import os
from argparse import Namespace
from typing import Union, Dict, Any, Optional
import hydra
import torch
import traceback
import sys
from lightning import Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import OmegaConf

from stamp_forming_sim.algorithm import get_algorithm
from stamp_forming_sim.logger.visualizations.log_visualizations import visualize_ml_trajectories
from stamp_forming_sim.util.initialization import load_omega_conf_resolvers, main_initialization, initialize_config, \
    initialize_seed, get_data
from stamp_forming_sim.util.own_types import ConfigDict

# full stack trace
os.environ['HYDRA_FULL_ERROR'] = '1'

# register OmegaConf resolver for hydra
load_omega_conf_resolvers()


class TestLogger(Logger):
    def __init__(self, output_path, exp_name, job_type, seed, meta_data: dict, eval_ds):
        super().__init__()
        # Initialize your logger (e.g., file writer, custom logging server, etc.)
        self.output_path = os.path.join(output_path, exp_name, job_type, seed)
        self._vis_path = os.path.join(self.output_path, "visualizations")
        self._eval_ds = eval_ds
        # create output path
        os.makedirs(self.output_path, exist_ok=True)
        # save metadata
        with open(os.path.join(self.output_path, "meta_data.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(meta_data))

    @property
    def name(self) -> Optional[str]:
        return "Test Logger"

    @property
    def version(self) -> Optional[Union[int, str]]:
        return "1.0"

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        pass

    def log_metrics(self, outputs, step):
        metrics = outputs["metrics"]
        config = outputs.get("config", None)
        visualizations = outputs.get("visualizations", [])
        if len(visualizations) > 0:
            for current_vis in visualizations:
                for context_size in current_vis.keys():
                    context_size_int = int(context_size[-1])
                    context_vis = current_vis[context_size]
                    visualize_ml_trajectories(context_vis, context_size_int, self._vis_path, self._eval_ds)
        # Implement how metrics are logged (e.g., to a file or external system)
        if isinstance(metrics, torch.Tensor):
            torch.save(metrics, os.path.join(self.output_path, f"metrics_all_time_steps.pt"))
            torch.save(metrics.mean(dim=-1), os.path.join(self.output_path, f"metrics_mean_over_time.pt"))

        if config is not None:
            # save config
            with open(os.path.join(self.output_path, "config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(config, resolve=True))
        print(f"Saved test metrics in {self.output_path}.")

    def save(self):
        # Optionally implement saving or finalizing logging state if needed
        pass

    def finalize(self, status):
        # Finalize logging when done (e.g., close connections)
        pass


def get_best_checkpoint(checkpoint_path):
    def extract_metric_from_filename(filename):
        # Example for "best-checkpoint-02-0.01234567.ckpt"
        parts = filename.split('-')
        metric = parts[-1].split("=")[1]
        return float(metric.replace('.ckpt', ''))

    files = os.listdir(checkpoint_path)
    files = [f for f in files if not f == "last.ckpt"]
    extract_metric_from_filename(files[0])
    best_checkpoint = min(files, key=extract_metric_from_filename)
    return best_checkpoint


@hydra.main(version_base=None, config_path="configs", config_name="test_config")
def test(config: ConfigDict) -> None:
    print("Debug print for LSDF")
    lsdf_value = os.getenv("LSDF")
    print(f"LSDF={lsdf_value}")
    try:
        exp_root = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        print(OmegaConf.to_yaml(config, resolve=True))
        initialize_config(config)  # put slurm stuff into config
        initialize_seed(config)  # set seed
        train_ds, eval_ds, train_dl, eval_dl = get_data(config.dataset, loading=config.loading.enable_loading)
        if config.trainer.get("matmul_precision", None) is not None:
            torch.set_float32_matmul_precision(config.trainer.matmul_precision)

        loading_path = config.loading.root_path
        if loading_path.endswith("/"):
            loading_path = loading_path[:-1]
        output_path = config.loading.output_path
        os.makedirs(output_path, exist_ok=True)
        # last folder is exp_name
        exp_name = loading_path.split("/")[-1]
        allowed_job_types = config.loading.get("job_types", None)
        allowed_seeds = config.loading.get("seeds", None)
        for job_type in os.listdir(loading_path):
            if allowed_job_types is not None and job_type not in allowed_job_types:
                continue
            job_type_path = os.path.join(loading_path, job_type)
            for seed in os.listdir(job_type_path):
                if allowed_seeds is not None and seed not in allowed_seeds:
                    continue
                seed_path = os.path.join(job_type_path, seed)
                checkpoint_path = os.path.join(seed_path, "checkpoints")
                if config.loading.checkpoint == "last":
                    checkpoint_path = os.path.join(checkpoint_path, "last.ckpt")
                elif config.loading.checkpoint == "best":
                    best_chkpt = get_best_checkpoint(checkpoint_path)
                    checkpoint_path = os.path.join(checkpoint_path, best_chkpt)
                algorithm = get_algorithm(config=config.algorithm, train_dl=train_dl, train_ds=train_ds,
                                          eval_ds=eval_ds, loading=True,
                                          checkpoint_path=checkpoint_path)

                # this is the correct parent directory to load the algorithm and the checkpoints
                test_logger = TestLogger(config.loading.output_path, exp_name, job_type, seed, meta_data={
                    "env_name": config.dataset.wandb_name,
                    "algorithm_name": algorithm.config.name,
                    "simulator_name": algorithm.config.simulator.name,
                },
                                         eval_ds=eval_ds)
                trainer = Trainer(
                    logger=test_logger,  # results will be written to disk
                    max_epochs=config.trainer.epochs,  # Max number of epochs for training
                    accelerator=config.trainer.accelerator,  # what type of accelerator to use
                    devices=config.trainer.devices,  # how many devices to use (if accelerator is not None)
                    precision=config.trainer.precision,  # Precision setting (e.g., 16-bit)
                    callbacks=None,
                    default_root_dir=exp_root,  # dummy path, will be used multiple times presumably
                    enable_checkpointing=False,  # no checkpointing
                )

                # Now, start the test
                trainer.test(algorithm, dataloaders=eval_dl)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == '__main__':
    test()
