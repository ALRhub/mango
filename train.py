import os
# deterministic cublas implementation (for reproducibility)
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import hydra
import torch
import traceback
import sys
from lightning import Trainer
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from mango.util.initialization import load_omega_conf_resolvers, main_initialization
from mango.util.own_types import ConfigDict

# full stack trace
os.environ['HYDRA_FULL_ERROR'] = '1'

# register OmegaConf resolver for hydra
load_omega_conf_resolvers()


@hydra.main(version_base=None, config_path="configs", config_name="training_config")
def train(config: ConfigDict) -> None:
    try:
        exp_root = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        print(OmegaConf.to_yaml(config, resolve=True))
        train_ds, eval_ds, train_dl, eval_dl, algorithm, wandb_logger = main_initialization(config)
        if config.trainer.get("matmul_precision", None) is not None:
            torch.set_float32_matmul_precision(config.trainer.matmul_precision)

        checkpoint_callback = ModelCheckpoint(
            monitor='val_mse',  # Metric to monitor
            dirpath=os.path.join(exp_root, "checkpoints"),  # Directory to save checkpoints
            filename='best-checkpoint-{epoch:02d}-{val_mse:.8f}',  # Checkpoint filename format
            save_top_k=3,  # Save only the best `k` models (1 = best model only)
            mode='min',  # Mode for monitoring ('min' for lower is better, 'max' for higher is better)
            save_last=True  # Optionally save the most recent model
        )
        learning_rate_monitor = LearningRateMonitor(logging_interval='epoch')

        callbacks = []
        if config.trainer.checkpointing:
            callbacks.append(checkpoint_callback)
        if wandb_logger:
            callbacks.append(learning_rate_monitor)
        if len(callbacks) == 0:
            callbacks = None

        trainer = Trainer(
            num_sanity_val_steps=config.trainer.num_sanity_val_steps,
            logger=wandb_logger,  # Use the wandb logger
            max_epochs=config.trainer.epochs,  # Max number of epochs for training
            accelerator=config.trainer.accelerator,  # what type of accelerator to use
            devices=config.trainer.devices,  # how many devices to use (if accelerator is not None)
            precision=config.trainer.precision,  # Precision setting (e.g., 16-bit)
            callbacks=callbacks,  # Checkpointing callback
            accumulate_grad_batches=config.trainer.accumulate_grad_batches,  # Gradient accumulation
            check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,  # How often to validate (in epochs)
            default_root_dir=exp_root,  # Where to save logs and checkpoints
            enable_checkpointing=config.trainer.checkpointing,  # Enable checkpointing
            enable_progress_bar=config.trainer.enable_progress_bar,  # Enable progress bar
        )

        # Now, start the training
        trainer.fit(algorithm, train_dataloaders=train_dl, val_dataloaders=eval_dl)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == '__main__':
    train()
