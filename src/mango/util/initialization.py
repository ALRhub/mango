import os
import warnings
import random

import numpy as np
import torch
from omegaconf import open_dict, OmegaConf
from torch_geometric.loader import DataLoader

from mango.algorithm import get_algorithm
from mango.dataset import get_dataset
from mango.logger import get_wandb_logger
from mango.util.job_type_resolver import shortener


def main_initialization(config):
    """
    Initializes the config, the seed, the device and the env, algorithm, evaluator and recorder.
    :param config:
    :param env: If None, the env will be initialized. Otherwise, the env will be used.
    :return:
    """
    initialize_config(config)  # put slurm stuff into config
    initialize_seed(config)  # set seed
    train_ds, eval_ds, train_dl, eval_dl = get_data(config.dataset, loading=config.loading.enable_loading)

    if config.loading.enable_loading:
        warnings.warn("Loading is enabled. Use test.py directly. This is deprecated. Make sure that the loading is set up correctly.")
        algorithm = get_algorithm(config=config.algorithm, train_dl=train_dl, train_ds=train_ds, eval_ds=eval_ds, loading=True,
                                  checkpoint_path="/home/philipp/projects/mango/output/hydra/training/2025-01-17/p5004_pb_mgno_test_debug/checkpoints/last.ckpt")
    else:
        algorithm = get_algorithm(config=config.algorithm, train_dl=train_dl, train_ds=train_ds, eval_ds=eval_ds)
    if config.logger.wandb.enabled:
        # Initialize WandB logger
        wandb_logger = get_wandb_logger(config=config, algorithm=algorithm)
    else:
        wandb_logger = False
    return train_ds, eval_ds, train_dl, eval_dl, algorithm, wandb_logger


def get_data(config, loading=False):
    train_ds, eval_ds = get_dataset(config, loading=loading)
    train_dl = DataLoader(train_ds, batch_size=config.train_dataset.batch_size, shuffle=True, num_workers=2)
    eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)
    return train_ds, eval_ds, train_dl, eval_dl


def initialize_config(config):
    try:
        with open_dict(config):
            config.slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", None)
            config.slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    except KeyError:
        pass


def initialize_seed(config):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    from pytorch_lightning import seed_everything
    seed_everything(config.seed)


def conditional_resolver(condition, if_true: str, if_false: str):
    if condition:
        return if_true
    else:
        return if_false


def load_omega_conf_resolvers():
    OmegaConf.register_new_resolver("sub_dir_shortener", shortener)
    OmegaConf.register_new_resolver("format", lambda inpt, formatter: formatter.format(inpt))
    OmegaConf.register_new_resolver("conditional_resolver", conditional_resolver)
