import hydra
import wandb
from omegaconf import OmegaConf

from mango.logger.custom_wandb_logger import CustomWandBLogger
from mango.logger.util.wandb_util import reset_wandb_env


def get_wandb_logger(config, algorithm):
    """
    Create a wandb logger with the given config and algorithm.
    Args:
        config:
        algorithm: An instance of the algorithm to run.

    Returns: A wandb logger to use.
    """
    reset_wandb_env()

    wandb_params = config.logger.wandb
    project_name = wandb_params.get("project_name")
    environment_name = config.dataset.wandb_name

    if wandb_params.get("task_name") is not None:
        project_name = project_name + "_" + wandb_params.get("task_name")
    elif environment_name is not None:
        project_name = project_name + "_" + environment_name
    else:
        # no further specification of the project, just use the initial project_name
        project_name = project_name

    groupname = wandb_params.get("group_name")
    if config.algorithm.evaluator.eval_only:
        groupname = config.eval_name + "_" + groupname
    if wandb_params.dev:
        groupname = "DEV_" + groupname
    groupname = groupname[-127:]
    runname = wandb_params.get("run_name")[-127:]
    job_type = wandb_params.get("job_type")[-64:]

    tags = wandb_params.get("tags", [])
    if tags is None:
        tags = []
    if config.get("algorithm").get("name") is not None:
        tags.append(config.get("algorithm").get("name"))
    if config.get("dataset").get("name") is not None:
        tags.append(config.get("dataset").get("name"))

    entity = wandb_params.get("entity")

    start_method = wandb_params.get("start_method")
    settings = wandb.Settings(start_method=start_method) if start_method is not None else None

    # Initialize WandB logger
    wandb_logger = CustomWandBLogger(
        config=OmegaConf.to_container(config, resolve=True),
        project=project_name,  # Name of your WandB project
        name=runname,  # Name of the current run
        group=groupname,  # Group name for the run
        tags=tags,  # List of tags for your run
        entity=entity,  # WandB username or team name
        settings=settings,  # Optional WandB settings
        job_type=job_type,  # Name of your experiment
        log_model=False,
        save_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    return wandb_logger
