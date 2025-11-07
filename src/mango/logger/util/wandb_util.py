import os
from typing import Union, Any

import hydra
import wandb
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from plotly.graph_objs import Figure as PlotlyFigure


def get_job_type(entity: str, project: str, group: str) -> str:
    """
    Starts the WandB API and searches for the group in the given project of the given entity.
    If there are no job types (e.g. no runs) returns a job type "run_00"
    Otherwise, it finds the highest job type number x and returns run_{x+1:02d}.
    """
    api = wandb.Api()
    group_runs = api.runs(f"{entity}/{project}", filters={"group": group})
    job_types = list(set([r.job_type for r in group_runs]))
    if len(job_types) == 0:
        return "run_00"
    else:
        indices = [int(job_type[job_type.rfind("_") + 1:]) for job_type in job_types]
        indices.sort()
        last_idx = indices[-1]
        new_job_type = f"run_{last_idx + 1:02d}"
        return new_job_type


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_START_METHOD",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def wandbfy(vis_figure: Union[plt.Figure, FuncAnimation, str, PlotlyFigure, Any], frame_duration: float | None = None) \
        -> Union[wandb.Image, wandb.Video, wandb.Html, Any]:
    """
    Converts the given figure to a wandb object that is loggable.
    Args:
        vis_figure:
        frame_duration: The duration of a single frame in seconds. Only relevant for animations in plotly figures.

    Returns:

    """
    if isinstance(vis_figure, plt.Figure):
        return wandb.Image(vis_figure)
    elif isinstance(vis_figure, FuncAnimation):
        try:
            return wandb.Video(vis_figure.to_html5_video())
            # todo here, debug and check.
        except Exception as e:
            print(f"Error converting animation to html: {e}. Resorting to (big) JS animation.")
            return wandb.Html(vis_figure.to_jshtml())

    elif isinstance(vis_figure, str):
        if vis_figure.endswith(".gif"):
            return wandb.Video(vis_figure)
        elif vis_figure.endswith(".png"):
            return wandb.Image(vis_figure)
    elif isinstance(vis_figure, PlotlyFigure) and len(vis_figure.frames) > 0:
        # if the plotly figure contains an animation (i.e., has non-empty frames), convert it to a html file
        # such that this animation can be properly displayed in the wandb dashboard
        from plotly.io import to_html
        return wandb.Html(to_html(vis_figure, include_plotlyjs="cdn", auto_play=True,
                                  animation_opts={"frame": {"duration": frame_duration,
                                                            "redraw": True  # must be true
                                                            },
                                                  "fromcurrent": True,
                                                  "transition": {"duration": frame_duration},
                                                  }))
    else:
        # just log it
        return vis_figure


def get_job_type_from_override(length: int = 2, ignore_keys=("seed", "+experiment", "+platform")) -> str:
    """Helper function to get the job type from the hydra overrides.

    Basically we take the last part of the override, split it by '_', and then take the first letter of each part.
    ==> agent_config.model_config.train_btm_image_prodmp_residual_mlp -> t_b_i_p_r_m (length=1) or tr_bi_im_pr_ml (length=2)

    Returns:
        str: The group name.
    """
    overrides = hydra.utils.HydraConfig.get()["overrides"]["task"]
    overrides_shortened = []
    for override in overrides:
        ignore = False
        for ignore_key in ignore_keys:
            if ignore_key in override:
                # ignore this kind of override as it is not a hyperparameter. Seed is usually the run name.
                ignore = True
                break
        if ignore:
            continue
        override_value = override.split("=")[-1]
        override_key = override.split("=")[-2]
        key_components = override_key.split(".")  # overrides from setting a param.value: val
        if len(key_components) == 1:
            key_components = key_components[0].split("/")  # overrides from overriding a config/value: config_name
        override_shortened = "_".join([o[:length] for o in key_components[-1].split("_")])
        overrides_shortened.append(f"{override_shortened}={override_value}")
    if len(overrides_shortened) == 0:
        return "default"
    output = ",".join(overrides_shortened)
    # truncate to last 64 characters
    if len(output) > 61:
        output = "..." + output[-61:]
    return output
