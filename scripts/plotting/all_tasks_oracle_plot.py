import os
from typing import List

import matplotlib
import numpy as np
import pandas as pd
import pylab
import scipy
import seaborn as sns
import tikzplotlib
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator, ScalarFormatter
from omegaconf import OmegaConf
from scipy.stats import trim_mean, t, sem, bootstrap

from scripts.plotting.configs import styles, names
from scripts.plotting.configs.names import env_names


class MagnitudeFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, exponent=None):
        super().__init__()
        self._fixed_exponent = exponent

    def _set_order_of_magnitude(self):
        if self._fixed_exponent:
            self.orderOfMagnitude = self._fixed_exponent
        else:
            super()._set_order_of_magnitude()

    def _set_format(self):
        self.format = "%1.1f"


def plot(data: dict,
         figure_width: int = 10,
         figure_height: int = 5,
         mode: str = "show",
         unique_name: str = None,
         formatter_magnitude: int = -4,
         use_major_formatter: bool = False,
         use_minor_formatter: bool = False,
         y_top_limit: float | None = None,
         y_bottom_limit: float | None = None,
         legend_order: List[int] | None = None,
         label_fontsize: int = 19,
         tick_fontsize: int = 16,
         linewidth: int = 3,
         boarder_linewidth: int = 3,
         border_color='darkgray',
         show_legend: bool = True,
         overwrite: bool = False):

    # load data
    df = {
        "env": [],
        "method": [],
        "context": [],
        "seed": [],
        "full_rollout_mean_mse": [],
    }
    for method_name, path in data.items():
        for seed in os.listdir(path):
            output_path = os.path.join(path, seed)
            meta_data = os.path.join(output_path, "meta_data.yaml")
            # load
            meta_data = OmegaConf.load(meta_data)
            results = os.path.join(output_path, "metrics_mean_over_time.pt")
            results = torch.load(results)
            for context_idx, mse_value in enumerate(results):
                context_size = context_idx + 1
                df["env"].append(meta_data.env_name)
                df["method"].append(method_name)
                df["context"].append(context_size)
                df["seed"].append(seed)
                df["full_rollout_mean_mse"].append(mse_value.item())
    # assert all methods have the same env
    assert len(set(df["env"])) == 1
    env_name = df["env"][0]
    methods = data.keys()
    data = pd.DataFrame(df)

    metric = "full_rollout_mean_mse"
    # call seaborn
    sns.set(rc={'figure.figsize': (figure_width, figure_height)})
    sns.set_theme()
    sns.set_style("whitegrid")
    plt.rcParams.update({"ytick.left": True})
    sns.lineplot(data=data, x="context", y=metric,
                 hue="method",
                 style="method",
                 palette=styles.method_colors,
                 markers=styles.method_markers,
                 markersize=10,
                 dashes=False,
                 # errorbar=("pi", 75),
                 errorbar="ci",
                 # n_boot=1000,
                 estimator=lambda scores: scipy.stats.trim_mean(scores, proportiontocut=0.25, axis=None),  # IQM
                 linewidth=linewidth,
                 )

    ax = plt.gca()

    # ticks and labels appearance
    x_labels = data["context"].unique()
    # only get numbers from context
    x_labels = [int(x) for x in x_labels]
    # remove trailing zeros
    x_labels = [str(int(x)) for x in x_labels]

    # plt.xticks(ticks=range(len(x_labels)), labels=x_labels)
    plt.xlabel("Context Size", fontdict={'size': label_fontsize})
    plt.ylabel(names.metric_names[metric], fontdict={'size': label_fontsize})

    # Set the fontsize  and colors for the numbers on the ticks and the offset text.
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=border_color, labelcolor="black")
    ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize, colors=border_color, labelcolor="black")
    ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)

    # Ticks number formatting and y scale
    # plt.yscale('log')
    fmt = MagnitudeFormatter(formatter_magnitude)
    if use_major_formatter:
        ax.yaxis.set_major_formatter(fmt)
    if use_minor_formatter:
        ax.yaxis.set_minor_formatter(fmt)

    # y limits
    if y_top_limit is not None:
        plt.ylim(top=y_top_limit)
    if y_bottom_limit is not None:
        plt.ylim(bottom=y_bottom_limit)

    # boarder
    plt.gca().spines['bottom'].set_linewidth(boarder_linewidth)
    plt.gca().spines['left'].set_linewidth(boarder_linewidth)
    plt.gca().spines['top'].set_linewidth(boarder_linewidth)
    plt.gca().spines['right'].set_linewidth(boarder_linewidth)
    plt.gca().spines['right'].set_color(border_color)
    plt.gca().spines['top'].set_color(border_color)
    plt.gca().spines['bottom'].set_color(border_color)
    plt.gca().spines['left'].set_color(border_color)

    # title
    plt.title(env_names[env_name], fontdict={'size': label_fontsize})

    # legend
    if show_legend:
        # Get the handles and labels of the current axes
        if legend_order is not None:
            handles, labels = plt.gca().get_legend_handles_labels()
            legend = plt.legend([handles[idx] for idx in legend_order], [labels[idx] for idx in legend_order])
        else:
            legend = plt.legend()

        # Translate the labels using the dictionary
        method_names = names.method_names
        for text in legend.get_texts():
            original_label = text.get_text()
            translated_label = method_names.get(original_label, original_label)
            text.set_text(translated_label)
            text.set_fontsize(tick_fontsize)
        # Set the legend title
        legend.set_title('Methods')
        legend.get_title().set_fontsize(label_fontsize)
    else:
        ax.get_legend().remove()

    # save or show
    if mode == "pdf" or mode == "tikz":
        if unique_name is not None:
            file_name = unique_name
        else:
            method_filename = "__".join(sorted(methods))
            file_name = f"{metric}___{method_filename}"
        out_path = f"output/figures/{env_name}/{file_name}.pdf"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # check if file exists and only write if overwrite is true
        if os.path.isfile(out_path) and not overwrite:
            print(f"File {out_path} already exists. Skipping...")
        else:
            # set tight layout explicit in savefig
            if mode == "pdf":
                plt.savefig(out_path, bbox_inches='tight', pad_inches=0, )
            elif mode == "tikz":
                out_path = out_path.replace(".pdf", ".tex")
                tikzplotlib.save(out_path)
            print(f"Saved figure to {out_path}")
    elif mode == "show":
        plt.show()
    else:
        raise ValueError(f"Unknown mode {mode}")



if __name__ == "__main__":
    methods = ["exp5101_pbv3_mgn_oracle", "exp5301_pbv3_dummy_mgno_oracle", "exp5701_pbv3_dummy_egno_oracle",
               "exp5103_dp_easy_v5_mgn_oracle", "exp5303_dp_easy_v5_dummy_mgno_oracle", "exp5703_dp_easy_v5_dummy_egno_oracle",
               "exp5103_dp_hard_v5_mgn_oracle", "exp5303_dp_hard_v5_dummy_mgno_oracle", "exp5703_dp_hard_v5_dummy_egno_oracle",
               "exp5103_sc_mgn_oracle", "exp5301_sc_dummy_mgno_oracle", "exp5701_sc_dummy_egno_oracle"
               ]
    env_order = ["pb_v3_ml", "dp_easy_v5_ml", "dp_hard_v5_ml", "sphere_cloth_ml"]
    algorithm_order = ["ml_dummy_mgno_time_conv", "ml_dummy_mgn","ml_dummy_egno"]
    method_names = ["MGNO (Oracle)", "MGN (Oracle)", "EGNO (Oracle)"]
    env_names = ["Plan. Bend.", "Def. Plate (Easy)", "Def. Plate (Hard)", "Sphere Cloth Coupl."]

    base_path = "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations"
    combined_data = torch.zeros((len(env_order), 3, 5)) + 0

    for method_name in methods:
        method_path = os.path.join(base_path, method_name)
        sub_folder = os.listdir(method_path)[0]
        method_path = os.path.join(method_path, sub_folder)
        for seed_idx, seed in enumerate(sorted(os.listdir(method_path))):
            output_path = os.path.join(method_path, seed)
            meta_data = os.path.join(output_path, "meta_data.yaml")
            # load
            meta_data = OmegaConf.load(meta_data)
            results = os.path.join(output_path, "metrics_mean_over_time.pt")
            results = torch.load(results)
            mse_value = torch.min(results)
            env = meta_data.env_name
            algorithm_name = meta_data.simulator_name
            algorithm_index = algorithm_order.index(algorithm_name)
            assert algorithm_index >= 0
            env_index = env_order.index(env)
            assert env_index >= 0
            combined_data[env_index, algorithm_index, seed_idx] = mse_value

    # egno oracle set to 0 for planar bending envs
    # combined_data[0, 0, :] = 0
    # combined_data[1, 0, :] = 0

    data = combined_data.numpy()
    # Compute aggregated scores
    aggregated_scores = np.apply_along_axis(lambda scores: trim_mean(scores, proportiontocut=0.0), axis=2,
                                            arr=data)

    # Compute bootstrap CI along the last dimension
    ci_lower = np.zeros((4, 3))
    ci_upper = np.zeros((4, 3))
    statistic = np.mean

    for i in range(4):  # Loop over environments
        for j in range(3):  # Loop over methods
            res = bootstrap((data[i, j, :],), statistic, confidence_level=0.95,
                            n_resamples=10000, method='percentile')
            ci_lower[i, j], ci_upper[i, j] = res.confidence_interval

    # Compute aggregated scores (trimmed mean or normal mean)
    aggregated_scores = np.mean(data, axis=2)  # Replace with trim_mean if needed

    # Compute error bars for visualization
    error_bars_lower = np.abs(aggregated_scores - ci_lower)  # Lower CI errors
    error_bars_upper = np.abs(ci_upper - aggregated_scores)


    # Plot
    num_envs, num_methods = aggregated_scores.shape
    x = np.arange(num_envs)  # Group positions
    bar_width = 0.2  # Width of bars

    fig, ax = plt.subplots(figsize=(20, 4))
    colors = ['tab:blue', 'tab:orange', 'tab:green']  # Colors for methods

    for i in range(num_methods):
        ax.bar(x + i * bar_width - (num_methods - 1) * bar_width / 2, aggregated_scores[:, i],
               width=bar_width, label=method_names[i], color=colors[i],
               yerr=[error_bars_lower[:, i], error_bars_upper[:, i]], capsize=5)
    # ax.set_yscale('log')

    # plt.title("Performance Comparison")

    label_fontsize: int = 19
    tick_fontsize: int = 16
    linewidth: int = 3
    boarder_linewidth: int = 3
    border_color = 'darkgray'
    # Formatting
    ax.legend(title="Methods", fontsize=tick_fontsize, title_fontsize=label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}" for e in env_names])  # Ensure it matches num_envs
    ax.set_ylabel("Full Rollout MSE", fontdict={'size': label_fontsize})

    # Set the fontsize  and colors for the numbers on the ticks and the offset text.
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=border_color, labelcolor="black")
    ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize, colors=border_color, labelcolor="black")
    ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)

    # Ticks number formatting and y scale
    plt.yscale('log')
    # fmt = MagnitudeFormatter(formatter_magnitude)
    # if use_major_formatter:
    #     ax.yaxis.set_major_formatter(fmt)
    # if use_minor_formatter:
    #     ax.yaxis.set_minor_formatter(fmt)
    #
    # # y limits
    # if y_top_limit is not None:
    #     plt.ylim(top=y_top_limit)
    # if y_bottom_limit is not None:
    #     plt.ylim(bottom=y_bottom_limit)

    # border
    plt.gca().spines['bottom'].set_linewidth(boarder_linewidth)
    plt.gca().spines['left'].set_linewidth(boarder_linewidth)
    plt.gca().spines['top'].set_linewidth(boarder_linewidth)
    plt.gca().spines['right'].set_linewidth(boarder_linewidth)
    plt.gca().spines['right'].set_color(border_color)
    plt.gca().spines['top'].set_color(border_color)
    plt.gca().spines['bottom'].set_color(border_color)
    plt.gca().spines['left'].set_color(border_color)
    # plt.savefig("/home/philipp/lsdf/for5339/ltsgnsv2/figures/quantitative/all_tasks_oracle.pdf", bbox_inches='tight', pad_inches=0)
    # plt.show()
    tikzplotlib.save("/home/philipp/lsdf/for5339/ltsgnsv2/figures/quantitative/all_tasks_oracle_log.tex")



