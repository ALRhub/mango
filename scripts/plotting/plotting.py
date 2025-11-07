import os
from typing import List

import matplotlib
import pandas as pd
import pylab
import scipy
import seaborn as sns
import matplot2tikz
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator, ScalarFormatter
from omegaconf import OmegaConf

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
         figure_width: int = 5,
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
         show_legend: bool = False,
         overwrite: bool = False,
         exclude_seeds=None,):

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
            if exclude_seeds is not None and seed in exclude_seeds:
                continue
            output_path = os.path.join(path, seed)
            meta_data = os.path.join(output_path, "meta_data.yaml")
            # load
            meta_data = OmegaConf.load(meta_data)
            results = os.path.join(output_path, "metrics_mean_over_time.pt")
            results = torch.load(results)
            for context_idx, mse_value in enumerate(results):
                context_size = context_idx + 1
                if context_size > 4:
                    continue
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
    ax = sns.lineplot(data=data, x="context", y=metric,
                 hue="method",
                 style="method",
                 palette=styles.method_colors,
                 # markers=styles.method_markers,
                 markersize=10,
                 dashes=styles.method_dashes,
                 linewidth=2,
                 # errorbar=("pi", 75),
                 errorbar="ci",
                 # n_boot=1000,
                 estimator=lambda scores: scipy.stats.trim_mean(scores, proportiontocut=0.0, axis=None),  # IQM
                 )

    for line, method in zip(ax.lines, data["method"].unique()):
        line.set_linewidth(styles.line_width[method])

    ax = plt.gca()

    # # ticks and labels appearance
    # x_labels = data["context"].unique()
    # # only get numbers from context
    # x_labels = [int(x) for x in x_labels]
    # # remove trailing zeros
    # x_labels = [str(int(x)) for x in x_labels]

    # plt.xticks(ticks=range(len(x_labels)), labels=x_labels)
    plt.xlabel("Context Size", fontdict={'size': label_fontsize})
    plt.ylabel(names.metric_names[metric], fontdict={'size': label_fontsize})

    # Set the fontsize  and colors for the numbers on the ticks and the offset text.
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=border_color, labelcolor="black")
    ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize, colors=border_color, labelcolor="black")
    ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)

    # Ticks number formatting and y scale
    plt.yscale('log')
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
        out_path = f"/home/philipp/lsdf/for5339/ltsgnsv2/figures/quantitative/{env_name}/{file_name}.pdf"
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
                matplot2tikz.save(out_path)
            print(f"Saved figure to {out_path}")
    elif mode == "show":
        plt.show()
    else:
        raise ValueError(f"Unknown mode {mode}")


def pbv3_easy(show=False):
    data = {
        "mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5001_pbv3_mgn/+ex_exp=planar_bending_v3,+pl=horeka_gpu_tai/",
        # "mgn_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5101_pbv3_mgn_oracle/+ex_exp_ora=planar_bending_v3,+pl=horeka_gpu_tai/",
        "ml_mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5502_planar_bending_v3_ml_mgn/+ex_exp_dee_mgn=planar_bending_v3,+pl=horeka_gpu_tai/",
        "dummy_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5201_pbv3_dummy_mgno/+ex_exp_mgn=planar_bending_v3,+pl=horeka_gpu_tai/",
        "dummy_mgno_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5301_pbv3_dummy_mgno_oracle/+ex_exp_mgn_ora=planar_bending_v3,+pl=horeka_gpu_tai/",
        "ml_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5401_planar_bending_v3_ml_mgno/+ex_exp_dee_mgn=planar_bending_v3,+pl=horeka_gpu_philipp/",
        "dummy_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5601_pbv3_dummy_egno/+ex_exp_egn=planar_bending_v3,+pl=horeka_gpu_philipp/",
        # "dummy_egno_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5701_pbv3_dummy_egno_oracle/+ex_exp_egn_ora=planar_bending_v3,+pl=horeka_gpu_philipp/",
        "ml_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5801_planar_bending_v3_ml_egno/+ex_exp_dee_egn=planar_bending_v3,+pl=horeka_gpu_dns/",
        "two_stages_mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5903_planar_bending_v3_two_stages_mgn/+ex_exp_dee_mgn_two_sta=planar_bending_v3,+pl=horeka_gpu_philipp/",
        "two_stages_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5803_planar_bending_v3_two_stages_mgno/+ex_exp_dee_mgn_two_sta=planar_bending_v3,+pl=horeka_gpu_philipp",
        "two_stages_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp6001_planar_bending_v3_two_stages_egno/+ex_exp_dee_egn_two_sta=planar_bending_v3,+pl=horeka_gpu_philipp/"
    }
    if show:
        plot(data, mode="show", unique_name="pbv3_easy_main")
    plt.close()
    plot(data, mode="pdf", unique_name="pbv3_easy_main", overwrite=True)
    plt.close()
    plot(data, mode="tikz", unique_name="pbv3_easy_main", overwrite=True)
    plt.close()


def pbv1_hard(show=False):
    data = {
        "mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5001_pb_hard_v1_mgn/+ex_exp=pb_hard,+pl=horeka_gpu_dns/",
        # "mgn_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5101_pb_hard_v1_mgn_oracle/+ex_exp_ora=pb_hard,+pl=horeka_gpu_dns/",
        "ml_mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5502_pb_hard_v1_ml_mgn/+ex_exp_dee_mgn=pb_hard,+pl=horeka_gpu_tai/",
        "dummy_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5201_pb_hard_v1_dummy_mgno/+ex_exp_mgn=pb_hard,+pl=horeka_gpu_tai/",
        "dummy_mgno_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5301_pb_hard_v1_dummy_mgno_oracle/+ex_exp_mgn_ora=pb_hard,+pl=horeka_gpu_tai/",
        "ml_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5401_pb_hard_v1_ml_mgno/+ex_exp_dee_mgn=pb_hard,+pl=horeka_gpu_tai/",
        "dummy_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5601_pb_hard_v1_dummy_egno/+ex_exp_egn=pb_hard,+pl=horeka_gpu_philipp/",
        # "dummy_egno_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5701_pbv3_dummy_egno_oracle/+ex_exp_egn_ora=planar_bending_v3,+pl=horeka_gpu_philipp/",
        "ml_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5801_pb_hard_v1_ml_egno/+ex_exp_dee_egn=pb_hard,+pl=horeka_gpu_philipp/"
    }
    if show:
        plot(data, mode="show", unique_name="pbv1_hard_main")
    plt.close()
    plot(data, mode="pdf", unique_name="pbv1_hard_main", overwrite=True)
    plt.close()
    plot(data, mode="tikz", unique_name="pbv1_hard_main", overwrite=True)
    plt.close()

def dp_easy(show=False):
    data = {
        "mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5003_dp_easy_v5_mgn/+ex_exp=dp_easy_v5,+pl=horeka_gpu_philipp/",
        # "mgn_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5101_pb_hard_v1_mgn_oracle/+ex_exp_ora=pb_hard,+pl=horeka_gpu_dns/",
        "ml_mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5504_dp_easy_v5_ml_mgn/+ex_exp_dee_mgn=dp_easy_v5,+pl=horeka_gpu_tai/",
        "dummy_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5203_dp_easy_v5_dummy_mgno/+ex_exp_mgn=dp_easy_v5,+pl=horeka_gpu_tai/",
        "dummy_mgno_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5303_dp_easy_v5_dummy_mgno_oracle/+ex_exp_mgn_ora=dp_easy_v5,+pl=horeka_gpu_tai/",
        "ml_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5403_dp_easy_v5_ml_mgno/+ex_exp_dee_mgn=dp_easy_v5,+pl=horeka_gpu_philipp/",
        "dummy_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5603_dp_easy_v5_dummy_egno/+ex_exp_egn=dp_easy_v5,+pl=horeka_gpu_dns/",
        # "dummy_egno_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5701_pbv3_dummy_egno_oracle/+ex_exp_egn_ora=planar_bending_v3,+pl=horeka_gpu_philipp/",
        "ml_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5803_dp_easy_v5_ml_egno/+ex_exp_dee_egn=dp_easy_v5,+pl=horeka_gpu_dns/",
        "two_stages_mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5904_dp_easy_v5_two_stages_mgn/+ex_exp_dee_mgn_two_sta=dp_easy_v5,+pl=horeka_gpu_philipp/",
        "two_stages_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5804_dp_easy_v5_two_stages_mgno/+ex_exp_dee_mgn_two_sta=dp_easy_v5,+pl=horeka_gpu_philipp/",
        "two_stages_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp6001_dp_easy_v5_two_stages_egno/+ex_exp_dee_egn_two_sta=dp_easy_v5,+pl=horeka_gpu_philipp/",
    }
    if show:
        plot(data, mode="show", unique_name="dp_easy_main", show_legend=True)
    plt.close()
    plot(data, mode="pdf", unique_name="dp_easy_main", overwrite=True)
    plt.close()
    plot(data, mode="tikz", unique_name="dp_easy_main", overwrite=True)
    plt.close()


def dp_easy_ood(show=False):
    data = {
        "dummy_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5203_dp_easy_v5_ood_dummy_mgno/+ex_exp_mgn=dp_easy_v5_ood,+pl=horeka_gpu_tai/",
        "dummy_mgno_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5303_dp_easy_v5_ood_dummy_mgno_oracle/+ex_exp_mgn_ora=dp_easy_v5_ood,+pl=horeka_gpu_tai/",
        "ml_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5403_dp_easy_v5_ood_ml_mgno/+ex_exp_dee_mgn=dp_easy_v5_ood,+pl=horeka_gpu_dns/",

    }
    if show:
        plot(data, mode="show", unique_name="dp_easy_ood")
    plt.close()
    plot(data, mode="pdf", unique_name="dp_easy_ood", overwrite=True)
    plt.close()
    plot(data, mode="tikz", unique_name="dp_easy_ood", overwrite=True)
    plt.close()

def dp_hard_ood(show=False):
    data = {
        "dummy_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5203_dp_hard_v5_ood_dummy_mgno/+ex_exp_mgn=dp_hard_v5_ood,+pl=kluster/",
        "dummy_mgno_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5303_dp_hard_v5_ood_dummy_mgno_oracle/+ex_exp_mgn_ora=dp_hard_v5_ood,+pl=kluster/",
        "ml_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5403_dp_hard_v5_ood_ml_mgno/+ex_exp_dee_mgn=dp_hard_v5_ood,+pl=kluster/",

    }
    if show:
        plot(data, mode="show", unique_name="dp_hard_ood")
    plt.close()
    plot(data, mode="pdf", unique_name="dp_hard_ood", overwrite=True)
    plt.close()
    plot(data, mode="tikz", unique_name="dp_hard_ood", overwrite=True)
    plt.close()

def dp_hard(show=False):
    data = {
        "mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5003_dp_hard_v5_mgn/+ex_exp=dp_hard_v5,+pl=horeka_gpu_philipp/",
        # "mgn_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5101_pb_hard_v1_mgn_oracle/+ex_exp_ora=pb_hard,+pl=horeka_gpu_dns/",
        "ml_mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5504_dp_hard_v5_ml_mgn/+ex_exp_dee_mgn=dp_hard_v5,+pl=horeka_gpu_tai/",
        "dummy_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5203_dp_hard_v5_dummy_mgno/+ex_exp_mgn=dp_hard_v5,+pl=horeka_gpu_tai/",
        "dummy_mgno_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5303_dp_hard_v5_dummy_mgno_oracle/+ex_exp_mgn_ora=dp_hard_v5,+pl=horeka_gpu_tai/",
        "ml_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5403_dp_hard_v5_ml_mgno/+ex_exp_dee_mgn=dp_hard_v5,+pl=horeka_gpu_philipp/",
        "dummy_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5603_dp_hard_v5_dummy_egno/+ex_exp_egn=dp_hard_v5,+pl=horeka_gpu_dns/",
        # "dummy_egno_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5701_pbv3_dummy_egno_oracle/+ex_exp_egn_ora=planar_bending_v3,+pl=horeka_gpu_philipp/",
        "ml_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5803_dp_hard_v5_ml_egno/+ex_exp_dee_egn=dp_hard_v5,+pl=horeka_gpu_dns/",
        "two_stages_mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5903_dp_hard_v5_two_stages_mgn/+ex_exp_dee_mgn_two_sta=dp_hard_v5,+pl=horeka_gpu_philipp/",
        "two_stages_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5803_dp_hard_v5_two_stages_mgno/+ex_exp_dee_mgn_two_sta=dp_hard_v5,+pl=horeka_gpu_philipp/",
        "two_stages_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp6001_dp_hard_v5_two_stages_egno/+ex_exp_dee_egn_two_sta=dp_hard_v5,+pl=horeka_gpu_philipp/",
    }
    if show:
        plot(data, mode="show", unique_name="dp_hard_main")
    plt.close()
    plot(data, mode="pdf", unique_name="dp_hard_main", overwrite=True)
    plt.close()
    plot(data, mode="tikz", unique_name="dp_hard_main", overwrite=True)
    plt.close()

def sc(show=False):
    data = {
        "mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5003_sc_mgn/+ex_exp=sphere_cloth,+pl=horeka_gpu_tai/",
        # "mgn_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5101_pb_hard_v1_mgn_oracle/+ex_exp_ora=pb_hard,+pl=horeka_gpu_dns/",
        "ml_mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5502_sc_ml_mgn/+ex_exp_dee_mgn=sphere_cloth,+pl=horeka_gpu_tai/",
        "dummy_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5201_sc_dummy_mgno/+ex_exp_mgn=sphere_cloth,+pl=horeka_gpu_philipp/",
        "dummy_mgno_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5301_sc_dummy_mgno_oracle/+ex_exp_mgn_ora=sphere_cloth,+pl=horeka_gpu_dns/",
        "ml_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5401_sc_ml_mgno/+ex_exp_dee_mgn=sphere_cloth,+pl=horeka_gpu_philipp/",
        "dummy_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5603_sc_dummy_egno/+ex_exp_egn=sphere_cloth,+pl=horeka_gpu_philipp/",
        # "dummy_egno_oracle": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5701_pbv3_dummy_egno_oracle/+ex_exp_egn_ora=planar_bending_v3,+pl=horeka_gpu_philipp/",
        "ml_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5801_sc_ml_egno/+ex_exp_dee_egn=sphere_cloth,+pl=horeka_gpu_philipp/",
        "two_stages_mgn": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5903_sc_two_stages_mgn/+ex_exp_dee_mgn_two_sta=sphere_cloth,+pl=horeka_gpu_philipp/",
        "two_stages_mgno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp5803_sc_two_stages_mgno/+ex_exp_dee_mgn_two_sta=sphere_cloth,+pl=horeka_gpu_philipp/",
        "two_stages_egno": "/home/philipp/lsdf/for5339/ltsgnsv2/evaluations/exp6001_sc_two_stages_egno/+ex_exp_dee_egn_two_sta=sphere_cloth,+pl=horeka_gpu_philipp/",
    }
    if show:
        plot(data, mode="show", unique_name="sc_main")
    plt.close()
    plot(data, mode="pdf", unique_name="sc_main", overwrite=True)
    plt.close()
    plot(data, mode="tikz", unique_name="sc_main", overwrite=True)
    plt.close()



def outside_plot():
    data = {
        "mgn": "/home/philipp/projects/stamp_forming_sim/output/test_results/outside/p1001_dpv2_exp_mgn/+ex_eas_v2=p10_exp_mgn,+pl=horeka_gpu_single/",
        "deepset_mgno_time_conv": "/home/philipp/projects/stamp_forming_sim/output/test_results/outside/p0801_dpv2_cnn_deepset_mp/+ex_eas_v2=p08_cnn_deepset_hpo,+pl=horeka_gpu_short,lr=1.0e-4,alg=adam,max_tar_siz=1,min_con_siz=1,min_tar_siz=1/",
        "dummy_mgno_time_conv": "/home/philipp/projects/stamp_forming_sim/output/test_results/outside/p1201_dpv2_easy_dummy_mgno/+ex_eas_v2=p12_dummy_mgno,+pl=kluster,alg=ml_dummy_mgno_spectral/"
    }
    plot(data, mode="show")


if __name__ == "__main__":
    # outside_plot()
    # data = {
    #     "mgn": "/home/philipp/projects/stamp_forming_sim/output/test_results/p2703_pb_mgn_baseline/+ex_ben=p27_mgn_baseline,+pl=kluster/",
    #     "deepset_mgn": "/home/philipp/projects/stamp_forming_sim/output/test_results/p2601_pb_deepset_ml/+ex_ben=p26_deepset_ml,+pl=kluster,alg=ml_deepset_mgn/",
    #     "deepset_mgno_spectral": "/home/philipp/projects/stamp_forming_sim/output/test_results/p2601_pb_deepset_ml/+ex_ben=p26_deepset_ml,+pl=kluster,alg=ml_deepset_mgno_spectral/",
    #     "deepset_mgno_time_conv": "/home/philipp/projects/stamp_forming_sim/output/test_results/p2601_pb_deepset_ml/+ex_ben=p26_deepset_ml,+pl=kluster,alg=ml_deepset_mgno_time_conv/"
    # }
    # plot(data, mode="show")
    show = True
    pbv3_easy(show)
    dp_easy(show)
    # dp_easy_ood(show)
    # dp_hard_ood(show)
    dp_hard(show)
    sc(show)



