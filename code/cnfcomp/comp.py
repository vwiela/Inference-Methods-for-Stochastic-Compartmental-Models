import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from epmodels.variant_model import seir2v_binomial, seir2v_gaussian, seir2v_forward
from epmodels.sir_model import sir_binomial, sir_gaussian


# ---------------- Simulation Helpers ---------------- #
def model(params, config, mode="binomial", batchsize=1):
    """Simulate trajectories given parameters and config."""
    assert params.ndim == 2, "params must be 2D (n_draws, n_params)"
    n_draws, n_params = params.shape

    if n_params == 2:
        if mode == "binomial":
            return sir_binomial(params, batchsize, config)
        else:  # gaussian/normal
            return sir_gaussian(params, batchsize, config)
    else:
        if mode == "binomial":
            return seir2v_binomial(params, batchsize, config)
        else:
            return seir2v_gaussian(params, batchsize, config)


def mean_squared_error(config, trajectories_1, trajectories_2):
    """Compute RMSE for infection and seroprevalence trajectories."""
    assert trajectories_1.ndim == 3 and trajectories_2.ndim == 3
    n_draws, batch_size, n_timepoints1 = trajectories_1.shape
    _, _, n_timepoints2 = trajectories_2.shape

    mse = np.zeros((n_draws, 2))
    for n in range(n_draws):
        mse1, mse2 = [], []
        for b in range(batch_size):
            mse1.append(np.sqrt(((trajectories_1[n, b] - config["obs1_nonmissing"]) ** 2).mean()))
            mse2.append(np.sqrt(((trajectories_2[n, b] - config["obs2_nonmissing"]) ** 2).mean()))
        mse[n, 0] = np.mean(mse1)
        mse[n, 1] = np.mean(mse2)
    return mse


# ---------------- Plotting ---------------- #


def plot_results(
    input,
    fontsize_label=36,
    fontsize_tick=28,
    errorbars=False,
    lower_b=None,
    upper_b=None,
    ticks_array=None,
):
    """Composite figure: posterior marginals, joint distributions, and trajectories."""

    # Colors
    cnf_colors = {"base": "#ff7f00", "20": "#fa935f", "50": "#d96b06", "80": "#9d552c"}
    pf_colors = {"base": "#377eb8", "20": "#7ab2ed", "50": "#418fd0", "80": "#426b94"}
    palette = {"CNF": cnf_colors["20"], "PF": pf_colors["20"]}

    n_draws, n_params = input["posterior_draws_cnf"].shape
    param_names = input["param_names"]

    # Bounds
    if n_params == 6:
        low = [-np.inf, -np.inf, -np.inf, -np.inf, 120, 10]
        high = [np.inf, np.inf, np.inf, np.inf, 360, 1000]
    elif n_params == 5:
        low, high = [0.95, 6, 1, 120, 10], [4, 30, 100, 360, 1000]
    elif n_params == 2:
        low, high = [0, 1], [1, 30]

    # Tick array
    if lower_b is not None and upper_b is not None:
        p20 = lower_b + 0.2 * (upper_b - lower_b)
        p80 = lower_b + 0.8 * (upper_b - lower_b)
        ticks_array = np.column_stack(
            [np.clip(np.round(p20, 2), low, high), np.clip(np.round(p80, 2), low, high)]
        )

    # Figure + GridSpec
    scale = 3
    n_rows = max(n_params, 4)
    figsize = 8.3 * scale, (n_rows + 0.0) * scale
    fig = plt.figure(figsize=figsize)
    height_ratios = [1] * n_rows
    width_ratios = [1] * 6 + [0.3] + [1] * 2
    gs = fig.add_gridspec(n_rows, 9, height_ratios=height_ratios, width_ratios=width_ratios)

    axs = []
    for i in range(n_params):
        axs.append([fig.add_subplot(gs[i, j]) for j in range(i + 1)])

    for i in range(n_params):
        sns.violinplot(
            data=input["df_posterior_draws"],
            y=param_names[i],
            hue="model",
            palette=palette,
            split=True,
            inner=None,
            fill=True,
            width=0.7,
            ax=axs[i][i],
            legend=False,
            linewidth=1,
        )
        if i > 0:
            axs[i][i].set_yticks([])
            axs[i][i].set_ylabel(None)

        patch_left = PathPatch(
            axs[i][i].collections[0].get_paths()[0], transform=axs[i][i].transData
        )
        patch_right = PathPatch(
            axs[i][i].collections[1].get_paths()[0], transform=axs[i][i].transData
        )

        if input["true_params"] is not None:
            line_left_true = axs[i][i].axhline(
                y=input["true_params"][i], color="black", linewidth=3
            )
            line_right_true = axs[i][i].axhline(
                y=input["true_params"][i], color="black", linewidth=3
            )
            line_left_true.set_clip_path(patch_left)
            line_right_true.set_clip_path(patch_right)

        if input["map_cnf"] is not None:
            line_left_map1 = axs[i][i].axhline(
                y=input["map_cnf"][i], color=cnf_colors["80"], linewidth=3
            )
            line_left_map1.set_clip_path(patch_left)

        if input["map_pf"] is not None:
            line_right_map2 = axs[i][i].axhline(
                y=input["map_pf"][i], color=pf_colors["80"], linewidth=3
            )
            line_right_map2.set_clip_path(patch_right)

        for spine in ["top", "right", "bottom"]:
            axs[i][i].spines[spine].set_visible(False)

        current_y_lower, current_y_upper = axs[i][i].get_ylim()
        new_y_lower = lower_b[i] if lower_b is not None else current_y_lower
        new_y_upper = upper_b[i] if upper_b is not None else current_y_upper
        axs[i][i].set_ylim(new_y_lower, new_y_upper)

    for i in range(n_params):
        for j in range(i):
            clip_x, clip_y = (low[j], high[j]), (low[i], high[i])
            sns.kdeplot(
                data=input["df_posterior_draws"].query('model == "CNF"'),
                x=param_names[j],
                y=param_names[i],
                color=cnf_colors["base"],
                levels=4,
                thresh=0.2,
                fill=True,
                alpha=0.7,
                ax=axs[i][j],
                legend=False,
                clip=(clip_x, clip_y),
            )

            sns.kdeplot(
                data=input["df_posterior_draws"].query('model == "PF"'),
                x=param_names[j],
                y=param_names[i],
                color=pf_colors["base"],
                levels=4,
                thresh=0.2,
                fill=True,
                alpha=0.7,
                ax=axs[i][j],
                legend=False,
                clip=(clip_x, clip_y),
            )

            axs[i][j].grid(alpha=0.5)
            for spine in ["top", "right"]:
                axs[i][j].spines[spine].set_visible(False)
            if j > 0 and i < n_params - 1:
                axs[i][j].set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
            if j == 0 and i < n_params - 1:
                axs[i][j].set(xlabel=None, xticklabels=[])
            if j > 0 and i == n_params - 1:
                axs[i][j].set(ylabel=None, yticklabels=[])

            current_x_lower, current_x_upper = axs[i][j].get_xlim()
            new_x_lower = lower_b[j] if lower_b is not None else current_x_lower
            new_x_upper = upper_b[j] if upper_b is not None else current_x_upper

            current_y_lower, current_y_upper = axs[i][j].get_ylim()
            new_y_lower = lower_b[i] if lower_b is not None else current_y_lower
            new_y_upper = upper_b[i] if upper_b is not None else current_y_upper

            axs[i][j].set_xlim(new_x_lower, new_x_upper)
            axs[i][j].set_ylim(new_y_lower, new_y_upper)

    # Synchronize y-ticks across each row (including diagonal)
    for i in range(n_params):
        ref_ax = axs[i][0]
        if ticks_array is None:
            ref_ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
        else:
            ref_ax.set_yticks(ticks_array[i])
        ref_yticks = ref_ax.get_yticks()
        for j in range(i + 1):  # include diagonal
            axs[i][j].set_yticks(ref_yticks)
        if j == i and i > 0:
            axs[i][j].set(yticklabels=[])

    for j in range(n_params):
        if j < n_params - 1:
            ref_ax = axs[-1][j]
            if ticks_array is None:
                ref_ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
            else:
                ref_ax.set_xticks(ticks_array[j])
            ref_xticks = ref_ax.get_xticks()
            for i in range(j + 1, n_params):
                axs[i][j].set_xticks(ref_xticks)

    for i in range(n_params):
        axs[-1][i].set_xlabel(param_names[i], fontsize=fontsize_label)
        axs[-1][i].tick_params(labelsize=fontsize_tick)
        axs[-1][i].tick_params(axis="x", labelrotation=45)
    for i in range(n_params):
        axs[i][0].set_ylabel(param_names[i], fontsize=fontsize_label)
        axs[i][0].tick_params(labelsize=fontsize_tick)

    ax_trajectories = fig.add_subplot(gs[0:3, 4:9])
    for spine in ["top", "right", "bottom", "left"]:
        ax_trajectories.spines[spine].set_visible(False)
    ax_trajectories.set(ylabel=None, xticklabels=[], yticklabels=[])
    ax_trajectories.tick_params(labelcolor="w", top=False, bottom=False, left=False, right=False)
    ax_trajectories.set_xlabel("Time t in days", fontsize=fontsize_label, labelpad=30)

    ax_trajectories.margins(x=0.1)

    ax_infc = fig.add_subplot(gs[0:3, 4:6])
    ax_seroprev = fig.add_subplot(gs[0:3, 7:9])

    timepoints1 = input["config"]["timepoints1_nonmissing"]
    ax_infc.fill_between(
        timepoints1,
        input["trajectories_cnf_1_95"][0],
        input["trajectories_cnf_1_95"][1],
        color=cnf_colors["20"],
        alpha=0.2,
        label="95% percentile",
    )
    ax_infc.fill_between(
        timepoints1,
        input["trajectories_cnf_1_90"][0],
        input["trajectories_cnf_1_90"][1],
        color=cnf_colors["50"],
        alpha=0.5,
        label="90% percentile",
    )
    ax_infc.fill_between(
        timepoints1,
        input["trajectories_cnf_1_50"][0],
        input["trajectories_cnf_1_50"][1],
        color=cnf_colors["80"],
        alpha=1,
        label="50% percentile",
    )
    ax_infc.fill_between(
        timepoints1,
        input["trajectories_pf_1_95"][0],
        input["trajectories_pf_1_95"][1],
        color=pf_colors["20"],
        alpha=0.2,
        label="95% percentile",
    )
    ax_infc.fill_between(
        timepoints1,
        input["trajectories_pf_1_90"][0],
        input["trajectories_pf_1_90"][1],
        color=pf_colors["50"],
        alpha=0.5,
        label="90% percentile",
    )
    ax_infc.fill_between(
        timepoints1,
        input["trajectories_pf_1_50"][0],
        input["trajectories_pf_1_50"][1],
        color=pf_colors["80"],
        alpha=1,
        label="50% percentile",
    )

    if errorbars == False:
        ax_infc.plot(
            timepoints1,
            input["config"]["obs1_nonmissing"],
            markersize=3,
            marker="o",
            linewidth=2,
            linestyle="dotted",
            color="black",
            label="data for experiments",
        )
    if errorbars == True:
        std1 = input["config"]["std1"][input["config"]["obs_data"][0, :, 2] == 1]
        ax_infc.errorbar(
            timepoints1,
            input["config"]["obs1_nonmissing"],
            yerr=std1,
            fmt="o",
            markersize=3,
            linestyle="dotted",
            color="black",
            ecolor="black",
            elinewidth=2.5,
            capsize=4,
            capthick=2,
            alpha=0.5,
            label="data for experiments",
        )
    ax_infc.grid(alpha=0.5)
    ax_infc.tick_params(labelsize=fontsize_tick)
    ax_infc.set_ylabel("Infection Count", fontsize=fontsize_label)
    for spine in ["top", "right", "left"]:
        ax_infc.spines[spine].set_visible(False)

    timepoints2 = input["config"]["timepoints2_nonmissing"]
    ax_seroprev.fill_between(
        timepoints2,
        input["trajectories_cnf_2_95"][0],
        input["trajectories_cnf_2_95"][1],
        color=cnf_colors["20"],
        alpha=0.3,
        label="95% percentile",
    )
    ax_seroprev.fill_between(
        timepoints2,
        input["trajectories_cnf_2_90"][0],
        input["trajectories_cnf_2_90"][1],
        color=cnf_colors["50"],
        alpha=0.5,
        label="90% percentile",
    )
    ax_seroprev.fill_between(
        timepoints2,
        input["trajectories_cnf_2_50"][0],
        input["trajectories_cnf_2_50"][1],
        color=cnf_colors["80"],
        alpha=0.8,
        label="50% percentile",
    )
    ax_seroprev.fill_between(
        timepoints2,
        input["trajectories_pf_2_95"][0],
        input["trajectories_pf_2_95"][1],
        color=pf_colors["20"],
        alpha=0.3,
        label="95% percentile",
    )
    ax_seroprev.fill_between(
        timepoints2,
        input["trajectories_pf_2_90"][0],
        input["trajectories_pf_2_90"][1],
        color=pf_colors["50"],
        alpha=0.5,
        label="90% percentile",
    )
    ax_seroprev.fill_between(
        timepoints2,
        input["trajectories_pf_2_50"][0],
        input["trajectories_pf_2_50"][1],
        color=pf_colors["80"],
        alpha=0.8,
        label="50% percentile",
    )

    if errorbars == False:
        ax_seroprev.plot(
            timepoints2,
            input["config"]["obs2_nonmissing"],
            markersize=3,
            marker="o",
            linewidth=2,
            linestyle="dotted",
            color="black",
            label="data for experiments",
        )
    if errorbars == True:
        std2 = input["config"]["std2"][input["config"]["obs_data"][0, :, 4] == 1]
        ax_seroprev.errorbar(
            timepoints2,
            input["config"]["obs2_nonmissing"],
            yerr=std2,
            fmt="o",
            markersize=3,
            linestyle="dotted",
            color="black",
            ecolor="black",
            elinewidth=2.5,
            capsize=4,
            capthick=2,
            alpha=0.5,
            label="data for experiments",
        )

    ax_seroprev.grid(alpha=0.5)
    ax_seroprev.set_axisbelow(True)
    ax_seroprev.tick_params(labelsize=fontsize_tick)
    ax_seroprev.set_ylabel("Seroprevalence", fontsize=fontsize_label)
    for spine in ["top", "right", "left"]:
        ax_seroprev.spines[spine].set_visible(False)

    # Add legend
    handles = [
        Line2D(xdata=[], ydata=[], color=cnf_colors["50"], lw=20),
        Line2D(xdata=[], ydata=[], color=pf_colors["50"], lw=20),
        Line2D(xdata=[], ydata=[], color="black", lw=2, linestyle="dotted"),
        Line2D(xdata=[], ydata=[], color="black", lw=3),
        Line2D(xdata=[], ydata=[], color=cnf_colors["80"], lw=3),
        Line2D(xdata=[], ydata=[], color=pf_colors["80"], lw=3),
    ]

    labels = [
        "Posterior Draws CNF",
        "Posterior Draws PF",
        "Inference Data",
        "True Parameters",
        "MAP CNF",
        "MAP PF",
    ]

    fig.align_labels()

    if n_params == 6:
        fig.legend(
            handles, labels, ncol=1, fontsize=32, loc="lower right", bbox_to_anchor=(0.95, 0.1)
        )
        axs[0][0].text(
            -0.75,
            1.0,
            "A",
            transform=axs[0][0].transAxes,
            fontsize=40,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="left",
        )
        ax_infc.text(
            -0.4,
            1.0,
            "B",
            transform=ax_infc.transAxes,
            fontsize=40,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="left",
        )
    if n_params == 5:
        fig.legend(
            handles, labels, ncol=1, fontsize=32, loc="lower right", bbox_to_anchor=(0.9, 0.03)
        )
        axs[0][0].text(
            -0.75,
            1.0,
            "A",
            transform=axs[0][0].transAxes,
            fontsize=40,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="left",
        )
        ax_infc.text(
            -0.35,
            1.0,
            "B",
            transform=ax_infc.transAxes,
            fontsize=40,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="left",
        )
    if n_params == 2:
        fig.legend(handles, labels, ncol=3, fontsize=32, loc="lower center")
        axs[0][0].text(
            -0.75,
            1.0,
            "A",
            transform=axs[0][0].transAxes,
            fontsize=40,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="left",
        )
        ax_infc.text(
            -0.4,
            1.0,
            "B",
            transform=ax_infc.transAxes,
            fontsize=40,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="left",
        )

    return fig


def plot_posterior_mountain(posterior_samples):
    """3D KDE surface plots for selected parameter pairs."""
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

    # parameter pairs to plot
    params = [(0, 1), (1, 4)]
    param_names = ["$\\gamma^{-1}$", "$\\kappa^{-1}$", "$\\beta$", "s", "$t_{var}$", "$I_0$"]
    axes = [fig.add_subplot(gs[0, 0], projection="3d"), fig.add_subplot(gs[0, 1], projection="3d")]

    # custom colormap
    cmap = plt.get_cmap("plasma")
    half_cmap = LinearSegmentedColormap.from_list("HalfPlasma", cmap(np.linspace(0.0, 0.7, 256)))

    for ax, (p1, p2) in zip(axes, params):
        data = posterior_samples[:, [p1, p2]]
        kde = gaussian_kde(data.T)

        # grid
        x_grid = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
        y_grid = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

        # surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=half_cmap, edgecolor="none", shade=True)
        ax.set_xlabel(param_names[p1])
        ax.set_ylabel(param_names[p2])
        ax.set_zlabel("")
        ax.set_zticks([])

    # colorbar
    cbar_ax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(surf, cax=cbar_ax)
    cbar.set_ticks([])
    fig.text(0.89, 0.08, "Low", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.text(0.89, 0.92, "High", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.text(0.92, 0.5, "Density", va="center", ha="center", fontsize=14, rotation=90)

    plt.tight_layout()
    return fig


def plot_histograms(
    prior, posterior_cnf, posterior_pf, param_names, post_1_color="#ff7f00", post_2_color="#377eb8"
):
    """Plot prior vs posterior histograms for each parameter."""

    n_params = prior.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(n_params):
        ax = axes[i]
        sns.histplot(prior[:, i], alpha=0.5, label="Prior", color="gray", ax=ax)
        sns.histplot(
            posterior_cnf[:, i],
            alpha=1,
            label="Posterior CNF",
            color=post_1_color,
            ax=ax,
            kde=True,
        )
        sns.histplot(
            posterior_pf[:, i], alpha=1, label="Posterior PF", color=post_2_color, ax=ax, kde=True
        )

        if i == 0 or i == 3:
            ax.set_ylabel("Count", fontsize=36)
        else:
            ax.set_ylabel("", fontsize=36)
        ax.set_xlabel(param_names[i], fontsize=36)
        ax.tick_params(labelsize=28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplot if n_params == 5
    if n_params == 5:
        axes[-1].axis("off")

    # Create a single legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.12),
        fontsize=40,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def input_result_plot(
    posterior_draws_cnf,
    posterior_draws_pf,
    config,
    mode,
    param_names,
    name,
    true_params,
    paramstring="True Param.",
):
    """Compute MAP estimates, trajectories, RMSEs, and package results into a dict."""

    values_cnf = np.transpose(posterior_draws_cnf)
    kernel_cnf = gaussian_kde(values_cnf, bw_method="silverman")
    height_cnf = kernel_cnf.pdf(values_cnf)
    map_cnf = values_cnf[:, np.argmax(height_cnf)]

    values_pf = np.transpose(posterior_draws_pf)
    kernel_pf = gaussian_kde(values_pf, bw_method="silverman")
    height_pf = kernel_pf.pdf(values_pf)
    map_pf = values_pf[:, np.argmax(height_pf)]

    trajectories_cnf_1, trajectories_cnf_2 = model(posterior_draws_cnf, config, mode=mode)
    trajectories_pf_1, trajectories_pf_2 = model(posterior_draws_pf, config, mode=mode)
    n_draws_cnf_1, batch_size_cnf_1, n_timepoints_cnf_1 = trajectories_cnf_1.shape
    n_draws_cnf_2, batch_size_cnf_2, n_timepoints_cnf_2 = trajectories_cnf_2.shape
    n_draws_pf_1, batch_size_pf_1, n_timepoints_pf_1 = trajectories_pf_1.shape
    n_draws_pf_2, batch_size_pf_2, n_timepoints_pf_2 = trajectories_pf_2.shape

    map_cnf_mse = mean_squared_error(config, trajectories_cnf_1, trajectories_cnf_2)
    map_pf_mse = mean_squared_error(config, trajectories_pf_1, trajectories_pf_2)

    trajectories_cnf_1 = trajectories_cnf_1.reshape(
        (n_draws_cnf_1 * batch_size_cnf_1, n_timepoints_cnf_1)
    )
    trajectories_cnf_2 = trajectories_cnf_2.reshape(
        (n_draws_cnf_2 * batch_size_cnf_2, n_timepoints_cnf_2)
    )
    trajectories_pf_1 = trajectories_pf_1.reshape(
        (n_draws_pf_1 * batch_size_pf_1, n_timepoints_pf_1)
    )
    trajectories_pf_2 = trajectories_pf_2.reshape(
        (n_draws_pf_2 * batch_size_pf_2, n_timepoints_pf_2)
    )

    # compute percentiles
    trajectories_cnf_1_50 = np.quantile(trajectories_cnf_1, q=[0.25, 0.75], axis=0)
    trajectories_cnf_1_90 = np.quantile(trajectories_cnf_1, q=[0.05, 0.95], axis=0)
    trajectories_cnf_1_95 = np.quantile(trajectories_cnf_1, q=[0.025, 0.975], axis=0)

    trajectories_cnf_2_50 = np.quantile(trajectories_cnf_2, q=[0.25, 0.75], axis=0)
    trajectories_cnf_2_90 = np.quantile(trajectories_cnf_2, q=[0.05, 0.95], axis=0)
    trajectories_cnf_2_95 = np.quantile(trajectories_cnf_2, q=[0.025, 0.975], axis=0)

    trajectories_pf_1_50 = np.quantile(trajectories_pf_1, q=[0.25, 0.75], axis=0)
    trajectories_pf_1_90 = np.quantile(trajectories_pf_1, q=[0.05, 0.95], axis=0)
    trajectories_pf_1_95 = np.quantile(trajectories_pf_1, q=[0.025, 0.975], axis=0)

    trajectories_pf_2_50 = np.quantile(trajectories_pf_2, q=[0.25, 0.75], axis=0)
    trajectories_pf_2_90 = np.quantile(trajectories_pf_2, q=[0.05, 0.95], axis=0)
    trajectories_pf_2_95 = np.quantile(trajectories_pf_2, q=[0.025, 0.975], axis=0)

    trajectories_true_1, trajectories_true_2 = model(
        true_params[np.newaxis, :], config, mode=mode, batchsize=n_draws_cnf_1
    )
    true_params_mse = mean_squared_error(config, trajectories_true_1, trajectories_true_2)

    df_cnf = pd.DataFrame(posterior_draws_cnf, columns=param_names, dtype="float")
    df_cnf["model"] = "CNF"
    df_pf = pd.DataFrame(posterior_draws_pf, columns=param_names, dtype="float")
    df_pf["model"] = "PF"
    df_true = pd.DataFrame(true_params[np.newaxis, :], columns=param_names, dtype="float")
    df_true[" "] = " "
    df_true["$RMSE_{infc}$"] = true_params_mse[0, 0]
    df_true["$RMSE_{sprev}$"] = true_params_mse[0, 1]
    df_map_cnf = pd.DataFrame(map_cnf[np.newaxis, :], columns=param_names, dtype="float")
    df_map_cnf[" "] = " "
    df_map_cnf["$RMSE_{infc}$"] = map_cnf_mse[0, 0]
    df_map_cnf["$RMSE_{sprev}$"] = map_cnf_mse[0, 1]
    df_map_pf = pd.DataFrame(map_pf[np.newaxis, :], columns=param_names, dtype="float")
    df_map_pf[" "] = " "
    df_map_pf["$RMSE_{infc}$"] = map_pf_mse[0, 0]
    df_map_pf["$RMSE_{sprev}$"] = map_pf_mse[0, 1]
    df_posterior_draws = pd.concat((df_cnf, df_pf), axis=0)
    df_map = pd.concat((df_true, df_map_cnf, df_map_pf), axis=0)
    df_map.index = [paramstring, "$CNF$", "$PF$"]

    for name in df_map.columns:
        if name in ["$\\beta$", "$RMSE_{infc}$", "$RMSE_{sprev}$"]:
            df_map[name] = [f"{x:.4f}" for x in df_map[name].round(4)]
        elif name != " ":
            df_map[name] = [f"{x:.2f}" for x in df_map[name].round(2)]

    dict = {
        "config": config,
        "mode": mode,
        "param_names": param_names,
        "name": name,
        "true_params": true_params,
        "posterior_draws_cnf": posterior_draws_cnf,
        "map_cnf": map_cnf,
        "trajectories_cnf_1_50": trajectories_cnf_1_50,
        "trajectories_cnf_1_90": trajectories_cnf_1_90,
        "trajectories_cnf_1_95": trajectories_cnf_1_95,
        "trajectories_cnf_2_50": trajectories_cnf_2_50,
        "trajectories_cnf_2_90": trajectories_cnf_2_90,
        "trajectories_cnf_2_95": trajectories_cnf_2_95,
        "posterior_draws_pf": posterior_draws_pf,
        "map_pf": map_pf,
        "trajectories_pf_1_50": trajectories_pf_1_50,
        "trajectories_pf_1_90": trajectories_pf_1_90,
        "trajectories_pf_1_95": trajectories_pf_1_95,
        "trajectories_pf_2_50": trajectories_pf_2_50,
        "trajectories_pf_2_90": trajectories_pf_2_90,
        "trajectories_pf_2_95": trajectories_pf_2_95,
        "df_posterior_draws": df_posterior_draws,
        "df_map": df_map,
    }

    return dict
