import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import ot

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.ticker import FuncFormatter

import seaborn as sns

from epmodels.variant_model import seir2v_binomial, seir2v_gaussian, seir2v_forward
from epmodels.sir_model import sir_binomial, sir_gaussian


# ---------------- Simulation Helpers ---------------- #
def model(params, config, mode="binomial", model=None, batchsize = 1):
    """Simulate trajectories given parameters and config."""
    assert params.ndim == 2, "params must be 2D (n_draws, n_params)"
    n_draws, n_params = params.shape

    if model == "sis":
        assert n_params == 2, "SIS has two parameters"
        if mode == "binomial":
            return sis_binomial(params, batchsize, config)
        else:  # gaussian/normal
            return sis_gaussian(params, batchsize, config)

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

# ---------------- Plotting Helpers --------#
def get_kde2d_limits(ax):
    xs, ys = [], []
    for coll in ax.collections:
        if hasattr(coll, "get_paths"):
            for path in coll.get_paths():
                v = path.vertices
                xs.append(v[:, 0])
                ys.append(v[:, 1])
    if not xs:
        return None
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    return xs.min(), xs.max(), ys.min(), ys.max()

def get_param_limits(vals, true_val = None, q_low=1, q_high=99, pad=0.05):
    lo, hi = np.percentile(vals, [q_low, q_high])
    span = hi - lo
    eps = pad * span

    lo -= eps
    hi += eps

    if true_val is not None:
        if true_val < lo:
            lo = true_val - eps
        elif true_val > hi:
            hi = true_val + eps
    return lo, hi

def draw_clipped_line(ax, x0, color, linewidth=3):
    kde_polys = []
    for coll in ax.collections:
        if hasattr(coll, "get_paths"):
            paths = coll.get_paths()
            if len(paths) > 0:
                kde_polys.append(paths[0])

    # Draw one clipped line per KDE polygon
    for poly in kde_polys:
        patch = PathPatch(poly, transform=ax.transData)
        line = ax.axvline(x0, color=color, linewidth=linewidth)
        line.set_clip_path(patch)

# ---------------- Plotting ---------------- #
def plot_results_from_metrics(
    metrics_list,
    config,
    true_params,
    experiment,
    fontsize_label=36,
    fontsize_tick=28,
    errorbars=False,
    ticks_array=None,
    method_colors=None,
):
    default_palette = {
        "CNF": {"base": "#ff7f00", "20": "#fa935f", "50": "#d96b06", "80": "#9d552c"},
        "PF":  {"base": "#377eb8", "20": "#7ab2ed", "50": "#418fd0", "80": "#426b94"},
        "HMC": {"base": "#4daf4a", "20": "#9be29b", "50": "#3b8f3a", "80": "#2f6b2f"},
    }
    if method_colors is None:
        method_colors = default_palette

    methods = []
    dfs = []
    for m in metrics_list:
        name = m["method"]
        colors = method_colors.get(name, default_palette["PF"])
        methods.append(
            {
                "name": name,
                "colors": colors,
                "map": m["map"],
                "posterior_draws": m["posterior_draws"],
                "traj1_q": m["traj1_quantiles"],
                "traj2_q": m["traj2_quantiles"],
            }
        )
        df_m = pd.DataFrame(m["posterior_draws"], columns=m["param_names"])
        df_m["model"] = name
        dfs.append(df_m)

    df = pd.concat(dfs, axis=0)
    param_names = metrics_list[0]["param_names"]
    n_draws, n_params = metrics_list[0]["posterior_draws"].shape

    if n_params == 6:
        low = [-np.inf, -np.inf, -np.inf, -np.inf, 120, 10]
        high = [np.inf, np.inf, np.inf, np.inf, 360, 1000]
    elif n_params == 5:
        low, high = [0.95, 6, 1, 120, 10], [4, 30, 100, 360, 1000]
    elif n_params == 2:
        low, high = [-np.inf, np.inf], [-np.inf, np.inf]

    scale = 3
    n_rows = max(n_params, 4)

    # --- Figure size ---
    if experiment == "sis":
        figsize = (6 * scale, (n_rows - 0.7) * scale)
    else:
        figsize = (8.3 * scale, n_rows * scale)

    fig = plt.figure(figsize=figsize)

    # --- Height ratios ---
    if n_params == 2:
        # last row slightly compressed
        height_ratios = [1] * (n_rows - 1) + [0.3]
    else:
        height_ratios = [1] * n_rows

    # --- Width ratios + number of columns ---
    if experiment == "sis":
        # SIS uses 6 columns
        width_ratios = [1] * 6
        n_cols = 6
    else:
        # SIR/SEIR/etc use 9 columns
        width_ratios = [1] * 6 + [0.3] + [1] * 2
        n_cols = 9

    # --- Create GridSpec ---
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        height_ratios=height_ratios,
        width_ratios=width_ratios
    )

    axs = []
    for i in range(n_params):
        axs.append([fig.add_subplot(gs[i, j]) for j in range(i + 1)])

    for i in range(n_params):
        ax = axs[i][i]
        for m in methods:
            sns.kdeplot(
                data=df.query(f'model == "{m["name"]}"'),
                x=param_names[i],
                fill=True,
                alpha=0.4,
                color=m["colors"]["base"],
                ax=ax,
            )

        if true_params is not None:
            draw_clipped_line(ax, true_params[i], color="black")
            ax.axvline(true_params[i], color="black", linewidth=3)
        for m in methods:
            if m["map"] is not None:
                draw_clipped_line(ax, m["map"][i], color=m["colors"]["80"])

        if i < n_params - 1:
            ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
        else:
            ax.set(ylabel=None, yticklabels=[])

    for i in range(n_params):
        for j in range(i):
            ax = axs[i][j]
            for m in methods:
                sns.kdeplot(
                    data=df.query(f'model == "{m["name"]}"'),
                    x=param_names[j],
                    y=param_names[i],
                    color=m["colors"]["base"],
                    thresh=0.2,
                    fill=True,
                    alpha=0.7,
                    ax=ax,
                    legend=False,
                )

            if true_params is not None:
                ax.scatter(
                    true_params[j], true_params[i],
                    marker="*", s=144,
                    facecolor="black", edgecolor="white", linewidth=1, zorder=6
                )
            for m in methods:
                if m["map"] is not None:
                    ax.scatter(
                        m["map"][j], m["map"][i],
                        marker="*", s=144,
                        facecolor=m["colors"]["80"], edgecolor="white",
                        linewidth=1.0, zorder=6
                    )

            ax.grid(alpha=0.5)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            if j > 0 and i < n_params - 1:
                ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
            if j == 0 and i < n_params - 1:
                ax.set(xlabel=None, xticklabels=[])
            if j > 0 and i == n_params - 1:
                ax.set(ylabel=None, yticklabels=[])

    param_limits = {i: {"min": np.inf, "max": -np.inf} for i in range(n_params)}
    for p in range(n_params):
        vals = df[param_names[p]].values
        lo, hi = get_param_limits(vals, true_params[p])
        param_limits[p]["min"] = lo
        param_limits[p]["max"] = hi

    for i in range(n_params):
        for j in range(i):
            axs[i][j].set_xlim(param_limits[j]["min"], param_limits[j]["max"])
            axs[i][j].set_ylim(param_limits[i]["min"], param_limits[i]["max"])
    for i in range(n_params):
        axs[i][i].set_xlim(param_limits[i]["min"], param_limits[i]["max"])

    if n_params == 2:  # SIS/SIR
        fmts = [".2f", ".1f"]

    elif n_params == 5:  # SEIR2V_reparam
        fmts = [".2f", ".1f", ".1f", ".0f", ".0f"]

    elif n_params == 6:  # SEIR2V_full
        fmts = [".1f", ".1f", ".2f", ".1f", ".0f", ".0f"]

    else:
        raise ValueError("Unknown model: no formatter spec defined")

    def make_formatter(fmt):
        return FuncFormatter(lambda x, pos: f"{x:{fmt}}")

    formatters = [make_formatter(fmt) for fmt in fmts]

    for j in range(n_params):
        lo, hi = axs[-1][j].get_xlim()
        xticks = [lo + 0.2 * (hi - lo), lo + 0.8 * (hi - lo)]

        # bottom row
        axs[-1][j].set_xticks(xticks)
        axs[-1][j].xaxis.set_major_formatter(formatters[j])
        # propagate upward
        for i in range(j, n_params - 1):
            axs[i][j].set_xticks(xticks)
    
    for i in range(1,n_params):
            lo, hi = axs[i][0].get_ylim()
            yticks = [lo + 0.2 * (hi - lo), lo + 0.8 * (hi - lo)]
            # first column
            axs[i][0].set_yticks(yticks)
            axs[i][0].yaxis.set_major_formatter(formatters[i])
            #propagate right
            for j in range(1, i - 1): #exclude diagonal y ticks
                axs[i][j].set_yticks(yticks)

    # Axis labels
    for i in range(n_params):
        axs[-1][i].set_xlabel(param_names[i], fontsize=fontsize_label)
        axs[-1][i].tick_params(labelsize=fontsize_tick)
        axs[-1][i].tick_params(axis="x", labelrotation=45)
    for i in range(n_params):
        axs[i][0].set_ylabel(param_names[i], fontsize=fontsize_label)
        axs[i][0].tick_params(labelsize=fontsize_tick)

    # Trajectory axis creation
    if experiment == "sis":
        # SIS uses 6 columns
        if n_params > 2:
            ax_trajectories = fig.add_subplot(gs[0:3, 4:6])
        else:
            ax_trajectories = fig.add_subplot(gs[0:2, 4:6])
    else:
        # SIR/SEIR/etc use 9 columns
        if n_params > 2:
            ax_trajectories = fig.add_subplot(gs[0:3, 4:9])
        else:
            ax_trajectories = fig.add_subplot(gs[0:2, 4:9])

    for spine in ["top", "right", "bottom", "left"]:
        ax_trajectories.spines[spine].set_visible(False)

    ax_trajectories.set(ylabel=None, xticklabels=[], yticklabels=[])
    ax_trajectories.tick_params(labelcolor="w", top=False, bottom=False, left=False, right=False)
    labelpad = 20 if n_params > 2 else  8
    ax_trajectories.set_xlabel("Time t in days", fontsize=fontsize_label, labelpad=labelpad) 
    ax_trajectories.xaxis.set_label_coords(0.5, -0.15 if n_params > 2 else -0.10)
    x = 0.1 if n_params > 2 else 0
    ax_trajectories.margins(x=x)

    ax_infc = fig.add_subplot(gs[0:3, 4:6]) if n_params > 2 else fig.add_subplot(gs[0:2, 4:6])
    if experiment != "sis": 
        ax_seroprev = fig.add_subplot(gs[0:3, 7:9]) if n_params > 2 else fig.add_subplot(gs[0:2, 7:9])

    timepoints1 = config["timepoints1_nonmissing"]
    timepoints2 = None if experiment == "sis" else config["timepoints2_nonmissing"]

    for m in methods:
        colors = m["colors"]
        q1 = m["traj1_q"]

        ax_infc.fill_between(timepoints1, q1["95"][0], q1["95"][1],
                             color=colors["20"], alpha=0.2)
        ax_infc.fill_between(timepoints1, q1["90"][0], q1["90"][1],
                             color=colors["50"], alpha=0.5)
        ax_infc.fill_between(timepoints1, q1["50"][0], q1["50"][1],
                             color=colors["80"], alpha=1.0)

        if experiment != "sis" and timepoints2 is not None:
            q2 = m["traj2_q"]
            ax_seroprev.fill_between(timepoints2, q2["95"][0], q2["95"][1],
                                     color=colors["20"], alpha=0.3)
            ax_seroprev.fill_between(timepoints2, q2["90"][0], q2["90"][1],
                                     color=colors["50"], alpha=0.5)
            ax_seroprev.fill_between(timepoints2, q2["50"][0], q2["50"][1],
                                     color=colors["80"], alpha=0.8)

    # Data overlay
    obs1 = config["obs1_nonmissing"]
    if not errorbars:
        ax_infc.plot(timepoints1, obs1, markersize=3, marker="o",
                     linewidth=2, linestyle="dotted", color="black")
    else:
        std1 = config["std1"][config["obs_data"][0, :, 2] == 1]
        ax_infc.errorbar(timepoints1, obs1, yerr=std1, fmt="o", markersize=3,
                         linestyle="dotted", color="black", ecolor="black",
                         elinewidth=2.5, capsize=4, capthick=2, alpha=0.5)

    ax_infc.grid(alpha=0.5)
    ax_infc.tick_params(labelsize=fontsize_tick)
    ax_infc.set_ylabel("Infection Count", fontsize=fontsize_label)
    for spine in ["top", "right", "left"]:
        ax_infc.spines[spine].set_visible(False)

    if experiment != "sis" and timepoints2 is not None:
        obs2 = config["obs2_nonmissing"]
        if not errorbars:
            ax_seroprev.plot(timepoints2, obs2, markersize=3, marker="o",
                             linewidth=2, linestyle="dotted", color="black")
        else:
            std2 = config["std2"][config["obs_data"][0, :, 4] == 1]
            ax_seroprev.errorbar(timepoints2, obs2, yerr=std2, fmt="o", markersize=3,
                                 linestyle="dotted", color="black", ecolor="black",
                                 elinewidth=2.5, capsize=4, capthick=2, alpha=0.5)

        ax_seroprev.grid(alpha=0.5)
        ax_seroprev.set_axisbelow(True)
        ax_seroprev.tick_params(labelsize=fontsize_tick)
        ax_seroprev.set_ylabel("Seroprevalence", fontsize=fontsize_label)
        for spine in ["top", "right", "left"]:
            ax_seroprev.spines[spine].set_visible(False)

    handles = []
    labels = []
    for m in methods:
        handles.append(Line2D([], [], color=m["colors"]["50"], lw=20))
        labels.append(f"Posterior Draws {m['name']}")
    handles.extend(
        [
            Line2D([], [], color="black", lw=2, linestyle="dotted"),
            Line2D([], [], color="black", lw=3),
        ]
    )
    labels.extend(["Inference Data", "True Parameters"])
    for m in methods:
        handles.append(Line2D([], [], color=m["colors"]["80"], lw=3))
        labels.append(f"MAP {m['name']}")

    fig.align_labels()

    if n_params == 6:
        fig.legend(handles, labels, ncol=1, fontsize=32, loc="lower right", bbox_to_anchor=(0.95, 0.1))
        axs[0][0].text(-0.75, 1.0, "A", transform=axs[0][0].transAxes,
                       fontsize=40, fontweight="bold", va="top", ha="left")
        ax_infc.text(-0.5, 1.0, "B", transform=ax_infc.transAxes,
                     fontsize=40, fontweight="bold", va="top", ha="left")
    elif n_params == 5:
        fig.legend(handles, labels, ncol=1, fontsize=32, loc="lower right", bbox_to_anchor=(0.9, 0.03))
        axs[0][0].text(-0.75, 1.0, "A", transform=axs[0][0].transAxes,
                       fontsize=40, fontweight="bold", va="top", ha="left")
        ax_infc.text(-0.35, 1.0, "B", transform=ax_infc.transAxes,
                     fontsize=40, fontweight="bold", va="top", ha="left")
    else:
        fig.legend(handles, labels, ncol=3, fontsize=32, loc="lower center")
        axs[0][0].text(-0.75, 1.0, "A", transform=axs[0][0].transAxes,
                       fontsize=40, fontweight="bold", va="top", ha="left")
        if experiment == "sis":
            ax_infc.text(-0.5, 1.0, "B", transform=ax_infc.transAxes,
                        fontsize=40, fontweight="bold", va="top", ha="left")
        else:
            ax_infc.text(-0.7, 1.0, "B", transform=ax_infc.transAxes,
                            fontsize=40, fontweight="bold", va="top", ha="left")

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

# ---------------Metrics------------------- #
def scale_parameters(params, mean=None, std=None):
    params = np.asarray(params)

    # Fit mode
    if mean is None or std is None:
        mean = params.mean(axis=0)
        std = params.std(axis=0)
    
    # Avoid division by zero
    #print(mean,std)
    std_safe = np.where(std == 0, 1.0, std)

    z_params = (params - mean) / std_safe
    return z_params, mean, std_safe

def scale_parameters_inv(z_params, mean, std):
    return z_params * std + mean

def compute_method_metrics(
    method_name,
    posterior_draws,
    true_params,
    config,
    mode,
    param_names,
    experiment,
    n_pairs=10000,
):
    print(f"Computing metrics for {method_name}")
    if method_name == "True":
        kernel = None
        map_estimate = true_params
        cov_scaled = None
        eig_vals = np.zeros(true_params.shape)
    else:
        posterior_draws_scaled = posterior_draws.copy()
        posterior_draws_scaled, mean, std = scale_parameters(posterior_draws_scaled)
        values = posterior_draws_scaled.T
        cov_scaled = np.cov(posterior_draws_scaled, rowvar=False)
        eig_vals = np.linalg.eigvalsh(cov_scaled)
        print(f"Approximating kernel for {method_name} in experiment {experiment}")
        kernel = gaussian_kde(values, bw_method="silverman")
        logp = kernel.logpdf(values)
        map_scaled = values[:, np.argmax(logp)]
        map_estimate = scale_parameters_inv(map_scaled, mean, std)

    if experiment == "sis":
        traj1 = model(posterior_draws, config, mode=mode, model="sis")
        traj2 = traj1
    else:
        traj1, traj2 = model(posterior_draws, config, mode=mode)

    n_draws, batch_size, T1 = traj1.shape
    _, _, T2 = traj2.shape

    # flatten for quantiles + energy score
    traj1_flat = traj1.reshape(n_draws * batch_size, T1)
    traj2_flat = traj2.reshape(n_draws * batch_size, T2)

    def quantiles(x):
        return {
            "50": np.quantile(x, [0.25, 0.75], axis=0),
            "90": np.quantile(x, [0.05, 0.95], axis=0),
            "95": np.quantile(x, [0.025, 0.975], axis=0),
        }

    traj1_q = quantiles(traj1_flat)
    traj2_q = quantiles(traj2_flat)

    q_low, q_high = 0.025, 0.975
    lowers = np.quantile(posterior_draws, q_low, axis=0)
    uppers = np.quantile(posterior_draws, q_high, axis=0)

    obs1 = config["obs1_nonmissing"]
    obs2 = None if experiment == "sis" else config["obs2_nonmissing"]

    loc1 = np.linalg.norm(traj1_flat - obs1[None, :], axis=1).mean()

    if experiment == "sis":
        loc2 = np.nan
    else:
        loc2 = np.linalg.norm(traj2_flat - obs2[None, :], axis=1).mean()

    rng = np.random.default_rng(2026)

    # infc dispersion
    idx_i = rng.integers(0, traj1_flat.shape[0], n_pairs)
    idx_j = rng.integers(0, traj1_flat.shape[0], n_pairs)
    diffs1 = traj1_flat[idx_i] - traj1_flat[idx_j]
    disp1 = np.linalg.norm(diffs1, axis=1).mean()

    # sprev dispersion
    if experiment == "sis":
        disp2 = np.nan
    else:
        idx_i = rng.integers(0, traj2_flat.shape[0], n_pairs)
        idx_j = rng.integers(0, traj2_flat.shape[0], n_pairs)
        diffs2 = traj2_flat[idx_i] - traj2_flat[idx_j]
        disp2 = np.linalg.norm(diffs2, axis=1).mean()

    # energy scores
    energy1 = loc1 - 0.5 * disp1
    energy2 = np.nan if experiment == "sis" else (loc2 - 0.5 * disp2)

    if experiment == "sis":
        energy_total = energy1
        loc_total = loc1
        disp_total = disp1
    else:
        y_obs_total = np.concatenate([obs1, obs2])
        y_pred_total = np.concatenate([traj1_flat, traj2_flat], axis=1)

        loc_total = np.linalg.norm(y_pred_total - y_obs_total[None, :], axis=1).mean()

        idx_i = rng.integers(0, y_pred_total.shape[0], n_pairs)
        idx_j = rng.integers(0, y_pred_total.shape[0], n_pairs)
        diffs_total = y_pred_total[idx_i] - y_pred_total[idx_j]
        disp_total = np.linalg.norm(diffs_total, axis=1).mean()

        energy_total = loc_total - 0.5 * disp_total

    return {
        "method": method_name,
        "experiment": experiment,
        "posterior_draws": posterior_draws,
        "kernel": kernel,
        "cov_scaled": cov_scaled,
        "eig_vals": eig_vals, 
        "map": map_estimate,
        "lowers": lowers,
        "uppers": uppers,

        # split metrics
        "loc_infc": loc1,
        "loc_sprev": loc2,
        "loc_total": loc_total,
        "disp_infc": disp1,
        "disp_sprev": disp2,
        "disp_total": disp_total,
        "energy_infc": energy1,
        "energy_sprev": energy2,
        "energy_total": energy_total,

        # trajectories + quantiles
        "traj1_quantiles": traj1_q,
        "traj2_quantiles": traj2_q,
        "traj1_flat": traj1_flat,
        "traj2_flat": traj2_flat,

        "param_names": param_names,
    }

def build_method_row(
    experiment_id,
    method_metrics,
    ws_dict,
    param_names,
    all_methods,
):
    method = method_metrics["method"]

    row = {
        "experiment_id": experiment_id,
        "method": method,
    }

    def sig(x, digits=4):
        if x is None or np.isnan(x):
            return np.nan
        return float(f"{x:.{digits}g}")
    for pname, w in zip(param_names, method_metrics["map"]):
        row[f"map_{pname}"] = sig(w)
    for pname, w in zip(param_names, method_metrics["lowers"]):
        row[f"lower_{pname}"] = sig(w)
    for pname, w in zip(param_names, method_metrics["uppers"]):
        row[f"upper_{pname}"] = sig(w)
    for pname, cov in zip(param_names, method_metrics["covered"]):
        row[f"covered_{pname}"] = sig(cov)

    row["covered_all"] = bool(method_metrics["covered_all"])

    row["energy_infc"] = sig(method_metrics["energy_infc"])
    row["energy_sprev"] = (
        np.nan if np.isnan(method_metrics["energy_sprev"]) else sig(method_metrics["energy_sprev"])
    )
    row["energy_total"] = sig(method_metrics["energy_total"])

    for other in all_methods:
        colname_ws = f"ws_{method}_vs_{other}"

        if other == method:
            row[colname_ws] = np.nan
            continue

        key_AB = (method, other)
        key_BA = (other, method)

        if key_AB in ws_dict:
            row[colname_ws] = sig(ws_dict[key_AB])
        elif key_BA in ws_dict:
            row[colname_ws] = sig(ws_dict[key_BA])
        else:
            row[colname_ws] = np.nan

    return row

def load_method_variants(method_name, experiment, k, samplesize):

    base_pattern = f"./results/{method_name}_samples/{experiment}/{method_name.lower()}_posterior_{experiment}_{k}"
    all_files = sorted(glob.glob(base_pattern + "*.csv"))

    # Regex: match exactly _k.csv or _k_<digits>.csv
    # Example: seir2v_reparam_dense_1.csv or seir2v_reparam_dense_1_2.csv
    regex = re.compile(rf"{experiment}_{k}(?:_\d+)?\.csv$")

    filtered = [f for f in all_files if regex.search(os.path.basename(f))]

    method_draws = {}
    for idx, file in enumerate(filtered):
        name = method_name if idx == 0 else f"{method_name}_{idx+1}"
        method_draws[name] = np.array(pd.read_csv(file))[:samplesize, :]
        print(f"  Loaded {name}: {os.path.basename(file)}")

    return method_draws

def build_experiment_table(
    valid_set,
    true_params,
    param_names,
    load_fn,
    mode="normal",
    experiment = "seir2v_full_dense",
    n_pairs=10000,
    samplesize=10000,
    savefig = False
):
    rows = []

    all_methods = ["CNF", "PF", "HMC"]

    for k in valid_set:
        print(f"=== Experiment {k} ===")

        try:
            config_name = f"./data/final_data/{experiment}/{experiment}_{k}.csv"
            print(config_name)
            experiment_name = f"{experiment}_{k}"
            config = load_fn(config_name, experiment_name)
        except FileNotFoundError:
            print(f"  -> Missing config for experiment {k}, skipping.")
            continue

        method_draws = {}

        # Load CNF variants
        cnf_draws = load_method_variants("CNF", experiment, k, samplesize)
        method_draws.update(cnf_draws)

        # Load PF variants
        pf_draws = load_method_variants("PF", experiment, k, samplesize)
        method_draws.update(pf_draws)

        # Load HMC variants
        hmc_draws = load_method_variants("HMC", experiment, k, samplesize)
        method_draws.update(hmc_draws)

        if len(method_draws) < 1:
            print(f"  -> Not enough posterior sources for experiment {k}, skipping.")
            continue

        # True parameters
        theta_true = true_params[k - 1]
        n_draws = method_draws["CNF"].shape[0]
        true_draws = np.repeat(theta_true[None, :], repeats=n_draws, axis=0)
        method_draws["True"] = true_draws

        metrics = {}
        for m, draws in method_draws.items():
            metrics[m] = compute_method_metrics(
                m, draws, theta_true, config,  mode,
                param_names, experiment=experiment, n_pairs=n_pairs
            )
        available = list(method_draws.keys())
        if savefig:
            metr = [metrics[a] for a in available if a != "True"]

            f = plot_results_from_metrics(metr, config, theta_true, experiment)
            f.savefig(f"./results/figures/{experiment}/{experiment}_{k}.png")
            f.savefig(f"./results/figures/{experiment}/{experiment}_{k}.pdf", bbox_inches='tight')         

        for m, mm in metrics.items():
            draws = mm["posterior_draws"]
            lowers = mm["lowers"]
            uppers = mm["uppers"]

            # marginal coverage
            covered = (lowers <= theta_true) & (theta_true <= uppers)
            mm["covered"] = covered

            # joint coverage
            mm["covered_all"] = bool(np.all(covered))

        ws = {}
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                A = available[i]
                B = available[j]
                if A == "True" or B == "True":
                    continue
                ws[(A, B)], W_A_B = compare_posteriors_wasserstein(metrics[A], metrics[B])
                pd.DataFrame(W_A_B).to_csv(f"./results/Wasserstein/{experiment}/W_{experiment}_{k}_{A}_{B}.csv")
                
        for m in available:
            rows.append(build_method_row(k, metrics[m], ws, param_names, all_methods))

        print(f"  -> Experiment {k} done.")

    return pd.DataFrame(rows)

def geometric_mean(x):
    x = np.asarray(x)
    x = x[x > 0]  # avoid log(0)
    return float(np.exp(np.mean(np.log(x))))

def aggregate_experiment(df, param_names):

    rows = []

    experiment_types = df["experiment_id"].unique()
    covered_cols = [f"covered_{p}" for p in param_names]

    for exp in experiment_types:

        df_exp = df[df["experiment_id"] == exp]

        df_cnf = df_exp[df_exp["method"] == "CNF"]
        df_pf  = df_exp[df_exp["method"] == "PF"]

        if df_cnf.empty or df_pf.empty:
            continue

        # Extract the single inflation and MMD values
        mmd_col       = "mmd_CNF_vs_PF"
        mmd_val       = float(df_cnf[mmd_col].iloc[0])

        energy_cnf = float(df_cnf["energy_total"].mean())
        energy_pf  = float(df_pf["energy_total"].mean())

        cov_cnf = float(df_cnf[covered_cols].astype(float).mean(axis=1).mean())
        cov_pf  = float(df_pf[covered_cols].astype(float).mean(axis=1).mean())

        rows.append({
            "experiment_id": exp,
            "mmd_pf_cnf": mmd_val,
            "energy_cnf": energy_cnf,
            "energy_pf": energy_pf,
            "coverage_cnf": cov_cnf,
            "coverage_pf": cov_pf,
        })

    rows = pd.DataFrame(rows)

    # Summary row
    avg_row = {
        "experiment_id": -1,
        "mmd_pf_cnf": rows["mmd_pf_cnf"].mean(),
        "energy_cnf": rows["energy_cnf"].mean(),
        "energy_pf": rows["energy_pf"].mean(),
        "coverage_cnf": rows["coverage_cnf"].mean(),
        "coverage_pf": rows["coverage_pf"].mean(),
    }

    rows = pd.concat([rows, pd.DataFrame([avg_row])], ignore_index=True)

    return rows

def directional_wasserstein_from_pools(A_pool, B_pool, n_sub=1000):
    # subsample
    nA = min(len(A_pool), n_sub)
    nB = min(len(B_pool), n_sub)
    X = A_pool[np.random.choice(len(A_pool), nA, replace=False)]
    Y = B_pool[np.random.choice(len(B_pool), nB, replace=False)]

    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)
    M = ot.dist(X, Y, metric='euclidean')
    W1_sq = ot.emd2(a, b, M)
    return float(np.sqrt(W1_sq))

def compare_posteriors_wasserstein(method_A, method_B,
                                   n_sub=1000, n_iter=20,
                                   mode="log",  # "log" or "whiten" or "log+whiten"
                                   ):
    A = method_A["posterior_draws"]  # (N, d)
    B = method_B["posterior_draws"]

    if mode in ("log", "log+whiten"):
        A = to_log_space(A)
        B = to_log_space(B)

    if mode in("whiten", "log+whiten"):
        X = np.vstack([A, B])
        mu = X.mean(axis=0)
        Sigma = np.cov(X, rowvar=False)
        L = np.linalg.cholesky(Sigma)
        L_inv = np.linalg.inv(L)
        A = (A - mu) @ L_inv.T
        B = (B - mu) @ L_inv.T

    #print(A, B)

    # 3) now just subsample + Wasserstein in this chosen space
    W_AB = [directional_wasserstein_from_pools(A, B, n_sub)
            for _ in range(n_iter)]
    W_AA = [directional_wasserstein_from_pools(A, A, n_sub)
            for _ in range(n_iter)]
    W_BB = [directional_wasserstein_from_pools(B, B, n_sub)
            for _ in range(n_iter)]

    #W_sym = [(W_AA[j] + W_BB[j]) / 2 for j in range(n_iter)]
    D_adj = np.mean(W_AB) - (np.mean(W_AA) + np.mean(W_BB)) / 2

    return float(D_adj), np.column_stack((W_AB, W_AA, W_BB))

def to_log_space(samples, eps=1e-8):
    samples = np.asarray(samples)
    return np.log(samples + eps)
