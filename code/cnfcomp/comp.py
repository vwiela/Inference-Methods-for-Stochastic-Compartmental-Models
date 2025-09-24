import numpy as np
import pandas as pd


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.colors import to_rgba
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

from scipy import stats
from scipy.optimize import minimize
from scipy.stats import binom, truncnorm, gaussian_kde
import seaborn as sns
from seaborn.utils import desaturate
from seaborn.external import husl

import math

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epmodels.variant_model import seir2v_binomial, seir2v_gaussian, seir2v_forward
from epmodels.sir_model import sir_binomial, sir_gaussian
from .coverage_utils import compute_trajectory_bounds, coverage_dataframe
from .coverage_utils import COVERAGE_QUANTILES, ALPHAS


n = 1000

def _cmap_from_color(color):
    """Return a sequential colormap given a color seed."""
    # Like so much else here, this is broadly useful, but keeping it
    # in this class to signify that I haven't thought overly hard about it...
    r, g, b, _ = to_rgba(color)
    h, s, _ = husl.rgb_to_husl(r, g, b)
    xx = np.linspace(-1, 1, int(1.15 * 256))[:256]
    ramp = np.zeros((256, 3))
    ramp[:, 0] = h
    ramp[:, 1] = s * np.cos(xx)
    ramp[:, 2] = np.linspace(35, 80, 256)
    colors = np.clip([husl.husl_to_rgb(*hsl) for hsl in ramp], 0, 1)
    return mpl.colors.ListedColormap(colors[::-1])

def model(
    params,
    config,
    mode = "binomial",
    batchsize = 1
    ):
    """Simulates trajectories given a set of parameters. 

    params : np.ndarray of shape (n_draws, n_params)
        Parameters to be simulated
    config : dictionary
        Configuration parameters for the simulation
    mode : string 
        Mode of the simulation, either "binomial", "gaussian" or "normal" (last two being the same)
        Default: "binomial"
    batchsize : int
        Number of simulations per parameter, to capture the stochastic nature of the model
        Default: 1 
    

    Returns
    -------
    arr1 : np.ndarray of shape (n_draws, batch_size, n_timepoints1)
    arr2 : np.ndarray of shape (n_draws, batch_size, n_timepoints2)

    """

    # Ensure correct shape
    assert (
        len(params.shape)
    ) == 2, "Shape of `param_draws` should be 2 dimensional!"

    # Obtain n_draws and n_params
    n_draws, n_params = params.shape

    # Ensure correct mode
    '''valid_modes = set(["binomial", "gaussian", "normal"])
    assert(
        mode in valid_modes
    ) == True, "Mode needs to be `binomial`, `gaussian` or `normal'''
    
    n_draws, n_params = params.shape
    if n_params == 2:
        if (mode == "binomial"):
            arr1, arr2 = sir_binomial(params, batchsize, config)
        if (mode == "gaussian" or mode == "normal"):
            arr1, arr2 = sir_gaussian(params, batchsize, config)
    else:
        if (mode == "binomial"):
            arr1, arr2 = seir2v_binomial(params, batchsize, config)
        if (mode == "gaussian" or mode == "normal"):
            arr1, arr2 = seir2v_gaussian(params, batchsize, config)

    return arr1, arr2

def mean_squared_error(
    config, 
    trajectories_1, 
    trajectories_2,
    ):
    """Calculates the mean squared error for given trajectories, averaged over `n_batchsize`

    config : dictionary containing model parameters and the true data
    trajectories_1   : np.ndarray of shape (n_draws, batch_size, n_timepoints1)
    trajectories_2   : np.ndarray of shape (n_draws, batch_size, n_timepoints2)
    """

    # Ensure correct shape
    assert (
        len(trajectories_1.shape)
    ) == 3, "Shape of `trajectories_1` should be 3 dimensional!"

    # Ensure correct shape
    assert (
        len(trajectories_2.shape)
    ) == 3, "Shape of `trajectories_2` should be 3 dimensional!"

    # Ensure config uses correct number of timepoints
    # ToDo

    n_draws_1, batch_size_1, n_timepoints1 = trajectories_1.shape
    n_draws_2, batch_size_2, n_timepoints2 = trajectories_2.shape

    mse = np.zeros((n_draws_1,2))
    mse1_out, mse2_out = np.zeros((n_draws_1,1)), np.zeros((n_draws_1,1))

    # Ensure correct dimensions of input trajectories
    assert (
        n_draws_1 == n_draws_2 and batch_size_1 == batch_size_2
    ), "n_draws and batch_size should be the same for trajectories_1 and trajectories_2!"

    for n in range(n_draws_1):
        mse1, mse2 = np.zeros(batch_size_1), np.zeros(batch_size_1)

        for batch in range(batch_size_1):
            mse1[batch] = math.sqrt(((trajectories_1[n,batch,:] - config["obs1_nonmissing"])**2).sum(axis=0) / n_timepoints1)
            mse2[batch] = math.sqrt(((trajectories_2[n,batch,:] - config["obs2_nonmissing"])**2).sum(axis=0) / n_timepoints2)
            
        mse[n,0] = mse1.mean(axis=0)
        mse[n,1] = mse2.mean(axis=0)
        mse1_out[n,0]=mse1.mean(axis=0)
        mse2_out[n,0]=mse2.mean(axis=0)

    return mse

def plot_percentile_bands(ax, timepoints, bounds_dict, cmap, label_prefix="", zorder=None):
    for i, (p, bounds) in enumerate(sorted(bounds_dict.items(), reverse=True)):
        lower, upper = bounds
        alpha = ALPHAS[p]
        color = cmap(alpha)
        ax.fill_between(
            timepoints, lower, upper,
            color=color,
            alpha=alpha,
            label=f"{label_prefix} {p}% percentile"
            #zorder=zorder if zorder is not None else (10 - i)
        )

def plot_posteriors_2d3(
    input,
    height=3,
    post_1_color='#ff7f00',
    post_2_color='#377eb8',
    table_1_color = '#ffefe0',
    table_2_color = '#eff4f9',
    fontsize_label = 20,
    fontsize_tick = 18
):

    # Obtain n_draws and n_params ONLY works if all posterior draws have the same shape
    n_draws_1, n_params_1 = input["posterior_draws_cnf"].shape

    # Attempt to determine parameter names
    if input["param_names"] is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params_1 + 1)]
    else:
        param_names = input["param_names"]

    cmap1 = _cmap_from_color(post_1_color)
    cmap2 = _cmap_from_color(post_2_color)
    palette = {"CNF": cmap1(0.2), "PF": cmap2(0.2)}


    figsize = 7.5 * height, (n_params_1 + 0.3)* height
    fig = plt.figure(figsize = figsize)
    # Define height ratios to control spacing
    height_ratios = [1] * n_params_1 + [0.3]# + [1] * n_params_1  # Fourth row (spacer) is half the height of the others
    #gs = fig.add_gridspec(2 * n_params_1 + 1, 7, height_ratios=height_ratios)
    width_ratios = [1] * 7 + [0.5]
    gs = fig.add_gridspec(n_params_1+1, 8, height_ratios=height_ratios, width_ratios = width_ratios)

    axs = []
    for i in range(n_params_1):
            axs.append([fig.add_subplot(gs[i, j]) for j in range(i+1)])
    
    cnf_df = input["df_posterior_draws"].query('model == "CNF"').drop(columns="model")
    pf_df = input["df_posterior_draws"].query('model == "PF"').drop(columns="model")
    percentiles = [0.005, 0.995]
    bounds_cnf = np.quantile(cnf_df, q=percentiles, axis=0)
    bounds_pf = np.quantile(pf_df, q=percentiles, axis=0)
    lower_bound = np.minimum(bounds_cnf[0], bounds_pf[0])
    upper_bound = np.maximum(bounds_cnf[1], bounds_pf[1])
    bounds = np.vstack([lower_bound, upper_bound])

    print(bounds)

    for i in range(n_params_1):
        sns.violinplot(data=input["df_posterior_draws"], y=param_names[i], hue='model', palette=palette, split=True, inner=None, fill=True,
                    width=0.7,  ax=axs[i][i], legend=False, linewidth=1)
        if i > 0:
            axs[i][i].set_yticks([])
            axs[i][i].set_ylabel(None)

        patch_left = PathPatch(axs[i][i].collections[0].get_paths()[0], transform=axs[i][i].transData)
        patch_right = PathPatch(axs[i][i].collections[1].get_paths()[0], transform=axs[i][i].transData)

        if input["true_params"] is not None:
            line_left_true = axs[i][i].axhline(y=input["true_params"] [i], color="black")
            line_right_true = axs[i][i].axhline(y=input["true_params"] [i], color="black")
            line_left_true.set_clip_path(patch_left)
            line_right_true.set_clip_path(patch_right)

        if input["map_cnf"] is not None:
            line_left_map1 = axs[i][i].axhline(y=input["map_cnf"][i], color=cmap1(0.9))
            line_left_map1.set_clip_path(patch_left)

        if input["map_pf"] is not None:
            line_right_map2 = axs[i][i].axhline(y=input["map_pf"][i], color=cmap2(0.9))
            line_right_map2.set_clip_path(patch_right)

        for spine in ['top', 'right', 'bottom']: 
            axs[i][i].spines[spine].set_visible(False)

        lower, upper = bounds[:, i]
        axs[i][i].set_ylim(lower, upper)



    for i in range(n_params_1):
        for j in range(i):
            sns.kdeplot(data=input["df_posterior_draws"].query('model == "CNF"'), x=param_names[j], y=param_names[i], color=post_1_color, 
                        levels=4, thresh=.2, fill=True, alpha = 0.7, ax=axs[i][j], legend=False)
    
            sns.kdeplot(data=input["df_posterior_draws"].query('model == "PF"'), x=param_names[j], y=param_names[i], color=post_2_color, 
                        levels=4, thresh=.2, fill=True, alpha = 0.7, ax=axs[i][j], legend=False)

            axs[i][j].grid(alpha=0.5)
            for spine in ['top', 'right']:
                axs[i][j].spines[spine].set_visible(False)
            if j > 0 and i < n_params_1 - 1:
                axs[i][j].set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
            if j == 0 and i < n_params_1 - 1:
                axs[i][j].set(xlabel=None, xticklabels=[])
            if j > 0 and i == n_params_1 - 1:
                axs[i][j].set(ylabel=None, yticklabels=[])

            x_lower, x_upper = bounds[:, j]
            y_lower, y_upper = bounds[:, i]
            axs[i][j].set_xlim(x_lower, x_upper)
            axs[i][j].set_ylim(y_lower, y_upper)


        """Add labels to the left and bottom Axes."""
    for i in range(n_params_1):
        axs[-1][i].set_xlabel(param_names[i], fontsize = fontsize_label)
        axs[-1][i].tick_params(labelsize = fontsize_tick)
    for i in range(n_params_1):
        axs[i][0].set_ylabel(param_names[i], fontsize = fontsize_label)
        axs[i][0].tick_params(labelsize = fontsize_tick)

    axs[0][0].text(-0.5, 1.0, 'A', transform=axs[0][0].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')


    ax_table = fig.add_subplot(gs[2:4,4:8])
    for spine in ['top', 'right', 'bottom', 'left']: 
        ax_table.spines[spine].set_visible(False)
    ax_table.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax_table.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    ax_table.text(-0.24, 0.88, 'C', transform=ax_table.transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
    
    row_colours = ["whitesmoke"] * len(input["df_map"].columns)
    cell_colours = [['w'] * len(input["df_map"].columns), [table_1_color] * len(input["df_map"].columns), [table_2_color] * len(input["df_map"].columns)]
    table = ax_table.table(cellText=input["df_map"].values, colLabels=input["df_map"].columns, rowLabels=input["df_map"].index, 
                          #bbox=([box.x0, box.y0, box.width * 0.8, box.height * 0.9]),
                          bbox=[0.0,0.5,1,0.4],  
                          #loc="upper center",
                          cellColours=cell_colours, rowColours = row_colours, colColours=row_colours, cellLoc = "center")
    table.auto_set_font_size(False)

    # Set font sizes
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_fontsize(fontsize_label)
        elif col == -1:  # Row labels
            cell.set_fontsize(fontsize_label)
        else:  # Data rows
            cell.set_fontsize(fontsize_tick)
    table.auto_set_column_width(col=list(range(len(input["df_map"].columns))))

    

    ax_trajectories = fig.add_subplot(gs[0:2,3:7])
    for spine in ['top', 'right', 'bottom', 'left']: 
        ax_trajectories.spines[spine].set_visible(False)
    ax_trajectories.set(ylabel=None, xticklabels=[], yticklabels=[])
    ax_trajectories.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax_trajectories.set_xlabel('Time t in days', fontsize=fontsize_label, labelpad=30)
   
    ax_trajectories.margins(x=0.1)

    ax_infc = fig.add_subplot(gs[0:2,3:5])
    ax_seroprev = fig.add_subplot(gs[0:2,5:7])

    # First Plot
    #ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_95"][0], input["trajectories_pf_1_95"][1], color = cmap2(0.2), alpha = 0.2, label = "95% percentile")
    #ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_90"][0], input["trajectories_pf_1_90"][1], color = cmap2(0.5), alpha = 0.5, label = "90% percentile")
    #ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_50"][0], input["trajectories_pf_1_50"][1], color = cmap2(0.8), alpha = 0.8, label = "50% percentile")  
    
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_cnf_1_95"][0], input["trajectories_cnf_1_95"][1], color = cmap1(0.2), alpha = 0.2, label = "95% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_cnf_1_90"][0], input["trajectories_cnf_1_90"][1], color = cmap1(0.5), alpha = 0.5, label = "90% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_cnf_1_50"][0], input["trajectories_cnf_1_50"][1], color = cmap1(0.8), alpha = 0.8, label = "50% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_95"][0], input["trajectories_pf_1_95"][1], color = cmap2(0.2), alpha = 0.2, label = "95% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_90"][0], input["trajectories_pf_1_90"][1], color = cmap2(0.5), alpha = 0.5, label = "90% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_50"][0], input["trajectories_pf_1_50"][1], color = cmap2(0.8), alpha = 0.8, label = "50% percentile")  
    ax_infc.plot(input["config"]["timepoints1_nonmissing"], input["config"]["obs1_nonmissing"], markersize = 3,  marker="o", linestyle="dotted", color = "black", label = "data for experiments")
    ax_infc.grid(alpha=0.5)
    ax_infc.tick_params(labelsize=fontsize_tick)
    ax_infc.set_ylabel("Infection Count", fontsize=fontsize_label)
    for spine in ['top', 'right', 'left']:  # remove part of the surrounding box, as it gets busy with the grid lines
        ax_infc.spines[spine].set_visible(False)

    ax_infc.text(-0.25, 1.0, 'B', transform=ax_infc.transAxes, fontsize=24,  fontweight='bold', verticalalignment='top', horizontalalignment='left')
      
    # Second Plot
    #ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_95"][0], input["trajectories_pf_2_95"][1], color = cmap2(0.2), alpha = 0.3, label = "95% percentile")
    #ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_90"][0], input["trajectories_pf_2_90"][1], color = cmap2(0.5), alpha = 0.5, label = "90% percentile")
    #ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_50"][0], input["trajectories_pf_2_50"][1], color = cmap2(0.8), alpha = 0.7, label = "50% percentile")
   
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_cnf_2_95"][0], input["trajectories_cnf_2_95"][1], color = cmap1(0.2), alpha = 0.3, label = "95% percentile")
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_cnf_2_90"][0], input["trajectories_cnf_2_90"][1], color = cmap1(0.5), alpha = 0.5, label = "90% percentile")
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_cnf_2_50"][0], input["trajectories_cnf_2_50"][1], color = cmap1(0.8), alpha = 0.7, label = "50% percentile")    
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_95"][0], input["trajectories_pf_2_95"][1], color = cmap2(0.2), alpha = 0.3, label = "95% percentile")
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_90"][0], input["trajectories_pf_2_90"][1], color = cmap2(0.5), alpha = 0.5, label = "90% percentile")
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_50"][0], input["trajectories_pf_2_50"][1], color = cmap2(0.8), alpha = 0.7, label = "50% percentile")
   
    ax_seroprev.plot(input["config"]["timepoints2_nonmissing"], input["config"]["obs2_nonmissing"], markersize=3, marker="o", linestyle="dotted", color = "black", label = "data for experiments")
    ax_seroprev.grid(alpha=0.5)
    ax_seroprev.set_axisbelow(True)
    ax_seroprev.tick_params(labelsize=fontsize_tick)
    ax_seroprev.set_ylabel("Seroprevalence", fontsize=fontsize_label)
    for spine in ['top', 'right', 'left']:
        ax_seroprev.spines[spine].set_visible(False)


    #Adding an empty row for spacing
    ax_spacer = fig.add_subplot(gs[n_params_1, 0:8])
    ax_spacer.set_frame_on(False)  
    ax_spacer.set_xticks([])  
    ax_spacer.set_yticks([])  
    ax_spacer.axis('off')

    # Add legend
    handles = [
        Line2D(xdata=[], ydata=[], color=cmap1(0.5), lw=12),
        Line2D(xdata=[], ydata=[], color=cmap2(0.5), lw=12),
        Line2D(xdata=[], ydata=[], color='black', lw=4,  linestyle='dotted'),
    ]

    labels = [
        "Posterior Draws CNF", 
        "Posterior Draws PF",
        "Inference Data"
    ]

    if input["true_params"] is not None:
        handles.append(Line2D(xdata=[], ydata=[], color='black', lw=2))
        labels.append("True Parameters")

    if input["map_cnf"] is not None:
        handles.append(Line2D(xdata=[], ydata=[], color=cmap1(0.9), lw=2))
        labels.append("MAP CNF")

    if input["map_pf"] is not None:
        handles.append(Line2D(xdata=[], ydata=[], color=cmap2(0.9), lw=2))
        labels.append("MAP PF")

    if n_params_1 == 6:
        fig.legend(handles, labels, ncol=1, fontsize=32, loc="lower right", bbox_to_anchor=(1, 0.25))
    if n_params_1 == 5:
        fig.legend(handles, labels, ncol=1, fontsize=32, loc="lower right", bbox_to_anchor=(1, 0.1))
    if n_params_1 == 2:
        fig.legend(handles, labels, ncol=3, fontsize=32, loc="lower center")
    fig.tight_layout(pad=2)
    if n_params_1 == 2:
        plt.subplots_adjust(wspace=0.5, hspace=0.3, bottom = 0.05)
    else:
        plt.subplots_adjust(wspace=0.5, hspace=0.3)

    return fig

def plot_posteriors_2d_sir(
    input,
    height=3,
    post_1_color='#ff7f00',
    post_2_color='#377eb8',
    table_1_color = '#ffefe0',
    table_2_color = '#eff4f9',
    fontsize_label = 20,
    fontsize_tick = 18
):

    # Obtain n_draws and n_params ONLY works if all posterior draws have the same shape
    n_draws_1, n_params_1 = input["posterior_draws_cnf"].shape

    # Attempt to determine parameter names
    if input["param_names"] is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params_1 + 1)]
    else:
        param_names = input["param_names"]

    cmap1 = _cmap_from_color(post_1_color)
    cmap2 = _cmap_from_color(post_2_color)
    palette = {"CNF": cmap1(0.2), "PF": cmap2(0.2)}

    figsize = 7.5 * height, (4 + 0.3)* height
    fig = plt.figure(figsize = figsize)
    # Define height ratios to control spacing
    height_ratios = [1] * 4 + [0.3]# + [1] * n_params_1  # Fourth row (spacer) is half the height of the others
    #gs = fig.add_gridspec(2 * n_params_1 + 1, 7, height_ratios=height_ratios)
    width_ratios = [1] * 7 + [0.5]
    gs = fig.add_gridspec(4+1, 8, height_ratios=height_ratios, width_ratios = width_ratios)

    axs = []
    for i in range(n_params_1):
            axs.append([fig.add_subplot(gs[i, j]) for j in range(i+1)])

    cnf_df = input["df_posterior_draws"].query('model == "CNF"').drop(columns="model")
    pf_df = input["df_posterior_draws"].query('model == "PF"').drop(columns="model")
    percentiles = [0.005, 0.995]
    bounds_cnf = np.quantile(cnf_df, q=percentiles, axis=0)
    bounds_pf = np.quantile(pf_df, q=percentiles, axis=0)
    lower_bound = np.minimum(bounds_cnf[0], bounds_pf[0])
    upper_bound = np.maximum(bounds_cnf[1], bounds_pf[1])
    bounds = np.vstack([lower_bound, upper_bound])

    for i in range(n_params_1):
        sns.violinplot(data=input["df_posterior_draws"], y=param_names[i], hue='model', palette=palette, split=True, inner=None, fill=True,
                    width=0.7,  ax=axs[i][i], legend=False, linewidth=1)
        if i > 0:
            axs[i][i].set_yticks([])
            axs[i][i].set_ylabel(None)

        patch_left = PathPatch(axs[i][i].collections[0].get_paths()[0], transform=axs[i][i].transData)
        patch_right = PathPatch(axs[i][i].collections[1].get_paths()[0], transform=axs[i][i].transData)

        if input["true_params"] is not None:
            line_left_true = axs[i][i].axhline(y=input["true_params"] [i], color="black", label="true parameter")
            line_right_true = axs[i][i].axhline(y=input["true_params"] [i], color="black", label="true parameter")
            line_left_true.set_clip_path(patch_left)
            line_right_true.set_clip_path(patch_right)

        if input["map_cnf"] is not None:
            line_left_map1 = axs[i][i].axhline(y=input["map_cnf"][i], color=cmap1(0.9), label="MAP CNF")
            line_left_map1.set_clip_path(patch_left)

        if input["map_pf"] is not None:
            line_right_map2 = axs[i][i].axhline(y=input["map_pf"][i], color=cmap2(0.9), label="MAP PF")
            line_right_map2.set_clip_path(patch_right)

        for spine in ['top', 'right', 'bottom']: 
            axs[i][i].spines[spine].set_visible(False)

        lower, upper = bounds[:, i]
        axs[i][i].set_ylim(lower, upper)

    posterior_draws_cnf_df = pd.DataFrame(input["posterior_draws_cnf"], columns=param_names)
    posterior_draws_pf_df = pd.DataFrame(input["posterior_draws_pf"], columns=param_names)

    for i in range(n_params_1):
        for j in range(i):
            sns.kdeplot(data=posterior_draws_cnf_df, x=param_names[j], y=param_names[i], color=post_1_color, 
                        levels=4, thresh=.2, fill=True, alpha = 0.7, ax=axs[i][j], legend=False)
    
            sns.kdeplot(data=posterior_draws_pf_df, x=param_names[j], y=param_names[i], color=post_2_color, 
                        levels=4, thresh=.2, fill=True, alpha = 0.7, ax=axs[i][j], legend=False)          


            axs[i][j].grid(alpha=0.5)
            for spine in ['top', 'right']:
                axs[i][j].spines[spine].set_visible(False)
            if j > 0 and i < n_params_1 - 1:
                axs[i][j].set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
            if j == 0 and i < n_params_1 - 1:
                axs[i][j].set(xlabel=None, xticklabels=[])
            if j > 0 and i == n_params_1 - 1:
                axs[i][j].set(ylabel=None, yticklabels=[])

        """Add labels to the left and bottom Axes."""
    for i in range(n_params_1):
        axs[-1][i].set_xlabel(param_names[i], fontsize = fontsize_label)
        axs[-1][i].tick_params(labelsize = fontsize_tick)
    for i in range(n_params_1):
        axs[i][0].set_ylabel(param_names[i], fontsize = fontsize_label)
        axs[i][0].tick_params(labelsize = fontsize_tick)

    axs[0][0].text(-0.7, 1.0, 'A', transform=axs[0][0].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
    #axs[0][0].set_ylim([0.09,0.12])
    #axs[1][1].set_ylim([16,23])
    #axs[1][0].set_xlim(axs[0][0].get_ylim())
    #axs[1][0].set_ylim(axs[1][1].get_ylim())
    # Reduce number of xticks instead of rounding values
    #axs[1][0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))


    ax_table = fig.add_subplot(gs[2:4,1:4])
    for spine in ['top', 'right', 'bottom', 'left']: 
        ax_table.spines[spine].set_visible(False)
    ax_table.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax_table.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    ax_table.text(-0.0, 0.6, 'C', transform=ax_table.transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
    
    row_colours = ["lightgrey"] * len(input["df_map"].columns)
    cell_colours = [['w'] * len(input["df_map"].columns), [table_1_color] * len(input["df_map"].columns), [table_2_color] * len(input["df_map"].columns)]
    table = ax_table.table(cellText=input["df_map"].values, colLabels=input["df_map"].columns, rowLabels=input["df_map"].index, 
                          bbox=[0.0,0.2,1.6,0.4],  cellColours=cell_colours, rowColours = row_colours, colColours=row_colours, cellLoc = "center", rowLoc="center")
    table.auto_set_font_size(False)

    # Set font sizes
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_fontsize(fontsize_label)
        elif col == -1:  # Row labels
            cell.set_fontsize(fontsize_label)
        else:  # Data rows
            cell.set_fontsize(fontsize_tick)
    #table.set_fontsize(fontsize_label)
    table.auto_set_column_width(col=list(range(len(input["df_map"].columns))))
    

    ax_trajectories = fig.add_subplot(gs[0:2,3:7])
    for spine in ['top', 'right', 'bottom', 'left']: 
        ax_trajectories.spines[spine].set_visible(False)
    ax_trajectories.set(ylabel=None, xticklabels=[], yticklabels=[])
    ax_trajectories.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax_trajectories.set_xlabel('Time t in days', fontsize=fontsize_label, labelpad=30)
   
    ax_trajectories.margins(x=0.1)

    ax_infc = fig.add_subplot(gs[0:2,3:5])
    ax_seroprev = fig.add_subplot(gs[0:2,5:7])

    # First Plot
    #ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_95"][0], input["trajectories_pf_1_95"][1], color = cmap2(0.2), alpha = 0.2, label = "95% percentile")
    #ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_90"][0], input["trajectories_pf_1_90"][1], color = cmap2(0.5), alpha = 0.5, label = "90% percentile")
    #ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_50"][0], input["trajectories_pf_1_50"][1], color = cmap2(0.8), alpha = 0.8, label = "50% percentile")  
    
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_cnf_1_95"][0], input["trajectories_cnf_1_95"][1], color = cmap1(0.2), alpha = 0.2, label = "95% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_cnf_1_90"][0], input["trajectories_cnf_1_90"][1], color = cmap1(0.5), alpha = 0.5, label = "90% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_cnf_1_50"][0], input["trajectories_cnf_1_50"][1], color = cmap1(0.8), alpha = 0.8, label = "50% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_95"][0], input["trajectories_pf_1_95"][1], color = cmap2(0.2), alpha = 0.2, label = "95% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_90"][0], input["trajectories_pf_1_90"][1], color = cmap2(0.5), alpha = 0.5, label = "90% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_50"][0], input["trajectories_pf_1_50"][1], color = cmap2(0.8), alpha = 0.8, label = "50% percentile")  
    ax_infc.plot(input["config"]["timepoints1_nonmissing"], input["config"]["obs1_nonmissing"], markersize = 3,  marker="o", linestyle="dotted", color = "black", label = "data for experiments")
    ax_infc.grid(alpha=0.5)
    ax_infc.tick_params(labelsize=fontsize_tick)
    ax_infc.set_ylabel("Infection Count", fontsize=fontsize_label)
    for spine in ['top', 'right', 'left']:  # remove part of the surrounding box, as it gets busy with the grid lines
        ax_infc.spines[spine].set_visible(False)

    ax_infc.text(-0.25, 1.0, 'B', transform=ax_infc.transAxes, fontsize=24,  fontweight='bold', verticalalignment='top', horizontalalignment='left')
      
    # Second Plot
    #ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_95"][0], input["trajectories_pf_2_95"][1], color = cmap2(0.2), alpha = 0.3, label = "95% percentile")
    #ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_90"][0], input["trajectories_pf_2_90"][1], color = cmap2(0.5), alpha = 0.5, label = "90% percentile")
    #ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_50"][0], input["trajectories_pf_2_50"][1], color = cmap2(0.8), alpha = 0.7, label = "50% percentile")
   
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_cnf_2_95"][0], input["trajectories_cnf_2_95"][1], color = cmap1(0.2), alpha = 0.3, label = "95% percentile")
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_cnf_2_90"][0], input["trajectories_cnf_2_90"][1], color = cmap1(0.5), alpha = 0.5, label = "90% percentile")
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_cnf_2_50"][0], input["trajectories_cnf_2_50"][1], color = cmap1(0.8), alpha = 0.7, label = "50% percentile")    
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_95"][0], input["trajectories_pf_2_95"][1], color = cmap2(0.2), alpha = 0.3, label = "95% percentile")
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_90"][0], input["trajectories_pf_2_90"][1], color = cmap2(0.5), alpha = 0.5, label = "90% percentile")
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_50"][0], input["trajectories_pf_2_50"][1], color = cmap2(0.8), alpha = 0.7, label = "50% percentile")
   
    ax_seroprev.plot(input["config"]["timepoints2_nonmissing"], input["config"]["obs2_nonmissing"], markersize=3, marker="o", linestyle="dotted", color = "black", label = "data for experiments")
    ax_seroprev.grid(alpha=0.5)
    ax_seroprev.set_axisbelow(True)
    ax_seroprev.tick_params(labelsize=fontsize_tick)
    ax_seroprev.set_ylabel("Seroprevalence", fontsize=fontsize_label)
    for spine in ['top', 'right', 'left']:
        ax_seroprev.spines[spine].set_visible(False)


    #Adding an empty row for spacing
    ax_spacer = fig.add_subplot(gs[n_params_1, 0:8])
    ax_spacer.set_frame_on(False)  
    ax_spacer.set_xticks([])  
    ax_spacer.set_yticks([])  
    ax_spacer.axis('off')

    # Add legend
    handles = [
        Line2D(xdata=[], ydata=[], color=cmap1(0.5), lw=12),
        Line2D(xdata=[], ydata=[], color=cmap2(0.5), lw=12),
        Line2D(xdata=[], ydata=[], color='black', lw=4,  linestyle='dotted'),
    ]

    labels = [
        "Posterior Draws CNF", 
        "Posterior Draws PF",
        "Inference Data"
    ]

    if input["true_params"] is not None:
        handles.append(Line2D(xdata=[], ydata=[], color='black', lw=2))
        labels.append("True Parameters")

    if input["map_cnf"] is not None:
        handles.append(Line2D(xdata=[], ydata=[], color=cmap1(0.9), lw=2))
        labels.append("MAP CNF")

    if input["map_pf"] is not None:
        handles.append(Line2D(xdata=[], ydata=[], color=cmap2(0.9), lw=2))
        labels.append("MAP PF")

    fig.legend(handles, labels, ncol=3, fontsize=32, loc="lower center")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.3)

    return fig

def plot_posterior_mountain(posterior_samples):

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)  # Grid layout

    params = [(0, 1), (1, 4)]  # (param_1, param_2), (param_1, param_6)
    param_names = ["$\\gamma^{-1}$", "$\\kappa^{-1}$", "$\\beta$", "s", "$t_{var}$", "$I_0$"]
    axes = [fig.add_subplot(gs[0, 0], projection='3d'), fig.add_subplot(gs[0, 1], projection='3d')]

    cmap = plt.get_cmap("plasma")
    colors = cmap(np.linspace(0.0, 0.7, 256))
    half_cmap = LinearSegmentedColormap.from_list("HalfPlasma", colors)

    for ax, (p1, p2) in zip(axes, params):
        data = posterior_samples[:, [p1, p2]]
        kde = gaussian_kde(data.T)

        x_min, x_max = data[:, 0].min(), data[:, 0].max()
        y_min, y_max = data[:, 1].min(), data[:, 1].max()
        x_grid, y_grid = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_grid, y_grid)

        grid_coords = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(grid_coords).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, cmap=half_cmap, edgecolor='none', shade=True, alpha=1)
        #ax.contour(X, Y, Z, levels=20, cmap=half_cmap, linewidths=2)
        ax.set_xlabel(param_names[p1])
        ax.set_ylabel(param_names[p2]) 
        ax.set_zlabel("")  # Remove z-axis label
        ax.set_zticks([])  # Remove z-axis ticks

    # Adjust color bar size & position
    cbar_ax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(surf, cax=cbar_ax, shrink=0.5)  # Further reduced size

    # Remove colorbar ticks and replace with custom labels
    cbar.set_ticks([])
    fig.text(0.89, 0.08, "Low", ha='center', va='center', fontsize=12, color='black', fontweight='bold')
    fig.text(0.89, 0.92, "High", ha='center', va='center', fontsize=12, color='black', fontweight='bold')

    # Add common density label next to the colorbar
    fig.text(0.92, 0.5, "Density", va='center', ha='center', fontsize=14, rotation=90)

    plt.tight_layout()
    return fig

def plot_trajectories(
    input,
    height=3,
    post_1_color='#ff7f00',
    post_2_color='#377eb8',
    fontsize_label = 20,
    fontsize_tick = 18
):

    # Obtain n_draws and n_params ONLY works if all posterior draws have the same shape
    n_draws_1, n_params_1 = input["posterior_draws_cnf"].shape

    # Attempt to determine parameter names
    if input["param_names"] is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params_1 + 1)]
    else:
        param_names = input["param_names"]

    cmap1 = _cmap_from_color(post_1_color)
    cmap2 = _cmap_from_color(post_2_color)
    palette = {"CNF": cmap1(0.2), "PF": cmap2(0.2)}


    figsize = 4.3* height, (2 + 0.3)* height
    fig = plt.figure(figsize = figsize)
    # Define height ratios to control spacing
    height_ratios = [1] * 2 + [0.3]# + [1] * n_params_1  # Fourth row (spacer) is half the height of the others
    #gs = fig.add_gridspec(2 * n_params_1 + 1, 7, height_ratios=height_ratios)
    width_ratios = [1] * 2 + [0.3] + [1] * 2
    gs = fig.add_gridspec(3, 5, height_ratios=height_ratios, width_ratios = width_ratios)
 
    ax_trajectories = fig.add_subplot(gs[0:2,0:5])
    for spine in ['top', 'right', 'bottom', 'left']: 
        ax_trajectories.spines[spine].set_visible(False)
    ax_trajectories.set(ylabel=None, xticklabels=[], yticklabels=[])
    ax_trajectories.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax_trajectories.set_xlabel('Time t in days', fontsize=fontsize_label, labelpad=30)
   
    ax_trajectories.margins(x=0.1)

    ax_infc = fig.add_subplot(gs[0:2,0:2])
    ax_seroprev = fig.add_subplot(gs[0:2,3:5])

    # First Plot
    #ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_95"][0], input["trajectories_pf_1_95"][1], color = cmap2(0.2), alpha = 0.2, label = "95% percentile")
    #ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_90"][0], input["trajectories_pf_1_90"][1], color = cmap2(0.5), alpha = 0.5, label = "90% percentile")
    #ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_50"][0], input["trajectories_pf_1_50"][1], color = cmap2(0.8), alpha = 0.8, label = "50% percentile")  
    
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_cnf_1_95"][0], input["trajectories_cnf_1_95"][1], color = cmap1(0.2), alpha = 0.2, label = "95% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_cnf_1_90"][0], input["trajectories_cnf_1_90"][1], color = cmap1(0.5), alpha = 0.5, label = "90% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_cnf_1_50"][0], input["trajectories_cnf_1_50"][1], color = cmap1(0.8), alpha = 0.8, label = "50% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_95"][0], input["trajectories_pf_1_95"][1], color = cmap2(0.2), alpha = 0.2, label = "95% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_90"][0], input["trajectories_pf_1_90"][1], color = cmap2(0.5), alpha = 0.5, label = "90% percentile")
    ax_infc.fill_between(input["config"]["timepoints1_nonmissing"], input["trajectories_pf_1_50"][0], input["trajectories_pf_1_50"][1], color = cmap2(0.8), alpha = 0.8, label = "50% percentile")  
    #ax_infc.plot(input["config"]["timepoints1_nonmissing"], input["config"]["obs1_nonmissing"], markersize = 3,  marker="o", linestyle="dotted", color = "black", label = "data for experiments")
    ax_infc.grid(alpha=0.5)
    ax_infc.tick_params(labelsize=fontsize_tick)
    ax_infc.set_ylabel("Infection Count", fontsize=fontsize_label)
    for spine in ['top', 'right', 'left']:  # remove part of the surrounding box, as it gets busy with the grid lines
        ax_infc.spines[spine].set_visible(False)

      
    # Second Plot
    #ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_95"][0], input["trajectories_pf_2_95"][1], color = cmap2(0.2), alpha = 0.3, label = "95% percentile")
    #ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_90"][0], input["trajectories_pf_2_90"][1], color = cmap2(0.5), alpha = 0.5, label = "90% percentile")
    #ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_50"][0], input["trajectories_pf_2_50"][1], color = cmap2(0.8), alpha = 0.7, label = "50% percentile")
   
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_cnf_2_95"][0], input["trajectories_cnf_2_95"][1], color = cmap1(0.2), alpha = 0.3, label = "95% percentile")
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_cnf_2_90"][0], input["trajectories_cnf_2_90"][1], color = cmap1(0.5), alpha = 0.5, label = "90% percentile")
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_cnf_2_50"][0], input["trajectories_cnf_2_50"][1], color = cmap1(0.8), alpha = 0.7, label = "50% percentile")    
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_95"][0], input["trajectories_pf_2_95"][1], color = cmap2(0.2), alpha = 0.3, label = "95% percentile")
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_90"][0], input["trajectories_pf_2_90"][1], color = cmap2(0.5), alpha = 0.5, label = "90% percentile")
    ax_seroprev.fill_between(input["config"]["timepoints2_nonmissing"], input["trajectories_pf_2_50"][0], input["trajectories_pf_2_50"][1], color = cmap2(0.8), alpha = 0.7, label = "50% percentile")
   
    #ax_seroprev.plot(input["config"]["timepoints2_nonmissing"], input["config"]["obs2_nonmissing"], markersize=3, marker="o", linestyle="dotted", color = "black", label = "data for experiments")
    ax_seroprev.grid(alpha=0.5)
    ax_seroprev.set_axisbelow(True)
    ax_seroprev.tick_params(labelsize=fontsize_tick)
    ax_seroprev.set_ylabel("Seroprevalence", fontsize=fontsize_label)
    for spine in ['top', 'right', 'left']:
        ax_seroprev.spines[spine].set_visible(False)


    # Add legend
    handles = [
        Line2D(xdata=[], ydata=[], color=cmap1(0.5), lw=12),
        Line2D(xdata=[], ydata=[], color=cmap2(0.5), lw=12),
        Line2D(xdata=[], ydata=[], color='black', lw=4,  linestyle='dotted'),
    ]

    labels = [
        "Posterior Draws CNF", 
        "Posterior Draws PF",
        "Inference Data"
    ]
    '''
    if n_params_1 == 6:
        fig.legend(handles, labels, ncol=1, fontsize=32, loc="lower right", bbox_to_anchor=(1, 0.25))
    if n_params_1 == 5:
        fig.legend(handles, labels, ncol=1, fontsize=32, loc="lower right", bbox_to_anchor=(1, 0.1))
    if n_params_1 == 2:
        fig.legend(handles, labels, ncol=3, fontsize=32, loc="lower center")
    '''
    fig.tight_layout(pad=2)
    if n_params_1 == 2:
        plt.subplots_adjust(wspace=0.5, hspace=0.3, bottom = 0.05)
    else:
        plt.subplots_adjust(wspace=0.5, hspace=0.3)

    return fig


def plot_input(posterior_draws_cnf, posterior_draws_pf, config, mode, param_names, name, true_params, paramstring = "True Param."):
    values_cnf = np.transpose(posterior_draws_cnf)
    kernel_cnf = gaussian_kde(values_cnf, bw_method='silverman')
    height_cnf = kernel_cnf.pdf(values_cnf)
    map_cnf = values_cnf[:,np.argmax(height_cnf)]
    df = pd.DataFrame(map_cnf[np.newaxis,:])

    values_pf = np.transpose(posterior_draws_pf)
    kernel_pf = gaussian_kde(values_pf, bw_method='silverman')
    height_pf = kernel_pf.pdf(values_pf)
    map_pf = values_pf[:,np.argmax(height_pf)]
    df = pd.DataFrame(map_pf[np.newaxis,:])

    trajectories_cnf_1, trajectories_cnf_2 = model(posterior_draws_cnf, config, mode=mode)
    trajectories_pf_1, trajectories_pf_2 = model(posterior_draws_pf, config, mode=mode) 
    n_draws_cnf_1, batch_size_cnf_1, n_timepoints_cnf_1 = trajectories_cnf_1.shape
    n_draws_cnf_2, batch_size_cnf_2, n_timepoints_cnf_2 = trajectories_cnf_2.shape
    n_draws_pf_1, batch_size_pf_1, n_timepoints_pf_1 = trajectories_pf_1.shape
    n_draws_pf_2, batch_size_pf_2, n_timepoints_pf_2 = trajectories_pf_2.shape

    map_cnf_mse = mean_squared_error(config, trajectories_cnf_1, trajectories_cnf_2)
    map_pf_mse = mean_squared_error(config, trajectories_pf_1, trajectories_pf_2)

    trajectories_cnf_1 = trajectories_cnf_1.reshape((n_draws_cnf_1 * batch_size_cnf_1, n_timepoints_cnf_1))
    trajectories_cnf_2 = trajectories_cnf_2.reshape((n_draws_cnf_2 * batch_size_cnf_2, n_timepoints_cnf_2))
    trajectories_pf_1 = trajectories_pf_1.reshape((n_draws_pf_1 * batch_size_pf_1, n_timepoints_pf_1))
    trajectories_pf_2 = trajectories_pf_2.reshape((n_draws_pf_2 * batch_size_pf_2, n_timepoints_pf_2))

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

    trajectories_true_1, trajectories_true_2 = model(true_params[np.newaxis,:], config, mode=mode, batchsize=n)
    #trajectories_map_cnf_1, trajectories_map_cnf_2 = model(map_cnf[np.newaxis,:], config, mode=mode, batchsize=n) 
    #trajectories_map_pf_1, trajectories_map_pf_2 = model(map_pf[np.newaxis,:], config, mode=mode, batchsize=n)    

    true_params_mse = mean_squared_error(config, trajectories_true_1, trajectories_true_2)


    df_cnf = pd.DataFrame(posterior_draws_cnf, columns=param_names,  dtype='float')
    df_cnf["model"] = "CNF"
    df_pf = pd.DataFrame(posterior_draws_pf, columns=param_names, dtype='float')
    df_pf["model"] = "PF"
    df_true = pd.DataFrame(true_params[np.newaxis,:], columns=param_names, dtype='float')
    df_true[" "] = " "
    df_true["$RMSE_{infc}$"] = true_params_mse[0, 0]
    df_true["$RMSE_{sprev}$"] = true_params_mse[0, 1]
    df_map_cnf = pd.DataFrame(map_cnf[np.newaxis,:], columns=param_names, dtype='float')
    df_map_cnf[" "] = " "
    df_map_cnf["$RMSE_{infc}$"] = map_cnf_mse[0, 0]
    df_map_cnf["$RMSE_{sprev}$"] = map_cnf_mse[0, 1]
    df_map_pf = pd.DataFrame(map_pf[np.newaxis,:], columns=param_names, dtype='float')
    df_map_pf[" "] = " "
    df_map_pf["$RMSE_{infc}$"] = map_pf_mse[0, 0]
    df_map_pf["$RMSE_{sprev}$"] = map_pf_mse[0, 1]
    df_posterior_draws = pd.concat((df_cnf,df_pf), axis=0)
    df_map = pd.concat((df_true, df_map_cnf, df_map_pf), axis=0)
    df_map.index = [paramstring, '$CNF$', '$PF$']
    
    for name in df_map.columns:
        if name in ["$\\beta$", "$RMSE_{infc}$", "$RMSE_{sprev}$"]:
            df_map[name] = [f"{x:.4f}" for x in df_map[name].round(4)]
        elif name != " ":
            df_map[name] = [f"{x:.2f}" for x in df_map[name].round(2)]

    dict = {"config": config,
            "mode": mode,
            "param_names" : param_names,
            "name" : name,
            "true_params" : true_params,
            "posterior_draws_cnf" : posterior_draws_cnf,
            "map_cnf" : map_cnf,
            "trajectories_cnf_1_50" : trajectories_cnf_1_50,
            "trajectories_cnf_1_90" : trajectories_cnf_1_90,
            "trajectories_cnf_1_95" : trajectories_cnf_1_95,
            "trajectories_cnf_2_50" : trajectories_cnf_2_50,
            "trajectories_cnf_2_90" : trajectories_cnf_2_90,
            "trajectories_cnf_2_95" : trajectories_cnf_2_95,
            "posterior_draws_pf" : posterior_draws_pf,
            "map_pf" : map_pf,
            "trajectories_pf_1_50" : trajectories_pf_1_50,
            "trajectories_pf_1_90" : trajectories_pf_1_90,
            "trajectories_pf_1_95" : trajectories_pf_1_95,
            "trajectories_pf_2_50" : trajectories_pf_2_50,
            "trajectories_pf_2_90" : trajectories_pf_2_90,
            "trajectories_pf_2_95" : trajectories_pf_2_95,
            "df_posterior_draws" : df_posterior_draws,
            "df_map" : df_map
            }
    
    return dict

def plot_input1(posterior_draws_cnf, posterior_draws_pf, config, mode, param_names, name, true_params):
    trajectories_cnf_1, trajectories_cnf_2 = model(posterior_draws_cnf, config, mode=mode)
    trajectories_pf_1, trajectories_pf_2 = model(posterior_draws_pf, config, mode=mode) 
    n_draws_cnf_1, batch_size_cnf_1, n_timepoints_cnf_1 = trajectories_cnf_1.shape
    n_draws_cnf_2, batch_size_cnf_2, n_timepoints_cnf_2 = trajectories_cnf_2.shape
    n_draws_pf_1, batch_size_pf_1, n_timepoints_pf_1 = trajectories_pf_1.shape
    n_draws_pf_2, batch_size_pf_2, n_timepoints_pf_2 = trajectories_pf_2.shape
    trajectories_cnf_1 = trajectories_cnf_1.reshape((n_draws_cnf_1 * batch_size_cnf_1, n_timepoints_cnf_1))
    trajectories_cnf_2 = trajectories_cnf_2.reshape((n_draws_cnf_2 * batch_size_cnf_2, n_timepoints_cnf_2))
    trajectories_pf_1 = trajectories_pf_1.reshape((n_draws_pf_1 * batch_size_pf_1, n_timepoints_pf_1))
    trajectories_pf_2 = trajectories_pf_2.reshape((n_draws_pf_2 * batch_size_pf_2, n_timepoints_pf_2))

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

    dict = {"config": config,
            "mode": mode,
            "param_names" : param_names,
            "name" : name,
            "true_params" : true_params,
            "posterior_draws_cnf" : posterior_draws_cnf,
            "trajectories_cnf_1_50" : trajectories_cnf_1_50,
            "trajectories_cnf_1_90" : trajectories_cnf_1_90,
            "trajectories_cnf_1_95" : trajectories_cnf_1_95,
            "trajectories_cnf_2_50" : trajectories_cnf_2_50,
            "trajectories_cnf_2_90" : trajectories_cnf_2_90,
            "trajectories_cnf_2_95" : trajectories_cnf_2_95,
            "posterior_draws_pf" : posterior_draws_pf,
            "trajectories_pf_1_50" : trajectories_pf_1_50,
            "trajectories_pf_1_90" : trajectories_pf_1_90,
            "trajectories_pf_1_95" : trajectories_pf_1_95,
            "trajectories_pf_2_50" : trajectories_pf_2_50,
            "trajectories_pf_2_90" : trajectories_pf_2_90,
            "trajectories_pf_2_95" : trajectories_pf_2_95
            }
    
    return dict

def input_plot_single(posterior_draws, true_params, config, mode, param_names, name):
    n=1000
    values_post = np.transpose(posterior_draws)
    kernel_post = gaussian_kde(values_post, bw_method='silverman')
    height_post = kernel_post.pdf(values_post)
    map = values_post[:,np.argmax(height_post)]
    #df = pd.DataFrame(map_abi[np.newaxis,:])

    trajectories_post_1, trajectories_post_2 = model(posterior_draws, config, mode=mode)
    n_draws_post_1, batch_size_post_1, n_timepoints_post_1 = trajectories_post_1.shape
    n_draws_post_2, batch_size_post_2, n_timepoints_post_2 = trajectories_post_2.shape
    trajectories_post_1 = trajectories_post_1.reshape((n_draws_post_1 * batch_size_post_1, n_timepoints_post_1))
    trajectories_post_2 = trajectories_post_2.reshape((n_draws_post_2 * batch_size_post_2, n_timepoints_post_2))

    trajectories_true_1, trajectories_true_2 = model(true_params[np.newaxis,:], config, mode=mode, batchsize=n)
    trajectories_map_1, trajectories_map_2 = model(map[np.newaxis,:], config, mode=mode, batchsize=n) 

    true_params_mse = mean_squared_error(config, trajectories_true_1, trajectories_true_2)
    map_mse = mean_squared_error(config, trajectories_map_1, trajectories_map_2)

    # compute percentiles
    n_draws_true_1, batch_size_true_1, n_timepoints_true_1 = trajectories_true_1.shape
    n_draws_true_2, batch_size_true_2, n_timepoints_true_2 = trajectories_true_2.shape
    trajectories_true_1 = trajectories_true_1.reshape((n_draws_true_1 * batch_size_true_1, n_timepoints_true_1))
    trajectories_true_2 = trajectories_true_2.reshape((n_draws_true_2 * batch_size_true_2, n_timepoints_true_2))

    n_draws_map_1, batch_size_map_1, n_timepoints_map_1 = trajectories_map_1.shape
    n_draws_map_2, batch_size_map_2, n_timepoints_map_2 = trajectories_map_2.shape
    trajectories_map_1 = trajectories_map_1.reshape((n_draws_map_1 * batch_size_map_1, n_timepoints_map_1))
    trajectories_map_2 = trajectories_map_2.reshape((n_draws_map_2 * batch_size_map_2, n_timepoints_map_2))

    trajectories_true_1_p = {}
    trajectories_true_2_p = {}
    trajectories_post_1_p = {}
    trajectories_post_2_p = {}
    trajectories_map_1_p = {}
    trajectories_map_2_p = {}

    for p, (q_low, q_high) in COVERAGE_QUANTILES.items():
        trajectories_true_1_p[p] = np.quantile(trajectories_true_1, q=[q_low, q_high], axis=0)
        trajectories_true_2_p[p] = np.quantile(trajectories_true_2, q=[q_low, q_high], axis=0)
        trajectories_post_1_p[p] = np.quantile(trajectories_post_1, q=[q_low, q_high], axis=0)
        trajectories_post_2_p[p] = np.quantile(trajectories_post_2, q=[q_low, q_high], axis=0)
        trajectories_map_1_p[p] = np.quantile(trajectories_map_1, q=[q_low, q_high], axis=0)
        trajectories_map_2_p[p] = np.quantile(trajectories_map_2, q=[q_low, q_high], axis=0)



    df_post = pd.DataFrame(posterior_draws, columns=param_names,  dtype='float')
    df_post["model"] = "CNF"
    df_true = pd.DataFrame(true_params[np.newaxis,:], columns=param_names, dtype='float')
    df_true[" "] = " "
    df_true["$RMSE_{infc}$"] = true_params_mse[0, 0]
    df_true["$RMSE_{sprev}$"] = true_params_mse[0, 1]    
    df_map = pd.DataFrame(map[np.newaxis,:], columns=param_names, dtype='float')
    df_map[" "] = " "
    df_map["$RMSE_{infc}$"] = map_mse[0, 0]
    df_map["$RMSE_{sprev}$"] = map_mse[0, 1]
    df_map = pd.concat((df_true, df_map), axis=0)
    df_map.index = ['True Param.', '$MAP$']
    
    # Simulated true prediction sets:
    bounds1_post = compute_trajectory_bounds(trajectories_post_1)
    bounds2_post = compute_trajectory_bounds(trajectories_post_2)
    bounds1_true = compute_trajectory_bounds(trajectories_true_1)
    bounds2_true = compute_trajectory_bounds(trajectories_true_2)

    # Compute DataFrame of observed coverage:
    df_cov_true = coverage_dataframe(config["obs1_nonmissing"], config["obs2_nonmissing"], bounds1_true, bounds2_true)
    df_cov_post = coverage_dataframe(config["obs1_nonmissing"], config["obs2_nonmissing"], bounds1_post, bounds2_post)
    df_cov = pd.concat((df_cov_true, df_cov_post), axis = 0)
    df_cov.index = ['True Param.', 'Post']

    for name in df_map.columns:
        if name in ["$\\beta$", "$RMSE_{infc}$", "$RMSE_{sprev}$"]:
            df_map[name] = [f"{x:.4f}" for x in df_map[name].round(4)]
        elif name != " ":
            df_map[name] = [f"{x:.2f}" for x in df_map[name].round(2)]

    for name in df_cov.columns:
        df_cov[name] = [f"{x:.2f}" for x in df_cov[name].round(2)]


    dict = {"config": config,
            "mode": mode,
            "param_names" : param_names,
            "name" : name,
            "true_params" : true_params,
            "posterior_draws" : posterior_draws,
            "map" : map,
            "trajectories_true_1_p" : trajectories_true_1_p,
            "trajectories_true_2_p" : trajectories_true_2_p,
            "trajectories_post_1_p" : trajectories_post_1_p,
            "trajectories_post_2_p" : trajectories_post_2_p,
            "trajectories_map_1_p" : trajectories_map_1_p,
            "trajectories_map_2_p" : trajectories_map_2_p,
            "df_post" : df_post,
            "df_map" : df_map,
            "df_cov" : df_cov
            }
    
    return dict

def plot_posteriors_single(
    input,
    height=3,
    post_1_color='#ff7f00',
    post_2_color='#377eb8',
    table_1_color = '#ffefe0',
    table_2_color = '#eff4f9',
    fontsize_label = 20,
    fontsize_tick = 18
):

    # Obtain n_draws and n_params ONLY works if all posterior draws have the same shape
    n_draws_1, n_params_1 = input["posterior_draws"].shape

    # Attempt to determine parameter names
    if input["param_names"] is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params_1 + 1)]
    else:
        param_names = input["param_names"]

    cmap1 = _cmap_from_color(post_1_color)
    cmap2 = _cmap_from_color(post_2_color)
    palette = {"CNF": cmap1(0.2), "PF": cmap2(0.2)}


    figsize = 7.5 * height, (n_params_1 + 0.3)* height
    fig = plt.figure(figsize = figsize)
    # Define height ratios to control spacing
    height_ratios = [1] * n_params_1 + [0.3]# + [1] * n_params_1  # Fourth row (spacer) is half the height of the others
    #gs = fig.add_gridspec(2 * n_params_1 + 1, 7, height_ratios=height_ratios)
    width_ratios = [1] * 7 + [0.5]
    gs = fig.add_gridspec(n_params_1+1, 8, height_ratios=height_ratios, width_ratios = width_ratios)

    axs = []
    for i in range(n_params_1):
            axs.append([fig.add_subplot(gs[i, j]) for j in range(i+1)])

    for i in range(n_params_1):
        sns.violinplot(data=input["df_post"], y=param_names[i], hue='model', palette=palette, inner=None, fill=True,
                    width=0.7,  ax=axs[i][i], legend=False, linewidth=1)
        if i > 0:
            axs[i][i].set_yticks([])
            axs[i][i].set_ylabel(None)

        patch_left = PathPatch(axs[i][i].collections[0].get_paths()[0], transform=axs[i][i].transData)

        if input["true_params"] is not None:
            line_left_true = axs[i][i].axhline(y=input["true_params"] [i], color="black")
            #line_right_true = axs[i][i].axhline(y=input["true_params"] [i], color="black")
            line_left_true.set_clip_path(patch_left)
            #line_right_true.set_clip_path(patch_right)

        if input["map"] is not None:
            line_left_map1 = axs[i][i].axhline(y=input["map"][i], color=cmap1(0.9))
            line_left_map1.set_clip_path(patch_left)


        for spine in ['top', 'right', 'bottom']: 
            axs[i][i].spines[spine].set_visible(False)

    posterior_draws_df = pd.DataFrame(input["posterior_draws"], columns=param_names)

    for i in range(n_params_1):
        for j in range(i):
            sns.kdeplot(data=posterior_draws_df, x=param_names[j], y=param_names[i], color=post_1_color, 
                        levels=4, thresh=.2, fill=True, alpha = 0.7, ax=axs[i][j], legend=False)

            axs[i][j].grid(alpha=0.5)
            for spine in ['top', 'right']:
                axs[i][j].spines[spine].set_visible(False)
            if j > 0 and i < n_params_1 - 1:
                axs[i][j].set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
            if j == 0 and i < n_params_1 - 1:
                axs[i][j].set(xlabel=None, xticklabels=[])
            if j > 0 and i == n_params_1 - 1:
                axs[i][j].set(ylabel=None, yticklabels=[])

        """Add labels to the left and bottom Axes."""
    for i in range(n_params_1):
        axs[-1][i].set_xlabel(param_names[i], fontsize = fontsize_label)
        axs[-1][i].tick_params(labelsize = fontsize_tick)
    for i in range(n_params_1):
        axs[i][0].set_ylabel(param_names[i], fontsize = fontsize_label)
        axs[i][0].tick_params(labelsize = fontsize_tick)

    axs[0][0].text(-0.5, 1.0, 'A', transform=axs[0][0].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')


    ax_table = fig.add_subplot(gs[2:4,4:8])
    for spine in ['top', 'right', 'bottom', 'left']: 
        ax_table.spines[spine].set_visible(False)
    ax_table.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax_table.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    ax_table.text(-0.05, 0.875, 'C', transform=ax_table.transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
    
    row_colours = ["whitesmoke"] * len(input["df_map"].columns)
    cell_colours = [['w'] * len(input["df_map"].columns), [table_1_color] * len(input["df_map"].columns)]
    table = ax_table.table(cellText=input["df_map"].values, colLabels=input["df_map"].columns, rowLabels=input["df_map"].index, 
                          #bbox=([box.x0, box.y0, box.width * 0.8, box.height * 0.9]),
                          bbox=[0.0,0.5,1,0.4],  
                          #loc="upper center",
                          cellColours=cell_colours, rowColours = row_colours, colColours=row_colours, cellLoc = "center")
    table.auto_set_font_size(False)

    # Set font sizes
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_fontsize(fontsize_label)
        elif col == -1:  # Row labels
            cell.set_fontsize(fontsize_label)
        else:  # Data rows
            cell.set_fontsize(fontsize_tick)
    table.auto_set_column_width(col=list(range(len(input["df_map"].columns))))
    #table.scale(1.8,1.8)
    
    ax_table2 = fig.add_subplot(gs[3,5:8])
    for spine in ['top', 'right', 'bottom', 'left']: 
        ax_table2.spines[spine].set_visible(False)
    ax_table2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax_table2.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    table2 = ax_table2.table(cellText=input["df_cov"].values, colLabels=input["df_cov"].columns, rowLabels=input["df_cov"].index, 
                          #bbox=([box.x0, box.y0, box.width * 0.8, box.height * 0.9]),
                          bbox=[0.0,0.5,1,0.4],  
                          #loc="upper center",
                          cellLoc = "center")
    
    for (row, col), cell in table2.get_celld().items():
        if row == 0:  # Header row
            cell.set_fontsize(fontsize_label)
        elif col == -1:  # Row labels
            cell.set_fontsize(fontsize_label)
        else:  # Data rows
            cell.set_fontsize(fontsize_tick)
    table2.auto_set_column_width(col=list(range(len(input["df_cov"].columns))))
    #ax_table.set_facecolor("blue")

    ax_trajectories = fig.add_subplot(gs[0:2,3:7])
    for spine in ['top', 'right', 'bottom', 'left']: 
        ax_trajectories.spines[spine].set_visible(False)
    ax_trajectories.set(ylabel=None, xticklabels=[], yticklabels=[])
    ax_trajectories.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax_trajectories.set_xlabel('Time t in days', fontsize=fontsize_label, labelpad=30)
   
    ax_trajectories.margins(x=0.1)

    ax_infc = fig.add_subplot(gs[0:2,3:5])
    ax_seroprev = fig.add_subplot(gs[0:2,5:7])

    # First Plot
    plot_percentile_bands(
        ax_infc,
        input["config"]["timepoints1_nonmissing"],
        input["trajectories_true_1_p"],
        cmap=cmap2,
        label_prefix="true"
    )

    plot_percentile_bands(
        ax_infc,
        input["config"]["timepoints1_nonmissing"],
        input["trajectories_post_1_p"],
        cmap=cmap1,
        label_prefix="posterior"
    )

    ax_infc.plot(
        input["config"]["timepoints1_nonmissing"],
        input["config"]["obs1_nonmissing"],
        markersize=3,
        marker="o",
        linestyle="dotted",
        color="black",
        label="data for experiments",
        zorder=20
    )
    ax_infc.grid(alpha=0.5)
    ax_infc.tick_params(labelsize=fontsize_tick)
    ax_infc.set_ylabel("Infection Count", fontsize=fontsize_label)
    for spine in ['top', 'right', 'left']:  # remove part of the surrounding box, as it gets busy with the grid lines
        ax_infc.spines[spine].set_visible(False)

    ax_infc.text(-0.25, 1.0, 'B', transform=ax_infc.transAxes, fontsize=24,  fontweight='bold', verticalalignment='top', horizontalalignment='left')
      
    # Second Plot
    plot_percentile_bands(
        ax_seroprev,
        input["config"]["timepoints2_nonmissing"],
        input["trajectories_true_2_p"],
        cmap=cmap2,
        label_prefix="true"
    )

    plot_percentile_bands(
        ax_seroprev,
        input["config"]["timepoints2_nonmissing"],
        input["trajectories_post_2_p"],
        cmap=cmap1,
        label_prefix="posterior"
    )

    ax_seroprev.plot(
        input["config"]["timepoints2_nonmissing"],
        input["config"]["obs2_nonmissing"],
        markersize=3,
        marker="o",
        linestyle="dotted",
        color="black",
        label="data for experiments",
        zorder=20
    )
    ax_seroprev.grid(alpha=0.5)
    ax_seroprev.set_axisbelow(True)
    ax_seroprev.tick_params(labelsize=fontsize_tick)
    ax_seroprev.set_ylabel("Seroprevalence", fontsize=fontsize_label)
    for spine in ['top', 'right', 'left']:
        ax_seroprev.spines[spine].set_visible(False)


    #Adding an empty row for spacing
    ax_spacer = fig.add_subplot(gs[n_params_1, 0:8])
    ax_spacer.set_frame_on(False)  
    ax_spacer.set_xticks([])  
    ax_spacer.set_yticks([])  
    ax_spacer.axis('off')

    # Add legend
    handles = [
        Line2D(xdata=[], ydata=[], color=cmap1(0.5), lw=12),
        Line2D(xdata=[], ydata=[], color='black', lw=4,  linestyle='dotted'),
    ]

    labels = [
        "Posterior Draws",
        "Inference Data"
    ]

    if input["true_params"] is not None:
        handles.append(Line2D(xdata=[], ydata=[], color='black', lw=2))
        labels.append("True Parameters")

    if input["map"] is not None:
        handles.append(Line2D(xdata=[], ydata=[], color=cmap1(0.9), lw=2))
        labels.append("MAP")

    if n_params_1 == 6:
        fig.legend(handles, labels, ncol=1, fontsize=32, loc="lower right", bbox_to_anchor=(1, 0.25))
    if n_params_1 == 5:
        fig.legend(handles, labels, ncol=1, fontsize=32, loc="lower right", bbox_to_anchor=(1, 0.1))
    if n_params_1 == 2:
        fig.legend(handles, labels, ncol=3, fontsize=32, loc="lower center")
    fig.tight_layout(pad=2)
    if n_params_1 == 2:
        plt.subplots_adjust(wspace=0.5, hspace=0.3, bottom = 0.05)
    else:
        plt.subplots_adjust(wspace=0.5, hspace=0.3)

    return fig

plt.show()
