from pathlib import Path
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from plot_results import save_figure, plots, model_order, syles


use_tex = True
if use_tex:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


if __name__ == "__main__":
    figsize = (0.01, 0.01)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    plot = "CLUSTER_as"
    for i, (model, sub_dir) in enumerate(plots[plot]["sub_dirs"].items()):
        model_name = model
        model_label = model
        idx = model_order[model_name]
        c_idx = idx
        m_idx = idx
        c = syles["color"][c_idx]
        l = syles["linestyle"][m_idx]
        #l = (2 * i, (3, 1, 4, 1))
        m = syles["marker"][m_idx]
        ms = syles["markersize"][m_idx]
        ax.plot([], [], alpha=0.9, label=model_label, color=c, marker=m, linestyle=l, markeredgewidth=0.0, markersize=ms)
        
    plt.axis('off')
    ax.legend(loc="center", ncols=9)

    save_figure(fig, "legend")
