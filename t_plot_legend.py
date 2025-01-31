import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pathlib import Path
from t_results_analysis import styles, save_figure, model_order


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
    file = Path("results_t")
    name = "legend"

    figsize = (0.01, 0.01)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # random pert
    l = "random_pert"                            
    c = styles[l]["color"]
    ls = styles[l]["linestyle"]
    m = styles[l]["marker"]
    ms = styles[l]["markersize"]
    label = f"random perturbation"
    ax.plot([], [], label=label, alpha=0.8, color=c, marker=m, linestyle=ls, markeredgewidth=0.0, markersize=ms)

    # random attack
    l = "random"                            
    c = styles[l]["color"]
    ls = styles[l]["linestyle"]
    m = styles[l]["marker"]
    ms = styles[l]["markersize"]
    label = f"random attack"
    ax.plot([], [], label=label, alpha=0.8, color=c, marker=m, linestyle=ls, markeredgewidth=0.0, markersize=ms)

    l = "transfer"                            
    c = styles[l]["color"]
    ls = styles[l]["linestyle"]
    m = styles[l]["marker"]
    ms = styles[l]["markersize"]

    # GCN PRBCD transfer
    idx = model_order["GCN"]
    c = c[idx]
    ls = ls[idx]
    m = m[idx]
    ms = ms[idx]
    label = f"GCN PRBCD transfer"
    ax.plot([], [], label=label, alpha=0.8, color=c, marker=m, linestyle=ls, markeredgewidth=0.0, markersize=ms)

    # best transfer
    c = styles[l]["color"][2]
    ls = styles[l]["linestyle"][0]
    m = styles[l]["marker"][0]
    ms = styles[l]["markersize"][0]
    label = f"best transfer (incl. ours)"
    #ax.plot([], [], label=label, alpha=0.8, color=c, marker=m, linestyle=ls, markeredgewidth=0.0, markersize=ms)

    # adaptive PRBCD
    l = "adaptive"                            
    c = styles[l]["color"]
    ls = styles[l]["linestyle"]
    m = styles[l]["marker"]
    ms = styles[l]["markersize"]
    label = f"adaptive PRBCD (ours)"
    ax.plot([], [], label=label, alpha=0.8, color=c, marker=m, linestyle=ls, markeredgewidth=0.0, markersize=ms)

    plt.axis('off')
    ax.legend(loc="center", ncols=5)

    save_figure(fig, file, name)
    ax.clear()
    plt.close(fig)
