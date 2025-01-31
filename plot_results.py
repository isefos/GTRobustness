from pathlib import Path
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pandas as pd


figures_dir = Path("final_figures")
figures_dir.mkdir(exist_ok=True)


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


def save_figure(fig, name, png=False):
    if png:
        file_name = name + ".png"
        fig.savefig(figures_dir / file_name, bbox_inches="tight", dpi=500)
        return
    file_name = name + ".pdf"
    fig.savefig(figures_dir / file_name, bbox_inches="tight")


plots = {
    "CLUSTER_cs": {"dataset": "CLUSTER_cs", "sub_dirs": {
        "Graphormer": "t_gph_cluster_cs_prel/0",
        "GRIT": "t_grt_cluster_cs_prel/0",
        "SAN": "t_san_cluster_cs_prel/0",
        "GPS": "t_gps_cluster_cs/0",
        "GPS-GCN": "t_gpsgcn_cluster_cs/0",
        "Polynormer": "t_ply_cluster_cs/0",
        "GATv2": "t_ga2_cluster_cs_prel/0",
        "GAT": "t_gat_cluster_cs_prel/0",
        "GCN": "t_gcn_cluster_cs_prel/0",
    }},
    "CLUSTER_as": {"dataset": "CLUSTER_as", "sub_dirs": {
        "Graphormer": "t_gph_cluster_as_prel/0",
        "GRIT": "t_grt_cluster_as_prel/0",
        "SAN": "t_san_cluster_as_prel/0",
        "GPS": "t_gps_cluster_as/0",
        "GPS-GCN": "t_gpsgcn_cluster_as/0",
        "Polynormer": "t_ply_cluster_as/0",
        "GATv2": "t_ga2_cluster_as_prel/0",
        "GAT": "t_gat_cluster_as_prel/0",
        "GCN": "t_gcn_cluster_as_prel/0",
    }},
    "UPFD_gos_bert": {"dataset": "UPFD_gos_bert", "sub_dirs": {
        "Graphormer": "t_gph_upfd_gos_bert_prel/0",
        "GRIT": "t_grt_upfd_gos_bert_prel/0",
        "SAN": "t_san_upfd_gos_bert_prel/0",
        "GPS": "t_gps_upfd_gos_bert/0",
        "Polynormer": "t_ply_upfd_gos_bert/0",
        "GCN": "t_gcn_upfd_gos_bert_prel/0",
    }},
    "UPFD_pol_bert": {"dataset": "UPFD_pol_bert", "sub_dirs": {
        "Graphormer": "t_gph_upfd_pol_bert_prel/0",
        "GRIT": "t_grt_upfd_pol_bert_prel/0",
        "SAN": "t_san_upfd_pol_bert_prel/0",
        "GPS": "t_gps_upfd_pol_bert/0",
        "Polynormer": "t_ply_upfd_pol_bert/1",
        "GCN": "t_gcn_upfd_pol_bert_prel/0",
    }},
    "reddit_threads": {"dataset": "reddit_threads", "sub_dirs": {
        "Graphormer": "t_gph_reddit_threads/small",
        "GRIT": "t_grt_reddit_threads/small",
        "SAN": "t_san_reddit_threads/medium2",
        "GPS": "t_gps_reddit_threads/small",
        "GCN": "t_gcn_reddit_threads/tiny",
    }},
    "adv_GCN_UPFD_gos_bert": {"dataset": "UPFD_gos_bert", "sub_dirs": {
        "GCN": "t_gcn_upfd_gos_bert_prel/0",
        r"GCN, b=5\%, k=4": "t_adv_gcn_upfd_gos_bert/11",
        r"GCN, b=10\%, k=4": "t_adv_gcn_upfd_gos_bert/15",
        r"GCN, b=15\%, k=5": "t_adv_gcn_upfd_gos_bert/25",
    }},
    "adv_GCN_UPFD_pol_bert": {"dataset": "UPFD_pol_bert", "sub_dirs": {
        "GCN": "t_gcn_upfd_pol_bert_prel/0",
        r"GCN, b=15\%, k=4": "t_adv_gcn_upfd_pol_bert/18",
        r"GCN, b=15\%, k=6": "t_adv_gcn_upfd_pol_bert/35",
    }},
    "adv_Graphormer_UPFD_gos_bert": {"dataset": "UPFD_gos_bert", "sub_dirs": {
        "Graphormer": "t_gph_upfd_gos_bert_prel/0",
        r"Graphormer, b=5\%, k=4": "t_adv_gph_upfd_gos_bert/12",
        r"Graphormer, b=10\%, k=4": "t_adv_gph_upfd_gos_bert/15",
        r"Graphormer, b=15\%, k=4": "t_adv_gph_upfd_gos_bert/18",
    }},
    "adv_Graphormer_UPFD_pol_bert": {"dataset": "UPFD_pol_bert", "sub_dirs": {
        "Graphormer": "t_gph_upfd_pol_bert_prel/0",
        r"Graphormer, b=15\%, k=4": "t_adv_gph_upfd_pol_bert/18",
        r"Graphormer, b=15\%, k=6": "t_adv_gph_upfd_pol_bert/35",
    }},
}
datasets = {
    "CLUSTER_cs": {
        "title": "CLUSTER (constrained attack)",
        "legend_loc": "upper right",
    },
    "CLUSTER_as": {
        "title": "CLUSTER",
        "legend_loc": "upper right",
    },
    "UPFD_gos_bert": {
        "title": "UPFD (Twitter) gossipcop",
        "legend_loc": "upper right",
    },
    "UPFD_pol_bert": {
        "title": "UPFD (Twitter) politifact",
        "legend_loc": "upper right",
    },
    "reddit_threads": {
        "title": "Reddit Threads",
        "legend_loc": "upper right",
    },
}
# In order of GT -> GNN, diverging color scheme
model_order = {
    "Graphormer": 0,
    "GRIT": 1,
    "SAN": 2,
    "GPS": 3,
    "GPS-GCN": 4,
    "Polynormer": 5,
    "GATv2": 6,
    "GAT": 7,
    "GCN": 8,
}
adv_order = {
    "UPFD_gos_bert": {
        "GCN": 9,
        r"GCN, b=5\%, k=4": 10,
        r"GCN, b=10\%, k=4": 11,
        r"GCN, b=15\%, k=5": 12,
        "Graphormer": 9,
        r"Graphormer, b=5\%, k=4": 10,
        r"Graphormer, b=10\%, k=4": 11,
        r"Graphormer, b=15\%, k=4": 12,
    },
    "UPFD_pol_bert": {
        "GCN": 9,
        r"GCN, b=15\%, k=4": 11,
        r"GCN, b=15\%, k=6": 12,
        "Graphormer": 9,
        r"Graphormer, b=15\%, k=4": 11,
        r"Graphormer, b=15\%, k=6": 12,
    },
}
syles = {
    "color": [
        #"#ad0d00", "#d65e2b", "#eda356", "#deba6b", "#aaac8c", "#92b4bf", "#689ab1", "#3e769d", "#0a3b87",
        #'#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499',
        '#A50026', '#DD3D2D', '#e66E3B', '#dD9346', '#cEaA5B', '#8A8C6C', '#92b4bF', '#5E96bD', '#364B9A',
        # for adversarial
        #'#cDdCaF', '#c0d7bA', '#b2d3c2', '#a5cDc8', '#98c8cC', '#8Bc2d1', '#7DbBd4', '#71b4d7', '#6BaCd7', '#6Ea2d4'
        #'#A80003', '#E40515', '#F94902', '#F6790B', '#F19903', '#E7B503', '#D5CE04', '#BBE453', '#A2F49B', '#C6F7D6', '#CEFFFF'
        '#E40515', '#F19903', '#D5CE04', '#A2F49B',
    ],
    "linestyle": [
        "--",
        (1, (3, 3, 1, 3, 2, 2)),
        "-.",
        (2, (3, 3, 1, 2, 1, 1)), 
        (3, (5, 10)), 
        (4, (5, 1)), 
        "dotted", 
        (5, (5, 2, 4, 1)),
        (6, (3, 2, 2, 1)),
        # now for the adv GCN b = 5, 10, 15, 15
        #(0, (3, 5, 1, 5)),
        #"-.",
        #(0, (3, 5, 1, 5, 1, 5)), 
        #(0, (5, 10)), 
        # now for the adv Graphormer b = 5, 10, 15, 15
        #(0, (5, 1)), 
        #"dotted", 
        #(0, (5, 10)),
        #(0, (3, 10, 1, 15)),
    ],
    "marker": [
        "X", "s", "v", "p", ">", "P", "^", "h", "d",
        # now for the adv GCN b = 5, 10, 15, 15
        #"P", "^", "h", "X",
        # now for the adv Graphormer b = 5, 10, 15, 15
        #"s", "v", "p", ">",
    ],
    "markersize": [
        7, 6, 8, 8, 8, 8, 8, 8, 7,
        # now for the adv GCN b = 5, 10, 15, 15
        #7, 7, 8, 7,
        # now for the adv Graphormer b = 5, 10, 15, 15
        #6, 8, 8, 7,
    ],
}
metrics = {"acc": "Accuracy (\%)", "asr": "Attack success rate (\%)", "margin": "Margin"}


def main(
    plot: str,
    metric: str,
    max_idx_small_budget: int,
    png: bool,
    legend: bool,
    legend_sb: bool,
    title: bool,
    y_label: bool,
    y_min: None | float,
    y_max: None | float,
    y_min_sb: None | float,
    y_max_sb: None | float,
    figsize: tuple[float, float],
):
    s = max_idx_small_budget
    dataset = plots[plot]["dataset"]
    result_dir = Path("results_t") / dataset
    if not result_dir.is_dir():
        raise ValueError(
            "Can't find results for that dataset. "
            "Please run the t_results_analysis.py script first to save the results."
        )

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig_sb, ax_sb = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    if title:
        ax.set_title(datasets[dataset]["title"])
        ax_sb.set_title(datasets[dataset]["title"])
    for i, (model, sub_dir) in enumerate(plots[plot]["sub_dirs"].items()):
        if plot.startswith("adv"):
            model_names = model.split(" ")
            if model in model_order:
                model_label = "normal"
            else:
                budget = model_names[1]
                steps = model_names[2]
                model_label = f"adv. {budget} {steps}"
            model_name = model_names[0]
            if model_name[-1] == ",":
                model_name = model_name[:-1]
        else:
            model_name = model
            model_label = model
        idx = model_order[model_name]
        if plot.startswith("adv"):
            c_idx = adv_order[dataset][model]
            m_idx = (idx + i) % len(model_order)
        else:
            c_idx = idx
            m_idx = idx
        file: Path = result_dir / model_name / sub_dir / "results" / f"strongest_{metric}.csv"
        if not file.is_file():
            raise Exception(f"The results file ({file}) does not exist, run transfer first!")
        c = syles["color"][c_idx]
        l = syles["linestyle"][m_idx]
        #l = (2 * i, (3, 1, 4, 1))
        m = syles["marker"][m_idx]
        ms = syles["markersize"][m_idx]

        df = pd.read_csv(file)
        x = df["budget"] * 100
        y = df["mean"]
        std = df["std"]
        if metric != "margin":
            y *= 100
            std *= 100
        ax.plot(x, y, alpha=0.9, label=model_label, color=c, marker=m, linestyle=l, markeredgewidth=0.0, markersize=ms)
        ax.fill_between(x, y-std, y+std, color=c, alpha=0.2, linewidth=0.0)
        ax_sb.plot(x[:s+1], y[:s+1], alpha=0.9, label=model_label, color=c, marker=m, linestyle=l, markeredgewidth=0.0, markersize=ms)
        ax_sb.fill_between(x[:s+1], y[:s+1]-std[:s+1], y[:s+1]+std[:s+1], color=c, alpha=0.2, linewidth=0.0)
        
    ax.set_xlabel("Edge modification budget (\%)")
    ax_sb.set_xlabel("Edge modification budget (\%)")
    ax.set_ylim(bottom=y_min, top=y_max)
    ax_sb.set_ylim(bottom=y_min_sb, top=y_max_sb)
    ax_sb.set_xlim(left=-0.05*x[s-1], right=1.1*x[s-1])
    if y_label:
        ax.set_ylabel(metrics[metric])
        ax_sb.set_ylabel(metrics[metric])
    if legend:
        #ax.legend(bbox_to_anchor=(1.01, 0), loc="lower left")  # loc=datasets[dataset]["legend_loc"])
        ax.legend(prop={'size': 8})  # loc='upper right')
        #ax.legend(bbox_to_anchor=(1.01, 0), loc="lower left", prop={'size': 8})
    if legend_sb:
        #ax_sb.legend(bbox_to_anchor=(1.01, 0), loc="lower left", prop={'size': 8})  # loc=datasets[dataset]["legend_loc"])
        ax_sb.legend(prop={'size': 8})  # loc='upper right')
    save_figure(fig, f"{plot}_{metric}", png)
    save_figure(fig_sb, f"{plot}_{metric}_sb", png)


        
parser = argparse.ArgumentParser(description='Processes the results of transfer attack.')
parser.add_argument("-p", "--plot")
parser.add_argument("-m", "--metric")
parser.add_argument("-s", "--small-budget-idx")
parser.add_argument("--png", action="store_true")
parser.add_argument("-l", "--legend", action="store_true")
parser.add_argument("--legend-sb", action="store_true")
parser.add_argument("-t", "--title", action="store_true")
parser.add_argument("-y", "--y-label", action="store_true")
parser.add_argument("--y-min", type=float, default=None)
parser.add_argument("--y-max", type=float, default=None)
parser.add_argument("--y-min-sb", type=float, default=None)
parser.add_argument("--y-max-sb", type=float, default=None)
parser.add_argument("--fs-w", type=float, default=2.5)
parser.add_argument("--fs-h", type=float, default=1.7)


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.metric in metrics, f"Invalid metric argument, must be one of {list(metrics.keys())}"
    main(
        plot=args.plot,
        metric=args.metric,
        max_idx_small_budget=int(args.small_budget_idx),
        png=args.png,
        legend=args.legend,
        legend_sb=args.legend_sb,
        title=args.title,
        y_label=args.y_label,
        y_min=args.y_min,
        y_max=args.y_max,
        y_min_sb=args.y_min_sb,
        y_max_sb=args.y_max_sb,
        figsize=(args.fs_w, args.fs_h),
    )
