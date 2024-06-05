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


def save_figure(fig, name, png=False):
    if png:
        file_name = name + ".png"
        fig.savefig(figures_dir / file_name, bbox_inches="tight", dpi=500)
        return
    file_name = name + ".pdf"
    fig.savefig(figures_dir / file_name, bbox_inches="tight")


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
}
models = {
    "Graphormer": {"color": "b", "linestyle": ":", "marker": "o", "markersize": 6},
    "SAN": {"color": "r", "linestyle": "--", "marker": "v", "markersize": 7},
    "GRIT": {"color": "g", "linestyle": "-.", "marker": "*", "markersize": 9},
    "GCN": {"color": "k", "linestyle": (0, (3, 5, 1, 5, 1, 5)), "marker": "s", "markersize": 6},
    "GAT": {"color": "orange", "linestyle": (0, (5, 10)), "marker": "X", "markersize": 8},
    "GATv2": {"color": "m", "linestyle": (0, (5, 1)), "marker": "p", "markersize": 8},
}
metrics = {"acc": "Accuracy (\%)", "asr": "Attack success rate (\%)", "margin": "Margin"}


def main(
    dataset: str,
    metric: str,
    max_idx_small_budget: int,
    png: bool,
    legend: bool,
    title: bool,
    y_label: bool,
):
    s = max_idx_small_budget
    result_dir = Path("results_t") / dataset
    if not result_dir.is_dir():
        raise ValueError(
            "Can't find results for that dataset. "
            "Please run the t_results_analysis.py script first to save the results."
        )
    figsize = (2.5, 1.7)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig_sb, ax_sb = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    if title:
        ax.set_title(datasets[dataset]["title"])
        ax_sb.set_title(datasets[dataset]["title"])
    for model_dir in result_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        c = models[model]["color"]
        l = models[model]["linestyle"]
        m = models[model]["marker"]
        ms = models[model]["markersize"]

        collections = [sd.name for sd in model_dir.iterdir()]
        assert len(collections), "Not implemented yet to handle more than one collection"
        coll_name = collections[0]

        df = pd.read_csv(model_dir / coll_name / "results" / f"strongest_{metric}.csv")
        x = df["budget"] * 100
        y = df["mean"]
        std = df["std"]
        if metric != "margin":
            y *= 100
            std *= 100
        ax.plot(x, y, alpha=0.7, label=model, color=c, marker=m, linestyle=l, markeredgewidth=0.0, markersize=ms)
        ax.fill_between(x, y-std, y+std, color=c, alpha=0.1, linewidth=0.0)
        ax_sb.plot(x[:s], y[:s], alpha=0.7, label=model, color=c, marker=m, linestyle=l, markeredgewidth=0.0, markersize=ms)
        ax_sb.fill_between(x[:s], y[:s]-std[:s], y[:s]+std[:s], color=c, alpha=0.1, linewidth=0.0)
        
    ax.set_xlabel("Edge modification budget (\%)")
    ax_sb.set_xlabel("Edge modification budget (\%)")
    if y_label:
        ax.set_ylabel(metrics[metric])
        ax_sb.set_ylabel(metrics[metric])
    if legend:
        ax.legend(bbox_to_anchor=(1.01, 0), loc="lower left")  # loc=datasets[dataset]["legend_loc"])
        #fig.legend(loc='outside right upper')
        ax_sb.legend(bbox_to_anchor=(1.01, 0), loc="lower left")  # loc=datasets[dataset]["legend_loc"])
        #fig_sb.legend(loc='outside right upper')
    save_figure(fig, f"{dataset}_{metric}", png)
    save_figure(fig_sb, f"{dataset}_{metric}_sb", png)


        
parser = argparse.ArgumentParser(description='Processes the results of transfer attack.')
parser.add_argument("-d", "--dataset")
parser.add_argument("-m", "--metric")
parser.add_argument("-s", "--small-budget-idx")
parser.add_argument("-p", "--png", action="store_true")
parser.add_argument("-l", "--legend", action="store_true")
parser.add_argument("-t", "--title", action="store_true")
parser.add_argument("-y", "--y-label", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.metric in metrics, f"Invalid metric argument, must be one of {list(metrics.keys())}"
    main(
        dataset=args.dataset,
        metric=args.metric,
        max_idx_small_budget=int(args.small_budget_idx),
        png=args.png,
        legend=args.legend,
        title=args.title,
        y_label=args.y_label,
    )
