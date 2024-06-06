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
    "CoraML": {
        "title": "CoraML",
        "legend_loc": "upper right",
    },
}
models = {
    "Graphormer": {"color": "b", "linestyle": ":", "marker": "o", "markersize": 3},
    "SAN": {"color": "r", "linestyle": "--", "marker": "v", "markersize": 3},
    "GRIT": {"color": "g", "linestyle": "-.", "marker": "*", "markersize": 3},
    "GCN": {"color": "k", "linestyle": (0, (3, 5, 1, 5, 1, 5)), "marker": "s", "markersize": 3},
    "GCN-hom": {"color": "orange", "linestyle": (0, (5, 10)), "marker": "X", "markersize": 3},
    #"GATv2": {"color": "m", "linestyle": (0, (5, 1)), "marker": "p", "markersize": 8},
}
metrics = {"acc": "Accuracy (\%)", "asr": "Attack success rate (\%)", "margin": "Margin"}


def main(
    dataset: str,
    png: bool,
    legend: bool,
    title: bool,
    y_label: bool,
):
    result_dir = Path("results_rut") / dataset
    if not result_dir.is_dir():
        raise ValueError(
            "Can't find results for that dataset. "
            "Please run the rut_results_analysis.py script first to save the results."
        )
    figsize = (3.5, 2.7)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    if title:
        ax.set_title(datasets[dataset]["title"])
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

        df = pd.read_csv(model_dir / coll_name / "results" / f"strongest.csv")
        x = df["budgets"] * 100
        y = df["Accuracy"] * 100
        ax.plot(x, y, alpha=0.7, label=model, color=c, marker=m, markeredgewidth=0.0, markersize=ms)  # linestyle=l
        
    ax.set_xlabel("Edge modification budget (\%)")
    if y_label:
        ax.set_ylabel(metrics["acc"])
    if legend:
        ax.legend(bbox_to_anchor=(1.01, 0), loc="lower left")  # loc=datasets[dataset]["legend_loc"])
        #fig_sb.legend(loc='outside right upper')
    save_figure(fig, f"{dataset}_acc", png)


        
parser = argparse.ArgumentParser(description='Processes the results of transfer attack.')
parser.add_argument("-d", "--dataset")
parser.add_argument("-p", "--png", action="store_true")
parser.add_argument("-l", "--legend", action="store_true")
parser.add_argument("-t", "--title", action="store_true")
parser.add_argument("-y", "--y-label", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        png=args.png,
        legend=args.legend,
        title=args.title,
        y_label=args.y_label,
    )
