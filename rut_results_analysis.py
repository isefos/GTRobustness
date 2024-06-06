import seml
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import argparse
import shutil


use_tex = True
if use_tex:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })


def save_figure(fig, name, figures_dir, png=False):
    if png:
        file_name = name + ".png"
        fig.savefig(figures_dir / file_name, bbox_inches="tight", dpi=500)
        return
    file_name = name + ".pdf"
    fig.savefig(figures_dir / file_name, bbox_inches="tight")


datasets = {
    "CoraML-RUT": {"format": "PyG-RobustnessUnitTest", "name": "cora_ml"},
    "Citeseer-RUT": {"format": "PyG-RobustnessUnitTest", "name": "citeseer"},
}
models = {
    "Graphormer": {"type": set(["Graphormer"]), "gnn_layer_type": None},
    "SAN": {"type": set(["SANTransformer", "WeightedSANTransformer"]), "gnn_layer_type": None},
    "GRIT": {"type": set(["GritTransformer"]), "gnn_layer_type": None},
    "GCN": {"type": set(["gnn"]), "gnn_layer_type": set(["gcnconvweighted", "gcnconv"])},
    "GCN-hom": {"type": set(["gnn"]), "gnn_layer_type": set(["gcnconvweighted", "gcnconv"])},
    "GAT": {"type": set(["gnn"]), "gnn_layer_type": set(["gatconvweighted", "gatconv"])},
    "GATv2": {"type": set(["gnn"]), "gnn_layer_type": set(["gatv2convweighted", "gatv2conv"])},
}
num_edges_clean = {
    "CoraML-RUT": 5069,
    "Citeseer-RUT": "TODO: replace with actual number",
}


attack_cols = {
    "Attack success rate": "attack_success_rate",
    "Accuracy": "correct_acc",
    "Average margin": "margin_mean",
}


# cols = ["clean", "gcn", "jaccard_gcn", "svd_gcn", "rgcn", "pro_gcn", "gnn_guard", "grand", "soft_median_gdc"]
# output_keys = ["correct_acc", "margin_mean", "margin_median", "margin_min", "margin_max"]

extra_runs_names = ["chain", "random", "radom0.1x", "random0.1x", "random10x"]


def clean_path(results_path: str):
    results_path = Path(results_path)
    if results_path.exists():
        shutil.rmtree(results_path)
    results_path.mkdir(parents=True)
    general_info_file = results_path / "runs_infos.txt"
    csv_dir = results_path / "results"
    csv_dir.mkdir()
    return results_path, general_info_file, csv_dir


def write_info_file(info_file, num_params, extras, run_dir, extra_runs):
    with open(info_file, "w") as f:
        f.write("RUT run infos:")
        f.write(f"\n\tnum_params: {num_params}")
        for k, v in extras.items():
            f.write(f"\n\t{k}: {v}")
        f.write(f"\n\trun_dir: {run_dir}")
        f.write("\n\textra perturbations:")
        for run_name, res in extra_runs.items():
            f.write(f"\n\t\t{run_name}")
            for k, v in res.items():
                f.write(f"\n\t\t\t{k}: {v}")
        f.write("\n")


def write_results(results_dir, res):
    run_dfs = {}
    extra_runs = {}
    clean_acc, clean_margin_mean = None, None
    for run_name, results in res["result"]["robustness_unit_test"].items():
        if run_name == "clean":
            clean_acc = results["correct_acc"]
            clean_margin_mean = results["margin_mean"]
            continue

        if run_name in extra_runs_names:
            extra_runs[run_name] = {
                "attack_success_rate": results["attack_success_rate"],
                "correct_acc": results["correct_acc"],
                "margin_mean": results["margin_mean"],
            }
            continue

        df_result = {title: results[col] for title, col in attack_cols.items()}
        df_result["budgets"] = results["budgets"]
        df = pd.DataFrame(df_result)
        run_dfs[run_name] = df
        df.to_csv(results_dir / f"{run_name}.csv")

    assert clean_acc is not None and clean_margin_mean is not None
    return run_dfs, clean_acc, clean_margin_mean, extra_runs


def save_plots(
    model,
    dataset,
    run_dfs,
    results_path,
    clean_acc,
    clean_margin_mean,
):
    plots_dir = results_path / "plots"
    plots_dir_runs = plots_dir / "individual"

    all_agg_results = {}
    # find the strongest attack:
    num_edges = num_edges_clean[dataset]
    e_budgets = [i * 0.0075 for i in range(21)]
    num_budgets = np.array([e * num_edges for e in e_budgets])
    strongest_acc = np.ones(21)

    for run_name, df in run_dfs.items():

        plots_dir_run = plots_dir_runs / run_name
        plots_dir_run.mkdir(parents=True)

        # plot aggregate
        for title in attack_cols:

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
            #ax.set_title(f"{model} - {dataset.replace('_', ' ')}")

            if title == "Accuracy":
                zb_val = clean_acc
            elif title == "Average margin":
                zb_val = clean_margin_mean
            elif title == "Attack success rate":
                zb_val = 0.0
            else:
                raise Exception("Probably changed the titles in attack_cols...")

            if title not in all_agg_results:
                all_agg_results[title] = {}

            x = np.concatenate((np.array([0.0]), np.array(df["budgets"])))
            y = np.concatenate((np.array([zb_val]), np.array(df[title])))
            ax.plot(x, y)

            all_agg_results[title][run_name] = {"x": x, "y": y}

            if title == "Accuracy":
                budget_idx = np.searchsorted(num_budgets, x)
                strongest_acc[budget_idx] = np.minimum(strongest_acc[budget_idx], y)

            ax.set_xlabel("Num. edges flipped")
            ax.set_ylabel(title)
            #ax.legend()
            save_figure(fig, f"{dataset}_{model}_{title.replace(' ', '_')}", plots_dir_run)
            ax.clear()
            plt.close(fig)

    plots_dir_all_runs = plots_dir / "all"
    plots_dir_all_runs.mkdir()

    for title, runs_stats in all_agg_results.items():
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
        #ax.set_title(f"{model} - {dataset.replace('_', ' ')}")
        for run_name, res in runs_stats.items():
            x = res["x"]
            y = res["y"]
            ax.plot(x, y, label=run_name)
        ax.set_xlabel("Num. edges flipped")
        ax.set_ylabel(title)
        ax.legend()
        save_figure(fig, f"{dataset}_{model}_{title.replace(' ', '_')}", plots_dir_all_runs)
        ax.clear()
        plt.close(fig)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    ax.plot(e_budgets, strongest_acc)
    ax.set_xlabel(r"Budget (\% edges flipped)")
    ax.set_ylabel(title)
    save_figure(fig, f"strongest_{dataset}_{model}_{title.replace(' ', '_')}", plots_dir_all_runs)
    ax.clear()
    plt.close(fig)

    df_strongest = pd.DataFrame({"budgets": e_budgets, "Accuracy": strongest_acc})
    df_strongest.to_csv(results_path / "results" / "strongest.csv")


def get_collection_results(collection, filter_dict):
    extra_fields = [
        'slurm.array_id', 'slurm.experiments_per_job', 'slurm.task_id', 'stats.real_time',
        'stats.pytorch.gpu_max_memory_bytes', 'stats.self.max_memory_bytes',
    ]
    results = seml.get_results(
        collection,
        ['config', 'result'] + extra_fields,
        filter_dict=filter_dict,
    )
    assert len(results) == 1, "Expected only one run containing all results"
    r = results[0]
    extras = dict()
    for key in extra_fields:
        r_e = r
        keys_list = key.split(".")
        for key_l in keys_list[:-1]:
            r_e = r_e.get(key_l, {})
        key_last = keys_list[-1]
        v = r_e.get(key_last, None)
        if v is not None:
            if key_last.endswith("bytes"):
                v = f"{v * 1e-9:.1f} GB"
            if key_last.endswith("time"):
                v = f"{v / 3600:.2f} hours"
        extras[key] = v
    run_dir = r["result"].get("run_dir")
    num_params = r["result"].get("num_params")
    return r, extras, run_dir, num_params


def check_input_result_match(res, dataset, model):
    df = res["config"]["graphgym"]["dataset"]["format"]
    dfg = datasets[dataset]["format"]
    dn = res["config"]["graphgym"]["dataset"]["name"]
    dng = datasets[dataset]["name"]
    assert df == dfg, (f"Dataset format was given to be `{dfg}`, but encountered `{df}`.")
    assert dn == dng, (f"Dataset name was given to be `{dng}`, but encountered `{dn}`.")
    
    mt = res["config"]["graphgym"]["model"]["type"]
    mtg = models[model]["type"]
    assert mt in mtg, (f"Model was given to be in {mtg}, but encountered `{mt}`.")
    mlg = models[model]["gnn_layer_type"]
    if mlg is not None:
        ml = res["config"]["graphgym"]["gnn"]["layer_type"]
        assert ml in mlg, (f"Model layer was given to be in {mlg}, but encountered `{ml}`.")


def main(
    collection: str,
    results_path: str,
    filter_dict,
    dataset: str,
    model: str,
):
    res, extras, run_dir, num_params = get_collection_results(collection, filter_dict)

    check_input_result_match(res, dataset, model)

    results_path, info_file, csv_dir = clean_path(results_path)
    # write results into file
    run_dfs, clean_acc, clean_margin_mean, extra_runs = write_results(csv_dir, res)
    write_info_file(info_file, num_params, extras, run_dir, extra_runs)
    # plots
    save_plots(
        model,
        dataset,
        run_dfs,
        results_path,
        clean_acc,
        clean_margin_mean,
    )


parser = argparse.ArgumentParser(description='Processes the results of transfer attack.')
parser.add_argument("-c", "--collection")
parser.add_argument("-d", "--dataset")
parser.add_argument("-m", "--model")


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.dataset in datasets
    results_path = f"results_rut/{args.dataset}/{args.model}/{args.collection}"
    # not implemented for argparse... but can manually change here
    filter_dict = None  # {"config.graphgym.attack.cluster_sampling": True}
    main(
        collection=args.collection,
        results_path=results_path,
        filter_dict=filter_dict,
        dataset=args.dataset,
        model=args.model,
    )
