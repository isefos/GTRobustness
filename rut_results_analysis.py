import seml
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import argparse
import shutil
from collections import defaultdict


datasets = {
    "CLUSTER": {"format":"PyG-GNNBenchmarkDataset", "name": "CLUSTER"},
    "CoraML-RUT": {"format": "PyG-RobustnessUnitTest", "name": "cora_ml"},
    "Citeseer-RUT": {"format": "PyG-RobustnessUnitTest", "name": "citeseer"},
    "UPFD_gos_bert": {"format": "PyG-UPFD", "name": "gossipcop-bert"},
    "UPFD_pol_bert": {"format": "PyG-UPFD", "name": "politifact-bert"},
}
models = {
    "Graphormer": {"type": set(["Graphormer"]), "gnn_layer_type": None},
    "SAN": {"type": set(["SANTransformer", "WeightedSANTransformer"]), "gnn_layer_type": None},
    "GRIT": {"type": set(["GritTransformer"]), "gnn_layer_type": None},
    "GCN": {"type": set(["gnn"]), "gnn_layer_type": set(["gcnconvweighted", "gcnconv"])},
    "GAT": {"type": set(["gnn"]), "gnn_layer_type": set(["gatconvweighted", "gatconv"])},
    "GATv2": {"type": set(["gnn"]), "gnn_layer_type": set(["gatv2convweighted", "gatv2conv"])},
}


attack_cols = {
    "Attack success rate": "attack_success_rate",
    "Accuracy": "correct_acc",
    "Average margin": "margin_mean",
}


# cols = ["clean", "gcn", "jaccard_gcn", "svd_gcn", "rgcn", "pro_gcn", "gnn_guard", "grand", "soft_median_gdc"]
output_keys = ["correct_acc", "margin_mean", "margin_median", "margin_min", "margin_max"]


def clean_path(results_path: str):
    results_path = Path(results_path)
    if results_path.exists():
        shutil.rmtree(results_path)
    results_path.mkdir(parents=True)
    general_info_file = results_path / "runs_infos.txt"
    csv_dir = results_path / "results"
    csv_dir.mkdir()
    return results_path, general_info_file, csv_dir


def write_info_file(info_file, num_params, extras, run_dir):
    with open(info_file, "w") as f:
        f.write("RUT run infos:")
        f.write(f"\n\tnum_params: {num_params}")
        for k, v in extras.items():
            f.write(f"\n\t{k}: {v}")
        f.write(f"\n\trun_dir: {run_dir}")
        f.write("\n")


def write_results(
    info_file,
    results_dir,
    res,
    extras,
    run_dir,
    num_params,
    attack_cols,
):
    write_info_file(info_file, num_params, extras, run_dir)
    run_dfs = {}
    clean_acc, clean_margin_mean = None, None
    for run_name, results in res["result"]["robustness_unit_test"].items():
        if run_name == "clean":
            clean_acc = results["correct_acc"]
            clean_margin_mean = results["margin_mean"]
            continue

        # TODO: maybe also add num clean graphs for each dataset to get relative budget used
        df_result = {title: results[col] for title, col in attack_cols.items()}
        df_result["budgets"] = results["budgets"]
        df = pd.DataFrame(df_result)
        run_dfs[run_name] = df
        df.to_csv(results_dir / f"{run_name}.csv")

    assert clean_acc is not None and clean_margin_mean is not None
    return run_dfs, clean_acc, clean_margin_mean


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

    for run_name, df in run_dfs.items():

        plots_dir_run = plots_dir_runs / run_name
        plots_dir_run.mkdir(parents=True)
        
        # aggregate seeds, mean and error std
        df_agg = pd.concat(list(seed_dfs.values()))
        df_agg_mean = df_agg.groupby("budget", as_index=False).mean()
        df_agg_std = df_agg.groupby("budget", as_index=False).std()

        # plot aggregate
        for title, col_group in attack_cols.items():

            for budget_measure, budget_dict in budget_measures.items():

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
                ax.set_title(f"{model} - {dataset.replace('_', ' ')}")

                if "clean" in col_group:
                    zb_val = df_agg_mean[col_group["clean"]][0]
                else:
                    zb_val = 0.0

                if title not in all_agg_results:
                    all_agg_results[title] = {budget_measure: {run_name: {}}}
                elif budget_measure not in all_agg_results[title]:
                    all_agg_results[title][budget_measure] = {run_name: {}}
                else:
                    all_agg_results[title][budget_measure][run_name] = {}

                for label, col in col_group.items():
                    if label == "clean":
                        continue  #  don't plot the clean (is equal to first step anyway)
                    x = np.concatenate((np.array([0.0]), np.array(df_agg_mean.index)))
                    x = np.concatenate(
                        (np.array([0.0]), np.array(df_agg_mean[budget_dict["key"]]))
                    ) * budget_dict["mul"]
                    y = np.concatenate((np.array([zb_val]), np.array(df_agg_mean[col])))
                    std = np.concatenate((np.array([0.0]), np.array(df_agg_std[col])))
                    ax.plot(x, y, label=label)
                    ax.fill_between(x, y-std, y+std, alpha=0.2)

                    all_agg_results[title][budget_measure][run_name][col] = {"x": x, "y": y, "std": std}

                ax.set_xlabel(budget_dict["label"])
                ax.set_ylabel(title)
                ax.legend()
                fig.savefig(plots_dir_run / f"{dataset}_{model}_{title.replace(' ', '_')}_{budget_measure}.png")
                ax.clear()
                plt.close(fig)

    plots_dir_all_runs = plots_dir / "all"
    plots_dir_all_runs.mkdir()

    for title, run_stats_b in all_agg_results.items():

        for budget_measure, budget_dict in budget_measures.items():

            run_stats = run_stats_b[budget_measure]
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
            ax.set_title(f"{model} - {dataset.replace('_', ' ')}")
            for run_name, agg_res in run_stats.items():
                for col, res in agg_res.items():
                    x = res["x"]
                    y = res["y"]
                    std = res["std"]
                    if run_name == model:
                        label = f"adaptive"
                    else:
                        label = f"transfer from {run_name}"
                    ax.plot(x, y, label=label)
                    ax.fill_between(x, y-std, y+std, alpha=0.2)
            ax.set_xlabel(budget_dict["label"])
            ax.set_ylabel(title)
            ax.legend()
            fig.savefig(plots_dir_all_runs / f"{dataset}_{model}_{title.replace(' ', '_')}_{budget_measure}.png")
            ax.clear()
            plt.close(fig)


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
        keys_list = key.split(".")
        for key_l in keys_list[:-1]:
            r = r.get(key_l, {})
        key_last = keys_list[-1]
        v = r.get(key_last, None)
        if v is not None:
            if key_last.endswith("bytes"):
                v = f"{v * 1e-9:.1f} GB"
            if key_last.endswith("time"):
                v = f"{v / 3600:.2f} hours"
        extras[key] = v
    run_dir = r["result"].get("run_dir")
    num_params = r["result"].get("num_params")
    return r, extras, run_dir, num_params


def main(
    collection: str,
    results_path: str,
    filter_dict,
    dataset: str,
    model: str,
):
    res, extras, run_dir, num_params = get_collection_results(collection, filter_dict)

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

    pred_level = res["config"]["graphgym"]["attack"]["prediction_level"]
    assert pred_level == "node"

    results_path, info_file, csv_dir = clean_path(results_path)
    # write results into file
    run_dfs, clean_acc, clean_margin_mean = write_results(
        info_file,
        csv_dir,
        res,
        extras,
        run_dir,
        num_params,
        attack_cols,
    )
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
