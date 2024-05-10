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
}


attack_cols_graph = {
    "Average attack success rate": {
        "adaptive": "avg_attack_success_rate", "random": "avg_attack_success_rate_random",
    },
    "Average accuracy": {
        "clean": "avg_correct_clean", "adaptive": "avg_correct_pert", "random": "avg_correct_pert_random",
    },
    "Average margin": {
        "clean": "avg_margin_clean", "adaptive": "avg_margin_pert", "random": "avg_margin_pert_random",
    },
}
attack_cols_node = {
    "Average attack success rate": {
        "adaptive": "avg_attack_success_rate", "random": "avg_attack_success_rate_random", 
    },
    "Average accuracy": {
        "clean": "avg_correct_acc_clean", "adaptive": "avg_correct_acc_pert", "random": "avg_correct_acc_pert_random",
    },
    "Average mean margin": {
        "clean": "avg_margin_mean_clean", "adaptive": "avg_margin_mean_pert", "random": "avg_margin_mean_pert_random",
    },
}


budget_measures = {
    "budget": {
        "keys": {"adaptive": "budget", "random": "budget"},
        "label": r"Budget (% edge modifications allowed)",
        "mul": 100,
    },
    "budget_used": {
        "keys": {"adaptive": "bu", "random": "bur"},
        "label": r"Average edges modified (%)",
        "mul": 100,
    },
    "num_modifications": {
        "keys": {"adaptive": "m", "random": "mr"},
        "label": r"Average num. edges modified",
        "mul": 1,
    },
}


def clean_path(results_path: str):
    results_path = Path(results_path)
    if results_path.exists():
        shutil.rmtree(results_path)
    results_path.mkdir(parents=True)
    general_info_file = results_path / "runs_infos.txt"
    seed_dir = results_path / "results"
    seed_dir.mkdir()
    return results_path, general_info_file, seed_dir


def write_info_file(info_file, run_ids, num_params, extras, seeds_graphgym, seeds_seml, run_dirs, budgets):
    with open(info_file, "w") as f:
        f.write("Attack run infos:")
        for i, run_id in enumerate(run_ids):
            f.write(f"\nrun: {run_id}")
            f.write(f"\n\te_budget: {budgets[i]}")
            f.write(f"\n\tnum_params: {num_params[i]}")
            for k, v in extras.items():
                f.write(f"\n\t{k}: {v[i]}")
            f.write(f"\n\tgraphgym.seed: {seeds_graphgym[i]}")
            f.write(f"\n\tseed(seml): {seeds_seml[i]}")
            f.write(f"\n\trun_dir: {run_dirs[i]}")
            f.write("\n")


def write_results(
    info_file,
    seed_dir,
    results,
    run_ids,
    extras,
    seeds_graphgym,
    seeds_seml,
    run_dirs,
    num_params,
    budgets,
    attack_cols,
):
    write_info_file(info_file, run_ids, num_params, extras, seeds_graphgym, seeds_seml, run_dirs, budgets)
    cols = []
    for col_group in attack_cols.values():
        cols.extend(list(col_group.values()))
    seed_results = {}
    for i, seed in enumerate(seeds_graphgym):
        if seed not in seed_results:
            seed_results[seed] = defaultdict(list)
        seed_results[seed]["run_ids"].append(run_ids[i])
        for budget_measure, budget in budgets[i].items():
            seed_results[seed][budget_measure].append(budget)
        for col in cols:
            seed_results[seed][col].append(results[i]["result"]["attack"]["avg"][col])
    seed_dataframes = {}
    for seed, results in seed_results.items():
        df = pd.DataFrame(results)
        seed_dataframes[seed] = df
        df.to_csv(seed_dir / f"seed_{seed}.csv")
    return seed_dataframes


def save_plots(
    model,
    dataset,
    seed_dfs,
    results_path,
    attack_cols,
):
    plots_dir = results_path / "plots"
    seed_plot_dir = plots_dir / "seeds"
    seed_plot_dir.mkdir(parents=True)

    for seed, df in seed_dfs.items():
        cur_plot_dir = seed_plot_dir / f"seed_{seed}"
        cur_plot_dir.mkdir()

        for title, col_group in attack_cols.items():

            for budget_measure, budget_dict in budget_measures.items():

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
                ax.set_title(f"{model} - {dataset.replace('_', ' ')}")

                if "clean" in col_group:
                    c = col_group["clean"]
                    zb_val = df[c][0]
                else:
                    zb_val = 0.0

                for label, col in col_group.items():
                    if label == "clean":
                        continue  #  don't plot the clean (is equal to first step anyway)
                    x = np.concatenate(
                        (np.array([0.0]), np.array(df[budget_dict["keys"][label]]))
                    ) * budget_dict["mul"]
                    y = np.concatenate((np.array([zb_val]), np.array(df[col])))
                    ax.plot(x, y, label=label)

                ax.set_xlabel(budget_dict["label"])
                ax.set_ylabel(title)
                ax.legend()
                fig.savefig(cur_plot_dir / f"{dataset}_{model}_{title.replace(' ', '_')}_{budget_measure}.png")
                ax.clear()
                plt.close(fig)

    agg_plot_dir = plots_dir / "agg"
    agg_plot_dir.mkdir(parents=True)
    
    # aggregate seeds, mean and error std
    df_agg = pd.concat(list(seed_dfs.values()))
    df_agg = df_agg.drop("run_ids", axis=1)
    df_agg_mean = df_agg.groupby("budget", as_index=False).mean()
    df_agg_std = df_agg.groupby("budget", as_index=False).std()

    # plot aggregate
    for title, col_group in attack_cols.items():

        for budget_measure, budget_dict in budget_measures.items():

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
            ax.set_title(f"{model} - {dataset.replace('_', ' ')}")

            if "clean" in col_group:
                c = col_group["clean"]
                zb_val = df[c][0]
            else:
                zb_val = 0.0

            for label, col in col_group.items():
                if label == "clean":
                        continue  #  don't plot the clean (is equal to first step anyway)
                x = np.concatenate(
                    (np.array([0.0]), np.array(df_agg_mean[budget_dict["keys"][label]]))
                ) * budget_dict["mul"]
                y = np.concatenate((np.array([zb_val]), np.array(df_agg_mean[col])))
                std = np.concatenate((np.array([0.0]), np.array(df_agg_std[col])))
                ax.plot(x, y, label=label)
                ax.fill_between(x, y-std, y+std, alpha=0.2)

            ax.set_xlabel(budget_dict["label"])
            ax.set_ylabel(title)
            ax.legend()
            fig.savefig(agg_plot_dir / f"{dataset}_{model}_{title.replace(' ', '_')}_{budget_measure}.png")
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
    run_ids = [result["_id"] for result in results]
    extras = dict()
    for key in extra_fields:
        values = []
        keys_list = key.split(".")
        for r in results:
            for key_l in keys_list[:-1]:
                r = r.get(key_l, {})
            key_last = keys_list[-1]
            v = r.get(key_last, None)
            if v is not None:
                if key_last.endswith("bytes"):
                    v = f"{v * 1e-9:.1f} GB"
                if key_last.endswith("time"):
                    v = f"{v / 3600:.2f} hours"
            values.append(v)
        extras[key] = values
    seeds_graphgym = [result["config"]["graphgym"]["seed"] for result in results]
    seeds_seml = [result["config"]["seed"] for result in results]
    run_dirs = [result["result"].get("run_dir") for result in results]
    num_params = [result["result"].get("num_params") for result in results]
    budgets_allowed = [result["config"]["graphgym"]["attack"]["e_budget"] for result in results]
    avg_num_edges_modified = [r["result"]["attack"]["avg"]["avg_budget_used"] for r in results]
    avg_num_edges_modified_random = [r["result"]["attack"]["avg"]["avg_budget_used_random"] for r in results]
    avg_num_edges_clean = [r["result"]["attack"]["avg"]["avg_num_edges_clean"] for r in results]
    avg_budget_used = [m / c for m, c in zip(avg_num_edges_modified, avg_num_edges_clean)]
    avg_budget_used_random = [m / c for m, c in zip(avg_num_edges_modified_random, avg_num_edges_clean)]
    budgets = [
        {"budget": b, "bu": bu, "bur": bur, "m": m, "mr": mr} for b, bu, bur, m, mr in zip(
            budgets_allowed,
            avg_budget_used,
            avg_budget_used_random,
            avg_num_edges_modified,
            avg_num_edges_modified_random,
        )
    ]
    return results, run_ids, extras, seeds_graphgym, seeds_seml, run_dirs, num_params, budgets


def main(
    collection: str,
    results_path: str,
    filter_dict,
    dataset: str,
    model: str,
):
    (
        results,
        run_ids,
        extras,
        seeds_graphgym,
        seeds_seml,
        run_dirs,
        num_params,
        budgets,
    ) = get_collection_results(collection, filter_dict)

    for res in results:
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

    pred_level = results[0]["config"]["graphgym"]["attack"]["prediction_level"]
    for r in results:
        assert r["config"]["graphgym"]["attack"]["prediction_level"] == pred_level, (
            "Why are there different prediction levels in same collection?"
        )
    if pred_level == "graph":
        attack_cols = attack_cols_graph
    elif pred_level == "node":
        attack_cols = attack_cols_node
    else:
        raise ValueError(f"Unknown prediction level: `{pred_level}`")

    results_path, info_file, seed_dir = clean_path(results_path)
    # write results into file
    seed_dataframes = write_results(
        info_file,
        seed_dir,
        results,
        run_ids,
        extras,
        seeds_graphgym,
        seeds_seml,
        run_dirs,
        num_params,
        budgets,
        attack_cols
    )
    # plots
    save_plots(
        model,
        dataset,
        seed_dataframes,
        results_path,
        attack_cols,
    )


parser = argparse.ArgumentParser(description='Processes the results of attack.')
parser.add_argument("-c", "--collection")
parser.add_argument("-d", "--dataset")
parser.add_argument("-m", "--model")


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.dataset in datasets
    results_path = f"results_a/{args.dataset}/{args.model}/{args.collection}"
    # not implemented for argparse... but can manually change here
    filter_dict = None  # {"config.graphgym.attack.cluster_sampling": True}
    main(
        collection=args.collection,
        results_path=results_path,
        filter_dict=filter_dict,
        dataset=args.dataset,
        model=args.model,
    )
