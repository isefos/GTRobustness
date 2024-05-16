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


attack_cols_graph = {
    "Average attack success rate": {
        "transfer": "avg_attack_success_rate", "random": "avg_attack_success_rate_random",
    },
    "Average accuracy": {
        "clean": "avg_correct_clean", "transfer": "avg_correct_pert", "random": "avg_correct_pert_random",
    },
    "Average margin": {
        "clean": "avg_margin_clean", "transfer": "avg_margin_pert", "random": "avg_margin_pert_random",
    },
}
attack_cols_node = {
    "Average attack success rate": {
        "transfer": "avg_attack_success_rate", "random": "avg_attack_success_rate_random",
    },
    "Average accuracy": {
        "clean": "avg_correct_acc_clean", "transfer": "avg_correct_acc_pert", "random": "avg_correct_acc_pert_random", 
    },
    "Average mean margin": {
        "clean": "avg_margin_mean_clean", "transfer": "avg_margin_mean_pert", "random": "avg_margin_mean_pert_random",
    },
}


budget_measures = {
    "budget": {
        "key": "budget",
        "label": r"Budget (% edge modifications allowed)",
        "mul": 100,
    },
    "budget_used": {
        "key": "bu",
        "label": r"Average edges modified (%)",
        "mul": 100,
    },
    "num_modifications": {
        "key": "m",
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


def write_info_file(info_file, run_ids, num_params, extras, run_dirs):
    with open(info_file, "w") as f:
        f.write("Attack run infos:")
        for i, run_id in enumerate(run_ids):
            f.write(f"\nrun: {run_id}")
            f.write(f"\n\tnum_params: {num_params[i]}")
            for k, v in extras.items():
                f.write(f"\n\t{k}: {v[i]}")
            f.write(f"\n\trun_dir: {run_dirs[i]}")
            f.write("\n")


def get_transfer_result_budgets(result):
    budgets_allowed = result["result"]["attack"]["budgets"]
    avg_num_edges_modified = [
        r.get("avg_num_edges_added_connected", r["avg_num_edges_added"]) + r["avg_num_edges_removed"]
        for r in result["result"]["attack"]["avg"]
    ]
    avg_num_edges_clean = [
        r["avg_num_edges_clean"]
        for r in result["result"]["attack"]["avg"]
    ]
    avg_budget_used = [m / c for m, c in zip(avg_num_edges_modified, avg_num_edges_clean)]
    budgets = [
        {"budget": b, "bu": bu, "m": m} for b, bu, m in zip(
            budgets_allowed,
            avg_budget_used,
            avg_num_edges_modified,
        )
    ]
    return budgets


def write_results(
    seed_dir,
    results,
    attack_cols,
):
    cols = []
    for col_group in attack_cols.values():
        cols.extend(list(col_group.values()))

    run_seed_dataframes = {}

    for result in results:
        run_name = result["transfer_model"]
        rand_run = run_name == "random"
        seeds = result["result"]["attack"]["seeds"]
        if rand_run:
            budgets = result["result"]["attack"]["budgets"]
        else:
            budgets = get_transfer_result_budgets(result)

        seed_results = {}
        for seed, budget, avg_results in zip(seeds, budgets, result["result"]["attack"]["avg"]):
            if seed not in seed_results:
                seed_results[seed] = defaultdict(list)
            for budget_measure, budget in budget.items():
                seed_results[seed][budget_measure].append(budget)
            for col in cols:
                rand_col = col.endswith("_random")
                clean_col = col.endswith("_clean") 
                if (rand_run and (rand_col or clean_col)) or not (rand_col or rand_run):
                    seed_results[seed][col].append(avg_results[col])

        seed_dataframes = {}
        for seed, r in seed_results.items():
            df = pd.DataFrame(r)
            seed_dataframes[seed] = df
            df.to_csv(seed_dir / f"{run_name}_s{seed}.csv")

        run_seed_dataframes[run_name] = seed_dataframes

    return run_seed_dataframes


def save_plots(
    model,
    dataset,
    run_seed_dfs,
    results_path,
    attack_cols,
    max_idx_small_budget: int,
):
    plots_dir = results_path / "plots"
    plots_dir_runs = plots_dir / "individual"

    all_agg_results = {}
    all_agg_results_sb = {}

    for run_name, seed_dfs in run_seed_dfs.items():
        rand_run = run_name == "random"
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
                fig_sb, ax_sb = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
                fig_title = f"{model} - {dataset.replace('_', ' ')}"
                ax.set_title(fig_title)
                ax_sb.set_title(fig_title)

                if "clean" in col_group:
                    zb_val = df_agg_mean[col_group["clean"]][0]
                else:
                    zb_val = 0.0

                for agg_res in [all_agg_results, all_agg_results_sb]:
                    if title not in agg_res:
                        agg_res[title] = {budget_measure: {run_name: {}}}
                    elif budget_measure not in agg_res[title]:
                        agg_res[title][budget_measure] = {run_name: {}}
                    else:
                        agg_res[title][budget_measure][run_name] = {}

                for label, col in col_group.items():
                    rand_label = label == "random"
                    if label == "clean":
                        continue  #  don't plot the clean (is equal to first step anyway)
                    if (rand_run and not rand_label) or (not rand_run and rand_label):
                        continue  # for random only plot the random col, for transfer dont plot rand col
                    x = np.concatenate((np.array([0.0]), np.array(df_agg_mean.index)))
                    x = np.concatenate(
                        (np.array([0.0]), np.array(df_agg_mean[budget_dict["key"]]))
                    ) * budget_dict["mul"]
                    y = np.concatenate((np.array([zb_val]), np.array(df_agg_mean[col])))
                    std = np.concatenate((np.array([0.0]), np.array(df_agg_std[col])))
                    ax.plot(x, y, label=label)
                    ax.fill_between(x, y-std, y+std, alpha=0.2)

                    x_sb = x[:max_idx_small_budget+1]
                    y_sb = y[:max_idx_small_budget+1]
                    std_sb = std[:max_idx_small_budget+1]
                    ax_sb.plot(x_sb, y_sb, label=label)
                    ax.fill_between(x_sb, y_sb-std_sb, y_sb+std_sb, alpha=0.2)

                    all_agg_results[title][budget_measure][run_name][col] = {"x": x, "y": y, "std": std}
                    all_agg_results_sb[title][budget_measure][run_name][col] = {"x": x_sb, "y": y_sb, "std": std_sb}

                ax.set_xlabel(budget_dict["label"])
                ax_sb.set_xlabel(budget_dict["label"])
                ax.set_ylabel(title)
                ax_sb.set_ylabel(title)
                ax.legend()
                ax_sb.legend()
                filename = f"{dataset}_{model}_{title.replace(' ', '_')}_{budget_measure}"
                fig.savefig(plots_dir_run / f"{filename}.png")
                fig_sb.savefig(plots_dir_run / f"{filename}_sb.png")
                ax.clear()
                ax_sb.clear()
                plt.close(fig)
                plt.close(fig_sb)

    plotdir_all = plots_dir / "agg_all_budgets"
    plotdir_all.mkdir()
    plotdir_small = plots_dir / "agg_small_budgets"
    plotdir_small.mkdir()

    for agg_res, pdir in zip([all_agg_results, all_agg_results_sb], [plotdir_all, plotdir_small]):

        for title, run_stats_b in agg_res.items():

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
                        elif run_name == "random":
                            label = "random"
                        else:
                            label = f"transfer from {run_name}"
                        ax.plot(x, y, label=label)
                        ax.fill_between(x, y-std, y+std, alpha=0.2)
                ax.set_xlabel(budget_dict["label"])
                ax.set_ylabel(title)
                ax.legend()
                fig.savefig(pdir / f"{dataset}_{model}_{title.replace(' ', '_')}_{budget_measure}.png")
                ax.clear()
                plt.close(fig)


def get_transfer_collection_results(collection, filter_dict):
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
    run_dirs = [result["result"].get("run_dir") for result in results]
    num_params = [result["result"].get("num_params") for result in results]
    return results, run_ids, extras, run_dirs, num_params


def get_random_attack_collection_results(collection, dataset, model, pred_level):
    results = seml.get_results(collection, ['config', 'result.attack.avg'])
    for r in results:
        check_correct_result(r, dataset, model, pred_level)
    a_results = [r["result"]["attack"]["avg"] for r in results]
    seeds_graphgym = [result["config"]["graphgym"]["seed"] for result in results]
    budgets_allowed = [result["config"]["graphgym"]["attack"]["e_budget"] for result in results]
    avg_num_edges_modified_random = [r["avg_budget_used_random"] for r in a_results]
    avg_num_edges_clean = [r["avg_num_edges_clean"] for r in a_results]
    avg_budget_used_random = [m / c for m, c in zip(avg_num_edges_modified_random, avg_num_edges_clean)]
    budgets = [
        {"budget": b, "bu": bur, "m": mr} for b, bur, mr in zip(
            budgets_allowed,
            avg_budget_used_random,
            avg_num_edges_modified_random,
        )
    ]
    random_transfer_result = {
        "transfer_model": "random",
        "result": {
            "attack": {
                "seeds": seeds_graphgym,
                "budgets": budgets,
                "avg": a_results,
            },
        },
    }
    return random_transfer_result


def check_correct_result(result, dataset, model, pred_level):
    df = result["config"]["graphgym"]["dataset"]["format"]
    dfg = datasets[dataset]["format"]
    dn = result["config"]["graphgym"]["dataset"]["name"]
    dng = datasets[dataset]["name"]
    assert df == dfg, (f"Dataset format was given to be `{dfg}`, but encountered `{df}`.")
    assert dn == dng, (f"Dataset name was given to be `{dng}`, but encountered `{dn}`.")
    
    mt = result["config"]["graphgym"]["model"]["type"]
    mtg = models[model]["type"]
    assert mt in mtg, (f"Model was given to be in {mtg}, but encountered `{mt}`.")
    mlg = models[model]["gnn_layer_type"]
    if mlg is not None:
        ml = result["config"]["graphgym"]["gnn"]["layer_type"]
        assert ml in mlg, (f"Model layer was given to be in {mlg}, but encountered `{ml}`.")
    
    assert result["config"]["graphgym"]["attack"]["prediction_level"] == pred_level, (
        "Why are there different prediction levels in same collection?"
    )


def get_transfer_model(result):
    tmp = result["config"]["graphgym"]["attack"]["transfer"]["perturbation_path"]
    transfer_model = None
    for m in models.keys():
        if m in tmp:
            if m == "GAT" and "GATv2" in tmp:
                transfer_model = "GATv2"
                break
            transfer_model = m
            break
    assert transfer_model is not None, "Could not determine the transfer model."
    return transfer_model


def main(
    collection: str,
    results_path: str,
    filter_dict,
    dataset: str,
    model: str,
    max_idx_small_budget: int,
    attack_collection: str,
):
    (
        results,
        run_ids,
        extras,
        run_dirs,
        num_params,
    ) = get_transfer_collection_results(collection, filter_dict)
    pred_level = results[0]["config"]["graphgym"]["attack"]["prediction_level"]
    if pred_level == "graph":
        attack_cols = attack_cols_graph
    elif pred_level == "node":
        attack_cols = attack_cols_node
    else:
        raise ValueError(f"Unknown prediction level: `{pred_level}`")
    for res in results:
        check_correct_result(res, dataset, model, pred_level)
        res["transfer_model"] = get_transfer_model(res)
    results_path, info_file, seed_dir = clean_path(results_path)
    write_info_file(info_file, run_ids, num_params, extras, run_dirs)

    rand_result = get_random_attack_collection_results(attack_collection, dataset, model, pred_level)
    results.append(rand_result)

    run_seed_dataframes = write_results(
        seed_dir,
        results,
        attack_cols,
    )
    # plots
    save_plots(
        model,
        dataset,
        run_seed_dataframes,
        results_path,
        attack_cols,
        max_idx_small_budget,
    )


parser = argparse.ArgumentParser(description='Processes the results of transfer attack.')
parser.add_argument("-c", "--collection")
parser.add_argument("-d", "--dataset")
parser.add_argument("-m", "--model")
parser.add_argument("-s", "--small-budget-idx")
parser.add_argument("-a", "--attack-collection")


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.dataset in datasets
    results_path = f"results_t/{args.dataset}/{args.model}/{args.collection}"
    # not implemented for argparse... but can manually change here
    filter_dict = None  # {"config.graphgym.attack.cluster_sampling": True}
    main(
        collection=args.collection,
        results_path=results_path,
        filter_dict=filter_dict,
        dataset=args.dataset,
        model=args.model,
        max_idx_small_budget=int(args.small_budget_idx),
        attack_collection=args.attack_collection,
    )
