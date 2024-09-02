import seml
import numpy as np
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import argparse
import shutil
from collections import defaultdict


use_tex = True
if use_tex:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })


def save_figure(fig, path, name, png=False):
    if png:
        file_name = name + ".png"
        fig.savefig(path / file_name, bbox_inches="tight", dpi=500)
        return
    file_name = name + ".pdf"
    fig.savefig(path / file_name, bbox_inches="tight")


datasets = {
    "CLUSTER_cs": {"format":"PyG-GNNBenchmarkDataset", "name": "CLUSTER"},
    "CLUSTER_as": {"format":"PyG-GNNBenchmarkDataset", "name": "CLUSTER"},
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
    "GPS": {"type": set(["GPSModel"]), "gnn_layer_type": None},
    "GPS-GCN": {"type": set(["GPSModel"]), "gnn_layer_type": None},
}


attack_cols_graph = {
    "Attack success rate": {
        "transfer": "avg_attack_success_rate", "random": "avg_attack_success_rate_random",
    },
    "Accuracy": {
        "clean": "avg_correct_clean", "transfer": "avg_correct_pert", "random": "avg_correct_pert_random",
    },
    "Margin": {
        "clean": "avg_margin_clean", "transfer": "avg_margin_pert", "random": "avg_margin_pert_random",
    },
}
attack_cols_node = {
    "Attack success rate": {
        "transfer": "avg_attack_success_rate", "random": "avg_attack_success_rate_random",
    },
    "Accuracy": {
        "clean": "avg_correct_acc_clean", "transfer": "avg_correct_acc_pert", "random": "avg_correct_acc_pert_random", 
    },
    "Margin": {
        "clean": "avg_margin_mean_clean", "transfer": "avg_margin_mean_pert", "random": "avg_margin_mean_pert_random",
    },
}


agg_cols_graph = {
    "strongest_asr.csv": (
        "max",
        ["avg_attack_success_rate", "avg_attack_success_rate_random"],
        0,
    ),
    "strongest_acc.csv": (
        "min",
        ["avg_correct_pert", "avg_correct_pert_random"],
        "avg_correct_clean",
    ),
    "strongest_margin.csv": (
        "min",
        ["avg_margin_pert", "avg_margin_pert_random"],
        "avg_margin_clean",
    ),
}
agg_cols_node = {
    "strongest_asr.csv": (
        "max",
        ["avg_attack_success_rate", "avg_attack_success_rate_random"],
        0,
    ),
    "strongest_acc.csv": (
        "min",
        ["avg_correct_acc_pert", "avg_correct_acc_pert_random"],
        "avg_correct_acc_clean",
    ),
    "strongest_margin.csv": (
        "min",
        ["avg_margin_mean_pert", "avg_margin_mean_pert_random"],
        "avg_margin_mean_clean",
    ),
}


budget_measures = {
    "budget": {
        "key": "budget",
        "label": r"Edge mod. budget (\%)",
        "mul": 100,
    },
    #"budget_used": {
    #    "key": "bu",
    #    "label": r"Avg. edges mod. (\%)",
    #    "mul": 100,
    #},
    #"num_modifications": {
    #    "key": "m",
    #    "label": r"Avg. edges mod.",
    #    "mul": 1,
    #},
}

styles = {
    "adaptive": {"color": ["b"], "linestyle": [":"], "marker": ["o"], "markersize": [6]},
    "random": {"color": ["g"], "linestyle": ["-."], "marker": ["*"], "markersize": [9]},
    "transfer": {
        "color": ["r", "k", "orange", "m", "aqua", "blue"],
        "linestyle": ["--", (0, (3, 5, 1, 5, 1, 5)), (0, (5, 10)), (0, (5, 1)), (0, (1, 1)), (0, (5, 10))],
        "marker": ["v", "s", "X", "p", "s", "s"],
        "markersize": [7, 6, 8, 8, 8, 8],
    },
}

figsize = (2.5, 2.0)
figsize_all = (4.5, 3.5)


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

    df_dir = seed_dir / "all"
    df_dir.mkdir()

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
            df.to_csv(df_dir / f"{run_name}_s{seed}.csv")

        run_seed_dataframes[run_name] = seed_dataframes

    return run_seed_dataframes


def save_plots(
    model,
    dataset,
    run_seed_dfs,
    results_path,
    attack_cols,
    max_idx_small_budget: int,
    agg_cols,
    seed_dir,
    y_label: bool,
    add_title: bool,
    add_legend: bool,
    y_min: float | None,
    y_max: float | None,
):
    plots_dir = results_path / "plots"
    plots_dir_runs = plots_dir / "individual"

    all_agg_results = {}
    all_agg_results_sb = {}

    all_dfs_mean = []
    all_dfs_std = []

    plot_individual = False

    for run_name, seed_dfs in run_seed_dfs.items():
        rand_run = run_name == "random"
        plots_dir_run = plots_dir_runs / run_name
        plots_dir_run.mkdir(parents=True)
        
        # aggregate seeds, mean and error std
        df_agg = pd.concat(list(seed_dfs.values()), ignore_index=True)
        df_agg_mean = df_agg.groupby("budget", as_index=False).mean()
        all_dfs_mean.append(df_agg_mean)
        df_agg_std = df_agg.groupby("budget", as_index=False).std()
        all_dfs_std.append(df_agg_std)

        # plot aggregate
        for title, col_group in attack_cols.items():

            for budget_measure, budget_dict in budget_measures.items():

                if plot_individual:
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
                    fig_sb, ax_sb = plt.subplots(nrows=1, ncols=1, figsize=figsize)
                    fig_title = f"{model} - {dataset.replace('_', ' ')}"
                    if add_title:
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

                label_occurrences = defaultdict(int)

                for label, col in col_group.items():
                    rand_label = label == "random"
                    if label == "clean":
                        continue  #  don't plot the clean (is equal to first step anyway)
                    if (rand_run and not rand_label) or (not rand_run and rand_label):
                        continue  # for random only plot the random col, for transfer dont plot rand col

                    l_count = label_occurrences[label]
                    c = styles[label]["color"][l_count]
                    l = styles[label]["linestyle"][l_count]
                    m = styles[label]["marker"][l_count]
                    ms = styles[label]["markersize"][l_count]
                    label_occurrences[label] += 1

                    x = np.concatenate((np.array([0.0]), np.array(df_agg_mean.index)))
                    x = np.concatenate(
                        (np.array([0.0]), np.array(df_agg_mean[budget_dict["key"]]))
                    ) * budget_dict["mul"]
                    y = np.concatenate((np.array([zb_val]), np.array(df_agg_mean[col])))
                    std = np.concatenate((np.array([0.0]), np.array(df_agg_std[col])))

                    if "margin" not in col:
                        y *= 100
                        std *= 100

                    if plot_individual:
                        ax.plot(x, y, label=label, alpha=0.7, color=c, marker=m, linestyle=l, markeredgewidth=0.0, markersize=ms)
                        ax.fill_between(x, y-std, y+std, color=c, alpha=0.1, linewidth=0.0)

                    x_sb = x[:max_idx_small_budget+1]
                    y_sb = y[:max_idx_small_budget+1]
                    std_sb = std[:max_idx_small_budget+1]

                    if plot_individual:
                        ax_sb.plot(x_sb, y_sb, label=label, alpha=0.7, color=c, marker=m, linestyle=l, markeredgewidth=0.0, markersize=ms)
                        ax.fill_between(x_sb, y_sb-std_sb, y_sb+std_sb, color=c, alpha=0.1, linewidth=0.0)

                    all_agg_results[title][budget_measure][run_name][col] = {"x": x, "y": y, "std": std}
                    all_agg_results_sb[title][budget_measure][run_name][col] = {"x": x_sb, "y": y_sb, "std": std_sb}

                if plot_individual:
                    ax.set_xlabel(budget_dict["label"])
                    ax_sb.set_xlabel(budget_dict["label"])
                    if y_label:
                        ax.set_ylabel(title)
                        ax_sb.set_ylabel(title)
                    ax.legend()
                    ax_sb.legend()
                    filename = f"{dataset}_{model}_{title.replace(' ', '_')}_{budget_measure}"
                    save_figure(fig, plots_dir_run, filename)
                    #fig.savefig(plots_dir_run / f"{filename}.png")
                    save_figure(fig_sb, plots_dir_run, f"{filename}_sb")
                    #fig_sb.savefig(plots_dir_run / f"{filename}_sb.png")
                    ax.clear()
                    ax_sb.clear()
                    plt.close(fig)
                    plt.close(fig_sb)

    # save the worst case attack to a file
    for filename, (agg_mode, cols, clean_col) in agg_cols.items():
        df_all_mean = pd.concat(all_dfs_mean, ignore_index=True)
        df_all_std = pd.concat(all_dfs_std, ignore_index=True)
        df_agg_mean_all = df_all_mean[["budget"] + cols].fillna(0.0)
        df_agg_std_all = df_all_std[["budget"] + cols].fillna(0.0)

        # assert that the colomns are complementary
        num_nonzero_mean_cols = df_agg_mean_all[cols].astype(bool).sum(axis=1)
        assert np.all(num_nonzero_mean_cols <= 1)
        df_agg_mean_all["stat"] = df_agg_mean_all[cols].sum(1)

        num_nonzero_std_cols = df_agg_std_all[cols].astype(bool).sum(axis=1)
        assert np.all(num_nonzero_std_cols <= 1)
        df_agg_std_all["stat"] = df_agg_std_all[cols].sum(1)

        df_mean_gb = df_agg_mean_all.groupby("budget")
        
        if agg_mode == "max":
            strongest_idx = df_mean_gb["stat"].idxmax().values
        else:
            strongest_idx = df_mean_gb["stat"].idxmin().values

        budgets = df_agg_mean_all["budget"].iloc[strongest_idx]
        strongest_mean = df_agg_mean_all["stat"].iloc[strongest_idx]
        strongest_std = df_agg_std_all["stat"].iloc[strongest_idx]

        budgets = np.concatenate([np.array([0]), budgets], axis=0)
        if not clean_col:
            clean_mean_val = 0.0
            clean_std_val = 0.0
        else:
            # the clean of the strongest first budget (should all be the same anyway) 
            clean_mean_val = df_all_mean[clean_col].iloc[strongest_idx[0]]
            clean_std_val = df_all_std[clean_col].iloc[strongest_idx[0]]
        strongest_mean = np.concatenate([np.array([clean_mean_val]), strongest_mean], axis=0)
        strongest_std = np.concatenate([np.array([clean_std_val]), strongest_std], axis=0)


        data = {"budget": budgets, "mean": strongest_mean, "std": strongest_std}
        df_strongest = pd.DataFrame(data)
        df_strongest.to_csv(seed_dir / filename)


    plotdir_all = plots_dir / "agg_all_budgets"
    plotdir_all.mkdir()
    plotdir_small = plots_dir / "agg_small_budgets"
    plotdir_small.mkdir()
    plotdir_bt = plots_dir / "agg_best_transfer"
    plotdir_bt.mkdir()

    for agg_res, pdir, bt in zip(
        [all_agg_results, all_agg_results_sb, all_agg_results_sb],
        [plotdir_all, plotdir_small, plotdir_bt],
        [False, False, True],
    ):

        for title, run_stats_b in agg_res.items():

            for budget_measure, budget_dict in budget_measures.items():

                if bt and (budget_measure != "budget" or "rate" in title):
                    # only get the best transfer for:
                    #  - allowed budget, where the x values should to align
                    #  - acc and margin where we find the min (to add asr, handle min/max)
                    continue

                x_best = None
                y_best = None
                std_best = None

                label_occurrences = defaultdict(int)
                run_stats = run_stats_b[budget_measure]
                if bt:
                    fs_ = figsize
                else:
                    fs_ = figsize_all
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
                if add_title:
                    ax.set_title(f"{model} - {dataset.replace('_', ' ')}")
                for run_name, agg_res_run in run_stats.items():
                    # not really a loop, should only have one entry...
                    assert len(agg_res_run) == 1
                    for col, res in agg_res_run.items():

                        if run_name == model:
                            label = "adaptive"
                            l = label
                        elif run_name == "random":
                            label = "random"
                            l = label
                        else:
                            if bt:
                                # save best
                                if x_best is None:
                                    x_best = res["x"]
                                    y_best = res["y"]
                                    std_best = res["std"]
                                else:
                                    x_new = res["x"].copy()
                                    y_new = res["y"].copy()
                                    std_new = res["std"].copy()
                                    n_n = x_new.shape[0]
                                    n_b = x_best.shape[0]
                                    if n_b > n_n:
                                        assert np.all(x_best[:n_n] == x_new)
                                        new_best = y_new < y_best[:n_n]
                                        y_best[:n_n][new_best] = y_new[new_best]
                                        std_best[:n_n][new_best] = std_new[new_best]
                                    elif n_n > n_b:
                                        assert np.all(x_best == x_new[:n_b])
                                        old_best = y_new[:n_b] > y_best
                                        y_new[:n_b][old_best] = y_best[old_best]
                                        std_new[:n_b][old_best] = std_best[old_best]
                                        x_best = x_new
                                        y_best = y_new
                                        std_best = std_new
                                    else:
                                        new_best = y_new < y_best
                                        y_best[new_best] = y_new[new_best]
                                        std_best[new_best] = std_new[new_best]
                                label = "transfer"
                                l = label
                                continue
                            else:
                                l = "transfer"
                                label = f"transfer from {run_name}"

                        l_count = label_occurrences[l]
                        c = styles[l]["color"][l_count]
                        ls = styles[l]["linestyle"][l_count]
                        m = styles[l]["marker"][l_count]
                        ms = styles[l]["markersize"][l_count]
                        label_occurrences[l] += 1

                        x = res["x"]
                        y = res["y"]
                        std = res["std"]

                        ax.plot(x, y, label=label, alpha=0.7, color=c, marker=m, linestyle=ls, markeredgewidth=0.0, markersize=ms)
                        ax.fill_between(x, y-std, y+std, color=c, alpha=0.1, linewidth=0.0)

                if bt:
                    l = "transfer"
                    c = styles[l]["color"][0]
                    ls = styles[l]["linestyle"][0]
                    m = styles[l]["marker"][0]
                    ms = styles[l]["markersize"][0]
                    label_occurrences[l] += 1
                    ax.plot(x_best, y_best, label=l, alpha=0.7, color=c, marker=m, linestyle=ls, markeredgewidth=0.0, markersize=ms)
                    ax.fill_between(x_best, y_best-std_best, y_best+std_best, color=c, alpha=0.1, linewidth=0.0)

                ax.set_xlabel(budget_dict["label"])
                if y_label:
                    if "margin" not in title:
                        ax.set_ylabel(title + r" (\%)")
                    else:
                        ax.set_ylabel(title)
                if add_legend:
                    ax.legend(fontsize="small", framealpha=0.5)  # bbox_to_anchor=(1.01, 0), loc="lower left")
                save_figure(fig, pdir, f"{dataset}_{model}_{title.replace(' ', '_')}_{budget_measure}")
                #fig.savefig(pdir / f"{dataset}_{model}_{title.replace(' ', '_')}_{budget_measure}.png")
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


def check_correct_result(result, dataset: str, model: str, pred_level: str):
    df = result["config"]["graphgym"]["dataset"]["format"]
    dfg = datasets[dataset]["format"]
    dn = result["config"]["graphgym"]["dataset"]["name"]
    dng = datasets[dataset]["name"]
    assert df == dfg, (f"Dataset format was given to be `{dfg}`, but encountered `{df}`.")
    assert dn == dng, (f"Dataset name was given to be `{dng}`, but encountered `{dn}`.")
    if dataset.startswith("CLUSTER"):
        constrained_attack = dataset.endswith("cs")
        if constrained_attack:
            assert result["config"]["graphgym"]["attack"]["cluster_sampling"]
        else:
            assert not result["config"]["graphgym"]["attack"]["cluster_sampling"]
    
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
            if (m == "GPS" or m == "GCN") and "GPS-GCN" in tmp:
                transfer_model = "GPS-GCN"
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
    y_label: bool,
    add_title: bool,
    add_legend: bool,
    y_min: float | None,
    y_max: float | None,
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
        agg_cols = agg_cols_graph
    elif pred_level == "node":
        attack_cols = attack_cols_node
        agg_cols = agg_cols_node
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
        agg_cols,
        seed_dir,
        y_label,
        add_title,
        add_legend,
        y_min,
        y_max,
    )


parser = argparse.ArgumentParser(description='Processes the results of transfer attack.')
parser.add_argument("-c", "--collection")
parser.add_argument("-d", "--dataset")
parser.add_argument("-m", "--model")
parser.add_argument("-s", "--small-budget-idx")
parser.add_argument("-a", "--attack-collection")
parser.add_argument("-y", "--y-label", action="store_true")
parser.add_argument("-t", "--title", action="store_true")
parser.add_argument("-l", "--legend", action="store_true")
parser.add_argument("-b", "--best-transfer-only", action="store_true")
parser.add_argument("--y-min", type=float, default=None)
parser.add_argument("--y-max", type=float, default=None)


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
        y_label=args.y_label,
        add_title=args.title,
        add_legend=args.legend,
        y_min=args.y_min,
        y_max=args.y_max,
    )
