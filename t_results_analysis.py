import seml
import numpy as np
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.use('Agg')
from matplotlib.ticker import AutoMinorLocator
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import argparse
import shutil
from collections import defaultdict
from plot_results import syles as final_styles


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
    "reddit_threads": {"format": "PyG-TUDataset", "name": "reddit_threads"},
    "ENZYMES": {"format": "PyG-TUDataset", "name": "ENZYMES"},
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
    "Polynormer": {"type": set(["WeightedPolynormer"]), "gnn_layer_type": None},
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
        #"label": r"Edge modification budget (\%)",
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

styles = {
    "adaptive": {"color": "#1b7837", "linestyle": (0, (1, 1)), "marker": "o", "markersize": 6},
    "random": {"color": "#9970ab", "linestyle": (0, (1, 1)), "marker": "*", "markersize": 9},
    "random_pert": {"color": "gray", "linestyle": (1, (3, 4, 1, 5)), "marker": (7, 1, 0), "markersize": 7},
    "transfer": final_styles
}


def clean_path(results_path: str, names: list[str]) -> tuple[dict, dict, dict]:
    results_path = Path(results_path)
    if results_path.exists():
        shutil.rmtree(results_path)
    results_paths, general_info_files, seed_dirs = {}, {}, {}
    for name in names:
        results_path_num = results_path / name
        results_path_num.mkdir(parents=True)
        results_paths[name] = results_path_num
        general_info_file = results_path_num / "runs_infos.txt"
        general_info_files[name] = general_info_file
        seed_dir = results_path_num / "results"
        seed_dir.mkdir()
        seed_dirs[name] = seed_dir
    return results_paths, general_info_files, seed_dirs


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
        rand_run = run_name.startswith("random")
        seeds = result["result"]["attack"]["seeds"]
        if rand_run or run_name == "adaptive":
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
    y_min_margin,
    y_max_margin,
    figsize,
    figsize_all,
):
    plots_dir = results_path / "plots"
    plots_dir_runs = plots_dir / "individual"

    all_agg_results = {}
    all_agg_results_sb = {}

    all_dfs_mean = []
    all_dfs_std = []

    plot_individual = False

    for run_name, seed_dfs in run_seed_dfs.items():
        rand_run = run_name.startswith("random")
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

                for label, col in col_group.items():
                    rand_label = label == "random"
                    if label == "clean":
                        continue  #  don't plot the clean (is equal to first step anyway)
                    if (rand_run and not rand_label) or (not rand_run and rand_label):
                        continue  # for random only plot the random col, for transfer dont plot rand col

                    k = label if run_name != "adaptive" else "adaptive"
                    c = styles[k]["color"]
                    l = styles[k]["linestyle"]
                    m = styles[k]["marker"]
                    ms = styles[k]["markersize"]
                    if k == "transfer":
                        idx = model_order[run_name]
                        c = c[idx]
                        l = l[idx]
                        m = m[idx]
                        ms = ms[idx]

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
                        ax.plot(x, y, label=label, alpha=0.9, color=c, marker=m, linestyle=l, markeredgewidth=0.0, markersize=ms)
                        ax.fill_between(x, y-std, y+std, color=c, alpha=0.1, linewidth=0.0)

                    x_sb = x[:max_idx_small_budget+1]
                    y_sb = y[:max_idx_small_budget+1]
                    std_sb = std[:max_idx_small_budget+1]

                    if plot_individual:
                        ax_sb.plot(x_sb, y_sb, label=label, alpha=0.9, color=c, marker=m, linestyle=l, markeredgewidth=0.0, markersize=ms)
                        ax.fill_between(x_sb, y_sb-std_sb, y_sb+std_sb, color=c, alpha=0.1, linewidth=0.0)

                    all_agg_results[title][budget_measure][run_name][col] = {"x": x, "y": y, "std": std}
                    all_agg_results_sb[title][budget_measure][run_name][col] = {"x": x_sb, "y": y_sb, "std": std_sb}

                if plot_individual:
                    ax.set_xlabel(budget_dict["label"])
                    ax_sb.set_xlabel(budget_dict["label"])
                    if y_label:
                        ax.set_ylabel(title)
                        ax_sb.set_ylabel(title)
                    if "argin" not in title:
                        ax.set_ylim(bottom=y_min, top=y_max)
                        ax_sb.set_ylim(bottom=y_min, top=y_max)
                    else:
                        ax.set_ylim(bottom=y_min_margin, top=y_max_margin)
                        ax_sb.set_ylim(bottom=y_min_margin, top=y_max_margin)
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


    plotdir_all = plots_dir / "agg_ab"
    plotdir_all.mkdir()
    plotdir_all_bt = plots_dir / "agg_ab_bt"
    plotdir_all_bt.mkdir()
    plotdir_small = plots_dir / "agg_sb"
    plotdir_small.mkdir()
    plotdir_small_bt = plots_dir / "agg_sb_bt"
    plotdir_small_bt.mkdir()

    for agg_res, pdir, bt, sb in zip(
        [all_agg_results, all_agg_results, all_agg_results_sb, all_agg_results_sb],
        [plotdir_all, plotdir_all_bt, plotdir_small, plotdir_small_bt],
        [False, True, False, True],
        [False, False, True, True],
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

                run_stats = run_stats_b[budget_measure]
                if bt:
                    fs_ = figsize
                else:
                    fs_ = figsize_all
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fs_)
                if add_title:
                    ax.set_title(f"{model} - {dataset.replace('_', ' ')}")
                res_rand_adap = {}
                for run_name, agg_res_run in run_stats.items():
                    # not really a loop, should only have one entry...
                    assert len(agg_res_run) == 1
                    for col, res in agg_res_run.items():

                        if run_name == "adaptive":
                            #label = "adaptive (PRBCD, ours)"
                            label = "adaptive"
                            l = "adaptive"
                        elif run_name == "random":
                            label = "random attack"
                            #label = "rand. attack"
                            l = "random"
                        elif run_name == "random_pert":
                            label = "random perturbation"
                            #label = "rand. pert."
                            l = "random_pert"
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
                                if run_name != "GCN":
                                    continue
                            l = "transfer"
                            #label = f"{run_name} PRBCD transfer"
                            label = f"{run_name} transfer"
                            #label = f"tr. {run_name}"
                            
                        c = styles[l]["color"]
                        ls = styles[l]["linestyle"]
                        m = styles[l]["marker"]
                        ms = styles[l]["markersize"]
                        if l == "transfer":
                            idx = model_order[run_name]
                            c = c[idx]
                            ls = ls[idx]
                            m = m[idx]
                            ms = ms[idx]

                        x = res["x"]
                        y = res["y"]
                        std = res["std"]

                        if run_name in ["adaptive", "random", "random_pert"]:
                            res_rand_adap[run_name] = {"x": x, "y": y, "std": std, "c": c, "m": m, "ls": ls, "ms": ms, "label": label}
                        else:
                            ax.plot(x, y, label=label, alpha=0.8, color=c, marker=m, linestyle=ls, markeredgewidth=0.0, markersize=ms)
                            ax.fill_between(x, y-std, y+std, color=c, alpha=0.1, linewidth=0.0)

                if bt:
                    l = "transfer"
                    if model == "GCN":
                        idx = model_order["GCN"]
                        c = styles[l]["color"][idx]
                        ls = styles[l]["linestyle"][idx]
                        m = styles[l]["marker"][idx]
                        ms = styles[l]["markersize"][idx]
                        ax.plot([], [], label="transfer GCN (PRBCD)", alpha=0.8, color=c, marker=m, linestyle=ls, markeredgewidth=0.0, markersize=ms)
                    #label = "best transfer (incl. ours)"
                    label = "best transfer"
                    c = styles[l]["color"][2]
                    ls = styles[l]["linestyle"][0]
                    m = styles[l]["marker"][0]
                    ms = styles[l]["markersize"][0]
                    ax.plot(x_best, y_best, label=label, alpha=0.8, color=c, marker=m, linestyle=ls, markeredgewidth=0.0, markersize=ms)
                    ax.fill_between(x_best, y_best-std_best, y_best+std_best, color=c, alpha=0.1, linewidth=0.0)

                for run_name, d in res_rand_adap.items():
                    if "rand" in run_name:
                        continue
                    # plot after all others, so they are on top
                    ax.plot(d["x"], d["y"], label=d["label"], alpha=0.8, color=d["c"], marker=d["m"], linestyle=d["ls"], markeredgewidth=0.0, markersize=d["ms"])
                    ax.fill_between(d["x"], d["y"]-d["std"], d["y"]+d["std"], color=d["c"], alpha=0.1, linewidth=0.0)


                ax.set_xlabel(budget_dict["label"])
                if sb:
                    x_lim = res_rand_adap["adaptive"]["x"][max_idx_small_budget-1]
                    ax.set_xlim(left=-0.05*x_lim, right=1.1*x_lim)
                if "argin" not in title:
                    ax.set_ylim(bottom=y_min, top=y_max)
                else:
                    ax.set_ylim(bottom=y_min_margin, top=y_max_margin)
                if y_label:
                    if "argin" not in title:
                        ax.set_ylabel(title + r" (\%)")
                    else:
                        ax.set_ylabel(title)
                if add_legend:
                    #ax.legend(fontsize="small", framealpha=0.4)  # bbox_to_anchor=(1.01, 0), loc="lower left")
                    #ax.legend(bbox_to_anchor=(1.01, 0), loc="lower left")  # , prop={'size': 8})
                    ax.legend(framealpha=0.4)
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


def get_attack_collection_results(collection, dataset, model, pred_level):
    all_results = seml.get_results(collection, ['config', 'result.attack.avg'])
    for r in all_results:
        check_correct_result(r, dataset, model, pred_level)

    # separate by the pretrained model
    per_pretrained = defaultdict(list)
    for r in all_results:
        pretrained = r["config"]["graphgym"]["pretrained"]["dir"]
        name = pretrained.split("/")[-1]
        per_pretrained[name].append(r)

    rand_results_per_pretrained = {}
    adaptive_results_per_pretrained = {}
    for name, results in per_pretrained.items():
        a_results = [r["result"]["attack"]["avg"] for r in results]
        seeds_graphgym = [result["config"]["graphgym"]["seed"] for result in results]
        budgets_allowed = [result["config"]["graphgym"]["attack"]["e_budget"] for result in results]
        avg_num_edges_modified_pert = [r["avg_budget_used"] for r in a_results]
        avg_num_edges_modified_random = [r["avg_budget_used_random"] for r in a_results]
        avg_num_edges_clean = [r["avg_num_edges_clean"] for r in a_results]
        for r, avg_edges_modified, transfer_model in zip(
            [adaptive_results_per_pretrained, rand_results_per_pretrained],
            [avg_num_edges_modified_pert, avg_num_edges_modified_random],
            ["adaptive", "random"]
        ):
            bs = [
                {"budget": b, "bu": bur, "m": mr} for b, bur, mr in zip(
                    budgets_allowed,
                    [m / c for m, c in zip(avg_edges_modified, avg_num_edges_clean)],
                    avg_edges_modified,
                )
            ]
            r[name] = {
                "transfer_model": transfer_model,
                "result": {
                    "attack": {
                        "seeds": seeds_graphgym,
                        "budgets": bs,
                        "avg": a_results,
                    },
                },
            }
    return adaptive_results_per_pretrained, rand_results_per_pretrained


def get_rand_pert_collection_results(collection, dataset, model, pred_level):
    all_results = seml.get_results(collection, ['config', 'result.attack.avg'])
    for r in all_results:
        check_correct_result(r, dataset, model, pred_level)

    # separate by the pretrained model
    per_pretrained = defaultdict(list)
    for r in all_results:
        pretrained = r["config"]["graphgym"]["pretrained"]["dir"]
        name = pretrained.split("/")[-1]
        per_pretrained[name].append(r)

    rand_pert_results_per_pretrained = {}
    for name, results in per_pretrained.items():
        a_results = [r["result"]["attack"]["avg"] for r in results]
        seeds_graphgym = [result["config"]["graphgym"]["seed"] for result in results]
        budgets_allowed = [result["config"]["graphgym"]["attack"]["e_budget"] for result in results]
        avg_num_edges_modified_random = [r["avg_budget_used_random"] for r in a_results]
        bs = [
            {"budget": b, "m": mr} for b, mr in zip(
                budgets_allowed,
                avg_num_edges_modified_random,
            )
        ]
        rand_pert_results_per_pretrained[name] = {
            "transfer_model": "random_pert",
            "result": {
                "attack": {
                    "seeds": seeds_graphgym,
                    "budgets": bs,
                    "avg": a_results,
                },
            },
        }
    return rand_pert_results_per_pretrained


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
    rand_pert_collection: str,
    y_label: bool,
    grid: bool,
    add_title: bool,
    add_legend: bool,
    y_min: float | None,
    y_max: float | None,
    y_min_margin: float | None,
    y_max_margin: float | None,
    fs_w: float,
    fs_h: float,
    fs_w_all: float,
    fs_h_all: float,
):
    figsize = (fs_w, fs_h)
    figsize_all = (fs_w_all, fs_h_all)
    (
        all_results,
        all_run_ids,
        extras,
        all_run_dirs,
        all_num_params,
    ) = get_transfer_collection_results(collection, filter_dict)
    pred_level = all_results[0]["config"]["graphgym"]["attack"]["prediction_level"]
    if pred_level == "graph":
        attack_cols = attack_cols_graph
        agg_cols = agg_cols_graph
    elif pred_level == "node":
        attack_cols = attack_cols_node
        agg_cols = agg_cols_node
    else:
        raise ValueError(f"Unknown prediction level: `{pred_level}`")
    for res in all_results:
        check_correct_result(res, dataset, model, pred_level)
        res["transfer_model"] = get_transfer_model(res)

    # separate by the pretrained model
    per_pretrained = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(all_results):
        pretrained = r["config"]["graphgym"]["pretrained"]["dir"]
        maybe_adv_trained, name = pretrained.split("/")[-2:]
        adv_trained = maybe_adv_trained == "adv"

        if not adv_trained and r["transfer_model"] == model:
            # get the "adpative" results from the attack collection (where we get random as well)
            continue 

        per_pretrained[name]["results"].append(r)
        for n, v in zip(
            [
                "run_ids",
                "run_dirs",
                "num_params",
            ],
            [
                all_run_ids[i],
                all_run_dirs[i],
                all_num_params[i],
            ]
        ):
            per_pretrained[name][n].append(v)

    all_adaptive_results, all_rand_results = get_attack_collection_results(
        attack_collection, dataset, model, pred_level,
    )
    
    all_rand_pert_results = get_rand_pert_collection_results(
        rand_pert_collection, dataset, model, pred_level,
    )
    
    # create the paths for each different model, dict
    results_paths, info_files, seed_dirs = clean_path(results_path, list(per_pretrained))

    for name, d in per_pretrained.items():
        write_info_file(info_files[name], d["run_ids"], d["num_params"], extras, d["run_dirs"])
        results = d["results"]
        # sort alphabetically
        results.sort(key=lambda x: model_order[x["transfer_model"]])
        # append random pert
        if name not in all_rand_pert_results:
            raise Exception(
                f"Did not find *random_pert* results for pretrained mode `{name}`"
                f"in rand_pert collection `{rand_pert_collection}`"
            )
        results.append(all_rand_pert_results[name])
        # append random and adaptive
        if name not in all_rand_results:
            raise Exception(
                f"Did not find *random* results for pretrained mode `{name}`"
                f"in attack collection `{attack_collection}`"
            )
        results.append(all_rand_results[name])
        if name not in all_adaptive_results:
            raise Exception(
                f"Did not find *adaptive* results for pretrained mode `{name}` "
                f"in attack collection `{attack_collection}`, (model={model}, dataset={dataset})"
            )
        results.append(all_adaptive_results[name])

        run_seed_dataframes = write_results(
            seed_dirs[name],
            results,
            attack_cols,
        )
        # plots
        save_plots(
            model,
            dataset,
            run_seed_dataframes,
            results_paths[name],
            attack_cols,
            max_idx_small_budget,
            agg_cols,
            seed_dirs[name],
            y_label,
            add_title,
            add_legend,
            y_min,
            y_max,
            y_min_margin,
            y_max_margin,
            figsize,
            figsize_all,
        )


parser = argparse.ArgumentParser(description='Processes the results of transfer attack.')
parser.add_argument("-c", "--collection")
parser.add_argument("-d", "--dataset")
parser.add_argument("-m", "--model")
parser.add_argument("-s", "--small-budget-idx", type=int, default=4)
parser.add_argument("-a", "--attack-collection")
parser.add_argument("-r", "--rand-pert-collection")
parser.add_argument("-y", "--y-label", action="store_true")
parser.add_argument("-g", "--grid", action="store_true")
parser.add_argument("-t", "--title", action="store_true")
parser.add_argument("-l", "--legend", action="store_true")
parser.add_argument("-b", "--best-transfer-only", action="store_true")
parser.add_argument("--y-min", type=float, default=None)
parser.add_argument("--y-max", type=float, default=None)
parser.add_argument("--y-min-margin", type=float, default=1)
parser.add_argument("--y-max-margin", type=float, default=-1)
parser.add_argument("--fs-w", type=float, default=2.5)
parser.add_argument("--fs-h", type=float, default=2.0)
parser.add_argument("--fs-w-all", type=float, default=4.5)
parser.add_argument("--fs-h-all", type=float, default=3.5)


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
        max_idx_small_budget=args.small_budget_idx,
        attack_collection=args.attack_collection,
        rand_pert_collection=args.rand_pert_collection,
        y_label=args.y_label,
        grid=args.grid,
        add_title=args.title,
        add_legend=args.legend,
        y_min=args.y_min,
        y_max=args.y_max,
        y_min_margin=args.y_min_margin,
        y_max_margin=args.y_max_margin,
        fs_w=args.fs_w,
        fs_h=args.fs_h,
        fs_w_all=args.fs_w_all,
        fs_h_all=args.fs_h_all,
    )
