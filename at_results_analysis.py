import seml
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pathlib import Path
from dash import Dash, html, dcc, Output, Input, Patch
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import argparse
import shutil


def clean_path(results_path: str):
    results_path = Path(results_path)
    if results_path.exists():
        shutil.rmtree(results_path)
    results_path.mkdir(parents=True)
    result_log_file = results_path / "hs_result_analysis.txt"
    return results_path, result_log_file


def get_split_metric_results(training_results, split: str, metric: str, agg: str):
    if agg == "max":
        agg_fun = max
    elif agg == "min":
        agg_fun = min
    else:
        raise NotImplementedError(f"agg={agg} is not implemented")
    all_values = []
    for r in training_results:
        all_values.append(r[split][metric])
    agg_values = [agg_fun(m) for m in all_values]
    epochs_of_agg_values = [np.arange(len(m))[np.array(m) == max_m].tolist() for m, max_m in zip(all_values, agg_values)]
    agg_value = max(agg_values)
    ind_of_agg_value = np.arange(len(training_results))[np.array(agg_values) == agg_value].tolist()
    return {
        "all_values": all_values,
        f"{agg}_values": agg_values,
        f"epochs_of_{agg}_values": epochs_of_agg_values,
        f"{agg}_value": agg_value,
        f"ind_of_{agg}_value": ind_of_agg_value,
    }


def get_highest_k_indices(values, k):
    ind = np.argpartition(values, -k)[-k:]
    best_ind = ind.tolist()
    best_ind_sorted = ind[np.argsort(np.array(values)[best_ind])].tolist()[::-1]
    return best_ind_sorted


def get_training_results(results):
    metric = results[0]["config"]["graphgym"]["metric_best"]
    for result in results:
        assert result["config"]["graphgym"]["metric_best"] == metric, "Not all metrics are the same"
    training_results = []
    for result in results:
        training_results.append(result["result"]["training"])
    N = len(training_results)
    # train
    train_metric = get_split_metric_results(training_results, "train", metric, "max")
    train_loss = get_split_metric_results(training_results, "train", "loss", "min")
    # val:
    val_metric = get_split_metric_results(training_results, "val", metric, "max")
    val_loss = get_split_metric_results(training_results, "val", "loss", "min")
    # test:
    test_metric = get_split_metric_results(training_results, "test", metric, "max")
    test_loss = get_split_metric_results(training_results, "test", "loss", "min")
    # val_adv:
    val_adv_metric = get_split_metric_results(training_results, "val_adv", metric, "max")
    val_adv_loss = get_split_metric_results(training_results, "val_adv", "loss", "min")
    # looking at interesting epochs of best experiments
    best_epochs = [
        sorted(
            set(
                val_metric["epochs_of_max_values"][i]
                + val_loss["epochs_of_min_values"][i]
                + test_metric["epochs_of_max_values"][i]
                + test_loss["epochs_of_min_values"][i]
                + val_adv_metric["epochs_of_max_values"][i]
                + val_adv_loss["epochs_of_min_values"][i]
            )
        ) 
        for i in range(N)
    ]
    return (
        metric,
        best_epochs,
        train_metric,
        val_metric,
        test_metric,
        val_adv_metric,
        train_loss,
        val_loss,
        test_loss,
        val_adv_loss,
    )


def write_results_into_file(
    result_log_file,
    results,
    configs,
    metric,
    best_epochs,
    train_metric,
    val_metric,
    test_metric,
    val_adv_metric,
    train_loss,
    val_loss,
    test_loss,
    val_adv_loss,
    run_ids,
    extras,
    seeds_graphgym,
    seeds_seml,
    run_dirs,
    num_params,
    adv_budgets,
    adv_num_replays,
    write_epochs=False,
    write_configs=False,
):
    with open(result_log_file, "w") as f:
        f.write("\nExperiments with best epochs:")
        for i, epochs in enumerate(best_epochs):
            f.write(f"\nexperiment: {run_ids[i]}")
            f.write(f"\n\tgraphgym.train.adv.e_budget: {adv_budgets[i]}")
            f.write(f"\n\tgraphgym.train.adv.num_replays: {adv_num_replays[i]}")
            f.write(f"\n\tgraphgym.seed: {seeds_graphgym[i]}")
            if write_configs:
                f.write("\n\tinfos:")
                p = num_params[i]
                if p is not None:
                    f.write(f"\n\t\tnum_params: {p}")
                for k, v in extras.items():
                    f.write(f"\n\t\t{k}: {v[i]}")
                f.write(f"\n\t\tseed(seml): {seeds_seml[i]}")
            run_dir = run_dirs[i]
            if run_dir is not None:
                f.write(f"\n\t\trun_dir: {run_dir}")
            f.write(f"\n\tbest val {metric}: {val_metric['max_values'][i]}, at epochs: {val_metric['epochs_of_max_values'][i]}")
            f.write(f"\n\tbest test {metric}: {test_metric['max_values'][i]}, at epochs: {test_metric['epochs_of_max_values'][i]}")
            f.write(f"\n\tbest val_adv {metric}: {val_adv_metric['max_values'][i]}, at epochs: {val_adv_metric['epochs_of_max_values'][i]}")
            f.write(f"\n\tbest val loss: {val_loss['min_values'][i]}, at epochs: {val_loss['epochs_of_min_values'][i]}")
            f.write(f"\n\tbest test loss: {test_loss['min_values'][i]}, at epochs: {test_loss['epochs_of_min_values'][i]}")
            f.write(f"\n\tbest val_adv loss: {val_adv_loss['min_values'][i]}, at epochs: {val_adv_loss['epochs_of_min_values'][i]}")
            if write_configs:
                f.write("\n\tconfigs:")
                for conf in configs:
                    sep_keys = conf.split(".")
                    val = results[i]["config"][sep_keys[0]]
                    j = 1
                    while isinstance(val, dict):
                        val = val.get(sep_keys[j])
                        j += 1
                    f.write(f"\n\t\t{conf}: {val}")
            if write_epochs:
                f.write("\n\tinteresting epochs:")
                for e in epochs:
                    f.write(f"\n\t\tepoch: {e}")
                    f.write(f"\n\t\t\ttrain {metric}: {train_metric['all_values'][i][e]}")
                    f.write(f"\n\t\t\tval {metric}: {val_metric['all_values'][i][e]}")
                    f.write(f"\n\t\t\ttest {metric}: {test_metric['all_values'][i][e]}")
                    f.write(f"\n\t\t\tval_adv {metric}: {val_adv_metric['all_values'][i][e]}")
                    f.write(f"\n\t\t\ttrain loss: {train_loss['all_values'][i][e]}")
                    f.write(f"\n\t\t\tval loss: {val_loss['all_values'][i][e]}")
                    f.write(f"\n\t\t\ttest loss: {test_loss['all_values'][i][e]}")
                    f.write(f"\n\t\t\tval_adv loss: {val_adv_loss['all_values'][i][e]}")
            f.write("\n")


def save_single_plots(
    results,
    results_path,
    metric,
    train_metric,
    val_metric,
    test_metric,
    val_adv_metric,
    train_loss,
    val_loss,
    test_loss,
    val_adv_loss,
    run_ids,
    adv_budgets,
    adv_num_replays,
):
    single_plots_dir = results_path / "single-plots"
    train_plot_dir = single_plots_dir / "training"
    train_plot_dir.mkdir(parents=True)
    fig, (((ax_metric)), (ax_loss)) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    for i in range(len(results)):
        (ax_metric).set_title(metric)
        ax_loss.set_title("Loss")
        x = np.arange(len(val_metric['all_values'][i]))
        (ax_metric).plot(x, train_metric['all_values'][i], "-g", label='train')
        (ax_metric).plot(x, val_metric['all_values'][i], "-b", label='val')
        (ax_metric).plot(x, test_metric['all_values'][i], "-r", label='test')
        (ax_metric).plot(x, val_adv_metric['all_values'][i], "-k", label='val_adv')
        (ax_metric).legend()
        ax_loss.plot(x, train_loss['all_values'][i], "-g", label='train')
        ax_loss.plot(x, val_loss['all_values'][i], "-b", label='val')
        ax_loss.plot(x, test_loss['all_values'][i], "-r", label='test')
        ax_loss.plot(x, val_adv_loss['all_values'][i], "-k", label='val_adv')
        ax_loss.legend()
        fig.savefig(train_plot_dir / f"{run_ids[i]}.png")
        plt.close(fig)
        (ax_metric).clear()
        ax_loss.clear()

    cfg_plot_dir = single_plots_dir / "configs"
    cfg_plot_dir.mkdir(parents=True)
    val_max_metric = np.array(val_metric["max_values"])
    test_max_metric = np.array(test_metric["max_values"])
    val_adv_max_metric = np.array(val_adv_metric["max_values"])

    for agg_values, name in zip([adv_budgets, adv_num_replays], ["adv_train_budget", "adv_num_replays"]):

        # aggregate onto different values
        idx = {}
        for e_budget in set(agg_values):
            idx[e_budget] = []
        for i in range(len(results)):
            idx[agg_values[i]].append(i)
        
        fig, ((ax_val), (ax_test), (ax_val_adv)) = plt.subplots(nrows=3, ncols=1, figsize=(18, 8))
        ax_val.set_title(f"max val {metric} / {name}")
        ax_test.set_title(f"max test {metric} / {name}")
        ax_val_adv.set_title(f"max val_adv {metric} / {name}")
        x = np.array(agg_values)
        values = np.array(sorted(idx))
        val_dataset = [val_max_metric[idx[v]] for v in values]
        test_dataset = [test_max_metric[idx[v]] for v in values]
        val_adv_dataset = [val_adv_max_metric[idx[v]] for v in values]
        min_gap = 1 if len(val_dataset) == 1 else np.min(np.diff(values))
        w = 0.4 * min_gap
        ax_val.violinplot(val_dataset, values, showmeans=True, showmedians=True, widths=w)
        ax_test.violinplot(test_dataset, values, showmeans=True, showmedians=True, widths=w)
        ax_val_adv.violinplot(val_adv_dataset, values, showmeans=True, showmedians=True, widths=w)
        x = x + w * (np.random.rand(len(x)) - 0.5)
        ax_val.scatter(x, val_max_metric, s=12, c="g")
        ax_test.scatter(x, test_max_metric, s=12, c="g")
        ax_val_adv.scatter(x, val_adv_max_metric, s=12, c="g")
        fig.savefig(cfg_plot_dir / f"{name}.png")
        plt.close(fig)


def get_collection_results(collection, filter_dict):
    extra_fields = [
        'slurm.array_id', 'slurm.experiments_per_job', 'slurm.task_id', 'stats.real_time',
        'stats.pytorch.gpu_max_memory_bytes', 'stats.self.max_memory_bytes',
    ]
    results = seml.get_results(
        collection,
        ['config', 'result'] + extra_fields,
        filter_dict=filter_dict
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
    for result in results:
        model = result["config"]["graphgym"]["model"]["type"]
        d_per_head = result["config"]["dims_per_head"]
        d_hidden = result["config"]["graphgym"]["gnn"]["dim_inner"]
        if d_per_head > 0 and d_hidden == 0:
            if model == "Graphormer":
                n_heads = result["config"]["graphgym"]["graphormer"]["num_heads"]
            elif model in ["GritTransformer", "WeightedSANTransformer", "SANTransformer", "GPSModel", "WeightedPolynormer"]:
                n_heads = result["config"]["graphgym"]["gt"]["n_heads"]
            else:
                raise NotImplementedError(f"model={model} is not implemented")
            d_hidden = d_per_head * n_heads
            result["config"]["graphgym"]["gnn"]["dim_inner"] = d_hidden
        dims_per_head_PE = result["config"].get("dims_per_head_PE", 0)
        if dims_per_head_PE > 0 and result["config"]["graphgym"]["posenc_WLapPE"]["dim_pe"] == 0:
            dim_pe = dims_per_head_PE * result["config"]["graphgym"]["posenc_WLapPE"]["n_heads"]
            result["config"]["graphgym"]["posenc_WLapPE"]["dim_pe"] = dim_pe
    return results, run_ids, extras, seeds_graphgym, seeds_seml, run_dirs, num_params


def main(
    collection: str,
    configs_all_info: list[tuple[str, bool, bool]],
    dataset: str,
    model: str,
    results_path: str,
    filter_dict,
):
    (
        results,
        run_ids,
        extras,
        seeds_graphgym,
        seeds_seml,
        run_dirs,
        num_params,
    ) = get_collection_results(collection, filter_dict)

    # sort by 1) e_budget 2) num_replays 3) seed
    adv_budgets = [x["config"]["graphgym"]["train"]["adv"]["e_budget"] for x in results]
    adv_num_replays = [x["config"]["graphgym"]["train"]["adv"]["num_replays"] for x in results]
    idx = sorted([
        (
            adv_budgets[i],
            adv_num_replays[i],
            seeds_graphgym[i],
            i,
        )
        for i in range(len(results))
    ])
    idx = [i[-1] for i in idx]
    new_results = []
    new_run_ids = []
    new_seeds_graphgym = []
    new_seeds_seml = []
    new_run_dirs = []
    new_num_params = []
    new_adv_budgets = []
    new_adv_num_replays = []
    for i in idx:
        new_results.append(results[i])
        new_run_ids.append(run_ids[i])
        new_seeds_graphgym.append(seeds_graphgym[i])
        new_seeds_seml.append(seeds_seml[i])
        new_run_dirs.append(run_dirs[i])
        new_num_params.append(num_params[i])
        new_adv_budgets.append(adv_budgets[i])
        new_adv_num_replays.append(adv_num_replays[i])
    results = new_results
    run_ids = new_run_ids
    seeds_graphgym = new_seeds_graphgym
    seeds_seml = new_seeds_seml
    run_dirs = new_run_dirs
    num_params = new_num_params
    adv_budgets = new_adv_budgets
    adv_num_replays = new_adv_num_replays
    

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

    configs = [c[0] for c in configs_all_info]
    (  # process results
        metric,
        best_epochs,
        train_metric,
        val_metric,
        test_metric,
        val_adv_metric,
        train_loss,
        val_loss,
        test_loss,
        val_adv_loss,
    ) = get_training_results(results)  
    results_path, result_log_file = clean_path(results_path)
    # write results into file
    write_results_into_file(
        result_log_file,
        results,
        configs,
        metric,
        best_epochs,
        train_metric,
        val_metric,
        test_metric,
        val_adv_metric,
        train_loss,
        val_loss,
        test_loss,
        val_adv_loss,
        run_ids,
        extras,
        seeds_graphgym,
        seeds_seml,
        run_dirs,
        num_params,
        adv_budgets,
        adv_num_replays,
    )
    # plots
    save_single_plots(
        results,
        results_path,
        metric,
        train_metric,
        val_metric,
        test_metric,
        val_adv_metric,
        train_loss,
        val_loss,
        test_loss,
        val_adv_loss,
        run_ids,
        adv_budgets,
        adv_num_replays,
    )


hyperparameters = {
    "Graphormer": [  # name, discrete, log_scale
        ("graphgym.optim.base_lr", False, True),
        ("graphgym.optim.weight_decay", False, True),
        ("graphgym.gnn.dim_inner", False, False),
        ("graphgym.graphormer.num_heads", True, False),
        ("graphgym.graphormer.num_layers", True, False),
        ("graphgym.gnn.layers_post_mp", True, False),
        ("graphgym.optim.num_warmup_epochs", False, False),
        ("graphgym.posenc_GraphormerBias.num_spatial_types", True, False),
        ("graphgym.posenc_GraphormerBias.num_in_degrees", False, False),
        ("graphgym.gnn.head", True, False),
        ("graphgym.train.batch_size", True, False),
        ("graphgym.train.homophily_regularization", False, False),
        ("graphgym.train.homophily_regularization_gt_weight", False, False),
        ("graphgym.graphormer.dropout", False, False),
        ("graphgym.graphormer.attention_dropout", False, False),
        ("graphgym.graphormer.mlp_dropout", False, False),
        ("graphgym.graphormer.input_dropout", False, False)
    ],
    "GCN": [
        ("graphgym.optim.base_lr", False, True),
        ("graphgym.optim.weight_decay", False, True),
        ("graphgym.gnn.dim_inner", False, False),
        ("graphgym.optim.num_warmup_epochs", True, False),
        ("graphgym.gnn.layers_pre_mp", True, False),
        ("graphgym.gnn.layers_mp", True, False),
        ("graphgym.gnn.head", True, False),
        ("graphgym.gnn.layers_post_mp", True, False),
        ("graphgym.gnn.act", True, False),
        ("graphgym.gnn.dropout", False, False),
        ("graphgym.gnn.agg", True, False),
        ("graphgym.train.batch_size", True, False),
        ("graphgym.train.homophily_regularization", False, False),
        ("graphgym.train.homophily_regularization_gt_weight", False, False),
    ],
    "GAT": [
        ("graphgym.optim.base_lr", False, True),
        ("graphgym.optim.weight_decay", False, True),
        ("graphgym.gnn.dim_inner", False, False),
        ("graphgym.optim.num_warmup_epochs", True, False),
        ("graphgym.gnn.layers_pre_mp", True, False),
        ("graphgym.gnn.layers_mp", True, False),
        ("graphgym.gnn.head", True, False),
        ("graphgym.gnn.layers_post_mp", True, False),
        ("graphgym.gnn.act", True, False),
        ("graphgym.gnn.dropout", False, False),
        ("graphgym.gnn.agg", True, False),
        ("graphgym.train.batch_size", True, False),
        ("graphgym.train.homophily_regularization", False, False),
        ("graphgym.train.homophily_regularization_gt_weight", False, False),
    ],
    "GATv2": [
        ("graphgym.optim.base_lr", False, True),
        ("graphgym.optim.weight_decay", False, True),
        ("graphgym.gnn.dim_inner", False, False),
        ("graphgym.optim.num_warmup_epochs", True, False),
        ("graphgym.gnn.layers_pre_mp", True, False),
        ("graphgym.gnn.layers_mp", True, False),
        ("graphgym.gnn.head", True, False),
        ("graphgym.gnn.layers_post_mp", True, False),
        ("graphgym.gnn.act", True, False),
        ("graphgym.gnn.dropout", False, False),
        ("graphgym.gnn.agg", True, False),
        ("graphgym.train.batch_size", True, False),
        ("graphgym.train.homophily_regularization", False, False),
        ("graphgym.train.homophily_regularization_gt_weight", False, False),
    ],
    "GRIT": [
        ("graphgym.optim.base_lr", False, True),
        ("graphgym.optim.weight_decay", False, True),
        ("graphgym.optim.num_warmup_epochs", True, False),
        ("graphgym.gnn.dim_inner", False, False),
        ("graphgym.gt.n_heads", True, False),
        ("graphgym.gt.layers", True, False),
        ("graphgym.gt.dropout", False, False),
        ("graphgym.gt.attn_dropout", False, False),
        ("graphgym.gt.bn_no_runner", True, False),
        ("graphgym.gt.bn_momentum", True, False),
        ("graphgym.posenc_RRWP.ksteps", True, False),
        ("graphgym.posenc_RRWP.w_add_dummy_edge", True, False),
        ("graphgym.gnn.head", True, False),
        ("graphgym.gnn.layers_post_mp", True, False),
        ("graphgym.train.batch_size", True, False),
        ("graphgym.train.homophily_regularization", False, False),
        ("graphgym.train.homophily_regularization_gt_weight", False, False),
    ],
    "SAN": [
        ("graphgym.optim.base_lr", False, True),
        ("graphgym.optim.weight_decay", False, True),
        ("graphgym.gnn.dim_inner", False, False),
        ("graphgym.gt.n_heads", True, False),
        ("graphgym.gt.layers", True, False),
        ("graphgym.gt.dropout", False, False),
        ("graphgym.posenc_WLapPE.w_add_dummy_edge", True, False),
        ("graphgym.gt.gamma", False, True),
        ("graphgym.gt.attn.clamp", True, False),
        ("graphgym.posenc_WLapPE.n_heads", True, False),
        ("graphgym.posenc_WLapPE.layers", True, False),
        ("graphgym.posenc_WLapPE.dim_pe", False, False),
        ("graphgym.posenc_WLapPE.eigen.max_freqs", False, False),
        ("graphgym.gnn.head", True, False),
        ("graphgym.gnn.layers_post_mp", True, False),
        ("graphgym.train.batch_size", True, False),
        ("graphgym.train.homophily_regularization", False, False),
        ("graphgym.train.homophily_regularization_gt_weight", False, False),
    ],
    "GPS": [
        ("graphgym.optim.base_lr", False, True),
        ("graphgym.optim.weight_decay", False, True),
        ("graphgym.gnn.dim_inner", False, False),
        ("graphgym.gt.n_heads", True, False),
        ("graphgym.gt.layers", True, False),
        ("graphgym.gt.dropout", False, False),
        ("graphgym.posenc_WLapPE.layers", True, False),
        ("graphgym.posenc_WLapPE.dim_pe", False, False),
        ("graphgym.posenc_WLapPE.eigen.max_freqs", False, False),
        ("graphgym.gnn.head", True, False),
        ("graphgym.gnn.layers_post_mp", True, False),
        ("graphgym.train.batch_size", True, False),
        ("graphgym.train.homophily_regularization", False, False),
        ("graphgym.train.homophily_regularization_gt_weight", False, False),
    ],
    "Polynormer": [
        ("graphgym.optim.base_lr", False, True),
        ("graphgym.optim.weight_decay", False, True),
        ("graphgym.gnn.dim_inner", False, False),
        ("graphgym.gt.n_heads", True, False),
        ("graphgym.gt.layers", True, False),
        ("graphgym.gt.dropout", False, False),
        ("graphgym.gnn.head", True, False),
        ("graphgym.gnn.dropout", False, False),
        ("graphgym.gnn.layers_post_mp", True, False),
        ("graphgym.gnn.layers_mp", True, False),
        ("graphgym.train.batch_size", True, False),
        ("graphgym.train.homophily_regularization", False, False),
        ("graphgym.train.homophily_regularization_gt_weight", False, False),
    ],
}

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
    "GPS": {"type": set(["GPSModel"]), "gnn_layer_type": None},
    "Polynormer": {"type": set(["WeightedPolynormer"]), "gnn_layer_type": None},
}


parser = argparse.ArgumentParser(description='Processes the results of hyperparameter search.')
parser.add_argument("-c", "--collection")
parser.add_argument("-d", "--dataset")
parser.add_argument("-m", "--model")


if __name__ == "__main__":
    args = parser.parse_args()
    results_path = f"results_at/{args.dataset}/{args.model}/{args.collection}"
    filter_dict = None  # {'slurm.array_id': 8919896}  # None  # not implemented for argparse... but can manually change here
    main(
        collection=args.collection,
        configs_all_info=hyperparameters[args.model],
        dataset=args.dataset,
        model=args.model,
        results_path=results_path,
        filter_dict=filter_dict,
    )
