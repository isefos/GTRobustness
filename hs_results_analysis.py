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


def get_split_metric_results(training_results, split: str, metric: str, agg: str, old_logging: bool):
    if agg == "max":
        agg_fun = max
    elif agg == "min":
        agg_fun = min
    else:
        raise NotImplementedError(f"agg={agg} is not implemented")
    if old_logging:
        # "old"
        split_results = [r[split] for r in training_results]
        all_values = [[e[metric] for e in r] for r in split_results]
    else:
        # "new"
        all_values = [r[split][metric] for r in training_results]
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


def get_ranked_results(results, k):
    metric = results[0]["config"]["graphgym"]["metric_best"]
    for result in results:
        assert result["config"]["graphgym"]["metric_best"] == metric, "Not all metrics are the same"
    old_logging = "train" in results[0]["result"]
    if old_logging:
        training_results = [result["result"]["train"] for result in results]
    else:
        training_results = [result["result"]["training"] for result in results]
    train_metric = get_split_metric_results(training_results, "train", metric, "max", old_logging)
    train_loss = get_split_metric_results(training_results, "train", "loss", "min", old_logging)
    # val:
    val_metric = get_split_metric_results(training_results, "val", metric, "max", old_logging)
    best_val_epochs = [r["best_val_epoch"] for r in training_results]
    for epochs_max_metric, best_val_epoch in zip(val_metric["epochs_of_max_values"], best_val_epochs):
        assert best_val_epoch in epochs_max_metric
    best_val_metric_ind = get_highest_k_indices(val_metric["max_values"], k)
    val_loss = get_split_metric_results(training_results, "val", "loss", "min", old_logging)
    # test:
    test_metric = get_split_metric_results(training_results, "test", metric, "max", old_logging)
    best_test_metric_ind = get_highest_k_indices(test_metric["max_values"], k)
    test_loss = get_split_metric_results(training_results, "test", "loss", "min", old_logging)
    # looking at interesting epochs of best experiments
    best_experiments = sorted(set(best_val_metric_ind + best_test_metric_ind))
    best_epochs = [
        sorted(
            set(
                val_metric["epochs_of_max_values"][i]
                + val_loss["epochs_of_min_values"][i]
                + test_metric["epochs_of_max_values"][i]
                + test_loss["epochs_of_min_values"][i]
            )
        ) 
        for i in best_experiments
    ]
    best_val_with_test_metric = []
    for i in best_val_metric_ind:
        best_val_with_test_metric.append((val_metric["max_values"][i], test_metric["all_values"][i][best_val_epochs[i]], i))
    best_val_with_test_metric = sorted(best_val_with_test_metric, reverse=True)
    return (
        metric,
        best_experiments,
        best_epochs,
        best_val_with_test_metric,
        best_test_metric_ind,
        train_metric,
        val_metric,
        test_metric,
        train_loss,
        val_loss,
        test_loss,
    )


def write_results_into_file(
    result_log_file,
    results,
    configs,
    metric,
    best_experiments,
    best_epochs,
    best_val_with_test_metric,
    best_test_metric_ind,
    train_metric,
    val_metric,
    test_metric,
    train_loss,
    val_loss,
    test_loss,
    run_ids,
    extras,
    seeds_graphgym,
    seeds_seml,
    run_dirs,
    num_params,
):
    with open(result_log_file, "w") as f:
        f.write(f"\nHighest val {metric}:")
        for (v_a, t_a, i) in best_val_with_test_metric:
            f.write(f"\n\tval {metric}: {v_a}, with {t_a} test {metric}, by experiment: {run_ids[i]}")
        f.write(f"\nHighest test {metric}:")
        for i in best_test_metric_ind:
            f.write(f"\n\ttest {metric}: {test_metric['max_values'][i]}, by experiment: {run_ids[i]}")
        f.write("\nBest experiments with best epochs:")
        for i, epochs in zip(best_experiments, best_epochs):
            f.write(f"\nexperiment: {run_ids[i]}")
            f.write("\n\tinfos:")
            p = num_params[i]
            if p is not None:
                f.write(f"\n\t\tnum_params: {p}")
            for k, v in extras.items():
                f.write(f"\n\t\t{k}: {v[i]}")
            f.write(f"\n\t\tgraphgym.seed: {seeds_graphgym[i]}")
            f.write(f"\n\t\tseed(seml): {seeds_seml[i]}")
            run_dir = run_dirs[i]
            if run_dir is not None:
                f.write(f"\n\t\trun_dir: {run_dir}")
            f.write(f"\n\tbest val {metric}: {val_metric['max_values'][i]}, at epochs: {val_metric['epochs_of_max_values'][i]}")
            f.write(f"\n\tbest test {metric}: {test_metric['max_values'][i]}, at epochs: {test_metric['epochs_of_max_values'][i]}")
            f.write(f"\n\tbest val loss: {val_loss['min_values'][i]}, at epochs: {val_loss['epochs_of_min_values'][i]}")
            f.write(f"\n\tbest test loss: {test_loss['min_values'][i]}, at epochs: {test_loss['epochs_of_min_values'][i]}")
            f.write("\n\tconfigs:")
            for conf in configs:
                sep_keys = conf.split(".")
                val = results[i]["config"][sep_keys[0]]
                j = 1
                while isinstance(val, dict):
                    val = val.get(sep_keys[j])
                    j += 1
                f.write(f"\n\t\t{conf}: {val}")
            f.write("\n\tinteresting epochs:")
            for e in epochs:
                f.write(f"\n\t\tepoch: {e}")
                f.write(f"\n\t\t\ttrain {metric}: {train_metric['all_values'][i][e]}")
                f.write(f"\n\t\t\tval {metric}: {val_metric['all_values'][i][e]}")
                f.write(f"\n\t\t\ttest {metric}: {test_metric['all_values'][i][e]}")
                f.write(f"\n\t\t\ttrain loss: {train_loss['all_values'][i][e]}")
                f.write(f"\n\t\t\tval loss: {val_loss['all_values'][i][e]}")
                f.write(f"\n\t\t\ttest loss: {test_loss['all_values'][i][e]}")
            f.write("\n")


def save_single_plots(
    results,
    results_path,
    metric,
    best_experiments,
    train_metric,
    val_metric,
    test_metric,
    train_loss,
    val_loss,
    test_loss,
    configs_all_info,
    run_ids,
):
    single_plots_dir = results_path / "single-plots"
    train_plot_dir = single_plots_dir / "training"
    train_plot_dir.mkdir(parents=True)
    fig, (((ax_metric)), (ax_loss)) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    for i in best_experiments:
        (ax_metric).set_title(metric)
        ax_loss.set_title("Loss")
        x = np.arange(len(val_metric['all_values'][i]))
        (ax_metric).plot(x, train_metric['all_values'][i], "-g", label='train')
        (ax_metric).plot(x, val_metric['all_values'][i], "-b", label='val')
        (ax_metric).plot(x, test_metric['all_values'][i], "-r", label='test')
        (ax_metric).legend()
        ax_loss.plot(x, train_loss['all_values'][i], "-g", label='train')
        ax_loss.plot(x, val_loss['all_values'][i], "-b", label='val')
        ax_loss.plot(x, test_loss['all_values'][i], "-r", label='test')
        ax_loss.legend()
        fig.savefig(train_plot_dir / f"{run_ids[i]}.png")
        plt.close(fig)
        (ax_metric).clear()
        ax_loss.clear()

    cfg_plot_dir = single_plots_dir / "configs"
    cfg_plot_dir.mkdir(parents=True)
    # filter results for lr < 1.3e-3
    # mask = [result["config"]["graphgym"]["optim"]["base_lr"] < 1.3e-3 for result in results]
    # results = [result for result, m in zip(results, mask) if m]
    # val_max_metric = [v for v, m in zip(val_metric["max_values"], mask) if m]
    # test_max_metric = [v for v, m in zip(test_metric["max_values"], mask) if m]
    val_max_metric = val_metric["max_values"]
    test_max_metric = test_metric["max_values"]

    for (conf, discrete, log) in configs_all_info:
        fig, ((ax_val), (ax_test)) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
        ax_val.set_title(f"max val {metric} / {conf}")
        ax_test.set_title(f"max test {metric} / {conf}")
        x = []
        for result in results:
            v = result["config"]
            conf_split = conf.split(".")
            for k in conf_split:
                v = v.get(k, None)
                if v is None:
                    break
            if v is None:
                continue
            x.append(v)
        if not x:
            continue
        if discrete:
            try:
                x = np.array(x)
                values, unique_inv = np.unique(x, return_inverse=True)
            except:
                x = np.array([f"{i}" for i in x])
                values, unique_inv = np.unique(x, return_inverse=True)
            tick_labels = None
            if values.dtype.type is np.str_:
                tick_labels = [str(a) for a in values]
                values = np.arange(1, values.size + 1)
                x = values[unique_inv]
            val_dataset = []
            test_dataset = []
            for v in values:
                val_dataset.append(np.array(val_max_metric)[x == v])
                test_dataset.append(np.array(test_max_metric)[x == v])
            ax_val.violinplot(val_dataset, values, showmeans=True, showmedians=True)
            ax_test.violinplot(test_dataset, values, showmeans=True, showmedians=True)
            x = x + 0.5 * (np.random.rand(len(x)) - 0.5)
            ax_val.scatter(x, val_max_metric, s=12, c="g")
            ax_test.scatter(x, test_max_metric, s=12, c="g")
            ax_val.set_xticks(values, labels=tick_labels)
            ax_test.set_xticks(values, labels=tick_labels)
        else:
            ax_val.scatter(x, val_max_metric)
            ax_test.scatter(x, test_max_metric)
            if log:
                ax_val.set_xscale('log')
                ax_test.set_xscale('log')
        fig.savefig(cfg_plot_dir / f"{conf}.png")
        plt.close(fig)


def get_collection_results(collection, model, filter_dict):
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
                r = r[key_l]
            key_last = keys_list[-1]
            v = r[key_last]
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
        d_per_head = result["config"]["dims_per_head"]
        d_hidden = result["config"]["graphgym"]["gnn"]["dim_inner"]
        if d_per_head > 0 and d_hidden == 0:
            if model == "Graphormer":
                n_heads = result["config"]["graphgym"]["graphormer"]["num_heads"]
            elif model in ["GRIT", "WeightedSANTransformer", "SANTransformer"]:
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
    k: int,
    results_path: str,
    filter_dict,
    model: str,
    old_plotting: bool = False,
):
    (
        results,
        run_ids,
        extras,
        seeds_graphgym,
        seeds_seml,
        run_dirs,
        num_params,
    ) = get_collection_results(collection, model, filter_dict)
    configs = [c[0] for c in configs_all_info]
    (  # process results
        metric,
        best_experiments,
        best_epochs,
        best_val_with_test_metric,
        best_test_metric_ind,
        train_metric,
        val_metric,
        test_metric,
        train_loss,
        val_loss,
        test_loss,
    ) = get_ranked_results(results, k)  
    results_path, result_log_file = clean_path(results_path)
    # write results into file
    write_results_into_file(
        result_log_file,
        results,
        configs,
        metric,
        best_experiments,
        best_epochs,
        best_val_with_test_metric,
        best_test_metric_ind,
        train_metric,
        val_metric,
        test_metric,
        train_loss,
        val_loss,
        test_loss,
        run_ids,
        extras,
        seeds_graphgym,
        seeds_seml,
        run_dirs,
        num_params,
    )
    # plots
    if old_plotting:
        save_single_plots(
            results,
            results_path,
            metric,
            best_experiments,
            train_metric,
            val_metric,
            test_metric,
            train_loss,
            val_loss,
            test_loss,
            configs_all_info,
            run_ids,
        )
        return None
    else:
        # TODO: make a dataframe
        plot_results = {}
        df = pd.DataFrame(plot_results)
        # make the dash app to display the plots
        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        dims = ["sepal_width", "sepal_length", "petal_width", "petal_length"]
        df = px.data.iris()
        fig = px.parallel_coordinates(
            df,
            color="species_id",
            dimensions=dims,
            color_continuous_scale=px.colors.diverging.Tealrose,
            color_continuous_midpoint=2,
        )
        app.layout = dbc.Container(
            [
                html.H4("Filtering a Datatable with Parallel Coordinates"),
                dcc.Graph(id="my-graph", figure=fig),
                dag.AgGrid(
                    id="table",
                    columnDefs=[{"field": i} for i in df.columns],
                    columnSize="sizeToFit",
                    defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth":125},
                    dashGridOptions={"rowSelection": "single", "animateRows": False},
                ),
                dcc.Store(id="activefilters", data={}),
                html.Br(),
                html.H5("Select a run to view traing curves:"),
                dcc.Dropdown(id="select-run"),
                html.Br(),
                dcc.Graph(figure={}, id="train-graph"),
            ]
        )

        @app.callback(
            Output("train-graph", "figure"),
            Input("select-run", "value"),
        )
        def update_train_graph(selected_run):
            if selected_run:
                dff = df.copy()
                # filter to get only from the selected species
                dff = dff[dff["species"] == selected_run]
                # select only the columns we want
                cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
                dff = dff[cols]
                # take the average of each column
                dff = dff.mean()
                # plot a bar for each column and its height is the mean
                fig = px.bar(dff)
            else:
                fig = {}
            return fig

        @app.callback(
            Output("select-run", "options"),
            Input("table", "rowData"),
        )
        def update_selectable_runs(row_data):
            if row_data:
                dff = pd.DataFrame(row_data)
                options = sorted(dff["species"].unique())
            else:
                options = []
            return options

        @app.callback(
            Output("table", "rowData"),
            Input("activefilters", "data"),
        )
        def udpate_table(data):
            if data:
                dff = df.copy()
                for col in data:
                    if data[col]:
                        rng = data[col][0]
                        if isinstance(rng[0], list):
                            # if multiple choices combine df
                            dff3 = pd.DataFrame(columns=df.columns)
                            for i in rng:
                                dff2 = dff[dff[col].between(i[0], i[1])]
                                dff3 = pd.concat([dff3, dff2])
                            dff = dff3
                        else:
                            # if one choice
                            dff = dff[dff[col].between(rng[0], rng[1])]
                return dff.to_dict("records")
            return df.to_dict("records")

        @app.callback(
            Output("activefilters", "data"),
            Input("my-graph", "restyleData"),
        )
        def updateFilters(data):
            if data:
                key = list(data[0].keys())[0]
                col = dims[int(key.split("[")[1].split("]")[0])]
                newData = Patch()
                newData[col] = data[0][key]
                return newData
            return {}
        
        return app


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
        ("graphgym.train.homophily_regularization", False, False),
        ("graphgym.train.homophily_regularization_gt_weight", False, False),
    ],
    "WeightedSANTransformer": [
        ("graphgym.optim.base_lr", False, True),
        ("graphgym.optim.weight_decay", False, True),
        ("graphgym.gnn.dim_inner", False, False),
        ("graphgym.gt.n_heads", True, False),
        ("graphgym.gt.layers", True, False),
        ("graphgym.gt.dropout", False, False),
        ("graphgym.gt.wsan_add_dummy_edges", True, False),
        ("graphgym.gt.gamma", False, True),
        ("graphgym.gt.attn.clamp", True, False),
        ("graphgym.posenc_WLapPE.n_heads", True, False),
        ("graphgym.posenc_WLapPE.layers", True, False),
        ("graphgym.posenc_WLapPE.dim_pe", False, False),
        ("graphgym.posenc_WLapPE.eigen.max_freqs", False, False),
        ("graphgym.gnn.head", True, False),
        ("graphgym.gnn.layers_post_mp", True, False),
        ("graphgym.train.homophily_regularization", False, False),
        ("graphgym.train.homophily_regularization_gt_weight", False, False),
    ],
}


parser = argparse.ArgumentParser(description='Processes the results of hyperparameter search.')
parser.add_argument("-c", "--collection")
parser.add_argument("-d", "--dataset")
parser.add_argument("-m", "--model")
parser.add_argument("-k", "--k-best")
parser.add_argument("-s", "--single-plots", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    results_path = f"hs_results/{args.dataset}/{args.model}/{args.collection}"
    filter_dict = None  # not implemented for argparse... but can manually change here
    app = main(
        collection=args.collection,
        configs_all_info=hyperparameters[args.model],
        k=int(args.k_best),
        results_path=results_path,
        filter_dict=filter_dict,
        model=args.model,
        old_plotting=args.single_plots,
    )
    if app is not None:
        app.run(debug=True)
