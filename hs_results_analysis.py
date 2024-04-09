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


def clean_path(results_path: str):
    results_path = Path(results_path)
    if results_path.exists():
        for file in results_path.glob('*.jpg'):
            file.unlink()
    else:
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
    agg_values = [agg_fun(acc) for acc in all_values]
    epochs_of_agg_values = [np.arange(len(acc))[np.array(acc) == max_acc].tolist() for acc, max_acc in zip(all_values, agg_values)]
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
    old_logging = "train" in results[0]["result"]
    if old_logging:
        training_results = [result["result"]["train"] for result in results]
    else:
        training_results = [result["result"]["training"] for result in results]
    train_acc = get_split_metric_results(training_results, "train", "accuracy", "max", old_logging)
    train_loss = get_split_metric_results(training_results, "train", "loss", "min", old_logging)
    # val:
    val_acc = get_split_metric_results(training_results, "val", "accuracy", "max", old_logging)
    best_val_epochs = [r["best_val_epoch"] for r in training_results]
    for epochs_max_acc, best_val_epoch in zip(val_acc["epochs_of_max_values"], best_val_epochs):
        assert best_val_epoch in epochs_max_acc
    best_val_acc_ind = get_highest_k_indices(val_acc["max_values"], k)
    val_loss = get_split_metric_results(training_results, "val", "loss", "min", old_logging)
    # test:
    test_acc = get_split_metric_results(training_results, "test", "accuracy", "max", old_logging)
    best_test_acc_ind = get_highest_k_indices(test_acc["max_values"], k)
    test_loss = get_split_metric_results(training_results, "test", "loss", "min", old_logging)
    # looking at interesting epochs of best experiments
    best_experiments = sorted(set(best_val_acc_ind + best_test_acc_ind))
    best_epochs = [
        sorted(
            set(
                val_acc["epochs_of_max_values"][i]
                + val_loss["epochs_of_min_values"][i]
                + test_acc["epochs_of_max_values"][i]
                + test_loss["epochs_of_min_values"][i]
            )
        ) 
        for i in best_experiments
    ]
    best_val_with_test_acc = []
    for i in best_val_acc_ind:
        best_val_with_test_acc.append((val_acc["max_values"][i], test_acc["all_values"][i][best_val_epochs[i]], i))
    best_val_with_test_acc = sorted(best_val_with_test_acc, reverse=True)
    return (
        best_experiments,
        best_epochs,
        best_val_with_test_acc,
        best_test_acc_ind,
        train_acc,
        val_acc,
        test_acc,
        train_loss,
        val_loss,
        test_loss,
    )


def write_results_into_file(
    result_log_file,
    results,
    configs,
    best_experiments,
    best_epochs,
    best_val_with_test_acc,
    best_test_acc_ind,
    train_acc,
    val_acc,
    test_acc,
    train_loss,
    val_loss,
    test_loss,
):
    with open(result_log_file, "w") as f:
        f.write("\nHighest val accurracies:")
        for (v_a, t_a, i) in best_val_with_test_acc:
            f.write(f"\n\tval acc: {v_a}, with {t_a} test acc, by experiment: {i}")
        f.write("\nHighest test accurracies:")
        for i in best_test_acc_ind:
            f.write(f"\n\ttest acc: {test_acc['max_values'][i]}, by experiment: {i}")
        f.write("\nBest experiments with best epochs:")
        for i, epochs in zip(best_experiments, best_epochs):
            f.write(f"\nexperiment: {i}")
            f.write(f"\n\tbest val acc: {val_acc['max_values'][i]}, at epochs: {val_acc['epochs_of_max_values'][i]}")
            f.write(f"\n\tbest test acc: {test_acc['max_values'][i]}, at epochs: {test_acc['epochs_of_max_values'][i]}")
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
                f.write(f"\n\t\t\ttrain acc: {train_acc['all_values'][i][e]}")
                f.write(f"\n\t\t\tval acc: {val_acc['all_values'][i][e]}")
                f.write(f"\n\t\t\ttest acc: {test_acc['all_values'][i][e]}")
                f.write(f"\n\t\t\ttrain loss: {train_loss['all_values'][i][e]}")
                f.write(f"\n\t\t\tval loss: {val_loss['all_values'][i][e]}")
                f.write(f"\n\t\t\ttest loss: {test_loss['all_values'][i][e]}")
            f.write("\n")


def save_single_plots(
    results,
    results_path,
    best_experiments,
    train_acc,
    val_acc,
    test_acc,
    train_loss,
    val_loss,
    test_loss,
    configs_all_info,
):
    fig, ((ax_acc), (ax_loss)) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    for i in best_experiments:
        ax_acc.set_title("Accuracy")
        ax_loss.set_title("Loss")
        x = np.arange(len(val_acc['all_values'][i]))
        ax_acc.plot(x, train_acc['all_values'][i], "-g", label='train')
        ax_acc.plot(x, val_acc['all_values'][i], "-b", label='val')
        ax_acc.plot(x, test_acc['all_values'][i], "-r", label='test')
        ax_acc.legend()
        ax_loss.plot(x, train_loss['all_values'][i], "-g", label='train')
        ax_loss.plot(x, val_loss['all_values'][i], "-b", label='val')
        ax_loss.plot(x, test_loss['all_values'][i], "-r", label='test')
        ax_loss.legend()
        fig.savefig(results_path / f"{i}.png")
        plt.close(fig)
        ax_acc.clear()
        ax_loss.clear()

    # filter results for lr < 1.3e-3
    # mask = [result["config"]["graphgym"]["optim"]["base_lr"] < 1.3e-3 for result in results]
    # results = [result for result, m in zip(results, mask) if m]
    # val_max_acc = [v for v, m in zip(val_acc["max_values"], mask) if m]
    # test_max_acc = [v for v, m in zip(test_acc["max_values"], mask) if m]
    val_max_acc = val_acc["max_values"]
    test_max_acc = test_acc["max_values"]

    for (conf, discrete, log) in configs_all_info:
        fig, ((ax_val), (ax_test)) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
        ax_val.set_title(f"max val accuracy / {conf}")
        ax_test.set_title(f"max test accuracy / {conf}")
        x = []
        for result in results:
            v = result["config"]
            conf_split = conf.split(".")
            for k in conf_split:
                v = v[k]
            x.append(v)
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
                val_dataset.append(np.array(val_max_acc)[x == v])
                test_dataset.append(np.array(test_max_acc)[x == v])
            ax_val.violinplot(val_dataset, values, showmeans=True, showmedians=True)
            ax_test.violinplot(test_dataset, values, showmeans=True, showmedians=True)
            x = x + 0.5 * (np.random.rand(len(x)) - 0.5)
            ax_val.scatter(x, val_max_acc, s=12, c="g")
            ax_test.scatter(x, test_max_acc, s=12, c="g")
            ax_val.set_xticks(values, labels=tick_labels)
            ax_test.set_xticks(values, labels=tick_labels)
        else:
            ax_val.scatter(x, val_max_acc)
            ax_test.scatter(x, test_max_acc)
            if log:
                ax_val.set_xscale('log')
                ax_test.set_xscale('log')
        fig.savefig(results_path / f"conf_{conf}.png")
        plt.close(fig)


def main(
    collection: str,
    configs_all_info: list[tuple[str, bool, bool]],
    k: int,
    results_path: str,
    filter_dict,
    model: str,
    old_plotting: bool = False,
):
    configs = [c[0] for c in configs_all_info]
    results = seml.get_results(collection, filter_dict=filter_dict)
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
    (  # process results
        best_experiments,
        best_epochs,
        best_val_with_test_acc,
        best_test_acc_ind,
        train_acc,
        val_acc,
        test_acc,
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
        best_experiments,
        best_epochs,
        best_val_with_test_acc,
        best_test_acc_ind,
        train_acc,
        val_acc,
        test_acc,
        train_loss,
        val_loss,
        test_loss,
    )
    # plots
    if old_plotting:
        save_single_plots(
            results,
            results_path,
            best_experiments,
            train_acc,
            val_acc,
            test_acc,
            train_loss,
            val_loss,
            test_loss,
            configs_all_info,
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
        ("graphgym.train.homophily_regularization", True, False),
        ("graphgym.train.homophily_regularization_gt_weight", True, False),
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
        ("graphgym.train.homophily_regularization", True, False),
        ("graphgym.train.homophily_regularization_gt_weight", True, False),
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
        ("graphgym.posenc_WLapPE.dim_pe", True, False),
        ("graphgym.posenc_WLapPE.eigen.max_freqs", True, False),
        ("graphgym.gnn.head", True, False),
        ("graphgym.gnn.layers_post_mp", True, False),
    ],
}


if __name__ == "__main__":

    collection_name = "hs_grt_coraml"
    dataset = "cora_ml"  # "upfd_pol_bert", "upfd_gos_bert", "cora_ml"
    model = "GRIT"  # "Graphormer", "GCN", "GRIT", "WeightedSANTransformer", "GAT"
    single_plots = True
    log_k_best = 5

    results_path = f"hs_results/{dataset}/{model}"
    filter_dict = None
    app = main(
        collection=collection_name,
        configs_all_info=hyperparameters[model],
        k=log_k_best,
        results_path=results_path,
        filter_dict=filter_dict,
        model=model,
        old_plotting=single_plots,
    )
    if app is not None:
        app.run(debug=True)
