import seml
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


def clean_path(results_path: str):
    results_path = Path(results_path)
    if results_path.exists():
        for file in results_path.glob('*.jpg'):
            file.unlink()
    else:
        results_path.mkdir()
    result_log_file = results_path / "hs_result_analysis.txt"
    return results_path, result_log_file


def get_split_metric_results(results, split: str, metric: str, agg: str):
    if agg == "max":
        agg_fun = max
    elif agg == "min":
        agg_fun = min
    else:
        raise NotImplementedError(f"agg={agg} is not implemented")
    split_results = [result["result"]["train"][split] for result in results]
    all_values = [[e[metric] for e in r] for r in split_results]
    agg_values = [agg_fun(acc) for acc in all_values]
    epochs_of_agg_values = [np.arange(len(acc))[np.array(acc) == max_acc].tolist() for acc, max_acc in zip(all_values, agg_values)]
    agg_value = max(agg_values)
    ind_of_agg_value = np.arange(len(results))[np.array(agg_values) == agg_value].tolist()
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



def main(collection: str, configs_all_info: list[tuple[str, bool, bool]], k: int, results_path: str, filter_dict):
    configs = [c[0] for c in configs_all_info]
    results = seml.get_results(collection, filter_dict=filter_dict)

    for result in results:
        if result["config"]["dims_per_head"] > 0 and result["config"]["graphgym"]["gnn"]["dim_inner"] == 0:
            dim_inner = result["config"]["dims_per_head"] * result["config"]["graphgym"]["graphormer"]["num_heads"]
            result["config"]["graphgym"]["graphormer"]["embed_dim"] = dim_inner
            result["config"]["graphgym"]["gnn"]["dim_inner"] = dim_inner
        
    results_path, result_log_file = clean_path(results_path)

    # train
    train_acc = get_split_metric_results(results, "train", "accuracy", "max")
    train_loss = get_split_metric_results(results, "train", "loss", "min")

    # val:
    val_acc = get_split_metric_results(results, "val", "accuracy", "max")
    best_val_epochs = [result["result"]["train"]["best_val_epoch"] for result in results]
    for epochs_max_acc, best_val_epoch in zip(val_acc["epochs_of_max_values"], best_val_epochs):
        assert best_val_epoch in epochs_max_acc
    best_val_acc_ind = get_highest_k_indices(val_acc["max_values"], k)
    val_loss = get_split_metric_results(results, "val", "loss", "min")


    # test:
    test_acc = get_split_metric_results(results, "test", "accuracy", "max")
    best_test_acc_ind = get_highest_k_indices(test_acc["max_values"], k)
    test_loss = get_split_metric_results(results, "test", "loss", "min")

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

    # write results into file
    
    with open(result_log_file, "w") as f:

        f.write("\nHighest val accurracies:")
        for i in best_val_acc_ind:
            f.write(f"\n\tval acc: {val_acc['max_values'][i]}, by experiment: {i}")

        f.write("\nHighest test accurracies:")
        for i in best_test_acc_ind:
            f.write(f"\n\tval acc: {test_acc['max_values'][i]}, by experiment: {i}")

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
    
    # save plots

    fig, ((ax_acc), (ax_loss)) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    ax_acc.set_title("Accuracy")
    ax_loss.set_title("Loss")
    for i in best_experiments:
        x = np.arange(len(val_acc['all_values'][i]))
        ax_acc.plot(x, train_acc['all_values'][i], "-g", label='train')
        ax_acc.plot(x, val_acc['all_values'][i], "-b", label='val')
        ax_acc.plot(x, test_acc['all_values'][i], "-r", label='test')
        ax_acc.legend()

        ax_loss.plot(x, train_loss['all_values'][i], "-g", label='train')
        ax_loss.plot(x, val_loss['all_values'][i], "-b", label='val')
        ax_loss.plot(x, test_loss['all_values'][i], "-r", label='test')
        ax_loss.legend()

        fig.savefig(results_path / f"{i}.jpg")
        plt.close(fig)
        ax_acc.clear()
        ax_loss.clear()

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
            x = np.array(x)
            values = np.unique(x)
            val_dataset = []
            test_dataset = []
            for v in values:
                val_dataset.append(np.array(val_acc["max_values"])[x == v])
                test_dataset.append(np.array(test_acc["max_values"])[x == v])
            ax_val.violinplot(val_dataset, values, showmeans=True, showmedians=True)
            ax_test.violinplot(test_dataset, values, showmeans=True, showmedians=True)
            x = x + 0.5 * (np.random.rand(len(x)) - 0.5)
            ax_val.scatter(x, val_acc["max_values"], s=12, c="g")
            ax_test.scatter(x, test_acc["max_values"], s=12, c="g")
        else:
            ax_val.scatter(x, val_acc["max_values"])
            ax_test.scatter(x, test_acc["max_values"])
            if log:
                ax_val.set_xscale('log')
                ax_test.set_xscale('log')

        fig.savefig(results_path / f"conf_{conf}.jpg")
        plt.close(fig)


if __name__ == "__main__":
    collection_name = "hyp_graphormer"
    configs_all_info = [  # name, is_discrete, is_log
        ("graphgym.optim.base_lr", False, True),
        ("graphgym.optim.weight_decay", False, True),
        ("graphgym.graphormer.num_heads", True, False),
        ("graphgym.gnn.dim_inner", False, False),
        ("graphgym.graphormer.num_layers", True, False),
        ("graphgym.posenc_GraphormerBias.num_spatial_types", True, False),
        ("graphgym.posenc_GraphormerBias.num_in_degrees", False, False),
        ("graphgym.gnn.layers_post_mp", True, False),
        ("graphgym.optim.num_warmup_epochs", True, False),
    ]
    k = 6
    results_path = "hs_results_analysis"
    filter_dict = None
    main(collection=collection_name, configs_all_info=configs_all_info, k=k, results_path=results_path, filter_dict=filter_dict)
