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



def main(collection: str, configs: list[str], k: int, results_path: str):
    results = seml.get_results(collection)
    results_path, result_log_file = clean_path(results_path)

    with open(result_log_file, "w") as f:
        # val:
        val_acc = get_split_metric_results(results, "val", "accuracy", "max")
        best_val_epochs = [result["result"]["train"]["best_val_epoch"] for result in results]
        for epochs_max_acc, best_val_epoch in zip(val_acc["epochs_of_max_values"], best_val_epochs):
            assert best_val_epoch in epochs_max_acc

        best_val_acc_ind = get_highest_k_indices(val_acc["max_values"], k)
        f.write("\nHighest val accurracies:")
        for i in best_val_acc_ind:
            f.write(f"\n\tval acc: {val_acc['max_values'][i]}, by experiment: {i}")

        val_loss = get_split_metric_results(results, "val", "loss", "min")


        # test:
        test_acc = get_split_metric_results(results, "test", "accuracy", "max")

        best_test_acc_ind = get_highest_k_indices(test_acc["max_values"], k)
        f.write("\nHighest test accurracies:")
        for i in best_test_acc_ind:
            f.write(f"\n\tval acc: {test_acc['max_values'][i]}, by experiment: {i}")

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

        fig, ((ax_acc), (ax_loss)) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
        ax_acc.set_title("Accuracy")
        ax_loss.set_title("Loss")

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
                f.write(f"\n\t\t\tval acc: {val_acc['all_values'][i][e]}")
                f.write(f"\n\t\t\ttest acc: {test_acc['all_values'][i][e]}")
                f.write(f"\n\t\t\tval loss: {val_loss['all_values'][i][e]}")
                f.write(f"\n\t\t\ttest loss: {test_loss['all_values'][i][e]}")

            x = np.arange(len(val_acc['all_values'][i]))
            ax_acc.plot(x, val_acc['all_values'][i], "-b", label='val')
            ax_acc.plot(x, test_acc['all_values'][i], "-r", label='test')
            ax_acc.legend()

            ax_loss.plot(x, val_loss['all_values'][i], "-b", label='val')
            ax_loss.plot(x, test_loss['all_values'][i], "-r", label='test')
            ax_loss.legend()

            fig.savefig(results_path / f"{i}.jpg")
            plt.close(fig)
            ax_acc.clear()
            ax_loss.clear()

            f.write("\n")


if __name__ == "__main__":
    collection_name = "hyp_search"
    configs = [
        "graphgym.posenc_GraphormerBias.num_spatial_types",
        "graphgym.posenc_GraphormerBias.num_in_degrees",
        "graphgym.graphormer.num_layers",
        "graphgym.graphormer.num_heads",
        "graphgym.graphormer.embed_dim",
        "graphgym.gnn.dim_inner",
        "graphgym.gnn.head",
        "graphgym.gnn.layers_pre_mp",
        "graphgym.gnn.layers_post_mp",
        "graphgym.optim.base_lr",
        "graphgym.optim.weight_decay",
        "graphgym.optim.num_warmup_epochs",
    ]
    k = 6
    results_path = "hs_results_analysis"
    main(collection=collection_name, configs=configs, k=k, results_path=results_path)
