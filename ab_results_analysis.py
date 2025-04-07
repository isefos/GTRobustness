import seml
import numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import pandas as pd
import argparse
import shutil
from collections import defaultdict


datasets = {
    "CLUSTER_as": {"format":"PyG-GNNBenchmarkDataset", "name": "CLUSTER"},
    "CLUSTER_cs": {"format":"PyG-GNNBenchmarkDataset", "name": "CLUSTER"},
    "CoraML-RUT": {"format": "PyG-RobustnessUnitTest", "name": "cora_ml"},
    "Citeseer-RUT": {"format": "PyG-RobustnessUnitTest", "name": "citeseer"},
    "UPFD_gos_bert": {"format": "PyG-UPFD", "name": "gossipcop-bert"},
    "UPFD_pol_bert": {"format": "PyG-UPFD", "name": "politifact-bert"},
    "reddit_threads": {"format": "PyG-TUDataset", "name": "reddit_threads"},
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


ablation_cols_graph = [
    "avg_attack_success_rate",
    "avg_correct_clean",
    "avg_correct_pert",
    "avg_margin_clean",
    "avg_margin_pert",
]
ablation_cols_node = [
    "avg_attack_success_rate",
    "avg_correct_acc_clean",
    "avg_correct_acc_pert",
    "avg_margin_mean_clean",
    "avg_margin_mean_pert",
]

random_cols_graph = [
    "avg_attack_success_rate_random",
    "avg_correct_clean",
    "avg_correct_pert_random",
    "avg_margin_clean",
    "avg_margin_pert_random",
]
random_cols_node = [
    "avg_attack_success_rate_random",
    "avg_correct_acc_clean",
    "avg_correct_acc_pert_random",
    "avg_margin_mean_clean",
    "avg_margin_mean_pert_random",
]


ablations_settings = {
    "Graphormer": {
        "w_deg": ["config.graphgym.attack.Graphormer.use_weighted_degrees"],
        "sp_find_w": ["config.graphgym.attack.Graphormer.sp_find_weighted"],
        "sp_use_w": ["config.graphgym.attack.Graphormer.sp_use_weighted"],
        "sp_grad_w": ["config.graphgym.attack.Graphormer.sp_use_gradient"],
        "sp_weight_fun": ["config.graphgym.attack.Graphormer.weight_function"],
        # TODO: add for node injection
        "node_prob": ["config.graphgym.attack.node_prob_enable"],
    },
    "SAN": {
        "cont_att": ["config.graphgym.attack.SAN.wsan_add_partially_fake", "config.graphgym.attack.SAN.add_partially_fake_edges"],
        "cont_att_grad": ["config.graphgym.attack.SAN.partially_fake_edge_grad"],
        "pert_grad": ["config.graphgym.attack.SAN.enable_pert_grad"],
        "BPDA": ["config.graphgym.attack.SAN.pert_BPDA"],
        "BPDA_match_signs": ["config.graphgym.attack.SAN.match_true_signs"],
        "zero_first": ["config.graphgym.attack.SAN.set_first_pert_zero"],
        "backprop": ["config.graphgym.attack.SAN.enable_eig_backprop"],
        # TODO: add for node injection
        "node_prob": ["config.graphgym.attack.node_prob_enable"],
    },
    "GRIT": {
        "rrwp_grad": ["config.graphgym.attack.GRIT.grad_RRWP"],
        "edge_w": ["config.graphgym.attack.GRIT.dummy_edge_weighting"],
        "deg_grad": ["config.graphgym.attack.GRIT.grad_degree"],
        # TODO: add for node injection
        "node_prob": ["config.graphgym.attack.node_prob_enable"],
    },
    "GPS": {
        "grad_MPNN": ["config.graphgym.attack.GPS.grad_MPNN"],
        "pert_appr": ["config.graphgym.attack.SAN.enable_pert_grad"],
        "backprop": ["config.graphgym.attack.SAN.enable_eig_backprop"],
        # TODO: add for node injection
        "node_prob": ["config.graphgym.attack.node_prob_enable"],
    },
}


random_results = {
    "CLUSTER_as": {
        "Graphormer": "a_gph_cluster_as_prel",
        "GRIT": "a_grt_cluster_as_prel",
        "SAN": "a_san_cluster_as_prel",
    },
    "CLUSTER_cs": {
        "Graphormer": "a_gph_cluster_cs_prel",
        "GRIT": "a_grt_cluster_cs_prel",
        "SAN": "a_san_cluster_cs_prel",
    },
    "UPFD_gos_bert": {
        "Graphormer": "a_gph_upfd_gos_bert_prel",
        "GRIT": "a_grt_upfd_gos_bert_prel",
        "SAN": "a_san_upfd_gos_bert_new",
        "GPS": "a_gps_upfd_gos_bert",
    },
    "UPFD_pol_bert": {
        "Graphormer": "a_gph_upfd_pol_bert_prel",
        "GRIT": "a_grt_upfd_pol_bert_prel",
        "SAN": "a_san_upfd_pol_bert_new",
    },
}


def get_abl_setting(result, possible_keys):
    value = None
    value_updated = False
    for key in possible_keys:
        key_levels = key.split(".")
        abl_value = result
        aborted = False
        for level_key in key_levels:
            abl_value = abl_value.get(level_key, None)
            if abl_value is None:
                aborted = True
                break
        if aborted:
            continue
        value_found = abl_value
        if value is None:
            value = value_found
            value_updated = True
            continue
        assert value_found == value, f"Conflicting configs for {possible_keys}"
    assert value_updated, f"No config for {possible_keys} found"
    return value


def clean_path(results_path: str):
    results_path = Path(results_path)
    if results_path.exists():
        shutil.rmtree(results_path)
    results_path.mkdir(parents=True)
    general_info_file = results_path / "runs_infos.txt"
    seed_dir = results_path / "results"
    seed_dir.mkdir()
    return results_path, general_info_file, seed_dir


def write_info_file(info_file, run_ids, num_params, extras, seeds_graphgym, seeds_seml, run_dirs, budgets, ablation_cfg):
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
            for k, v in ablation_cfg[i].items():
                f.write(f"\n\t{k}: {v}")
            f.write("\n")


def write_results(
    seed_dir,
    results,
    seeds_graphgym,
    budgets,
    result_cols,
    random_cols,
    ablation_cfg,
    ablation_settings,
    random_path,
):
    seed_results = {}
    budget = None
    df_dir = seed_dir / "all"
    df_dir.mkdir()
    for i, seed in enumerate(seeds_graphgym):
        if budget is None:
            budget = budgets[i]
        else:
            assert budget == budgets[i], "For ablations the same budget should be used!"
        if seed not in seed_results:
            seed_results[seed] = defaultdict(list)
        for col in result_cols:
            seed_results[seed][col].append(results[i]["result"]["attack"]["avg"][col])
        for col in ablation_settings:
            seed_results[seed][col].append(ablation_cfg[i][col])
    seed_dataframes = {}
    for seed, results in seed_results.items():
        df = pd.DataFrame(results)
        seed_dataframes[seed] = df
        df.to_csv(df_dir / f"seed_{seed}.csv")

    df_random_mean = pd.read_csv(random_path / "agg_mean.csv")
    df_random_mean = df_random_mean[df_random_mean["budget"] == budget][random_cols]
    df_random_mean.columns = [c.split("_random")[0] for c in df_random_mean.columns]
    df_random_std = pd.read_csv(random_path / "agg_std.csv")
    df_random_std = df_random_std[df_random_std["budget"] == budget][random_cols]
    df_random_std.columns = [c.split("_random")[0] for c in df_random_std.columns]
    
    df_agg = pd.concat(list(seed_dataframes.values()))
    df_agg_mean = df_agg.groupby(list(ablation_settings.keys()), as_index=False).mean()
    df_agg_std = df_agg.groupby(list(ablation_settings.keys()), as_index=False).std()

    df_all_mean = pd.concat((df_agg_mean, df_random_mean), ignore_index=True).fillna("random")
    df_all_std = pd.concat((df_agg_std, df_random_std), ignore_index=True).fillna("random")

    df_all_mean.to_csv(seed_dir / f"agg_mean.csv")
    df_all_std.to_csv(seed_dir / f"agg_std.csv")


def get_collection_results(collection, filter_dict, model):
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
    budgets = [result["config"]["graphgym"]["attack"]["e_budget"] for result in results]
    ablation_cfg = [
        {name: get_abl_setting(r, keys) for name, keys in ablations_settings[model].items()}
        for r in results
    ]
    return results, run_ids, extras, seeds_graphgym, seeds_seml, run_dirs, num_params, budgets, ablation_cfg


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


def main(
    collection: str,
    results_path: str,
    filter_dict,
    dataset: str,
    model: str,
    name: str,
):
    results_path, info_file, seed_dir = clean_path(results_path)
    (
        results,
        run_ids,
        extras,
        seeds_graphgym,
        seeds_seml,
        run_dirs,
        num_params,
        budgets,
        ablation_cfg,
    ) = get_collection_results(collection, filter_dict, model)
    write_info_file(info_file, run_ids, num_params, extras, seeds_graphgym, seeds_seml, run_dirs, budgets, ablation_cfg)
    pred_level = results[0]["config"]["graphgym"]["attack"]["prediction_level"]
    if pred_level == "graph":
        ablation_cols = ablation_cols_graph
        random_cols = random_cols_graph
    elif pred_level == "node":
        ablation_cols = ablation_cols_node
        random_cols = random_cols_node
    else:
        raise ValueError(f"Unknown prediction level: `{pred_level}`")
    for r in results:
        check_correct_result(r, dataset, model, pred_level)

    random_path = Path("results_a") / dataset / model / random_results[dataset][model] / name / "results"
    assert random_path.is_dir(), f"No results for random attack found in {random_path}"

    # write results into file
    write_results(
        seed_dir,
        results,
        seeds_graphgym,
        budgets,
        ablation_cols,
        random_cols,
        ablation_cfg,
        ablation_settings=ablations_settings[model],
        random_path=random_path,
    )


parser = argparse.ArgumentParser(description='Processes the results of attack.')
parser.add_argument("-c", "--collection")
parser.add_argument("-d", "--dataset")
parser.add_argument("-m", "--model")
parser.add_argument("-n", "--name-pretrained", default="0")


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.dataset in datasets
    results_path = f"results_ab/{args.dataset}/{args.model}/{args.collection}"
    # not implemented for argparse... but can manually change here
    filter_dict = None  # {"config.graphgym.attack.cluster_sampling": True}
    main(
        collection=args.collection,
        results_path=results_path,
        filter_dict=filter_dict,
        dataset=args.dataset,
        model=args.model,
        name=args.name_pretrained,
    )
