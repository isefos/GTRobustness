import argparse
import os
import shutil
import seml
import yaml


attack_configs = {
    "attack": [
        "split",
        "prediction_level",
        "remove_isolated_components",
        "is_undirected",
        "root_node_idx",
        "cluster_sampling",
        {
            "node_injection": [
                "enable",
                "from_train",
                "from_val",
                "from_test",
                "allow_existing_graph_pert",
                "include_root_nodes",
                "sample_only_trees",
            ]
        },
    ],
    "dataset": [
        "format",
        "name",
        "split_index",
        "split_mode",
        "task",
        "task_type",
        "to_undirected",
        "transductive",
    ],
}


def _add_subkeys_(d_from: dict, d_to: dict, subkeys: list):
    for k in subkeys:
        if isinstance(k, dict):
            for kk, ssubkeys in k.items():
                d_to[kk] = {}
                current_cfg = d_from[kk]
                _add_subkeys_(current_cfg, d_to[kk], ssubkeys)
        else:
            v_from = d_from[k]
            if isinstance(v_from, dict):
                d_to[k] = {}
                if "enable" in v_from and not v_from["enable"]:
                    d_to[k]["enable"] = False
                else:
                    _add_subkeys_(v_from, d_to[k], list(v_from.keys()))
            else:
                d_to[k] = v_from


def get_attack_configs(graphgym_configs: dict) -> dict:
    attack_cfg = {}
    for key, subkeys in attack_configs.items():
        attack_cfg[key] = {}
        current_cfg = graphgym_configs[key]
        _add_subkeys_(current_cfg, attack_cfg[key], subkeys)
    return attack_cfg


def check_equal_configs(configs_given: dict, currect_configs: dict) -> None:
    for k, value_given in configs_given.items():
        value_current = currect_configs[k]
        if isinstance(value_given, dict):
            assert isinstance(value_current, dict)
            check_equal_configs(value_given, value_current)
        else:
            assert value_given == currect_configs[k]


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
    "GPS-GCN": {"type": set(["GPSModel"]), "gnn_layer_type": None},
}


def get_run_dirs_and_attack_cfg(collection: str, filter_dict, model: str, dataset: str) -> tuple[list[str], dict]:
    results = seml.get_results(collection, ['config', 'result'], filter_dict=filter_dict)
    run_dirs = []
    attack_cfg = None
    for res in results:
        if attack_cfg is None:
            attack_cfg = get_attack_configs(res["config"]["graphgym"])
        else:
            check_equal_configs(attack_cfg, get_attack_configs(res["config"]["graphgym"]))
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

        run_dirs.append(res["result"].get("run_dir"))
    return run_dirs, attack_cfg


def collect_perturbations_from_run_dirs(collection, filter_dict, model, dataset, dest_base_dir, add_to_name):
    base_dir = os.path.join(dest_base_dir, dataset, model + add_to_name)
    os.makedirs(base_dir, exist_ok=True)
    run_dirs, attack_cfg = get_run_dirs_and_attack_cfg(collection, filter_dict, model, dataset)

    cfg_file = os.path.join(base_dir, "attack_configs.yaml")
    with open(cfg_file, "w") as f:
        yaml.dump(attack_cfg, f)

    for r_dir in run_dirs:
        # Extract the seed from the directory name (e.g., "sN-...")
        subdir_name = r_dir.split("/")[-1]
        seed = int(subdir_name.split("-")[0][1:])
        dest_dir = os.path.join(base_dir, f"s{seed}")
        os.makedirs(dest_dir, exist_ok=True)

        json_files = [f for f in os.listdir(r_dir) if f.endswith(".json")]
        for file in json_files:
            src_file = os.path.join(r_dir, file)
            dest_file = os.path.join(dest_dir, file)
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Find the saved perturbation files from the corresponding "
            "runs and save them in the provided loacation"
        )
    )
    parser.add_argument("-c", "--collection")
    parser.add_argument("-d", "--dataset")
    parser.add_argument("-m", "--model")
    parser.add_argument("-l", "--copy-location")
    parser.add_argument("-a", "--add-name", default="")
    args = parser.parse_args()

    filter_dict = None

    collect_perturbations_from_run_dirs(
        collection=args.collection,
        filter_dict=filter_dict,
        model=args.model,
        dataset=args.dataset,
        dest_base_dir=args.copy_location,
        add_to_name=args.add_name,
    )
