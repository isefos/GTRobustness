import argparse
import shutil
from pathlib import Path
import yaml


datasets = {
    "CLUSTER": {"format":"PyG-GNNBenchmarkDataset", "name": "CLUSTER"},
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
    "NettackGCN": {"type": set(["NettackGCN"]), "gnn_layer_type": set(["gcnconvweighted", "gcnconv"])},
    "GAT": {"type": set(["gnn"]), "gnn_layer_type": set(["gatconvweighted", "gatconv"])},
    "GATv2": {"type": set(["gnn"]), "gnn_layer_type": set(["gatv2convweighted", "gatv2conv"])},
    "GPS": {"type": set(["GPSModel"]), "gnn_layer_type": None},
    "GPS-GCN": {"type": set(["GPSModel"]), "gnn_layer_type": None},
    "Polynormer": {"type": set(["WeightedPolynormer"]), "gnn_layer_type": None},
}


def check_cfg(cfg_from_file, model, dataset):
    df = cfg_from_file["dataset"]["format"]
    dfg = datasets[dataset]["format"]
    dn = cfg_from_file["dataset"]["name"]
    dng = datasets[dataset]["name"]
    assert df == dfg, (f"Dataset format was given to be `{dfg}`, but encountered `{df}`.")
    assert dn == dng, (f"Dataset name was given to be `{dng}`, but encountered `{dn}`.")

    mt = cfg_from_file["model"]["type"]
    mtg = models[model]["type"]
    assert mt in mtg, (f"Model was given to be in {mtg}, but encountered `{mt}`.")
    mlg = models[model]["gnn_layer_type"]
    if mlg is not None:
        ml = cfg_from_file["gnn"]["layer_type"]
        assert ml in mlg, (f"Model layer was given to be in {mlg}, but encountered `{ml}`.")


def collect_model(
    source_dir,
    model,
    dataset,
    copy_location,
    adversarial,
    name,
):
    source_dir = Path(source_dir)
    cfg_file = source_dir / "config.yaml"
    assert cfg_file.is_file(), "No `config.yaml` file in given source directory!"
    cfg_seml_file = source_dir / "configs_from_seml.yaml"
    assert cfg_seml_file.is_file(), "No `configs_from_seml.yaml` file in given source directory!"

    with open(cfg_seml_file, "r") as f:
        cfg_from_file = yaml.safe_load(f)
    check_cfg(cfg_from_file, model, dataset)

    ckpt_dir = source_dir / "ckpt"
    assert ckpt_dir.is_dir(), "No `ckpt` subdirectory in given source directory!"

    best_ckpt_file = None
    for ckpt_file in ckpt_dir.iterdir():
        assert best_ckpt_file is None, "More than one file in `ckpt` subdirectory, not sure which is best!"
        best_ckpt_file = ckpt_file        

    base_dir = Path(copy_location) / model / dataset
    if adversarial:
        base_dir = base_dir / "adv"
    destination_dir: Path = base_dir / name
    destination_dir.mkdir(parents=True, exist_ok=True)

    for s_file, d_name in zip(
        [cfg_file, cfg_seml_file, best_ckpt_file],
        ["config.yaml", "config_seml.yaml", "best.ckpt"]
    ):
        d_file = destination_dir / d_name
        shutil.copy(s_file, d_file)
        print(f"Copied {s_file} to {d_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Copy model checkpoint and configuration to provided loacation"
        )
    )
    parser.add_argument("-s", "--source-dir")
    parser.add_argument("-d", "--dataset")
    parser.add_argument("-m", "--model")
    parser.add_argument("-l", "--copy-location")
    parser.add_argument("-a", "--adversarial", action="store_true")
    parser.add_argument("-n", "--name")
    args = parser.parse_args()

    collect_model(
        source_dir=args.source_dir,
        model=args.model,
        dataset=args.dataset,
        copy_location=args.copy_location,
        adversarial=args.adversarial,
        name=args.name,
    )
