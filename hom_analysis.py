import graphgps  # need to import to generate some stuff
from torch_geometric.graphgym.config import cfg, set_cfg
set_cfg(cfg)
from yacs.config import CfgNode as CN
import torch
from torch_geometric.utils import to_undirected, coalesce, degree, scatter
from torch_geometric.graphgym.loader import create_loader
import json
from pathlib import Path
from torch_geometric.data import Data, Batch
from graphgps.attack.preprocessing import remove_isolated_components
from graphgps.attack.dataset_attack import get_total_dataset_graphs, get_attack_datasets
from graphgps.attack.attack import get_attack_loader, get_attack_graph
import pickle


def get_node_homophilies(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    # note that graphs are undirected -> each edge is included in both directions
    # num_nodes = x.size(0)
    # dim_x = x.size(1)
    # num_edges = edge_index.size(1)
    degrees = degree(edge_index[0])
    assert (degrees == degree(edge_index[1])).all()
    inv_sqrt_d = 1 / degrees.sqrt()
    r = inv_sqrt_d[:, None] * scatter(
        src=(inv_sqrt_d[:, None] * x)[edge_index[0, :], :],
        index=edge_index[1, :],
        dim=0,
    )
    dot_prod = (r * x).sum(1)
    sim = dot_prod / (l2(r) * l2(x))
    return sim


def l2(x: torch.Tensor) -> torch.Tensor:
    return ((x ** 2).sum(1)).sqrt()


def check_equal_configs(configs_given: CN, currect_configs: CN) -> None:
    for k, value_given in configs_given.items():
        if isinstance(value_given, CN):
            check_equal_configs(value_given, currect_configs[k])
        else:
            value = currect_configs[k]
            if k == "name" and configs_given.get("format", "") == "PyG-UPFD":
                # for UPFD only the dataset politifact / gossipcop has to match, the features don't matter
                value = value.split("-")[0]
                value_given = value_given.split("-")[0]
            assert value == value_given, "The transfer attack configs are not compatible with the current ones."


def transfer_attack_graph(
    i: int,
    clean_data: Data,
    perturbation: list[list[int]],
    total_attack_dataset_graph: Data | None,
    attack_dataset_slices: list[tuple[int, int]] | None,
    total_additional_datasets_graph: Data | None,
) -> torch.Tensor:
    # AUGMENT GRAPH
    assert attack_dataset_slices is not None
    attack_graph_data = get_attack_graph(
        graph_data=clean_data,
        total_attack_dataset_graph=total_attack_dataset_graph,
        attack_dataset_slice=attack_dataset_slices[i],
        total_additional_datasets_graph=total_additional_datasets_graph,
    )
    # APPLY PERTURBATION
    pert_edge_index = get_perturbed_edge_index(attack_graph_data, perturbation)
    data = Data(x=attack_graph_data.x.clone(), edge_index=pert_edge_index.clone())
    data, _ = remove_isolated_components(data)
    return get_node_homophilies(data.x, data.edge_index)


def get_perturbed_edge_index(data: Data, perturbation):
    N = data.x.size(0)
    E_clean = data.edge_index.size(1)
    pert_edge_index = torch.tensor(perturbation, dtype=torch.long, device=data.x.device)
    if cfg.attack.is_undirected:
        pert_edge_index = to_undirected(pert_edge_index, num_nodes=N)
    E_pert = pert_edge_index.size(1)
    modified_edge_index = torch.cat((data.edge_index, pert_edge_index), dim=-1)
    modified_edge_weight = torch.ones(E_clean + E_pert)
    modified_edge_index, modified_edge_weight = coalesce(
        modified_edge_index,
        modified_edge_weight,
        num_nodes=N,
        reduce='sum',
    )
    removed_edges_mask = ~(modified_edge_weight == 2)
    modified_edge_index = modified_edge_index[:, removed_edges_mask]
    return modified_edge_index


def main(
        dataset_option = "upfd_pol",
        pert_dir = "../gtr-saved/perturbations",
        model_name = "GRIT",
    ):
    """
    dataset_option: upfd_gos, upfd_pol
    pert_dir: where the saved perturbations are stored locally
    model_name: GCN / GPS / Graphormer / SAN / Polynormer / GRIT
    """
    print(f"\nStarting for {dataset_option} - {model_name}...\n")
    # config file: just need any that loads the right dataset later
    # TODO: changing this specific local config later could break things, if broken:
    #  load a configs_seml config, see loading jupyter to get valid config by removing the seml stuff
    config_file = Path("./configs_local_debug").resolve() / "GCN" / f"{dataset_option}.yaml"
    with open(config_file, "r") as f:
        cfg_file = CN._load_cfg_from_file(f)
    cfg_file = cfg_file.graphgym
    cfg.merge_from_other_cfg(cfg_file)
    print("\nConfig loaded...\n")

    # SAVED PERTURBATION FILES
    dataset_base_name = "UPFD_gos_bert" if dataset_option == "upfd_gos" else "UPFD_pol_bert"
    perturbation_path = Path(pert_dir).resolve() / dataset_base_name / model_name
    # check config -> not really needed, if this throws error, just comment
    with open(perturbation_path / "attack_configs.yaml", "r") as f:
        attack_configs = CN.load_cfg(f)
    check_equal_configs(attack_configs, cfg)
    # extract perturbations from perturbation path
    perturbations, seeds, budgets = [], [], []
    for child in perturbation_path.iterdir():
        if not child.is_dir():
            continue
        # for each seed
        seed = int(child.name[1:])
        for pert_file in child.iterdir():
            # for each budget
            budget = float(pert_file.name[15:-5])
            with open(pert_file, "r") as f:
                perturbations.append(json.load(f))
            seeds.append(seed)
            budgets.append(budget)
        # TODO: if only want to evaluate for one seed set a break here:
        # break
    print("\nPerturbations loaded....\n")

    # DATASETS
    loaders = create_loader()
    assert len(loaders) == 3, "this script assumes train, val and test sets"
    assert cfg.attack.split == "test", "this script assumes that the test set was attacked"

    # TRAIN AND VAL SETS
    node_homophily_results = {}
    for loader, split in zip(loaders[:2], ["train", "val"]):
        node_homophily_results[split] = []
        for batch in loader:
            batch: Batch
            # standard loader generates batches with multiple graphs
            for graph in batch.to_data_list():
                graph: Data
                graph_node_homophilies = get_node_homophilies(graph.x, graph.edge_index)
                node_homophily_results[split].append(graph_node_homophilies)

    # PREPARE NIA ATTACK DATASET
    dataset_to_attack, additional_injection_datasets, inject_nodes_from_attack_dataset = get_attack_datasets(loaders)
    assert cfg.attack.node_injection.enable, "only for NIA"
    total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph = get_total_dataset_graphs(
        inject_nodes_from_attack_dataset=inject_nodes_from_attack_dataset,
        dataset_to_attack=dataset_to_attack,
        additional_injection_datasets=additional_injection_datasets,
        include_root_nodes=cfg.attack.node_injection.include_root_nodes,
    )

    # TEST AND TEST_PERTURBED
    node_homophily_results["test"] = []
    # the pert results are saved in dicts by budget and seed
    node_homophily_results["pert"] = []
    clean_loader = get_attack_loader(dataset_to_attack)
    for i, clean_data in enumerate(clean_loader):
        clean_data: Batch
        # attack loader generates batches with a single graph
        assert clean_data.num_graphs == 1
        clean_graph_node_homophilies = get_node_homophilies(clean_data.x, clean_data.edge_index)
        node_homophily_results["test"].append(clean_graph_node_homophilies)
        pert_homophily_results = dict()
        for seed, budget, perts in zip(seeds, budgets, perturbations):
            # get the pert of this graph (for all seeds and budgets)
            perturbation = perts.get(str(i), None)
            if perturbation is None:
                continue
            if budget not in pert_homophily_results:
                pert_homophily_results[budget] = dict()
            assert seed not in pert_homophily_results[budget]
            pert_node_homophily = transfer_attack_graph(
                i,
                clean_data.get_example(0),
                perturbation,
                total_attack_dataset_graph,
                attack_dataset_slices,
                total_additional_datasets_graph,
            )
            pert_homophily_results[budget][seed] = pert_node_homophily
        node_homophily_results["pert"].append(pert_homophily_results)

    print("\nCollected all node homophilies, saving to file...\n")
    results_dir = Path("results_hom").resolve()
    with open(results_dir / f"{dataset_option}_{model_name}.pkl", 'wb') as f:
        pickle.dump(node_homophily_results, f)
    
    # TODO: process the results and generate plots!
    #  potentially aggregate over different seeds
    #  and then processes over different budgets


if __name__ == "__main__":
    # TODO: change the pert dir if necessary
    pert_dir = "../gtr-saved/perturbations"
    for model in ["GCN", "GPS", "SAN", "Polynormer", "GRIT", "Graphormer"]:
        for dataset in ["upfd_gos", "upfd_pol"]:
            main(dataset_option=dataset, model_name=model, pert_dir=pert_dir)
