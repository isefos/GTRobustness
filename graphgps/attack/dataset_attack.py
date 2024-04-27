import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data.collate import collate
import logging
from torch_geometric.graphgym.config import cfg


def get_attack_datasets(loaders):
    splits = ["train", "val", "test"]
    split_to_attack_idx = splits.index(cfg.attack.split)
    dataset_to_attack = loaders[split_to_attack_idx].dataset

    additional_injection_datasets = None
    inject_nodes_from_attack_dataset = False

    if cfg.attack.node_injection.enable:
        include_additional_datasets = [
            cfg.attack.node_injection.from_train,
            cfg.attack.node_injection.from_val,
            cfg.attack.node_injection.from_test,
        ]
        inject_nodes_from_attack_dataset = include_additional_datasets[split_to_attack_idx]
        include_additional_datasets[split_to_attack_idx] = False
        additional_injection_datasets = [l.dataset for i, l in enumerate(loaders) if include_additional_datasets[i]]

    return dataset_to_attack, additional_injection_datasets, inject_nodes_from_attack_dataset


def filter_out_root_node(graph: Data) -> Data:
    root_node_idx = cfg.attack.root_node_idx
    assert root_node_idx is not None, "If specified to not inlcude root nodes, must also specify the root node index!"
    assert isinstance(root_node_idx, int), "Root node index must be an integer!"
    n = graph.x.size(0)
    graph = graph.subgraph(
        subset=torch.tensor(
            [j for j in range(n) if j != root_node_idx],
            dtype=torch.long,
        )
    )
    return graph


_collate_exclude_keys = [
        "y", 
        # Graphormer
        "graph_index", "in_degrees", "out_degrees", "degrees", "spatial_types",
        # GRIT
        "rrwp", "rrwp_index", "rrwp_val", "log_deg",
        # SAN
        "EigVecs", "EigVals",
    ]


def get_total_dataset_graphs(
    inject_nodes_from_attack_dataset: bool,
    dataset_to_attack: Dataset,
    additional_injection_datasets: None | list[Dataset],
    include_root_nodes: bool,
) -> tuple[None | Data, None | list[tuple[int, int]], None | Data]:
    """
    """
    if not inject_nodes_from_attack_dataset:
        total_attack_dataset_graph = None
        attack_dataset_slices = None
    else:
        graphs_to_join: list[Data] = []
        attack_dataset_slices: list[tuple[int, int]] = []
        current_idx = 0
        for graph_to_add in dataset_to_attack:
            graph_to_add.edge_index = torch.empty((2, 0), dtype=torch.long)
            if not include_root_nodes:
                graph_to_add = filter_out_root_node(graph_to_add)
            graphs_to_join.append(graph_to_add)
            next_idx = current_idx + graph_to_add.x.size(0)
            attack_dataset_slices.append((current_idx, next_idx))
            current_idx = next_idx
        merge_result = collate(
            cls=Data,
            data_list=graphs_to_join,
            increment=False,
            add_batch=False,
            exclude_keys=_collate_exclude_keys,
        )
        total_attack_dataset_graph = merge_result[0]

    if additional_injection_datasets is None:
        total_additional_datasets_graph = None
    else:
        graphs_to_join: list[Data] = []
        for additional_dataset in additional_injection_datasets:
            for graph_to_add in additional_dataset:
                graph_to_add.edge_index = torch.empty((2, 0), dtype=torch.long)
                if not include_root_nodes:
                    graph_to_add = filter_out_root_node(graph_to_add)
                graphs_to_join.append(graph_to_add)
        merge_result = collate(
            cls=Data,
            data_list=graphs_to_join,
            increment=False,
            add_batch=False,
            exclude_keys=_collate_exclude_keys,
        )
        total_additional_datasets_graph = merge_result[0]
    
    return total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph


def get_augmented_graph(
    graph: Data,
    total_attack_dataset_graph: None | Data,
    attack_dataset_slice: tuple[int, int],
    total_additional_datasets_graph: None | Data,
)-> Data:
    """
    """
    if total_attack_dataset_graph is None and total_additional_datasets_graph is None:
        logging.info("Augmenting graph for node injection, but no additional nodes for injection were configured...")
        return graph.clone()
    if total_attack_dataset_graph is None:
        attack_dataset_graph_to_add = None
    else:
        n = total_attack_dataset_graph.x.size(0)
        current_mask = torch.ones(n, dtype=torch.bool, device="cpu")
        current_mask[attack_dataset_slice[0]:attack_dataset_slice[1]] = 0
        attack_dataset_graph_to_add = total_attack_dataset_graph.subgraph(current_mask)
    graphs_to_join = [g for g in (graph, attack_dataset_graph_to_add, total_additional_datasets_graph) if g is not None]
    merge_result = collate(
        cls=Data,
        data_list=graphs_to_join,
        increment=False,
        add_batch=False,
        exclude_keys=_collate_exclude_keys,
    )
    augmented_graph = merge_result[0]
    augmented_graph.y = graph.y
    return augmented_graph