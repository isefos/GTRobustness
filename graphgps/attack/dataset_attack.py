import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data.collate import collate
import logging


def filter_out_root_node(graph: Data, root_node_idx: None | int) -> Data:
    assert root_node_idx is not None, "If specified to not inlcude root nodes, must also specify the root node index!"
    n = graph.x.size(0)
    graph = graph.subgraph(
        subset=torch.tensor(
            [j for j in range(n) if j != root_node_idx],
            dtype=torch.long,
        )
    )
    return graph


def get_total_dataset_graphs(
    inject_nodes_from_attack_dataset: bool,
    dataset_to_attack: Dataset,
    additional_injection_datasets: None | list[Dataset],
    include_root_nodes: bool,
    root_node_idx: None | int,
    device,
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
                graph_to_add = filter_out_root_node(graph_to_add, root_node_idx)
            graphs_to_join.append(graph_to_add)
            next_idx = current_idx + graph_to_add.x.size(0)
            attack_dataset_slices.append((current_idx, next_idx))
            current_idx = next_idx
        merge_result = collate(
            cls=Data,
            data_list=graphs_to_join,
            increment=False,
            add_batch=False,
            exclude_keys=["y"],
        )
        total_attack_dataset_graph = merge_result[0].to(device=device)

    if additional_injection_datasets is None:
        total_additional_datasets_graph = None
    else:
        graphs_to_join: list[Data] = []
        for additional_dataset in additional_injection_datasets:
            for graph_to_add in additional_dataset:
                graph_to_add.edge_index = torch.empty((2, 0), dtype=torch.long)
                if not include_root_nodes:
                    graph_to_add = filter_out_root_node(graph_to_add, root_node_idx)
                graphs_to_join.append(graph_to_add)
        merge_result = collate(
            cls=Data,
            data_list=graphs_to_join,
            increment=False,
            add_batch=False,
            exclude_keys=["y"],
        )
        total_additional_datasets_graph = merge_result[0].to(device=device)
    
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
        current_mask = torch.ones(n, dtype=torch.bool, device=total_attack_dataset_graph.x.device)
        current_mask[attack_dataset_slice[0]:attack_dataset_slice[1]] = 0
        attack_dataset_graph_to_add = total_attack_dataset_graph.subgraph(current_mask)
    graphs_to_join = [g for g in (graph, attack_dataset_graph_to_add, total_additional_datasets_graph) if g is not None]
    merge_result = collate(
        cls=Data,
        data_list=graphs_to_join,
        increment=False,
        add_batch=False,
        exclude_keys=["y"],
    )
    augmented_graph = merge_result[0]
    augmented_graph.y = graph.y
    return augmented_graph