import torch
from graphgps.attack.utils_attack import get_reached_nodes


# TODO: make default and specific versions to get loaded dependent on cfg (existing is super specific to UPFD?)


def get_global_indices(
    nodes: set[int],
    edges: set[frozenset[int]],
    all_nodes: torch.Tensor,
    node_features: torch.Tensor,
    root_mask: torch.Tensor,
    local_root_offset: torch.Tensor,
    node_ids: list[str],
) -> tuple[set[int], list[str], set[frozenset[int]], list[frozenset[str]]]:
    """reverse root mask index shift"""
    nodes_indices = torch.Tensor(list(nodes))
    masked_roots = torch.nonzero(~root_mask).squeeze()
    local_masked_roots = masked_roots - local_root_offset
    root_mask_offset = torch.searchsorted(
        local_masked_roots,
        nodes_indices,
        right=True,
    )
    nodes_mapping: dict[int, int] = {}
    for i, node_index in enumerate(nodes_indices):
        nodes_mapping[int(node_index)] = int(node_index + root_mask_offset[i])
    # check that it is correct -> indexes the same features
    for local_node_index, global_node_index in nodes_mapping.items():
        assert torch.allclose(all_nodes[global_node_index, :], node_features[local_node_index, :])
    global_nodes = set(nodes_mapping.values())
    # additionally construct it for the ids (not unique -> use list)
    nodes_ids = [node_ids[i] for i in global_nodes]
    # for the edges
    global_edges = set(frozenset(nodes_mapping[node_index] for node_index in edge) for edge in edges)
    edges_ids = [frozenset(node_ids[i] for i in edge) for edge in global_edges]
    return global_nodes, nodes_ids, global_edges, edges_ids


def post_process_attack(
    edge_index: torch.Tensor,
    pert_edge_index: torch.Tensor,
    all_nodes: torch.Tensor,
    node_features: torch.Tensor,
    roots_mask: torch.Tensor,
    local_root_offset: torch.Tensor,
    node_ids: list[str],
    num_nodes_added,
    num_nodes_added_connected,
    num_nodes_removed,
    num_edges_added,
    num_edges_added_connected,
    num_edges_removed,
    count_nodes_added_index,
    count_nodes_added_connected_index,
    count_nodes_removed_index,
):
    def to_global(edges: set[frozenset[int]]):
        nodes = set()
        for edge in edges:
            nodes.update(edge)
        nodes, nodes_ids, edges, edges_ids = get_global_indices(
            nodes, edges, all_nodes, node_features, roots_mask, local_root_offset, node_ids,
        )
        return {"nodes": nodes, "nodes_ids": nodes_ids, "edges": edges, "edges_ids": edges_ids}

    root = int(edge_index[0, 0])
    reached = get_reached_nodes(root, pert_edge_index)
    # edges:
    edges = set(frozenset(int(i) for i in e.squeeze()) for e in torch.split(edge_index, 1, dim=1))
    pert_edges = set(frozenset(int(i) for i in e.squeeze()) for e in torch.split(pert_edge_index, 1, dim=1))
    pert_r_edges = set(e for e in pert_edges if all(n in reached for n in e))
    # nodes:
    result_dict = {}
    for category, category_edges in zip(["clean", "pert", "pert_r"], [edges, pert_edges, pert_r_edges]):
        result_dict[category] = to_global(category_edges)
    # edges
    edges = result_dict["clean"]["edges"]
    pert_edges = result_dict["pert"]["edges"]
    pert_edges_connected = result_dict["pert_r"]["edges"]
    added_edges = pert_edges - edges
    added_edges_connected = pert_edges_connected - edges
    removed_edges = edges - pert_edges
    num_edges_added.append(len(added_edges))
    num_edges_added_connected.append(len(added_edges_connected))
    num_edges_removed.append(len(removed_edges))
    # nodes
    nodes = result_dict["clean"]["nodes"]
    pert_nodes = result_dict["pert"]["nodes"]
    pert_nodes_connected = result_dict["pert_r"]["nodes"]
    added_nodes = pert_nodes - nodes
    added_nodes_connected = pert_nodes_connected - nodes
    removed_nodes = nodes - pert_nodes
    num_nodes_added.append(len(added_nodes))
    for node in added_nodes:
        if node in count_nodes_added_index:
            count_nodes_added_index[node] += 1
        else:
            count_nodes_added_index[node] = 1
    num_nodes_added_connected.append(len(added_nodes_connected))
    for node in added_nodes_connected:
        if node in count_nodes_added_connected_index:
            count_nodes_added_connected_index[node] += 1
        else:
            count_nodes_added_connected_index[node] = 1
    num_nodes_removed.append(len(removed_nodes))
    for node in removed_nodes:
        if node in count_nodes_removed_index:
            count_nodes_removed_index[node] += 1
        else:
            count_nodes_removed_index[node] = 1
