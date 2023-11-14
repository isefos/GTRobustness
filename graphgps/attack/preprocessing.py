import torch
from torch_geometric.utils import to_dense_adj, remove_self_loops


def node_in_graph_prob_undirected(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    batch: torch.Tensor,
    num_iterations: int = 5,
    root_node: None | int = None,
) -> torch.Tensor:
    assert num_iterations > 0, "Must do at least one iteration"
    assert len(edge_weights.shape) == 1, "Only scalar edge weights are supported"
    adj_edge_index, adj_edge_weights = remove_self_loops(edge_index, edge_weights)
    adj = to_dense_adj(adj_edge_index, batch, adj_edge_weights).squeeze_()
    prob_nodes = torch.ones(batch.size(0))
    for _ in range(num_iterations):
        prob_nodes_new = 1 - torch.prod(1 - adj * prob_nodes[None, :], dim=1)
        if root_node is not None:
            prob_nodes_new[root_node] = 1
        prob_nodes = prob_nodes_new
    return prob_nodes


def node_in_graph_prob_directed(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    batch: torch.Tensor,
    num_iterations: int = 5,
    root_node: None | int = None,
) -> torch.Tensor:
    raise NotImplementedError("Must implement vectorized version, otherwise too slow")
    assert num_iterations > 0, "Must do at least one iteration"
    prob_nodes = torch.ones(num_nodes)
    for _ in range(num_iterations):
        prob_nodes_new = torch.zeros(num_nodes)
        for node_idx in range(num_nodes):
            edges_out_mask = edge_index[0, :] == node_idx
            neighbors_out = edge_index[1, edges_out_mask]
            neigh_out_weights = edge_weights[edges_out_mask]
            edges_in_mask = edge_index[1, :] == node_idx
            neighbors_in = edge_index[0, edges_in_mask]
            neigh_in_weights = edge_weights[edges_in_mask]
            neighbors = torch.cat((neighbors_out, neighbors_in), dim=0)
            edges_neigh_weights = torch.cat((neigh_out_weights, neigh_in_weights), dim=0)
            prob_nodes_new[node_idx] = 1 - torch.prod(1 - prob_nodes[neighbors] * edges_neigh_weights)
        prob_nodes = prob_nodes_new
        if root_node is not None:
            prob_nodes[root_node] = 1
    return prob_nodes


def node_in_graph_prob(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    batch: torch.Tensor,
    undirected: bool,
    num_iterations: int = 5,
    root_node: None | int = None,
):
    if edge_weights is None:
        return torch.ones(batch.size(0))
    fun = node_in_graph_prob_undirected if undirected else node_in_graph_prob_directed
    node_prob = fun(
            edge_index=edge_index,
            edge_weights=edge_weights,
            batch=batch,
            num_iterations=num_iterations,
            root_node=root_node,
    )
    return node_prob