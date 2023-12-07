import torch
from torch_geometric.utils import remove_self_loops, scatter


# TODO: also save maximum edge weight of each node -> for shortest path pruning
# just in the first iteration with max reduction, almost no additional cost


def node_in_graph_prob_undirected(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    batch: torch.Tensor,
    num_iterations: int = 5,
    root_node: None | int = None,
) -> torch.Tensor:
    """
    Sparse implementation as a form of message passing
    """
    assert num_iterations > 0, "Must do at least one iteration"
    assert len(edge_weights.shape) == 1, "Only scalar edge weights are supported"
    num_nodes = batch.size(0)
    edge_index_nsl, edge_weights_nsl = remove_self_loops(edge_index, edge_weights)
    prob_nodes = torch.ones(num_nodes)
    for _ in range(num_iterations):
        msg = 1 - edge_weights_nsl * prob_nodes[edge_index_nsl[1, :]]
        out = scatter(msg, edge_index_nsl[0, :], dim=0, dim_size=num_nodes, reduce='mul')
        prob_nodes_new = 1 - out
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
    """
    Sparse implementation as a form of message passing
    """
    assert num_iterations > 0, "Must do at least one iteration"
    assert len(edge_weights.shape) == 1, "Only scalar edge weights are supported"
    num_nodes = batch.size(0)
    edge_index_nsl, edge_weights_nsl = remove_self_loops(edge_index, edge_weights)
    prob_nodes = torch.ones(num_nodes)
    for _ in range(num_iterations):
        msg_out = 1 - edge_weights_nsl * prob_nodes[edge_index_nsl[1, :]]
        msg_in = 1 - edge_weights_nsl * prob_nodes[edge_index_nsl[0, :]]
        out = scatter(
            torch.cat((msg_out, msg_in), dim=0),
            torch.cat((edge_index_nsl[0, :], edge_index_nsl[1, :]), dim=0),
            dim=0,
            dim_size=num_nodes,
            reduce='mul',
        )
        prob_nodes_new = 1 - out
        if root_node is not None:
            prob_nodes_new[root_node] = 1
        prob_nodes = prob_nodes_new
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