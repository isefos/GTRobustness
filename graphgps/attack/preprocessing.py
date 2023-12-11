import torch
from torch_geometric.utils import remove_self_loops, scatter
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_scipy_sparse_matrix, index_to_mask, subgraph
from scipy.sparse.csgraph import breadth_first_order
from typing import Callable
import functools


# TODO: also save maximum edge weight of each node -> for shortest path pruning
# just in the first iteration with max reduction, almost no additional cost


def forward_wrapper(forward: Callable, is_undirected: bool) -> Callable:

    @functools.wraps(forward)
    def wrapped_forward(data, root_node: None | int = None, remove_not_connected: bool = False):
        if remove_not_connected:
            assert root_node is not None, "The root node must be specified"
            new_data, new_root_node = get_only_root_graph(data, root_node)

            # TODO: where to get bool: undirected from?
            new_data.node_probs = node_in_graph_prob(
                edge_index=new_data.edge_index,
                edge_weights=new_data.edge_attr,
                batch=new_data.batch,
                undirected=is_undirected,
                root_node=new_root_node,
                num_iterations=4,
            )
            return forward(new_data)[0]
        else:
            return forward(data)[0]

    return wrapped_forward


def get_only_root_graph(data, root_node: int):
    # only works for a single graph
    assert torch.all(data.batch == 0)
    num_nodes = data.x.size(0)
    # do graph traversal starting from root to find the root component
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)
    bfs_order = breadth_first_order(adj, root_node, return_predecessors=False)
    subset_mask = index_to_mask(torch.tensor(bfs_order, dtype=torch.long), size=num_nodes)
    edge_index, edge_attr = subgraph(subset_mask, data.edge_index, data.edge_attr, relabel_nodes=True)
    new_data = Batch.from_data_list(
        [Data(x=data.x[subset_mask, :].clone(), edge_index=edge_index, edge_attr=edge_attr)]
    )
    new_root_node = int(subset_mask[:root_node].sum().item())
    assert torch.allclose(data.x[root_node, :], new_data.x[new_root_node, :])
    return new_data, new_root_node


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