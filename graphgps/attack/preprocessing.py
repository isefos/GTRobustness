import torch
from torch_geometric.utils import remove_self_loops, scatter
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_scipy_sparse_matrix, index_to_mask
from scipy.sparse.csgraph import breadth_first_order
from torch_geometric.transforms import LargestConnectedComponents
from typing import Callable
import functools
from torch_geometric.graphgym.config import cfg


get_largest_connected_subgraph = LargestConnectedComponents(num_components=1, connection="weak")


def remove_isolated_components(data: Batch):
    root_node = cfg.attack.root_node_idx
    if root_node is not None:
        data, root_node = get_only_root_graph(data, root_node)
    else:
        data = get_largest_connected_subgraph(data)
    return data, root_node


def forward_wrapper(forward: Callable, is_undirected: bool) -> Callable:

    @functools.wraps(forward)
    def wrapped_forward(data: Batch, unmodified: bool = False):
        if not unmodified:
            assert data.num_graphs == 1
            if cfg.attack.remove_isolated_components:
                data, root_node = remove_isolated_components(data)
            else:
                root_node = cfg.attack.root_node_idx
                
            use_64bit = False
            num_nodes = data.x.size(0)

            data.node_logprob = get_node_logprob(
                edge_index=data.edge_index,
                edge_weights=data.edge_attr,
                num_nodes=num_nodes,
                is_undirected=is_undirected,
                root_node=root_node,
                num_iterations=cfg.attack.iterations_node_prob,
                use_64bit=use_64bit,
            )
            data.recompute_preprocessing = True
        else:
            data.recompute_preprocessing = False
        model_prediction, _ = forward(data)  # don't need the y ground truth
        return model_prediction

    return wrapped_forward


def get_only_root_graph(batch: Batch, root_node: int):
    # only works for a single graph
    assert torch.all(batch.batch == 0)
    data: Data = batch.get_example(0)
    num_nodes = data.x.size(0)
    device = data.x.device
    # do graph traversal starting from root to find the root component
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)
    bfs_order = breadth_first_order(adj, root_node, return_predecessors=False)
    subset_mask = index_to_mask(torch.tensor(bfs_order, dtype=torch.long, device=device), size=num_nodes)
    data_root_component = data.subgraph(subset_mask)
    new_data = Batch.from_data_list([data_root_component])
    new_root_node = int(subset_mask[:root_node].sum().item())
    assert torch.allclose(data.x[root_node, :], new_data.x[new_root_node, :])
    return new_data, new_root_node


def get_node_logprob(
    edge_index: torch.Tensor,
    edge_weights: None | torch.Tensor,
    num_nodes: int,
    is_undirected: bool,
    num_iterations: int,
    root_node: None | int = None,
    use_64bit: bool = False,
) -> torch.Tensor:
    """
    Sparse implementation as a form of message passing

    needs either a root node or at least one edge with probability 1,
    otherwise the entire graph existence probabalility is in question, 
    and we would need to weight the node probabilities by that as well. 
    (with current implementation and a graph that is not guaranteed to exist,
    iterataions will keep decreasing node probabilities forever without convergence)
    """
    device = edge_index.device
    if edge_weights is None:
        return torch.zeros((num_nodes, ), device=device)

    assert len(edge_weights.shape) == 1, "Only scalar edge weights are supported"
    m = "Edge weights represent probabilities, so they must be between 0 and 1"
    assert torch.all(edge_weights > 0) and torch.all(edge_weights <= 1), m

    edge_index_nsl, prob_edge = remove_self_loops(edge_index, edge_weights)
    if use_64bit:
        prob_edge = prob_edge.to(torch.float64)
    mask_prob_edge_one = prob_edge == 1

    if mask_prob_edge_one.all():
        return torch.zeros((num_nodes, ), device=device)

    m = "The graph must have a root node or at least one edge with probability 1"
    assert root_node is not None or torch.any(mask_prob_edge_one), m

    log1m_prob_edge = torch.full_like(prob_edge, -float('inf'))
    log1m_prob_edge[~mask_prob_edge_one] = (-prob_edge[~mask_prob_edge_one]).log1p()
    logprob_not_node = torch.full((num_nodes, ), -float('inf'), device=device)
    if use_64bit:
        logprob_not_node = logprob_not_node.to(torch.float64)
    for _ in range(num_iterations):
        msg_out = get_msg(
            logprob_not_node=logprob_not_node[edge_index_nsl[1, :]],
            prob_edge=prob_edge,
            log1m_prob_edge=log1m_prob_edge,
            mask_prob_edge_one=mask_prob_edge_one,
        )
        if is_undirected:
            logprob_not_node = scatter(msg_out, edge_index_nsl[0, :], dim=0, dim_size=num_nodes, reduce='sum')
        else:
            msg_in = get_msg(
                logprob_not_node=logprob_not_node[edge_index_nsl[0, :]],
                prob_edge=prob_edge,
                log1m_prob_edge=log1m_prob_edge,
                mask_prob_edge_one=mask_prob_edge_one,
            )
            logprob_not_node = scatter(
                torch.cat((msg_out, msg_in), dim=0),
                edge_index_nsl.flatten(),
                dim=0,
                dim_size=num_nodes,
                reduce='sum',
            )
        assert torch.all(logprob_not_node < 0)
        # reset the root node to prob_not_node = 0 -> logprob_not_node = -inf
        if root_node is not None:
            logprob_not_node[root_node] = -float('inf')

    logprob_node = log1mexp(logprob_not_node)
    assert torch.all(logprob_node <= 0)
    assert torch.all(-float("inf") < logprob_node)
    if use_64bit:
        # convert back to 32bit after computations
        logprob_node = logprob_node.to(torch.float32)
    return logprob_node


def get_msg(
    logprob_not_node: torch.Tensor,
    prob_edge: torch.Tensor,
    log1m_prob_edge: torch.Tensor,
    mask_prob_edge_one: torch.Tensor,
) -> torch.Tensor:
    """
    """
    # when both logprob_not_node = -inf and prob_edge = 1, the message is -inf
    msg = torch.full_like(logprob_not_node, -float('inf'))

    # special case: logprob_not_node = -inf simplifies to: log1p(-prob_edge)
    inf_mask = logprob_not_node == -float('inf')
    mask = torch.logical_and(inf_mask, ~mask_prob_edge_one)
    msg[mask] = log1m_prob_edge[mask]

    # special case: prob_edge = 1 simplifies to: logprob_not_node
    mask = torch.logical_and(~inf_mask, mask_prob_edge_one)
    msg[mask] = logprob_not_node[mask]

    # general case: log1p(prob_edge * expm1(logprob_not_node))
    rest_mask = torch.logical_and(~inf_mask, ~mask_prob_edge_one)
    msg[rest_mask] = (prob_edge[rest_mask] * logprob_not_node[rest_mask].expm1()).log1p()
    
    assert torch.all(msg <= 0)
    return msg


def get_node_logprob_alternative(
    edge_index: torch.Tensor,
    edge_weights: None | torch.Tensor,
    num_nodes: int,
    is_undirected: bool,
    num_iterations: int,
    root_node: None | int = None,
    use_64bit: bool = False,
) -> torch.Tensor:
    """
    Sparse implementation as a form of message passing
    """
    if edge_weights is None:
        return torch.zeros((num_nodes, ))
    assert len(edge_weights.shape) == 1, "Only scalar edge weights are supported"
    m = "Edge weights represent probabilities, so they must be between 0 and 1"
    assert torch.all(edge_weights > 0) and torch.all(edge_weights <= 1), m

    edge_index_nsl, prob_edge = remove_self_loops(edge_index, edge_weights)

    m = "The graph must have a root node or at least one edge with probability 1"
    assert root_node is not None or torch.any(prob_edge == 1), m

    if use_64bit:
        prob_edge = prob_edge.to(torch.float64)

    logprob_edge = prob_edge.log()
    logprob_node = torch.zeros((num_nodes, ))
    if use_64bit:
        logprob_node = logprob_node.to(torch.float64)
    for _ in range(num_iterations):
        msg_out = log1mexp(logprob_node[edge_index_nsl[1, :]] + logprob_edge)
        if is_undirected:
            logprob_not_node = scatter(msg_out, edge_index_nsl[0, :], dim=0, dim_size=num_nodes, reduce='sum')
        else:
            msg_in = log1mexp(logprob_node[edge_index_nsl[0, :]] + logprob_edge)
            logprob_not_node = scatter(
                torch.cat((msg_out, msg_in), dim=0),
                edge_index_nsl.flatten(),
                dim=0,
                dim_size=num_nodes,
                reduce='sum',
            )
        logprob_node = log1mexp(logprob_not_node)
        # reset the root node to 
        #   prob_not_node = 0 -> logprob_not_node = -inf
        #   logprob_node = 1 -> logprob_node = 0 
        if root_node is not None:
            logprob_not_node[root_node] = -float('inf')
            logprob_node[root_node] = 0

    assert torch.all(logprob_node <= 0)
    assert torch.all(-float("inf") < logprob_node)
    return logprob_node.to(torch.float32)


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Calculates log(1 - exp(x)) for x < 0 with "high precision" for both
      - x very close to zero
      - x very large negative

    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    assert torch.all(x <= 0), "The input must be non-positive"
    result = torch.zeros_like(x)

    # when x is exactly zero returns -inf
    zero_mask = x == 0
    result[zero_mask] = -float('inf')

    # when x is -inf returns zero (does not change the entries in result)
    finite_mask = x > -float('inf')
    valid_mask = torch.logical_and(~zero_mask, finite_mask)

    # for values finite values x < 0 the result of log(1 - exp(x)) is calculated in two regimes
    cuttoff_mask = x < -0.69314718056  # -log(2)

    # for x < -log(2) -> exp(x) = p < 0.5, the result is calculated as log1p(-exp(x))
    mask = torch.logical_and(valid_mask, cuttoff_mask)
    result[mask] = (-x[mask].exp()).log1p()

    # for x >= -log(2) -> exp(x) = p >= 0.5, the result is calculated as log(-expm1(x))
    mask = torch.logical_and(valid_mask, ~cuttoff_mask)
    result[mask] = (-x[mask].expm1()).log()
    return result


if __name__ == "__main__":
    from time import perf_counter

    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10],
        [1, 6, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 7, 6, 8, 7, 9, 8, 10, 9],
    ], dtype=torch.long)
    edge_attr = torch.tensor(
        [
            1e-7, 1-1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7,
            1-1e-7, 1-1e-7, 1-1e-7, 1-1e-7, 1-1e-7, 1-1e-7, 1-1e-7, 1-1e-7, 1-1e-7,
        ]
    )

    is_undirected = True
    root_node = 0
    num_nodes = edge_index.max().item() + 1
    num_iterations = 10
    use_64bit = False

    t = perf_counter()

    node_logprob_1 = get_node_logprob(
        edge_index=edge_index,
        edge_weights=edge_attr,
        num_nodes=num_nodes,
        is_undirected=is_undirected,
        root_node=root_node,
        num_iterations=num_iterations,
        use_64bit=use_64bit,
    )

    print(f"Elapsed time 1: {perf_counter() - t:.6f}")

    t = perf_counter()

    node_logprob_2 = get_node_logprob_alternative(
        edge_index=edge_index,
        edge_weights=edge_attr,
        num_nodes=num_nodes,
        is_undirected=is_undirected,
        root_node=root_node,
        num_iterations=num_iterations,
        use_64bit=use_64bit,
    )

    print(f"Elapsed time 2: {perf_counter() - t:.6f}")

    assert torch.all(node_logprob_1 == node_logprob_2)

    print("\nLogprob 1 and 2:")
    for logprob_1, logprob_2 in zip(node_logprob_1.numpy(), node_logprob_2.numpy()):
        print(f"{repr(logprob_1):>20} {repr(logprob_2):>20}")

    print("\nProb 1 and 2:")
    for prob_1, prob_2 in zip(node_logprob_1.exp().numpy(), node_logprob_2.exp().numpy()):
        print(f"{repr(prob_1):>20} {repr(prob_2):>20}")
