import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix, scatter
from itertools import combinations
from graphgps.encoder.graphormer_encoder import add_graph_token
from scipy.sparse import csgraph
import time

# Permutes from (batch, node, node, head) to (batch, head, node, node)
BATCH_HEAD_NODE_NODE = (0, 3, 1, 2)

# Inserts a leading 0 row and a leading 0 column with F.pad
INSERT_GRAPH_TOKEN = (1, 0, 1, 0)


def node_degree_weights_undirected(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    undirected_extra_checks: bool = False,
    remove_zero_degree: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    """
    # TODO: put on the correct device
    node_degree_info = get_node_degree_info(
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_weights=edge_weights,
        undirected=True,
        undirected_extra_checks=undirected_extra_checks,
    )
    node_stats, all_degrees = node_degree_info["undirected"]
    all_degree_indices, all_degree_weights = get_degree_weights(
        node_stats=node_stats, all_degrees=all_degrees, remove_zero_degree=remove_zero_degree,
    )
    return all_degree_indices, all_degree_weights


def get_node_degree_info(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    undirected: bool,
    undirected_extra_checks: bool = False,
) -> dict[str, tuple[list[tuple[int, int, torch.Tensor]], set[int]]]:
    """
    """
    if undirected:
        degree_edge_dims = {"undirected": 0}
        node_degree_info = {"undirected": ([], set())}
    else:
        degree_edge_dims = {"out": 0, "in": 1}
        node_degree_info = {"out": ([], set()), "in": ([], set())}
    for i in range(num_nodes):
        summed_degree = 0
        for degree_mode, idx in degree_edge_dims.items():
            edges_mask = edge_index[idx, :] == i
            max_degree = int(edges_mask.sum().item())
            summed_degree += max_degree
            uncertain_edge_mask = edges_mask * ((edges_mask * edge_weights) < 1)
            uncertain_edge_weights = edge_weights[uncertain_edge_mask]
            n_uncertain_edges = int(uncertain_edge_mask.sum().item())
            # assert max_in_degree - n_uncertain_edges == (edge_weights[edges_to_node_mask] == 1).sum().item()
            certain_degree = max_degree - n_uncertain_edges
            node_degree_info[degree_mode][0].append((certain_degree, n_uncertain_edges, uncertain_edge_weights))
            node_degree_info[degree_mode][1].update(range(certain_degree, max_degree+1))
            if undirected and undirected_extra_checks:
                opposite_edges_mask = edge_index[int(not idx), :] == i
                assert max_degree == opposite_edges_mask.sum(), "not undirected: different in / out degrees"
                assert torch.allclose(
                    edge_weights[edges_mask], edge_weights[opposite_edges_mask]
                ), "weight different for in/out"
        assert summed_degree > 0, "A totally isolated node should not be included"
    return node_degree_info


def get_degree_weights(
    node_stats: list[tuple[int, int, torch.Tensor]],
    all_degrees: set[int],
    remove_zero_degree: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    sorted_degrees = sorted(all_degrees)
    degree_index_map = {degree: i for i, degree in enumerate(sorted_degrees)}
    all_degree_indices = torch.tensor(sorted_degrees, dtype=torch.long)
    # dim = (N, (max_degree-min_degree)) -> maximally (N, N+1)
    all_degree_weights = torch.zeros((len(node_stats), all_degree_indices.size(0)))
    for node, (certain_degree, n, uncertain_edge_weights) in enumerate(node_stats):
        # can compute the indices as sum on top of min index safely, since all must be included
        min_idx = degree_index_map[certain_degree]
        if n == 0 and not certain_degree == 0:
            # a node with only certain edges
            all_degree_weights[node,  min_idx] += 1
            continue
        if certain_degree == 0 and n == 1:
            # a node with only a single uncertain edge
            # set degree to one, as we don't allow degree zero (does not appear in undirected graphs)
            all_degree_weights[node,  degree_index_map[1]] += uncertain_edge_weights.squeeze()
            continue
        weights_selection = torch.concat((1 - uncertain_edge_weights[None, :], uncertain_edge_weights[None, :]), dim=0)
        if n == 1:
            # only a single uncertain edge -> 2 options, can find directly:
            assert weights_selection.size(1) == 1
            all_degree_weights[node, [min_idx, min_idx+1]] += weights_selection.squeeze()
            continue
        # compute the combinations (n, k), start with the 0 and n: only a single combination
        all_degree_weights[node, min_idx] += torch.prod(weights_selection[0, :])
        all_degree_weights[node, min_idx+n] += torch.prod(weights_selection[1, :])
        if n > 2:
            # can skip for n == 2, uses symmetry for k and n-k
            # num_combinations = 1
            indices = tuple(range(n))
            max_k = (n + 1) // 2
            for k in range(1, max_k):
                # num_combinations *= (n + 1 - k) // k
                for comb in combinations(indices, k):
                    mask = torch.zeros_like(weights_selection).to(dtype=torch.bool)
                    mask[0, :] = 1
                    mask[0, comb] = 0
                    mask[1, comb] = 1
                    # for k
                    all_degree_weights[node, min_idx+k] += torch.prod((mask * weights_selection).sum(dim=0))
                    # for n-k
                    all_degree_weights[node, min_idx+n-k] += torch.prod((~mask * weights_selection).sum(dim=0))
        if n % 2 == 0:
            # even -> the last entry in the 'middle' is still missing
            # TODO: does it make a difference if we compute it in this simplified manner for the gradients?
            all_degree_weights[node, min_idx+n//2] += 1 - all_degree_weights[node, :].sum()
    if remove_zero_degree and sorted_degrees[0] == 0:
        # don't allow the option of no edges existing, since that would lead to an isolated node
        all_degree_indices = all_degree_indices[1:]
        all_degree_weights = all_degree_weights[:, 1:]
    return all_degree_indices, all_degree_weights


def weighted_graphormer_pre_processing(
    data,
    distance: int,
    directed_graphs: bool,
    combinations_degree: bool,
    num_in_degrees: int,
    num_out_degrees: int,
    use_weighted_path_distance: bool,
):
    """
    """
    # TODO: only undirected implemented for now, add directed later
    if directed_graphs:
        raise NotImplementedError

    n = data.x.size(0)
    # logging.info(f"num nodes: {n}, num edges: {data.edge_index.size(1)}")

    t = time.perf_counter()
    if combinations_degree:
        # TODO: rewrite vectorized -> scatter:
        #in_degrees_max = torch.scatter(
        #    input=torch.zeros(n),
        #    value=1,
        #    index=data.edge_index[1, :],
        #    dim=0,
        #    reduce="add"
        #)
        #certain_edges = data.edge_index[:, data.edge_attr == 1]
        #in_degrees_min = torch.scatter(
        #    input=torch.zeros(n),
        #    value=1,
        #    index=certain_edges[1, :],
        #    dim=0,
        #    reduce="add"
        #)
        #ns = in_degrees_max - in_degrees_min
        #n_max = ns.max()
        raise NotImplementedError
        # TODO: handle too large degree -> clamping to max
        data.degree_indices, data.degree_weights = node_degree_weights_undirected(
            data.x.size(0), data.edge_index, data.edge_attr,
        )
    else:
        in_degrees_weighted = torch.scatter_add(
            input=torch.zeros(n),
            src=data.edge_attr,
            index=data.edge_index[1, :],
            dim=0,
        )
        # clamp to maximum degree
        max_degree = max(num_in_degrees, num_out_degrees) - 1
        in_degrees_weighted[in_degrees_weighted > max_degree] = max_degree
        in_degrees_low = torch.floor(in_degrees_weighted).to(dtype=torch.long)
        in_degrees_high = torch.ceil(in_degrees_weighted).to(dtype=torch.long)
        weights_high = in_degrees_weighted - in_degrees_low
        weights_low = in_degrees_high - in_degrees_weighted
        weights_low[in_degrees_low == in_degrees_high] = 1
        # unweighted edges, remove zero degree encoding 
        # (never used during training, can't appear after discretization)
        weights_low[in_degrees_low == 0] = 0
        data.degree_indices = torch.cat((in_degrees_low[:, None], in_degrees_high[:, None]), dim=1)
        data.degree_weights = torch.cat((weights_low[:, None], weights_high[:, None]), dim=1)
    t = time.perf_counter() - t
    # logging.info(f"computed degrees in: {t:.5f}")

    if cfg.posenc_GraphormerBias.node_degrees_only:
        return data
    
    data.graph_index = torch.cat(
        (
            torch.arange(n, dtype=torch.long).repeat_interleave(n)[None, :],
            torch.arange(n, dtype=torch.long).repeat(n)[None, :]),
        dim=0,
    )

    # TODO: maybe use the actual tensor edge weight in the computation to get gradient
    # to get dense inverted adj matrix
    # idx = n * data.edge_index[0] + data.edge_index[1]
    # inv_adj = scatter(1 / data.edge_attr, idx, dim=0, dim_size=n**2, reduce='sum')
    # inv_adj = inv_adj.view((n, n))
    # inv_adj_np = inv_adj.detach().numpy()

    t = time.perf_counter()

    # for now calculate the fixed distance values, without any gradient
    inv_edge_attr = 1 / data.edge_attr.detach()
    adj_weighted = to_scipy_sparse_matrix(data.edge_index, edge_attr=inv_edge_attr, num_nodes=n).tocsc()
    distances_weighted_np, predecessors_np = csgraph.shortest_path(
        adj_weighted, method="auto", directed=directed_graphs, return_predecessors=True, unweighted=False,
    )
    distances_weighted_np = distances_weighted_np.reshape(n**2)
    clamped_distance_mask_np = distances_weighted_np > distance - 1
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).tocsc()
    distances_hop_np = csgraph.construct_dist_matrix(adj, predecessors_np, directed=directed_graphs)
    distances_hop_np[distances_hop_np > distance - 1] = distance - 1
    max_hops = int(distances_hop_np.max())

    if use_weighted_path_distance:

        use_weighted_gradient = True

        if use_weighted_gradient:
            # get dense inv adj for easier indexing in path reconstruction
            inv_adj = scatter(
                inv_edge_attr,
                n * data.edge_index[0] + data.edge_index[1],
                dim=0,
                dim_size=n**2,
                reduce='sum',
            )
            # hack: because we know that the self loops are zero, 
            # we can just put all the negative indices to (0, 0) -> 0
            # then we are technically always adding that first self-loop edge for "non-edges", 
            # but doesn't matter since we are just adding 0
            assert inv_adj[0].item() == 0
            clamped_distance_mask = torch.tensor(clamped_distance_mask_np, dtype=torch.bool)
            true_distance_mask = ~clamped_distance_mask
            distances_weighted = torch.zeros(n**2)

            p_original = torch.tensor(predecessors_np.reshape((n**2)), dtype=torch.long)
            adj_col_idx = p_original[true_distance_mask]
            neg_mask = adj_col_idx < 0
            adj_col_idx[neg_mask] = 0
            # ends: torch.arange(n, dtype=torch.long).repeat(n)
            adj_row_idx = torch.arange(n, dtype=torch.long).repeat(n)[true_distance_mask]
            adj_row_idx[neg_mask] = 0
            adj_lin_idx = n * adj_col_idx + adj_row_idx
            distances_weighted[true_distance_mask] += inv_adj[adj_lin_idx]

            row_idx = torch.zeros((n**2), dtype=torch.long)
            col_idx = torch.arange(n, dtype=torch.long).repeat_interleave(n)
            p_prev = p_original
            for i in range(max_hops - 1):
                row_idx = p_prev.clone()
                row_idx[row_idx < 0] = col_idx[row_idx < 0]
                p_next = p_original[n * col_idx + row_idx]
                adj_col_idx = p_prev[true_distance_mask]
                adj_row_idx = p_next[true_distance_mask]
                neg_mask = torch.logical_or(adj_col_idx < 0, adj_row_idx < 0)
                adj_lin_idx = n * adj_col_idx + adj_row_idx
                adj_lin_idx[neg_mask] = 0
                distances_weighted[true_distance_mask] += inv_adj[adj_lin_idx]
                p_prev = p_next
            distances_weighted[clamped_distance_mask] = distance - 1

            debug = True
            if debug:
                distances_clamped = torch.tensor(distances_weighted_np, dtype=torch.float32)
                distances_clamped[clamped_distance_mask] = distance - 1
                assert torch.allclose(distances_weighted, distances_clamped)

        else:
            distances_weighted = torch.tensor(distances_weighted_np, dtype=torch.float32)
            distances_weighted[distances_weighted > distance - 1] = distance - 1

        dw_low = torch.floor(distances_weighted).to(dtype=torch.long)
        dw_high = torch.ceil(distances_weighted).to(dtype=torch.long)
        weights_high = distances_weighted - dw_low
        weights_low = dw_high - distances_weighted
        weights_low[dw_low == dw_high] = 1
        spatial_types = torch.cat((dw_low[:, None], dw_high[:, None]), dim=1)
        data.spatial_types_weights = torch.cat((weights_low[:, None], weights_high[:, None]), dim=1)
    else:
        spatial_types = torch.tensor(distances_hop_np.reshape(n ** 2), dtype=torch.long)
    data.spatial_types = spatial_types

    t = time.perf_counter() - t
    # logging.info(f"computed shortest paths in: {t:.5f}")

    return data


class WeightedBiasEncoder(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_spatial_types: int,
        use_graph_token: bool,
        use_weighted_path_distance: bool,
    ):
        """Implementation of the bias encoder of Graphormer modified for attackable model

        Args:
            num_heads: The number of heads of the Graphormer model
            num_spatial_types: The total number of different spatial types
            num_edge_types: The total number of different edge types
            use_graph_token: If True, pads the attn_bias to account for the
            additional graph token that can be added by the ``NodeEncoder``.
        """
        super().__init__()
        self.num_heads = num_heads
        self.spatial_encoder = torch.nn.Embedding(num_spatial_types, num_heads)
        self.use_graph_token = use_graph_token
        if self.use_graph_token:
            self.graph_token = torch.nn.Parameter(torch.zeros(1, num_heads, 1))
        self.use_weighted_path_distance = use_weighted_path_distance
        self.reset_parameters()

    def reset_parameters(self):
        self.spatial_encoder.weight.data.normal_(std=0.02)
        if self.use_graph_token:
            self.graph_token.data.normal_(std=0.02)

    def forward(self, data):
        """Computes the bias matrix that can be induced into multi-head attention
        via the attention mask.

        Adds the tensor ``attn_bias`` to the data object, optionally accounting
        for the graph token.
        """
        # To convert 2D matrices to dense-batch mode, one needs to decompose
        # them into index and value. One example is the adjacency matrix
        # but this generalizes actually to any 2D matrix
        if self.use_weighted_path_distance and hasattr(data, "spatial_types_weights"):
            # TODO: multiply and sum like in degrees
            spatial_types: torch.Tensor = (
                data.spatial_types_weights[:, :, None] * self.spatial_encoder(data.spatial_types)
            ).sum(1)
        else:
            spatial_types: torch.Tensor = self.spatial_encoder(data.spatial_types)
        spatial_encodings = to_dense_adj(data.graph_index, data.batch, spatial_types)
        bias = spatial_encodings.permute(BATCH_HEAD_NODE_NODE)

        # during attack: adds bias for node's probability of being in graph
        if hasattr(data, "node_probs"):
            n = data.node_probs.size(0)
            assert n == bias.size(2) == bias.size(3)
            # if p is so small that log(p) = -inf, gradient is undefined, so just set -inf for very small p
            # TODO: or scale to be same as min!
            l = torch.zeros_like(data.node_probs) - torch.inf
            min_prob_mask = data.node_probs > 1e-30
            l[min_prob_mask] = data.node_probs[min_prob_mask].log()
            bias += l[None, None, None, :]

        if self.use_graph_token:
            bias = F.pad(bias, INSERT_GRAPH_TOKEN)
            bias[:, :, 1:, 0] = self.graph_token
            bias[:, :, 0, :] = self.graph_token
        B, H, N, _ = bias.shape
        data.attn_bias = bias.reshape(B * H, N, N)
        return data


class WeightedNodeEncoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        num_in_degree,
        num_out_degree,
        directed_graphs: bool,
        combinations_degree: bool,
        input_dropout=0.0,
        use_graph_token: bool = True,
    ):
        """Implementation of the node encoder of Graphormer.
        This encoder is based on the implementation at:
        https://github.com/microsoft/Graphormer/tree/v1.0
        Note that this refers to v1 of Graphormer.

        Args:
            embed_dim: The number of hidden dimensions of the model
            num_in_degree: Maximum size of in-degree to encode
            num_out_degree: Maximum size of out-degree to encode
            input_dropout: Dropout applied to the input features
            use_graph_token: If True, adds the graph token to the incoming batch.
        """
        super().__init__()
        self.directed_graphs = directed_graphs
        self.combinations_degree = combinations_degree
        if self.directed_graphs:
            self.in_degree_encoder = torch.nn.Embedding(num_in_degree, embed_dim)
            self.out_degree_encoder = torch.nn.Embedding(num_out_degree, embed_dim)
        else:
            max_degree = max(num_in_degree, num_out_degree)
            self.degree_encoder = torch.nn.Embedding(max_degree, embed_dim)
        self.use_graph_token = use_graph_token
        if self.use_graph_token:
            self.graph_token = torch.nn.Parameter(torch.zeros(1, embed_dim))
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.reset_parameters()

    def forward(self, data):
        if self.directed_graphs:
            if hasattr(data, "in_degrees"):
                # precomputed
                in_degree_encoding = self.in_degree_encoder(data.in_degrees)
                out_degree_encoding = self.out_degree_encoder(data.out_degrees)
            else:
                # continuous relaxation during attack
                if self.combinations_degree:
                    in_degree_encoding = data.degree_weights @ self.in_degree_encoder(data.degree_indices)
                    out_degree_encoding = data.degree_weights @ self.out_degree_encoder(data.degree_indices)
                else:
                    raise NotImplementedError
            degree_encoding = in_degree_encoding + out_degree_encoding
        else:
            if hasattr(data, "degrees"):
                # precomputed
                degree_encoding = self.degree_encoder(data.degrees)
            else:
                # continuous relaxation during attack
                if self.combinations_degree:
                    degree_encoding = data.degree_weights @ self.degree_encoder(data.degree_indices)
                else:
                    degree_encoding = (
                        data.degree_weights[:, :, None] * self.degree_encoder(data.degree_indices)
                    ).sum(1)

        if data.x.size(1) > 0:
            data.x = data.x + degree_encoding
        else:
            data.x = degree_encoding

        if self.use_graph_token:
            data = add_graph_token(data, self.graph_token)
        data.x = self.input_dropout(data.x)
        return data

    def reset_parameters(self):
        if self.directed_graphs:
            self.in_degree_encoder.weight.data.normal_(std=0.02)
            self.out_degree_encoder.weight.data.normal_(std=0.02)
        else:
            self.degree_encoder.weight.data.normal_(std=0.02)
        if self.use_graph_token:
            self.graph_token.data.normal_(std=0.02)


class WeightedPreprocessing(torch.nn.Module):
    def __init__(
        self,
        distance: int,
        directed_graphs: bool,
        combinations_degree: bool,
        num_in_degrees: int,
        num_out_degrees: int,
        use_weighted_path_distance: bool,
    ):
        """Implementation of the node encoder of Graphormer.
        This encoder is based on the implementation at:
        https://github.com/microsoft/Graphormer/tree/v1.0
        Note that this refers to v1 of Graphormer.

        Args:
            embed_dim: The number of hidden dimensions of the model
            num_in_degree: Maximum size of in-degree to encode
            num_out_degree: Maximum size of out-degree to encode
            input_dropout: Dropout applied to the input features
            use_graph_token: If True, adds the graph token to the incoming batch.
        """
        super().__init__()
        self.distance = distance
        self.directed_graphs = directed_graphs
        self.combinations_degree = combinations_degree
        self.num_in_degrees = num_in_degrees
        self.num_out_degrees = num_out_degrees
        self.use_weighted_path_distance = use_weighted_path_distance

    def forward(self, data):
        if hasattr(data, "recompute_preprocessing") and data.recompute_preprocessing:
            data = weighted_graphormer_pre_processing(
                data,
                self.distance,
                self.directed_graphs,
                self.combinations_degree,
                self.num_in_degrees,
                self.num_out_degrees,
                self.use_weighted_path_distance,
            )
        return data



@register_node_encoder("WeightedGraphormerBias")
class WeightedGraphormerEncoder(torch.nn.Sequential):
    def __init__(self, dim_emb, *args, **kwargs):
        encoders = [
            WeightedPreprocessing(
                cfg.posenc_GraphormerBias.num_spatial_types,
                cfg.posenc_GraphormerBias.directed_graphs,
                cfg.posenc_GraphormerBias.combinations_degree,
                cfg.posenc_GraphormerBias.num_in_degrees,
                cfg.posenc_GraphormerBias.num_out_degrees,
                cfg.posenc_GraphormerBias.use_weighted_path_distance,
            ),
            WeightedBiasEncoder(
                cfg.graphormer.num_heads,
                cfg.posenc_GraphormerBias.num_spatial_types,
                cfg.graphormer.use_graph_token,
                cfg.posenc_GraphormerBias.use_weighted_path_distance,
            ),
            WeightedNodeEncoder(
                dim_emb,
                cfg.posenc_GraphormerBias.num_in_degrees,
                cfg.posenc_GraphormerBias.num_out_degrees,
                cfg.posenc_GraphormerBias.directed_graphs,
                cfg.posenc_GraphormerBias.combinations_degree,
                cfg.graphormer.input_dropout,
                cfg.graphormer.use_graph_token
            ),
        ]
        if cfg.posenc_GraphormerBias.node_degrees_only:  # No attn. bias encoder
            encoders = encoders[1:]
        super().__init__(*encoders)
