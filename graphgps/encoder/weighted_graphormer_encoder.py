import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.utils import to_dense_adj, to_networkx
from itertools import combinations

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


def weighted_graphormer_pre_processing(data, distance):
    """Implementation of Graphormer pre-processing. Computes in- and out-degrees
    for node encodings, as well as spatial types (via shortest-path lengths) and
    prepares edge encodings along shortest paths. The function adds the following
    properties to the data object:

    - spatial_types
    - graph_index: An edge_index type tensor that contains all possible directed edges 
                  (see more below)
    - shortest_path_types: Populates edge attributes along all shortest paths between two nodes

    Similar to the adjacency matrix, any matrix can be batched in PyG by decomposing it
    into a 1D tensor of values and a 2D tensor of indices. Once batched, the graph-specific
    matrix can be recovered (while appropriately padded) via ``to_dense_adj``. We use this 
    concept to decompose the spatial type matrix and the shortest path edge type tensor
    via the ``graph_index`` tensor.

    Args:
        data: A PyG data object holding a single graph
        distance: The distance up to which types are calculated

    Returns:
        The augmented data object.
    """
    # data.in_degrees = torch.tensor([d for _, d in graph.in_degree()])
    # data.out_degrees = torch.tensor([d for _, d in graph.out_degree()])

    # TODO: only undirected implemented for now, add directed later
    # TODO: handle too large degree -> clamping to max
    data.degree_indices, data.degree_weights = node_degree_weights_undirected(
        data.x.size(0), data.edge_index, data.edge_attr,
    )

    if cfg.posenc_GraphormerBias.node_degrees_only:
        return data

    data.w_inv = data.edge_attr.reciprocal()
    graph: nx.DiGraph = to_networkx(data, edge_attrs=["w_inv"])

    N = len(graph.nodes)
    shortest_paths = nx.shortest_path(graph, weight="w_inv")

    # if we want the weighted path lengths (interpolate, get gradients for w)
    # shortest_paths_lengths = dict(nx.all_pairs_shortest_path_length(graph)))

    spatial_types = torch.empty(N ** 2, dtype=torch.long).fill_(distance)
    graph_index = torch.empty(2, N ** 2, dtype=torch.long)

    for i in range(N):
        for j in range(N):
            graph_index[0, i * N + j] = i
            graph_index[1, i * N + j] = j

    for i, paths in shortest_paths.items():
        for j, path in paths.items():
            if len(path) > distance:
                path = path[:distance]

            assert len(path) >= 1
            spatial_types[i * N + j] = len(path) - 1

    data.spatial_types = spatial_types
    data.graph_index = graph_index
    return data


class WeightedBiasEncoder(torch.nn.Module):
    def __init__(self, num_heads: int, num_spatial_types: int,
                 num_edge_types: int, use_graph_token: bool = True):
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

        # Takes into account disconnected nodes
        self.spatial_encoder = torch.nn.Embedding(
            num_spatial_types + 1, num_heads)

        self.use_graph_token = use_graph_token
        if self.use_graph_token:
            self.graph_token = torch.nn.Parameter(torch.zeros(1, num_heads, 1))
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
        spatial_types: torch.Tensor = self.spatial_encoder(data.spatial_types)
        spatial_encodings = to_dense_adj(data.graph_index,
                                         data.batch,
                                         spatial_types)
        bias = spatial_encodings.permute(BATCH_HEAD_NODE_NODE)

        if self.use_graph_token:
            bias = F.pad(bias, INSERT_GRAPH_TOKEN)
            bias[:, :, 1:, 0] = self.graph_token
            bias[:, :, 0, :] = self.graph_token

        B, H, N, _ = bias.shape
        data.attn_bias = bias.reshape(B * H, N, N)

        # TODO: add bias for node ex prob
        if hasattr(data, "node_probs"):
            assert N == data.node_probs.size(0)
            data.attn_bias += torch.log2(data.node_probs.repeat(N, 1))[None, :, :]

        return data


def add_graph_token(data, token):
    """Helper function to augment a batch of PyG graphs
    with a graph token each. Note that the token is
    automatically replicated to fit the batch.

    Args:
        data: A PyG data object holding a single graph
        token: A tensor containing the graph token values

    Returns:
        The augmented data object.
    """
    B = len(data.batch.unique())
    tokens = torch.repeat_interleave(token, B, 0)
    data.x = torch.cat([tokens, data.x], 0)
    data.batch = torch.cat(
        [torch.arange(0, B, device=data.x.device, dtype=torch.long), data.batch]
    )
    data.batch, sort_idx = torch.sort(data.batch)
    data.x = data.x[sort_idx]
    # TODO: check if works 
    if hasattr(data, "node_probs"):
        # add 1 as prob of graph token
        data.node_probs = torch.cat([torch.ones(B), data.node_probs], 0)
        data.node_probs = data.node_probs[sort_idx]
    return data


class WeightedNodeEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_in_degree, num_out_degree,
                 input_dropout=0.0, use_graph_token: bool = True):
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
        self.in_degree_encoder = torch.nn.Embedding(num_in_degree, embed_dim)
        self.out_degree_encoder = torch.nn.Embedding(num_out_degree, embed_dim)

        self.use_graph_token = use_graph_token
        if self.use_graph_token:
            self.graph_token = torch.nn.Parameter(torch.zeros(1, embed_dim))
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.reset_parameters()

    def forward(self, data):
        if hasattr(data, "in_degrees"):
            in_degree_encoding = self.in_degree_encoder(data.in_degrees)
            out_degree_encoding = self.out_degree_encoder(data.out_degrees)
        else:
            in_degree_encoding = data.degree_weights @ self.in_degree_encoder(data.degree_indices)
            out_degree_encoding = data.degree_weights @ self.out_degree_encoder(data.degree_indices)

        if data.x.size(1) > 0:
            data.x = data.x + in_degree_encoding + out_degree_encoding
        else:
            data.x = in_degree_encoding + out_degree_encoding

        if self.use_graph_token:
            data = add_graph_token(data, self.graph_token)
        data.x = self.input_dropout(data.x)
        return data

    def reset_parameters(self):
        self.in_degree_encoder.weight.data.normal_(std=0.02)
        self.out_degree_encoder.weight.data.normal_(std=0.02)
        if self.use_graph_token:
            self.graph_token.data.normal_(std=0.02)


class WeightedPreprocessing(torch.nn.Module):
    def __init__(self, distance):
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

    def forward(self, data):
        if not hasattr(data, "in_degrees"):
            data = weighted_graphormer_pre_processing(data, self.distance)
        return data



@register_node_encoder("WeightedGraphormerBias")
class WeightedGraphormerEncoder(torch.nn.Sequential):
    def __init__(self, dim_emb, *args, **kwargs):
        encoders = [
            WeightedPreprocessing(cfg.posenc_GraphormerBias.num_spatial_types),
            WeightedBiasEncoder(
                cfg.graphormer.num_heads,
                cfg.posenc_GraphormerBias.num_spatial_types,
                cfg.dataset.edge_encoder_num_types,
                cfg.graphormer.use_graph_token
            ),
            WeightedNodeEncoder(
                dim_emb,
                cfg.posenc_GraphormerBias.num_in_degrees,
                cfg.posenc_GraphormerBias.num_out_degrees,
                cfg.graphormer.input_dropout,
                cfg.graphormer.use_graph_token
            ),
        ]
        if cfg.posenc_GraphormerBias.node_degrees_only:  # No attn. bias encoder
            encoders = encoders[1:]
        super().__init__(*encoders)
