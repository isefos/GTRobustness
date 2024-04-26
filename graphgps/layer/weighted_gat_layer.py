import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.utils import softmax


class WeightedGATConvLayer(pyg_nn.GATConv):
    """Extended GAT to allow for weighted edges."""
    def edge_update(
        self,
        alpha_j: torch.Tensor,
        alpha_i: None | torch.Tensor,
        edge_attr: None | torch.Tensor,
        index: torch.Tensor,
        ptr: None | torch.Tensor,
        size_i: None | int,
    ) -> torch.Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        if edge_attr is not None:
            assert edge_attr.dim() == 1, 'Only scalar edge weights supported'
            edge_attr = edge_attr.view(-1, 1)
            # `alpha` unchanged if edge_attr == 1 and -Inf if edge_attr == 0;
            # We choose log to counteract underflow in subsequent exp/softmax
            alpha = alpha + torch.log2(edge_attr)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


@register_layer('gatconvweighted')
class WeightedGATConvGraphGymLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        heads = cfg.gnn.att_heads
        per_head_dim_out = layer_config.dim_out // heads
        if (heads * per_head_dim_out) != layer_config.dim_out:
            raise ValueError(
                f"The given hidden channel size ({layer_config.dim_out}) is not divisible by the number of "
                f"attention heads ({heads}). Hidden channel size was set to {per_head_dim_out * heads}."
            )
        self.model = WeightedGATConvLayer(
            layer_config.dim_in,
            per_head_dim_out,
            bias=layer_config.has_bias,
            heads=heads,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch
