from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_graphormer')
def set_cfg_gt(cfg):
    cfg.graphormer = CN()
    cfg.graphormer.num_layers = 6
    cfg.graphormer.embed_dim = 80
    cfg.graphormer.num_heads = 4
    cfg.graphormer.dropout = 0.0
    cfg.graphormer.attention_dropout = 0.0
    cfg.graphormer.mlp_dropout = 0.0
    cfg.graphormer.input_dropout = 0.0
    cfg.graphormer.use_graph_token = True

    cfg.posenc_GraphormerBias = CN()
    cfg.posenc_GraphormerBias.enable = False
    cfg.posenc_GraphormerBias.node_degrees_only = False
    # I think dim_pe = 0 is only used for composed_encoders.py because here the PEs are not concatenated:
    cfg.posenc_GraphormerBias.dim_pe = 0
    cfg.posenc_GraphormerBias.num_spatial_types = None
    cfg.posenc_GraphormerBias.num_in_degrees = None
    cfg.posenc_GraphormerBias.num_out_degrees = None
    cfg.posenc_GraphormerBias.directed_graphs = False
    cfg.posenc_GraphormerBias.has_edge_attr = True
    cfg.posenc_GraphormerBias.use_weighted_degrees = True
    cfg.posenc_GraphormerBias.combinations_degree = False
    # use reciprocal edge weight to find the shortest paths, if False, will ignore `sp_use_weighted` and `sp_use_gradient`
    cfg.posenc_GraphormerBias.sp_find_weighted = True
    # use the weighted distances for the found paths (if False will use the hop distance, and will ignore `sp_use_gradient`)
    cfg.posenc_GraphormerBias.sp_use_weighted = True
    # use the gradient of the weighted shortest path distances
    cfg.posenc_GraphormerBias.sp_use_gradient = True
