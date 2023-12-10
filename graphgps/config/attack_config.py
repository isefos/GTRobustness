from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('attack')
def dataset_cfg(cfg):
    """Attack config options.
    """
    # example argument group
    cfg.attack = CN()

    # whether to attack or not
    cfg.attack.enabled = False

    # set to 0 to attack all, n for only the first n
    cfg.attack.num_attacked_graphs = 0

    # bla
    cfg.attack.lr = 4_000

    # bla
    cfg.attack.block_size = 2_000

    # bla
    cfg.attack.e_budget = 0.1

    # bla
    cfg.attack.existing_node_prob_multiplier = 1_000

    # bla
    cfg.attack.allow_existing_graph_pert = False

    # TODO: where to get bool: undirected from?
    cfg.attack.is_undirected = True

    # bla
    cfg.attack.loss = "train"

    # TODO: add much more (which dataset to attack, node_injection / edge_perturbation)...
