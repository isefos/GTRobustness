from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('attack')
def dataset_cfg(cfg):
    """Attack config options.
    """
    # example argument group
    cfg.attack = CN()

    # whether to attack or not
    cfg.attack.enable = False

    # load the best validation model before attack or not
    cfg.attack.load_best_model = True

    # which split to attack, "train", "val", or "test"
    cfg.attack.split = "test"

    # set to 0 to attack all, n for only the first n
    cfg.attack.num_attacked_graphs = 0

    # bla
    cfg.attack.lr = 4_000

    # bla
    cfg.attack.block_size = 2_000

    # bla
    cfg.attack.e_budget = 0.1

    # bla
    cfg.attack.is_undirected = True

    # bla
    cfg.attack.loss = "train"

    # set True to do a node injection attack
    cfg.attack.enable_node_injection = False

    # when doing node injection attack, include nodes from train split to consider for injection
    cfg.attack.node_injection_from_train = True

    # when doing node injection attack, include nodes from val split to consider for injection
    cfg.attack.node_injection_from_val = True

    # when doing node injection attack, include nodes from test split to consider for injection
    cfg.attack.node_injection_from_test = True

    # bla
    cfg.attack.existing_node_prob_multiplier = 1

    # bla
    cfg.attack.allow_existing_graph_pert = True

    # bla
    cfg.attack.remove_isolated_components = True

    # None or int -> give an int (e.g. 0) to tell the execution that the root node of a graph will always be on the given index
    cfg.attack.root_node_idx: None | int = None

    # bla
    cfg.attack.include_root_nodes_for_injection = True

    # bla
    cfg.attack.sample_only_connected = False

    # bla
    cfg.attack.sample_only_trees = False
