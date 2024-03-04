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

    # whether to also run a random baseline attack
    cfg.attack.run_random_baseline = True

    # load the best validation model before attack or not
    cfg.attack.load_best_model = True

    # show progress bar or not
    cfg.attack.log_progress = True

    # which split to attack, "train", "val", or "test"
    cfg.attack.split = "test"

    # set to 0 to attack all, n for only the first n
    cfg.attack.num_attacked_graphs = 0

    # bla
    cfg.attack.epochs = 125

    # bla
    cfg.attack.epochs_resampling = 100

    # bla
    cfg.attack.max_final_samples = 20

    # bla
    cfg.attack.max_trials_sampling = 20
    
    # bla
    cfg.attack.with_early_stopping = True

    # bla
    cfg.attack.eps = 1e-7

    # bla
    cfg.attack.lr = 4_000

    # bla
    cfg.attack.block_size = 2_000

    # bla
    cfg.attack.e_budget = 0.1

    # bla
    cfg.attack.minimum_budget = 0

    # bla
    cfg.attack.is_undirected = True

    # 'train', 'masked', 'margin', 'prob_margin', or 'tanh_margin' (or callable)
    cfg.attack.loss = "train"

    # None (set to same as loss), 'train', 'masked', 'margin', 'prob_margin', or 'tanh_margin' (or callable)
    cfg.attack.metric = None

    # how many iterations of the node probability approximation computation to do
    cfg.attack.iterations_node_prob = 5

    # bla
    cfg.attack.skip_incorrect_graph_classification = True

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
    cfg.attack.root_node_idx = None

    # bla
    cfg.attack.include_root_nodes_for_injection = True

    # bla
    cfg.attack.sample_only_connected = False

    # bla
    cfg.attack.sample_only_trees = False
