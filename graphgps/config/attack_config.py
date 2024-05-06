from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('attack')
def dataset_cfg(cfg):
    """Attack config options.
    """
    # for transfer attacks from the robustness unit test
    cfg.robustness_unit_test = CN()
    cfg.robustness_unit_test.enable = False
    cfg.robustness_unit_test.load_best_model = True

    # for adaptive attack
    cfg.attack = CN()

    # whether to attack or not
    cfg.attack.enable = False

    # whether to also run a random baseline attack
    cfg.attack.run_random_baseline = True

    # load the best validation model before attack or not
    cfg.attack.load_best_model = True

    # show progress bar or not
    cfg.attack.log_progress = True

    # whether to log all results (for each graph), or only the average in database observer
    # (can easily re-run with the saved perturbations to get per-graph results again)
    cfg.attack.only_return_avg = True

    # which split to attack, "train", "val", or "test"
    cfg.attack.split = "test"

    # set to 0 to attack all, n for only the first n
    cfg.attack.num_attacked_graphs = 0

    # bla
    cfg.attack.epochs = 125

    # bla
    cfg.attack.epochs_resampling = 100

    # how many gradient step updates before new edges are sampled
    cfg.attack.resample_period = 1

    # bla
    cfg.attack.max_final_samples = 20

    # bla
    cfg.attack.max_trials_sampling = 20
    
    # bla
    cfg.attack.with_early_stopping = True

    # bla
    cfg.attack.eps = 1e-7

    # Instead of initializing all new edges with eps, add some random variation
    cfg.attack.eps_init_noised = False

    # used for gradient clipping, to disable gradient clipping set to 0.0
    cfg.attack.max_edge_weight_update = 0.0

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

    # what the predictions are for, node or graph
    cfg.attack.prediction_level = "graph"

    # 'train', 'masked', 'margin', 'prob_margin', or 'tanh_margin' (or callable)
    cfg.attack.loss = "train"

    # None (set to same as loss), or 'neg_accuracy'
    cfg.attack.metric = None

    # is important for node injection attacks, where graph is huge, but only some nodes get added, rest is disconnected
    cfg.attack.remove_isolated_components = False

    # None or int -> give an int (e.g. 0) to define that the root node of a graph will always be on the given index (used for removing isolated components)
    cfg.attack.root_node_idx = None

    # do we want to compute the node probability approximation or not (more important for node injection attacks)
    cfg.attack.node_prob_enable = False

    # how many iterations of the node probability approximation computation to do
    cfg.attack.node_prob_iterations = 3

    # compute the node probability approximation directly (faster) or in log space (better for numerical stability)
    cfg.attack.node_prob_log = True

    # will not attack a graph which is already incorrectly classified (faster, but if we want to transfer attack should keep False)
    cfg.attack.skip_incorrect_graph_classification = False

    # specifically for the CLUSTER dataset, to not sample edges to labeled nodes
    cfg.attack.cluster_sampling = False

    # For node injection attacks (sampling all edges independently):
    cfg.attack.node_injection = CN()

    # set True to do a node injection attack
    cfg.attack.node_injection.enable = False

    # when doing node injection attack, include nodes from train split to consider for injection
    cfg.attack.node_injection.from_train = True

    # when doing node injection attack, include nodes from val split to consider for injection
    cfg.attack.node_injection.from_val = True

    # when doing node injection attack, include nodes from test split to consider for injection
    cfg.attack.node_injection.from_test = True

    # whether the existing graph edges can be changed, or only new edges added
    cfg.attack.node_injection.allow_existing_graph_pert = True

    # sample only edges from existing nodes to new nodes, not from new to new
    cfg.attack.node_injection.sample_only_connected = False

    # when also sampling new-new edges (may be much more), can set a higher weight to sample edges from existing nodes (often minority, but more useful)
    cfg.attack.node_injection.existing_node_prob_multiplier = 1

    # for some dataset (e.g. UPFD) the root nodes are special, and each graph should only have one, therefore shouldn't be included for injection
    cfg.attack.node_injection.include_root_nodes = True

    # for tree datasets, when we inject a new node, we need to make sure it still has a tree structure
    cfg.attack.node_injection.sample_only_trees = False

    # node sampling, sample the nodes to inject first, then the edges to those nodes, more efficient in the sense that can sample more edges while adding less nodes
    cfg.attack.node_injection.node_sampling = CN()

    cfg.attack.node_injection.node_sampling.enable = False

    cfg.attack.node_injection.node_sampling.min_add_nodes = 100

    cfg.attack.node_injection.node_sampling.min_total_nodes = 1000

    cfg.attack.node_injection.node_sampling.max_block_size = 20_000

    # for transfer attack
    cfg.attack.transfer = CN()
    cfg.attack.transfer.enable = False
    cfg.attack.transfer.perturbation_path = ""
