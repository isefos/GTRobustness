import datetime
import os
import torch
import logging

import graphgps  # noqa, register custom modules
from graphgps.agg_runs import agg_runs
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger
from graphgps.attack.attack import prbcd_attack_test_dataset


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()
        model = create_model()
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head, seed=cfg.seed
            )
        optimizer = create_optimizer(model.parameters(),
                                     new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the "
                                "default train.mode, set it to `custom`")
            datamodule = GraphGymDataModule()
            train(model, datamodule, logger=True)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
            

        # ATTACK:
        # (temporary hacky way of adding attack to every run,
        #  will only work for UPDF and weighted / generalized models though... Work in Progress)
            
        # TODO: add all of this to a separate function / class and make it enabled / disabled by config
        #  also add all parameters to config
        n_attacks = 20
        attack_lr = 4_000  # 4_000
        attack_b = 10_000
        attack_e = 0.3  # 0.3
        existing_node_prob_multiplier = 10.
        allow_existing_graph_pert = False  # False
        # TODO: where to get bool: undirected from?
        is_undirected = True

        # TODO: add this to the actual UPFD dataset (so it gets downloaded directly)
        dataset_name = "politifact"
        data_path = os.path.join(os.getcwd(), "datasets", "UPFD")

        # TODO: to run in current state, must manually download these files and put into correct dir
        # from: https://github.com/safe-graph/GNN-FakeNews/blob/main/data/gos_id_twitter_mapping.pkl
        # and https://github.com/safe-graph/GNN-FakeNews/blob/main/data/pol_id_twitter_mapping.pkl
        id_mapping_files = {
            'politifact': os.path.join(data_path, "pol_id_twitter_mapping.pkl"),
            'gossipcop': os.path.join(data_path, "gos_id_twitter_mapping.pkl"),
        }
        id_mapping_path = id_mapping_files[dataset_name]
        raw_data_path = os.path.join(data_path, dataset_name, "raw")
        graph_indices_paths = {
            "train": os.path.join(raw_data_path, "train_idx.npy"),
            "val": os.path.join(raw_data_path, "val_idx.npy"),
            "test": os.path.join(raw_data_path, "test_idx.npy"),
        }

        # TODO: attack loss wrapper for normal training loss, put into attack, and let it be configurable 
        def attack_loss(pred, true, node_cls_mask):
            return compute_loss(pred, true)[0]
        
        # TODO: instead of giving train val test, 
        # already compute total nodes x before and pass, 
        # then just pass attack dataset
        
        results = prbcd_attack_test_dataset(
            model=model,
            datasets={split: l.dataset for split, l in zip(["train", "val", "test"], loaders)},
            device=torch.device(cfg.accelerator),
            attack_loss=attack_loss,
            id_mapping_path=id_mapping_path,
            graph_indices_paths=graph_indices_paths,
            limit_number_attacks=n_attacks,
            e_budget=attack_e,
            block_size=attack_b,
            lr=attack_lr,
            is_undirected=is_undirected,
            sigmoid_threshold=cfg.model.thresh,
            existing_node_prob_multiplier=existing_node_prob_multiplier,
            allow_existing_graph_pert=allow_existing_graph_pert,
        )
        print(
            f"PRBCD: Accuracy clean: {results['clean_acc']:.3f},  Perturbed: {results['pert_acc']:.3f}.\n"
            f"Average number of edges added: {sum(results['num_edges_added']) / len(results['num_edges_added']):.3f}\n"
            f"Average number of edges removed: "
            f"{sum(results['num_edges_removed']) / len(results['num_edges_removed']):.3f}\n"
            f"Average number of nodes added: {sum(results['num_nodes_added']) / len(results['num_nodes_added']):.3f}\n"
            f"Average number of nodes removed: "
            f"{sum(results['num_nodes_removed']) / len(results['num_nodes_removed']):.3f}\n"
            f"Nodes most frequently added - (freq, global_node_index): {results['most_added_nodes'][:10]}\n"
        )


    # Aggregate results from different seeds
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")
