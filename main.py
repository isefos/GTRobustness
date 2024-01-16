import datetime
import os
import torch
import logging
import seml
import yaml
from yacs.config import CfgNode
from argparse import Namespace
from sacred import Experiment

import graphgps  # noqa, register custom modules
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from torch_geometric.graphgym.config import (
    cfg, dump_cfg, set_cfg, load_cfg,
)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.optim import (
    create_optimizer, create_scheduler, OptimizerConfig,
)
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything
from graphgps.finetuning import (
    load_pretrained_model_cfg, init_model_from_pretrained,
)
from graphgps.logger import create_logger
from graphgps.attack.attack import prbcd_attack_dataset


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def new_optimizer_config(cfg):
    return OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum
    )


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period,
    )


def get_attack_cfg(loaders):
    splits = ["train", "val", "test"]
    split_to_attack_idx = splits.index(cfg.attack.split)
    dataset_to_attack = loaders[split_to_attack_idx].dataset

    additional_injection_datasets = None
    inject_nodes_from_attack_dataset = False

    if cfg.attack.enable_node_injection:
        include_additional_datasets = [
            cfg.attack.node_injection_from_train,
            cfg.attack.node_injection_from_val,
            cfg.attack.node_injection_from_test,
        ]
        inject_nodes_from_attack_dataset = include_additional_datasets[split_to_attack_idx]
        include_additional_datasets[split_to_attack_idx] = False
        additional_injection_datasets = [l.dataset for i, l in enumerate(loaders) if include_additional_datasets[i]]

    return dataset_to_attack, additional_injection_datasets, inject_nodes_from_attack_dataset

    
def main(cfg):
    # Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    seed_everything(cfg.seed)
    auto_select_device()
    
    # Finetuning / loading pretrained
    if cfg.pretrained.dir:
        cfg = load_pretrained_model_cfg(cfg)

    # Machine learning pipeline
    loaders = create_loader()
    loggers = create_logger()
    model = create_model()
    if cfg.pretrained.dir:
        model = init_model_from_pretrained(
            model, cfg.pretrained.dir, cfg.pretrained.freeze_main, cfg.pretrained.reset_prediction_head, seed=cfg.seed,
        )
    optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
    scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

    # Print model info
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)
    logging.info(f"[*] Starting now: {datetime.datetime.now()}, with seed={cfg.seed}, running on {cfg.accelerator}")

    # Train
    if cfg.train.mode == 'standard':
        if cfg.wandb.use:
            logging.warning("[W] WandB logging is not supported with the "
                            "default train.mode, set it to `custom`")
        datamodule = GraphGymDataModule()
        train(model, datamodule, logger=True)
    else:
        train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

    # Attack
    if cfg.attack.enable:
        dataset_to_attack, additional_injection_datasets, inject_nodes_from_attack_dataset = get_attack_cfg(loaders)
        # TODO: if specified, load best model checkpoint before attack
        # TODO: return results
        prbcd_attack_dataset(
            model=model,
            dataset_to_attack=dataset_to_attack,
            node_injection_attack=cfg.attack.enable_node_injection,
            additional_injection_datasets=additional_injection_datasets,
            inject_nodes_from_attack_dataset=inject_nodes_from_attack_dataset,
            device=torch.device(cfg.accelerator),
            attack_loss=cfg.attack.loss,
            num_attacked_graphs=cfg.attack.num_attacked_graphs,
            e_budget=cfg.attack.e_budget,
            block_size=cfg.attack.block_size,
            lr=cfg.attack.lr,
            is_undirected=cfg.attack.is_undirected,
            sigmoid_threshold=cfg.model.thresh,
            existing_node_prob_multiplier=cfg.attack.existing_node_prob_multiplier,
            allow_existing_graph_pert=cfg.attack.allow_existing_graph_pert,
            remove_isolated_components=cfg.attack.remove_isolated_components,
            root_node_idx=cfg.attack.root_node_idx,
            include_root_nodes_for_injection=cfg.attack.include_root_nodes_for_injection,
        )

    logging.info(f"[*] Finished now: {datetime.datetime.now()}")

    # TODO: return results
    results = {"example": 10}
    return results


ex = Experiment()
seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


def convert_cfg_to_dict(cfg_node):
    cfg_dict = {}
    for k, v in cfg_node.items():
        if isinstance(v, CfgNode):
            cfg_dict[k] = convert_cfg_to_dict(v)
        else:
            cfg_dict[k] = v
    return cfg_dict


set_cfg(cfg)
cfg_dict = convert_cfg_to_dict(cfg)
ex.add_config({"graphgym": cfg_dict})


@ex.automain
def run(graphgym: dict):
    set_cfg(cfg)

    seed = graphgym.get("seed", cfg.seed)
    ex_identifier = (
        graphgym["dataset"]["format"]
        + "-" + graphgym["dataset"]["name"]
        + "-"+ graphgym["model"]["type"]
    )
    output_dir = os.path.join(graphgym["out_dir"], ex_identifier)

    run_identifier = f"s{seed}-{datetime.datetime.now().strftime('d%Y%m%d-t%H%M%S')}"
    run_dir = os.path.join(output_dir, run_identifier)
    os.makedirs(run_dir)

    graphgym_cfg_file = os.path.join(run_dir, "configs_from_seml.yaml")
    with open(graphgym_cfg_file, 'w') as f:
        yaml.dump(graphgym, f)
    args = Namespace(cfg_file=str(graphgym_cfg_file), opts=[])

    cfg.out_dir = output_dir
    cfg.run_dir = run_dir
    cfg.run_id = ex_identifier + "-" + run_identifier
    cfg.cfg_dest = run_identifier + "/configs_all.yaml"

    load_cfg(cfg, args)
    dump_cfg(cfg)

    return main(cfg)
