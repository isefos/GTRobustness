import numpy  # noqa, fixes mkl error
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


def get_attack_datasets(loaders):
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
    assert cfg.train.mode != 'standard', "Default train.mode not supported, use `custom` (or other specific mode)"
    train_results = train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

    # Attack
    attack_results = None
    if cfg.attack.enable:

        if cfg.attack.load_best_model:
            assert cfg.train.enable_ckpt and cfg.train.ckpt_best, (
                "To load best model, enable checkpointing and set ckpt_best"
            )
            # load best model checkpoint before attack
            from torch_geometric.graphgym.checkpoint import MODEL_STATE

            ckpt_file = os.path.join(cfg.run_dir, "ckpt", f"{train_results['best_val_epoch']}.ckpt")
            ckpt = torch.load(ckpt_file, map_location=torch.device('cpu'))
            best_model_dict = ckpt[MODEL_STATE]
            model_dict = model.state_dict()
            model_dict.update(best_model_dict)
            model.load_state_dict(model_dict)

        attack_dataset, injection_datasets, inject_nodes_from_attack_dataset = get_attack_datasets(loaders)
        
        attack_results = prbcd_attack_dataset(
            model=model,
            dataset_to_attack=attack_dataset,
            node_injection_attack=cfg.attack.enable_node_injection,
            additional_injection_datasets=injection_datasets,
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
            sample_only_trees=cfg.attack.sample_only_trees,
        )

    logging.info(f"[*] Finished now: {datetime.datetime.now()}")
    results = {"train": train_results, "attack": attack_results}
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


def convert_readonly_to_dict(readonly_dict):
    new_dict = {}
    for k, v in readonly_dict.items():
        if isinstance(v, dict):
            new_dict[k] = convert_readonly_to_dict(v)
        else:
            new_dict[k] = v
    return new_dict


set_cfg(cfg)
cfg_dict = convert_cfg_to_dict(cfg)
ex.add_config({"graphgym": cfg_dict, "dims_per_head": 0})


os.makedirs("configs_seml/logs", exist_ok=True)


@ex.automain
def run(seed, graphgym, dims_per_head: int):
    graphgym = convert_readonly_to_dict(graphgym)
    model_type = graphgym["model"]["type"]
    if dims_per_head > 0 and model_type in ["Graphormer"] and graphgym["gnn"]["dim_inner"] == 0:
        if model_type == "Graphormer":
            dim_inner = dims_per_head * graphgym["graphormer"]["num_heads"]
            graphgym["graphormer"]["embed_dim"] = dim_inner
            graphgym["gnn"]["dim_inner"] = dim_inner
        else:
            raise NotImplementedError(f"Please add a case for {model_type} (very easy)!")
        
    set_cfg(cfg)

    ex_identifier = (
        graphgym["dataset"]["format"]
        + "-" + graphgym["dataset"]["name"]
        + "-"+ graphgym["model"]["type"]
    )
    output_dir = os.path.join(graphgym["out_dir"], ex_identifier)
    os.makedirs(output_dir, exist_ok=True)
    graphgym["out_dir"] = output_dir

    seed_graphgym = graphgym.get("seed", cfg.seed)
    run_identifier = f"s{seed_graphgym}-{datetime.datetime.now().strftime('d%Y%m%d-t%H%M%S%f')}-{seed}"
    run_dir = os.path.join(output_dir, run_identifier)
    os.makedirs(run_dir)

    graphgym_cfg_file = os.path.join(run_dir, "configs_from_seml.yaml")
    with open(graphgym_cfg_file, 'w') as f:
        yaml.dump(graphgym, f)
    args = Namespace(cfg_file=str(graphgym_cfg_file), opts=[])

    load_cfg(cfg, args)

    cfg.run_dir = run_dir
    cfg.cfg_dest = f"{run_identifier}/config.yaml"

    dump_cfg(cfg)

    return main(cfg)
