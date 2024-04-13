import numpy  # noqa, fixes mkl error
import datetime
import os
import torch
import logging
import yaml
from yacs.config import CfgNode
from argparse import Namespace
from seml.experiment import Experiment
from seml.config import read_config, generate_configs

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
from torch_geometric.graphgym.checkpoint import MODEL_STATE
from torch_geometric import seed_everything
from graphgps.finetuning import (
    load_pretrained_model_cfg, init_model_from_pretrained,
)
from graphgps.logger import create_logger
from graphgps.attack.attack import prbcd_attack_dataset
from graphgps.attack.transfer_unit_test import transfer_unit_test
from graphgps.loader.dataset.robust_unittest import RobustnessUnitTest


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


def load_model(model, ckpt_file):
    ckpt = torch.load(ckpt_file, map_location=torch.device('cpu'))
    best_model_dict = ckpt[MODEL_STATE]
    model_dict = model.state_dict()
    model_dict.update(best_model_dict)
    model.load_state_dict(model_dict)
    return model


def load_best_val_model(model, training_results):
    assert cfg.train.enable_ckpt and cfg.train.ckpt_best, (
        "To load best model, enable checkpointing and set ckpt_best"
    )
    # load best model checkpoint before attack
    ckpt_file = os.path.join(cfg.run_dir, "ckpt", f"{training_results['best_val_epoch']}.ckpt")
    return load_model(model, ckpt_file)


def initialize_run(cfg):
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
        if cfg.pretrained.finetune:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main, cfg.pretrained.reset_prediction_head, seed=cfg.seed,
            )
        else:
            ckpt_file = os.path.join(cfg.pretrained.dir, "best.ckpt")
            logging.info(f"Loading saved model (from {ckpt_file})")
            model = load_model(model, ckpt_file)
    optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
    scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

    # Print model info
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)
    logging.info(f"[*] Starting now: {datetime.datetime.now()}, with seed={cfg.seed}, running on {cfg.accelerator}")
    return loaders, loggers, model, optimizer, scheduler

    
def main(cfg):
    loaders, loggers, model, optimizer, scheduler = initialize_run(cfg)

    # Train
    if cfg.pretrained.dir and not cfg.pretrained.finetune:
        training_results = None
    else:
        assert cfg.train.mode != 'standard', "Default train.mode not supported, use `custom` (or other specific mode)"
        training_results = train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

    # Robustness unit test
    rut_results = None
    if cfg.robustness_unit_test.enable:
        assert isinstance(loaders[0].dataset, RobustnessUnitTest), (
            "To run the robustness unit test, the model should be trained on a "
            "corresponding RobustnessUnitTest dataset,"
        )
        if cfg.robustness_unit_test.load_best_model:
            logging.info(f"Loading best val. model (from epoch {training_results['best_val_epoch']})")
            model = load_best_val_model(model, training_results)
        logging.info("Running robustness unit test")
        rut_results = transfer_unit_test(model, loaders[2])

    # Attack
    attack_results = None
    if cfg.attack.enable:
        if cfg.attack.load_best_model:
            logging.info(f"Loading best val. model before attack (from epoch {training_results['best_val_epoch']})")
            model = load_best_val_model(model, training_results)
        attack_results = prbcd_attack_dataset(model, loaders)

    logging.info(f"[*] Finished now: {datetime.datetime.now()}")
    results = {"training": training_results, "attack": attack_results, "robustness_unit_test": rut_results}
    return results


ex = Experiment()


# TODO: there is already a function in graphgps.utils that does this
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
ex.add_config({"graphgym": cfg_dict, "dims_per_head": 0, "dims_per_head_PE": 0})


os.makedirs("configs_seml/logs", exist_ok=True)


@ex.automain
def run(seed, graphgym, dims_per_head: int, dims_per_head_PE: int):
    graphgym = convert_readonly_to_dict(graphgym)
    setup_run(graphgym, dims_per_head, dims_per_head_PE, seed)
    results = main(cfg)
    results["run_dir"] = str(cfg.run_dir)
    results["num_params"] = cfg.params
    return results


def setup_run(graphgym, dims_per_head, dims_per_head_PE, seml_seed=None, jupyter=False):
    # calculate the composed configs
    model_type = graphgym["model"]["type"]
    if dims_per_head > 0 and graphgym["gnn"]["dim_inner"] == 0:
        if model_type == "Graphormer":
            dim_inner = dims_per_head * graphgym["graphormer"]["num_heads"]
            graphgym["graphormer"]["embed_dim"] = dim_inner
        elif model_type in ["SANTransformer", "WeightedSANTransformer", "GritTransformer"]:
            dim_inner = dims_per_head * graphgym["gt"]["n_heads"]
            graphgym["gt"]["dim_hidden"] = dim_inner
        else:
            raise NotImplementedError(f"Please add a case for {model_type} (very easy)!")
        graphgym["gnn"]["dim_inner"] = dim_inner
    if dims_per_head_PE > 0 and graphgym["posenc_WLapPE"]["dim_pe"] == 0:
        dim_pe = dims_per_head_PE * graphgym["posenc_WLapPE"]["n_heads"]
        graphgym["posenc_WLapPE"]["dim_pe"] = dim_pe
    # set defaults
    set_cfg(cfg)
    # find and set the paths
    if jupyter:
        graphgym["out_dir"] += "-jupyter"
    run_type = "train"
    if graphgym.get("robustness_unit_test", {}).get("enable", False):
        run_type = "rut"
    elif graphgym.get("attack", {}).get("enable", False):
        run_type = "attack"
    output_dir = os.path.join(
        graphgym["out_dir"],
        model_type,
        graphgym["dataset"]["format"] + "-" + graphgym["dataset"]["name"],
        run_type
    )
    os.makedirs(output_dir, exist_ok=True)
    graphgym["out_dir"] = output_dir
    seed_graphgym = graphgym.get("seed", cfg.seed)
    run_identifier = f"s{seed_graphgym}-{datetime.datetime.now().strftime('d%Y%m%d-t%H%M%S%f')}"
    if seml_seed is not None:
        run_identifier += f"-{seml_seed}"
    run_dir = os.path.join(output_dir, run_identifier)
    os.makedirs(run_dir)
    # save the config and load using YACS
    graphgym_cfg_file = os.path.join(run_dir, "configs_from_seml.yaml")
    with open(graphgym_cfg_file, 'w') as f:
        yaml.dump(graphgym, f)
    args = Namespace(cfg_file=str(graphgym_cfg_file), opts=[])
    load_cfg(cfg, args)
    # set last configs and dump final
    cfg.run_dir = run_dir
    cfg.cfg_dest = f"{run_identifier}/config.yaml"
    dump_cfg(cfg)


def _convert_numpy_to_float(cfg_with_np: dict):
    for k, v in cfg_with_np.items():
        if isinstance(v, dict):
            _convert_numpy_to_float(v)
        elif isinstance(v, numpy.floating):
            cfg_with_np[k] = float(v)


def setup_jupyter(cfg_path, exp_index: int = 0):
    _, _, exp_cfg = read_config(cfg_path)
    exp_cfgs = generate_configs(exp_cfg)
    num_exp = len(exp_cfgs)
    if exp_index >= num_exp:
        raise ValueError(f"Given exp_index={exp_index}, but config only generates {num_exp} experiments.")
    g_cfg = exp_cfgs[exp_index]["graphgym"]
    _convert_numpy_to_float(g_cfg)
    dims_per_head = exp_cfgs[exp_index].get("dims_per_head", 0)
    dims_per_head_PE = exp_cfgs[exp_index].get("dims_per_head_PE", 0)
    setup_run(g_cfg, dims_per_head, dims_per_head_PE, jupyter=False)
    loaders, loggers, model, optimizer, scheduler = initialize_run(cfg)
    return loaders, loggers, model, optimizer, scheduler
