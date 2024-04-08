from graphgps.loader.dataset.robust_unittest import RUTAttack
from torch_geometric.graphgym.config import cfg
from functools import partial
from graphgps.transform.transforms import pre_transform_in_memory
from graphgps.transform.task_preprocessing import task_specific_preprocessing
from graphgps.loader.master_loader import compute_PE_stats_
from graphgps.loader.split_generator import prepare_splits
from torch_geometric.graphgym.loader import set_dataset_info
from torch_geometric.loader import DataLoader
import os.path as osp
import torch
import torch.nn.functional as F
from graphgps.attack.postprocessing import log_node_classification_output_stats
import logging
from itertools import compress


def get_RUT_loader(transfer_model: str, budget: int):
    dataset_dir = osp.join(cfg.dataset.dir, "RobustnessUnitTest")
    dataset = RUTAttack(
        root=dataset_dir,
        name=cfg.dataset.name,
        split=cfg.dataset.split_index,
        scenario="evasion",
        model=transfer_model,
        budget=budget,
    )
    pre_transform_in_memory(dataset, partial(task_specific_preprocessing, cfg=cfg))
    # Precompute necessary statistics for positional encodings.
    compute_PE_stats_(dataset)
    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)
    set_dataset_info(dataset)
    pw = cfg.num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=pw
    )
    return loader


@torch.no_grad
def transfer_unit_test(model, clean_test_loader):
    results = {}
    assert len(clean_test_loader) == 1
    for batch in clean_test_loader:
        E_clean = batch.edge_index.size(1)
        batch.split = "test"
        output_clean, y = model(batch)
    # performance of clean
    output_stats_clean = get_output_stats(y, output_clean)
    results["clean"] = output_stats_clean
    # transfer attacks
    budgets = clean_test_loader.dataset.budgets[cfg.dataset.split_index]
    for transfer_model in RUTAttack.models:
        model_budgets = budgets[transfer_model]["evasion"]
        r = {"budgets": model_budgets, "attack_success_rate": []}
        for k in output_keys:
            r[k] = []
        for b in model_budgets:
            loader = get_RUT_loader(transfer_model, b)
            assert len(loader) == 1
            for batch in loader:
                batch.split = "test"
                output_pert, y = model(batch)
            # check performance of pert
            output_stats_pert = get_output_stats(y, output_pert)
            for k, v in output_stats_pert.items():
                r[k].append(v)
            logging.info(f"Transfer model: {transfer_model}, budget: {b} ({100 * (2 * b) / E_clean:.2f}%)")
            log_node_classification_output_stats(output_stats_pert, output_stats_clean)
            pert_c = list(compress(output_stats_pert["correct"], output_stats_clean["correct"]))
            asr = 1 - (sum(pert_c) / len(pert_c))
            r["attack_success_rate"].append(asr)
            logging.info(f"Attack success rate: {asr:.3f}")
        results[transfer_model] = r
    return results


output_keys = [
    "probs",
    "logits",
    "correct",
    "correct_acc",
    "margin",
    "margin_mean",
    "margin_median",
    "margin_min",
    "margin_max",
]


def get_output_stats(y_gt, logits):
    class_idx_pred = logits.argmax(dim=1)
    probs = logits.softmax(dim=1)
    num_classes = probs.size(1)
    y_correct_mask = F.one_hot(y_gt, num_classes).to(dtype=torch.bool)
    assert probs.shape == y_correct_mask.shape
    margin = probs[y_correct_mask] - probs[~y_correct_mask].reshape(-1, num_classes-1).max(dim=1)[0]
    margin_mean = margin.mean().item()
    margin_median = margin.median().item()
    margin_min = margin.min().item()
    margin_max = margin.max().item()
    correct = class_idx_pred == y_gt
    acc = correct.float().mean().item()
    output_stats = {
        "probs": probs.tolist(),
        "logits": logits.tolist(),
        "correct": correct.tolist(),
        "correct_acc": acc,
        "margin": margin.tolist(),
        "margin_mean": margin_mean,
        "margin_median": margin_median,
        "margin_min": margin_min,
        "margin_max": margin_max,
    }
    return output_stats