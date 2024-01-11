import torch
import logging
from graphgps.attack.utils_attack import get_reached_nodes


def output_comparison(y_gt, output_clean, output_pert, sigmoid_threshold):
    logit_clean = output_clean[0, :]
    logit_pert = output_pert[0, :]

    y_multiclass: torch.Tensor
    class_index_pred_clean: int
    class_index_pred_pred: int
    prob_clean: torch.Tensor
    prob_pert: torch.Tensor

    if y_gt.size(0) == 1:
        # binary classification
        y_binary = int(y_gt[0].item())
        y_multiclass = torch.zeros(2, dtype=torch.long)
        y_multiclass[y_binary] = 1
        prob_clean_binary = torch.sigmoid(logit_clean)
        prob_pert_binary = torch.sigmoid(logit_pert)
        class_index_pred_clean = int(prob_clean_binary > sigmoid_threshold)
        class_index_pred_pred = int(prob_pert_binary > sigmoid_threshold)
        prob_clean = torch.cat([1 - prob_clean_binary, prob_clean_binary], dim=0)
        prob_pert = torch.cat([1 - prob_pert_binary, prob_pert_binary], dim=0)
    else:
        y_multiclass = y_gt
        class_index_pred_clean = int(logit_clean.argmax().item())
        class_index_pred_pred = int(logit_pert.argmax().item())
        prob_clean = logit_clean.softmax(dim=0)
        prob_pert = logit_pert.softmax(dim=0)

    class_index_gt = int(y_multiclass.argmax().item())
    y_correct_mask = y_multiclass.to(dtype=torch.bool)
    margin_clean = float((prob_clean[y_correct_mask] - prob_clean[~y_correct_mask].max()).item())
    margin_pert = float((prob_pert[y_correct_mask] - prob_pert[~y_correct_mask].max()).item())
    correct_clean = class_index_pred_clean == class_index_gt
    correct_pert = class_index_pred_pred == class_index_gt

    logging_stats = [
        ("Clean", prob_clean, logit_clean, correct_clean, margin_clean),
        ("Perturbed", prob_pert, logit_pert, correct_pert, margin_pert),
    ]
    for (name, prob, logit, correct, margin) in logging_stats:
        prob_str = ", ".join((f"{p:.3f}" for p in prob.tolist()))
        logit_str = ", ".join([f"{l:.2f}" for l in logit.tolist()])
        logging.info(
            f"{name + ':':<10}\tcorrect (margin) [prob] <logits>:\t"
            f"{str(correct):5} ({f'{margin:.4}':>7}) [{prob_str}] <{logit_str}>"
        )
    return correct_clean, correct_pert


def basic_edge_and_node_stats(
    edge_index: torch.Tensor,
    pert_edge_index: torch.Tensor,
    root: None | int,
):
    if root is not None:
        reached = get_reached_nodes(root, pert_edge_index)
    else:
        # TODO: add option when root is None -> select the largest conected subgraph as new graph
        raise NotImplementedError
    edges = set()
    nodes = set()
    for edge in torch.split(edge_index, 1, dim=1):
        edge = tuple(int(n) for n in edge.squeeze().tolist())
        edges.add(edge)
        nodes.update(edge)

    edges_pert = set()
    nodes_pert = set()
    nodes_pert_connected = set()
    for edge in torch.split(pert_edge_index, 1, dim=1):
        edge = tuple(int(n) for n in edge.squeeze().tolist())
        edges_pert.add(edge)
        nodes_pert.update(edge)
        nodes_pert_connected.update([n for n in edge if n in reached])
    edges_pert_connected = set(e for e in edges_pert if all(n in reached for n in e))

    edges_added = edges_pert - edges
    edges_added_connected = edges_pert_connected - edges
    edges_removed = edges - edges_pert

    nodes_added = nodes_pert - nodes
    nodes_added_connected = nodes_pert_connected - nodes
    nodes_removed = nodes - nodes_pert

    stats = {
        "edges": {
            "clean": edges,
            "pert": edges_pert,
            "pert_connected": edges_pert_connected,
            "added": edges_added,
            "added_connected": edges_added_connected,
            "removed": edges_removed
        },
        "nodes": {
            "clean": nodes,
            "pert": nodes_pert,
            "pert_connected": nodes_pert_connected,
            "added": nodes_added,
            "added_connected": nodes_added_connected,
            "removed": nodes_removed
        },
    }
    num_stats = {}
    for key1 in stats:
        num_key = "num_" + key1
        num_stats[num_key] = {}
        for key2, value in stats[key1].items():
            num_stats[num_key][key2] = len(value)
    return stats | num_stats


def log_and_accumulate_stats(accumulated_stats, stats):
    stat_keys = [
        ("num_edges", "clean"), ("num_edges", "added"), ("num_edges", "added_connected"), ("num_edges", "removed"),
        ("num_nodes", "clean"), ("num_nodes", "added"), ("num_nodes", "added_connected"), ("num_nodes", "removed"),

    ]
    acc_keys = [
        "num_edges", "num_edges_added", "num_edges_added_connected", "num_edges_removed",
        "num_nodes", "num_nodes_added", "num_nodes_added_connected", "num_nodes_removed"
    ]
    info_texts = [
        "Original number of edges", "Added edges", "Added edges (connected)", "Removed edges",
        "Original number of nodes", "Added nodes", "Added nodes (connected)", "Removed nodes"
    ]

    for stat_key, acc_key, info_text in zip(stat_keys, acc_keys, info_texts):
        current_stat = stats[stat_key[0]][stat_key[1]]
        accumulated_stats[acc_key].append(current_stat)
        logging.info(f"{info_text + ':':<26} {current_stat:>7}")


def log_summary_stats(accumulated_stats):
    logging.info("SUMMARY (averages over all attacked graphs):")
    for key, current_stat in accumulated_stats.items():
        name = "avg_" + key
        avg = sum(current_stat) / len(current_stat)
        logging.info(f"{name + ':':<30} {f'{avg:.2f}':>10}")
