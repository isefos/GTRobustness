import torch
import logging
from graphgps.attack.utils_attack import get_reached_nodes


def get_prediction_stats(y_gt, model_output, sigmoid_threshold) -> tuple[torch.Tensor, torch.Tensor, bool, float]:
    logits = model_output[0, :]

    y_multiclass: torch.Tensor
    class_index_pred: int
    probs: torch.Tensor

    if y_gt.size(0) == 1:
        # binary classification
        y_binary = int(y_gt[0].item())
        y_multiclass = torch.zeros(2, dtype=torch.long)
        y_multiclass[y_binary] = 1
        prob_binary = torch.sigmoid(logits)
        class_index_pred = int(prob_binary > sigmoid_threshold)
        probs = torch.cat([1 - prob_binary, prob_binary], dim=0)
    else:
        y_multiclass = y_gt
        class_index_pred = int(logits.argmax().item())
        probs = logits.softmax(dim=0)

    class_index_gt = int(y_multiclass.argmax().item())
    y_correct_mask = y_multiclass.to(dtype=torch.bool)
    margin = float((probs[y_correct_mask] - probs[~y_correct_mask].max()).item())
    correct = class_index_pred == class_index_gt
    return probs, logits, correct, margin



def log_and_accumulate_output(
    y_gt,
    probs_clean,
    logits_clean,
    correct_clean,
    margin_clean,
    output_pert,
    sigmoid_threshold,
    accumulated_stats,
    random=False,
):
    probs_pert, logits_pert, correct_pert, margin_pert = get_prediction_stats(
        y_gt, output_pert, sigmoid_threshold,
    )
    logging_stats = [
        ("clean", probs_clean, logits_clean, correct_clean, margin_clean),
        ("pert", probs_pert, logits_pert, correct_pert, margin_pert),
    ]
    for (name, prob, logit, correct, margin) in logging_stats:
        prob_str = ", ".join((f"{p:.3f}" for p in prob.tolist()))
        logit_str = ", ".join([f"{l:.3f}" for l in logit.tolist()])
        logging.info(
            f"{name + ':':<10}\tcorrect (margin) [prob] <logits>:\t"
            f"{str(correct):5} ({f'{margin:.4}':>7}) [{prob_str}] <{logit_str}>"
        )
    
    if random:
        accumulated_stats["correct_pert_random"].append(correct_pert)
        accumulated_stats["margin_pert_random"].append(margin_pert)
    else:
        accumulated_stats["correct_clean"].append(correct_clean)
        accumulated_stats["correct_pert"].append(correct_pert)
        accumulated_stats["margin_clean"].append(margin_clean)
        accumulated_stats["margin_pert"].append(margin_pert)


def basic_edge_and_node_stats(
    edge_index: torch.Tensor,
    pert_edge_index: torch.Tensor,
    root: None | int,
    num_edges: int,
    num_nodes: int,
    is_undirected: bool,
) -> tuple[dict[str, dict[str, set[tuple[int, int]]]], dict[str, dict[str, int]]]:
    if root is not None:
        reached = get_reached_nodes(root, pert_edge_index)
    else:
        # TODO: add option when root is None -> select the largest connected subgraph as main graph
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
            v = len(value)
            if is_undirected and key1 == "edges":
                v = v // 2
            num_stats[num_key][key2] = v

    assert num_edges == num_stats["num_edges"]["clean"]
    assert num_nodes == num_stats["num_nodes"]["clean"]

    return stats, num_stats


def log_and_accumulate_num_stats(accumulated_stats, num_stats, random=False):
    stat_keys = [
        ("num_edges", "clean"), ("num_edges", "added"), ("num_edges", "added_connected"), ("num_edges", "removed"),
        ("num_nodes", "clean"), ("num_nodes", "added"), ("num_nodes", "added_connected"), ("num_nodes", "removed"),

    ]
    info_texts = [
        "Original number of edges", "Added edges", "Added edges (connected)", "Removed edges",
        "Original number of nodes", "Added nodes", "Added nodes (connected)", "Removed nodes"
    ]

    for stat_key, info_text in zip(stat_keys, info_texts):
        current_stat = num_stats[stat_key[0]][stat_key[1]]
        logging.info(f"{info_text + ':':<26} {current_stat:>7}")
        acc_key = stat_key[0] + "_" + stat_key[1]
        if random:
            if stat_key[1] == "clean":
                continue
            acc_key += "_random"
        accumulated_stats[acc_key].append(current_stat)


def log_summary_stats(accumulated_stats):
    summary_stats = {}
    logging.info("Attack stats summary (averages over all attacked graphs):")
    for key, current_stat in accumulated_stats.items():
        name = "avg_" + key
        filtered_stat = [s for s in current_stat if s is not None]
        if filtered_stat:
            avg = sum(current_stat) / len(current_stat)
            logging.info(f"\t{name + ':':<30} {f'{avg:.2f}':>10}")
        else:
            avg = None
            logging.info(f"\t{name + ':':<30} {'nan':>10}")
        summary_stats[name] = avg
    return summary_stats
