import torch
import logging
from torch_geometric.utils import to_scipy_sparse_matrix, index_to_mask, subgraph
from scipy.sparse.csgraph import breadth_first_order
from torch_geometric.graphgym.config import cfg


def get_empty_accumulated_stats():
    keys = [
        "budget_used",
        "num_edges_added",
        "num_edges_added_connected",
        "num_edges_removed",
        "num_nodes_added",
        "num_nodes_added_connected",
        "num_nodes_removed",
    ]

    if cfg.attack.run_random_baseline:
        keys.extend([k + "_random" for k in keys])

    keys.extend(["num_edges_clean", "num_nodes_clean"])

    if cfg.dataset.task == "node":
        # TODO: implement for node classification
        raise NotImplementedError
    
    elif cfg.dataset.task == "graph":
        if cfg.dataset.task_type.startswith("classification"):
            graph_cls_keys = ["probs", "logits", "correct", "margin"]
            keys.extend([k + "_clean" for k in graph_cls_keys])
            keys.extend([k + "_pert" for k in graph_cls_keys])
            if cfg.attack.run_random_baseline:
                keys.extend([k + "_pert_random" for k in graph_cls_keys])

        else:
            raise NotImplementedError
    
    else:
        raise NotImplementedError
    
    return {k: [] for k in sorted(keys)}


def get_output_stats(y_gt, model_output):
    if cfg.dataset.task == "node":
        # TODO: implement for node classification
        raise NotImplementedError
    
    elif cfg.dataset.task == "graph":
        return get_graph_output_stats(y_gt, model_output)
    
    else:
        raise NotImplementedError


def get_graph_output_stats(y_gt, model_output) -> tuple[torch.Tensor, torch.Tensor, bool, float]:

    if cfg.dataset.task_type.startswith("classification"):
        logits = model_output[0, :]
        y_multiclass: torch.Tensor
        class_index_pred: int
        probs: torch.Tensor

        if cfg.dataset.task_type == "classification_binary":
            y_binary = int(y_gt[0].item())
            y_multiclass = torch.zeros(2, dtype=torch.long)
            y_multiclass[y_binary] = 1
            prob_binary = torch.sigmoid(logits)
            class_index_pred = int(prob_binary > cfg.model.thresh)
            probs = torch.cat([1 - prob_binary, prob_binary], dim=0)
        
        elif cfg.dataset.task_type == "classification":
            y_multiclass = y_gt
            class_index_pred = int(logits.argmax().item())
            probs = logits.softmax(dim=0)

        class_index_gt = int(y_multiclass.argmax().item())
        y_correct_mask = y_multiclass.to(dtype=torch.bool)
        margin = float((probs[y_correct_mask] - probs[~y_correct_mask].max()).item())
        correct = class_index_pred == class_index_gt
        output_stats = {
            "probs": probs.tolist(),
            "logits": logits.tolist(),
            "correct": correct,
            "margin": margin,
        }
        return output_stats

    else:
        raise NotImplementedError


def log_and_accumulate_pert_output_stats(
    y_gt,
    output_pert,
    output_stats_clean,
    accumulated_stats,
    random=False,
):
    output_stats_pert = get_output_stats(y_gt, output_pert)  
    accumulate_output_stats(accumulated_stats, output_stats_pert, mode="pert", random=random)

    if cfg.dataset.task == "graph":
        if cfg.dataset.task_type.startswith("classification"):
            log_graph_classification_output_stats(output_stats_pert, output_stats_clean, random)
        else:
            raise NotImplementedError

    elif cfg.dataset.task == "node":
        # TODO: implement for node classification
        raise NotImplementedError
    
    else:
        raise NotImplementedError
    

def accumulate_output_stats(
    accumulated_stats,
    output_stats,
    mode: str,
    random: bool,
):
    for key, stat in output_stats.items():
        k = key + "_" + mode
        if random:
            k = k + "_random"
        accumulated_stats[k].append(stat)


def log_graph_classification_output_stats(
    output_stats_pert,
    output_stats_clean,
    random=False,
):
    for name, output_stats in zip(
        ["clean", "pert rand" if random else "pert"],
        [output_stats_clean, output_stats_pert]
    ):
        correct_str = f"{str(output_stats['correct']):5}"
        margin = output_stats['margin']
        margin_str = f"{f'{margin:.4}':>7}"
        prob_str = ", ".join((f"{p:.3f}" for p in output_stats['probs']))
        logit_str = ", ".join([f"{l:.3f}" for l in output_stats['logits']])
        logging.info(
            f"{name + ':':<10}   correct  (margin) [probs] <logits>:   "
            f"{correct_str} ({margin_str}) [{prob_str}] <{logit_str}>"
        )


def _get_edges_nodes_from_index(edge_index):
    edges = set()
    nodes = set()
    for edge in torch.split(edge_index, 1, dim=1):
        edge = tuple(int(n) for n in edge.squeeze().tolist())
        edges.add(edge)
        nodes.update(edge)
    return edges, nodes


def basic_edge_and_node_stats(
    edge_index: torch.Tensor,
    edge_index_pert: torch.Tensor,
    num_edges_clean: int,
    num_nodes_clean: int,
    is_undirected: bool,
) -> tuple[dict[str, dict[str, set[tuple[int, int]]]], dict[str, dict[str, int]]]:
    
    num_nodes_pert = edge_index_pert.max().item() + 1
    root_node = cfg.attack.root_node_idx
    adj = to_scipy_sparse_matrix(edge_index_pert, num_nodes=num_nodes_pert)
    if root_node is not None:
        bfs_order = breadth_first_order(adj, root_node, return_predecessors=False)
        subset_mask = index_to_mask(
            torch.tensor(bfs_order, dtype=torch.long, device=edge_index.device),
            size=num_nodes_pert,
        )
        edge_index_pert_connected = subgraph(subset_mask, edge_index_pert)[0]
    else:
        # TODO: add option when root is None -> select the largest connected subgraph as new graph
        raise NotImplementedError
    
    edges, nodes = _get_edges_nodes_from_index(edge_index)
    edges_pert, nodes_pert = _get_edges_nodes_from_index(edge_index_pert)
    edges_pert_connected, nodes_pert_connected = _get_edges_nodes_from_index(edge_index_pert_connected)

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

    num_edges_clean = num_edges_clean // 2 if is_undirected else num_edges_clean

    assert num_edges_clean == num_stats["num_edges"]["clean"]
    assert num_nodes_clean == num_stats["num_nodes"]["clean"]

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
        if key.startswith("probs") or key.startswith("logits"):
            continue
        name = "avg_" + key
        filtered_stat = [s for s in current_stat if s is not None]
        if filtered_stat:
            avg = sum(filtered_stat) / len(filtered_stat)
            logging.info(f"  {name + ':':<37} {f'{avg:.2f}':>8}")
        else:
            avg = None
            logging.info(f"  {name + ':':<37} {'nan':>8}")
        summary_stats[name] = avg
    return summary_stats
