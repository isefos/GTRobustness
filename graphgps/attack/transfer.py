import torch
import logging
import json
from pathlib import Path
from torch_geometric.graphgym.config import cfg
from torch_geometric.data import Data, Batch
from graphgps.attack.preprocessing import forward_wrapper, remove_isolated_components
from graphgps.attack.dataset_attack import (
    get_total_dataset_graphs,
    get_attack_datasets,
)
from graphgps.attack.postprocessing import (
    get_accumulated_stat_keys,
    accumulate_output_stats,
    accumulate_output_stats_pert,
    get_output_stats,
    log_pert_output_stats,
    basic_edge_and_node_stats,
    log_and_accumulate_num_stats,
    log_summary_stats,
)
from graphgps.attack.attack import (
    apply_node_mask,
    get_attack_loader,
    get_attack_graph,
)


def transfer_attack_dataset(model, loaders, perturbation_path):
    """
    """

    # TODO: save the attack config with the perturbations, then check if compatible
    # (only some are important, e.g. the attack dataset options)

    # TODO: extract perturbations from perturbation path
    perturbations = None
     
    if cfg.dataset.task == "node" and (cfg.attack.node_injection.enable or cfg.attack.remove_isolated_components):
        raise NotImplementedError(
            "Need to handle the node mask (also to calculate attack success rate) "
            "with node injection or pruning away isolated components."
        )
    logging.info("Start of transfer attack:")
    model.eval()
    model.forward = forward_wrapper(model.forward)
    stat_keys = get_accumulated_stat_keys()
    all_stats = {k: [] for k in sorted(stat_keys)}

    # PREPARE DATASETS
    dataset_to_attack, additional_injection_datasets, inject_nodes_from_attack_dataset = get_attack_datasets(loaders)
    total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph = None, None, None
    if cfg.attack.node_injection.enable:
        total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph = get_total_dataset_graphs(
            inject_nodes_from_attack_dataset=inject_nodes_from_attack_dataset,
            dataset_to_attack=dataset_to_attack,
            additional_injection_datasets=additional_injection_datasets,
            include_root_nodes=cfg.attack.node_injection.include_root_nodes,
        )
    clean_loader = get_attack_loader(dataset_to_attack)
    for i, clean_data in enumerate(clean_loader):
        clean_data: Batch
        assert clean_data.num_graphs == 1
        perturbation = perturbations.get(i, None)
        if perturbation is None:
            logging.info(f"Skipping graph {i} attack because no perturbation for it is available.")
            continue
        attack_or_skip_graph(
            i,
            model,
            clean_data.get_example(0),
            all_stats,
            perturbation,
            total_attack_dataset_graph,
            attack_dataset_slices,
            total_additional_datasets_graph,
        )
    model.forward = model.forward.__wrapped__
    logging.info("End of transfer attack.")

    # summarize results
    summary_stats = log_summary_stats(all_stats)
    results = {"avg": summary_stats}
    if not cfg.attack.only_return_avg:
        results["all"] = all_stats
    return results


def attack_or_skip_graph(
    i: int,
    model,
    clean_data: Data,
    all_stats: dict[str, list],
    perturbation: list[list[int]],
    total_attack_dataset_graph: Data | None,
    attack_dataset_slices: list[tuple[int, int]] | None,
    total_additional_datasets_graph: Data | None,
):
    """
    """
    logging.info(f"Attacking graph {i}")
    num_nodes = clean_data.x.size(0)
    num_edges = clean_data.edge_index.size(1)
    node_mask = clean_data.get(f'{cfg.attack.split}_mask')
    if node_mask is not None:
        node_mask = node_mask.to(device=cfg.accelerator)
        assert not cfg.attack.prediction_level == "graph"

    # CHECK CLEAN GRAPH
    output_clean = model(clean_data.clone().to(device=cfg.accelerator), unmodified=True)
    output_clean = apply_node_mask(output_clean, node_mask)
    y_gt = apply_node_mask(clean_data.y.to(device=cfg.accelerator), node_mask)
    output_stats_clean = get_output_stats(y_gt, output_clean)
    accumulate_output_stats(all_stats, output_stats_clean, mode="clean", random=False)

    # AUGMENT GRAPH (ONLY WHEN NODE INJECTION)
    attack_graph_data = get_attack_graph(
        graph_data=clean_data,
        total_attack_dataset_graph=total_attack_dataset_graph,
        attack_dataset_slice=None if attack_dataset_slices is None else attack_dataset_slices[i],
        total_additional_datasets_graph=total_additional_datasets_graph,
    )

    # APPLY TRANSFER ATTACK
    # TODO: get from perturbation
    pert_edge_index = None
    
    # CHECK OUTPUT
    data = Data(x=attack_graph_data.x.clone(), edge_index=pert_edge_index.cpu().clone())
    data, _ = remove_isolated_components(data)
    output_pert = model(data.to(device=cfg.accelerator), unmodified=True)
    output_pert = apply_node_mask(output_pert, node_mask)
    output_stats_pert = get_output_stats(y_gt, output_pert)
    log_pert_output_stats(output_stats_pert, output_stats_clean=output_stats_clean, random=False)
    stats, num_stats = basic_edge_and_node_stats(clean_data.edge_index, pert_edge_index, num_edges, num_nodes)
    accumulate_output_stats_pert(all_stats, output_stats_pert, output_stats_clean, False)
    log_and_accumulate_num_stats(all_stats, num_stats, random=False)
