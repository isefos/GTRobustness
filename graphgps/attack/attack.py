import torch
import logging
from torch_geometric.graphgym.config import cfg
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from graphgps.attack.prbcd import PRBCDAttack
from graphgps.attack.preprocessing import forward_wrapper, remove_isolated_components
from graphgps.attack.dataset_attack import (
    get_total_dataset_graphs,
    get_augmented_graph,
    get_attack_datasets,
)
from graphgps.attack.postprocessing import (
    get_empty_accumulated_stats,
    accumulate_output_stats,
    accumulate_output_stats_pert,
    get_output_stats,
    log_pert_output_stats,
    basic_edge_and_node_stats,
    zero_budget_edge_and_node_stats,
    log_and_accumulate_num_stats,
    log_summary_stats,
)


def prbcd_attack_dataset(model, loaders):
    """
    """
    if cfg.dataset.task == "node" and (cfg.attack.enable_node_injection or cfg.attack.remove_isolated_components):
        raise NotImplementedError(
            "Need to handle the node mask (also to calculate attack success rate) "
            "with node injection or pruning away isolated components."
        )
    logging.info("Start of attack:")
    model.eval()
    model.forward = forward_wrapper(model.forward)
    prbcd = PRBCDAttack(model)
    attack_epoch_stats = []
    all_stats, all_stats_zb = get_empty_accumulated_stats()
    # PREPARE DATASETS
    dataset_to_attack, additional_injection_datasets, inject_nodes_from_attack_dataset = get_attack_datasets(loaders)
    total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph = None, None, None
    if cfg.attack.enable_node_injection:
        # TODO: attach a global index to all possible nodes, that can later be used to trace which nodes where added
        # how many times
        total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph = get_total_dataset_graphs(
            inject_nodes_from_attack_dataset=inject_nodes_from_attack_dataset,
            dataset_to_attack=dataset_to_attack,
            additional_injection_datasets=additional_injection_datasets,
            include_root_nodes=cfg.attack.include_root_nodes_for_injection,
        )
    clean_loader = get_attack_loader(dataset_to_attack)
    for i, clean_data in enumerate(clean_loader):
        if cfg.attack.num_attacked_graphs and i >= cfg.attack.num_attacked_graphs:
            break
        attack_or_skip_graph(
            i,
            model,
            prbcd,
            clean_data,
            all_stats,
            all_stats_zb,
            attack_epoch_stats,
            total_attack_dataset_graph,
            attack_dataset_slices,
            total_additional_datasets_graph,
        )
    summary_stats = log_summary_stats(all_stats)
    summary_stats_zb = log_summary_stats(all_stats_zb, zb=True)
    model.forward = model.forward.__wrapped__
    logging.info("End of attack.")
    results = {
        "avg": summary_stats,
        "avg_including_zero_budget": summary_stats_zb,
        "all": all_stats,
        "all_including_zero_budget": all_stats_zb,
        "attack_stats": attack_epoch_stats,
    }
    return results


def attack_or_skip_graph(
    i,
    model,
    prbcd,
    clean_data,
    all_stats,
    all_stats_zb,
    attack_epoch_stats,
    total_attack_dataset_graph,
    attack_dataset_slices,
    total_additional_datasets_graph,
):
    """
    """
    num_nodes = clean_data.x.size(0)
    num_edges = clean_data.edge_index.size(1)
    # currently only global attack (entire split), but could attack specific nodes by using this node mask
    node_mask = clean_data.get(f'{cfg.attack.split}_mask')
    if node_mask is not None:
        node_mask = node_mask.to(device=cfg.accelerator)
        assert not cfg.attack.prediction_level == "graph"

    # CHECK CLEAN GRAPH
    output_clean = model(clean_data.clone().to(device=cfg.accelerator), unmodified=True)
    output_clean = apply_node_mask(output_clean, node_mask)
    y_gt = apply_node_mask(clean_data.y.to(device=cfg.accelerator), node_mask)
    output_stats_clean = get_output_stats(y_gt, output_clean)

    # SKIP SCENARIO 1 - INCORRECT GRAPH CLASSIFICATION
    if (
        cfg.attack.prediction_level == "graph"
        and cfg.attack.skip_incorrect_graph_classification
        and not output_stats_clean.get("correct", True)
    ):
        log_incorrect_graph_skip(all_stats, all_stats_zb, attack_epoch_stats)
        return

    # BUDGET DEFINITION
    budget_edges, global_budget = get_budget(num_edges)

    # SKIP SCENARIO 2 - NO BUDGET (SMALL GRAPH)
    if global_budget == 0:
        log_budget_skip(
            clean_data, num_edges, num_nodes, budget_edges, output_stats_clean, all_stats_zb, attack_epoch_stats
        )
        return

    # GRAPH WAS NOT SKIPPED - ATTACK
    logging.info(f"Attacking graph {i + 1}")
    for d in [all_stats, all_stats_zb]:
        accumulate_output_stats(d, output_stats_clean, mode="clean", random=False)

    # AUGMENT GRAPH (ONLY WHEN NODE INJECTION)
    attack_graph_data = get_attack_graph(
        graph_data=clean_data.get_example(0),
        total_attack_dataset_graph=total_attack_dataset_graph,
        attack_dataset_slice=None if attack_dataset_slices is None else attack_dataset_slices[i],
        total_additional_datasets_graph=total_additional_datasets_graph,
    )

    for is_random_attack in [False, True] if cfg.attack.run_random_baseline else [False]:
        # ATTACK
        pert_edge_index, perts = attack_single_graph(
            attack_graph_data=attack_graph_data,
            model=model,
            attack=prbcd,
            global_budget=global_budget,
            random_attack=is_random_attack,
            node_mask=node_mask,
            _model_forward_already_wrapped=True,
            _keep_forward_wrapped=True,
        )
        if not is_random_attack:
            attack_epoch_stats.append(dict(prbcd.attack_statistics))
            all_stats["perturbation"].append(perts.tolist())
        log_used_budget(all_stats, all_stats_zb, global_budget, perts, is_random_attack)
        
        # CHECK OUTPUT
        data = Batch.from_data_list([Data(x=attack_graph_data.x.clone(), edge_index=pert_edge_index.clone())])
        data, _ = remove_isolated_components(data)
        output_pert = model(data.to(device=cfg.accelerator), unmodified=True)
        output_pert = apply_node_mask(output_pert, node_mask)
        output_stats_pert = get_output_stats(y_gt, output_pert)
        log_pert_output_stats(output_stats_pert, output_stats_clean=output_stats_clean, random=is_random_attack)
        stats, num_stats = basic_edge_and_node_stats(clean_data.edge_index, pert_edge_index, num_edges, num_nodes)
        for (d, zb) in [(all_stats, False), (all_stats_zb, True)]:
            accumulate_output_stats_pert(d, output_stats_pert, output_stats_clean, is_random_attack, zb)
            log_and_accumulate_num_stats(d, num_stats, random=is_random_attack, zero_budget=zb)


def apply_node_mask(tensor_to_mask, mask):
    if mask is not None:
        return tensor_to_mask[mask]
    return tensor_to_mask


def get_budget(num_edges):
    # TODO: allow for other ways to define the budget
    budget_edges = num_edges / 2 if cfg.attack.is_undirected else num_edges
    global_budget = int(cfg.attack.e_budget * budget_edges)
    if cfg.attack.minimum_budget > global_budget:
        global_budget = cfg.attack.minimum_budget
        logging.info(
            f"Budget smaller than minimum, thus set to minimum: relative budget "
            f"effectively increased from {cfg.attack.e_budget} to "
            f"{cfg.attack.minimum_budget / budget_edges} for this graph."
        )
    return budget_edges, global_budget


def get_attack_graph(
    graph_data: Data,
    total_attack_dataset_graph: None | Data,
    attack_dataset_slice: None | tuple[int, int],
    total_additional_datasets_graph: None | Data,
) -> tuple[Data, torch.Tensor, torch.Tensor]:
    """
    """
    num_edges = graph_data.edge_index.size(1)
    if cfg.attack.enable_node_injection:
        assert attack_dataset_slice is not None
        graph_data_augmented = get_augmented_graph(
            graph=graph_data.clone(),
            total_attack_dataset_graph=total_attack_dataset_graph,
            attack_dataset_slice=attack_dataset_slice,
            total_additional_datasets_graph=total_additional_datasets_graph,
        )
        assert graph_data_augmented.edge_index.size(1) == num_edges
    else:
        graph_data_augmented = graph_data
    return graph_data_augmented


def attack_single_graph(
    attack_graph_data: Data,
    model,
    attack: PRBCDAttack,
    global_budget: int,
    random_attack: bool = False,
    node_mask: None | torch.Tensor = None,
    _model_forward_already_wrapped: bool = False,
    _keep_forward_wrapped: bool = False,
) -> tuple[Data, torch.Tensor, torch.Tensor]:
    """
    """
    if not _model_forward_already_wrapped:
        model.forward = forward_wrapper(model.forward)

    attack_fun = attack.attack_random_baseline if random_attack else attack.attack

    pert_edge_index, perts = attack_fun(
        attack_graph_data.x.to(device=cfg.accelerator),
        attack_graph_data.edge_index.to(device=cfg.accelerator),
        attack_graph_data.y.to(device=cfg.accelerator),
        budget=global_budget,
        idx_attack=node_mask,
    )

    if not _keep_forward_wrapped:
        model.forward = model.forward.__wrapped__

    return pert_edge_index, perts


def get_attack_loader(dataset_to_attack):
    pw = cfg.num_workers > 0
    loader = DataLoader(
        dataset_to_attack,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=pw
    )
    return loader


def log_incorrect_graph_skip(all_stats, all_stats_zb, attack_epoch_stats):
    logging.info("Skipping graph attack because it is already incorrectly classified by model.")
    attack_epoch_stats.append(None)
    # set correct_ to False (for the accuracy calculations)
    for k in all_stats:
        if k.startswith("correct"):
            all_stats[k].append(False)
        else:
            all_stats[k].append(None)
    for k in all_stats_zb:
        if k.startswith("correct"):
            all_stats_zb[k].append(False)
        else:
            all_stats_zb[k].append(None)


def log_budget_skip(clean_data, E, N, budget_edges, output_stats_clean, all_stats_zb, attack_epoch_stats):
    logging.info(
        f"Skipping graph attack because maximum budget is less than 1 "
        f"({cfg.attack.e_budget} of {budget_edges}), so cannot make perturbations."
    )
    attack_epoch_stats.append(None)
    # In this case we only accumulate the stats for the clean graph in the zero budget dict
    all_stats_zb["budget"].append(0)
    for k in ["budget_used", "budget_used_rel"]:
        all_stats_zb[k].append(0)
        all_stats_zb[k + "_random"].append(0)
    accumulate_output_stats(all_stats_zb, output_stats_clean, mode="clean", random=False)
    for random in [False, True]:
        accumulate_output_stats_pert(all_stats_zb, output_stats_clean, output_stats_clean, random, True)
    _, num_stats = zero_budget_edge_and_node_stats(clean_data.edge_index, E, N)
    for random in [False, True]:
        log_and_accumulate_num_stats(all_stats_zb, num_stats, random, zero_budget=True)


def log_used_budget(all_stats, all_stats_zb, global_budget, perts, is_random_attack):
    all_stats["budget"].append(global_budget)
    all_stats_zb["budget"].append(global_budget)
    E_mod = perts.size(1)
    b_rel = E_mod / global_budget
    for key, value in zip(["budget_used", "budget_used_rel"], [E_mod, b_rel]):
        if is_random_attack:
            key += "_random"
        all_stats[key].append(value)
        all_stats_zb[key].append(value)
    m = "Random perturbation" if is_random_attack else "Perturbation"
    logging.info(f"{m} uses {100 * b_rel:.1f}% [{E_mod}/{global_budget}] of the given attack budget.")


def check_augmentation_correctness(model, loaders):
    """
    The attacked model's ooutput should be the same for the clean graph and the augmented one.
    Augmentation adds disconected new nodes, should be pruned away.
    Test was more important in previous code versions, when the augmentation would also permute the existing nodes.
    Now the extra disconnected nodes are always appended after the existing ones, so as long as the connected component
    pruning is used and works correctly, this 'test' should pass (does not require model node permutation invariance
    anymore).
    """
    dataset_to_attack, additional_injection_datasets, inject_nodes_from_attack_dataset = get_attack_datasets(loaders)
    total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph = None, None, None
    if cfg.attack.enable_node_injection:
        total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph = get_total_dataset_graphs(
            inject_nodes_from_attack_dataset=inject_nodes_from_attack_dataset,
            dataset_to_attack=dataset_to_attack,
            additional_injection_datasets=additional_injection_datasets,
            include_root_nodes=cfg.attack.include_root_nodes_for_injection,
        )
    clean_loader = get_attack_loader(dataset_to_attack)
    for i, clean_data in enumerate(clean_loader):
        node_mask = clean_data.get(f'{cfg.attack.split}_mask')
        if node_mask is not None:
            node_mask = node_mask.to(device=cfg.accelerator)
            assert not cfg.attack.prediction_level == "graph"
        # check the prediction of the clean graph
        output_clean = model(clean_data.clone().to(device=cfg.accelerator), unmodified=True)
        output_clean = apply_node_mask(output_clean, node_mask)
        # get the graph to attack (potentially augmented)
        attack_graph_data = get_attack_graph(
            graph_data=clean_data.get_example(0),
            total_attack_dataset_graph=total_attack_dataset_graph,
            attack_dataset_slice=None if attack_dataset_slices is None else attack_dataset_slices[i],
            total_additional_datasets_graph=total_additional_datasets_graph,
        )
        attack_graph_data.edge_attr = torch.ones(attack_graph_data.edge_index.size(1))
        data = Batch.from_data_list([attack_graph_data.clone()])
        augmented_output = model(data)
        augmented_output = apply_node_mask(augmented_output, node_mask)
        assert torch.allclose(output_clean, augmented_output, atol=0.001, rtol=0.001)
