import torch
from torch_geometric.loader import DataLoader
from graphgps.attack.prbcd import PRBCDAttack
from torch_geometric.data import Data, Batch
from graphgps.attack.dataset_attack import (
    get_total_dataset_graphs,
    get_augmented_graph,
    get_attack_datasets,
)
from graphgps.attack.preprocessing import forward_wrapper
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
import logging
from torch_geometric.graphgym.config import cfg


# TODO: dataset analysis:
#  - Plot the graphs!
#  - How many levels from root?
#  - Always trees?
#  - How often are node ids repeated?
#     - Within same graph?
#     - In other graphs?


def prbcd_attack_dataset(model, loaders):
    logging.info("Start of attack:")
    model.eval()
    model.forward = forward_wrapper(model.forward)
    prbcd = PRBCDAttack(model)
    accumulated_stats, accumulated_stats_zb = get_empty_accumulated_stats()

    if cfg.dataset.task == "node" and (cfg.attack.enable_node_injection or cfg.attack.remove_isolated_components):
        raise NotImplementedError(
            "Need to handle the node mask (also to calculate attack success rate) "
            "with node injection or pruning away isolated components."
        )

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

    clean_loader = DataLoader(dataset_to_attack, batch_size=1, shuffle=False)
    for i, clean_data in enumerate(clean_loader):
        if cfg.attack.num_attacked_graphs and i >= cfg.attack.num_attacked_graphs:
            break

        num_nodes = clean_data.x.size(0)
        num_edges = clean_data.edge_index.size(1)
        # currently only does global attack (entire split),
        # but could technically select any subset of (single / local)
        # nodes to attack by passing a node mask as an argument
        node_mask = clean_data.get(f'{cfg.attack.split}_mask')
        if node_mask is not None:
            node_mask = node_mask.to(device=cfg.accelerator)
            assert not cfg.attack.prediction_level == "graph"

        # check the prediction of the clean graph
        with torch.no_grad():
            output_clean = model(clean_data.clone().to(device=cfg.accelerator), unmodified=True)
        output_clean = apply_node_mask(output_clean, node_mask)
        y_gt = apply_node_mask(clean_data.y.to(device=cfg.accelerator), node_mask)
        output_stats_clean = get_output_stats(y_gt, output_clean)

        if (
            cfg.attack.prediction_level == "graph"
            and cfg.attack.skip_incorrect_graph_classification
            and "correct" in output_stats_clean
            and not output_stats_clean["correct"]
        ):
            # graph prediction is incorrect, no need to attack
            logging.info("Skipping graph attack because it is already incorrectly classified by model.")
            # In this case:
            # - set correct.* to False (for the accuracy calculations)
            # - all the rest to None
            for k in accumulated_stats:
                if k.startswith("correct"):
                    accumulated_stats[k].append(False)
                else:
                    accumulated_stats[k].append(None)
            for k in accumulated_stats_zb:
                if k.startswith("correct"):
                    accumulated_stats_zb[k].append(False)
                else:
                    accumulated_stats_zb[k].append(None)
            continue

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

        if global_budget == 0:
            # there is no budget for the attack, can't attack
            logging.info(
                f"Skipping graph attack because maximum budget is less than 1 "
                f"({cfg.attack.e_budget} of {budget_edges}), so cannot make perturbations."
            )
            # In this case we only accumulate the stats for the clean graph in the zero budget dict
            for k in ["budget_used", "budget_used_random"]:
                accumulated_stats_zb[k].append(0)
            accumulate_output_stats(
                    accumulated_stats_zb,
                    output_stats_clean,
                    mode="clean",
                    random=False,
                )
            for random in [False, True]:
                accumulate_output_stats_pert(
                    accumulated_stats_zb,
                    output_stats_clean,
                    output_stats_clean,
                    random,
                    True,
                )
            _, num_stats = zero_budget_edge_and_node_stats(
                clean_data.edge_index,
                num_edges=num_edges,
                num_nodes=num_nodes
            )
            for random in [False, True]:
                log_and_accumulate_num_stats(accumulated_stats_zb, num_stats, random, zero_budget=True)
            continue

        logging.info(f"Attacking graph {i + 1}")

        for d in [accumulated_stats, accumulated_stats_zb]:
            accumulate_output_stats(
                d,
                output_stats_clean,
                mode="clean",
                random=False,
            )

        # get the graph to attack (potentially augmented)
        attack_graph_data = get_attack_graph(
            graph_data=clean_data.get_example(0),
            total_attack_dataset_graph=total_attack_dataset_graph,
            attack_dataset_slice=None if attack_dataset_slices is None else attack_dataset_slices[i],
            total_additional_datasets_graph=total_additional_datasets_graph,
        )

        # TODO: make debugging argument
        _check_augmentation_correctness = False
        if _check_augmentation_correctness:
            attack_graph_data.edge_attr = torch.ones(attack_graph_data.edge_index.size(1))
            with torch.no_grad():
                data = Batch.from_data_list([attack_graph_data.clone()])
                augmented_output = model(data)
            augmented_output = apply_node_mask(augmented_output, node_mask)
            assert torch.allclose(output_clean, augmented_output, atol=0.001, rtol=0.001)

        # attack (with PRBCD then random baseline if specified)
        random_attacks = [False, True] if cfg.attack.run_random_baseline else [False]
        for random_attack in random_attacks:

            pert_edge_index, perts = attack_single_graph(
                attack_graph_data=attack_graph_data,
                model=model,
                attack=prbcd,
                global_budget=global_budget,
                random_attack=random_attack,
                node_mask=node_mask,
                _model_forward_already_wrapped=True,
                _keep_forward_wrapped=True,
            )

            if not random_attack:
                accumulated_stats["perturbation"].append(perts.tolist())

            num_modified_edges = perts.size(1)
            relative_budget_used = num_modified_edges / global_budget
            key = "budget_used_random" if random_attack else "budget_used"
            accumulated_stats[key].append(relative_budget_used)
            accumulated_stats_zb[key].append(relative_budget_used)
            m = "Random perturbation" if random_attack else "Perturbation"
            logging.info(
                f"{m} uses {100 * relative_budget_used:.1f}% "
                f"[{num_modified_edges}/{global_budget}] of the given attack budget."
            )

            pert_data = attack_graph_data.clone()
            pert_data.edge_index = pert_edge_index
            pert_data.edge_attr = torch.ones(pert_edge_index.size(1))
            with torch.no_grad():
                data = Batch.from_data_list([pert_data.clone()])
                output_pert = model(data.to(device=cfg.accelerator))
            output_pert = apply_node_mask(output_pert, node_mask)

            output_stats_pert = get_output_stats(y_gt, output_pert)
            log_pert_output_stats(
                output_stats_pert=output_stats_pert,
                output_stats_clean=output_stats_clean,
                random=random_attack,
            )

            stats, num_stats = basic_edge_and_node_stats(
                clean_data.edge_index,
                pert_edge_index,
                num_edges_clean=num_edges,
                num_nodes_clean=num_nodes,
            )

            for (d, zb) in [(accumulated_stats, False), (accumulated_stats_zb, True)]:
                accumulate_output_stats_pert(
                    d,
                    output_stats_pert,
                    output_stats_clean,
                    random_attack,
                    zb,
                )
                log_and_accumulate_num_stats(
                    d,
                    num_stats,
                    random=random_attack,
                    zero_budget=zb,
                )
        
    summary_stats = log_summary_stats(accumulated_stats)
    summary_stats_zb = log_summary_stats(accumulated_stats_zb, zb=True)
    model.forward = model.forward.__wrapped__
    logging.info("End of attack.")
    results = {
        "avg": summary_stats,
        "avg_including_zero_budget": summary_stats_zb,
        "all": accumulated_stats,
        "all_including_zero_budget": accumulated_stats_zb,
    }
    return results


def apply_node_mask(tensor_to_mask, mask):
    if mask is not None:
        return tensor_to_mask[mask]
    return tensor_to_mask


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