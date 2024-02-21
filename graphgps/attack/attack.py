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
    get_output_stats,
    log_and_accumulate_pert_output_stats,
    basic_edge_and_node_stats,
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
    is_undirected = cfg.attack.is_undirected
    model.eval()
    model.forward = forward_wrapper(model.forward, is_undirected)

    prbcd = PRBCDAttack(model, is_undirected)

    accumulated_stats = get_empty_accumulated_stats()

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

        skip, skip_msg, global_budget, output_clean, output_stats_clean = check_budget_and_clean_data(
            model, clean_data,
        )

        if skip:
            logging.info(f"Skipping graph {i + 1}, {skip_msg}")
            # still need to log that it is incorrect, to compute the accuracy
            for key, s in accumulated_stats.items():
                if key.startswith("correct"):
                    s.append(False)
                else:
                    s.append(None)
            continue

        logging.info(f"Attacking graph {i + 1}")

        if skip_msg:
            logging.info(skip_msg)

        accumulate_output_stats(accumulated_stats, output_stats_clean, mode="clean", random=False)

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
                augmented_output = model(Batch.from_data_list([attack_graph_data.clone()]))
            assert torch.allclose(output_clean, augmented_output, atol=0.001, rtol=0.001)

        # attack (with PRBCD then random baseline if specified)
        random_attacks = [False, True] if cfg.attack.run_random_baseline else [False]
        for random_attack in random_attacks:

            pert_edge_index, perts = attack_single_graph(
                attack_graph_data=attack_graph_data,
                is_undirected=is_undirected,
                model=model,
                attack=prbcd,
                global_budget=global_budget,
                random_attack=random_attack,
                _model_forward_already_wrapped=True,
                _keep_forward_wrapped=True,
            )

            num_modified_edges = perts.size(1)
            relative_budget_used = num_modified_edges / global_budget
            key = "budget_used_random" if random_attack else "budget_used"
            accumulated_stats[key].append(relative_budget_used)
            m = "Random perturbation" if random_attack else "Perturbation"
            logging.info(f"{m} uses {100 * relative_budget_used:.1f}% of the given attack budget.")

            # check output of the perturbed graph
            pert_data = attack_graph_data.clone()
            pert_data.edge_index = pert_edge_index
            pert_data.edge_attr = torch.ones(pert_edge_index.size(1), device=cfg.accelerator)
            with torch.no_grad():
                output_pert = model(Batch.from_data_list([pert_data.clone()]))

            log_and_accumulate_pert_output_stats(
                y_gt=clean_data.y,
                output_pert=output_pert,
                output_stats_clean=output_stats_clean,
                accumulated_stats=accumulated_stats,
                random=random_attack,
            )
            stats, num_stats = basic_edge_and_node_stats(
                clean_data.edge_index,
                pert_edge_index,
                num_edges_clean=num_edges,
                num_nodes_clean=num_nodes,
                is_undirected=is_undirected,
            )
            log_and_accumulate_num_stats(
                accumulated_stats,
                num_stats,
                random=random_attack,
            )

        
    summary_stats = log_summary_stats(accumulated_stats)
    model.forward = model.forward.__wrapped__
    logging.info("End of attack.")
    return {"avg": summary_stats, "all": dict(accumulated_stats)}


def check_budget_and_clean_data(model, clean_data):
    skip = False
    skip_msg = ""

    # TODO: allow for other ways to define the budget
    num_edges = clean_data.edge_index.size(1)
    budget_edges = num_edges / 2 if cfg.attack.is_undirected else num_edges
    global_budget = int(cfg.attack.e_budget * budget_edges)

    if cfg.attack.minimum_budget > global_budget:
        global_budget = cfg.attack.minimum_budget
        skip_msg = (
            f"Budget would be smaller than set minimum. Budget set to minimum, this means the relative budget "
            f"was effectively increased from {cfg.attack.e_budget} to {cfg.attack.minimum_budget / budget_edges} "
            f"for this graph."
        )
    if not global_budget:
        skip_msg = "because budget is 0."
        skip = True

    if not skip:
        clean_data.to(device=cfg.accelerator)
        with torch.no_grad():
            output_clean = model(clean_data.clone(), unmodified=True)

        output_stats_clean = get_output_stats(clean_data.y, output_clean)
        if (
            cfg.dataset.task == "graph"
            and cfg.attack.skip_incorrect_graph_classification
            and "correct" in output_stats_clean
            and not output_stats_clean["correct"]
        ):
            skip_msg = "because it is already incorrectly classified by model."
            skip = True

    return skip, skip_msg, global_budget, output_clean, output_stats_clean


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
    is_undirected: bool,
    model,
    attack: PRBCDAttack,
    global_budget: int,
    random_attack: bool = False,
    _model_forward_already_wrapped: bool = False,
    _keep_forward_wrapped: bool = False,
) -> tuple[Data, torch.Tensor, torch.Tensor]:
    """
    """
    if not _model_forward_already_wrapped:
        model.forward = forward_wrapper(model.forward, is_undirected)

    attack_fun = attack.attack_random_baseline if random_attack else attack.attack

    pert_edge_index, perts = attack_fun(
        attack_graph_data.x,
        attack_graph_data.edge_index,
        attack_graph_data.y,
        budget=global_budget,
    )

    if not _keep_forward_wrapped:
        model.forward = model.forward.__wrapped__

    return pert_edge_index, perts