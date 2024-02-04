import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from graphgps.attack.prbcd import PRBCDAttack
from torch_geometric.data import Data, Batch
from graphgps.attack.dataset_attack import get_total_dataset_graphs, get_augmented_graph
from graphgps.attack.utils_attack import check_if_tree
from graphgps.attack.preprocessing import forward_wrapper
from graphgps.attack.postprocessing import (
    get_prediction_stats,
    log_and_accumulate_output,
    basic_edge_and_node_stats,
    log_and_accumulate_num_stats,
    log_summary_stats,
)
import logging


# TODO: dataset analysis:
#  - Plot the graphs!
#  - How many levels from root?
#  - Always trees?
#  - How often are node ids repeated?
#     - Within same graph?
#     - In other graphs?


def prbcd_attack_dataset(
    model,
    dataset_to_attack: Dataset,
    node_injection_attack: bool,
    additional_injection_datasets: None | list[Dataset],
    inject_nodes_from_attack_dataset: bool,
    device: torch.device,
    attack_loss: str,
    num_attacked_graphs: int,
    e_budget: float,
    block_size: int,  # e.g. 1_000
    lr: float,  # e.g. 1_000
    is_undirected: bool,
    sigmoid_threshold: float,
    existing_node_prob_multiplier: int,
    allow_existing_graph_pert: bool,
    remove_isolated_components: bool,
    root_node_idx: None | int,
    include_root_nodes_for_injection: bool,
):
    logging.info("Start of attack:")
    model.eval()
    model.forward = forward_wrapper(model.forward, is_undirected)

    prbcd = PRBCDAttack(
        model,
        block_size=block_size,
        lr=lr,
        is_undirected=is_undirected,
        loss=attack_loss,
        existing_node_prob_multiplier=existing_node_prob_multiplier,
        allow_existing_graph_pert=allow_existing_graph_pert,
    )

    accumulated_stats = {
        "correct_clean": [],
        "correct_pert": [],
        "correct_pert_random": [],
        "margin_clean": [],
        "margin_pert": [],
        "margin_pert_random": [],
        "num_edges_clean": [],
        "num_edges_added": [],
        "num_edges_added_random": [],
        "num_edges_added_connected": [],
        "num_edges_added_connected_random": [],
        "num_edges_removed": [],
        "num_edges_removed_random": [],
        "num_nodes_clean": [],
        "num_nodes_added": [],
        "num_nodes_added_random": [],
        "num_nodes_added_connected": [],
        "num_nodes_added_connected_random": [],
        "num_nodes_removed": [],
        "num_nodes_removed_random": [],
    }

    total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph = None, None, None
    if node_injection_attack:
        
        # TODO: attach a global index to all possible nodes, that can later be used to trace which nodes where added
        # how many times

        total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph = get_total_dataset_graphs(
            inject_nodes_from_attack_dataset=inject_nodes_from_attack_dataset,
            dataset_to_attack=dataset_to_attack,
            additional_injection_datasets=additional_injection_datasets,
            include_root_nodes=include_root_nodes_for_injection,
            root_node_idx=root_node_idx,
        )

    clean_loader = DataLoader(dataset_to_attack, batch_size=1, shuffle=False)

    for i, clean_data in enumerate(clean_loader):
        if num_attacked_graphs and i >= num_attacked_graphs:
            break

        num_nodes = clean_data.x.size(0)
        num_edges = clean_data.edge_index.size(1)

        skip = False
        skip_msg = ""

        # TODO: allow for other ways to define the budget
            
        budget_edges = num_edges / 2 if is_undirected else num_edges
        global_budget = int(e_budget * budget_edges)

        # TODO: make argument
        set_minimum_budget = False
        if not global_budget:
            if set_minimum_budget:
                global_budget = 1
                logging.info("Budget would be 0, setting it to 1 to enable attack.")
            else:
                skip_msg = "because budget is 0."
                skip = True

        if not skip:
            clean_data.to(device)
            with torch.no_grad():
                output_clean = model(clean_data.clone())
            probs_clean, logits_clean, correct_clean, margin_clean = get_prediction_stats(
                clean_data.y, output_clean, sigmoid_threshold,
            )
            if not correct_clean:
                skip_msg = "because it is already incorrectly classified by model."
                skip = True

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

        # get the graph to attack (potentioally augmented)

        attack_graph_data = get_attack_graph(
            graph_data=clean_data.get_example(0),
            node_injection_attack=node_injection_attack,
            total_attack_dataset_graph=total_attack_dataset_graph,
            attack_dataset_slice=None if attack_dataset_slices is None else attack_dataset_slices[i],
            total_additional_datasets_graph=total_additional_datasets_graph,
        )

        # TODO: make debugging argument
        _check_augmentation_correctness = False
        if _check_augmentation_correctness:
            attack_graph_data.edge_attr = torch.ones(attack_graph_data.edge_index.size(1))
            with torch.no_grad():
                augmented_output = model(
                    Batch.from_data_list([attack_graph_data.clone()]),
                    root_node=root_node_idx,
                    remove_not_connected=remove_isolated_components,
                    recompute_preprocessing=True,
                    unmodified=False,
                )
            assert torch.allclose(output_clean, augmented_output, atol=0.001, rtol=0.001)

        # attack

        pert_edge_index, perts = attack_single_graph(
            attack_graph_data=attack_graph_data,
            is_undirected=is_undirected,
            model=model,
            attack=prbcd,
            global_budget=global_budget,
            root_node_idx=root_node_idx,
            remove_isolated_components=remove_isolated_components,
            random_attack=False,
            _model_forward_already_wrapped=True,
            _keep_forward_wrapped=True,
        )

        num_modified_edges = perts.size(1)
        logging.info(f"Perturbation uses {100 * num_modified_edges / global_budget:.1f}% of the given attack budget.")

        if not check_if_tree(pert_edge_index):
            # TODO: add tree generation (minimum spanning tree) to attack (when specified)
            logging.info("WARNING: PERTURBATION IS NOT A TREE ANYMORE!")

        # check output of the perturbed graph

        pert_data = attack_graph_data.clone()
        pert_data.edge_index = pert_edge_index
        pert_data.edge_attr = torch.ones(pert_edge_index.size(1))
        with torch.no_grad():
            output_pert = model(
                Batch.from_data_list([pert_data.clone()]),
                root_node=root_node_idx,
                remove_not_connected=remove_isolated_components,
                recompute_preprocessing=True,
                unmodified=False,
            )

        log_and_accumulate_output(
            clean_data.y,
            probs_clean,
            logits_clean,
            correct_clean,
            margin_clean,
            output_pert,
            sigmoid_threshold,
            accumulated_stats,
            random=False,
        )
        stats, num_stats = basic_edge_and_node_stats(
            clean_data.edge_index,
            pert_edge_index,
            root=root_node_idx,
            num_edges=num_edges,
            num_nodes=num_nodes,
            is_undirected=is_undirected,
        )
        log_and_accumulate_num_stats(accumulated_stats, num_stats, random=False)

        # random baseline attack

        rand_pert_edge_index, rand_perts = attack_single_graph(
            attack_graph_data=attack_graph_data,
            is_undirected=is_undirected,
            model=model,
            attack=prbcd,
            global_budget=global_budget,
            root_node_idx=root_node_idx,
            remove_isolated_components=remove_isolated_components,
            random_attack=True,
            _model_forward_already_wrapped=True,
            _keep_forward_wrapped=True,
        )

        num_modified_edges = rand_perts.size(1)
        logging.info(
            f"Random perturbation uses {100 * num_modified_edges / global_budget:.1f}% of the given attack budget."
        )

        if not check_if_tree(rand_pert_edge_index):
            logging.info("WARNING: RANDOM PERTURBATION IS NOT A TREE ANYMORE!")

        rand_pert_data = attack_graph_data.clone()
        rand_pert_data.edge_index = rand_pert_edge_index
        rand_pert_data.edge_attr = torch.ones(rand_pert_edge_index.size(1))
        with torch.no_grad():
            output_rand_pert = model(
                Batch.from_data_list([rand_pert_data.clone()]),
                root_node=root_node_idx,
                remove_not_connected=remove_isolated_components,
                recompute_preprocessing=True,
                unmodified=False,
            )
        
        log_and_accumulate_output(
            clean_data.y,
            probs_clean,
            logits_clean,
            correct_clean,
            margin_clean,
            output_rand_pert,
            sigmoid_threshold,
            accumulated_stats,
            random=True,
        )
        stats_rand, num_stats_rand = basic_edge_and_node_stats(
            clean_data.edge_index,
            rand_pert_edge_index,
            root=root_node_idx,
            num_edges=num_edges,
            num_nodes=num_nodes,
            is_undirected=is_undirected,
        )
        log_and_accumulate_num_stats(accumulated_stats, num_stats_rand, random=True)

    summary_stats = log_summary_stats(accumulated_stats)
    model.forward = model.forward.__wrapped__
    logging.info("End of attack.")
    return {"avg": summary_stats, "all": accumulated_stats}


def get_attack_graph(
    graph_data: Data,
    node_injection_attack: bool,
    total_attack_dataset_graph: None | Data,
    attack_dataset_slice: None | tuple[int, int],
    total_additional_datasets_graph: None | Data,
) -> tuple[Data, torch.Tensor, torch.Tensor]:
    """
    """
    num_edges = graph_data.edge_index.size(1)
    if node_injection_attack:
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
    root_node_idx: None | int,
    remove_isolated_components: bool,
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
        root_node=root_node_idx,
        remove_not_connected=remove_isolated_components,
        recompute_preprocessing=True,
        unmodified=False,
    )

    if not _keep_forward_wrapped:
        model.forward = model.forward.__wrapped__

    return pert_edge_index, perts