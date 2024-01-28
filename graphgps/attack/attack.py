import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from graphgps.attack.prbcd import PRBCDAttack
from torch_geometric.data import Data, Batch
from graphgps.attack.dataset_attack import get_total_dataset_graphs, get_augmented_graph
from graphgps.attack.utils_attack import check_if_tree
from graphgps.attack.preprocessing import forward_wrapper
from graphgps.attack.postprocessing import (
    log_and_accumulate_output,
    basic_edge_and_node_stats,
    log_and_accumulate_pert_stats,
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
        "num_edges": [],
        "num_edges_added": [],
        "num_edges_added_connected": [],
        "num_edges_removed": [],
        "num_nodes": [],
        "num_nodes_added": [],
        "num_nodes_added_connected": [],
        "num_nodes_removed": [],
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
        logging.info(f"Attacking graph {i + 1}")

        num_nodes = clean_data.x.size(0)
        num_edges = clean_data.edge_index.size(1)
        clean_data.to(device)
        with torch.no_grad():
            clean_output = model(clean_data.clone())

        if clean_data.y.size(0) == 1:
            # binary classification
            y_binary = int(clean_data.y[0].item())
            y_multiclass = torch.zeros(2, dtype=torch.long)
            y_multiclass[y_binary] = 1
            prob_clean_binary = torch.sigmoid(clean_output[0, :])
            class_index_pred_clean = int(prob_clean_binary > sigmoid_threshold)
        else:
            y_multiclass = clean_data.y
            class_index_pred_clean = int(clean_output[0, :].argmax().item())

        class_index_gt = int(y_multiclass.argmax().item())
        correct = class_index_pred_clean == class_index_gt
        # if not correct:
        #     logging.info("Skipping graph, because it is already not correctly classified.")
        #     continue

        # TODO: allow for other ways to define the budget
            
        budget_edges = num_edges / 2 if is_undirected else num_edges
        global_budget = int(e_budget * budget_edges)

        clean_data_augmented, pert_edge_index, perts = attack_single_graph(
            graph_data=clean_data.get_example(0),
            is_undirected=is_undirected,
            model=model,
            attack=prbcd,
            global_budget=global_budget,
            node_injection_attack=node_injection_attack,
            total_attack_dataset_graph=total_attack_dataset_graph,
            attack_dataset_slice=None if attack_dataset_slices is None else attack_dataset_slices[i],
            total_additional_datasets_graph=total_additional_datasets_graph,
            root_node_idx=root_node_idx,
            remove_isolated_components=remove_isolated_components,
            _model_forward_already_wrapped=True,
            _keep_forward_wrapped=True,
            _check_augmentation_correctness=False,
            _check_against_precomputed_output=clean_output,
        )

        num_modified_edges = perts.size(1)
        logging.info(f"Perturbation uses {100 * num_modified_edges / global_budget:.1f}% of the given attack budget.")

        # TODO: add tree generation (minimum spanning tree) to attack (when specified)

        # TODO: add dataset specific analysis functions:
        #   - UPFD: check the twitter IDs, are we adding same ID (user) multiple times? 
        #       Maybe even chained (retweeting himself)

        if not check_if_tree(pert_edge_index):
            logging.info("WARNING: PERTURBATION IS NOT A TREE ANYMORE!")

        pert_data = clean_data_augmented.clone()
        pert_data.edge_index = pert_edge_index
        pert_data.edge_attr = torch.ones(pert_edge_index.size(1))
        with torch.no_grad():
            pert_output = model(
                Batch.from_data_list([pert_data.clone()]),
                root_node=root_node_idx,
                remove_not_connected=remove_isolated_components,
                recompute_preprocessing=True,
                unmodified=False,
            )
        
        log_and_accumulate_output(
            clean_data.y, clean_output, pert_output, sigmoid_threshold, accumulated_stats,
        )

        stats, num_stats = basic_edge_and_node_stats(clean_data.edge_index, pert_edge_index, root=root_node_idx)
        assert num_edges == num_stats["num_edges"]["clean"]
        assert num_nodes == num_stats["num_nodes"]["clean"]
        if is_undirected:
            for key, value in num_stats["num_edges"].items():
                num_stats["num_edges"][key] = value // 2
        log_and_accumulate_pert_stats(accumulated_stats, num_stats)

        ######## RANDOM

        clean_data_augmented, pert_edge_index, perts = attack_single_graph(
            graph_data=clean_data.get_example(0),
            is_undirected=is_undirected,
            model=model,
            attack=prbcd,
            global_budget=global_budget,
            node_injection_attack=node_injection_attack,
            total_attack_dataset_graph=total_attack_dataset_graph,
            attack_dataset_slice=None if attack_dataset_slices is None else attack_dataset_slices[i],
            total_additional_datasets_graph=total_additional_datasets_graph,
            root_node_idx=root_node_idx,
            remove_isolated_components=remove_isolated_components,
            random_attack=True,
            _model_forward_already_wrapped=True,
            _keep_forward_wrapped=True,
            _check_augmentation_correctness=False,
            _check_against_precomputed_output=clean_output,
        )

        if not check_if_tree(pert_edge_index):
            logging.info("WARNING: RANDOM PERTURBATION IS NOT A TREE ANYMORE!")

        pert_data = clean_data_augmented.clone()
        pert_data.edge_index = pert_edge_index
        pert_data.edge_attr = torch.ones(pert_edge_index.size(1))
        with torch.no_grad():
            pert_output = model(
                Batch.from_data_list([pert_data.clone()]),
                root_node=root_node_idx,
                remove_not_connected=remove_isolated_components,
                recompute_preprocessing=True,
                unmodified=False,
            )
        
        log_and_accumulate_output(
            clean_data.y, clean_output, pert_output, sigmoid_threshold, accumulated_stats, random=True,
        )

        # stats, num_stats = basic_edge_and_node_stats(clean_data.edge_index, pert_edge_index, root=root_node_idx)
        # assert num_edges == num_stats["num_edges"]["clean"]
        # assert num_nodes == num_stats["num_nodes"]["clean"]
        # if is_undirected:
        #     for key, value in num_stats["num_edges"].items():
        #         num_stats["num_edges"][key] = value // 2
        # log_and_accumulate_pert_stats(accumulated_stats, num_stats)

    summary_stats = log_summary_stats(accumulated_stats)
    model.forward = model.forward.__wrapped__
    logging.info("End of attack.")
    return {"avg": summary_stats, "all": accumulated_stats}


def attack_single_graph(
    graph_data: Data,
    is_undirected: bool,
    model,
    attack: PRBCDAttack,
    global_budget: int,
    node_injection_attack: bool,
    total_attack_dataset_graph: None | Data,
    attack_dataset_slice: None | tuple[int, int],
    total_additional_datasets_graph: None | Data,
    root_node_idx: None | int,
    remove_isolated_components: bool,
    random_attack: bool = False,
    _model_forward_already_wrapped: bool = False,
    _keep_forward_wrapped: bool = False,
    _check_augmentation_correctness: bool = False,
    _check_against_precomputed_output: None | torch.Tensor = None,
) -> tuple[Data, torch.Tensor, torch.Tensor]:
    """
    """
    if not _model_forward_already_wrapped:
        model.forward = forward_wrapper(model.forward, is_undirected)

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

        if _check_augmentation_correctness:
            graph_data_augmented.edge_attr = torch.ones(graph_data_augmented.edge_index.size(1))
            with torch.no_grad():
                augmented_output = model(
                    Batch.from_data_list([graph_data_augmented.clone()]),
                    root_node=root_node_idx,
                    remove_not_connected=remove_isolated_components,
                    recompute_preprocessing=True,
                    unmodified=False,
                )
            if _check_against_precomputed_output is None:
                with torch.no_grad():
                    _check_against_precomputed_output = model(graph_data.clone())
            assert torch.allclose(_check_against_precomputed_output, augmented_output, atol=0.001, rtol=0.001)
    else:
        graph_data_augmented = graph_data

    if random_attack:
        pert_edge_index, perts = attack.attack_random_baseline(
            graph_data_augmented.x,
            graph_data_augmented.edge_index,
            graph_data.y,
            budget=global_budget,
            root_node=root_node_idx,
            remove_not_connected=remove_isolated_components,
            recompute_preprocessing=True,
            unmodified=False,
        )
    else:
        pert_edge_index, perts = attack.attack(
            graph_data_augmented.x,
            graph_data_augmented.edge_index,
            graph_data.y,
            budget=global_budget,
            root_node=root_node_idx,
            remove_not_connected=remove_isolated_components,
            recompute_preprocessing=True,
            unmodified=False,
        )

    if not _keep_forward_wrapped:
        model.forward = model.forward.__wrapped__

    return graph_data_augmented, pert_edge_index, perts