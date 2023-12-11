import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from graphgps.attack.prbcd import PRBCDAttack
from torch_geometric.data import Data, Batch
from graphgps.attack.dataset_attack import get_complete_graph
from graphgps.attack.utils_attack import check_if_tree
from graphgps.attack.preprocessing import forward_wrapper
from graphgps.attack.postprocessing import post_process_attack
import logging


# TODO: dataset analysis:
#  - Plot the graphs!
#  - How many levels from root?
#  - Always strict trees?
#  - Are the assumed rules never violated? No retweeting your own tweet/ retweet.
#       -----> no chaining same ids -> not feasible to retweet your own tweet?
#  - Are graph ids (news) repeated, or is each graph on a different news?
#  - How often are node ids repeated?
#     - Within same graph?
#     - In other graphs?
#     - On which levels? Only as multiple tweets of same article link?
#     - Or additionally as multiple retweets of different tweets that have the same link?
#     - Or mixed, one post of the article and a retweet of other users tweet of same article?


def prbcd_attack_test_dataset(
    model,
    datasets: dict[str, Dataset],
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
):
    # TODO: make dataset agnostic with default, but let dataset specific methods be overloaded
    complete_graph_output = get_complete_graph(datasets, device)
    all_nodes, global_test_edge_index, root_masks, root_indices, node_ids = complete_graph_output

    # TODO: undo wrapping through: model.forward = model.forward.__wrapped__ after attack 
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
    
    local_root_offset = torch.arange(len(root_indices) - 1, dtype=torch.int)
    total_examples = 0
    total_clean_correct = 0
    total_pert_correct = 0
    num_edges_added = []
    num_edges_added_connected = []
    num_edges_removed = []
    num_nodes_added = []
    num_nodes_added_connected = []
    num_nodes_removed = []
    count_nodes_removed_index = {}
    # count_nodes_removed_id = {}
    count_nodes_added_index = {}
    # count_nodes_added_id = {}
    count_nodes_added_connected_index = {}
    # count_nodes_added_connected_id = {}
    attack_test_loader = DataLoader(datasets["test"], batch_size=1, shuffle=False)
    for i, clean_data in enumerate(attack_test_loader):
        if num_attacked_graphs and i >= num_attacked_graphs:
            break
        logging.info(f"\nAttacking test graph {i + 1}")
        # ATTACK
        try:
            # we want to attack each graph individually (data loader should have batch size 1)
            model.eval()
            assert clean_data.num_graphs == 1
            num_nodes = clean_data.x.size(0)
            num_edges = clean_data.edge_index.size(1)
            clean_data = clean_data.to(device)
            with torch.no_grad():
                clean_output = model(clean_data)
            # Perturb e_budget of edges:
            global_budget = int(e_budget * num_edges / 2)
            # mask out the other root nodes (should not be able to attack using those!)
            other_roots_mask = root_masks[i, :]
            node_features = all_nodes[other_roots_mask, :]
            # get the global edge index
            edge_index = global_test_edge_index[i]
            root_node = int(edge_index[0, 0].item())
            # check that using the complete graph with all user nodes is equivalent
            global_clean_data = Batch.from_data_list(
                [Data(x=node_features.clone(), edge_index=edge_index, edge_attr=torch.ones(edge_index.size(1)))]
            )
            with torch.no_grad():
                global_graph_output = model(
                    global_clean_data, root_node=root_node, remove_not_connected=True,
                )
            assert torch.allclose(clean_output, global_graph_output, atol=0.001, rtol=0.001)

            # attack: find perturbations
            pert_edge_index, perts = prbcd.attack(
                node_features,
                edge_index,
                clean_data.y,
                budget=global_budget,
                root_node=root_node,
                remove_not_connected=True,
            )

            # check the result of the attack on the model
            # TODO: check if the perturbation violates any other structure rules
            #  (maybe no chaining together 2 same ids?)
            if not check_if_tree(pert_edge_index):
                logging.info("\n\nWARNING: PERTURBATION IS NOT A TREE ANYMORE!!!\n\n")

            pert_data = Batch.from_data_list(
                [
                    Data(
                        x=node_features.clone(),
                        edge_index=pert_edge_index,
                        edge_attr=torch.ones(pert_edge_index.size(1)),
                    )
                ]
            )
            with torch.no_grad():
                pert_output = model(
                    pert_data, root_node=root_node, remove_not_connected=True,
                )
        except KeyboardInterrupt:
            logging.info("Attacks interrupted by user.")
            break
        # RESULTS OF THE ATTACK
        # TODO: also look at the raw logits (before softax, so when something changed from 1.0 to 0.99 
        #   -> could have required a massive change depending on how strong the 1.0 was, or simply weak attack, 
        #      with logits we can see the differnce)
        y_correct = int(clean_data.y)
        y_wrong = int(not y_correct)
        clean_output_prob = torch.sigmoid(clean_output)
        clean_pred = int(clean_output_prob > sigmoid_threshold)
        clean_prob = [1 - float(clean_output_prob.squeeze()), float(clean_output_prob.squeeze())]
        pert_output_prob = torch.sigmoid(pert_output)
        pert_pred = int(pert_output_prob > sigmoid_threshold)
        pert_prob = [1 - float(pert_output_prob.squeeze()), float(pert_output_prob.squeeze())]
        clean_margin = clean_prob[y_correct] - clean_prob[y_wrong]
        pert_margin = pert_prob[y_correct] - pert_prob[y_wrong]
        clean_correct = clean_pred == y_correct
        pert_correct = pert_pred == y_correct
        probs = [f"{p:.4f}" for p in clean_prob]
        logging.info(
            f"CLEAN:     \tcorrect (margin) [prob]:\t"
            f"{str(clean_correct):5} ({f'{clean_margin:.4f}':>7}) [{probs[0]}, {probs[1]}]"
        )
        probs = [f"{p:.4f}" for p in pert_prob]
        logging.info(
            f"PERTURBED: \tcorrect (margin) [prob]:\t"
            f"{str(pert_correct):5} ({f'{pert_margin:.4f}':>7}) [{probs[0]}, {probs[1]}]"
        )
        total_clean_correct += int(clean_correct)
        total_pert_correct += int(pert_correct)
        total_examples += 1
        # TODO: analyze the attack, loading the post_process_function that is defined in cfg
        # TODO: log all interesting stats and append in a list so at the end we still all hav all
        post_process_attack(
            edge_index=edge_index,
            pert_edge_index=pert_edge_index,
            all_nodes=all_nodes,
            node_features=node_features,
            roots_mask=other_roots_mask,
            local_root_offset=local_root_offset,
            node_ids=node_ids,
            num_nodes_added=num_nodes_added,
            num_nodes_added_connected=num_nodes_added_connected,
            num_nodes_removed=num_nodes_removed,
            num_edges_added=num_edges_added,
            num_edges_added_connected=num_edges_added_connected,
            num_edges_removed=num_edges_removed,
            count_nodes_added_index=count_nodes_added_index,
            count_nodes_added_connected_index=count_nodes_added_connected_index,
            count_nodes_removed_index=count_nodes_removed_index,
        )
        logging.info(f"Original number of edges: {num_edges:>5}")
        logging.info(f"Added edges:              {num_edges_added[-1]:>5}")
        logging.info(f"Added edges (connected):  {num_edges_added_connected[-1]:>5}")
        logging.info(f"Removed edges:            {num_edges_removed[-1]:>5}")
        logging.info(f"Original number of nodes: {num_nodes:>5}")
        logging.info(f"Added nodes:              {num_nodes_added[-1]:>5}")
        logging.info(f"Added nodes (connected):  {num_nodes_added_connected[-1]:>5}")
        logging.info(f"Removed nodes:            {num_nodes_removed[-1]:>5}")
    # summary of results and analysis
    clean_acc = total_clean_correct / total_examples
    pert_acc = total_pert_correct / total_examples
    most_added_nodes = sorted([(v, k) for k, v in count_nodes_added_index.items()], reverse=True)
    logging.info(f"PRBCD: Accuracy clean:           {clean_acc:.3f},  Perturbed: {pert_acc:.3f}.")
    logging.info(f"Average number of edges added:   {sum(num_edges_added) / len(num_edges_added):.3f}")
    logging.info(f"Average number of edges removed: {sum(num_edges_removed) / len(num_edges_removed):.3f}")
    logging.info(f"Average number of nodes added:   {sum(num_nodes_added) / len(num_nodes_added):.3f}")
    logging.info(f"Average number of nodes removed: {sum(num_nodes_removed) / len(num_nodes_removed):.3f}")
    logging.info(f"Nodes most frequently added - (freq, global_node_index):\n{most_added_nodes[:10]}")
