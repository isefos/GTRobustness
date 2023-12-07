import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from graphgps.attack.prbcd import PRBCDAttack
from typing import Callable
import functools
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_scipy_sparse_matrix, index_to_mask, subgraph
from scipy.sparse.csgraph import breadth_first_order
from graphgps.attack.preprocessing import node_in_graph_prob
import pickle
import numpy as np


def forward_wrapper(forward: Callable, is_undirected: bool) -> Callable:

    @functools.wraps(forward)
    def wrapped_forward(data, root_node: None | int = None, remove_not_connected: bool = False):
        if remove_not_connected:
            assert root_node is not None, "The root node must be specified"
            new_data, new_root_node = get_only_root_graph(data, root_node)

            # TODO: where to get bool: undirected from?
            new_data.node_probs = node_in_graph_prob(
                edge_index=new_data.edge_index,
                edge_weights=new_data.edge_attr,
                batch=new_data.batch,
                undirected=is_undirected,
                root_node=new_root_node,
                num_iterations=4,
            )
            return forward(new_data)[0]
        else:
            return forward(data)[0]

    return wrapped_forward



def prbcd_attack_test_dataset(
    model,
    datasets: dict[str, Dataset],
    device: torch.device,
    attack_loss: Callable,
    id_mapping_path,
    graph_indices_paths,
    limit_number_attacks: int,
    e_budget: float,
    block_size: int,  # e.g. 1_000
    lr: float,  # e.g. 1_000
    is_undirected: bool,
    sigmoid_threshold: float,
    existing_node_prob_multiplier: int,
    allow_existing_graph_pert: bool,
):
    # TODO: make dataset agnostic with default, but let dataset specific methods be overloaded 
    # TODO: use logger instead of print
    complete_graph_output = get_complete_graph(datasets, device, id_mapping_path, graph_indices_paths)
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
        if i >= limit_number_attacks:
            break
        print(f"\nAttacking test graph {i + 1}")
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
                print("\n\nWARNING: PERTURBATION IS NOT A TREE ANYMORE!!!\n\n")

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
            print("Attacks interrupted by user.")
            break
        # RESULTS OF THE ATTACK
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
        print(
            f"CLEAN:     \tcorrect (margin) [prob]:\t"
            f"{str(clean_correct):5} ({f'{clean_margin:.4f}':>7}) [{probs[0]}, {probs[1]}]"
        )
        probs = [f"{p:.4f}" for p in pert_prob]
        print(
            f"PERTURBED: \tcorrect (margin) [prob]:\t"
            f"{str(pert_correct):5} ({f'{pert_margin:.4f}':>7}) [{probs[0]}, {probs[1]}]"
        )
        total_clean_correct += int(clean_correct)
        total_pert_correct += int(pert_correct)
        total_examples += 1
        # analyze the attack
        result_dict = post_process_attack(
            edge_index=edge_index,
            pert_edge_index=pert_edge_index,
            all_nodes=all_nodes,
            node_features=node_features,
            roots_mask=other_roots_mask,
            local_root_offset=local_root_offset,
            node_ids=node_ids,
        )

        # TODO: analysis on the ids -> have we basically added the same node twice (but with different indices)?
        #  check if some of the added nodes have the same id (as any other node in graph) using node_ids
        #  -> are we violating any graph properties (eg. no chaining same id?)
        # edges
        edges = result_dict["clean"]["edges"]
        pert_edges = result_dict["pert"]["edges"]
        pert_edges_connected = result_dict["pert_r"]["edges"]
        added_edges = pert_edges - edges
        added_edges_connected = pert_edges_connected - edges
        removed_edges = edges - pert_edges
        num_edges_added.append(len(added_edges))
        num_edges_added_connected.append(len(added_edges_connected))
        num_edges_removed.append(len(removed_edges))
        # nodes
        nodes = result_dict["clean"]["nodes"]
        pert_nodes = result_dict["pert"]["nodes"]
        pert_nodes_connected = result_dict["pert_r"]["nodes"]
        added_nodes = pert_nodes - nodes
        added_nodes_connected = pert_nodes_connected - nodes
        removed_nodes = nodes - pert_nodes
        num_nodes_added.append(len(added_nodes))
        for node in added_nodes:
            if node in count_nodes_added_index:
                count_nodes_added_index[node] += 1
            else:
                count_nodes_added_index[node] = 1
        num_nodes_added_connected.append(len(added_nodes_connected))
        for node in added_nodes_connected:
            if node in count_nodes_added_connected_index:
                count_nodes_added_connected_index[node] += 1
            else:
                count_nodes_added_connected_index[node] = 1
        num_nodes_removed.append(len(removed_nodes))
        for node in removed_nodes:
            if node in count_nodes_removed_index:
                count_nodes_removed_index[node] += 1
            else:
                count_nodes_removed_index[node] = 1
        print(f"Original number of edges: {num_edges:>5}")
        print(f"Added edges:              {len(added_edges):>5}")
        print(f"Added edges (connected):  {len(added_edges_connected):>5}")
        print(f"Removed edges:            {len(removed_edges):>5}")
        print(f"Original number of nodes: {num_nodes:>5}")
        print(f"Added nodes:              {len(added_nodes):>5}")
        print(f"Added nodes (connected):  {len(added_nodes_connected):>5}")
        print(f"Removed nodes:            {len(removed_nodes):>5}")
    # summary of results and analysis
    clean_acc = total_clean_correct / total_examples
    pert_acc = total_pert_correct / total_examples
    most_added_nodes = sorted([(v, k) for k, v in count_nodes_added_index.items()], reverse=True)
    return_dict = {
        "clean_acc": clean_acc,
        "pert_acc": pert_acc,
        "num_edges_added": num_edges_added,
        "num_edges_removed": num_edges_removed,
        "num_nodes_added": num_nodes_added,
        "num_nodes_removed": num_nodes_removed,
        "most_added_nodes": most_added_nodes,
    }
    return return_dict


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


def get_global_indices(
    nodes: set[int],
    edges: set[frozenset[int]],
    all_nodes: torch.Tensor,
    node_features: torch.Tensor,
    root_mask: torch.Tensor,
    local_root_offset: torch.Tensor,
    node_ids: list[str],
) -> tuple[set[int], list[str], set[frozenset[int]], list[frozenset[str]]]:
    """reverse root mask index shift"""
    nodes_indices = torch.Tensor(list(nodes))
    masked_roots = torch.nonzero(~root_mask).squeeze()
    local_masked_roots = masked_roots - local_root_offset
    root_mask_offset = torch.searchsorted(
        local_masked_roots,
        nodes_indices,
        right=True,
    )
    nodes_mapping: dict[int, int] = {}
    for i, node_index in enumerate(nodes_indices):
        nodes_mapping[int(node_index)] = int(node_index + root_mask_offset[i])
    # check that it is correct -> indexes the same features
    for local_node_index, global_node_index in nodes_mapping.items():
        assert torch.allclose(all_nodes[global_node_index, :], node_features[local_node_index, :])
    global_nodes = set(nodes_mapping.values())
    # additionally construct it for the ids (not unique -> use list)
    nodes_ids = [node_ids[i] for i in global_nodes]
    # for the edges
    global_edges = set(frozenset(nodes_mapping[node_index] for node_index in edge) for edge in edges)
    edges_ids = [frozenset(node_ids[i] for i in edge) for edge in global_edges]
    return global_nodes, nodes_ids, global_edges, edges_ids


def post_process_attack(
    edge_index: torch.Tensor,
    pert_edge_index: torch.Tensor,
    all_nodes: torch.Tensor,
    node_features: torch.Tensor,
    roots_mask: torch.Tensor,
    local_root_offset: torch.Tensor,
    node_ids: list[str],
):
    def to_global(edges: set[frozenset[int]]):
        nodes = set()
        for edge in edges:
            nodes.update(edge)
        nodes, nodes_ids, edges, edges_ids = get_global_indices(
            nodes, edges, all_nodes, node_features, roots_mask, local_root_offset, node_ids,
        )
        return {"nodes": nodes, "nodes_ids": nodes_ids, "edges": edges, "edges_ids": edges_ids}

    root = int(edge_index[0, 0])
    reached = get_reached_nodes(root, pert_edge_index)
    # edges:
    edges = set(frozenset(int(i) for i in e.squeeze()) for e in torch.split(edge_index, 1, dim=1))
    pert_edges = set(frozenset(int(i) for i in e.squeeze()) for e in torch.split(pert_edge_index, 1, dim=1))
    pert_r_edges = set(e for e in pert_edges if all(n in reached for n in e))
    # nodes:
    result_dict = {}
    for category, category_edges in zip(["clean", "pert", "pert_r"], [edges, pert_edges, pert_r_edges]):
        result_dict[category] = to_global(category_edges)
    return result_dict


def add_graph(
    graph,
    graph_node_ids: list[str],
    node_ids: list[str],
    id_to_indices: dict[str, list[int]],
    all_nodes: list[torch.Tensor],
    roots: set[int],
) -> tuple[None | int, list[int]]:
    """helper function for adding a single graph to the complete graph"""
    # make sure the first node (root, index 0) is always the first entry in the edge_index as well
    assert graph.edge_index[0, 0] == 0
    node_features = torch.split(graph.x, 1, dim=0)
    # check that we have an id for each node
    assert len(graph_node_ids) == len(node_features)
    global_indices_in_graph: set[int] = set()
    # first one is root, after that set to False
    is_root = True
    root_global_index = None
    # for remembering the map of local indices to global indices:
    node_index_mapping: list[int] = []
    for node_f, node_id in zip(node_features, graph_node_ids):
        global_indices = id_to_indices.get(node_id, None)
        new_id = global_indices is None
        if new_id or is_root:
            # roots need to have their own node features so they can be masked out
            # new node, needs to be added
            global_index = len(all_nodes)
            all_nodes.append(node_f)
            id_to_indices[node_id] = [global_index]
            node_ids.append(node_id)
            if is_root:
                # check wether root id is already saved, should be unique...
                assert new_id
                roots.add(global_index)
                root_global_index = global_index
                # all the next nodes are not root anymore
                is_root = False
        else:
            # since is_root=False, assert that we are are not taking a root index (would be masked out!)
            assert not roots.intersection(global_indices)
            # existing node
            needs_new_entry = True
            global_index = len(all_nodes)
            for saved_index in global_indices:
                assert node_ids[saved_index] == node_id
                if saved_index in global_indices_in_graph:
                    continue
                # check that the node with same id actually has the same features
                if not torch.allclose(node_f, all_nodes[saved_index]):
                    continue
                needs_new_entry = False
                global_index = saved_index
            if needs_new_entry:
                all_nodes.append(node_f)
                id_to_indices[node_id].append(global_index)
                node_ids.append(node_id)
        global_indices_in_graph.add(global_index)
        node_index_mapping.append(global_index)
    return root_global_index, node_index_mapping


def get_complete_graph(
    datasets, device, id_mapping_path, graph_indices_paths,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, list[int], list[str]]:
    """
    Root nodes are special -> they are not twitter users, but rather news articles (URLs in tweets)
    themselves. So it doesn't make sense to include them in the total graph. (Cannot connect a root
    node as a retweet to other root node). So we have to mask the other root nodes out.
    """
    # first we need to get the id mappings from the file:
    with open(id_mapping_path, "rb") as f:
        id_map: dict[int, str] = pickle.load(f)
    # the roots are strings (e.g. politifact1111), the users are integers
    # we can use this to separate the ids by graph
    id_mapping_per_graph: list[list[str]] = []
    for identifier in id_map.values():
        begin_new_graph: bool = False
        try:
            int(identifier)
        except ValueError:
            begin_new_graph = True
        if begin_new_graph:
            id_mapping_per_graph.append([identifier])
        else:
            id_mapping_per_graph[-1].append(identifier)
    # then we need to get the graph indexes to be able to look up the correct ids
    graph_indices: dict[str, list[int]] = {}
    for dataset_mode in ["train", "val", "test"]:
        with open(graph_indices_paths[dataset_mode], "rb") as f:
            graph_indices[dataset_mode] = [int(i) for i in np.load(f)]

    # now we can start constructing the complete graph
    all_nodes: list[torch.Tensor] = []
    all_node_ids: list[str] = []
    id_to_indices: dict[str, list[int]] = {}
    roots: set[int] = set()
    root_indices: list[int] = []
    # first for the train and val sets
    graph_index: int = 0
    for dataset_mode in ["train", "val"]:
        dataset = datasets[dataset_mode]
        dataset_graph_indices = graph_indices[dataset_mode]
        for graph_index, graph in zip(dataset_graph_indices, dataset):
            assert check_if_tree(graph.edge_index)
            graph_node_ids = id_mapping_per_graph[graph_index]
            root_index, _ = add_graph(graph, graph_node_ids, all_node_ids, id_to_indices, all_nodes, roots)
            assert root_index is not None
            root_indices.append(root_index)
    # for the test set (which we will attack) we also need to remap the edges
    test_edge_index: list[torch.Tensor] = []
    test_root_indices: list[int] = []
    dataset = datasets["test"]
    dataset_graph_indices = graph_indices["test"]
    for graph_index, graph in zip(dataset_graph_indices, dataset):
        assert check_if_tree(graph.edge_index)
        graph_node_ids = id_mapping_per_graph[graph_index]
        root_index, node_index_mapping = add_graph(
            graph, graph_node_ids, all_node_ids, id_to_indices, all_nodes, roots,
        )
        assert root_index is not None
        test_root_indices.append(root_index)
        # remap the edge indices to global, additionally take into consideration the root masks
        node_index_mapping = torch.tensor(node_index_mapping, dtype=torch.long, device=device)
        root_mask_offset = torch.searchsorted(torch.Tensor(root_indices), node_index_mapping)
        node_index_mapping -= root_mask_offset
        test_edge_index.append(torch.reshape(node_index_mapping[graph.edge_index.flatten()], (2, -1)))
        root_indices.append(root_index)
    # construct the root masks
    root_masks = torch.ones((len(test_edge_index), len(all_nodes)), dtype=torch.bool)
    for i, root_index in enumerate(test_root_indices):
        root_masks[i, list(roots - set([root_index]))] = False
    # finally the complete graph node features
    complete_graph = torch.cat(all_nodes, dim=0).to(device)
    return complete_graph, test_edge_index, root_masks, root_indices, all_node_ids


def get_undirected_graph_node_children(edge_index: torch.Tensor) -> dict[int, set[int]]:
    children_nodes: dict[int, set[int]] = {}
    for i in range(edge_index.size(1)):
        edge_node_1 = int(edge_index[0, i])
        edge_node_2 = int(edge_index[1, i])
        if edge_node_1 not in children_nodes:
            children_nodes[edge_node_1] = set()
        if edge_node_2 not in children_nodes:
            children_nodes[edge_node_2] = set()
        # TODO: if edge weight is nonzero?
        children_nodes[edge_node_1].add(edge_node_2)
        # TODO: if undirected and edge weight is nonzero?
        children_nodes[edge_node_2].add(edge_node_1)
    return children_nodes


def check_if_tree(edge_index: torch.Tensor) -> bool:
    root = int(edge_index[0, 0])
    children_nodes = get_undirected_graph_node_children(edge_index)
    # DFS check wheather a child has already been visited from different parent
    to_explore: list[tuple[int, None | int]] = [(root, None)]
    reached: set[int] = set((root, ))
    is_tree = True
    while to_explore:
        current_node, parent_node = to_explore.pop()
        children = children_nodes[current_node]
        for child_node in children:
            if child_node == parent_node:
                continue
            if child_node in reached:
                is_tree = False
                break
            to_explore.append((child_node, current_node))
        if not is_tree:
            break
        reached |= children
    return is_tree


def get_reached_nodes(root: int, edge_index: torch.Tensor) -> set[int]:
    children_nodes = get_undirected_graph_node_children(edge_index)
    # DFS to find all reachable nodes:
    to_explore = [root]
    reached: set[int] = set((root, ))
    while to_explore:
        current_node = to_explore.pop()
        children = children_nodes[current_node]
        for child_node in children:
            if child_node in reached:
                continue
            to_explore.append(child_node)
        reached |= children
    return reached


def get_only_root_graph(data, root_node: int):
    # only works for a single graph
    assert torch.all(data.batch == 0)
    num_nodes = data.x.size(0)
    # do graph traversal starting from root to find the root component
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)
    bfs_order = breadth_first_order(adj, root_node, return_predecessors=False)
    subset_mask = index_to_mask(torch.tensor(bfs_order, dtype=torch.long), size=num_nodes)
    edge_index, edge_attr = subgraph(subset_mask, data.edge_index, data.edge_attr, relabel_nodes=True)
    new_data = Batch.from_data_list(
        [Data(x=data.x[subset_mask, :].clone(), edge_index=edge_index, edge_attr=edge_attr)]
    )
    new_root_node = int(subset_mask[:root_node].sum().item())
    assert torch.allclose(data.x[root_node, :], new_data.x[new_root_node, :])
    return new_data, new_root_node
