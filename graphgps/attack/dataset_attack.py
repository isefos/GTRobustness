import torch
from graphgps.attack.utils_attack import check_if_tree


# TODO: implement default for graph classification -> injection, just take all node in dataset together
# default for transductive -> just return the same graph...
# specific: e.g. UPFD


def get_complete_graph(
    datasets, device,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, list[int], list[str]]:
    """
    Root nodes are special -> they are not just twitter users, but rather include the news articles
    themselves. So it doesn't make sense to include them in the total graph. (Cannot connect a root
    node as a retweet to other root node). So we have to mask the other root nodes out.
    """
    # construct the complete graph
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