import graphgps
from torch_geometric.graphgym.config import cfg, set_cfg
set_cfg(cfg)
from yacs.config import CfgNode as CN
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csgraph
from graphgps.loader.master_loader import load_dataset_master, log_loaded_dataset
import logging
from torch_geometric.graphgym.loader import set_dataset_info
from torch_geometric.utils import mask_to_index


logging.basicConfig(level=logging.INFO)
config_file = "./configs_local_debug/GCN/ENZYMES.yaml"
with open(config_file, "r") as f:
    cfg_file = CN._load_cfg_from_file(f)
cfg_file = cfg_file.graphgym
cfg.merge_from_other_cfg(cfg_file)
dataset = load_dataset_master(cfg.dataset.format, cfg.dataset.name, "./datasets")
log_loaded_dataset(dataset, cfg.dataset.format, cfg.dataset.name)


datasets = []
set_dataset_info(dataset)
# train loader
if cfg.dataset.task == 'graph':
    id = dataset.data['train_graph_index']
    datasets.append(dataset[id])
    delattr(dataset.data, 'train_graph_index')
    for i in range(cfg.share.num_splits - 1):
        split_names = ['val_graph_index', 'test_graph_index']
        id = dataset.data[split_names[i]]
        datasets.append(dataset[id])
        delattr(dataset.data, split_names[i])
else:
    datasets = [dataset]

splits = ["train", "val", "test"][:len(datasets)]


for dataset, split in zip(datasets, splits):
    print(f"\n{split}")

    num_graphs = len(dataset)
    print(f"num graphs: {num_graphs}")
    num_nodes = dataset.x.size(0)
    print(f"num nodes: {num_nodes}")
    num_nodes_per_graph = []
    num_edges = dataset.edge_index.size(1)
    print(f"num edges: {num_edges}")
    degrees = {}
    max_shortest_paths = []

    for i in range(num_graphs):
        graph = dataset[i]

        train_mask = graph.get("train_mask")
        if train_mask is not None:
            train_nodes = mask_to_index(train_mask)
            print(f"train nodes: {train_nodes}")
            print(f"val nodes: {mask_to_index(graph.get('val_mask'))}")
            # print(f"test nodes: {mask_to_index(graph.get('test_mask'))}")

        n = graph.x.size(0)
        num_nodes_per_graph.append(n)
        node_degrees = torch.zeros(n, dtype=torch.long).scatter_(value=1, index=graph.edge_index[1, :], dim=0, reduce="add")
        deg, count = torch.unique(node_degrees, return_counts=True)
        for d, c in zip(deg, count):
            d, c = d.item(), c.item()
            if d in degrees:
                degrees[d] += c
            else:
                degrees[d] = c
        adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=n).tocsc()
        distances = csgraph.shortest_path(adj, method="auto", directed=False, unweighted=False)
        max_shortest_path = distances.max()
        max_shortest_paths.append(max_shortest_path)

    num_nodes_per_graph = torch.tensor(num_nodes_per_graph)
    print(f"Min nodes: {num_nodes_per_graph.min().item()}")
    print(f"Avg nodes: {torch.mean(num_nodes_per_graph.float()).item():.2f}")
    print(f"Max nodes: {num_nodes_per_graph.max().item()}")

    max_shortest_paths = torch.tensor(max_shortest_paths)
    msps, counts = torch.unique(max_shortest_paths, return_counts=True)
    print("Encountered longest shortest paths:")
    for msp, c in zip(msps, counts):
        print(f"  {msp.item()}: {int(c.item())}")
    print(f"Min longest shortest path: {max_shortest_paths.min().item()}")
    print(f"Avg longest shortest path: {max_shortest_paths.mean().item():.2f}")
    print(f"Max longest shortest path: {max_shortest_paths.max().item()}")

    degrees = sorted([(k, v) for k, v in degrees.items()])
    print("degrees:")
    for (d, c) in degrees:
        print(f"  {d}: {c}")
