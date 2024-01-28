import os.path as osp
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csgraph
from graphgps.loader.master_loader import preformat_UPFD


format = "PyG-UPFD"  # cfg.dataset.format
name = "politifact-profile"  # cfg.dataset.name
dataset_dir = "./datasets"  # cfg.dataset.dir
pyg_dataset_id = format.split('-', 1)[1]
dataset_dir = osp.join(dataset_dir, pyg_dataset_id)
dataset = preformat_UPFD(dataset_dir, name)

degrees = []
max_shortest_paths = []

for graph in dataset:
    n = graph.x.size(0)
    node_degrees = torch.zeros(n, dtype=torch.long).scatter_(value=1, index=graph.edge_index[1, :], dim=0, reduce="add")
    degrees.append(node_degrees)
    adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=n).tocsc()
    distances = csgraph.shortest_path(adj, method="auto", directed=False, unweighted=False)
    max_shortest_path = distances.max()
    max_shortest_paths.append(max_shortest_path)

print("e")
