import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor


def add_full_rrwp(data: Data, walk_length: int):
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(
        edge_index,
        edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_mask = deg != 0
    deg_inv = torch.zeros_like(deg)
    deg_inv[deg_mask] = 1.0 / deg[deg_mask]
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = [
        torch.eye(num_nodes, dtype=torch.float, device=edge_index.device),
        adj,
    ]

    out = adj
    for _ in range(walk_length - 2):
        out = out @ adj
        pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1) # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1) # n x k

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)

    data.rrwp = abs_pe
    data.rrwp_index = rel_pe_idx
    data.rrwp_val = rel_pe_val
    data.log_deg = torch.log1p(deg)

    return data
