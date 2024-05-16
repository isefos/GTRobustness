import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.utils import get_laplacian, coalesce
from torch_sparse import spmm
from graphgps.transform.lap_eig import (
    get_dense_eigh,
    get_lap_decomp_stats,
    eigvec_normalizer,
    invert_wrong_signs,
)
import numpy as np


@register_node_encoder("WLapPE")
class WeightedLapPENodeEncoder(torch.nn.Module):
    """Weighted Laplace Positional Embedding node encoder.

    for simplicity, just allow transformer (original)

    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in  # Expected original input node features dim

        pecfg = cfg.posenc_WLapPE
        dim_pe = pecfg.dim_pe  # Size of Laplace PE embedding
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        n_heads = pecfg.n_heads  # Num. attention heads in Trf PE encoder
        post_n_layers = pecfg.post_layers  # Num. layers to apply after pooling
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"LapPE size {dim_pe} is too large for desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_A = nn.Linear(2, dim_pe)

        activation = nn.ReLU
        # Transformer model for LapPE
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_pe, nhead=n_heads, batch_first=True)
        self.pe_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.post_mlp = None
        if post_n_layers > 0:
            # MLP to apply post pooling
            layers = []
            if post_n_layers == 1:
                layers.append(nn.Linear(dim_pe, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                layers.append(activation())
                for _ in range(post_n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.post_mlp = nn.Sequential(*layers)


    def forward(self, batch):
        if batch.get("EigVals") is None:
            add_eig(batch)

        EigVals = batch.EigVals
        EigVecs = batch.EigVecs

        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe

        # PE encoder (Transformer)
        pos_enc = self.pe_encoder(src=pos_enc, src_key_padding_mask=empty_mask[:, :, 0])

        # Set masked parts back to zero
        pos_enc = pos_enc * (~empty_mask[:, :, [0]]).float()

        # Sum pooling
        pos_enc = pos_enc.sum(1)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            batch.pe_LapPE = pos_enc
        return batch


def add_eig(batch):
    assert batch.num_graphs == 1, "On the fly getting PE currently only works for single graphs."
    num_nodes = batch.x.size(0)
    pe_cfg = cfg.posenc_WLapPE
    max_freqs = pe_cfg.eigen.max_freqs
    lap_norm_type = pe_cfg.eigen.laplacian_norm
    lap_norm_type = lap_norm_type.lower() if lap_norm_type is not None else None
    eigvec_norm = pe_cfg.eigen.eigvec_norm
    if lap_norm_type == 'none':
        lap_norm_type = None
    attack_mode = batch.get("attack_mode", False)

    if (not attack_mode) or (batch.edge_attr is None) or not cfg.posenc_WLapPE.eigen.enable_pert_grad:
        pe_cfg = cfg.posenc_WLapPE
        batch.EigVals, batch.EigVecs = get_lap_decomp_stats(
            batch.edge_index,
            batch.edge_attr,
            batch.x.size(0),
            lap_norm_type,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm,
        )
        batch.EigVals = batch.EigVals.repeat(num_nodes, 1)[:, :, None]
        return
    
    # update original eig with perturbation approximation as function of laplacian perturbation
    check_approx = False
    if check_approx:
        out = get_lap_decomp_stats(
            batch.edge_index,
            batch.edge_attr,
            num_nodes,
            lap_norm_type,
            max_freqs,
            eigvec_norm,
            return_lap=True,
        )
        eigenvalues_true, eigenvectors_true, lap_attack_edge_index, lap_attack_edge_attr = out
    else:
        lap_attack_edge_index, lap_attack_edge_attr = get_laplacian(
            batch.edge_index, batch.edge_attr, lap_norm_type, num_nodes=num_nodes,
        )

    delta_lap_edge_index, delta_lap_edge_attr = coalesce(
        torch.cat((lap_attack_edge_index, batch.lap_clean_edge_index), 1),
        torch.cat((lap_attack_edge_attr, -batch.lap_clean_edge_attr)),
        num_nodes=num_nodes,
        reduce="add",
    )
    zero_mask = delta_lap_edge_attr != 0
    delta_lap_edge_index = delta_lap_edge_index[:, zero_mask]
    delta_lap_edge_attr = delta_lap_edge_attr[zero_mask]

    E_clean = batch.E_clean
    U_clean = batch.U_clean
    P = E_clean[:, None] - E_clean[None, :]
    P[P == 0] = float("inf")
    P_inv = 1 / P
    if batch.E_rep_slices_min.size(0) > 0:
        # handle repeated eigenvalues
        basis_transform_repeated_eigenvectors_(
            U_clean,
            delta_lap_edge_index,
            delta_lap_edge_attr,
            num_nodes,
            batch.E_rep_slices_min,
            batch.E_rep_slices_max,
            P_inv,
        )

    # calculate the eigenperturbation from the formulas of matrix perturbation theory
    Q = U_clean.T @ spmm(delta_lap_edge_index, delta_lap_edge_attr, num_nodes, num_nodes, U_clean)
    E_delta = torch.diag(Q)
    U_delta = - U_clean @ (P_inv * Q)
    E_pert = E_clean + E_delta
    U_pert = U_clean + U_delta

    # sort by smallest eigenvalue and select only the smallest max_freqs
    E_pert, idx = torch.topk(E_pert, max_freqs, largest=False, sorted=True)
    U_pert = U_pert[:, idx]

    # normalize the eigenvectors
    U_pert = eigvec_normalizer(U_pert, E_pert, eigvec_norm)

    # pad if max_freq > num_nodes
    if num_nodes < max_freqs:
        evals = F.pad(evals, (0, max_freqs - num_nodes), value=float('nan'))
        evects = F.pad(evects, (0, max_freqs - num_nodes), value=float('nan'))

    eigenvalues = torch.zeros((max_freqs, ), device=batch.x.device)
    eigenvalues[1:] = E_pert[1:]

    if check_approx:
        E_error = torch.nan_to_num(E_pert - eigenvalues_true, nan=0.0, posinf=0.0, neginf=0.0)
        E_error = E_error.abs()
        U_pert_correct_sign = invert_wrong_signs(eigenvectors_true, U_pert)
        U_error = torch.nan_to_num(U_pert_correct_sign - eigenvectors_true, nan=0.0, posinf=0.0, neginf=0.0)
        U_error = (U_error ** 2).sum(0).sqrt()
        print(f"Min eigval error: {E_error.min().item()}")
        print(f"Avg eigval error: {E_error.mean().item()}")
        print(f"Max eigval error: {E_error.max().item()}")
        print(f"Min eigvec error: {U_error.min().item()}")
        print(f"Avg eigvec error: {U_error.mean().item()}")
        print(f"Max eigvec error: {U_error.max().item()}")

    eigenvalues = eigenvalues.repeat(num_nodes, 1)[:, :, None]
    batch.EigVals, batch.EigVecs = eigenvalues, U_pert


@torch.no_grad
def basis_transform_repeated_eigenvectors_(
    U,
    delta_lap_edge_index,
    delta_lap_edge_attr,
    num_nodes,
    slices_min,
    slices_max,
    P_inv,
):
    # find the repeated eigenvalue blocks, set the corresponding eigenvector entries
    for i in range(slices_min.size(0)):
        start, end = slices_min[i].item(), slices_max[i].item()
        U_block = U[:, start:end]  # n x m
        # project L_delta into U basis -- m x m <- ((m x n) x (n x n) x (n x m))
        L_delta_block_p: torch.Tensor
        L_delta_block_p = U_block.T @ spmm(delta_lap_edge_index, delta_lap_edge_attr, num_nodes, num_nodes, U_block)
        # do the eigendecomposition in scipy
        _, U_p = get_dense_eigh(L_delta_block_p.cpu().numpy().astype(np.float64), need_full=True)
        U_p = torch.from_numpy(U_p).float().to(U.device)
        # set the block in U to be equal to the eigenvectors of L_delta_block_p projected back
        U[:, start:end] = U_block @ U_p
        # set the entries in P_inv to zero
        P_inv[start:end, start:end] = 0
