import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import numpy as np


@torch.no_grad
def get_lap_decomp_stats(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int | None,
    lap_norm_type: str | None,
    max_freqs: int,
    eigvec_norm: str,
    pad_too_small: bool = True,
    need_full: bool = False,
    return_lap: bool = False,
):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.
    """
    lap_norm_type = lap_norm_type.lower() if lap_norm_type is not None else None
    if lap_norm_type == 'none':
        lap_norm_type = None

    L_edge_index, L_edge_attr = get_laplacian(edge_index, edge_attr, lap_norm_type, num_nodes=num_nodes)
    L = to_scipy_sparse_matrix(L_edge_index, L_edge_attr, num_nodes)

    E, U = None, None
    if not need_full and (4 * max_freqs) < num_nodes:
        try:
            E, U = eigsh(L, k=max_freqs, which='SM', return_eigenvectors=True)
        except:
            pass
    if E is None:
        # do dense calculation
        E, U = eigh(L.toarray())

    if not need_full:
        E = E[:max_freqs]
        U = U[:, :max_freqs]

    idx = E.argsort()
    E = E[idx]
    U = U[:, idx]

    evals = torch.from_numpy(E).float().clamp_min(0).to(edge_index.device)
    evals[0] = 0
    evects = torch.from_numpy(U).float().to(edge_index.device)

    # Normalize and pad eigen vectors.
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)

    # Pad if less than max_freqs.
    if num_nodes < max_freqs and pad_too_small:
        evals = F.pad(evals, (0, max_freqs - num_nodes), value=float('nan'))
        evects = F.pad(evects, (0, max_freqs - num_nodes), value=float('nan'))

    if return_lap:
        return evals, evects, L_edge_index, L_edge_attr

    return evals, evects


def eigvec_normalizer(EigVecs, EigVals, normalization):
    """
    Implement different eigenvector normalizations.
    """
    eps=1e-12
    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / torch.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs


def get_pert_diff_eigenvects(E, U):
    # eta is the minimal space we would like between two eigenvalues
    eta = 1e-5

    E_diff = torch.diff(E)

    # check if it has "repeated" eigenvalues
    needs_pert = torch.any(E_diff < eta)

    if not needs_pert:
        return None
    
    P = get_ev_pert(E, E_diff, U, eta)
    return P


def get_repeated_eigenvalue_slices(E, eta):
    E_diff = torch.diff(E)
    # check if it has "repeated" eigenvalues
    if not torch.any(E_diff < eta):
        return None, None

    pad = E_diff.new_zeros((1, ), dtype=bool)
    edges = torch.diff(torch.cat((pad, E_diff < eta, pad)).to(dtype=torch.int64))
    slices_min = torch.nonzero(edges == 1).flatten()
    slices_max = torch.nonzero(edges == -1).flatten() + 1
    return slices_min, slices_max


def get_ev_pert(E: torch.Tensor, E_diff: torch.Tensor, U: torch.Tensor, eta: float):
    pad = E_diff.new_zeros((1, ), dtype=bool)
    edges = torch.diff(torch.cat((pad, E_diff < eta, pad)).to(dtype=torch.int64))
    slices_min = torch.nonzero(edges == 1).flatten()
    slices_max = torch.nonzero(edges == -1).flatten() + 1
    multiplicities = slices_max - slices_min

    # space to left of smallest and to the right of larges are zero
    # (so we ensure that the total range of eigenvalues stays the same)
    pad = E_diff.new_zeros((1, ))
    spaces = torch.cat((pad, E_diff, pad))
    # where two repeated eigenvalues are next to each other, each gets half the space to split and expand
    spaces[1:-1][torch.diff(edges) == 2] /= 2
    space_left = spaces[slices_min]
    space_right = spaces[slices_max]

    eig_pert_diag = torch.zeros_like(E)
    for rep in range(slices_min.size(0)):
        m = multiplicities[rep]
        needed_space = (m - 1) * eta
        min_range = (-needed_space / 2).item()
        min_possible = -(space_left[rep]).item() + eta
        right_offset_needed = max(0, min_possible - min_range)
        max_range = (needed_space / 2).item()
        max_possible = (space_right[rep]).item() - eta
        left_offset_needed = max(0, max_range - max_possible)
        left = max(min_range - left_offset_needed, min_possible)
        right = min(max_range + right_offset_needed, max_possible)
        pert_entries = torch.linspace(left, right, m, dtype=torch.float64, device=E.device)
        eig_pert_diag[slices_min[rep]:slices_max[rep]] += pert_entries

    U_no_grad = U.detach()
    P_ev = U_no_grad @ torch.diag(eig_pert_diag) @ U_no_grad.T
    return P_ev


def invert_wrong_signs(U: torch.Tensor, U_noised: torch.Tensor):
    inverted_sign = (U_noised - U).abs().sum(0) > (U_noised + U).abs().sum(0)
    flipper = U.new_ones((U.size(1), ))
    flipper[inverted_sign] = -1
    return U_noised * flipper[None, :]
