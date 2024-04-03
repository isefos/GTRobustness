import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.graphgym.config import cfg


def get_lap_decomp_stats(data, lap_norm_type, max_freqs, eigvec_norm, requires_grad=False):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.
    """
    N = data.num_nodes

    L_sp = get_laplacian(data.edge_index, data.edge_attr, normalization=lap_norm_type)
    L = to_dense_adj(L_sp[0], None, L_sp[1])[0].to(dtype=torch.float64)
    
    E, U = torch.linalg.eigh(L)

    # for attack, when we want to compute the gradient of the eigenvalues and eigenvectors
    if requires_grad:
        perturbation = get_pert_diff_eigenvects(E, U)
        if perturbation is not None:
            _, U_pert = torch.linalg.eigh(L + perturbation)

            if cfg.posenc_WLapPE.eigen.correct_pert_eigvec_sign:
                U_pert = invert_wrong_signs(U, U_pert)

            if cfg.posenc_WLapPE.eigen.straight_through_estimator:
                U = U.detach() + U_pert - U_pert.detach()
            else:
                U = U_pert

    # now that the solver is finished, transform back to float32
    E = E.to(dtype=torch.float32)
    U = U.to(dtype=torch.float32)

    # set first eigenvalue to zero (is non-zero because of numerical error)
    evals = torch.zeros_like(E)
    evals[1:] = E[1:]
    evects = U

    # Keep up to the maximum desired number of frequencies, output is already sorted
    evals = evals[:max_freqs]
    evects = evects[:, :max_freqs]

    # Normalize and pad eigen vectors.
    if eigvec_norm != "L2":
        # solver automatically normalizes eigenvectors to L2 norm.
        evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


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


def get_ev_pert(E, E_diff, U, eta):
    edges = torch.diff(torch.cat((torch.tensor([False]), E_diff < eta, torch.tensor([False]))).to(dtype=torch.int64))
    slices_min = torch.nonzero(edges == 1).flatten()
    slices_max = torch.nonzero(edges == -1).flatten() + 1

    # space to left of smallest and to the right of larges are zero
    # (so we ensure that the total range of eigenvalues stays the same)
    spaces = torch.cat((torch.tensor([0]), E_diff, torch.tensor([0])))
    # where two repeated eigenvalues are next to each other, each gets half the space to split and expand
    spaces[1:-1][torch.diff(edges) == 2] /= 2
    space_left = spaces[slices_min]
    space_right = spaces[slices_max]

    multiplicities = slices_max - slices_min

    eig_pert_diag = torch.zeros(E.size(0), dtype=torch.float64)
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
        pert_entries = torch.linspace(left, right, m, dtype=torch.float64)
        eig_pert_diag[slices_min[rep]:slices_max[rep]] = pert_entries

    U_no_grad = U.detach()
    P_ev = U_no_grad @ torch.diag(eig_pert_diag) @ U_no_grad.T
    return P_ev


def invert_wrong_signs(U, U_noised):
    inverted_sign = (U_noised - U).abs().sum(0) > (U_noised + U).abs().sum(0)
    flipper = torch.ones(U.size(0))
    flipper[inverted_sign] = -1
    return U_noised * flipper[None, :]
