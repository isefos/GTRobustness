import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from graphgps.transform.lap_eig import get_lap_decomp_stats


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
        attack_mode = batch.get("attack_mode", False)
        if attack_mode or batch.get("EigVals") is None:
            # for attack
            assert batch.num_graphs == 1, "On the fly preprocessing only works for single graphs"
            pe_cfg = cfg.posenc_WLapPE
            lap_norm_type = pe_cfg.eigen.laplacian_norm.lower()
            if lap_norm_type == 'none':
                lap_norm_type = None
            max_freqs = pe_cfg.eigen.max_freqs
            eigvec_norm = pe_cfg.eigen.eigvec_norm
            if attack_mode and pe_cfg.eigen.use_gradient:
                batch.EigVals, batch.EigVecs = get_lap_decomp_stats(
                    batch,
                    lap_norm_type,
                    max_freqs=max_freqs,
                    eigvec_norm=eigvec_norm,
                    requires_grad=True,
                )
            else:
                with torch.no_grad():
                    batch.EigVals, batch.EigVecs = get_lap_decomp_stats(
                        batch,
                        lap_norm_type,
                        max_freqs=max_freqs,
                        eigvec_norm=eigvec_norm,
                    )

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
