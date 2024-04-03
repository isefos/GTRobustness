import torch
from torch_geometric.graphgym.register import register_edge_encoder
from torch_geometric.utils import coalesce
from graphgps.utils import negate_edge_index
from torch_geometric.graphgym.config import cfg


@register_edge_encoder("WeightedSANDummyEdge")
class WeightedSANDummyEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        if cfg.gt.wsan_add_dummy_edges:
            self.real_encoder = torch.nn.Embedding(num_embeddings=1, embedding_dim=emb_dim)
            if cfg.gt.full_graph:
                self.fake_encoder = torch.nn.Embedding(num_embeddings=1, embedding_dim=emb_dim)

    def forward(self, batch):
        if cfg.gt.wsan_add_dummy_edges:
            batch.edge_attr_dummy = self.real_encoder(batch.edge_index.new_zeros((1, )))
            if cfg.gt.full_graph:
                batch.edge_attr_dummy_fake = self.fake_encoder(batch.edge_index.new_zeros((1, )))

        # same for "fake" edges
        fake_edge_index = negate_edge_index(batch.edge_index, batch.batch)

        if batch.get("attack_mode", False):
            # for attack with weighted edges, edge probability in batch.edge_attr
            batch.edge_logprob = batch.edge_attr.log()
            # find the edges with weight less than one, add to fake_edge_index as partially fake edges
            maybe_fake_mask = batch.edge_attr < 1
            logprob_fake = (-batch.edge_attr[maybe_fake_mask]).log1p()

            cat_index = torch.cat([fake_edge_index, batch.edge_index[:, maybe_fake_mask]], dim=1)
            cat_logprob = torch.cat([fake_edge_index.new_zeros((fake_edge_index.size(1), )), logprob_fake], dim=0)
            fake_edge_index, logprob_fake = coalesce(
                cat_index,
                cat_logprob,
                batch.num_nodes,
                reduce="sum",
            )
            batch.edge_logprob_fake = logprob_fake

        batch.edge_index_fake = fake_edge_index
        return batch
