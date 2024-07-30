import torch
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('DummyEdge')
class DummyEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_embeddings=1,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):

        # TODO: if there are already 1D probability edge attr, then set those to batch.edge_weights
        #  (needed dor e.g. for GPS with GatedGCN where edge attributes are used and updated, could 
        #   also embed this dummy edge into there directly to avoid this problem)
        assert batch.edge_attr is None

        dummy_attr = batch.edge_index.new_zeros(batch.edge_index.shape[1])
        batch.edge_attr = self.encoder(dummy_attr)
        return batch
