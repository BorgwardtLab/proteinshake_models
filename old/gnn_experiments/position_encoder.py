import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper.
    """
    def __init__(self, embed_dim=32, init_len=3000, temperature=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.init_len = init_len
        self.temperature = temperature

        self.register_buffer("_float_tensor", torch.FloatTensor(1))

        self.weights = PositionEmbeddingSine.get_embedding(
            init_len, embed_dim, temperature
        )

    @staticmethod
    def get_embedding(max_len, embed_dim, temperature):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embed_dim // 2
        emb = math.log(temperature) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_len, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            max_len, -1
        )
        if embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(max_len, 1)], dim=1)
        return emb

    def forward(self, data):
        max_len = data.ptr.diff().max().item() + 1
        if self.weights is None or max_len > self.weights.shape[0]:
            # recompute/expand embeddings if needed
            self.weights = PositionalEmbeddingSine1D.get_embedding(
                max_len, self.embed_dim, self.temperature
            )
        self.weights = self.weights.to(self._float_tensor)
        return self.weights.index_select(0, data.residue_idx)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, embed_dim=32, max_len=3000):
        super().__init__()
        self.embed = nn.Embedding(max_len, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embed.weight)

    def forward(self, data):
        pos = self.embed(data.residue_idx)
        return pos


def build_position_encoding(embed_dim, pe='learned'):
    if pe is None:
        return pe
    if pe == 'sine':
        position_embedding = PositionEmbeddingSine(embed_dim)
    elif pe == 'learned':
        position_embedding = PositionEmbeddingLearned(embed_dim)
    else:
        raise ValueError(f"not supported {pe}")

    return position_embedding
