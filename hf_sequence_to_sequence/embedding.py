from torch import Tensor
import torch
import torch.nn as nn
import math

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, maxlen: int = 6400):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


self.num_categories = len(categories)
self.num_unique_categories = sum(categories)

# create category embeddings table

self.num_special_tokens = num_special_tokens
total_tokens = self.num_unique_categories + num_special_tokens

# for automatically offsetting unique category ids to the correct position in the categories embedding table

categories_offset = F.pad(
    torch.tensor(list(categories)), (1, 0), value=num_special_tokens
)
categories_offset = categories_offset.cumsum(dim=-1)[:-1]
self.register_buffer("categories_offset", categories_offset)

# categorical embedding

self.categorical_embeds = nn.Embedding(total_tokens, dim)
