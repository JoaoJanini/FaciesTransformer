from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
from model.embedding import TokenEmbedding, PositionalEncoding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        d_input: int,
        num_decoder_layers: int,
        d_model: int,
        nhead: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        device=DEVICE,
    ):
        super(Seq2SeqTransformer, self).__init__()

        self.transformer = Transformer(
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.embedding_input = torch.nn.Linear(d_input, d_model)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        channel_encoding = self.embedding_input(src.transpose(-1, -2))
        channel_encoding = channel_encoding.transpose(0, 1)
        outs = self.transformer(
            channel_encoding,
            tgt_emb,
            None,
            tgt_mask,
            None,
            None,
            None,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        channel_encoding = self.embedding_input(src.transpose(-1, -2))
        channel_encoding = channel_encoding.transpose(0, 1)
        return self.transformer.encoder(
           channel_encoding, src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, PAD_IDX=None):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
