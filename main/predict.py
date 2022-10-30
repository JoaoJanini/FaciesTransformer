"""
@author : Hyunwoong
@when : 2019-12-19
@homepage : https://github.com/gusdnd852
"""

import math
from collections import Counter

import numpy as np

from datasource import *
import torchmetrics
import math
import time
from torch import nn, optim
from torch.optim import Adam
import torch
from transformer.transformer import Transformer
from util.epoch_timer import epoch_time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = Transformer(
    d_model=d_model,
    d_channel=d_channel,
    d_input=d_input,
    dec_voc_size=dec_voc_size,
    max_len=max_len,
    ffn_hidden=ffn_hidden,
    n_head=n_heads,
    n_layers=n_layers,
    drop_prob=0.00,
    device=device,
).to(device)

model.load_state_dict(
    torch.load("/home/joao/code/learning/seq2seq/models/model-0.12528906492516398.pt")
)


def greedy_decode(model, src, max_len=SEQUENCE_LEN, start_symbol=0):
    src = src.to(device)

    memory = model.encode(src)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        memory_mask = (
            torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        )
        if i == 0:
            tgt_mask = make_no_peak_mask(ys.size(0), ys.size(0), device).type(
                torch.bool
            )
        else:
            tgt_mask = None
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0
        ).type(torch.long)
    return ys


def translate(model, src):
    model.eval()
    tgt_tokens = greedy_decode(model, src, max_len=SEQUENCE_LEN, start_symbol=0)
    return tgt_tokens


test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

for i, (src, tgt) in enumerate(test_loader):
    ys = translate(model, src)
    print("ys")
    print(ys)
    print("tgt")
    print(tgt)
