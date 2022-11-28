import torch
import torch.nn as nn
import torch.optim as opt

torch.set_printoptions(linewidth=120)
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    TrainingArguments,
    Trainer,
    logging,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from hf_sequence_to_sequence.model import FaciesForConditionalGeneration
from hf_sequence_to_sequence.configuration import FaciesConfig
import torchmetrics
import math
import time
from torch import nn, optim
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader
from dataset.dataset import WellsDataset
from torch.utils.data import random_split
from typing import List
from datetime import datetime
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.hyperopt import HyperOptSearch
from ray import tune

# define function to compue metrics
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
SEQUENCE_LEN = 15
TRAINING_RATIO = 0.05
WIRELINE_LOGS_HEADER = ["GR", "NPHI", "RSHA", "DTC", "RHOB", "SP"]
LABEL_COLUMN_HEADER = ["FORCE_2020_LITHOFACIES_LITHOLOGY"]
model_directory = f"saved_models/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

train_dataset = WellsDataset(
    dataset_type="train",
    sequence_len=SEQUENCE_LEN,
    model_type="seq2seq",
    feature_columns=WIRELINE_LOGS_HEADER,
    label_columns=LABEL_COLUMN_HEADER,
)

DATA_LEN = train_dataset.train_len
d_input = train_dataset.input_len
d_output = train_dataset.output_len
d_channel = train_dataset.channel_len
tgt_vocab_size = train_dataset.output_len + len(train_dataset.special_symbols)


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        tgt_batch.append(tgt_sample)
        src_batch.append(src_sample)

    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)

    model_input = {"input_ids": src_batch.to(DEVICE), "labels": tgt_batch.to(DEVICE)}
    return model_input


# function to collate data samples into batch tesors
facies_config = {
    "vocab_size": tgt_vocab_size,
    "max_position_embeddings": 1024,
    "encoder_layers": 6,
    "encoder_ffn_dim": 1024,
    "encoder_attention_heads": 8,
    "decoder_layers": 4,
    "decoder_ffn_dim": 1024,
    "decoder_attention_heads": 8,
    "encoder_layerdrop": 0.0,
    "decoder_layerdrop": 0.0,
    "activation_function": "relu",
    "d_model": 512,
    "n_input_features": d_channel,
    "n_output_features": d_output,
    "sequence_len": SEQUENCE_LEN,
    "dropout": 0.2,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "init_std": 0.02,
    "classifier_dropout": 0.0,
    "scale_embedding": False,
    "use_cache": False,
    "num_labels": tgt_vocab_size,
    "pad_token_id": train_dataset.PAD_IDX,
    "bos_token_id": train_dataset.PAD_IDX,
    "eos_token_id": train_dataset.PAD_IDX,
    "is_encoder_decoder": True,
    "decoder_start_token_id": train_dataset.PAD_IDX,
    "forced_eos_token_id": train_dataset.PAD_IDX,
    "return_dict": False,
}


facies_transformer_config = FaciesConfig(**facies_config)
facies_transformer_config.save_pretrained(
    f"{model_directory}/facies-transformer-config"
)
facies_transformer_config = FaciesConfig.from_pretrained(
    f"{model_directory}/facies-transformer-config"
)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

model = FaciesForConditionalGeneration(facies_transformer_config).to(DEVICE)
model_input = next(iter(train_loader))
input_ids = model_input["input_ids"]
labels = model_input["labels"]

tb = SummaryWriter()
tb.add_graph(model, (input_ids, labels), use_strict_trace=False)
tb.close()
