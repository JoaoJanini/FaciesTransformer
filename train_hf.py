from transformers import TrainingArguments, Trainer, logging
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
from utils import epoch_time
from conf import DEVICE
from conf import (
    batch_size,
    max_len,
    d_model,
    n_layers,
    n_heads,
    ffn_hidden,
    drop_prob,
    model_path,
)
from conf import (
    init_lr,
    factor,
    adam_eps,
    patience,
    warmup,
    epoch,
    clip,
    weight_decay,
    inf,
)
from conf import WIRELINE_LOGS_HEADER, LABEL_COLUMN_HEADER, SEQUENCE_LEN, TRAINING_RATIO
from typing import List

train_dataset = WellsDataset(
    dataset_type="train",
    sequence_len=SEQUENCE_LEN,
    model_type="seq2seq",
    feature_columns=WIRELINE_LOGS_HEADER,
    label_columns=LABEL_COLUMN_HEADER,
)
test_dataset = WellsDataset(
    dataset_type="test",
    sequence_len=SEQUENCE_LEN,
    model_type="seq2seq",
    feature_columns=WIRELINE_LOGS_HEADER,
    label_columns=LABEL_COLUMN_HEADER,
    scaler=train_dataset.scaler,
    output_len=train_dataset.output_len,
)


DATA_LEN = train_dataset.train_len
d_input = train_dataset.input_len
d_output = train_dataset.output_len
d_channel = train_dataset.channel_len
tgt_vocab_size = train_dataset.output_len + len(train_dataset.special_symbols)
TRAIN_DATA_LEN = int(DATA_LEN * TRAINING_RATIO)


train_data, validation_data = random_split(
    train_dataset, lengths=[TRAIN_DATA_LEN, DATA_LEN - TRAIN_DATA_LEN]
)


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices

# src and tgt language text transforms to convert raw strings into tensors indices

# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        tgt_batch.append(tgt_sample)
        src_batch.append(src_sample)

    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)

    model_input = {"input_ids": src_batch.to(DEVICE), "labels": tgt_batch.to(DEVICE)}
    return model_input

print("data structure: [lines, timesteps, features]")
print(f"train data size: [{DATA_LEN, d_input, d_channel}]")
print(f"Number of classes: {d_output}")

facies_transformer_config = FaciesConfig(eos_token_id=EOS_IDX, pad_token_id=PAD_IDX, bos_token_id=BOS_IDX, vocab_size=tgt_vocab_size, d_input=d_input)
facies_transformer_config.save_pretrained("facies-transformer-config")
facies_transformer_config = FaciesConfig.from_pretrained("facies-transformer-config")


facies_transformer = FaciesForConditionalGeneration(facies_transformer_config)


trainer = Trainer(model=facies_transformer, train_dataset=train_data, data_collator=collate_fn )
result = trainer.train()
print_summary(result)