from transformers import TrainingArguments, Trainer, logging
from transformers import AutoModelForSequenceClassification
import torchmetrics
import math
import time
from torch import nn, optim
from torch.optim import Adam
import torch
from model.transformer import Seq2SeqTransformer, create_mask
from torch.utils.data import DataLoader
from dataset.dataset import WellsDataset
from torch.utils.data import random_split
from utils import epoch_time
import numpy as np
from datasets import Dataset



ds = Dataset.from_dict(dummy_data)
ds.set_format("pt")
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
seq_len, dataset_size = 512, 512
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size)),
}

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
def tgt_transform(token_ids: List[int]):
    return torch.cat(
        (
            torch.tensor([train_dataset.BOS_IDX]),
            torch.tensor(token_ids),
            torch.tensor([train_dataset.EOS_IDX]),
        )
    )


def src_transform(token_ids: List[int]):
    return torch.cat(
        (
            torch.ones(1, d_channel) * train_dataset.PAD_IDX,
            torch.tensor(token_ids),
            torch.ones(1, d_channel) * train_dataset.PAD_IDX,
        )
    )


# src and tgt language text transforms to convert raw strings into tensors indices
transforms = {}
transforms["tgt"] = sequential_transforms(
    tgt_transform  # Add BOS/EOS and create tensor
)
transforms["src"] = sequential_transforms(
    src_transform  # Add BOS/EOS and create tensor
)

# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        tgt_batch.append(transforms["tgt"](tgt_sample))
        src_batch.append(src_sample)

    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    return src_batch, tgt_batch


test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
validation_loader = DataLoader(
    dataset=validation_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)

print("data structure: [lines, timesteps, features]")
print(f"train data size: [{DATA_LEN, d_input, d_channel}]")
print(f"Number of classes: {d_output}")



model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to("cuda")
print_gpu_utilization()


logging.set_verbosity_error()


default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}

training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)