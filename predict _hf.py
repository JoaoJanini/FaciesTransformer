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
from typing import List


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 640
SEQUENCE_LEN = 30
TRAINING_RATIO = 0.90
WIRELINE_LOGS_HEADER = ["DEPTH_MD", "GR", "NPHI"]
LABEL_COLUMN_HEADER = ["FORCE_2020_LITHOFACIES_LITHOLOGY"]

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

    model_input = {"input_ids": src_batch, "labels": tgt_batch}
    return model_input


print("data structure: [lines, timesteps, features]")
print(f"train data size: [{DATA_LEN, d_input, d_channel}]")
print(f"Number of classes: {d_output}")


test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

facies_transformer_config = FaciesConfig.from_pretrained("/home/joao/code/tcc/seq2seq/saved_models/checkpoint-500")

facies_transformer = FaciesForConditionalGeneration(facies_transformer_config).to(DEVICE)

# Loop for generating the output of a sequence for all the data in the test dataloader using model.generate
for i, batch in enumerate(test_loader):
    input_ids = batch["input_ids"].to(DEVICE)
    outputs = facies_transformer.generate(
        input_ids=input_ids,
        bos_token_id=2,
        bad_words_ids=[[2,1,0,3]],
        num_beams=2,
        num_return_sequences=1,
        max_new_tokens=SEQUENCE_LEN+1
    )