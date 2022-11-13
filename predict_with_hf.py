import torchmetrics

import torch
from model.faciestransformer import Seq2SeqTransformer, generate_square_subsequent_mask
from torch.utils.data import DataLoader
from dataset.dataset import WellsDataset
from torch.utils.data import random_split
from model.huggingface.custom_hf_transformer import (
    SeqtoSeqForFaciesClassification,
    Seq2SeqConfig,
)
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
        tgt_batch.append(tgt_sample)
        src_batch.append(src_sample)

    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    return src_batch, tgt_batch


test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

print("data structure: [lines, timesteps, features]")
print(f"train data size: [{DATA_LEN, d_input, d_channel}]")
print(f"Number of classes: {d_output}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = Seq2SeqTransformer(
    num_encoder_layers=n_layers,
    d_input=d_input,
    num_decoder_layers=n_layers,
    d_model=d_model,
    nhead=n_heads,
    tgt_vocab_size=tgt_vocab_size,
    dim_feedforward=ffn_hidden,
    dropout=drop_prob,
    device=DEVICE,
).to(DEVICE)


model.load_state_dict(
    torch.load("/home/joao/code/tcc/seq2seq/saved_models/model_-0.35968838930130004.pt")
)

NUM_EPOCHS = 18


# Code for loading trained model
seq_to_seq_facies_config = Seq2SeqConfig(
    num_encoder_layers=n_layers,
    d_input=d_input,
    num_decoder_layers=n_layers,
    d_model=d_model,
    nhead=n_heads,
    tgt_vocab_size=tgt_vocab_size,
    dim_feedforward=ffn_hidden,
    dropout=drop_prob,
)

seq_to_seq_facies_config.save_pretrained("custom_seq_to_seq_facies_config")
seq_to_seq_facies_config.from_pretrained("custom_seq_to_seq_facies_config")
seq_to_seq_facies = SeqtoSeqForFaciesClassification(seq_to_seq_facies_config)
seq_to_seq_facies.model.load_state_dict(model.state_dict())


print(" oi")
