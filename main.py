"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
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
            torch.tensor(token_ids),
            torch.tensor([train_dataset.EOS_IDX]),
        )
    )


# src and tgt language text transforms to convert raw strings into tensors indices
transforms = {}
transforms["tgt"] = sequential_transforms(
    tgt_transform  # Add BOS/EOS and create tensor
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


print(f"The model has {count_parameters(model):,} trainable parameters")
optimizer = Adam(
    params=model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, verbose=True, factor=factor, patience=patience
)

criterion = nn.CrossEntropyLoss()


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, tgt) in enumerate(iterator):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt = tgt.transpose_(0, 1)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, PAD_IDX=train_dataset.PAD_IDX
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print("step :", round((i / len(iterator)) * 100, 2), "% , loss :", loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        accuracy = torchmetrics.Accuracy().to(DEVICE)
        for i, (src, tgt) in enumerate(iterator):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            tgt = tgt.transpose_(0, 1)
            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input, PAD_IDX=train_dataset.PAD_IDX
            )

            logits = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )

            tgt_out = tgt[1:-1, :]
            logits = logits[:-1, :, :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            epoch_loss += loss.item()

            _, label_index = torch.max(
                logits.reshape(-1, logits.shape[-1]).data, dim=-1
            )
            acc = accuracy(label_index, tgt_out.reshape(-1))
        acc = accuracy.compute()

        accuracy.reset()
    return epoch_loss / len(iterator), acc


def run(total_epoch, best_loss):
    train_losses, test_losses = [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
        valid_loss, validation_accuracy = evaluate(model, validation_loader, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_path + "-{0}.pt".format(valid_loss))

        f = open(model_path + "-train_loss.txt", "w")
        f.write(str(train_losses))
        f.close()

        f = open(model_path + "-test_loss.txt", "w")
        f.write(str(test_losses))
        f.close()

        print(f"Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
        )
        print(f"\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}")
        print(f"\tVal Accuracy : {validation_accuracy}")
        print(f"\tTest Loss: {test_loss:.3f} |  Val PPL: {math.exp(test_loss):7.3f}")
        print(f"\tTest Accuracy : {test_accuracy}")


run(total_epoch=epoch, best_loss=inf)
