import torchmetrics

import torch
from model.transformer import Seq2SeqTransformer, generate_square_subsequent_mask
from torch.utils.data import DataLoader
from dataset.dataset import WellsDataset
from torch.utils.data import random_split
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

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, start_symbol):
    src = src.to(DEVICE)
    memory = model.encode(src, None)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(SEQUENCE_LEN + 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            DEVICE
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0
        ).type(torch.long)
        if next_word == test_dataset.EOS_IDX:
            break
    return ys[1:-1]


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    # src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    tgt_tokens = greedy_decode(
        model, src_sentence, start_symbol=test_dataset.BOS_IDX
    ).flatten()
    return tgt_tokens


test_loader = DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
)
# run translation function on all data
def translate_all(model: torch.nn.Module, data_loader: DataLoader):
    predicted_sequences = []
    real_sequences = []
    model.eval()

    for i, (src_sequence, tgt_sequence) in enumerate(data_loader):
        src_sequence = src_sequence.to(DEVICE)

        tgt_token = translate(model, src_sequence)

        predicted_sequences = predicted_sequences + tgt_token.tolist()
        real_sequences = real_sequences + tgt_sequence[0].tolist()
        if (i + 1) % 1000 == 0:
            comparison = torch.Tensor(predicted_sequences) == torch.Tensor(
                real_sequences
            )
            print(
                f"Current Accuracy for {i+1} sequences: {sum(comparison.tolist())/len(real_sequences)}"
            )

    return predicted_sequences, real_sequences


predicted_sequences, real_sequences = translate_all(model, test_loader)

comparison = torch.Tensor(predicted_sequences) == torch.Tensor(real_sequences)
print(f"Final Accuracy: {sum(comparison.tolist())/len(real_sequences)}")
