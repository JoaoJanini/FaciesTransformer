"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
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


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


model = Transformer(
    d_model=d_model,
    d_channel=d_channel,
    d_input=d_input,
    dec_voc_size=dec_voc_size,
    max_len=max_len,
    ffn_hidden=ffn_hidden,
    n_head=n_heads,
    n_layers=n_layers,
    drop_prob=drop_prob,
    device=device,
).to(device)

print(f"The model has {count_parameters(model):,} trainable parameters")
model.apply(initialize_weights)
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
    for i, (src, trg) in enumerate(iterator):
        optimizer.zero_grad()
        src = src.to(device)
        trg = trg.to(device)
        # Add <sos> token to the beginning of the target sentence
        trg = torch.cat(
            ((torch.zeros((trg.shape[0], 1)).to(device)), trg), dim=1
        ).long()
        trg_mask = make_no_peak_mask(SEQUENCE_LEN, device)
        output = model(src, trg[:, :-1], trg_mask)

        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print("step :", round((i / len(iterator)) * 100, 2), "% , loss :", loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []

    with torch.no_grad():
        accuracy = torchmetrics.Accuracy().to(device)
        for i, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)
            trg = torch.cat(
                ((torch.zeros((trg.shape[0], 1)).to(device)), trg), dim=1
            ).long()
            trg_mask = make_no_peak_mask(SEQUENCE_LEN, device)

            output = model(src, trg[:, :-1], trg_mask)
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()
            _, label_index = torch.max(output_reshape.data, dim=-1)
            acc = accuracy(label_index, trg)
        acc = accuracy.compute()

        accuracy.reset()
    return epoch_loss / len(iterator), acc


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
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

        f = open(model_path + "-bleu.txt", "w")
        f.write(str(bleus))
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
