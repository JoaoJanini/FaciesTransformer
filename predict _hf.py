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
import numpy as np 
import utils
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
run_path = "/home/joao/code/tcc/seq2seq/saved_models/2022-11-23_02-14-14"
config_path = "/home/joao/code/tcc/seq2seq/saved_models/2022-11-23_02-14-14/facies-transformer-config"
model_path = "/home/joao/code/tcc/seq2seq/saved_models/2022-11-23_02-14-14/facies-transformer/facies_transformer_state_dict.pt"

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for (src_sample, tgt_sample) in batch:
        tgt_batch.append(tgt_sample)
        src_batch.append(src_sample)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)

    model_input = {"input_ids": src_batch, "labels": tgt_batch}
    return model_input


facies_transformer_config = FaciesConfig.from_pretrained(f"{config_path}")

facies_transformer = FaciesForConditionalGeneration(facies_transformer_config).to(
    DEVICE
)
facies_transformer.load_state_dict(torch.load(model_path))
BATCH_SIZE = 128
TRAINING_RATIO = 0.90
WIRELINE_LOGS_HEADER = ["GR", "NPHI", "RSHA", "DTC", "RHOB", "SP"]
LABEL_COLUMN_HEADER = ["FORCE_2020_LITHOFACIES_LITHOLOGY"]
train_dataset = WellsDataset(
    dataset_type="train",
    sequence_len=facies_transformer_config.sequence_len,
    model_type="seq2seq",
    feature_columns=WIRELINE_LOGS_HEADER,
    label_columns=LABEL_COLUMN_HEADER,
)
test_dataset = WellsDataset(
    dataset_type="test",
    sequence_len=facies_transformer_config.sequence_len,
    model_type="seq2seq",
    feature_columns=WIRELINE_LOGS_HEADER,
    label_columns=LABEL_COLUMN_HEADER,
    scaler=train_dataset.scaler,
    output_len=train_dataset.output_len,
)


test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# Loop for generating the output of a sequence for all the data in the test dataloader using model.generate

decoded_labels = torch.empty(0, dtype=torch.long).to(DEVICE)
for i, batch in enumerate(test_loader):
    input_ids = batch["input_ids"].to(DEVICE)
    outputs = facies_transformer.generate(
        input_ids=input_ids,
        bos_token_id=test_dataset.PAD_IDX,
        pad_token_id=test_dataset.PAD_IDX,
        eos_token_id=test_dataset.PAD_IDX,
        num_return_sequences=1,
        num_beams=7,
        max_new_tokens=facies_transformer_config.sequence_len + 1,
    )

    decoded_labels = torch.cat((decoded_labels, outputs[:, 1:-1].flatten()))

labels = test_dataset.train_label.flatten().to(DEVICE)
decoded_labels = decoded_labels[labels != 0]
labels = labels[labels != 0]


# Calculate the accuracy of the model
correct = (decoded_labels == labels).sum().item()
accuracy = correct / len(labels)
print(f"Accuracy: {accuracy}")
# Save to file

wells_depth = test_dataset.df_position
index_to_lith_code = {v: k for k, v in utils.get_lithology_numbers().items()}
y_pred_decoded = np.array([*map(index_to_lith_code.get, decoded_labels.cpu().numpy())])
y_true_decoded = np.array([*map(index_to_lith_code.get, labels.cpu().numpy())])

wells_depth["FORCE_2020_LITHOFACIES_LITHOLOGY"] = y_pred_decoded
wells_depth.to_csv(f"{run_path}/facies_prediction.csv")
wells_depth["FORCE_2020_LITHOFACIES_LITHOLOGY"] = y_true_decoded
wells_depth.to_csv(f"{run_path}/facies.csv")
# Increase print limit for torch tensor
torch.set_printoptions(threshold=10000)
