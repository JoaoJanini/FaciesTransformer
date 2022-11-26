from transformers import TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer
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
from datasets import load_dataset, load_metric

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
SEQUENCE_LEN = 20
TRAINING_RATIO = 0.95
WIRELINE_LOGS_HEADER = ["GR", "NPHI", "RSHA", "DTC", "RHOB"]
CATEGORICAL_COLUMNS = ["FORMATION", "GROUP"]
LABEL_COLUMN_HEADER = ["FORCE_2020_LITHOFACIES_LITHOLOGY"]

train_dataset = WellsDataset(
    dataset_type="train",
    sequence_len=SEQUENCE_LEN,
    model_type="seq2seq",
    feature_columns=WIRELINE_LOGS_HEADER,
    categorical_features_columns=CATEGORICAL_COLUMNS,
    label_columns=LABEL_COLUMN_HEADER,
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


facies_config = {
    "vocab_size": tgt_vocab_size,
    "max_position_embeddings": 1024,
    "encoder_layers": 5,
    "encoder_ffn_dim": 1024,
    "encoder_attention_heads": 8,
    "decoder_layers": 5,
    "decoder_ffn_dim": 512,
    "cat_features_indexes": train_dataset.categorical_columns_indexes,
    "decoder_attention_heads": 8,
    "encoder_layerdrop": 0.3,
    "decoder_layerdrop": 0.1,
    "activation_function": "relu",
    "d_model": 512,
    "n_input_features": d_channel,
    "n_output_features": d_output,
    "sequence_len": SEQUENCE_LEN,
    "dropout": 0.3,
    "attention_dropout": 0.3,
    "activation_dropout": 0.3,
    "init_std": 0.02,
    "classifier_dropout": 0.3,
    "scale_embedding": False,
    "use_cache": False,
    "num_labels": tgt_vocab_size,
    "pad_token_id": train_dataset.PAD_IDX,
    "bos_token_id": train_dataset.PAD_IDX,
    "eos_token_id": train_dataset.PAD_IDX,
    "is_encoder_decoder": True,
    "decoder_start_token_id": train_dataset.PAD_IDX,
    "forced_eos_token_id": train_dataset.PAD_IDX,
}
model_directory = f"saved_models/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
facies_transformer_config = FaciesConfig(**facies_config)
facies_transformer_config.save_pretrained(
    f"{model_directory}/facies-transformer-config"
)
facies_transformer_config = FaciesConfig.from_pretrained(
    f"{model_directory}/facies-transformer-config"
)

facies_transformer = FaciesForConditionalGeneration(facies_transformer_config)
def compute_metrics_fn(eval_preds):
    metrics = dict()
    accuracy_metric = load_metric("accuracy")
    preds = eval_preds.predictions[:, 1:-1]
    preds = preds.flatten()
    labels = eval_preds.label_ids[:,:-2]
    labels = labels.flatten()
    preds = preds[labels != 0]
    labels = labels[labels != 0]

    metrics.update(accuracy_metric.compute(predictions=preds, references=labels))

    return metrics


test_dataset = WellsDataset(
    dataset_type="test",
    sequence_len=facies_transformer_config.sequence_len,
    model_type="seq2seq",
    feature_columns=WIRELINE_LOGS_HEADER,
    label_columns=LABEL_COLUMN_HEADER,
    scaler=train_dataset.scaler,
    output_len=train_dataset.output_len,
    categories_label_encoders=train_dataset.categories_label_encoders,
)
training_args = Seq2SeqTrainingArguments(
    output_dir=f"{model_directory}/facies-transformer",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    num_train_epochs=10,
    generation_max_length=SEQUENCE_LEN+2,
    generation_num_beams=4,
    predict_with_generate=True
)


trainer = Seq2SeqTrainer(
    model=facies_transformer,
    train_dataset=train_data,
    data_collator=collate_fn,
    eval_dataset=test_dataset,
    args=training_args,
    compute_metrics=compute_metrics_fn,
)
result = trainer.train()

torch.save(
    facies_transformer.state_dict(),
    f=f"{model_directory}/facies-transformer/facies_transformer_state_dict.pt",
)
# Write the model directory to a text file called current_model.txt
with open("current_model.txt", "w") as f:
    f.write(model_directory)

# decoded_labels = torch.empty(0, dtype=torch.long).to(DEVICE)
# for i, batch in enumerate(test_loader):
#     input_ids = batch["input_ids"].to(DEVICE)
#     outputs = facies_transformer.generate(
#         input_ids=input_ids,
#         bos_token_id=test_dataset.PAD_IDX,
#         pad_token_id=test_dataset.PAD_IDX,
#         eos_token_id=test_dataset.PAD_IDX,
#         num_return_sequences=1,
#         num_beams=3,
#         max_new_tokens=facies_transformer_config.sequence_len + 1,
#         temperature=0.8,
#     )

#     decoded_labels = torch.cat((decoded_labels, outputs[:, 1:-1].flatten()))
# print(decoded_labels)
# labels = test_dataset.train_label.flatten().to(DEVICE)
# # Calculate the accuracy of the model
# correct = (decoded_labels == labels).sum().item()
# accuracy = correct / len(labels)
# print(f"Accuracy: {accuracy}")
