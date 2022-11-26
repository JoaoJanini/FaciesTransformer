from transformers import TrainingArguments, Trainer, logging, Seq2SeqTrainingArguments, Seq2SeqTrainer
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
from utils import collate_fn
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.hyperopt import HyperOptSearch
from ray import tune
# define function to compute metrics
import numpy as np


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
SEQUENCE_LEN = 15
TRAINING_RATIO = 0.95
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
test_dataset = WellsDataset(
    dataset_type="test",
    sequence_len=SEQUENCE_LEN,
    model_type="seq2seq",
    feature_columns=WIRELINE_LOGS_HEADER,
    label_columns=LABEL_COLUMN_HEADER,
    scaler=train_dataset.scaler,
    output_len=train_dataset.output_len,
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
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
}
facies_transformer_config = FaciesConfig(**facies_config)
facies_transformer_config.save_pretrained(
    f"{model_directory}/facies-transformer-config"
)
facies_transformer_config = FaciesConfig.from_pretrained(
    f"{model_directory}/facies-transformer-config"
)


# function to collate data samples into batch tesors
def ray_hp_space(trial):
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([16]),
        "weight_decay": tune.uniform(0.0, 0.3),
        "num_train_epochs": tune.choice([2]),
    }


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


def model_init(trial):

    return FaciesForConditionalGeneration(facies_transformer_config)


training_args = Seq2SeqTrainingArguments(
    output_dir=f"{model_directory}/facies-transformer",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    disable_tqdm=True,
    evaluation_strategy="steps",
    num_train_epochs=10,
    eval_steps=500,
    generation_max_length=SEQUENCE_LEN+2,
    generation_num_beams=4,
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=None,
    train_dataset=train_data,
    data_collator=collate_fn,
    eval_dataset=validation_data,
    args=training_args,
    model_init = model_init,
    compute_metrics=compute_metrics_fn
)

best_model = trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    n_trials=10,
    search_alg=HyperOptSearch(metric="objective", mode="max"),
    hp_space=ray_hp_space,
    local_dir=f"{model_directory}/ray_results",
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
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

torch.save(
    best_model.state_dict(),
    f=f"{model_directory}/facies-transformer/facies_transformer_state_dict.pt",
)
