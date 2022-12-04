import os
from datetime import datetime

base_path = "/home/joao/code/tcc/seq2seq/data"

models = {
    "seq2seq": {
        "folder_path": "seq2seq",
    },
    "xgb": {"folder_path": "xgb"},
}

model_choice = "xgb"
folder_path = models[model_choice]["folder_path"]
last_model = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

for directory in ["trained_models", "predictions", "runs"]:
    models[model_choice][
        directory
    ] = f"{base_path}/{directory}/{folder_path}/{last_model}"
    try:
        os.mkdir(f"{base_path}/{directory}/{folder_path}")
    except FileExistsError:
        pass

trained_models_path = models[model_choice]["trained_models"]
predictions_path = models[model_choice]["predictions"]
runs_path = models[model_choice]["runs"]

if __name__ == "__main__":
    if model_choice == "seq2seq":

        from transformers import (
            TrainingArguments,
            Trainer,
            Seq2SeqTrainingArguments,
            Seq2SeqTrainer,
        )
        from model import FaciesForConditionalGeneration
        from configuration import FaciesConfig
        import torch
        from dataset.dataset import WellsDataset
        from torch.utils.data import random_split
        from datetime import datetime
        from datasets import load_metric

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        BATCH_SIZE = 10
        SEQUENCE_LEN = 5
        TRAINING_RATIO = 0.05
        WIRELINE_LOGS_HEADER = ["GR", "NPHI"]
        LABEL_COLUMN_HEADER = ["FORCE_2020_LITHOFACIES_LITHOLOGY"]
        train_dataset = WellsDataset(
            dataset_type="train",
            sequence_len=SEQUENCE_LEN,
            model_type="seq2seq",
            feature_columns=WIRELINE_LOGS_HEADER,
            categorical_features_columns=[],
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
            categories_label_encoders=train_dataset.categories_label_encoders,
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

        def create_missing_mask(src_batch):
            missing_mask = torch.isnan(src_batch)
            return missing_mask

        # function to collate data samples into batch tesors
        def collate_fn(batch):
            src_batch, tgt_batch, missing_mask = [], [], []
            for src_sample, tgt_sample in batch:
                tgt_batch.append(tgt_sample)
                src_batch.append(src_sample)
                missing_mask.append(create_missing_mask(src_sample))

            src_batch = torch.stack(src_batch)
            tgt_batch = torch.stack(tgt_batch)
            attn_mask_batch = torch.stack(missing_mask)
            model_input = {
                "input_ids": src_batch,
                "labels": tgt_batch,
                "attention_mask": attn_mask_batch,
            }
            return model_input

        facies_config = {
            "vocab_size": tgt_vocab_size,
            "max_position_embeddings": 1024,
            "encoder_layers": 6,
            "encoder_ffn_dim": 512,
            "encoder_attention_heads": 8,
            "decoder_layers": 6,
            "decoder_ffn_dim": 512,
            "decoder_attention_heads": 8,
            "encoder_layerdrop": 0.1,
            "decoder_layerdrop": 0.1,
            "activation_function": "relu",
            "d_model": 512,
            "n_input_features": d_input,
            "n_output_features": d_output,
            "sequence_len": SEQUENCE_LEN,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.1,
            "init_std": 0.02,
            "classifier_dropout": 0.1,
            "scale_embedding": False,
            "use_cache": False,
            "num_labels": tgt_vocab_size,
            "pad_token_id": train_dataset.PAD_IDX,
            "bos_token_id": train_dataset.PAD_IDX,
            "eos_token_id": train_dataset.PAD_IDX,
            "is_encoder_decoder": True,
            "decoder_start_token_id": train_dataset.PAD_IDX,
            "forced_eos_token_id": train_dataset.PAD_IDX,
            "return_dict": False,
        }
        os.mkdir(path=f"{trained_models_path}")
        facies_transformer_config = FaciesConfig(**facies_config)
        facies_transformer_config.save_pretrained(f"{trained_models_path}/config")
        facies_transformer_config = FaciesConfig.from_pretrained(
            f"{trained_models_path}/config"
        )

        facies_transformer = FaciesForConditionalGeneration(facies_transformer_config)

        def compute_metrics_fn(eval_preds):
            metrics = dict()
            accuracy_metric = load_metric("accuracy")
            preds = eval_preds.predictions[:, 1:-1]
            preds = preds.flatten()
            labels = eval_preds.label_ids[:, :-2]
            labels = labels.flatten()
            preds = preds[labels != 0]
            labels = labels[labels != 0]

            metrics.update(
                accuracy_metric.compute(predictions=preds, references=labels)
            )

            return metrics

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"{runs_path}",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=256,
            evaluation_strategy="steps",
            num_train_epochs=4,
            generation_max_length=SEQUENCE_LEN + 2,
            generation_num_beams=7,
            predict_with_generate=True,
            learning_rate=3.40752e-05,
            weight_decay=0.0624145,
        )

        trainer = Seq2SeqTrainer(
            model=facies_transformer,
            train_dataset=train_dataset,
            data_collator=collate_fn,
            eval_dataset=test_dataset,
            args=training_args,
            compute_metrics=compute_metrics_fn,
        )
        try:
            best_model = trainer.train()
        except:
            torch.save(
                facies_transformer.state_dict(),
                f=f"{trained_models_path}/model.pt",
            )
        # Write the model directory to a text file called current_model.txt
    elif model_choice == "xgb":
        from xgboost import XGBClassifier
        from dataset.dataset import WellsDataset
        from datetime import datetime

        seed = 7
        WIRELINE_LOGS_HEADER = ["GR", "NPHI", "RSHA", "DTC", "RHOB", "SP"]
        LABEL_COLUMN_HEADER = ["FORCE_2020_LITHOFACIES_LITHOLOGY"]
        train_dataset = WellsDataset(
            dataset_type="train",
            model_type="label2label",
            feature_columns=WIRELINE_LOGS_HEADER,
            label_columns=LABEL_COLUMN_HEADER,
        )

        # fit model no training data
        model = XGBClassifier(
            n_estimators=100,
            max_depth=10,
            booster="gbtree",
            objective="multi:softprob",
            learning_rate=0.1,
            random_state=0,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            tree_method="gpu_hist",
            verbose=2020,
            reg_lambda=1500,
        )
        model.fit(train_dataset.X, train_dataset.y)
        # Increase print limit for torch tensor
        os.mkdir(path=f"{trained_models_path}")
        model.save_model(f"{trained_models_path}/model.json")
