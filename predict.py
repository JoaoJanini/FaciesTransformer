import os

model_choice = "seq2label"

base_path = "/home/joao/code/tcc/seq2seq/data"
models = {
    "seq2seq": {
        "folder_path": "seq2seq",
    },
    "xgb": {"folder_path": "xgb"},
    "seq2label": {"folder_path": "seq2label"},
}


def get_last_model(models_directories):
    last_model = sorted(models_directories)[-1]
    for directory in ["trained_models", "predictions", "runs"]:
        models[model_choice][
            directory
        ] = f"{base_path}/{directory}/{folder_path}/{last_model}"
        os.makedirs(f"{base_path}/{directory}/{folder_path}", exist_ok=True)

    trained_models_path = models[model_choice]["trained_models"]
    predictions_path = models[model_choice]["predictions"]
    runs_path = models[model_choice]["runs"]
    model_path = None
    if model_choice == "xgb":
        return trained_models_path, model_path, predictions_path, runs_path
    try:
        if f"{trained_models_path}/model.pt" not in os.listdir(trained_models_path):
            model_path = os.path.join(
                runs_path,
                list(
                    filter(
                        lambda path: "checkpoint" in path, sorted(os.listdir(runs_path))
                    )
                )[-1],
                "pytorch_model.bin",
            )
        else:
            model_path = f"{trained_models_path}/model.pt"
    except IndexError:
        trained_models_path, model_path, predictions_path, runs_path = get_last_model(sorted(models_directories)[:-1])

    return trained_models_path, model_path, predictions_path, runs_path


folder_path = models[model_choice]["folder_path"]
trained_models_path, model_path, predictions_path, runs_paths = get_last_model(os.listdir(f"{base_path}/trained_models/{folder_path}"))
model_path = f"/home/joao/code/tcc/seq2seq/data/runs/seq2label/2022-12-07_06-53-12/checkpoint-1000/pytorch_model.bin"
if __name__ == "__main__":
    if model_choice == "seq2seq":
        from transformers import Trainer, logging
        from models.seq2seq.model import FaciesForConditionalGeneration
        from models.seq2seq.configuration import FaciesConfig
        import torch
        from torch.utils.data import DataLoader
        from dataset.dataset import WellsDataset
        import numpy as np

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # read string from current_model.txt
        def collate_fn(batch):
            src_batch, tgt_batch = [], []
            for (src_sample, tgt_sample) in batch:
                tgt_batch.append(tgt_sample)
                src_batch.append(src_sample)
            src_batch = torch.stack(src_batch)
            tgt_batch = torch.stack(tgt_batch)

            model_input = {"input_ids": src_batch, "labels": tgt_batch}
            return model_input

        facies_transformer_config = FaciesConfig.from_pretrained(
            f"{trained_models_path}/config"
        )

        facies_transformer = FaciesForConditionalGeneration(
            facies_transformer_config
        ).to(DEVICE)

        facies_transformer.load_state_dict(torch.load(model_path))

        BATCH_SIZE = 128
        SEQUENCE_LEN = 5
        TRAINING_RATIO = 0.95
        WIRELINE_LOGS_HEADER = ["GR", "NPHI", "RSHA", "DTC", "RHOB", "SP"]
        LABEL_COLUMN_HEADER = ["FORCE_2020_LITHOFACIES_LITHOLOGY"]
        CATEGORICAL_COLUMNS = ["FORMATION", "GROUP"]
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
            categories_label_encoders=train_dataset.categories_label_encoders,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
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
                num_beams=3,
                max_new_tokens=facies_transformer_config.sequence_len + 1,
            )

            decoded_labels = torch.cat((decoded_labels, outputs[:, 1:-1].flatten()))

        from datasets import load_metric

        metrics = dict()
        accuracy_metric = load_metric("accuracy")
        labels = test_dataset.train_label.flatten().to(DEVICE)
        decoded_labels = decoded_labels[labels != 12]
        labels = labels[labels != 12]
        metrics.update(
            accuracy_metric.compute(predictions=decoded_labels, references=labels)
        )

        wells_depth = test_dataset.df_position
        wells_depth["FORCE_2020_LITHOFACIES_LITHOLOGY"] = decoded_labels.cpu().numpy()

        os.makedirs(predictions_path, exist_ok=True)
        wells_depth.to_csv(f"{predictions_path}/facies_prediction.csv")

    elif model_choice == "seq2label":

        from transformers import Trainer, logging
        from models.seq2seq.model import (
            FaciesForConditionalGeneration,
            FaciesForClassification,
        )
        from models.seq2seq.configuration import FaciesConfig
        import torch
        from torch.utils.data import DataLoader
        from dataset.dataset import WellsDataset
        import numpy as np

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # read string from current_model.txt
        def collate_fn(batch):
            src_batch, tgt_batch = [], []
            for (src_sample, tgt_sample) in batch:
                tgt_batch.append(tgt_sample)
                src_batch.append(src_sample)
            src_batch = torch.stack(src_batch)
            tgt_batch = torch.stack(tgt_batch)

            model_input = {"input_ids": src_batch, "labels": tgt_batch}
            return model_input

        facies_transformer_config = FaciesConfig.from_pretrained(
            f"{trained_models_path}/config"
        )

        facies_transformer = FaciesForClassification(facies_transformer_config).to(
            DEVICE
        )
        facies_transformer.load_state_dict(torch.load(model_path))

        BATCH_SIZE = 128
        SEQUENCE_LEN = 5
        TRAINING_RATIO = 0.95
        WIRELINE_LOGS_HEADER = ["GR", "NPHI", "RSHA", "DTC", "RHOB", "SP"]
        LABEL_COLUMN_HEADER = ["FORCE_2020_LITHOFACIES_LITHOLOGY"]
        CATEGORICAL_COLUMNS = ["FORMATION", "GROUP"]
        train_dataset = WellsDataset(
            dataset_type="train",
            sequence_len=facies_transformer_config.sequence_len,
            model_type="seq2label",
            feature_columns=WIRELINE_LOGS_HEADER,
            label_columns=LABEL_COLUMN_HEADER,
        )
        test_dataset = WellsDataset(
            dataset_type="test",
            sequence_len=facies_transformer_config.sequence_len,
            model_type="seq2label",
            feature_columns=WIRELINE_LOGS_HEADER,
            label_columns=LABEL_COLUMN_HEADER,
            scaler=train_dataset.scaler,
            output_len=train_dataset.output_len,
            categories_label_encoders=train_dataset.categories_label_encoders,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Loop for generating the output of a sequence for all the data in the test dataloader using model.generate

        decoded_labels = torch.empty(0, dtype=torch.long).to(DEVICE)
        for i, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            outputs = facies_transformer(input_ids)
            decoded_labels = torch.cat(
                (decoded_labels, outputs[0].argmax(-1).flatten())
            )

        from datasets import load_metric

        metrics = dict()
        accuracy_metric = load_metric("accuracy")
        labels = test_dataset.train_label.flatten().to(DEVICE)
        decoded_labels = decoded_labels[labels != 12]
        labels = labels[labels != 12]
        metrics.update(
            accuracy_metric.compute(predictions=decoded_labels, references=labels)
        )
        wells_depth = test_dataset.df_position
        wells_depth.loc[
            :, ["FORCE_2020_LITHOFACIES_LITHOLOGY"]
        ] = decoded_labels.cpu().numpy()

        os.makedirs(predictions_path, exist_ok=True)
        wells_depth.to_csv(f"{predictions_path}/facies_prediction.csv")

        # Write the model directory to a text file called current_model.txt
    elif model_choice == "xgb":

        from xgboost import XGBClassifier
        from dataset.dataset import WellsDataset
        import os

        seed = 7
        WIRELINE_LOGS_HEADER = ["GR", "NPHI", "RSHA", "DTC", "RHOB", "SP"]
        LABEL_COLUMN_HEADER = ["FORCE_2020_LITHOFACIES_LITHOLOGY"]

        test_dataset = WellsDataset(
            dataset_type="test",
            model_type="label2label",
            feature_columns=WIRELINE_LOGS_HEADER,
            label_columns=LABEL_COLUMN_HEADER,
        )

        model = XGBClassifier()
        model.load_model(f"{trained_models_path}/model.json")
        # save to Json
        x_test = test_dataset.X
        y_test = test_dataset.y
        y_pred = model.predict(x_test)
        # save predictions to file

        test_dataset.df_position["FORCE_2020_LITHOFACIES_LITHOLOGY"] = y_pred

        os.makedirs(predictions_path, exist_ok=True)
        test_dataset.df_position.to_csv(f"{predictions_path}/facies_prediction.csv")
