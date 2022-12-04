import os


base_path = "/home/joao/code/tcc/seq2seq/data/"

models = {
    "seq2seq": {
        "folder_path": f"seq2seq-transformer",
    },
    "xgb": {"folder_path": f"xgboost"},
}

model = "seq2seq"
folder_path = models[model]["folder_path"]

models_directories = os.listdir(f"{base_path}/trained_models/{folder_path}")
last_model = models_directories.sort()[-1]
if __name__ == "__main__":
    if model == "seq2seq":

        model_path = f"{base_path}/trained_models/{folder_path}/{last_model}"
        prediction_path = f"{base_path}/predictions/{folder_path}/{last_model}"


        from transformers import Trainer, logging
        from model import FaciesForConditionalGeneration
        from configuration import FaciesConfig
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

        facies_transformer_config = FaciesConfig.from_pretrained(f"{model_path}/config")

        facies_transformer = FaciesForConditionalGeneration(
            facies_transformer_config
        ).to(DEVICE)
        facies_transformer.load_state_dict(torch.load(f"{model_path}/model.pt"))

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
                num_beams=7,
                max_new_tokens=facies_transformer_config.sequence_len + 1,
            )

            decoded_labels = torch.cat((decoded_labels, outputs[:, 1:-1].flatten()))

        labels = test_dataset.train_label.flatten().to(DEVICE)
        decoded_labels = decoded_labels[labels != 0].cpu().numpy()

        wells_depth = test_dataset.df_position
        wells_depth["FORCE_2020_LITHOFACIES_LITHOLOGY"] = decoded_labels

        os.mkdir(path=prediction_path)
        wells_depth.to_csv(f"{prediction_path}/facies_prediction.csv")

        # Write the model directory to a text file called current_model.txt
    elif model == "xgb":

        model_path = f"{base_path}/trained_models/{folder_path}/{last_model}"
        prediction_path = f"{base_path}/predictions/{folder_path}/{last_model}"

        from numpy import loadtxt
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from dataset.dataset import WellsDataset
        from datetime import datetime
        import os
        import numpy as np
        import xgboost

        seed = 7
        WIRELINE_LOGS_HEADER = ["GR", "NPHI", "RSHA", "DTC", "RHOB", "SP"]
        LABEL_COLUMN_HEADER = ["FORCE_2020_LITHOFACIES_LITHOLOGY"]
        train_dataset = WellsDataset(
            dataset_type="train",
            model_type="label2label",
            feature_columns=WIRELINE_LOGS_HEADER,
            label_columns=LABEL_COLUMN_HEADER,
        )

        test_dataset = WellsDataset(
            dataset_type="test",
            model_type="label2label",
            feature_columns=WIRELINE_LOGS_HEADER,
            label_columns=LABEL_COLUMN_HEADER,
        )

        model = XGBClassifier()
        model.load_model(f"{model_path}/model.json")
        # save to Json
        x_test = test_dataset.X
        y_test = test_dataset.y
        y_pred = model.predict(x_test)
        # save predictions to file

        test_dataset.df_position["FORCE_2020_LITHOFACIES_LITHOLOGY"] = y_pred

        os.mkdir(path=prediction_path)
        test_dataset.df_position.to_csv(f"{prediction_path}/facies_prediction.csv")
