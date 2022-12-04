import torch
import numpy as np
import pandas as pd
from utils import makeplot
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from dataset.dataset import get_lithology_names, get_lithology_numbers
with open("current_model.txt", "r") as f:
    current_model = f.read()

base_path = "/home/joao/code/tcc/seq2seq/data"

models_results = {
    "y_true": {"path": "/home/joao/code/tcc/seq2seq/data/raw/test.csv"},
    "seq2seq": {},
    "xgb": {},
    "seq2label": {}
}

for model_name, model_info in models_results.items():
    if model_info.get("path", {}) is {}:
        continue

    models_directories = os.listdir(f"{base_path}/predictions/{model_name}")
    last_model = models_directories.sort()[-1]
    models_results[model_name]["path"] = f"{base_path}/predictions/{model_name}/{last_model}/facies_prediction.csv"


labels = list(get_lithology_names().values())[1:]
for model_name, model in models_results.items():

    predictions_df = pd.read_csv(f"{ model['path']}", sep=",")

    y_pred_decoded = np.array(predictions_df["FORCE_2020_LITHOFACIES_LITHOLOGY"])
    y_pred_decoded_lith_names = np.array(
        [*map(get_lithology_names().get, y_pred_decoded)]
    )

    y_pred = np.array(
        [*map(get_lithology_numbers().get, y_pred_decoded)]
    )

    models_results[model_name]["regular"] = y_pred
    models_results[model_name]["decoded"] = y_pred_decoded
    models_results[model_name]["lith_name"] = y_pred_decoded_lith_names
    models_results[model_name]["predictions_df"] = predictions_df


# utils.get_lithology_numbers to map index to label in both y_true and y_pred
# utils.get_confusion_matrix(y_true_decoded_lith_names, y_pred_decoded_lith_names, labels)
# utils.get_metrics(y_true_decoded_lith_names, y_pred_decoded_lith_names, labels)
# metrics = pd.DataFrame(columns=["Model", "F1", "Precision", "Recall", "Accuracy"])

# for model_name, model in models_results.items():
#     if model_name == "y_true":
#         continue
#     f1 = f1_score(models_results["y_true"]["lith_name"], model["lith_name"],labels= labels, average='macro')
#     accuracy = accuracy_score(models_results["y_true"]["lith_name"], model["lith_name"])
#     precision = precision_score(models_results["y_true"]["lith_name"], model["lith_name"], labels= labels, average='macro')
#     recall = recall_score(models_results["y_true"]["lith_name"], model["lith_name"], labels= labels, average='macro')
#     # competition_score = utils.score(
#     #     models_results["y_true"]["regular"], model["regular"]
#     # )

#     metrics = metrics.append([model_name, f1, precision, recall, accuracy], ignore_index=True)
#     # metrics = utils.get_metrics(
#     #     models_results["y_true"]["lith_name"], model["lith_name"], labels
#     # )
#     # confusion_matrix = utils.get_confusion_matrix(
#     #     models_results["y_true"]["lith_name"], model["lith_name"], labels
#     # )
#     # competition_score = utils.score = utils.get_metrics(
#     #     models_results["y_true"]["regular"], model["regular"], labels
#     # )

# print(metrics)

wells = models_results["y_true"]["predictions_df"]["WELL"].unique()
total_wells = len(wells)
for position, well in enumerate(wells):
    makeplot(models=models_results, well_name=well)
