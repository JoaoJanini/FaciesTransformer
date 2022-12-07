import torch
import numpy as np
import pandas as pd
from utils import makeplot
from utils import *
import os
import copy
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from dataset.dataset import (
    get_lithology_names,
    get_index_to_lithology_number,
    get_lithology_numbers,
)
from datetime import datetime

base_path = "/home/joao/code/tcc/seq2seq/data"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
tcc_path = f"{base_path}/tcc/{timestamp}"

models_results = {
    "y_true": {
        "predictions": f"{base_path}/raw",
        "facies_file": "test.csv",
        "sep": ";",
        "lith_column": "FORCE_2020_LITHOFACIES_LITHOLOGY",
    },
    "olawale": {
        "predictions": f"{base_path}/predictions/olawale",
        "facies_file": "all.csv",
        "sep": " ",
        "lith_column": "prediction",
    },
    "seq2seq": {"facies_file": "facies_prediction.csv"},
    "xgb": {
        "facies_file": "facies_prediction.csv",
        "lith_column": "FORCE_2020_LITHOFACIES_LITHOLOGY",
    },
    "seq2label": {"facies_file": "facies_prediction.csv"},
}

for model_name, model_info in models_results.items():
    if model_info.get("predictions", "") == "":
        models_directories = os.listdir(f"{base_path}/predictions/{model_name}")
        last_model = sorted(models_directories)[-1]
        for directory in ["trained_models", "predictions", "runs"]:
            models_results[model_name][
                directory
            ] = f"{base_path}/{directory}/{model_name}/{last_model}"

os.mkdir(tcc_path)

models_data = copy.deepcopy(models_results)
labels = list(get_lithology_names().values())[1:]
for model_name, model in models_data.items():
    predictions_df = pd.read_csv(
        f"{model['predictions']}/{model['facies_file']}", sep=model.get("sep", ",")
    )
    facies_predictions = predictions_df[
        model.get("lith_column", "FORCE_2020_LITHOFACIES_LITHOLOGY")
    ]

    if max(facies_predictions.values) < 20:
        predictions_df["FORCE_2020_LITHOFACIES_LITHOLOGY"] = np.array(
            [*map(get_index_to_lithology_number().get, facies_predictions.values)]
        )

    models_data[model_name]["pred-lith-code"] = np.array(
        predictions_df["FORCE_2020_LITHOFACIES_LITHOLOGY"]
    )
    models_data[model_name]["pred-index"] = np.array(
        [
            *map(
                get_lithology_numbers().get,
                models_data[model_name]["pred-lith-code"],
            )
        ]
    )
    models_data[model_name]["pred-lith-name"] = np.array(
        [*map(get_lithology_names().get, models_data[model_name]["pred-lith-code"])]
    )
    models_data[model_name]["predictions_df"] = predictions_df
    print(predictions_df.shape)
# pandas dataframe with metrics as columns and model_names as rows
metrics_df = pd.DataFrame(
    columns=["accuracy", "precision", "recall", "f1", "competition_score"],
    index=list(models_data.keys() - {"y_true"}),
)
cms_path = f"{tcc_path}/confusion_matrices"
os.mkdir(cms_path)
for model_name, model in models_data.items():
    if model_name == "y_true":
        continue
    f1 = f1_score(
        models_data["y_true"]["pred-lith-name"],
        model["pred-lith-name"],
        labels=labels,
        average="macro",
    )
    accuracy = accuracy_score(
        models_data["y_true"]["pred-lith-name"], model["pred-lith-name"]
    )
    precision = precision_score(
        models_data["y_true"]["pred-lith-name"],
        model["pred-lith-name"],
        labels=labels,
        average="macro",
    )
    recall = recall_score(
        models_data["y_true"]["pred-lith-name"],
        model["pred-lith-name"],
        labels=labels,
        average="macro",
    )
    competition_score = score(models_data["y_true"]["pred-index"], model["pred-index"])

    metrics_df.loc[model_name] = [accuracy, precision, recall, f1, competition_score]

    # metrics = utils.get_metrics(
    #     models_data["y_true"]["pred-lith-name"], model["pred-lith-name"], labels
    # )

    get_confusion_matrix(
        model_name,
        models_data["y_true"]["pred-lith-name"],
        model["pred-lith-name"],
        labels,
        cms_path,
    )
    # competition_score = utils.score = utils.get_metrics(
    #     models_data["y_true"]["regular"], model["regular"], labels
    # )


metrics_df.to_markdown(f"{tcc_path}/metrics.md")

y_true_df = models_data["y_true"]["predictions_df"]
wells = y_true_df["WELL"].unique()
total_wells = len(wells)
wells_paths = f"{tcc_path}/wells_plots"
os.mkdir(wells_paths)
for position, well in enumerate(wells):
    current_well = y_true_df.loc[y_true_df["WELL"] == well][
        ["WELL", "DEPTH_MD", "FORCE_2020_LITHOFACIES_LITHOLOGY"]
    ]

    depth = current_well["DEPTH_MD"].values
    top_depth = max(depth)
    bottom_depth = min(depth)
    print(well)

    makeplot(
        models=models_data,
        well_name=well,
        depth=depth,
        top_depth=top_depth,
        bottom_depth=bottom_depth,
        path=wells_paths,
    )
