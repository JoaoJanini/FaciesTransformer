import torch
import numpy as np
import pandas as pd
from utils import makeplot, score, get_confusion_matrix, get_metrics, lith_code_to_index, lith_code_to_name, index_to_lith_code
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
from copy import deepcopy

base_path = "/home/joao/code/tcc/seq2seq/data"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
tcc_path = f"{base_path}/tcc/{timestamp}"

models_results = {
    "y_true": {
        "predictions": f"{base_path}/raw",
        "facies_file": "test.csv",
        "sep": ";",
        "lith_column": "FORCE_2020_LITHOFACIES_LITHOLOGY",
        "plot_title": "True Labels",
    },
    "olawale": {
        "predictions": f"{base_path}/predictions/olawale",
        "facies_file": "all.csv",
        "sep": " ",
        "lith_column": "prediction",
        "plot_title": "Olawale's Model",
    },
    "seq2seq": {"facies_file": "facies_prediction.csv", "plot_title": "Encoder-Decoder Model"},
    "xgb": {
        "facies_file": "facies_prediction.csv",
        "lith_column": "FORCE_2020_LITHOFACIES_LITHOLOGY",
        "plot_title": "XGBoost Model",
    },
    "seq2label": {"facies_file": "facies_prediction.csv", "plot_title": "Encoder-Only Model"},
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
    predictions_df["FORCE_2020_LITHOFACIES_LITHOLOGY"] =  predictions_df[
        model.get("lith_column", "FORCE_2020_LITHOFACIES_LITHOLOGY")
    ]
    if max(predictions_df["FORCE_2020_LITHOFACIES_LITHOLOGY"].values) < 20:
        predictions_df["FORCE_2020_LITHOFACIES_LITHOLOGY"] = index_to_lith_code(predictions_df["FORCE_2020_LITHOFACIES_LITHOLOGY"])

    predictions_df["FORCE_2020_LITHOFACIES_LITHOLOGY_ENCODED"] = lith_code_to_index(predictions_df["FORCE_2020_LITHOFACIES_LITHOLOGY"])
    predictions_df["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"] = lith_code_to_name(predictions_df["FORCE_2020_LITHOFACIES_LITHOLOGY"])

    models_data[model_name]["predictions_df"] = predictions_df
    print(predictions_df.shape)
y_true_df = models_data["y_true"]["predictions_df"]

# plot titles besides y_true
plot_titles = [models_data[model]["plot_title"] for model in  list(models_data.keys() - {"y_true"})]
# plot
# pandas dataframe with metrics as columns and model_names as rows
metrics_df = pd.DataFrame(
    columns=["accuracy", "precision", "recall", "f1", "competition_score"],
    index=plot_titles
)

metrics_per_lith_df = pd.DataFrame(
    columns=labels, index=plot_titles
)
accuracy_by_well_df = pd.DataFrame(
    columns=plot_titles, index=sorted(predictions_df["WELL"].unique())
)
score_by_well_df = deepcopy(accuracy_by_well_df)

cms_path = f"{tcc_path}/confusion_matrices"
os.mkdir(cms_path)
for model_name, model in models_data.items():
    if model_name == "y_true":
        continue

    f1 = f1_score(
        y_true_df["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"],
        model["predictions_df"]["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"],
        labels=labels,
        average="macro",
    )
    accuracy = accuracy_score(
        y_true_df["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"], model["predictions_df"]["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"]
    )
    precision = precision_score(
        y_true_df["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"],
        model["predictions_df"]["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"],
        labels=labels,
        average="macro",
    )
    recall = recall_score(
        y_true_df["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"],
        model["predictions_df"]["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"],
        labels=labels,
        average="macro",
    )

    a = pd.crosstab(y_true_df["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"],model["predictions_df"]["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"])
    metrics_per_lith_df.loc[model["plot_title"]] = a.max(axis=1)/a.sum(axis=1)

    competition_score = score(y_true_df["FORCE_2020_LITHOFACIES_LITHOLOGY_ENCODED"], model["predictions_df"]["FORCE_2020_LITHOFACIES_LITHOLOGY_ENCODED"])

    metrics_df.loc[model["plot_title"]] = [accuracy, precision, recall, f1, competition_score]

    accuracy_by_well_df.loc[:,model["plot_title"]] = y_true_df.groupby("WELL").apply(
        lambda x: accuracy_score(x["FORCE_2020_LITHOFACIES_LITHOLOGY_ENCODED"], model["predictions_df"].loc[x.index]["FORCE_2020_LITHOFACIES_LITHOLOGY_ENCODED"])
    )

    score_by_well_df.loc[:,model["plot_title"]] = y_true_df.groupby("WELL").apply(
        lambda x: score(x["FORCE_2020_LITHOFACIES_LITHOLOGY_ENCODED"], model["predictions_df"].loc[x.index]["FORCE_2020_LITHOFACIES_LITHOLOGY_ENCODED"])
    )
    # metrics = utils.get_metrics(
    #     y_true_df["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"], model["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"], labels
    # )

    get_confusion_matrix(
        model["plot_title"],
        y_true_df["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"],
        model["predictions_df"]["FORCE_2020_LITHOFACIES_LITHOLOGY_NAME"],
        labels,
        cms_path,
    )

    # competition_score = utils.score = utils.get_metrics(
    #     y_true_df["regular"], model["regular"], labels
    # )


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

metrics_per_lith_df = metrics_per_lith_df.style.apply(
    lambda x: ["font-weight: bold" if v == x.max() else "" for v in x], axis=1
)
accuracy_by_well_df = accuracy_by_well_df.style.apply(
    lambda x: ["font-weight: bold" if v == x.max() else "" for v in x], axis=1
)
score_by_well_df = score_by_well_df.style.apply(
    lambda x: ["font-weight: bold" if v == x.min() else "" for v in x], axis=1
)

# do the same for the metrics_df, but use min when comparing the competition score
metrics_df.style.apply(
    lambda x: ["font-weight: bold" if v == x.max() else "" for v in x], axis=1
)


metrics_df.to_latex(f"{tcc_path}/metrics.tex", label="tab:metricas-todos-pocos")
metrics_per_lith_df.to_latex(f"{tcc_path}/metrics_per_lith.tex", label="tab:metricas-por-litologia")
accuracy_by_well_df.to_latex(f"{tcc_path}/accuracy_by_well.tex", label="tab:acuracia-por-poco")
score_by_well_df.to_latex(f"{tcc_path}/score_by_well.tex", label="tab:score-por-poco")
