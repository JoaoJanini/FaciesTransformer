import torch
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import makeplot, make_plot_facies_only

run_path = "/home/joao/code/tcc/seq2seq/saved_models/2022-11-23_02-14-14"
facies_prediction = pd.read_csv(f"{run_path}/facies_prediction.csv", sep=",")
facies = pd.read_csv(f"{run_path}/facies.csv", sep=",")

y_pred_decoded = np.array(facies_prediction["FORCE_2020_LITHOFACIES_LITHOLOGY"])
y_true_decoded = np.array(facies["FORCE_2020_LITHOFACIES_LITHOLOGY"])
y_pred_decoded_lith_names = np.array(
    [*map(utils.get_lithology_names().get, y_pred_decoded)]
)
y_true_decoded_lith_names = np.array(
    [*map(utils.get_lithology_names().get, y_true_decoded)]
)
labels = list(utils.get_lithology_names().values())[1:]
# utils.get_lithology_numbers to map index to label in both y_true and y_pred
utils.get_confusion_matrix(y_true_decoded_lith_names, y_pred_decoded_lith_names, labels)
utils.get_metrics(y_true_decoded_lith_names, y_pred_decoded_lith_names, labels)
# print(utils.score(y_true, y_pred))

lithology_numbers = {
    30000: {"lith": "Sandstone", "lith_num": 1, "hatch": "..", "color": "#ffff00"},
    65030: {
        "lith": "Sandstone/Shale",
        "lith_num": 2,
        "hatch": "-.",
        "color": "#ffe119",
    },
    65000: {"lith": "Shale", "lith_num": 3, "hatch": "--", "color": "#bebebe"},
    80000: {"lith": "Marl", "lith_num": 4, "hatch": "", "color": "#7cfc00"},
    74000: {"lith": "Dolomite", "lith_num": 5, "hatch": "-/", "color": "#8080ff"},
    70000: {"lith": "Limestone", "lith_num": 6, "hatch": "+", "color": "#80ffff"},
    70032: {"lith": "Chalk", "lith_num": 7, "hatch": "..", "color": "#80ffff"},
    88000: {"lith": "Halite", "lith_num": 8, "hatch": "x", "color": "#7ddfbe"},
    86000: {"lith": "Anhydrite", "lith_num": 9, "hatch": "", "color": "#ff80ff"},
    99000: {"lith": "Tuff", "lith_num": 10, "hatch": "||", "color": "#ff8c00"},
    90000: {"lith": "Coal", "lith_num": 11, "hatch": "", "color": "black"},
    93000: {"lith": "Basement", "lith_num": 12, "hatch": "-|", "color": "#ef138a"},
}

# plt.show()
axes = []
fig, ax_master = plt.subplots(figsize=(30, 20))
wells = facies_prediction["WELL"].unique()
total_wells = len(wells)
for position, well in enumerate(facies_prediction["WELL"].unique()):
    data = facies_prediction.loc[facies_prediction["WELL"] == well][
        ["WELL", "DEPTH_MD", "FORCE_2020_LITHOFACIES_LITHOLOGY"]
    ]
    ax = make_plot_facies_only(data, lithology_numbers, position, total_wells)
    axes.append(ax)

for axi in axes:
    plt.setp(ax.get_yticklabels(), visible=False)

plt.tight_layout()
fig.subplots_adjust(wspace=0.1)
plt.show()
plt.savefig("test_facies.png")
print("teste")


axes = []
fig, ax_master = plt.subplots(figsize=(30, 20))
wells = facies["WELL"].unique()
total_wells = len(wells)
for position, well in enumerate(facies["WELL"].unique()):
    data = facies.loc[facies["WELL"] == well][
        ["WELL", "DEPTH_MD", "FORCE_2020_LITHOFACIES_LITHOLOGY"]
    ]
    ax = make_plot_facies_only(data, lithology_numbers, position, total_wells)
    axes.append(ax)

for axi in axes:
    plt.setp(ax.get_yticklabels(), visible=False)

plt.tight_layout()
fig.subplots_adjust(wspace=0.1)
plt.show()
plt.savefig("true_facies.png")
print("teste")
