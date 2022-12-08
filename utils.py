from sklearn import metrics
import numpy as np
import pandas as pd

import torch
import matplotlib.patches as mpatches
from dataset.dataset import (
    get_lithology_names,
    get_index_to_lithology_number,
    get_lithology_numbers,
)
def lith_code_to_index(lith_encoded_df):
    indexed_facies = np.array(
            [
                *map(get_lithology_numbers().get, lith_encoded_df)
            ]
    )
    return indexed_facies

def lith_code_to_name(lith_encoded_df):
    lith_names = np.array(
            [
                *map(get_lithology_names().get, lith_encoded_df)
            ]
    )
    return lith_names
def index_to_lith_code(indexed_facies):
    lith_encoded_df = np.array(
            [
                *map(get_index_to_lithology_number().get, indexed_facies)
            ]
    )
    return lith_encoded_df

def create_missing_mask():
    def get_mask_batch2(self, tensor):
        mask = torch.zeros(tensor.shape[0], tensor.shape[1], tensor.shape[1])
        mask = tensor.unsqueeze(1) & tensor.unsqueeze(2)
        return mask

    step_wise_attention_mask = torch.isnan(input_ids[:, :, n_features - 1])

    step_wise_attention_mask = ~self.get_mask_batch2(~step_wise_attention_mask)
    step_wise_attention_mask = step_wise_attention_mask.unsqueeze(1).expand(
        -1, num_heads, -1, -1
    )
    step_wise_attention_mask = step_wise_attention_mask.reshape(
        -1, sequence_len, sequence_len
    )


def get_confusion_matrix(title, y_trues, y_preds, labels, path):
    import seaborn as sn
    import matplotlib
    confusion_matrix = metrics.confusion_matrix(
        y_trues, y_preds, labels=labels, normalize="true"
    )
    df_cm = pd.DataFrame(confusion_matrix, labels, labels)
    # colormap: see this and choose your more dear
    df_cm.drop(columns=["Basement"], index=["Basement"])
    fig, ax = matplotlib.pyplot.subplots(figsize=(32.0, 32.0))
    # for label size
    sn.heatmap(
        df_cm, annot=True, fmt="g", annot_kws={"size": 16}, cmap="Oranges", ax=ax
    )  # font size
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(f"{title}", fontsize=32)
    matplotlib.pyplot.yticks(rotation=45)
    fig.show()
    matplotlib.pyplot.show(block=False)
    fig.savefig(f"{path}/{title}.jpg")

    return confusion_matrix


def f1_score(y_trues, y_preds, labels):
    from sklearn.metrics import f1_score

    f1_score(y_trues, y_preds, average="macro")


# get table with f1, precision, recall, accuracy using sklearn
def get_metrics(y_trues, y_preds, labels):
    from sklearn.metrics import classification_report

    cr = classification_report(y_trues, y_preds, labels=labels)
    print(cr)
    return cr


# Get the confusion matrix from y_trues and y_preds
# Print the confusion matrix
# get pandas dataframe


def score(y_true, y_pred):
    A = np.load("/home/joao/code/tcc/seq2seq/data/raw/penalty_matrix.npy")
    S = 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(0, y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S / y_true.shape[0]


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
# REDO THE FOLLOWING CODE SO THAT IT PLOTS ON LITHOFACIE TRACK FOR EACH MODEL PREDCITION, AND ONE FOR THE ACTUAL PREDICTION. Dont plot thhe GR, NPHI and RHOB tracks. Just the lithofacies track.


def makeplot(models, well_name, depth, top_depth, bottom_depth, path):
    import matplotlib
    # matplotlib.use("pgf")
    # matplotlib.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })


    fig, ax = matplotlib.pyplot.subplots(figsize=(15, 10))

    # Set up the plot axes
    ax1 = matplotlib.pyplot.subplot2grid((1, 5), (0, 0), rowspan=1, colspan=1)
    ax2 = matplotlib.pyplot.subplot2grid((1, 5), (0, 1), rowspan=1, colspan=1, sharey=ax1)
    ax3 = matplotlib.pyplot.subplot2grid((1, 5), (0, 2), rowspan=1, colspan=1, sharey=ax1)
    ax4 = matplotlib.pyplot.subplot2grid((1, 5), (0, 3), rowspan=1, colspan=1, sharey=ax1)
    ax5 = matplotlib.pyplot.subplot2grid((1, 5), (0, 4), rowspan=1, colspan=1, sharey=ax1)

    for model_name, ax in zip(models, [ax1, ax2, ax3, ax4, ax5]):
        models[model_name]["ax"] = ax

    # As our curve scales will be detached from the top of the track,
    # this code adds the top border back in without dealing with splines

    for model_name, model in models.items():
        ax = model["ax"]
        well = model["predictions_df"].loc[
            (model["predictions_df"]["WELL"] == well_name)
            | (model["predictions_df"]["WELL"] == well_name.replace(" ", ""))
        ][["WELL", "DEPTH_MD", "FORCE_2020_LITHOFACIES_LITHOLOGY"]]

        ax.plot(
            well["FORCE_2020_LITHOFACIES_LITHOLOGY"],
            depth,
            color="black",
            linewidth=0.5,
        )
        ax.set_xlim(0, 1)
        ax.xaxis.label.set_color("black")
        ax.spines["top"].set_edgecolor("black")
        ax.set_xlabel(model["plot_title"])
        if ax == ax1:
            ax.set_ylabel("Depth (m)")
        for key in lithology_numbers.keys():
            color = lithology_numbers[key]["color"]
            hatch = lithology_numbers[key]["hatch"]
            ax.fill_betweenx(
                depth,
                0,
                well["FORCE_2020_LITHOFACIES_LITHOLOGY"],
                where=(well["FORCE_2020_LITHOFACIES_LITHOLOGY"] == key),
                facecolor=color,
                hatch=hatch,
            )
    #############################################################################################################

    # Common functions for setting up the plot can be extracted into
    # a for loop. This saves repeating code.
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_ylim(bottom_depth, top_depth)
        ax.xaxis.set_label_position("top")
        matplotlib.pyplot.setp(ax.get_xticklabels(), visible=False)

    for ax in [ax2, ax3, ax4, ax5]:
        matplotlib.pyplot.setp(ax.get_yticklabels(), visible=False)

    patches = [
        mpatches.Patch(
            facecolor=lithology_numbers[key]["color"],
            hatch=lithology_numbers[key]["hatch"],
            label=lithology_numbers[key]["lith"],
        )
        for key in lithology_numbers.keys()
    ]

    # user legend_fig as legend to the main plot
    ax5.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        mode="expand",
    )
    fig.subplots_adjust(wspace=0.15)
    fig.suptitle(f"{well_name} Lithology", fontsize=16)
    # remove / from well name to save as png
    well_name = well_name.replace("/", "_")
    fig.show()
    matplotlib.pyplot.show(block=False)
    fig.savefig(f"{path}/{well_name}.jpg")
