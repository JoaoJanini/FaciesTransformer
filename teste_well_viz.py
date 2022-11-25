import pandas as pd
import matplotlib.pyplot as plt

from utils import makeplot, make_plot_facies_only

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

facies_prediction = pd.read_csv(
    "/home/joao/code/tcc/seq2seq/facies_prediction.csv", sep=","
)
facies = pd.read_csv("/home/joao/code/tcc/seq2seq/facies.csv", sep=",")


# df_lith = pd.DataFrame.from_dict(lithology_numbers, orient="index")
# df_lith.index.name = "LITHOLOGY"
# df_lith

y = [0, 1]
x = [1, 1]

# fig, axes = plt.subplots(
#     ncols=4,
#     nrows=3,
#     sharex=True,
#     sharey=True,
#     figsize=(10, 5),
#     subplot_kw={"xticks": [], "yticks": []},
# )

# for ax, key in zip(axes.flat, lithology_numbers.keys()):
#     ax.plot(x, y)
#     ax.fill_betweenx(
#         y,
#         0,
#         1,
#         facecolor=lithology_numbers[key]["color"],
#         hatch=lithology_numbers[key]["hatch"],
#     )
#     ax.set_xlim(0, 0.1)
#     ax.set_ylim(0, 1)
#     ax.set_title(str(lithology_numbers[key]["lith"]))

# plt.tight_layout()

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
