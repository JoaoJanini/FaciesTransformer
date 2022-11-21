from sklearn import metrics

import numpy as np
import pandas as pd
from pretty_confusion_matrix import pp_matrix
import seaborn as sn
import matplotlib.pyplot as plt

def get_lithology_numbers():
    lithology_numbers = {
        0: 0,
        30000: 1,
        65030: 2,
        65000: 3,
        80000: 4,
        74000: 5,
        70000: 6,
        70032: 7,
        88000: 8,
        86000: 9,
        99000: 10,
        90000: 11,
        93000: 12,
    }
    return lithology_numbers

def get_lithology_names():
    # Define special symbols and indices
    lithology_names = {
        0: "<pad>",
        30000: "Sandstone",
        65030: "Sandstone/Shale",
        65000: "Shale",
        80000: "Marl",
        74000: "Dolomite",
        70000: "Limestone",
        70032: "Chalk",
        88000: "Halite",
        86000: "Anhydrite",
        99000: "Tuff",
        90000: "Coal",
        93000: "Basement",
    }
    return lithology_names

def get_confusion_matrix(y_trues, y_preds, labels):
    confusion_matrix = metrics.confusion_matrix(y_trues, y_preds, labels=labels, normalize='true')
    df_cm = pd.DataFrame(confusion_matrix, labels, labels)
    # colormap: see this and choose your more dear
    empty_columns = list(df_cm.columns[(df_cm == 0).all()])
    df_cm = df_cm[[c for c in df_cm if c not in empty_columns] + empty_columns]
    sn.set(font_scale=1.4) # for label size
    sn.set(rc={'figure.figsize':(32.0,32.0)})
    sn.set()
    sn.heatmap(df_cm, annot=True,fmt='g', annot_kws={"size": 16}, cmap='Oranges') # font size
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.yticks(rotation=45)
    plt.title('Confusion Matrix for Lithology Classification', fontsize=32)

    plt.show()
    return confusion_matrix

def f1_score(y_trues, y_preds, labels):
    from sklearn.metrics import f1_score
    f1_score(y_trues, y_preds, average='macro')

# get table with f1, precision, recall, accuracy using sklearn
def get_metrics(y_trues, y_preds, labels):
    from sklearn.metrics import classification_report
    print(classification_report(y_trues, y_preds, labels=labels))

# Get the confusion matrix from y_trues and y_preds
# Print the confusion matrix
# get pandas dataframe

def score(y_true, y_pred):
    A = np.load('/home/joao/code/tcc/seq2seq/data/penalty_matrix.npy')
    S = 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(0, y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S/y_true.shape[0]


def plot_facies_wells(data):
    # https://github.com/andymcdgeo/Petrophysics-Python-Series/blob/master/14%20-%20Displaying%20Lithology%20Data.ipynb

    def makeplot(well, top_depth, bottom_depth):
        fig, ax = plt.subplots(figsize=(15,10))

        #Set up the plot axes
        ax1 = plt.subplot2grid((1,3), (0,0), rowspan=1, colspan = 1)
        ax2 = plt.subplot2grid((1,3), (0,1), rowspan=1, colspan = 1, sharey = ax1)
        ax3 = ax2.twiny() #Twins the y-axis for the density track with the neutron track
        ax4 = plt.subplot2grid((1,3), (0,2), rowspan=1, colspan = 1, sharey = ax1)

        # As our curve scales will be detached from the top of the track,
        # this code adds the top border back in without dealing with splines
        ax10 = ax1.twiny()
        ax10.xaxis.set_visible(False)
        ax11 = ax2.twiny()
        ax11.xaxis.set_visible(False)
        ax13 = ax4.twiny()
        ax13.xaxis.set_visible(False)

        # Gamma Ray track
        ax1.plot(well["GR"], well['DEPTH_MD'], color = "green", linewidth = 0.5)
        ax1.set_xlabel("Gamma")
        ax1.xaxis.label.set_color("green")
        ax1.set_xlim(0, 200)
        ax1.set_ylabel("Depth (m)")
        ax1.tick_params(axis='x', colors="green")
        ax1.spines["top"].set_edgecolor("green")
        ax1.title.set_color('green')
        ax1.set_xticks([0, 50, 100, 150, 200])

        # Density track
        ax2.plot(well["RHOB"], well['DEPTH_MD'], color = "red", linewidth = 0.5)
        ax2.set_xlabel("Density")
        ax2.set_xlim(1.95, 2.95)
        ax2.xaxis.label.set_color("red")
        ax2.tick_params(axis='x', colors="red")
        ax2.spines["top"].set_edgecolor("red")
        ax2.set_xticks([1.95, 2.45, 2.95])

        # Neutron track placed ontop of density track
        ax3.plot(well["NPHI"], well['DEPTH_MD'], color = "blue", linewidth = 0.5)
        ax3.set_xlabel('Neutron')
        ax3.xaxis.label.set_color("blue")
        ax3.set_xlim(0.45, -0.15)
        ax3.tick_params(axis='x', colors="blue")
        ax3.spines["top"].set_position(("axes", 1.08))
        ax3.spines["top"].set_visible(True)
        ax3.spines["top"].set_edgecolor("blue")
        ax3.set_xticks([0.45,  0.15, -0.15])

        # Lithology track
        ax4.plot(well["LITHOLOGY"], well['DEPTH_MD'], color = "black", linewidth = 0.5)
        ax4.set_xlabel("Lithology")
        ax4.set_xlim(0, 1)
        ax4.xaxis.label.set_color("black")
        ax4.tick_params(axis='x', colors="black")
        ax4.spines["top"].set_edgecolor("black")

        for key in lithology_numbers.keys():
            color = lithology_numbers[key]['color']
            hatch = lithology_numbers[key]['hatch']
            ax4.fill_betweenx(well['DEPTH_MD'], 0, well['LITHOLOGY'], where=(well['LITHOLOGY']==key),
                            facecolor=color, hatch=hatch)
            

        ax4.set_xticks([0, 1])

        # Common functions for setting up the plot can be extracted into
        # a for loop. This saves repeating code.
        for ax in [ax1, ax2, ax4]:
            ax.set_ylim(bottom_depth, top_depth)
            ax.grid(which='major', color='lightgrey', linestyle='-')
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")
            ax.spines["top"].set_position(("axes", 1.02))
            
            
        for ax in [ax2, ax3, ax4]:
            plt.setp(ax.get_yticklabels(), visible = False)
            
        plt.tight_layout()
        fig.subplots_adjust(wspace = 0.15)

def plot_facies(data):

    ## https://www.linkedin.com/pulse/pda-series-2-facies-classification-from-well-logs-yohanes-nuwara
    # Display logs with facies
    logs = data.columns[1:]
    rows,cols = 1,5
    fig,ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,6), sharey=True)

    plt.suptitle('WELL F02-1', size=15)
    for i in range(cols):
        if i < cols-1:
            ax[i].plot(data[logs[i]], data.DEPTH, color='b', lw=0.5)
            ax[i].set_title('%s' % logs[i])
            ax[i].minorticks_on()
            ax[i].grid(which='major', linestyle='-', linewidth='0.5', color='lime')
            ax[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
            ax[i].set_ylim(max(data.DEPTH), min(data.DEPTH))
        elif i==cols-1:
            F = np.vstack((facies,facies)).T
            ax[i].imshow(F, aspect='auto', extent=[0,1,max(data.DEPTH), min(data.DEPTH)])
            ax[i].set_title('FACIES')