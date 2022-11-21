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
