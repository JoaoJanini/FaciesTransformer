from sklearn import model_selection
import torch

# GPU device setting
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "saved_models/model_"

# dataset settings
SEQUENCE_LEN = 4
TRAINING_RATIO = 0.95
WIRELINE_LOGS_HEADER = ["DEPTH_MD", "GR", "NPHI"]
LABEL_COLUMN_HEADER = ["FORCE_2020_LITHOFACIES_LITHOLOGY"]

# model parameter setting
batch_size = 640
max_len = 256
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

# optimizer parameter setting
init_lr = 1e-4
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 10
clip = 1.0
weight_decay = 5e-4
inf = float("inf")

correct_on_train = []
correct_on_test = []

"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
