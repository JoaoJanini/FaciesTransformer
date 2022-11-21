import torch
import utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, logging
from hf_sequence_to_sequence.model import FaciesForConditionalGeneration
from hf_sequence_to_sequence.configuration import FaciesConfig
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config_path = "/home/joao/code/tcc/seq2seq/saved_models/2022-11-20_18-38-58/facies-transformer-config"
model_path = "/home/joao/code/tcc/seq2seq/saved_models/2022-11-20_18-38-58/facies-transformer/facies_transformer_state_dict.pt"


# default `log_dir` is "runs" - we'll be more specific here
facies_transformer_config = FaciesConfig.from_pretrained(
    f"{config_path}"
)

facies_transformer = FaciesForConditionalGeneration(facies_transformer_config).to(DEVICE)

summary(facies_transformer, input_size=(3, 30), batch_size=640, device="cuda")



index_to_lith_code = {v: k for k, v in utils.get_lithology_numbers().items()}
y_pred = np.array(torch.load('y_pred.pt').cpu())
y_true = np.array(torch.load('y_true.pt').cpu())
y_pred_decoded = np.array([*map(index_to_lith_code.get, y_pred)])
y_true_decoded = np.array([*map(index_to_lith_code.get, y_true)])
y_pred_decoded_lith_names = np.array([*map(utils.get_lithology_names().get, y_pred_decoded)])
y_true_decoded_lith_names = np.array([*map(utils.get_lithology_names().get, y_true_decoded)])
labels = list(utils.get_lithology_names().values())[1:]
# utils.get_lithology_numbers to map index to label in both y_true and y_pred
utils.get_confusion_matrix(y_true_decoded_lith_names, y_pred_decoded_lith_names, labels)
utils.get_metrics(y_true_decoded_lith_names, y_pred_decoded_lith_names, labels)
print(utils.score(y_true, y_pred))
