from re import I
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch
from sklearn import preprocessing
from urllib.request import urlopen
import numpy as np

WIRELINE_LOGS = [
    "WELL",
    "DEPTH_MD",
    "X_LOC",
    "Y_LOC",
    "Z_LOC",
    "GROUP",
    "FORMATION",
    "CALI",
    "RSHA",
    "RMED",
    "RDEP",
    "RHOB",
    "GR",
    "SGR",
    "NPHI",
    "PEF",
    "DTC",
    "SP",
    "BS",
    "ROP",
    "DTS",
    "DCAL",
    "DRHO",
    "MUDWEIGHT",
    "RMIC",
    "ROPA",
    "RXO",
]

original_lithology_numbers = {
    30000: 0,
    65030: 1,
    65000: 2,
    80000: 3,
    74000: 4,
    70000: 5,
    70032: 6,
    88000: 7,
    86000: 8,
    99000: 9,
    90000: 10,
    93000: 11,
}


class WellsDataset(Dataset):
    def __init__(
        self,
        sequence_len=5,
        model_type="seq2seq",
        dataset_type="train",
        download_data_locally=True,
        path="data",
        label_columns=["FORCE_2020_LITHOFACIES_LITHOLOGY"],
        feature_columns=WIRELINE_LOGS,
        scaler=None,
        output_len=None,
    ):
        """
        Dataset objects for training datasets and test datasets
        :param path: dataset path
        :param dataset_type: distinguish whether to get the training set or the test set
        :param download_data_locally: whether to download the data locally or to just get it from the url everytime
        :param label_columns: the header of the target column

        """
        urls = {
            "train": "https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/raw/master/lithology_competition/data/train.zip",
            "test": "https://raw.githubusercontent.com/bolgebrygg/Force-2020-Machine-Learning-competition/master/lithology_competition/data/hidden_test.csv",
        }

        super(WellsDataset, self).__init__()
        self.dataset_type = dataset_type
        self.download_data_locally = download_data_locally
        self.path = f"{path}/{dataset_type}.csv"
        self.url = urls[self.dataset_type]
        self.data_df = self.download_data()
        self.feature_columns = feature_columns
        self.target = label_columns
        self.model_type = model_type
        self.sequence_len = sequence_len
        self.scaler = scaler
        self.output_len = output_len
        # Define special symbols and indices
        self.PAD_IDX = 0
        self.special_symbols = [self.PAD_IDX]
        # Make sure the tokens are in order of their indices to properly insert them in vocab
        if self.output_len is None:
            self.output_len = len(tuple(set(self.data_df[self.target[0]].to_numpy())))

        self.wells = list(self.data_df["WELL"].unique())
        self.data = self.data_df[
            ["WELL"] + ["DEPTH_MD"] + label_columns + feature_columns
        ].dropna()
        self.well_indexes = pd.DataFrame(
            self.data["WELL"].apply(lambda x: self.wells.index(x)), columns=["WELL"]
        )
        self.X = self.prepare_X()
        self.y = self.prepare_y()

        if self.model_type == "seq2seq":
            self.X, self.y = self.separate_by_well()
            (
                self.train_len,
                self.input_len,
                self.channel_len,
                self.train_dataset,
                self.train_label,
            ) = self.prepare_sequences_to_sequences()

    def __getitem__(self, index):
        return (self.train_dataset[index], self.train_label[index])

    def __len__(self):
        return self.train_len

    def download_data(self):
        if self.download_data_locally:
            try:
                data = self.load_data(self.path)
            except:
                data = self.load_data(self.url)
                data.to_csv(self.path, sep=";")
        else:
            data = self.load_data(self.url)
        return data

    def load_data(self, path):
        data = pd.read_csv(path, sep=";")
        return data

    def prepare_X(self):
        X = self.data[self.feature_columns]
        if self.scaler is None:
            self.scaler = preprocessing.StandardScaler().fit(X)
        scaled_X = self.scaler.transform(X)
        self.df_position = self.data[["WELL"] + ["DEPTH_MD"]]
        X_df = pd.DataFrame(scaled_X, columns=X.columns, index=X.index)
        return X_df

    def prepare_y(self):
        y = self.data[self.target]
        y_tmp = (
            y[self.target[0]]
            .apply(lambda x: self.get_lithology_numbers()[x])
            .to_numpy()
        )
        # encoded_yi = np.array(torch.nn.functional.one_hot(torch.Tensor(y_tmp).reshape(-1, 1).long(), self.output_len)).squeeze()
        encoded_yi = np.array(torch.Tensor(y_tmp).reshape(-1, 1).long()).squeeze()

        y_df = pd.DataFrame(encoded_yi, index=y.index)
        return y_df

    def get_positions_df(self):
        return self.df_position

    def separate_by_well(self):
        training_data = []
        training_labels = []

        wells = list(self.well_indexes["WELL"].unique())
        for well_index, _ in enumerate(wells):
            well_rows_index = self.well_indexes[
                self.well_indexes["WELL"] == well_index
            ].index
            xi = self.X.loc[well_rows_index].to_numpy()
            yi = self.y.loc[well_rows_index].to_numpy()

            training_data.append(xi)
            training_labels.append(yi)
        return training_data, training_labels

    def prepare_sequences_to_sequences(self):
        train_dataset = []
        train_label = []
        for x1, y1 in zip(self.X, self.y):
            train_dataset_well = list(torch.split(torch.as_tensor(x1).float(), self.sequence_len))
            train_dataset_label = list(torch.split(torch.as_tensor(y1).float(), self.sequence_len))

            if len(train_dataset_well[-1]) < self.sequence_len:
                train_dataset_well[-1] = torch.nn.functional.pad(
                    train_dataset_well[-1],
                    (0, 0, 0, self.sequence_len - len(train_dataset_well[-1])),
                    "constant",
                    self.PAD_IDX,
                )
                train_dataset_label[-1] = torch.nn.functional.pad(
                    train_dataset_label[-1],
                    (0, 0, 0, self.sequence_len - train_dataset_label[-1].shape[0]),
                )
            train_dataset = train_dataset + train_dataset_well
            train_label =  train_label + train_dataset_label
            # pad last torch tensor from train_label with zeros so that shape is (sequence_len, 1)

        train_dataset = torch.stack(train_dataset, dim=0).permute(0, 1, 2)
        train_label = torch.stack(train_label, dim=0).permute(0, 1, 2).long().squeeze()
        train_len = train_dataset.shape[0]
        channel_len = train_dataset[0].shape[-1]
        input_len = train_dataset[0].shape[-2]
        return train_len, input_len, channel_len, train_dataset, train_label

    def get_lithology_numbers(self):
        lithology_numbers = {
            self.PAD_IDX: self.PAD_IDX,
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

    def get_lithology_names(self):
        # Define special symbols and indices
        lithology_names = {
            self.PAD_IDX: "<pad>",
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


def main():
    WIRELINE_LOGS_HEADER = ["DEPTH_MD", "GR", "NPHI"]
    LABEL_COLUMN_HEADER = ["FORCE_2020_LITHOFACIES_LITHOLOGY"]
    seq2seq = WellsDataset(
        model_type="seq2seq",
        feature_columns=WIRELINE_LOGS_HEADER,
        label_columns=LABEL_COLUMN_HEADER,
        sequence_len=5,
    )
    print(seq2seq.train_dataset.shape, seq2seq.train_label.shape)


if __name__ == "__main__":
    main()
