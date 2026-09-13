import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


DATASETS_REQUIRING_GLOBAL_NORMALIZATION = {
    'AllGestureWiimoteX',
    'AllGestureWiimoteY',
    'AllGestureWiimoteZ',
    'BME',
    'Chinatown',
    'Crop',
    'EOGHorizontalSignal',
    'EOGVerticalSignal',
    'Fungi',
    'GestureMidAirD1',
    'GestureMidAirD2',
    'GestureMidAirD3',
    'GesturePebbleZ1',
    'GesturePebbleZ2',
    'GunPointAgeSpan',
    'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung',
    'HouseTwenty',
    'InsectEPGRegularTrain',
    'InsectEPGSmallTrain',
    'MelbournePedestrian',
    'PickupGestureWiimoteZ',
    'PigAirwayPressure',
    'PigArtPressure',
    'PigCVP',
    'PLAID',
    'PowerCons',
    'Rock',
    'SemgHandGenderCh2',
    'SemgHandMovementCh2',
    'SemgHandSubjectCh2',
    'ShakeGestureWiimoteZ',
    'SmoothSubspace',
    'UMD',
}


def load_csv_data(data_path, data_name, split='TRAIN'):
    data_file = os.path.join(data_path, f"{split}.csv")
    label_file = os.path.join(data_path, f"{split}_label.csv")

    data = pd.read_csv(data_file, header=None).values
    labels = pd.read_csv(label_file, header=None).values.flatten()
    indices = np.arange(data.shape[0])
    data = data.reshape(data.shape[0], -1, 1)

    if data_name in DATASETS_REQUIRING_GLOBAL_NORMALIZATION:
        mean = np.nanmean(data)
        std = np.nanstd(data)
        if std == 0:
            std = 1.0
        data = (data - mean) / std

    return data, labels, indices


def create_data_loader(data, labels, index, batch_size, shuffle=True):
    temporal_missing = np.isnan(data).all(axis=-1).any(axis=0)
    if temporal_missing[0] or temporal_missing[-1]:
        data = centerize_vary_length_series(data)

    data = data[~np.isnan(data).all(axis=2).all(axis=1)]

    tensor_data = torch.tensor(data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    tensor_index = torch.tensor(index, dtype=torch.long)
    dataset = TensorDataset(tensor_data, tensor_labels, tensor_index)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]
