import os
import re
from pathlib import Path
import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dataset_utils.dataset_reader import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_data_3D(
        data_array: np.ndarray,
        test_array: np.ndarray,
        y_data: np.ndarray,
        train_size: float,
        val_size: float,
        normalization="mean-std"
):
    """Splits 3D data (nbr_batches x nbr_time_steps x nbr_features) into train/val/test sets and normalizes the data.

    Args:
        data_array: 3D ndarray of shape `(nbr_batches, nbr_time_steps, nbr_features)`
        test_array: 3D ndarray of shape `(nbr_batches, nbr_time_steps, nbr_features)` for testing
        train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the train split.
        val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the validation split.
        normalization (normalize data):  "mean-std" : standard scaler norm - "0-1": MinMax norm

    Returns:
        `train_array`, `val_array`, `test_array` normalized with respect to all batches.
    """

    # Get original shape
    nbr_batches, nbr_time_steps, nbr_features = data_array.shape

    # Reshape to 2D: (nbr_batches * nbr_time_steps, nbr_features)
    reshaped_data = data_array.reshape(-1, nbr_features)
    reshaped_test = test_array.reshape(-1, nbr_features)

    # Apply normalization
    if normalization == "mean-std":
        scaler = StandardScaler()
        reshaped_data = scaler.fit_transform(reshaped_data)
        reshaped_test = scaler.transform(reshaped_test)
    elif normalization == "0-1":
        scaler = MinMaxScaler()
        reshaped_data = scaler.fit_transform(reshaped_data)
        reshaped_test = scaler.transform(reshaped_test)
    else:
        print(f'returning raw data')


    # Reshape the data back to 3D: (nbr_batches, nbr_time_steps, nbr_features)
    data_array = reshaped_data.reshape(nbr_batches, nbr_time_steps, nbr_features)
    test_array = reshaped_test.reshape(test_array.shape[0], test_array.shape[1], nbr_features)

    train_array, val_array, y_train, y_valid = train_test_split(
        data_array, y_data, test_size=0.2, random_state=42, shuffle=True
    )

    return train_array.astype('float32'), val_array.astype('float32'), test_array.astype('float32'),y_train, y_valid



def read_data(module_path, dataset_name, dataset_trace=None,
              preprocess="0-1"):
    # prepare dataset
    mydata, labels = datasets[dataset_name](module_path)
    labels=np.array(labels)
    labels= labels.max(axis=2)
    mydata = np.array(mydata)

    trainset, testset, y_data, y_test = train_test_split(
        mydata, labels, test_size=0.2, random_state=42, shuffle=True
    )

    # Preprocess datasets
    trainset, valset, testset,y_train, y_valid = preprocess_data_3D(trainset, testset, y_data,train_size=0.9, val_size=0.1,
                                                   normalization=preprocess)

    return trainset, valset, testset, y_train, y_valid, y_test
