"""
functions for reading in benchmark datasets:
    - WADI
    - SWaT
    - smd
    - smap
    - msl
    - UCR/IB
"""

from pathlib import Path
from typing import Union, Tuple, Dict, Callable, List
from enum import Enum
import pandas as pd
import os
import numpy as np
# import h5py
import glob

DATA_DIR = 'resources/processed_datasets'
"""the name of the data directory"""


def find_files_in_path(directory: str, file_ending: str = ".npy") -> List[str]:
    """list all the files with ending in a path.

    :param directory: the directory where you'd like to find the files

    :param file_ending: the type of file. Default: .npy

    :return: a list of matched files
    """
    # initialize the output list
    found_files = []

    # Walk through the files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_ending):
                # add a found file to the list
                found_files.append(file)
    return found_files


class GeneralDatasetNames(Enum):
    wadi_127 = 'wadi_127'
    wadi_112 = 'wadi_112'
    swat = 'swat'
    smd = 'smd'
    msl = 'msl'
    smap = 'smap'
    ucr_IB_16 = 'ucr_IB_16'
    ucr_IB_17 = 'ucr_IB_17'
    ucr_IB_18 = 'ucr_IB_18'
    ucr_IB_19 = 'ucr_IB_19'
    ucr_IB = 'ucr_IB'
    ourBench='ourBench'
    ourBench2= 'ourBench2'
    ourBench3 = 'ourBench3'
    ourClassdata = 'ourClassdata'


def load_wadi_127(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "WADI_127"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def load_wadi_112(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "WADI_112"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def load_swat(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "SWaT"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


# load only the trace used in TransAD
def load_ucr_IB_17(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "UCR"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'136_{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def load_ucr_IB_16(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "UCR"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'135_{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels

def load_ucr_IB_18(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "UCR"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'137_{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def load_ucr_IB_19(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "UCR"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'138_{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels

def load_ucr_IB(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # load all four UCR/IB traces
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "UCR"

    data_traces = list(set([f'{filename.split("_")[0]}_' for filename in find_files_in_path(directory=files_path)]))
    
    print(f'[INFO:] UCR contains {len(data_traces)} data traces')

    data = []

    for file in ['train', 'test', 'labels']:
        trace_data = []
        for traces in data_traces:
            file_ = traces + file
            trace_data.append(np.load(Path(files_path, f'{file_}.npy')))

        data.append(trace_data)  
    
    return data[0], data[1], data[2]


def load_smd(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "smd"

    data_traces = list(set([f'{filename.split("_")[0]}_' for filename in find_files_in_path(directory=files_path)]))
    
    print(f'[INFO:] smd contains {len(data_traces)} data traces.')

    data = []

    for file in ['train', 'test', 'labels']:
        trace_data = []
        for traces in data_traces:
            file_ = traces + file
            trace_data.append(np.load(Path(files_path, f'{file_}.npy')))

        data.append(trace_data)  
    # Please note that the `labels` is a 2D array
    return data[0], data[1], data[2]


def load_msl(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "msl"

    data_traces = list(set([f'{filename.split("_")[0]}_' for filename in find_files_in_path(directory=files_path)]))
    if "P-2_" in data_traces:
        data_traces.remove("P-2_")
    print(f'[INFO:] msl contains {len(data_traces)} data traces.')

    data = []

    for file in ['train', 'test', 'labels']:
        trace_data = []
        for traces in data_traces:
            file_ = traces + file
            trace_data.append(np.load(Path(files_path, f'{file_}.npy')))

        data.append(trace_data)

    # Ensure the number of data points
    assert np.concatenate(data[0]).shape[0] == 58317
    assert np.concatenate(data[1]).shape[0] == 73729
    assert np.concatenate(data[2]).shape[0] == 73729
    assert np.concatenate(
        data[0]).shape[1] == np.concatenate(
        data[1]).shape[1] == np.concatenate(
            data[2]).shape[1] == 55

    # Please notes that the `labels` is a 2D array
    return data[0], data[1], data[2]


def load_smap(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "smap"

    data_traces = list(set([f'{filename.split("_")[0]}_' for filename in find_files_in_path(directory=files_path)]))
    
    print(f'[INFO:] smap contains {len(data_traces)} data traces.')

    data = []

    for file in ['train', 'test', 'labels']:
        trace_data = []
        for traces in data_traces:
            file_ = traces + file
            trace_data.append(np.load(Path(files_path, f'{file_}.npy')))

        data.append(trace_data)

    # Please note that the `labels` is a 2D array
    return data[0], data[1], data[2]

'''
def load_ourBench(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "ourBench"

    data_traces = list(set([f'{filename.split("_")[0]}_' for filename in find_files_in_path(directory=files_path)]))

    print(f'[INFO:] ourBench contains {len(data_traces)} data traces.')

    data = []

    for file in ['train', 'test', 'labels']:
        trace_data = []
        for traces in data_traces:
            file_ = traces + file
            trace_data.append(np.load(Path(files_path, f'{file_}.npy')))

        data.append(trace_data)
        # Please note that the `labels` is a 2D array
    return data[0], data[1], data[2]
'''
def load_ourBench(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "ourBench"
    # Directory where your files are located
    directory = files_path

    # Define file patterns
    train_pattern = os.path.join(directory, 'train_Ndist_*_period_*_fsize_*_train.npy')
    test_pattern = os.path.join(directory, 'test_Ndist_*_period_*_fsize_*_test.npy')
    labels_pattern = os.path.join(directory, 'test_Ndist_*_period_*_fsize_*_labels.npy')

    # Gather file paths
    train_files = glob.glob(train_pattern)
    test_files = glob.glob(test_pattern)
    labels_files = glob.glob(labels_pattern)

    # Sort files if needed (to ensure consistent order)
    train_files.sort()
    test_files.sort()
    labels_files.sort()

    # Load data
    data = []
    data.append([np.load(file) for file in train_files])  # Load and store train files
    data.append([np.load(file) for file in test_files])    # Load and store test files
    data.append([np.load(file) for file in labels_files])  # Load and store label files

    # Example usage
    print("Number of train files:", len(data[0]))
    print("Number of test files:", len(data[1]))
    print("Number of label files:", len(data[2]))

    return data[0], data[1], data[2]

def load_ourBench2(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "ourBench2"
    # Directory where your files are located
    directory = files_path

    # Define file patterns
    train_pattern = os.path.join(directory, 'train_Ndist_*_period_*_fsize_*_train.npy')
    test_pattern = os.path.join(directory, 'test_Ndist_*_period_*_fsize_*_test.npy')
    labels_pattern = os.path.join(directory, 'test_Ndist_*_period_*_fsize_*_labels.npy')

    # Gather file paths
    train_files = glob.glob(train_pattern)
    test_files = glob.glob(test_pattern)
    labels_files = glob.glob(labels_pattern)

    # Sort files if needed (to ensure consistent order)
    train_files.sort()
    test_files.sort()
    labels_files.sort()

    # Load data
    data = []
    data.append([np.load(file) for file in train_files])  # Load and store train files
    data.append([np.load(file) for file in test_files])    # Load and store test files
    data.append([np.load(file) for file in labels_files])  # Load and store label files

    # Example usage
    print("Number of train files:", len(data[0]))
    print("Number of test files:", len(data[1]))
    print("Number of label files:", len(data[2]))

    return data[0], data[1], data[2]

def load_ourBench3(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "ourBench3"
    # Directory where your files are located
    directory = files_path

    # Define file patterns
    train_pattern = os.path.join(directory, 'train_Ndist_*_period_*_fsize_*_train.npy')
    test_pattern = os.path.join(directory, 'test_Ndist_*_period_*_fsize_*_test.npy')
    labels_pattern = os.path.join(directory, 'test_Ndist_*_period_*_fsize_*_labels.npy')

    # Gather file paths
    train_files = glob.glob(train_pattern)
    test_files = glob.glob(test_pattern)
    labels_files = glob.glob(labels_pattern)

    # Sort files if needed (to ensure consistent order)
    train_files.sort()
    test_files.sort()
    labels_files.sort()

    # Load data
    data = []
    data.append([np.load(file) for file in train_files])  # Load and store train files
    data.append([np.load(file) for file in test_files])    # Load and store test files
    data.append([np.load(file) for file in labels_files])  # Load and store label files

    # Example usage
    print("Number of train files:", len(data[0]))
    print("Number of test files:", len(data[1]))
    print("Number of label files:", len(data[2]))

    return data[0], data[1], data[2]


def load_ourClassdata(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "ourClassdata"
    # Directory where your files are located
    directory = files_path

    # Define file patterns
    data_pattern = os.path.join(directory, 'Ndist_*_period_*_fsize_*_Nfaults_*_train.npy')
    labels_pattern = os.path.join(directory, 'Ndist_*_period_*_fsize_*_Nfaults_*_labels.npy')

    # Gather file paths
    data_files = glob.glob(data_pattern)
    labels_files = glob.glob(labels_pattern)

    # Sort files if needed (to ensure consistent order)
    data_files.sort()
    labels_files.sort()

    # Load data
    data = []
    data.append([np.load(file) for file in data_files])  # Load and store train files
    data.append([np.load(file) for file in labels_files])  # Load and store label files

    # Example usage
    print("Number of data files:", len(data[0]))
    print("Number of label files:", len(data[1]))

    return data[0], data[1]

datasets:  \
    Dict[str, Callable[[Union[str, Path]], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        GeneralDatasetNames.wadi_127.value: load_wadi_127,
        GeneralDatasetNames.wadi_112.value: load_wadi_112,
        GeneralDatasetNames.swat.value: load_swat,
        GeneralDatasetNames.smd.value: load_smd,
        GeneralDatasetNames.msl.value: load_msl,
        GeneralDatasetNames.smap.value: load_smap,
        GeneralDatasetNames.ucr_IB_17.value: load_ucr_IB_17,
        GeneralDatasetNames.ucr_IB_16.value: load_ucr_IB_16,
        GeneralDatasetNames.ucr_IB_18.value: load_ucr_IB_18,
        GeneralDatasetNames.ucr_IB_19.value: load_ucr_IB_19,    
        GeneralDatasetNames.ucr_IB.value: load_ucr_IB,
        GeneralDatasetNames.ourBench.value: load_ourBench,
        GeneralDatasetNames.ourBench2.value: load_ourBench2,
        GeneralDatasetNames.ourBench3.value: load_ourBench3,
        GeneralDatasetNames.ourClassdata.value: load_ourClassdata,
    }

