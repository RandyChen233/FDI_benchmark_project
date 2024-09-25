import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import numpy as np
from quovadis_tad import config_data3D


def create_tf_dataset(
    data_array: np.ndarray,    
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    reconstruction_dataset=False,
    shuffle=False,
    multi_horizon=False,
    input_for_gnn=False
    ):
    """Creates tensorflow dataset from numpy array.

    This function creates a dataset where each element is a tuple `(inputs, targets)`.
    `inputs` is a Tensor
    of shape `(batch_size, input_sequence_length, num_routes, 1)` containing
    the `input_sequence_length` past values of the timeseries for each node.
    `targets` is a Tensor of shape `(batch_size, forecast_horizon, num_routes)`
    containing the `forecast_horizon`
    future values of the timeseries for each node.

    Args:
        data_array: np.ndarray with shape `(num_time_steps, num_routes)`
        input_sequence_length: Length of the input sequence (in number of timesteps).
        forecast_horizon: If `multi_horizon=True`, the target will be the values of the timeseries for 1 to
            `forecast_horizon` timesteps ahead. If `multi_horizon=False`, the target will be the value of the
            timeseries `forecast_horizon` steps ahead (only one value).
        
        reconstruction_dataset: False for forecasting mode. if True will return the input batch as target (for reconstruction tasks, auto encoders etc.,)
        batch_size: Number of timeseries samples in each batch.
        shuffle: Whether to shuffle output samples, or instead draw them in chronological order.
        multi_horizon: See `forecast_horizon`.

    Returns:
        A tf.data.Dataset instance.
    """
    if reconstruction_dataset:
        shuffle_if = shuffle
    else:
        shuffle_if = False
        
    inputs = timeseries_dataset_from_array(
        data_array[:-forecast_horizon],
        #np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=shuffle_if,
        batch_size=batch_size,
    )
    
    if not reconstruction_dataset:
        target_offset = (
            input_sequence_length
            if multi_horizon
            else input_sequence_length + forecast_horizon - 1
        )
        target_seq_length = forecast_horizon if multi_horizon else 1
        targets = timeseries_dataset_from_array(
            data_array[target_offset:],
            None,
            sequence_length=target_seq_length,
            shuffle=False,
            batch_size=batch_size,
        )
    else:
        targets = inputs
    #labels = label_array[target_offset:]
    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    if config_data3D.data3D:
        return dataset
    else:
        return dataset.prefetch(16).cache()


def create_tf_dataset_from_3d_array(data_array,
                      input_sequence_length=12,
                      forecast_horizon=1,
                      batch_size=1,
                      reconstruction_dataset=False,
                      shuffle=False):
    # Create a list to hold per-scenario datasets
    datasets = []

    for scenario_data in data_array:
        # Create a dataset for each scenario without shuffling
        dataset = create_tf_dataset(
            data_array=scenario_data,
            input_sequence_length=input_sequence_length,
            forecast_horizon=forecast_horizon,
            batch_size=batch_size,
            reconstruction_dataset=reconstruction_dataset,
            shuffle=False  # Do not shuffle individual scenario datasets
        )
        datasets.append(dataset)

    # Create a dataset of datasets
    dataset_of_datasets = tf.data.Dataset.from_tensor_slices(datasets)

    # Interleave the datasets to cycle through scenarios
    combined_dataset = dataset_of_datasets.interleave(
        lambda x: x,
        cycle_length=len(datasets),
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        combined_dataset = combined_dataset.shuffle(buffer_size=100)

    return combined_dataset.prefetch(16).cache()

'''
def create_tf_dataset_from_3d_array(
        data_array_3d,
        input_sequence_length,
        forecast_horizon,
        batch_size=128,
        reconstruction_dataset=False,
        shuffle=False,
        multi_horizon=False
):
    datasets = []
    num_scenarios = data_array_3d.shape[0]

    for i in range(num_scenarios):
        data_array = data_array_3d[i]  # Shape: (num_time_steps, num_features)

        # Use the existing function to create dataset for this scenario
        dataset = create_tf_dataset(
            data_array=data_array,
            input_sequence_length=input_sequence_length,
            forecast_horizon=forecast_horizon,
            batch_size=batch_size,
            reconstruction_dataset=reconstruction_dataset,
            shuffle=shuffle,
            multi_horizon=multi_horizon
        )

        datasets.append(dataset)

    # Combine all datasets
    combined_dataset = datasets[0]
    for ds in datasets[1:]:
        combined_dataset = combined_dataset.concatenate(ds)

    # Shuffle the combined dataset if desired
    if shuffle:
        combined_dataset = combined_dataset.shuffle(buffer_size=1000)

    #combined_dataset = combined_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return combined_dataset.prefetch(buffer_size=tf.data.AUTOTUNE).cache()
    #return combined_dataset.prefetch(16).cache()
    #prefetch(buffer_size=tf.data.AUTOTUNE)
'''