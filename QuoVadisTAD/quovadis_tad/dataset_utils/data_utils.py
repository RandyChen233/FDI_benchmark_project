import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import iqr


def preprocess_data(
        data_array: np.ndarray,
        test_array: np.ndarray,
        train_size: float,
        val_size: float,
        normalization="mean-std"
):
    """Splits data into train/val/test sets and normalizes the data.

    Args:
        data_array: ndarray of shape `(num_time_steps, num_routes)`
        train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the train split.
        val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the validation split.
        normalization (normalize data):  "mean-std" : standard scaler norm - "0-1": MinMax norm
    Returns:
        `train_array`, `val_array`, `test_array`
    """
    if normalization == "mean-std":
        scaler = StandardScaler()
        data_array = scaler.fit_transform(data_array)
        test_array = scaler.transform(test_array)
    elif normalization == "0-1":
        scaler = MinMaxScaler()
        data_array = scaler.fit_transform(data_array)
        test_array = scaler.transform(test_array)
    else:
        pass
        print(f'returning raw data')
    
    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    train_array = data_array[:num_train]
    val_array = data_array[num_train: (num_train + num_val)]
    return train_array.astype('float32'), val_array.astype('float32'), test_array.astype('float32')


def preprocess_data_3D(
        data_array: np.ndarray,
        test_array: np.ndarray,
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

    # Split into train and validation
    num_train = int(nbr_time_steps * train_size)
    num_val = int(nbr_time_steps * val_size)

    train_array = data_array[:, :num_train, :]
    val_array = data_array[:, num_train:(num_train + num_val), :]

    return train_array.astype('float32'), val_array.astype('float32'), test_array.astype('float32')


def normalise_scores(test_delta, norm="median-iqr", smooth=True, smooth_window=5):
    """
    Args:
        norm: None, "mean-std" or "median-iqr"
    """
    if norm == "mean-std":
        err_scores = StandardScaler().fit_transform(test_delta)
    elif norm == "median-iqr":
        n_err_mid = np.median(test_delta, axis=0)
        n_err_iqr = iqr(test_delta, axis=0)
        epsilon = 1e-2

        err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)
    elif norm is None:
        err_scores = test_delta
    else:
        raise ValueError('specified normalisation not implemented, please use one of {None, "mean-std", "median-iqr"}')    
    
    if smooth:
        smoothed_err_scores = np.zeros(err_scores.shape)

        for i in range(smooth_window, len(err_scores)):
            smoothed_err_scores[i] = np.mean(err_scores[i - smooth_window: i + smooth_window - 1], axis=0)
        return smoothed_err_scores
    else:
        return err_scores


def concatenate_windows_feat(arr, window_size=5):
    i = 0
    # Initialize an empty list to store
    arr = np.vstack([np.repeat(arr[0][None, :], window_size - 1, axis=0), arr])
    cat_feats = []

    # Loop through the array to consider every window
    while i < len(arr) - window_size + 1:

        # Concatenate current window
        cat_f = arr[i:i + window_size].flatten('F')

        # window list
        cat_feats.append(cat_f)

        # Shift window to right by one position
        i += 1
    cat_feats = np.array(cat_feats)    
    
    return cat_feats
