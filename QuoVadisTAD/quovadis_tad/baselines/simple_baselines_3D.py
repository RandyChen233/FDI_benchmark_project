from sklearn import metrics

from typing import Optional, Union

import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA

from quovadis_tad.baselines.utils import check_timeseries_shape_3D, TADMethodEstimator

class NNDistance_3D(TADMethodEstimator):
    def __init__(self, distance: str = 'euclidean') -> None:
        """Computes an anomaly score as the distance to the nearest neighbor in the train set.

        Args:
            distance: The distance metric to be used, defaults to Euclidean.
        """
        self.distance = distance
        self.train_data = None

    def fit(self, x: np.ndarray, univariate: bool = False, verbose: bool = False) -> None:
        """
        Stores the training data in its original 3D form.

        Args:
            x: Training data with shape [num_batches, num_train_samples, num_features].
        """
        check_timeseries_shape_3D(x)
        self.train_data = x  # Store the 3D training data directly

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Computes nearest neighbor distances for each batch in the test data.

        Args:
            x: Test data with shape [num_batches, num_test_samples, num_features].

        Returns:
            min_distances: Minimum distance to the nearest neighbor for each test sample.
        """
        check_timeseries_shape_3D(x)

        num_test_batches, num_test_samples, num_features = x.shape
        num_train_batches = self.train_data.shape[0]

        # Initialize an array to store the minimum distances for each test sample
        min_distances = np.full((num_test_batches, num_test_samples), np.inf)

        # Loop over each batch in the test data
        for test_batch_idx in range(num_test_batches):
            test_batch = x[test_batch_idx]  # Shape: [num_test_samples, num_features]

            # Loop over each batch in the train data
            for train_batch_idx in range(num_train_batches):
                train_batch = self.train_data[train_batch_idx]  # Shape: [num_train_samples, num_features]

                # Compute pairwise distances between the current test batch and train batch
                neighbor_distances = metrics.pairwise.pairwise_distances(
                    test_batch,
                    train_batch,
                    metric=self.distance
                )

                # Update the minimum distances for this test batch
                min_distances[test_batch_idx] = np.minimum(min_distances[test_batch_idx],
                                                           neighbor_distances.min(axis=1))

        # Flatten the min_distances array to return a 1D array with the results for all batches
        #return min_distances.flatten()
        return min_distances

class LNorm_3D(TADMethodEstimator):
    def __init__(self, ord: int = 2) -> None:
        """
        Computes the L^n norm as the anomaly score of a sequence for 3D datasets (num_batches, num_samples, num_features).
        The method defaults to the L2 norm, but the power can be changed if needed.

        Args:
            ord: The power n of the L^n norm. Defaults to 2.
        """
        self.ord = ord

    def fit(self, x: np.ndarray, univariate: bool = False, verbose: bool = False) -> None:
        check_timeseries_shape_3D(x)
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the L^n norm for each time step and feature across batches.

        Args:
            x: Test data with shape [num_batches, num_samples, num_features].

        Returns:
            A 2D array of shape [num_batches, num_samples], where each value represents
            the L^n norm of the corresponding time step and feature set.
        """
        check_timeseries_shape_3D(x)

        # Compute the L^n norm across the feature axis (axis=2)
        return np.linalg.norm(x, ord=self.ord, axis=2)

class Random_3D(TADMethodEstimator):
    def __init__(self, seed: Optional[int] = None) -> None:
        """The method randomly selects an anomaly value of 0 or 1 per timestamp for each batch.
        The method does not make sense in any practical setting, but it is useful to expose issues in scoring,
        for example when using F1 score with point adjust (PA).

        Args:
            seed: An optional random seed for repeatability.
        """
        self.seed = seed

    def fit(self, x: np.ndarray, univariate: bool = False, verbose: bool = False) -> None:
        check_timeseries_shape_3D(x)
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        check_timeseries_shape_3D(x)
        np.random.seed(self.seed)

        # Output shape should be (nb batches, nbr time steps)
        return np.random.randint(
            low=0,
            high=2,
            size=(x.shape[0], x.shape[1])  # size=(nb batches, nbr time steps)
        )