import numpy as np
from knn_robustness.utils import kth_max, kth_min
from knn_robustness.utils import top_k_min_indices
from knn_robustness.utils import KnnPredictor


class ErgodicVerifier:

    def __init__(self, X_train, y_train, n_neighbors):
        self._X_train = X_train
        self._y_train = y_train
        self._n_neighbors = n_neighbors
        self._predictor = KnnPredictor(X_train, y_train, n_neighbors)

    def predict_batch(self, X_eval):
        return self._predictor.predict_batch(X_eval)

    def predict_individual(self, x_eval):
        return self._predictor.predict_individual(x_eval)

    def __call__(self, x_eval):
        X_pos, X_neg = self._partition(x_eval)
        bounds = np.empty(X_neg.shape[0])
        for j, x_neg in enumerate(X_neg):
            bounds[j] = kth_max(
                np.maximum(
                    np.sum(
                        np.multiply(2 * x_eval - X_pos - x_neg, X_pos - x_neg),
                        axis=1
                    ),
                    0
                ) / (2 * np.linalg.norm(X_pos - x_neg, axis=1)),
                k=(self._n_neighbors+1)//2
            )
        return kth_min(bounds, k=(self._n_neighbors+1)//2)

    def _partition(self, x_eval):
        y_pred = self.predict_individual(x_eval)
        mask = (self._y_train == y_pred)
        X_pos = self._X_train[mask]
        X_neg = self._X_train[~mask]
        return X_pos, X_neg


class RelaxVerifier(ErgodicVerifier):
    """Only use part of positive instances"""

    def __init__(self, X_train, y_train, n_neighbors, n_selective):
        super().__init__(X_train, y_train, n_neighbors)
        self._n_selective = n_selective

    def _partition(self, x_eval):
        X_pos, X_neg = super()._partition(x_eval)

        indices = top_k_min_indices(
            np.linalg.norm(X_pos - x_eval, axis=1),
            self._n_selective
        )
        return X_pos[indices], X_neg
