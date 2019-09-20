import numpy as np

from .base import Attack
from knn_robustness.utils import BinarySearch
from knn_robustness.utils import top_k_min_indices


class NaiveAttack(Attack):

    def __init__(self, X_train, y_train, n_neighbors, n_trials):
        super().__init__(X_train, y_train, n_neighbors)
        self._n_trials = n_trials
        self._binary_search = BinarySearch(self._predictor)

    def __call__(self, x_eval):
        y_eval = self.predict_individual(x_eval)

        min_perturbation_norm = np.inf
        best_perturbation = None

        for x_pivot, y_pivot in self._pivots_generator(x_eval, y_eval):
            x_target = self._compute_target(x_eval, y_eval, x_pivot, y_pivot)
            if self.predict_individual(x_target) == y_eval:
                continue
            else:
                perturbation = self._binary_search(
                    x_eval, y_eval, x_target
                ) - x_eval

                perturbation_norm = np.linalg.norm(perturbation)
                if perturbation_norm < min_perturbation_norm:
                    min_perturbation_norm = perturbation_norm
                    best_perturbation = perturbation

        return best_perturbation

    def _pivots_generator(self, x_eval, y_eval):
        mask = self._y_train != y_eval
        X_cand = self._X_train[mask]
        y_cand = self._y_train[mask]
        indices = np.argsort(
            np.linalg.norm(X_cand - x_eval, axis=1)
        )[:self._n_trials]
        for i in indices:
            yield X_cand[i], y_cand[i]

    def _compute_target(self, x_eval, y_eval, x_pivot, y_pivot):
        if self._n_neighbors == 1:
            return x_pivot
        X_cand = self._X_train[
            y_pivot == self._y_train
        ]
        indices = top_k_min_indices(
            np.linalg.norm(
                X_cand - x_pivot, axis=1
            ), (self._n_neighbors + 1) // 2
        )
        return X_cand[indices].mean(axis=0)
