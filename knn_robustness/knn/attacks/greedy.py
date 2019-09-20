import numpy as np
import math

from .base import Attack
from ..subsolvers import Subsolver

from knn_robustness.utils import top_k_min_indices


class GreedyAttack(Attack):

    def __init__(
            self, X_train, y_train, n_neighbors,
            subsolver: Subsolver,
            n_far, max_trials, min_trials, eps=1e-6
    ):
        assert max_trials >= min_trials
        assert min_trials > 0

        super().__init__(X_train, y_train, n_neighbors)
        self._subsolver = subsolver
        self._n_far = n_far
        self._max_trials = max_trials
        self._min_trials = min_trials
        self._eps = eps

    def __call__(self, x_eval):

        y_eval = self.predict_individual(x_eval)

        best_perturbation = None
        min_perturbation_norm = math.inf

        for i, (x_pivot, y_pivot) in enumerate(
                self._pivots_generator(x_eval, y_eval)
        ):

            if self._time_to_stop(i, best_perturbation):
                break

            X_far = self._compute_far(x_eval, y_eval, x_pivot, y_pivot)
            X_near = self._compute_near(x_eval, y_eval, x_pivot, y_pivot)
            perturbation = self._subsolver(x_eval, X_far, X_near)

            if self._successful(x_eval, y_eval, perturbation):

                perturbation_norm = np.linalg.norm(perturbation)
                if perturbation_norm < min_perturbation_norm:
                    best_perturbation = perturbation
                    min_perturbation_norm = perturbation_norm

        return best_perturbation

    def _pivots_generator(self, x_eval, y_eval):
        mask = self._y_train != y_eval
        X_cand = self._X_train[mask]
        y_cand = self._y_train[mask]
        indices = np.argsort(
            np.linalg.norm(X_cand - x_eval, axis=1)
        )[:self._max_trials]
        for i in indices:
            yield X_cand[i], y_cand[i]

    def _time_to_stop(self, i, best_perturbation):
        return i >= self._min_trials and best_perturbation is not None

    def _compute_far(self, x_eval, y_eval, x_pivot, y_pivot):
        X_cand = self._compute_complete_far(x_eval, y_eval, x_pivot, y_pivot)
        indices = top_k_min_indices(
            np.linalg.norm(
                X_cand - x_eval, axis=1
            ), self._n_far
        )
        return X_cand[indices]

    def _compute_complete_far(self, x_eval, y_eval, x_pivot, y_pivot):
        if self._n_neighbors == 1:
            return self._X_train[
                self._y_train == y_eval
            ]
        X_cand = self._X_train[
            y_pivot != self._y_train
        ]
        indices = top_k_min_indices(
            -np.linalg.norm(
                X_cand - x_pivot, axis=1
            ), X_cand.shape[0] - (self._n_neighbors - 1) // 2
        )
        return X_cand[indices]

    def _compute_near(self, x_eval, y_eval, x_pivot, y_pivot):
        if self._n_neighbors == 1:
            return x_pivot[np.newaxis, :]
        X_cand = self._X_train[
            y_pivot == self._y_train
        ]
        indices = top_k_min_indices(
            np.linalg.norm(
                X_cand - x_pivot, axis=1
            ), (self._n_neighbors + 1) // 2
        )
        return X_cand[indices]

    def _successful(self, x_eval, y_eval, perturbation):
        return ((perturbation is not None)
                and (
                y_eval != self.predict_individual(
                    x_eval + perturbation)
                or y_eval != self.predict_individual(
                    x_eval + perturbation + self._eps)
                ))
