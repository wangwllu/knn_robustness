import math
import numpy as np

from knn_robustness.utils import top_k_min_indices
from knn_robustness.utils import KnnPredictor
from knn_robustness.utils import QpSolver


class ExactSolver:

    def __init__(
            self, X_train, y_train, qp_solver: QpSolver,
            n_pos_for_screen, bounded, upper=1., lower=0.
    ):
        self._X_train = X_train
        self._y_train = y_train
        self._qp_solver = qp_solver

        self._n_pos_for_screen = n_pos_for_screen
        self._bounded = bounded
        self._upper = upper
        self._lower = lower

        self._predictor = KnnPredictor(X_train, y_train, n_neighbors=1)

    def predict_batch(self, X_eval):
        return self._predictor.predict_batch(X_eval)

    def predict_individual(self, x_eval):
        return self._predictor.predict_individual(x_eval)

    def __call__(self, x_eval):
        X_pos, X_neg = self._partition(x_eval)
        X_screen = self._compute_pos_for_screen(x_eval, X_pos)
        inner_product_pos = X_pos @ X_pos.T

        best_perturbation = None
        min_perturbation_norm = math.inf

        for x_neg in self._neg_generator(x_eval, X_neg):
            if self._screenable(
                    x_eval, x_neg, X_screen, min_perturbation_norm
            ):
                continue
            else:
                perturbation = self._solve_subproblem(
                    x_eval, x_neg, X_pos, inner_product_pos
                )
                perturbation_norm = np.linalg.norm(perturbation)
                if perturbation_norm < min_perturbation_norm:
                    min_perturbation_norm = perturbation_norm
                    best_perturbation = perturbation

        return best_perturbation

    def _partition(self, x_eval):
        y_pred = self.predict_individual(x_eval)
        mask = (self._y_train == y_pred)
        X_pos = self._X_train[mask]
        X_neg = self._X_train[~mask]
        return X_pos, X_neg

    def _compute_pos_for_screen(self, x_eval, X_pos):
        indices = top_k_min_indices(
            np.linalg.norm(x_eval - X_pos, axis=1),
            self._n_pos_for_screen
        )
        return X_pos[indices]

    def _neg_generator(self, x_eval, X_neg):
        indices = np.argsort(
            np.linalg.norm(
                X_neg - x_eval, axis=1
            )
        )
        for i in indices:
            yield X_neg[i]

    def _screenable(self, x_eval, x_neg, X_screen, threshold):
        return threshold <= np.max(
            np.maximum(
                np.sum(
                    np.multiply(
                        2 * x_eval - X_screen - x_neg, X_screen - x_neg
                    ),
                    axis=1
                ),
                0
            ) / (2 * np.linalg.norm(X_screen - x_neg, axis=1))
        )

    def _solve_subproblem(self, x_eval, x_neg, X_pos, inner_product_pos=None):

        A, b, Q = self._compute_qp_params(
            x_eval, x_neg, X_pos, inner_product_pos
        )
        lamda = self._qp_solver(Q, b)
        return -A.T @ lamda

    def _compute_qp_params(
            self, x_eval, x_neg, X_pos, inner_product_pos
    ):
        if inner_product_pos is None:
            inner_product_pos = X_pos @ X_pos.T

        # A @ u <= b
        A = 2 * (X_pos - x_neg)

        # test: this one is much more efficient due to less multiplications
        b = np.sum(np.multiply(X_pos + x_neg - 2 * x_eval,
                               X_pos - x_neg), axis=1)

        # X @ y
        temp = X_pos @ x_neg

        # A @ A.T = 4 * (X @ X.T - X @ y - (X @ y).T + y.T @ y)
        Q = 4 * (inner_product_pos - temp[np.newaxis, :]
                 - temp[:, np.newaxis] + x_neg @ x_neg)

        # min 0.5 * v.T @ P @ v + v.T @ b, v >= 0
        # max - 0.5 * v.T @ P @ v - v.T @ b, v >= 0
        if not self._bounded:
            return A, b, Q

        else:
            # upper bound
            # A1 @ delta <= b1
            # z + delta <= upper
            A1 = np.identity(X_pos.shape[1], dtype=X_pos.dtype)
            b1 = self._upper - x_eval

            # lower bound
            # A2 @ delta <= b2
            # z + delta >= lower
            A2 = -np.identity(X_pos.shape[1], dtype=X_pos.dtype)
            b2 = x_eval - self._lower

            # A_full @ A_full.T
            Q_full = np.block([
                [Q, A, -A],
                [A.T, A1, A2],
                [-A.T, A2, A1],
            ])

            A_full = np.block([
                [A],
                [A1],
                [A2]
            ])

            b_full = np.concatenate([b, b1, b2])
            return A_full, b_full, Q_full
