from .base import Attack
from knn_robustness.utils import BinarySearch


import numpy as np


class MeanAttack(Attack):

    def __init__(self, X_train, y_train, n_neighbors):
        super().__init__(X_train, y_train, n_neighbors)
        self._means = self._compute_means(X_train, y_train)
        self._binary_search = BinarySearch(self._predictor)

    def _compute_means(self, X_train, y_train):
        n_labels = y_train.max() + 1
        n_dim = X_train.shape[1]

        means = np.empty((n_labels, n_dim))
        for label in range(n_labels):
            means[label] = np.mean(
                X_train[y_train == label], axis=0
            )
        return means

    def __call__(self, x_eval):
        y_eval = self.predict_individual(x_eval)
        neg_mean = self._compute_nearest_neg_mean(x_eval, y_eval)
        if self.predict_individual(neg_mean) == y_eval:
            return None
        else:
            return self._binary_search(x_eval, y_eval, neg_mean) - x_eval

    def _compute_nearest_neg_mean(self, x_eval, y_eval):
        masked_means = np.ma.array(self._means, mask=False)
        masked_means.mask[y_eval] = True
        idx = np.argmin(
            np.sum((masked_means - x_eval)**2, axis=1)
        )
        return masked_means[idx].data
