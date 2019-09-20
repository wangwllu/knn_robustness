from abc import ABC, abstractmethod
from knn_robustness.utils import KnnPredictor


class Attack(ABC):

    def __init__(self, X_train, y_train, n_neighbors):
        self._X_train = X_train
        self._y_train = y_train
        self._n_neighbors = n_neighbors
        self._predictor = KnnPredictor(X_train, y_train, n_neighbors)

    def predict_batch(self, X_eval):
        return self._predictor.predict_batch(X_eval)

    def predict_individual(self, x_eval):
        return self._predictor.predict_individual(x_eval)

    @abstractmethod
    def __call__(self, x_eval):
        pass
