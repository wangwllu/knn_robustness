from abc import ABC, abstractmethod

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class Predictor(ABC):

    @abstractmethod
    def predict_batch(self, X_eval):
        pass

    def predict_individual(self, x_eval):
        return self.predict_batch(x_eval[np.newaxis, :]).item()


class KnnPredictor(Predictor):

    def __init__(self, X_train, y_train, n_neighbors, algorithm='brute'):
        self._classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors, algorithm=algorithm
        )
        self._classifier.fit(X_train, y_train)

    def predict_batch(self, X_eval):
        return self._classifier.predict(X_eval)
