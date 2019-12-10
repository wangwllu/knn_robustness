import unittest
import numpy as np

from knn_robustness.nn_only import ExactSolver
from knn_robustness.utils import QpSolverFactory


class TestExactSolver(unittest.TestCase):

    def setUp(self):
        self.X_train = np.array(
            [[0.5, 0.5], [1, 0.5], [0, 1]], dtype=np.float32
        )
        self.y_train = np.array([1, 1, 0], dtype=np.int)

        self.x_eval = np.array([0, 0], dtype=np.float)

    def test_unbounded(self):
        solver = ExactSolver(
            self.X_train, self.y_train,
            QpSolverFactory().create('gcd'), 1, False
        )
        perturbation = solver(self.x_eval)
        self.assertTrue(
            np.allclose(
                perturbation,
                np.array([-0.25, 0.25], dtype=self.X_train.dtype)
            )
        )
        self.assertEqual(
            perturbation.dtype, self.X_train.dtype
        )

    def test_bounded(self):
        solver = ExactSolver(
            self.X_train, self.y_train,
            QpSolverFactory().create('gcd'), 1, True
        )
        perturbation = solver(self.x_eval)
        self.assertTrue(
            np.allclose(
                perturbation,
                np.array([0, 0.5], dtype=self.X_train.dtype)
            )
        )
        self.assertEqual(
            perturbation.dtype, self.X_train.dtype
        )
