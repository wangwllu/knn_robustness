import numpy as np
from knn_robustness.utils import QpSolver
from knn_robustness.utils import QpSolverFactory


class Subsolver:

    def __init__(self, qp_solver: QpSolver, feasibility_eps=1e-5):
        self._qp_solver = qp_solver
        self._feasibility_eps = feasibility_eps

    def __call__(self, x_eval, X_far, X_near, inner_product_far=None):
        """Return the minimum perturbation such that
        x+delta is closer to X_near than X_far"""

        b = self._compute_b(x_eval, X_far, X_near)
        A = self._compute_A(X_far, X_near)
        Q = self._compute_Q(X_far, X_near, inner_product_far, A)

        lamda = self._qp_solver(Q, b)

        if np.all(Q @ lamda + b + self._feasibility_eps >= 0):
            return A.T @ lamda
        else:
            None

    def _compute_b(self, x_eval, X_far, X_near):
        b_list = []
        for x_near in X_near:
            b_list.append(
                0.5 * np.sum(
                    np.multiply(
                        X_far + x_near - 2 * x_eval, X_far - x_near
                    ), axis=1
                )
            )
        return np.concatenate(b_list)

    def _compute_A(self, X_far, X_near):
        A_list = []
        for x_near in X_near:
            A_list.append(
                x_near - X_far
            )
        return np.concatenate(A_list)

    def _compute_Q(self, X_far, X_near, inner_product_far, A):
        """efficient computation for A @ A.T"""

        if inner_product_far is None:
            inner_product_far = X_far @ X_far.T

        inner_product_far_near = X_far @ X_near.T
        inner_product_near = X_near @ X_near.T

        Q_block = []
        for i in range(X_near.shape[0]):
            Q_block.append([])
            for j in range(X_near.shape[0]):
                if i > j:
                    Q_block[i].append(
                        Q_block[j][i].T
                    )
                else:
                    Q_block[i].append(
                        inner_product_far
                        - inner_product_far_near[:, i][np.newaxis, :]
                        - inner_product_far_near[:, j][:, np.newaxis]
                        + inner_product_near[i, j]
                    )
        return np.block(Q_block)


class SubsolverFactory:
    def create(self, name):
        qpsolver_factory = QpSolverFactory()
        return Subsolver(
            qpsolver_factory.create(name)
        )
