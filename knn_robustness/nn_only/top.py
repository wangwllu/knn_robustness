from .exact import ExactSolver
from knn_robustness.utils import QpSolver


class TopAttack(ExactSolver):

    def __init__(
            self, X_train, y_train, qp_solver: QpSolver,
            n_pos_for_screen, bounded, n_top, upper=1., lower=0.,
    ):
        super().__init__(
            X_train, y_train, qp_solver,
            n_pos_for_screen, bounded, upper=1., lower=0.
        )
        self._n_top = n_top

    def _neg_generator(self, x_eval, X_neg):
        for i, x_neg in enumerate(
                super()._neg_generator(x_eval, X_neg)
        ):
            if i >= self._n_top:
                break
            yield x_neg
