import os
import numpy as np
import pandas as pd

from knn_robustness.nn_only import ExactSolver
from knn_robustness.utils import QpSolverFactory

from knn_robustness.utils import initialize_params
from knn_robustness.utils import initialize_data


params = initialize_params('exact')
X_train, y_train, X_test, y_test = initialize_data(params)

exact_solver = ExactSolver(
    X_train=X_train,
    y_train=y_train,
    qp_solver=QpSolverFactory().create(params.get('qpsolver')),
    n_pos_for_screen=params.getint('n_pos_for_screen'),
    bounded=params.getboolean('bounded'),
)

count = 0
perturbation_norms = []
for instance, label in zip(X_test, y_test):
    if exact_solver.predict_individual(instance) != label:
        continue
    perturbation = exact_solver(instance)

    perturbation_norm = np.linalg.norm(perturbation)

    perturbation_norms.append(perturbation_norm)

    details = pd.DataFrame({
        'perturbation': perturbation_norms
    })
    details.to_csv(os.path.join(params.get('result_dir'), 'detail.csv'))

    count += 1
    print(f'{count:03d} {perturbation_norm:.9f}')
    if count >= params.getint('n_evaluate'):
        break

summary = pd.DataFrame({
    'num': [count],
    'mean': [details['perturbation'].mean()],
    'median': [details['perturbation'].median()]
})

summary.to_csv(os.path.join(params.get('result_dir'), 'summary.csv'))
print(summary)
