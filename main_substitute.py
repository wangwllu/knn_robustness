import os
import numpy as np
import pandas as pd
import math

from knn_robustness.nn_only import SubstituteAttack

from knn_robustness.utils import initialize_params
from knn_robustness.utils import initialize_data


params = initialize_params('substitute')
X_train, y_train, X_test, y_test = initialize_data(params)

substitute_attack = SubstituteAttack(
    X_train=X_train,
    y_train=y_train,
    step_size=params.getfloat('step_size'),
    max_iter=params.getint('max_iter')
)

count = 0
success_notes = []
perturbation_norms = []
for instance, label in zip(X_test, y_test):
    if substitute_attack.predict_individual(instance) != label:
        continue
    perturbation = substitute_attack(instance)
    if perturbation is None:
        success = False
        perturbation_norm = math.inf
    else:
        success = True
        perturbation_norm = np.linalg.norm(perturbation)

    success_notes.append(success)
    perturbation_norms.append(perturbation_norm)

    details = pd.DataFrame({
        'success': success_notes,
        'perturbation': perturbation_norms
    })
    details.to_csv(os.path.join(params.get('result_dir'), 'detail.csv'))

    count += 1
    print(f'{count:03d} {success} {perturbation_norm:.7f}')
    if count >= params.getint('n_evaluate'):
        break

summary = pd.DataFrame({
    'num': [count],
    'success_rate': [details['success'].sum()/count],
    'mean': [details['perturbation'][details['success']].mean()],
    'median': [details['perturbation'][details['success']].median()]
})

summary.to_csv(os.path.join(params.get('result_dir'), 'summary.csv'))
print(summary)
