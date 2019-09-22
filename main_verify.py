import os
import pandas as pd

from knn_robustness.knn import RelaxVerifier

from knn_robustness.utils import initialize_params
from knn_robustness.utils import initialize_data


params = initialize_params('verify')
X_train, y_train, X_test, y_test = initialize_data(params)

verifier = RelaxVerifier(
    X_train=X_train,
    y_train=y_train,
    n_neighbors=params.getint('n_neighbors'),
    n_selective=params.getint('n_selective')
)

count = 0
lower_bounds = []
for instance, label in zip(X_test, y_test):
    if verifier.predict_individual(instance) != label:
        continue
    lower_bound = verifier(instance)
    lower_bounds.append(lower_bound)

    details = pd.DataFrame(lower_bounds, columns=['lower bound'])
    details.to_csv(os.path.join(params.get('result_dir'), 'detail.csv'))

    count += 1
    print(f'{count:03d} {lower_bound:.3f}')
    if count >= params.getint('n_evaluate'):
        break

summary = pd.DataFrame({
    'num': [count],
    'mean': [details['lower bound'].mean()],
    'median': [details['lower bound'].median()]
})

summary.to_csv(os.path.join(params.get('result_dir'), 'summary.csv'))
print(summary)
