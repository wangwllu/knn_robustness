# K-NN Adversairal Robustness

The official implementations of algorithms proposed in the paper [Evaluating the Robustness of Nearest Neighbor Classifiers: A Primal-Dual Perspective](https://arxiv.org/abs/1906.03972).

Also see the [repo](https://github.com/wangwllu/provably_robust_metric_learning) for our NeurIPS 2020 paper [Provably Robust Metric Learning](https://arxiv.org/abs/2006.07024).

## Implemented algorithms

- **QP-exact**: computes exact minimum adversarial perturbations for 1-NN.
- **QP-top**: computes upper bounds (attack) of minimum adversarial perturbations for 1-NN.
- **QP-verify**: computes lower bounds (verification) of minimum adversarial perturbations for general **K-NN**.
- **QP-greedy**: computes upper bounds (attack) of minimum adversarial perturbations for general **K-NN**.

Moreover, other compared attack algorithms are also implemented.

## Getting started with the code

Our program is tested on Python 3.7.
The required packages are

- numpy
- scikit-learn
- pandas (only used to collect results)
- pytorch (only used to load MNIST and Fashion-MNIST, and the substitute attack)

For example, if you want to run **QP-exact** on the `Letter` dataset,

1. Edit the `dataset` field in `config/exact.ini`;
2. Move `letter.scale` and `letter.scale.t` to the directory appointed by the field `dataset_dir`;
3. Run `python main_exact.py`.

Datasets `Letter`, `Pendigits`, `USPS` and `Satimage` can be downloaded from [LibSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).
`MNIST` and `Fashion-MNIST` should be downloaded by PyTorch.
Other datasets can be easily supported by implementing the abstract class `Loader` in `knn_robustness/utils/loaders`.
