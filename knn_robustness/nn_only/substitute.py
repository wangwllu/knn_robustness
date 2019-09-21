import torch
import torch.nn as nn
import torch.nn.functional as F

from ..knn import Attack


def squared_distance(X, Y):
    out = -2 * torch.matmul(X, Y.t())
    out += (X**2).sum(dim=-1, keepdim=True)
    out += (Y**2).sum(dim=-1, keepdim=True).t()
    return out


class DiffNN(nn.Module):
    def __init__(self, X_train, y_train, smoothness=1., eps=1e-8):
        super().__init__()
        self._X_train = X_train
        self._y_train = y_train
        self._smoothness = smoothness
        self._eps = eps

    def forward(self, X):
        D = squared_distance(X, self._X_train)
        logits = - D * (self._smoothness ** 2)
        w = F.softmax(logits, dim=1)
        p_list = []
        for i in range(self._y_train.max()+1):
            p_list.append(w[:, self._y_train == i].sum(dim=1))
        return torch.stack(p_list, dim=1).clamp(min=self._eps).log()


class SubstituteAttack(Attack):
    def __init__(
            self, X_train, y_train,
            step_size, max_iter,
            lower=0., upper=1.
    ):
        super().__init__(X_train, y_train, n_neighbors=1)
        self._step_size = step_size
        self._max_iter = max_iter
        self._lower = lower
        self._upper = upper
        self._substitute = DiffNN(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        )

    def __call__(self, x_eval):
        y_eval = self.predict_individual(x_eval)

        x_eval = torch.from_numpy(x_eval)
        y_eval = torch.tensor(y_eval)

        perturbation = torch.zeros_like(x_eval, requires_grad=True)

        criterion = nn.NLLLoss()
        for _ in range(self._max_iter):
            loss = criterion(
                self._substitute(
                    (x_eval+perturbation).unsqueeze(0)),
                y_eval.unsqueeze(0)
            )
            loss.backward()
            perturbation.data = (
                perturbation.data
                + self._step_size
                * perturbation.grad.data
                / torch.norm(perturbation.grad.data, p=2, keepdim=True)
            ).min(self._upper - x_eval).max(self._lower - x_eval)
            perturbation.grad.zero_()

            if self.predict_individual(
                (x_eval+perturbation).detach().numpy()
            ) != y_eval.item():
                return perturbation.detach().numpy()

        return None
