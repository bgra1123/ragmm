import warnings

import numpy as np
import pytest
import torch

from risk_averse_tuning import GMM

torch.set_default_dtype(torch.float64)

Q = torch.tensor([0.3, 1.0, 2.5])
B = torch.tensor([1.0, -2.0, 0.5])


def grad(x):
    return Q * x - B                      # f(x) = 0.5 x^T diag(Q) x - b^T x


def reference_path(alpha, beta, gamma, x0, steps):
    """The paper's form: x_{k+1} = (1+beta) x_k - beta x_{k-1} - alpha grad f(y_k), y_k = (1+gamma)x_k - gamma x_{k-1}."""
    x_prev, x = x0.clone(), x0.clone()
    path = [x.clone()]
    for _ in range(steps):
        y = (1 + gamma) * x - gamma * x_prev
        x_prev, x = x, (1 + beta) * x - beta * x_prev - alpha * grad(y)
        path.append(x.clone())
    return path


CONFIGS = {
    "gd": (0.3, 0.0, 0.0),
    "heavy_ball": (0.2, 0.7, 0.0),
    "nesterov": (0.25, 0.8, 0.8),
    "general": (0.15, 0.6, 0.35),
    "gamma_above_one": (0.1, 0.5, 1.7),      # RA-GMM can select gamma > 1 (Theorem 2, set S_+)
}


def run_closure(alpha, beta, gamma, x0, steps):
    x = x0.clone().requires_grad_(True)
    opt = GMM([x], lr=alpha, beta=beta, gamma=gamma)
    path = [x.detach().clone()]

    def closure():
        opt.zero_grad()
        loss = 0.5 * (Q * x * x).sum() - (B * x).sum()
        loss.backward()
        return loss

    for _ in range(steps):
        opt.step(closure)
        path.append(x.detach().clone())
    return path


@pytest.mark.parametrize("name", CONFIGS)
def test_closure_path_matches_paper_recursion(name):
    x0 = torch.tensor([2.0, -1.0, 3.0])
    got = run_closure(*CONFIGS[name], x0, 25)
    want = reference_path(*CONFIGS[name], x0, 25)
    for g, w in zip(got, want):
        assert torch.allclose(g, w, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("name", CONFIGS)
def test_apply_lookahead_path_matches_closure_path(name):
    alpha, beta, gamma = CONFIGS[name]
    x0 = torch.tensor([2.0, -1.0, 3.0])
    x = x0.clone().requires_grad_(True)
    opt = GMM([x], lr=alpha, beta=beta, gamma=gamma)
    path = [x.detach().clone()]
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # no lookahead warning expected
        for _ in range(25):
            opt.apply_lookahead()
            opt.zero_grad()
            (0.5 * (Q * x * x).sum() - (B * x).sum()).backward()
            opt.step()
            path.append(x.detach().clone())
    for g, w in zip(path, run_closure(alpha, beta, gamma, x0, 25)):
        assert torch.allclose(g, w, atol=1e-12)


def test_step_without_lookahead_warns_when_gamma_nonzero():
    x = torch.tensor([2.0, -1.0, 3.0], requires_grad=True)
    opt = GMM([x], lr=0.2, beta=0.7, gamma=0.5)
    def plain_step():
        opt.zero_grad()
        (0.5 * (Q * x * x).sum() - (B * x).sum()).backward()
        opt.step()
    plain_step()                                             # first step: d_0 = 0, y_0 = x_0, no warning needed
    with pytest.warns(RuntimeWarning, match="lookahead"):
        plain_step()


def test_no_warning_when_gamma_zero():
    x = torch.tensor([2.0, -1.0, 3.0], requires_grad=True)
    opt = GMM([x], lr=0.2, beta=0.7, gamma=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(3):
            opt.zero_grad()
            (0.5 * (Q * x * x).sum()).backward()
            opt.step()


def test_x_k_is_restored_for_a_parameter_that_loses_its_gradient():
    """A parameter with a momentum buffer is shifted to y_k by the lookahead; it must come back even if
    the closure produces no gradient for it (old code skipped the restore and left it at y_k)."""
    a = torch.tensor([1.0, 2.0], requires_grad=True)
    b = torch.tensor([3.0], requires_grad=True)
    opt = GMM([a, b], lr=0.1, beta=0.5, gamma=0.5)

    def both():
        opt.zero_grad()
        loss = (a ** 2).sum() + (b ** 2).sum()
        loss.backward()
        return loss

    def only_a():
        opt.zero_grad(set_to_none=True)                 # b receives no gradient this step
        loss = (a ** 2).sum()
        loss.backward()
        return loss

    opt.step(both)                                      # b now has a momentum buffer
    b_after_first = b.detach().clone()
    opt.step(only_a)
    assert torch.equal(b.detach(), b_after_first)       # not left at y_k = x_k + gamma d_k
    assert all('x_k_backup' not in st for st in opt.state.values())


def test_double_apply_lookahead_is_rejected():
    x = torch.tensor([1.0], requires_grad=True)
    opt = GMM([x], lr=0.1, beta=0.5, gamma=0.5)
    x.grad = torch.ones(1)
    opt.step()                                               # creates a momentum buffer
    opt.apply_lookahead()
    with pytest.raises(RuntimeError, match="already"):
        opt.apply_lookahead()


@pytest.mark.parametrize("kwargs", [dict(lr=0.0), dict(lr=-1.0), dict(beta=-0.1), dict(gamma=-0.1)])
def test_invalid_hyperparameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GMM([torch.zeros(1, requires_grad=True)], **kwargs)


# ---------------------------------------------------------------------- #
# Special cases against torch.optim.SGD                                    #
# ---------------------------------------------------------------------- #

def _noisy_quadratic_grad(x, step):
    """A deterministic 'noisy' gradient so the comparison is not only on a smooth path."""
    return Q * x - B + 0.3 * torch.sin(torch.arange(3.0) * (step + 1.0))


def _run_gmm_with_grad_fn(alpha, beta, gamma, x0, steps):
    x = x0.clone().requires_grad_(True)
    opt = GMM([x], lr=alpha, beta=beta, gamma=gamma)
    xs, ys = [x.detach().clone()], [x.detach().clone()]
    for k in range(steps):
        opt.apply_lookahead()                                # gradient at y_k
        x.grad = _noisy_quadratic_grad(x.detach(), k)
        opt.step()
        xs.append(x.detach().clone())
        buf = opt.state[x]['momentum_buffer']
        ys.append(x.detach() + gamma * buf)                  # y_{k+1} = x_{k+1} + gamma d_{k+1}
    return xs, ys


def _run_torch_sgd(alpha, momentum, nesterov, x0, steps):
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.SGD([x], lr=alpha, momentum=momentum, nesterov=nesterov)
    path = [x.detach().clone()]
    for k in range(steps):
        x.grad = _noisy_quadratic_grad(x.detach(), k)
        opt.step()
        path.append(x.detach().clone())
    return path


def test_beta_gamma_zero_is_sgd():
    x0 = torch.tensor([2.0, -1.0, 3.0])
    xs, _ = _run_gmm_with_grad_fn(0.2, 0.0, 0.0, x0, 30)
    for g, w in zip(xs, _run_torch_sgd(0.2, 0.0, False, x0, 30)):
        assert torch.allclose(g, w, atol=1e-13)


def test_gamma_zero_is_sgd_with_heavy_ball_momentum():
    x0 = torch.tensor([2.0, -1.0, 3.0])
    xs, _ = _run_gmm_with_grad_fn(0.2, 0.7, 0.0, x0, 30)
    for g, w in zip(xs, _run_torch_sgd(0.2, 0.7, False, x0, 30)):
        assert torch.allclose(g, w, atol=1e-13)


def test_gamma_equals_beta_is_nesterov():
    """torch's Nesterov SGD tracks the lookahead point y_k, so compare it with y_k = x_k + gamma d_k."""
    x0 = torch.tensor([2.0, -1.0, 3.0])
    _, ys = _run_gmm_with_grad_fn(0.2, 0.7, 0.7, x0, 30)
    for g, w in zip(ys, _run_torch_sgd(0.2, 0.7, True, x0, 30)):
        assert torch.allclose(g, w, atol=1e-12)
