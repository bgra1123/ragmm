import math
import warnings

import pytest
import torch

from risk_averse_tuning import GMM, RAGMM

torch.set_default_dtype(torch.float64)


def quadratic(eigs, x0):
    x = torch.tensor(x0, requires_grad=True)
    q = torch.tensor(eigs)
    return x, q


def make(L=1.0, mu=0.1, d=4, **kw):
    x = torch.linspace(1.0, 2.0, d).requires_grad_(True)
    gmm = GMM([x], lr=1 / (L + mu))
    return x, gmm, RAGMM(gmm, L=L, mu=mu, grid_size=30, **kw)


def test_parameters_are_selected_at_construction_and_applied():
    x, gmm, opt = make(rate_slack=0.25)
    assert opt.dimension == 4
    assert (gmm.param_groups[0]['lr'], gmm.param_groups[0]['beta'], gmm.param_groups[0]['gamma']) == \
        (opt.solution.alpha, opt.solution.beta, opt.solution.gamma)
    assert opt.psi == opt.solution.psi
    assert opt.solution.rate_ratio <= 1.25 + 1e-12


def test_dimension_defaults_to_the_number_of_trainable_parameters():
    a = torch.zeros(3, 2, requires_grad=True)
    frozen = torch.zeros(7)                                      # not a trainable parameter
    b = torch.zeros(4, requires_grad=True)
    b2 = torch.nn.Parameter(torch.zeros(5), requires_grad=False)
    opt = RAGMM(GMM([a, b, b2], lr=0.1), L=1.0, mu=0.1, grid_size=10)
    assert opt.dimension == 10
    assert RAGMM(GMM([a], lr=0.1), L=1.0, mu=0.1, grid_size=10, dimension=123).dimension == 123


def test_closure_evaluates_the_gradient_at_the_lookahead_point():
    x, gmm, opt = make(rate_slack=0.25)
    opt._apply_hyperparams(0.2, 0.6, 0.5)                         # force gamma != 0
    seen = []

    def closure():
        seen.append(x.detach().clone())
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)                                             # first step: d_0 = 0
    x_1 = x.detach().clone()
    d_1 = x_1 - seen[0]
    opt.step(closure)
    assert torch.allclose(seen[1], x_1 + 0.5 * d_1)               # y_1 = x_1 + gamma d_1


def test_x_k_is_restored_after_the_closure_and_nan_recovery_uses_it():
    """Old code backed up the lookahead point y_k instead of x_k, so NaN recovery left the iterate at y_k."""
    x, gmm, opt = make(rate_slack=0.25)
    opt._apply_hyperparams(0.2, 0.6, 0.5)
    def good():
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        return loss
    opt.step(good)
    opt.step(good)
    x_k = x.detach().clone()
    assert gmm.state[x]['momentum_buffer'].abs().sum() > 0

    def bad():
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        x.grad.fill_(float('inf'))
        return loss

    stats = opt.step(bad)
    assert stats['nan_detected']
    assert torch.equal(x.detach(), x_k)                           # x_k, not y_k
    assert gmm.state[x]['momentum_buffer'].abs().sum() == 0
    assert 'x_k_backup' not in gmm.state[x]


def test_stats_are_reported_consistently():
    x, gmm, opt = make(rate_slack=0.25)
    def closure():
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        return loss
    first = opt.step(closure)
    second = opt.step(closure)
    assert first['tuned'] and not second['tuned']
    assert first['psi'] == second['psi'] == opt.solution.psi       # old code reported psi = 0.0 after step 1
    assert first['evar_bound'] == second['evar_bound'] == opt.solution.evar_bound
    assert first['step'] == 1 and second['step'] == 2
    assert {'alpha', 'beta', 'gamma', 'sigma_hat', 'nan_detected', 'loss'} <= set(first)


def test_step_without_closure_works_with_manual_lookahead():
    x, gmm, opt = make(rate_slack=0.25)
    x2 = x.detach().clone().requires_grad_(True)
    opt2 = RAGMM(GMM([x2], lr=0.1), L=1.0, mu=0.1, grid_size=30, rate_slack=0.25)
    def closure():
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        return loss
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(6):
            opt.step(closure)
            opt2.apply_lookahead()
            opt2.zero_grad()
            (x2 ** 2).sum().backward()
            opt2.step()
    assert torch.allclose(x, x2)


def test_noiseless_convergence_at_the_selected_parameters():
    """Theorem 2: at the selected parameters the noiseless quadratic converges at rate <= rho^2 per step."""
    L, mu = 1.0, 0.1
    eigs = torch.linspace(mu, L, 8)
    x = torch.ones(8).requires_grad_(True)
    opt = RAGMM(GMM([x], lr=0.5), L=L, mu=mu, grid_size=30, rate_slack=0.25)
    f0 = 0.5 * (eigs * x.detach() ** 2).sum().item()
    def closure():
        opt.zero_grad()
        loss = 0.5 * (eigs * x ** 2).sum()
        loss.backward()
        return loss
    for _ in range(150):
        opt.step(closure)
    f = 0.5 * (eigs * x.detach() ** 2).sum().item()
    assert f < 1e-8 * f0


@pytest.mark.parametrize("kwargs", [dict(alpha_range=(1e-6, 0.1)), dict(psi_range=(1e-5, 0.95)), dict(eps=1e-8)])
def test_removed_arguments_are_ignored_with_a_deprecation_warning(kwargs):
    with pytest.warns(DeprecationWarning):
        RAGMM(GMM([torch.zeros(2, requires_grad=True)], lr=0.1), L=1.0, mu=0.1, grid_size=10, **kwargs)


def test_unknown_arguments_and_non_gmm_optimizers_are_rejected():
    p = torch.zeros(2, requires_grad=True)
    with pytest.raises(TypeError):
        RAGMM(GMM([p], lr=0.1), L=1.0, mu=0.1, bogus=1)
    with pytest.raises(TypeError):
        RAGMM(torch.optim.SGD([p], lr=0.1), L=1.0, mu=0.1)


def test_tuning_arguments_are_keyword_only():
    """v0.1.0 took grid_size, eps, ... positionally; silently reinterpreting them would be worse than failing."""
    p = torch.zeros(2, requires_grad=True)
    with pytest.raises(TypeError):
        RAGMM(GMM([p], lr=0.1), 1.0, 0.1, 0.05, 10)


def test_ra_agd_wrapper_sets_gamma_equal_to_beta():
    x = torch.linspace(1.0, 2.0, 4).requires_grad_(True)
    gmm = GMM([x], lr=0.1)
    opt = RAGMM(gmm, L=1.0, mu=0.1, grid_size=30, rate_slack=0.25, method="agd")
    assert opt.method == "agd" and opt.solution.region == "S_0"
    assert opt.gamma == opt.beta
    assert opt.beta == pytest.approx((1 - math.sqrt(opt.alpha * 0.1)) / (1 + math.sqrt(opt.alpha * 0.1)))
