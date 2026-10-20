# Risk-Averse Tuning for Momentum Methods

PyTorch implementation of the **RA-GMM** (Risk-Averse Generalized Momentum Method) optimizer from:

> **Entropic Risk-Averse Generalized Momentum Methods**
> Bugra Can, Mert Gürbüzbalaban
> *Optimization Methods and Software*, 40(6), 1535–1583, 2025
> https://doi.org/10.1080/10556788.2025.2549356

## Quick start: your own table

Version 0.3.0 adds a NumPy/pandas API for tabular yes/no decisions (PyTorch is only needed for the optimizers below):

```bash
pip install .
ragmm applications.csv --target repaid --positive yes --drop applicant_id --report tail_report.md
```

This trains an L2 logistic regression with the paper's RA-GMM settings (`--method sgd` or `careful` for the two
baselines) and, with `--report`, retrains each method 200 times on a held-out split and reports how many decisions
change between training runs, which rows are affected, and which rows are at risk (found from one fit). In Python:
`RiskAverseClassifier`, `TabularEncoder`, `tail_report`. The story and evidence behind it:
[`research/report/REPORT.md`](research/report/REPORT.md).

## Stability adapter (new in 0.4.0)

Retrain a model with a different random seed and some of its decisions change. The adapter measures this and reduces it for
a model you already have:

```python
from risk_averse_tuning import StabilityAdapter, stability_check
from xgboost import XGBClassifier

model = StabilityAdapter(XGBClassifier(subsample=0.8), n_seeds=10).fit(X_train, y_train)   # averages 10 seeds
report = stability_check(lambda s: XGBClassifier(subsample=0.8, random_state=s), X_train, y_train, X_test, y_test,
                         seeds=30, groups={"housing for free": mask})                       # how much do decisions move?
```

For logistic models (`SGDClassifier(loss="log_loss")`, `LogisticRegression`, `RiskAverseClassifier`, a network's last layer)
the adapter instead polishes the trained weights with deterministic full-batch steps; `step="entropic"` uses a risk-averse
line search based on the entropic risk measure of the paper. Data that does not fit in memory: `polish_logistic_streaming` / `StabilityAdapter(batch_size=...)` compute the step search from batch sums. Command line: `ragmm-adapter data.csv --target y --model xgboost`.
Benchmark (pre-registered, German Credit): disagreement between retrains SGD logistic 15.5% → 0.1%, XGBoost 6.5% → 2.1%,
MLP 14.4% → 5.1%, with no loss of accuracy; for tree models 10× more trees is an equally good fix
([`research/product/RESULTS.md`](research/product/RESULTS.md)).

* Tutorial: [`docs/TUTORIAL.md`](docs/TUTORIAL.md) · runnable: [`examples/tutorial.py`](examples/tutorial.py)
* The story: [`research/blog/BLOG.md`](research/blog/BLOG.md) · videos and shorts: [`content/`](content/README.md)

## Overview

The Generalized Momentum Method (GMM) covers gradient descent (GD), Nesterov's accelerated gradient (AGD) and the heavy-ball (HB) method through three parameters: step size $\alpha$, momentum $\beta$ and lookahead $\gamma$.

$$y_k = x_k + \gamma \, d_k, \qquad d_{k+1} = \beta \, d_k - \alpha \, \nabla f(y_k), \qquad x_{k+1} = x_k + d_{k+1}$$

With noisy gradients, the standard choices of $(\alpha, \beta, \gamma)$ trade a fast rate for a heavy tail of bad outcomes. **RA-GMM** picks the parameters by solving the paper's problem (51): it minimizes the entropic value at risk (EV@R) bound of Theorem 3 over the stable parametrization $(\vartheta, \psi)$ of Theorem 2, subject to a convergence-rate constraint

$$\min_{(\vartheta,\psi)\in S_c\cup S_0}\ \bar{E}_{1-\zeta}(\vartheta,\psi) \quad \text{s.t.}\quad \rho^2_{\vartheta,\psi}/\rho_*^2 \le 1+\varepsilon, \qquad \rho_*^2 = 1-\sqrt{\mu/L},$$

with $\alpha = \frac{1-\vartheta}{L(1-\psi)}$, $\beta = \frac{1-\sqrt{\vartheta\alpha\mu}}{1-\alpha\psi\mu}\bigl(1-\sqrt{\alpha\mu/\vartheta}\bigr)$, $\gamma = \psi\beta$ and $\rho^2 = 1-\sqrt{\vartheta\alpha\mu}$. The choice depends only on $(L, \mu, d, \zeta, \varepsilon)$ and is made once, when the wrapper is created.

Two special cases are available for comparison with standard methods: `method="agd"` gives the paper's **RA-AGD** (problem (51) restricted to $\gamma = \beta$, i.e. Nesterov's method with a risk-averse step size), and `RAGMMBounds.gd_evar_bound` evaluates the paper's Theorem 4 bound for plain gradient descent ($\beta = \gamma = 0$). The paper gives no selection problem for GD in the strongly convex case, so there is no "RA-GD". `GMM` itself reduces exactly to `torch.optim.SGD` ($\beta=\gamma=0$), SGD with momentum ($\gamma=0$) and Nesterov SGD ($\gamma=\beta$, in terms of the lookahead point $y_k$); this is tested.

**Assumptions of the theory:** a strongly convex, smooth objective with known constants $L > \mu > 0$, and unbiased gradients with additive noise.

## Installation

```bash
pip install -e .                 # tabular API and CLI (numpy, pandas)
pip install -e ".[torch]"        # plus the PyTorch optimizers GMM / RAGMM
pip install -e ".[boosting]"     # plus scikit-learn, XGBoost, LightGBM for the stability adapter
pip install -e ".[dev]" && pytest   # optional: run the tests
```

## Usage

```python
import torch
from risk_averse_tuning import GMM, RAGMM

# L = gradient-Lipschitz constant, mu = strong-convexity constant (L > mu > 0)
L, mu = 10.0, 0.1

gmm = GMM(model.parameters(), lr=1 / (L + mu))
opt = RAGMM(gmm, L=L, mu=mu, zeta=0.05, rate_slack=0.05)   # alpha, beta, gamma are chosen here

def closure():
    opt.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    return loss

for x, y in dataloader:
    stats = opt.step(closure)       # the gradient is evaluated at y_k = x_k + gamma d_k
```

**Pass a closure.** When $\gamma \ne 0$ the gradient must be taken at the lookahead point $y_k$; `step(closure)` moves there, calls the closure, restores $x_k$ and updates. `step()` without a closure evaluates the gradient at $x_k$ and issues a `RuntimeWarning`; to use it correctly, call `opt.apply_lookahead()` before the forward/backward pass.

`opt.step(closure)` returns a diagnostics dict: `alpha`, `beta`, `gamma`, `psi`, `evar_bound`, `sigma_hat`, `step`, `tuned`, `nan_detected` and `loss`. `opt.solution` holds the full solution of (51) (`vartheta`, `psi`, `rate`, `rate_ratio`, `region`).

## Components

| Module | Description |
|---|---|
| `optimizer.py` | `GMM`: PyTorch `Optimizer` implementing the recursion above. `RAGMM`: wrapper that sets `GMM`'s parameters by solving problem (51). |
| `risk.py` | `RAGMMBounds`: stable sets (Theorem 2), EV@R bound (Proposition 3, Theorem 3) and the grid search for (51). |

## RAGMM parameters

| Parameter | Default | Description |
|---|---|---|
| `L` | required | Gradient-Lipschitz constant $L$ |
| `mu` | required | Strong-convexity constant $\mu$, with $L > \mu > 0$ |
| `zeta` | `0.05` | Tail probability: $P\{f(x_k)-f^\star \ge \mathrm{EV@R}_{1-\zeta}\} \le \zeta$. Small is demanding (see below) |
| `rate_slack` | `0.05` | The trade-off $\varepsilon$ in (51b). The paper uses 0.25 (quadratic example) and 0.05 (logistic regression) |
| `grid_size` | `100` | Bins per axis of the $(\vartheta,\psi)$ grid, as in the authors' reference code |
| `phi` | `0.99` | The constant $\varphi \in (0,1)$ of Theorem 3 |
| `dimension` | number of trainable parameters | Dimension $d$ of the optimization variable |
| `method` | `"gmm"` | `"gmm"` solves (51); `"agd"` is RA-AGD ($\gamma = \beta$, search over $\alpha \in (0, 1/L]$) |

## Conventions worth knowing

* **`zeta` is a tail probability.** It enters the bounds only through $\log(1/\zeta)$. The paper's experiments pass `zeta = 0.95` (and describe it as a "confidence level"), which is a very weak guarantee ($\log(1/0.95) \approx 0.05$); `zeta = 0.05` is a 95% guarantee.
* **Bounds are stated for noise variance $\sigma^2 = 1$** and scale linearly in $\sigma^2$. The minimizer of (51) does not depend on $\sigma$.
* **`rate_slack` can be vacuous.** Since $\rho^2 < 1$, the constraint (51b) does nothing when $1+\varepsilon \ge 1/\rho_*^2$ (for $\kappa = 100$, when $\varepsilon \ge 0.11$). The minimizer then drifts toward ever smaller step sizes and depends on `grid_size`. Check `opt.solution.rate_ratio`.
* **Keep `grid_size >= 100`.** The feasible band of the set $S_+$ ($\psi > 1$) is narrow; a coarse grid can miss it.
* **$\gamma$ may exceed 1** (the set $S_+$). This is allowed by Theorem 2.
* **The strongly-convex EV@R bounds (Theorem 3, problem (51)) are conservative; for quadratic objectives use `risk_averse_tuning.quadratic` (paper Section 3: exact EV@R, Theorem 1, problem (32)).** On the noisy-quadratic example of `paper_experiments/audit_reference_code.py` (Part A) the bound is 165–817x the exact EV@R: valid, but conservative.

## Changes in 0.2.0

Fixes so that the code follows the paper (each is covered by a test):

* the rate constraint (51b) now uses the paper's fixed $\rho_*^2 = 1-\sqrt{\mu/L}$ instead of an $\alpha$-dependent benchmark that made the constraint vacuous for small $\alpha$ (v0.1.0 could select $\alpha \approx 10^{-6}$ and barely train);
* $\beta$ uses $1-\alpha\psi\mu$ in the denominator (eq. 35b), not $1-\alpha\psi L$;
* the search runs over the stable sets $S_c \cup S_0$ of Theorem 2, including $\psi > 1$, and no longer rejects $\gamma \ge 1$; the `psi_range`, `alpha_range` and `eps` arguments are removed (ignored with a `DeprecationWarning`), and the tuning arguments after `zeta` are keyword-only;
* the EV@R bound takes $\varphi$ as a parameter (`phi`) and requires the dimension instead of silently using $d = 1$;
* parameters are chosen at construction; `psi` is reported on every step;
* `GMM`: the iterate $x_k$ is always restored after a lookahead (also for parameters without a gradient); `apply_lookahead()` twice is an error; `step()` without lookahead and with $\gamma \ne 0$ warns; hyperparameters are validated;
* `RAGMM.step`: NaN recovery restores $x_k$ (v0.1.0 restored the lookahead point $y_k$); `step(closure)` returns the loss in `stats`;
* `pyproject.toml`: build backend fixed (`setuptools.build_meta`; the old value made `pip install -e .` fail).

## Tests and audit

`pytest` runs the test-suite, including a formula-independent check that the Theorem 3 bound is never below the exact EV@R of a noisy quadratic. `paper_experiments/AUDIT.md` documents differences found between the paper's formulas and the code in `paper_experiments/EntropicsFOMs`.

## Reference

```bibtex
@article{can2025entropic,
  title   = {Entropic risk-averse generalized momentum methods},
  author  = {Can, Bugra and G\"{u}rb\"{u}zbalaban, Mert},
  journal = {Optimization Methods and Software},
  volume  = {40},
  number  = {6},
  pages   = {1535--1583},
  year    = {2025},
  doi     = {10.1080/10556788.2025.2549356}
}
```
