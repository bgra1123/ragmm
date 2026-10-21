"""
Stability adapter: make a trained classifier give the same answer when it is retrained with a different random seed.

Two strategies, chosen by what causes the variation (evidence: research/adapter/DESIGN.md, research/sota2/RESULTS.md):

* ``"polish"`` - for logistic / linear heads (``SGDClassifier(loss="log_loss")``, ``LogisticRegression``,
  ``RiskAverseClassifier``, or the last layer of a network given its features). The objective is strongly convex, so
  the seed only changes where the optimizer stopped. A few deterministic full-batch steps from the trained weights move
  every run to (nearly) the same point. Step rules:
    - ``"line_search"``: exact line search along the full gradient;
    - ``"entropic"``: the risk-averse line search  alpha* = argmin_a (1/theta) log mean_i exp(-theta a g^T grad f_i(w)),
      i.e. the entropic risk of the first-order per-row loss change (research/linesearch);
    - ``"entropic_exact"``: the same with the exact per-row loss change, which protects the rows a step makes worse;
    - ``"newton"``: solve the head exactly (Newton with backtracking).
* ``"average"`` - for models whose structure depends on the seed (XGBoost / LightGBM with subsampling, random forests,
  neural networks): average the predicted probabilities of ``n_seeds`` copies trained with different seeds.

``calibrate=True`` adds out-of-fold Platt scaling (needed when decisions use a cost-based cutoff, not 0.5).

``stability_check`` retrains any model many times and measures how much its decisions depend on the seed, overall and
by group; ``compare_stability`` puts several such results side by side (e.g. without and with the adapter).

The theory behind the risk-averse choices is in Can & Gurbuzbalaban, "Entropic Risk-Averse Generalized Momentum
Methods", Optimization Methods and Software 40(6), 2025 (arXiv:2204.11292).
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from .classifier import sigmoid

STEPS = ("line_search", "entropic", "entropic_exact", "newton")
STRATEGIES = ("auto", "polish", "average")
DISCLAIMER = ("These numbers describe how much a model's decisions vary between training runs, overall and by group. "
              "They are a statistical diagnostic, not a finding of bias or wrongdoing, and not legal, credit or "
              "compliance advice.")


# ---------------------------------------------------------------------------------------------------------------------
# polishing a logistic head


def _lme(z):
    m = z.max()
    return float(m + math.log(np.mean(np.exp(z - m))))


def _minimize_1d(phi, hi, lo=None, grid=60, refine=30):
    """Minimize a scalar function on (0, hi]: geometric grid, then golden-section in the bracket of the best point."""
    lo = lo if lo is not None else hi * 1e-6
    a = np.geomspace(lo, hi, grid)
    v = np.array([phi(x) for x in a])
    i = int(np.argmin(v))
    left, right = a[max(i - 1, 0)], a[min(i + 1, grid - 1)]
    for _ in range(refine):
        c, d = left + 0.382 * (right - left), left + 0.618 * (right - left)
        if phi(c) < phi(d):
            right = d
        else:
            left = c
    x = 0.5 * (left + right)
    return x if phi(x) <= v[i] else float(a[i])


def logistic_objective(w, X, y, lam, mask=None):
    """(1/n) sum log(1 + exp(-y x^T w)) + (lam/2) ||mask * w||^2, y in {+1, -1}."""
    m = np.ones_like(w) if mask is None else mask
    return float(np.logaddexp(0.0, -y * (X @ w)).mean() + 0.5 * lam * np.sum((m * w) ** 2))


def polish_logistic(w0, X, y, lam, epochs=20, step="line_search", theta=1.0, mask=None, tol=1e-8):
    """
    Deterministic polish of a logistic head from the starting weights ``w0``.

    X : (n, d) features (include an intercept column yourself), y : labels in {+1, -1}, lam : L2 strength (> 0 makes
    the solution unique), mask : 1 for penalized coordinates, 0 for unpenalized (e.g. the intercept).
    Each epoch is one full pass over the data (one gradient) plus a 1-D step search. Returns (w, info).
    """
    if step not in STEPS:
        raise ValueError(f"step must be one of {STEPS}, got {step!r}")
    X, y = np.asarray(X, float), np.asarray(y, float)
    w = np.asarray(w0, float).copy()
    m = np.ones_like(w) if mask is None else np.asarray(mask, float)
    n = len(y)
    f0 = logistic_objective(w, X, y, lam, m)
    L = float(np.linalg.eigvalsh(X.T @ X / n).max() / 4.0 + lam)
    a_max = 1.0 / lam if lam > 0 else 1e3 / L
    if step == "newton":
        used = 0
        for used in range(1, 101):
            s = sigmoid(-y * (X @ w))
            g = -(X.T @ (y * s)) / n + lam * m * w
            if np.linalg.norm(g) < tol:
                break
            H = (X.T * (s * (1 - s))) @ X / n + lam * np.diag(m) + 1e-12 * np.eye(len(w))
            d = np.linalg.solve(H, g)
            t, fw = 1.0, logistic_objective(w, X, y, lam, m)
            while logistic_objective(w - t * d, X, y, lam, m) > fw - 0.25 * t * g @ d and t > 1e-12:
                t *= 0.5
            w = w - t * d
        return w, dict(step=step, epochs=used, objective_before=f0, objective_after=logistic_objective(w, X, y, lam, m))
    alphas = []
    used = 0
    for used in range(1, epochs + 1):
        M = y * (X @ w)
        s = sigmoid(-M)
        g = -(X.T @ (y * s)) / n + lam * m * w
        if np.linalg.norm(g) < tol:
            used -= 1
            break
        yXg = y * (X @ g)
        reg = lambda a: 0.5 * lam * np.sum((m * (w - a * g)) ** 2)  # noqa: E731
        if step == "line_search":
            phi = lambda a: float(np.logaddexp(0.0, -(M - a * yXg)).mean() + reg(a))  # noqa: E731
        elif step == "entropic":
            # a_i = g^T grad f_i(w): first-order decrease of row i's loss per unit step
            a_i = -s * yXg + lam * np.sum(m * w * g)
            phi = lambda a: _lme(-theta * a * a_i) / theta  # noqa: E731
        else:
            f_old = np.logaddexp(0.0, -M)
            pen_old = 0.5 * lam * np.sum((m * w) ** 2)
            phi = lambda a: _lme(theta * (np.logaddexp(0.0, -(M - a * yXg)) - f_old + reg(a) - pen_old)) / theta  # noqa: E731
        a = _minimize_1d(phi, a_max)
        w = w - a * g
        alphas.append(a)
    return w, dict(step=step, epochs=used, objective_before=f0, objective_after=logistic_objective(w, X, y, lam, m),
                   mean_step_times_L=float(np.mean(alphas) * L) if alphas else 0.0)


def iter_batches(X, y, batch_size):
    """Re-iterable batch source for ``polish_logistic_streaming``: ``batches = iter_batches(X, y, 1000)``."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    return lambda: ((X[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(y), batch_size))


def _lse_merge(m, s, z):
    """Merge a batch of terms z (shape (G, b)) into running (max, sum of exp(z - max)) per row of z."""
    bm = z.max(1)
    nm = np.maximum(m, bm)
    s = s * np.exp(m - nm) + np.exp(z - nm[:, None]).sum(1)
    return nm, s


def _newton_1d(h1, h2, lo, hi, iters=60, tol=1e-12):
    """Minimize a convex scalar function on [lo, hi] given its first and second derivatives (safeguarded Newton)."""
    if h1(lo) >= 0:
        return lo
    if h1(hi) <= 0:
        return hi
    a, b = lo, hi
    x = float(np.sqrt(lo * hi))
    for _ in range(iters):
        d1 = h1(x)
        if d1 < 0:
            a = x
        else:
            b = x
        d2 = h2(x)
        xn = x - d1 / d2 if d2 > 0 else 0.5 * (a + b)
        if not (a < xn < b):
            xn = 0.5 * (a + b)
        if abs(xn - x) <= tol * max(x, 1e-30):
            return xn
        x = xn
    return x


def _newton_streaming(w0, batches, lam, max_iter, mask, tol):
    """Damped Newton on the L2 logistic objective from batch sums. Per iteration: one pass for the gradient, the Hessian
    (d x d, memory O(d^2)) and the objective, one pass for the objective at a grid of trial step lengths (Armijo backtracking)."""
    w = np.asarray(w0, float).copy()
    m = np.ones_like(w) if mask is None else np.asarray(mask, float)
    ts = 0.5 ** np.arange(0, 12)
    passes, used = 0, 0
    f_first = None
    for used in range(1, max_iter + 1):
        gs, Hs, fs, n = np.zeros_like(w), np.zeros((len(w), len(w))), 0.0, 0
        for Xb, yb in batches():
            Mb = yb * (Xb @ w)
            s = sigmoid(-Mb)
            gs += -(Xb.T @ (yb * s))
            Hs += (Xb.T * (s * (1 - s))) @ Xb
            fs += float(np.logaddexp(0.0, -Mb).sum())
            n += len(yb)
        passes += 1
        pen = 0.5 * lam * np.sum((m * w) ** 2)
        f0 = fs / n + pen
        if f_first is None:
            f_first = f0
        g = gs / n + lam * m * w
        if np.linalg.norm(g) < tol:
            used -= 1
            break
        H = Hs / n + lam * np.diag(m) + 1e-12 * np.eye(len(w))
        p = np.linalg.solve(H, g)
        gp = float(g @ p)
        S = np.zeros(len(ts))
        for Xb, yb in batches():
            Mb = yb * (Xb @ w)
            Cb = yb * (Xb @ p)
            S += np.logaddexp(0.0, -(Mb[None, :] - ts[:, None] * Cb[None, :])).sum(1)
        passes += 1
        reg = 0.5 * lam * np.array([np.sum((m * (w - t * p)) ** 2) for t in ts])
        ok = np.flatnonzero(S / n + reg <= f0 - 0.25 * ts * gp)
        t = float(ts[ok[0]]) if len(ok) else float(ts[-1])
        w = w - t * p
    return w, dict(step="newton", search="exact", epochs=used, iterations=used, passes=passes, objective_before=f_first,
                   objective_after=None)


def polish_logistic_streaming(w0, batches, lam, epochs=20, step="line_search", theta=1.0, mask=None, search="newton",
                              grid=120, tol=1e-8):
    """
    ``polish_logistic`` for data that is read in batches (so X never has to be in memory).

    batches : callable returning a fresh iterable of (X_batch, y_batch) each time it is called (see ``iter_batches``);
              rows are features including the intercept column; labels in {+1, -1}.
    Each epoch makes two passes: (1) the full gradient g, (2) the quantities the step rule needs along g.
    search="grid": accumulate sums on a geometric grid of ``grid`` step lengths in one pass (memory O(grid)), then refine
    by a parabola; search="newton": keep three length-n vectors (margin, margin change along g, directional derivative)
    and solve the convex 1-D problem by safeguarded Newton with no further pass over the data (memory O(n)).
    step: "line_search", "entropic" (the entropic risk-averse line search) or "entropic_exact". Returns (w, info).
    """
    if step not in STEPS:
        raise ValueError(f"step must be one of {STEPS}, got {step!r}")
    if search not in ("grid", "newton"):
        raise ValueError("search must be 'grid' or 'newton'")
    if step == "newton":
        return _newton_streaming(w0, batches, lam, epochs, mask, tol)
    w = np.asarray(w0, float).copy()
    m = np.ones_like(w) if mask is None else np.asarray(mask, float)
    alphas_used, passes = [], 0
    n = None

    def count_and_grad(w):
        gs, cnt, obj = np.zeros_like(w), 0, 0.0
        for Xb, yb in batches():
            Mb = yb * (Xb @ w)
            gs += -(Xb.T @ (yb * sigmoid(-Mb)))
            cnt += len(yb)
            obj += float(np.logaddexp(0.0, -Mb).sum())
        return gs, cnt, obj

    gs, n, obj0 = count_and_grad(w)
    passes += 1
    f0 = obj0 / n + 0.5 * lam * np.sum((m * w) ** 2)
    used = 0
    for used in range(1, epochs + 1):
        if used > 1:
            gs, _, _ = count_and_grad(w)
            passes += 1
        g = gs / n + lam * m * w
        if np.linalg.norm(g) < tol:
            used -= 1
            break
        pen0 = 0.5 * lam * np.sum((m * w) ** 2)
        r1 = -lam * np.sum(m * w * g)
        r2 = 0.5 * lam * np.sum(m * g * g)
        reg = lambda a: pen0 + r1 * a + r2 * a * a   # noqa: E731  (value of the penalty at w - a g)
        dreg = lambda a: r1 + 2 * r2 * a             # noqa: E731
        ddreg = 2 * r2
        wg = lam * np.sum(m * w * g)
        a_max = 1.0 / lam if lam > 0 else 1e3
        a_min = a_max * 1e-6
        alphas = np.geomspace(a_min, a_max, grid)
        if search == "grid":
            S = np.zeros(grid)
            lm, ls = np.full(grid, -np.inf), np.zeros(grid)
        else:
            Ms, Cs = [], []
        for Xb, yb in batches():
            Mb = yb * (Xb @ w)
            Cb = yb * (Xb @ g)
            if search == "newton":
                Ms.append(Mb)
                Cs.append(Cb)
                continue
            if step == "line_search":
                S += np.logaddexp(0.0, -(Mb[None, :] - alphas[:, None] * Cb[None, :])).sum(1)
            elif step == "entropic":
                ab = -sigmoid(-Mb) * Cb + wg
                lm, ls = _lse_merge(lm, ls, -theta * alphas[:, None] * ab[None, :])
            else:
                du = np.logaddexp(0.0, -(Mb[None, :] - alphas[:, None] * Cb[None, :])) - np.logaddexp(0.0, -Mb)[None, :]
                lm, ls = _lse_merge(lm, ls, theta * du)
        passes += 1
        if search == "grid":
            if step == "line_search":
                h = S / n + np.array([reg(a) for a in alphas])
            elif step == "entropic":
                h = (lm + np.log(ls) - np.log(n)) / theta
            else:
                h = (lm + np.log(ls) - np.log(n)) / theta + np.array([reg(a) for a in alphas]) - pen0
            k = int(np.argmin(h))
            alpha = float(alphas[k])
            if 0 < k < grid - 1:
                x0, x1, x2 = alphas[k - 1:k + 2]
                y0, y1, y2 = h[k - 1:k + 2]
                den = (x0 - x1) * (x0 - x2) * (x1 - x2)
                A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
                B = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / den
                if A > 0:
                    alpha = float(np.clip(-B / (2 * A), x0, x2))
        else:
            M, C = np.concatenate(Ms), np.concatenate(Cs)
            if step == "line_search":
                h1 = lambda a: float(np.mean(sigmoid(-(M - a * C)) * C) + dreg(a))                      # noqa: E731
                h2 = lambda a: float(np.mean(sigmoid(-(M - a * C)) * sigmoid(M - a * C) * C * C) + ddreg)    # noqa: E731
            elif step == "entropic":
                A_ = -sigmoid(-M) * C + wg

                def _p(a):
                    z = -theta * a * A_
                    p = np.exp(z - z.max())
                    return p / p.sum()
                h1 = lambda a: float(-(_p(a) @ A_))                                                 # noqa: E731
                h2 = lambda a: float(theta * (lambda p: p @ A_ ** 2 - (p @ A_) ** 2)(_p(a)))         # noqa: E731
            else:
                f_old = np.logaddexp(0.0, -M)

                def _q(a):
                    z = theta * (np.logaddexp(0.0, -(M - a * C)) - f_old)
                    p = np.exp(z - z.max())
                    return p / p.sum()
                h1 = lambda a: float(_q(a) @ (sigmoid(-(M - a * C)) * C) + dreg(a))                      # noqa: E731

                def h2(a):
                    p = _q(a)
                    u1 = sigmoid(-(M - a * C)) * C
                    u2 = sigmoid(-(M - a * C)) * sigmoid(M - a * C) * C * C
                    return float(p @ u2 + theta * (p @ u1 ** 2 - (p @ u1) ** 2) + ddreg)
            alpha = _newton_1d(h1, h2, a_min, a_max)
        w = w - alpha * g
        alphas_used.append(alpha)
    _, _, objf = count_and_grad(w)
    passes += 1
    return w, dict(step=step, search=search, epochs=used, passes=passes, objective_before=f0,
                   objective_after=objf / n + 0.5 * lam * np.sum((m * w) ** 2),
                   mean_alpha=float(np.mean(alphas_used)) if alphas_used else 0.0)


def platt_fit(p, y01):
    """1-D logistic calibration of probabilities p on 0/1 labels. Returns (a, b): p_cal = sigmoid(a logit(p) + b)."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    F = np.column_stack([z, np.ones_like(z)])
    w, _ = polish_logistic(np.array([1.0, 0.0]), F, 2.0 * np.asarray(y01, float) - 1.0, 1e-4, step="newton")
    return float(w[0]), float(w[1])


def platt_apply(p, ab):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return sigmoid(ab[0] * z + ab[1])


# ---------------------------------------------------------------------------------------------------------------------
# the adapter around an estimator


def _set_seed(est, seed, seed_param=None):
    est = _clone(est)
    if seed_param:
        est.set_params(**{seed_param: seed})
    elif hasattr(est, "get_params"):
        params = est.get_params()
        for k in ("random_state", "seed"):
            if k in params:
                est.set_params(**{k: seed})
                break
    elif hasattr(est, "seed"):
        est.seed = seed
    return est


def _clone(est):
    try:
        from sklearn.base import clone
        return clone(est)
    except Exception:
        return copy.deepcopy(est)


def _is_logistic(est):
    name = type(est).__name__
    if name == "RiskAverseClassifier":
        return True
    if name == "LogisticRegression":
        return getattr(est, "penalty", "l2") in ("l2", None, "none")
    if name == "SGDClassifier":
        return est.loss in ("log_loss", "log") and est.penalty in ("l2", None)
    return False


def _proba1(model, X):
    return np.asarray(model.predict_proba(X))[:, 1]


class StabilityAdapter:
    """
    Wrap a classifier so that retraining it with a different seed changes as few decisions as possible.

        from sklearn.linear_model import SGDClassifier
        from xgboost import XGBClassifier
        StabilityAdapter(SGDClassifier(loss="log_loss", alpha=0.01)).fit(X, y)       # strategy "polish"
        StabilityAdapter(XGBClassifier(subsample=0.8), n_seeds=10).fit(X, y)          # strategy "average"

    Parameters
    ----------
    estimator : unfitted binary classifier with fit / predict_proba (scikit-learn API).
    strategy : "auto" (polish for logistic models, average otherwise), "polish" or "average".
    n_seeds : copies averaged by the "average" strategy.
    polish_epochs, step, theta : the polish (see ``polish_logistic``). lam : L2 strength of the polished objective;
        by default taken from the estimator (SGDClassifier alpha, LogisticRegression 1/(C n), RiskAverseClassifier lam).
    calibrate : add out-of-fold Platt scaling (``cv`` folds; costs cv extra fits of the whole adapter).
    random_state : base seed; copy i uses seed ``random_state * 1000 + i``.
    batch_size : if set, the polish reads the data in batches of this many rows (``polish_logistic_streaming``): the same
        result with the step search done from batch sums. For ``step="newton"`` the Hessian (d x d) is accumulated over batches.
    """

    def __init__(self, estimator, strategy="auto", n_seeds=10, polish_epochs=20, step="line_search", theta=1.0,
                 lam=None, calibrate=False, cv=5, random_state=0, seed_param=None, batch_size=None):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        if step not in STEPS:
            raise ValueError(f"step must be one of {STEPS}, got {step!r}")
        if n_seeds < 1:
            raise ValueError("n_seeds must be >= 1")
        self.estimator, self.strategy, self.n_seeds = estimator, strategy, n_seeds
        self.polish_epochs, self.step, self.theta, self.lam = polish_epochs, step, theta, lam
        self.calibrate, self.cv, self.random_state, self.seed_param = calibrate, cv, random_state, seed_param
        self.batch_size = batch_size

    # -- fitting ---------------------------------------------------------------------------------------------------
    def _resolve(self):
        if self.strategy != "auto":
            return self.strategy
        return "polish" if _is_logistic(self.estimator) else "average"

    def _fit_core(self, X, y, seed):
        """Fit the uncalibrated adapter; return a function X -> p(positive)."""
        strategy = self._resolve()
        if strategy == "average":
            models = [_set_seed(self.estimator, seed * 1000 + i, self.seed_param).fit(X, y) for i in range(self.n_seeds)]
            return (lambda Z: np.mean([_proba1(m, Z) for m in models], axis=0)), dict(models=models)
        est = _set_seed(self.estimator, seed * 1000, self.seed_param).fit(X, y)
        name = type(est).__name__
        Xn = np.asarray(X, float)
        if name == "RiskAverseClassifier":
            w0, F, lam, mask = est.coef_, Xn, est.lam, None
            yy = est._labels(y)
        else:
            if len(est.classes_) != 2:
                raise ValueError("the polish strategy supports binary classification only")
            w0 = np.concatenate([np.ravel(est.coef_), np.ravel(est.intercept_)])
            F = np.column_stack([Xn, np.ones(len(Xn))])
            mask = np.concatenate([np.ones(Xn.shape[1]), [0.0]])
            yy = np.where(np.asarray(y) == est.classes_[1], 1.0, -1.0)
            if self.lam is not None:
                lam = self.lam
            elif name == "SGDClassifier":
                lam = est.alpha
            elif name == "LogisticRegression":
                lam = 1.0 / (est.C * len(yy)) if getattr(est, "penalty", "l2") == "l2" else 0.0
            else:
                lam = 1e-2
        lam = self.lam if self.lam is not None else lam
        if self.batch_size:
            w, info = polish_logistic_streaming(w0, iter_batches(F, yy, self.batch_size), lam, epochs=self.polish_epochs,
                                                step=self.step, theta=self.theta, mask=mask)
        else:
            w, info = polish_logistic(w0, F, yy, lam, epochs=self.polish_epochs, step=self.step, theta=self.theta, mask=mask)
        info.update(lam=lam)
        if name == "RiskAverseClassifier":
            f = lambda Z: sigmoid(np.asarray(Z, float) @ w)  # noqa: E731
        else:
            f = lambda Z: sigmoid(np.asarray(Z, float) @ w[:-1] + w[-1])  # noqa: E731
        return f, dict(model=est, coef=w, polish=info)

    def fit(self, X, y):
        t0 = time.perf_counter()
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("StabilityAdapter supports binary classification only")
        self.strategy_ = self._resolve()
        self._f, self.fitted_ = self._fit_core(X, y, self.random_state)
        n_fits = self.n_seeds if self.strategy_ == "average" else 1
        self.calibration_ = None
        if self.calibrate:
            y01 = (y == self.classes_[1]).astype(float)
            oof = np.zeros(len(y))
            folds = np.random.default_rng(self.random_state).permutation(len(y)) % self.cv
            Xa = np.asarray(X) if not hasattr(X, "iloc") else X
            for k in range(self.cv):
                tr, te = folds != k, folds == k
                sub = Xa.iloc[tr] if hasattr(Xa, "iloc") else Xa[tr]
                f_k, _ = self._fit_core(sub, y[tr], self.random_state)
                oof[te] = f_k(Xa.iloc[te] if hasattr(Xa, "iloc") else Xa[te])
            self.calibration_ = platt_fit(oof, y01)
            n_fits *= self.cv + 1
        self.info_ = dict(strategy=self.strategy_, fits=n_fits, fit_seconds=time.perf_counter() - t0,
                          calibrated=bool(self.calibrate))
        if "polish" in self.fitted_:
            self.info_["polish"] = self.fitted_["polish"]
        return self

    # -- prediction ------------------------------------------------------------------------------------------------
    def predict_proba(self, X):
        p = self._f(X)
        if self.calibration_ is not None:
            p = platt_apply(p, self.calibration_)
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return np.where(self.predict_proba(X)[:, 1] >= threshold, self.classes_[1], self.classes_[0])


# ---------------------------------------------------------------------------------------------------------------------
# measuring stability


@dataclass
class StabilityResult:
    name: str
    summary: dict
    groups: dict = field(default_factory=dict)
    probabilities: Optional[np.ndarray] = None

    def to_dict(self):
        return dict(name=self.name, summary=self.summary, groups=self.groups, disclaimer=DISCLAIMER)

    def to_markdown(self):
        s = self.summary
        lines = [f"## {self.name}", "",
                 f"- Retrains: {s['retrains']}; rows checked: {s['rows']}; seconds per retrain: {s['fit_seconds']:.2f}",
                 f"- Chance that two retrains disagree on a row: {s['disagreement']:.1%}",
                 f"- Unstable rows (decision differs from the majority in at least 10% of retrains): {s['unstable_share']:.1%}",
                 f"- Decisions that differ from the majority, per retrain: {s['changes_mean']:.1f} on average, "
                 f"{s['changes_worst5']:.1f} in the worst 5% of retrains"]
        if s.get("wrong") is not None:
            lines.append(f"- Chance of a wrong decision: {s['wrong']:.1%} for one retrain, {s['wrong_majority']:.1%} "
                         f"for the majority vote ({s['wrong_excess']:+.1%} due to training randomness)")
        if self.groups:
            lines += ["", "| Group | Rows | Wrong (one retrain) | Wrong (majority) | Due to randomness | Unstable rows |",
                      "|---|---|---|---|---|---|"]
            for g, v in self.groups.items():
                wr = "-" if v.get("wrong") is None else f"{v['wrong']:.1%}"
                wm = "-" if v.get("wrong_majority") is None else f"{v['wrong_majority']:.1%}"
                ex = "-" if v.get("wrong_excess") is None else f"{v['wrong_excess']:+.1%}"
                lines.append(f"| {g} | {v['rows']} | {wr} | {wm} | {ex} | {v['unstable_share']:.1%} |")
        return "\n".join(lines)

    def statements(self, top=3):
        """Plain sentences for the groups with the largest part of wrong decisions due to training randomness."""
        rows = [(g, v) for g, v in self.groups.items() if (v.get("wrong_excess") or 0) > 0]
        rows.sort(key=lambda gv: -gv[1]["wrong_excess"])
        return [f"For rows in '{g}', a single retrain of {self.name} makes a wrong decision {v['wrong']:.1%} of the time; "
                f"the majority vote of {self.summary['retrains']} retrains does so {v['wrong_majority']:.1%} of the time, "
                f"so {100 * v['wrong_excess']:.1f} percentage points come from training randomness."
                for g, v in rows[:top]]


def _stats(D, truth):
    S = D.shape[0]
    p = D.mean(0)
    maj = p >= 0.5
    changes = (D != maj[None]).sum(1)
    q = np.quantile(changes, 0.95)
    out = dict(rows=int(D.shape[1]), disagreement=float(np.mean(2 * p * (1 - p)) * S / max(S - 1, 1)),
               unstable_share=float(np.mean(np.minimum(p, 1 - p) >= 0.10)),
               changes_mean=float(changes.mean()), changes_worst5=float(changes[changes >= q].mean()))
    if truth is not None:
        wrong = float((D != truth[None]).mean())
        wm = float((maj != truth).mean())
        out.update(wrong=wrong, wrong_majority=wm, wrong_excess=wrong - wm)
    return out


def stability_check(make_model: Callable[[int], object], X_train, y_train, X_test, y_test=None, seeds=20,
                    groups: Optional[dict] = None, threshold=0.5, name="model", min_group_size=15,
                    keep_probabilities=False) -> StabilityResult:
    """
    Retrain ``make_model(seed)`` for each seed on the same training data and measure its decisions on X_test.

    make_model : seed -> unfitted classifier (fit / predict_proba). Wrap the adapter here to measure it, e.g.
        ``lambda s: StabilityAdapter(XGBClassifier(subsample=0.8), random_state=s)``.
    y_test : optional true labels (enables the chance of a wrong decision). The reference for "due to randomness" is
        the majority decision over all retrains.
    groups : optional {name: boolean mask over the test rows}; groups smaller than ``min_group_size`` are skipped.
    """
    seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    if len(seeds) < 2:
        raise ValueError("need at least 2 seeds")
    P, secs = [], []
    for s in seeds:
        t0 = time.perf_counter()
        m = make_model(s).fit(X_train, y_train)
        secs.append(time.perf_counter() - t0)
        P.append(_proba1(m, X_test))
    P = np.array(P)
    D = P >= threshold
    truth = None
    if y_test is not None:
        yt = np.asarray(y_test)
        positive = np.unique(np.asarray(y_train))[-1]
        truth = yt == positive
    summary = _stats(D, truth)
    summary.update(retrains=len(seeds), fit_seconds=float(np.mean(secs)), threshold=threshold)
    res_groups = {}
    for g, mask in (groups or {}).items():
        mask = np.asarray(mask, bool)
        if mask.sum() >= min_group_size:
            res_groups[g] = _stats(D[:, mask], None if truth is None else truth[mask])
    return StabilityResult(name=name, summary=summary, groups=res_groups,
                           probabilities=P if keep_probabilities else None)


def compare_stability(results: Iterable[StabilityResult], group=None) -> str:
    """Markdown table with one row per result (overall, or for one group)."""
    lines = ["| Model | Two retrains disagree | Unstable rows | Changes per retrain (avg / worst 5%) | "
             "Wrong decision (one retrain) | Due to randomness | Seconds per fit |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        s = r.summary if group is None else r.groups.get(group)
        if s is None:
            continue
        wr = "-" if s.get("wrong") is None else f"{s['wrong']:.1%}"
        ex = "-" if s.get("wrong_excess") is None else f"{s['wrong_excess']:+.1%}"
        lines.append(f"| {r.name} | {s['disagreement']:.1%} | {s['unstable_share']:.1%} | "
                     f"{s['changes_mean']:.1f} / {s['changes_worst5']:.1f} | {wr} | {ex} | "
                     f"{r.summary['fit_seconds']:.2f} |")
    return "\n".join(lines)
