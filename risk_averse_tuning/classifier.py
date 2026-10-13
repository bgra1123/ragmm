"""
Logistic regression trained by stochastic GMM, with the step size and momentum chosen by RA-GMM (problem (51) of
Can & Gürbüzbalaban, 2025) or by two baselines. NumPy only.

Objective: f(w) = (1/n) sum_i log(1 + exp(-y_i x_i^T w)) + (lam/2) ||w||^2, y in {+1, -1}.
It is mu-strongly convex with mu = lam and L-smooth with L = lambda_max(X^T X / n) / 4 + lam.

methods
    "ragmm"    : (alpha, beta, gamma) from problem (51) at L, mu, d (no tuning runs).
    "sgd"      : plain SGD with alpha = 1/L (the textbook step).
    "careful"  : SGD with tail averaging, step chosen on 20 tuning runs to minimize the mean objective.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .risk import RAGMMBounds

METHODS = ("ragmm", "sgd", "careful")


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def objective(W, X, y, lam):
    W2 = np.atleast_2d(W)
    return np.logaddexp(0.0, -(W2 @ X.T) * y).mean(1) + 0.5 * lam * (W2 ** 2).sum(1)


def smoothness(X, lam):
    return float(np.linalg.eigvalsh(X.T @ X / X.shape[0]).max() / 4.0 + lam)


def solve_exact(X, y, lam, tol=1e-10, max_iter=100):
    """Newton's method with backtracking. Returns (w*, f*)."""
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(max_iter):
        s = sigmoid(-y * (X @ w))
        g = -(X.T @ (y * s)) / n + lam * w
        if np.linalg.norm(g) < tol:
            break
        H = (X.T * (s * (1 - s))) @ X / n + lam * np.eye(d)
        step = np.linalg.solve(H, g)
        t, f0 = 1.0, objective(w, X, y, lam)[0]
        while objective(w - t * step, X, y, lam)[0] > f0 - 0.25 * t * g @ step and t > 1e-12:
            t *= 0.5
        w = w - t * step
    return w, float(objective(w, X, y, lam)[0])


def batch_indices(seeds, steps, n, batch):
    return np.stack([np.random.default_rng(s).integers(0, n, size=(steps, batch)) for s in seeds]).astype(np.int64)


def train_gmm(X, y, lam, idx, alpha, beta, gamma, tail_average=False):
    """
    Stochastic GMM for all seeds at once (idx has shape (seeds, steps, batch)):
        y_k = x_k + gamma d_k ; d_{k+1} = beta d_k - alpha g(y_k) ; x_{k+1} = x_k + d_{k+1}.
    Returns the final iterates (or tail averages), shape (seeds, d).
    """
    S, K, b = idx.shape
    W = np.zeros((S, X.shape[1]))
    D = np.zeros_like(W)
    avg, n_avg = np.zeros_like(W), 0
    for k in range(K):
        Y = W + gamma * D
        Xb, yb = X[idx[:, k]], y[idx[:, k]]
        coef = -yb * sigmoid(-np.einsum("sbd,sd->sb", Xb, Y) * yb)
        G = np.einsum("sbd,sb->sd", Xb, coef) / b + lam * Y
        D = beta * D - alpha * G
        W = W + D
        if tail_average and k >= K // 2:
            n_avg += 1
            avg += (W - avg) / n_avg
    return avg if tail_average else W


def choose_parameters(method, X, y, lam, steps, batch, zeta=0.05, rate_slack=0.25):
    """Return dict(alpha, beta, gamma, tail_average) and a dict of diagnostics."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    L, mu, d = smoothness(X, lam), float(lam), X.shape[1]
    info = dict(L=L, mu=mu, kappa=L / mu)
    if method == "sgd":
        return dict(alpha=1.0 / L, beta=0.0, gamma=0.0, tail_average=False), info
    if method == "ragmm":
        sol = RAGMMBounds(zeta=zeta).solve(L, mu, d, rate_slack=rate_slack)
        info.update(evar_bound=sol.evar_bound, rate=sol.rate, rate_ratio=sol.rate_ratio, region=sol.region)
        return dict(alpha=sol.alpha, beta=sol.beta, gamma=sol.gamma, tail_average=False), info
    _, fstar = solve_exact(X, y, lam)
    idx = batch_indices(range(10_000, 10_020), steps, X.shape[0], batch)
    best, best_val = None, math.inf
    for c in np.logspace(-2, math.log10(2.0), 12):
        W = train_gmm(X, y, lam, idx, c / L, 0.0, 0.0, tail_average=True)
        sub = objective(W, X, y, lam) - fstar
        val = float(np.mean(sub)) if np.all(np.isfinite(sub)) else math.inf
        if val < best_val:
            best, best_val = c / L, val
    info["tuning_runs"] = 12 * 20
    return dict(alpha=best, beta=0.0, gamma=0.0, tail_average=True), info


class RiskAverseClassifier:
    """
    Binary classifier (logistic regression) trained with stochastic GMM.

    Parameters: method ("ragmm" default, "sgd", "careful"), lam (L2 strength), steps, batch_size,
    zeta (tail probability of the EV@R guarantee), rate_slack (epsilon of problem (51)), seed.
    Works on a numeric matrix X (add an intercept column yourself, or use ``TabularEncoder``) and labels in {+1, -1}
    or {1, 0}.
    """

    def __init__(self, method="ragmm", lam=0.01, steps=2000, batch_size=16, zeta=0.05, rate_slack=0.25, seed=0):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if lam <= 0:
            raise ValueError("lam must be > 0 (the method needs strong convexity)")
        if steps < 1 or batch_size < 1:
            raise ValueError("steps and batch_size must be >= 1")
        self.method, self.lam, self.steps, self.batch_size = method, lam, steps, batch_size
        self.zeta, self.rate_slack, self.seed = zeta, rate_slack, seed

    @staticmethod
    def _labels(y):
        y = np.asarray(y, float)
        if set(np.unique(y)) <= {0.0, 1.0}:
            y = 2 * y - 1
        if not set(np.unique(y)) <= {-1.0, 1.0}:
            raise ValueError("labels must be in {+1, -1} or {1, 0}")
        return y

    def fit(self, X, y, params: Optional[dict] = None):
        X = np.asarray(X, float)
        y = self._labels(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D with one row per label")
        if params is None:
            params, info = choose_parameters(self.method, X, y, self.lam, self.steps, self.batch_size,
                                             self.zeta, self.rate_slack)
        else:
            info = {}
        self.params_, self.info_ = dict(params), info
        idx = batch_indices([self.seed], self.steps, len(y), self.batch_size)
        W = train_gmm(X, y, self.lam, idx, params["alpha"], params["beta"], params["gamma"],
                      params.get("tail_average", False))
        if not np.all(np.isfinite(W)):
            raise FloatingPointError("training diverged; try method='careful' or a larger lam")
        self.coef_ = W[0]
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.coef_

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return np.where(self.predict_proba(X)[:, 1] >= threshold, 1, -1)
