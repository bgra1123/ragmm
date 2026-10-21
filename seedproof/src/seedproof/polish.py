"""
Deterministic polish of a logistic head: continue from trained weights with exact full-batch steps until every seed lands
on (nearly) the same model.

Step rules (``step=``):
  "line_search"     exact line search along the full gradient;
  "entropic"        the entropic risk-averse line search, alpha* = argmin_a (1/theta) log mean_i exp(-theta a g^T grad f_i(w)),
                    i.e. the entropic risk of the first-order per-row loss change (builds on Can & Gurbuzbalaban, 2025);
  "entropic_exact"  the same with the exact per-row loss change;
  "newton"          solve the head exactly (damped Newton).
Every rule has an in-memory version (``polish_logistic``) and a streaming version that reads the data in batches
(``polish_logistic_streaming``).
"""
from __future__ import annotations

import math

import numpy as np

from ._util import sigmoid

STEPS = ("line_search", "entropic", "entropic_exact", "newton")


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
