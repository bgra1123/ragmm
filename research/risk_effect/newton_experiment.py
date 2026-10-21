"""Newton test (PLAN.md section 13). Run: python research/risk_effect/newton_experiment.py german compas pima heart adult
Output: results/newton_<name>.json"""
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stepmatch_experiment as sm  # noqa: E402
import risk_effect_experiment as re_  # noqa: E402
import loan_experiment as le  # noqa: E402
import run_gate0 as g0  # noqa: E402
from risk_averse_tuning.adapter import iter_batches, polish_logistic_streaming  # noqa: E402

LAM = sm.LAM
K = 6
sig = le.sigmoid


def newton_iterates(w0, X, y, k):
    """Damped Newton (Armijo) on the L2 logistic objective; returns the weights after 1..k iterations and the per-iteration seconds."""
    n, d = X.shape
    w = w0.copy()
    out, secs = [], []

    def F(v):
        return np.logaddexp(0.0, -y * (X @ v)).mean() + 0.5 * LAM * v @ v

    for _ in range(k):
        t0 = time.perf_counter()
        M = y * (X @ w)
        s = sig(-M)
        g = -(X.T @ (y * s)) / n + LAM * w
        if np.linalg.norm(g) < 1e-12:
            out.append(w.copy())
            secs.append(0.0)
            continue
        H = (X.T * (s * (1 - s))) @ X / n + LAM * np.eye(d)
        p = np.linalg.solve(H, g)
        t, f0 = 1.0, F(w)
        while F(w - t * p) > f0 - 0.25 * t * (g @ p) and t > 1e-12:
            t *= 0.5
        w = w - t * p
        secs.append(time.perf_counter() - t0)
        out.append(w.copy())
    return out, secs


def timeit(f, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def run(name):
    t0 = time.time()
    if name == "adult":
        X, y, Xte, yte = g0.prepare("adult")
        seeds = 100
    else:
        X, y, Xte, yte = g0.prepare(name)
        seeds = sm.SEEDS
    sm.SEEDS = seeds
    n, d = X.shape
    L, _ = le.constants(X, LAM)
    wstar, fstar = le.solve_exact(X, y, LAM)
    ex = sig(Xte @ wstar) >= 0.5
    out = dict(dataset=name, n=n, d=d, seeds=seeds, stage1={})
    for opt in ("SGD", "Adam"):
        W1 = sm.stage1(opt, X, y, L)
        iters = [[] for _ in range(K)]
        secs = [[] for _ in range(K)]
        for i in range(seeds):
            ws, ss = newton_iterates(W1[i], X, y, K)
            for k in range(K):
                iters[k].append(ws[k])
                secs[k].append(ss[k])
        res = {"stage1": sm.arm_stats(W1, W1, X, y, Xte, yte, ex, fstar, None)[0]}
        for k in range(K):
            Wk = np.array(iters[k])
            s, _, _ = sm.arm_stats(Wk, W1, X, y, Xte, yte, ex, fstar, None)
            s["max_dist_to_exact"] = float(np.abs(Wk - wstar).max())
            s["seconds_per_model"] = float(np.mean([sum(x[:k + 1]) for x in [[secs[j][i] for j in range(K)] for i in range(seeds)]]))
            res[f"newton_{k + 1}"] = s
        # reference arms at the same data: 20 line-search steps (streaming, Newton search), 1 epoch average-and-continue
        w1 = W1[0]
        bs = max(1000, n // 20)
        tl = timeit(lambda: polish_logistic_streaming(w1, iter_batches(X, y, bs), LAM, epochs=20, step="line_search", search="newton"), 1 if n > 20000 else 3)
        tn = timeit(lambda: polish_logistic_streaming(w1, iter_batches(X, y, bs), LAM, epochs=50, step="newton"), 1 if n > 20000 else 3)
        wn, info = polish_logistic_streaming(w1, iter_batches(X, y, bs), LAM, epochs=50, step="newton")
        idx2 = le.batch_indices([100_000], max(1, round(n / sm.BATCH)), n, sm.BATCH)
        tc = timeit(lambda: re_.continue_training(W1[:1], X, y, idx2, opt, 1 / L if opt == "SGD" else 0.01, "avg"), 1 if n > 20000 else 3)
        snaps, _ = sm.trajectory(W1, X, y, 20, "LS")
        for B in (5, 20):
            s, _, _ = sm.arm_stats(snaps[B], W1, X, y, Xte, yte, ex, fstar, None)
            res[f"line_search_{B}"] = s
        idx5 = le.batch_indices(range(100_000, 100_000 + seeds), max(1, round(5 * n / sm.BATCH)), n, sm.BATCH)
        s, _, _ = sm.arm_stats(re_.continue_training(W1, X, y, idx5, opt, 1 / L if opt == "SGD" else 0.01, "avg"), W1, X, y, Xte, yte, ex, fstar, None)
        res["C3_5_epochs"] = s
        res["timing_seconds_one_model"] = dict(line_search_20_steps_streaming=tl, newton_to_convergence_streaming=tn, newton_iterations=info["iterations"],
                                               newton_passes=info["passes"], C3_one_epoch=tc)
        out["stage1"][opt] = res
        print(name, opt, f"[{time.time() - t0:.0f}s]", res["timing_seconds_one_model"], flush=True)
    return out


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["german", "compas", "pima", "heart"]):
        (HERE / "results" / f"newton_{nm}.json").write_text(json.dumps(run(nm), indent=2))
