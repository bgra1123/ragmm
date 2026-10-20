"""Wall-clock cost of the step search: one model, B = 1 / 5 / 20 extra epochs. Run: python research/risk_effect/cost_benchmark.py
Single-threaded (OMP_NUM_THREADS=1), median of 3 repeats. Real data for German, COMPAS, Adult when available; the 200,000-row
case is synthetic (timing only, no accuracy claim)."""
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
for sub in ("model_multiplicity", "gate0"):
    sys.path.insert(0, str(HERE.parent / sub))
import loan_experiment as le  # noqa: E402
import run_gate0 as g0  # noqa: E402
from risk_averse_tuning.adapter import iter_batches, polish_logistic, polish_logistic_streaming  # noqa: E402

LAM = 0.01


def sgd_epochs(w0, X, y, epochs, lr, batch=16, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    w = w0.copy()
    for _ in range(round(epochs * n / batch)):
        i = rng.integers(0, n, batch)
        Xb, yb = X[i], y[i]
        w -= lr * (-(Xb.T @ (yb * le.sigmoid(-yb * (Xb @ w)))) / batch + LAM * w)
    return w


def timeit(f, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def datasets():
    out = {}
    for name in ("german", "compas", "adult"):
        try:
            X, y, *_ = g0.prepare(name)
            out[name] = (X, y)
        except Exception as e:  # noqa: BLE001
            print("skip", name, repr(e)[:80])
    rng = np.random.default_rng(0)
    n, d = 200_000, 50
    Xs = np.column_stack([rng.normal(size=(n, d)), np.ones(n)])
    ws = rng.normal(size=d + 1) / 4
    out["synthetic 200k x 51 (timing only)"] = (Xs, np.where(rng.random(n) < 1 / (1 + np.exp(-Xs @ ws)), 1.0, -1.0))
    return out


def main():
    res = {}
    for name, (X, y) in datasets().items():
        n, d = X.shape
        L = float(np.linalg.eigvalsh(X.T @ X / n).max() / 4 + LAM)
        w1 = sgd_epochs(np.zeros(d), X, y, 2000 * 16 / n if n > 2000 else 3, 1 / L)
        mask = np.ones(d)
        cells = {}
        for B in (1, 5, 20):
            row = {"SGD continue (batch 16)": timeit(lambda: sgd_epochs(w1, X, y, B, 1 / L), 1 if n > 50_000 else 3)}
            for step in ("line_search", "entropic"):
                row[f"{step}: in memory (grid + golden)"] = timeit(lambda: polish_logistic(w1, X, y, LAM, epochs=B, step=step, mask=mask), 1 if n > 50_000 else 3)
                bs = max(1000, n // 20)
                row[f"{step}: streaming, grid sums"] = timeit(lambda: polish_logistic_streaming(w1, iter_batches(X, y, bs), LAM, epochs=B, step=step, mask=mask, search="grid"), 1 if n > 50_000 else 3)
                row[f"{step}: streaming, Newton on cached scalars"] = timeit(lambda: polish_logistic_streaming(w1, iter_batches(X, y, bs), LAM, epochs=B, step=step, mask=mask, search="newton"), 1 if n > 50_000 else 3)
            cells[str(B)] = row
            print(name, B, {k: round(v, 3) for k, v in row.items()}, flush=True)
        res[name] = dict(n=n, d=d, seconds=cells)
    (HERE / "results" / "cost.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
