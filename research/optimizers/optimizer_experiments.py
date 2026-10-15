"""
Other optimizers and the polish adapter. Design: PLAN.md (pre-registered).
Run from the repository root:  python research/optimizers/optimizer_experiments.py --datasets german compas pima heart
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "polish"))
sys.path.insert(0, str(HERE.parent / "model_multiplicity"))
sys.path.insert(0, str(HERE.parent / "gate0"))
import loan_experiment as le  # noqa: E402
import polish_experiments as pe  # noqa: E402
import run_gate0 as g0  # noqa: E402
from risk_averse_tuning import RAGMMBounds  # noqa: E402

LAM, BATCH, K1, K2 = 0.01, 16, 2000, 1000
OPTS = {"SGD": [1e-3, 1e-2, 1e-1], "SGD+momentum": [1e-3, 1e-2, 1e-1], "SGD+Nesterov": [1e-3, 1e-2, 1e-1],
        "Adam": [1e-3, 1e-2, 1e-1], "RMSprop": [1e-3, 1e-2, 1e-1], "Adagrad": [1e-3, 1e-2, 1e-1]}


def run_opt(name, X, y, idx, lr, W0=None):
    """Vectorized torch-semantics optimizers over seeds. idx: (S, K, b)."""
    S, K, b = idx.shape
    W = np.zeros((S, X.shape[1])) if W0 is None else W0.copy()
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    for k in range(K):
        t = k + 1
        Xb, yb = X[idx[:, k]], y[idx[:, k]]

        def grad(P):
            coef = -yb * le.sigmoid(-np.einsum("sbd,sd->sb", Xb, P) * yb)
            return np.einsum("sbd,sb->sd", Xb, coef) / b + LAM * P

        g = grad(W)
        if name == "SGD":
            W = W - lr * g
        elif name in ("SGD+momentum", "SGD+Nesterov"):
            m = g if k == 0 else 0.9 * m + g                      # torch: buf = momentum*buf + grad (first step: buf = grad)
            W = W - lr * (g + 0.9 * m if name == "SGD+Nesterov" else m)
        elif name == "Adam":
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            W = W - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        elif name == "RMSprop":
            v = 0.99 * v + 0.01 * g * g
            W = W - lr * g / (np.sqrt(v) + 1e-8)
        elif name == "Adagrad":
            v = v + g * g
            W = W - lr * g / (np.sqrt(v) + 1e-10)
        else:
            raise ValueError(name)
    return W


def check_against_torch(X, y, steps=60):
    import torch
    idx = le.batch_indices([5], steps, X.shape[0], BATCH)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    makers = {"SGD": lambda p, lr: torch.optim.SGD([p], lr=lr),
              "SGD+momentum": lambda p, lr: torch.optim.SGD([p], lr=lr, momentum=0.9),
              "SGD+Nesterov": lambda p, lr: torch.optim.SGD([p], lr=lr, momentum=0.9, nesterov=True),
              "Adam": lambda p, lr: torch.optim.Adam([p], lr=lr),
              "RMSprop": lambda p, lr: torch.optim.RMSprop([p], lr=lr),
              "Adagrad": lambda p, lr: torch.optim.Adagrad([p], lr=lr)}
    worst = 0.0
    for name, mk in makers.items():
        lr = 0.05
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        opt = mk(w, lr)
        for k in range(steps):
            bi = torch.tensor(idx[0, k], dtype=torch.long)
            opt.zero_grad()
            (torch.nn.functional.softplus(-(Xt[bi] @ w) * yt[bi]).mean() + 0.5 * LAM * (w ** 2).sum()).backward()
            opt.step()
        err = np.abs(run_opt(name, X, y, idx, lr)[0] - w.detach().numpy()).max()
        worst = max(worst, err)
        assert err < 1e-8, f"{name} deviates from torch: {err}"
    return worst


def stage_metrics(W, X, y, Xte, yte, wstar, fstar):
    S = le.objective(W, X, y, LAM) - fstar
    if not np.all(np.isfinite(S)):
        return dict(diverged=int((~np.isfinite(S)).sum())), None
    q95 = float(np.quantile(S, 0.95))
    P = le.sigmoid(W @ Xte.T)
    dec = P >= 0.5
    ex = le.sigmoid(Xte @ wstar) >= 0.5
    ch = (dec != ex[None]).sum(1)
    p = dec.mean(0)
    k5 = max(1, len(S) // 20)
    aucs = np.array([le.auc(P[s], yte) for s in range(0, len(W), max(1, len(W) // 60))])
    return dict(S_mean=float(S.mean()), S_p95=q95, S_cvar5=float(S[S >= q95].mean()),
                changed_mean=float(ch.mean()), changed_worst5=float(np.sort(ch)[-k5:].mean()),
                retrain_disagreement=float(np.mean(2 * p * (1 - p)) * len(W) / (len(W) - 1)),
                bias=float(np.linalg.norm(W.mean(0) - wstar) / np.linalg.norm(wstar)),
                auc=float(aucs.mean())), S


def run_dataset(name, seeds):
    t0 = time.time()
    X, y, Xte, yte = g0.prepare(name)
    n, d = X.shape
    L, mu = le.constants(X, LAM)
    wstar, fstar = le.solve_exact(X, y, LAM)
    out = dict(dataset=name, seeds=seeds, L=L, kappa=L / mu, torch_check_max_err=float(check_against_torch(X, y)), cells=[])
    ra = RAGMMBounds(zeta=0.05).solve(L, mu, d, rate_slack=0.25)
    idx1 = le.batch_indices(range(seeds), K1, n, BATCH)
    rng = np.random.default_rng(0)
    for opt, lrs in OPTS.items():
        for lr in lrs:
            W1 = run_opt(opt, X, y, idx1, lr)
            m1, S1 = stage_metrics(W1, X, y, Xte, yte, wstar, fstar)
            cell = dict(optimizer=opt, lr=lr, stage1=m1, polish={})
            if S1 is not None:
                idx2 = le.batch_indices(range(100_000, 100_000 + seeds), K2, n, BATCH)
                sel, _ = pe.select_polish(X, y, W1, K2, rng)
                pol = {"none": None, "average": dict(alpha=1 / L, beta=0.0, gamma=0.0, average=True),
                       "decay": dict(alpha=0.05 / L, beta=0.0, gamma=0.0),
                       "RA-GMM (51)": dict(alpha=ra.alpha, beta=ra.beta, gamma=ra.gamma),
                       "Exact-EV@R polish": sel}
                Ss = {}
                for pn, cfg in pol.items():
                    W2 = W1 if cfg is None else pe.run_from(W1, X, y, idx2, **cfg)
                    mm, SS = stage_metrics(W2, X, y, Xte, yte, wstar, fstar)
                    cell["polish"][pn] = mm
                    if SS is not None:
                        Ss[pn] = SS
                cell["selected"] = {k: float(v) for k, v in sel.items()}
                for pn in Ss:
                    if pn != "none":
                        cell["polish"][pn]["share_seeds_improved"] = float(np.mean(Ss[pn] < Ss["none"]))
                        cell["polish"][pn]["cvar5_vs_none"] = pe.boot_diff(Ss[pn], Ss["none"], pe.cvar5, n=1000)
            out["cells"].append(cell)
            p = cell["polish"]
            print(f"{name} {opt:13s} lr={lr:<6g} S1 cvar {m1.get('S_cvar5', float('nan')):.4f} ch {m1.get('changed_mean', float('nan')):5.1f} "
                  f"dis {m1.get('retrain_disagreement', float('nan')):.3f} bias {m1.get('bias', float('nan')):.2f} | polish cvar: "
                  + " ".join(f"{k.split(' ')[0]}={v.get('S_cvar5', float('nan')):.4f}" for k, v in p.items() if k != "none")
                  + f" [{time.time() - t0:.0f}s]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["german", "compas", "pima", "heart"])
    ap.add_argument("--seeds", type=int, default=300)
    args = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    for name in args.datasets:
        res = run_dataset(name, args.seeds)
        (HERE / "results" / f"{name}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
