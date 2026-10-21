"""Step-multiplier test (PLAN.md section 12). Run: python research/risk_effect/stepscale_experiment.py german compas pima heart adult
Output: results/stepscale_<name>.json"""
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

OMEGAS = (1.0, 1.5, 2.0, 3.0)
STEPS = (1, 2, 5)
RULES = {"L": ("LS", 0.0), "R1(theta=1)": ("R1", 1.0), "R1(theta=5)": ("R1", 5.0), "R3(theta=5)": ("R3", 5.0)}


def run(name):
    t0 = time.time()
    if name == "adult":
        Xall, yall, Xte, yte = g0.prepare("adult")
        X, y, seeds = Xall, yall, 100
    else:
        X, y, Xte, yte = g0.prepare(name)
        seeds = sm.SEEDS
    n, d = X.shape
    L, _ = le.constants(X, sm.LAM)
    wstar, fstar = le.solve_exact(X, y, sm.LAM)
    ex = le.sigmoid(Xte @ wstar) >= 0.5
    good = yte > 0
    out = dict(dataset=name, n=n, seeds=seeds, stage1={})
    sm.SEEDS = seeds
    for opt in ("SGD", "Adam"):
        W1 = sm.stage1(opt, X, y, L)
        D1 = le.sigmoid(W1 @ Xte.T) >= 0.5
        lr0 = 1 / L if opt == "SGD" else 0.01
        stats, per, dec = {}, {}, {}

        def record(key, W):
            s, p, D = sm.arm_stats(W, W1, X, y, Xte, yte, ex, fstar, None)
            rb, ra = D1 == good[None], D == good[None]
            s["neg"] = float((rb & ~ra).mean())
            s["pos"] = float((~rb & ra).mean())
            stats[key], per[key], dec[key] = s, p, D

        s1, _, _ = sm.arm_stats(W1, W1, X, y, Xte, yte, ex, fstar, None)
        for B in STEPS:
            idx2 = le.batch_indices(range(100_000, 100_000 + seeds), max(1, round(B * n / sm.BATCH)), n, sm.BATCH)
            for nm, mode in (("C1", "const"), ("C3", "avg")):
                record((B, nm, 1.0), re_.continue_training(W1, X, y, idx2, opt, lr0, mode))
        for nm, (rule, th) in RULES.items():
            for om in OMEGAS:
                snaps, al = sm.trajectory(W1, X, y, max(STEPS), rule, th, scale=om)
                for B in STEPS:
                    record((B, nm, om), snaps[B])
                    stats[(B, nm, om)]["mean_alpha_times_L"] = float(al[:B].mean() * L)
            print(name, opt, nm, f"[{time.time() - t0:.0f}s]", flush=True)
        res = {}
        for B in STEPS:
            arms = {f"{nm}|{om:g}": stats[(B, nm, om)] for (b, nm, om) in stats if b == B}
            tests = {}
            for nm in RULES:
                for om in OMEGAS:
                    k = (B, nm, om)
                    for ref_key, ref in (("C3", (B, "C3", 1.0)), ("omega1", (B, nm, 1.0))):
                        if ref_key == "omega1" and om == 1.0:
                            continue
                        tests[f"{nm}|{om:g} vs {ref_key}"] = sm.boot_pair(per[k], per[ref], dec[k], dec[ref], None)
            res[str(B)] = dict(arms=arms, tests=tests)
        out["stage1"][opt] = dict(stage1=s1, budgets=res)
    return out


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["german", "compas", "pima", "heart"]):
        r = run(nm)
        (HERE / "results" / f"stepscale_{nm}.json").write_text(json.dumps(r, indent=2))
