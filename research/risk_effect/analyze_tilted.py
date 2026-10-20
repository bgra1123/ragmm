"""Tables for Experiment B from results/tilted_<dataset>.json."""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DS = ["german", "compas", "pima", "heart"]
TH = [0.5, 1.0, 2.0, 5.0]
res = {d: json.loads((HERE / "results" / f"tilted_{d}.json").read_text())["cells"] for d in DS}
rng = np.random.default_rng(0)


def metric(cells, arm, kind, groups=None, idx=None):
    idx = range(len(cells)) if idx is None else idx
    if kind in ("acc", "worst_class_err", "fnr", "fpr"):
        return float(np.mean([cells[i][arm][kind] for i in idx]))
    # worst group error, pooled over the resampled splits, over the fixed group list
    errs = {g: sum(cells[i][arm]["group_err_counts"][g][0] for i in idx) / sum(cells[i][arm]["group_err_counts"][g][1] for i in idx) for g in groups}
    return max(errs.values())


def paired(cells, a, b, kind, groups=None, n=1000):
    S = len(cells)
    d0 = metric(cells, a, kind, groups) - metric(cells, b, kind, groups)
    bs = []
    for _ in range(n):
        idx = rng.integers(0, S, S)
        bs.append(metric(cells, a, kind, groups, idx) - metric(cells, b, kind, groups, idx))
    return d0, *np.quantile(bs, [0.025, 0.975])


def fmt(t, scale=100, p=1):
    return f"{scale * t[0]:+.{p}f} [{scale * t[1]:+.{p}f}, {scale * t[2]:+.{p}f}]"


g_list = None
cells = res["german"]
pooled = {g: sum(c["ERM"]["group_err_counts"][g][1] for c in cells) for g in cells[0]["ERM"]["group_err_counts"]}
g_list = [g for g, n in pooled.items() if n >= 200]
print("German groups used for the worst-group metric (pooled test rows over 20 splits >= 200):", {g: pooled[g] for g in g_list}, "\n")

print("## Level of each arm (mean over 20 splits): accuracy | worst-class error | German: worst-group error\n")
print("| Dataset | Arm | accuracy | false-negative rate | false-positive rate | worst-class error | worst-group error (German) |")
print("|---|---|---|---|---|---|---|")
for d in DS:
    arms = list(res[d][0].keys())
    for a in arms:
        wg = f"{100 * metric(res[d], a, 'wg', g_list):.1f}%" if d == "german" else "-"
        print(f"| {d} | {a} | {100 * metric(res[d], a, 'acc'):.1f}% | {100 * metric(res[d], a, 'fnr'):.1f}% | {100 * metric(res[d], a, 'fpr'):.1f}% | {100 * metric(res[d], a, 'worst_class_err'):.1f}% | {wg} |")

print("\n## Primary: tilted minus matched reweighting (percentage points; negative = tilted better), 95% CI over 20 splits\n")
print("| Dataset | theta | accuracy gap | worst-class error | worst-group error (German) |")
print("|---|---|---|---|---|")
for d in DS:
    for th in TH:
        a, b = f"tilt theta={th:g}", f"reweight (matched to theta={th:g})"
        wg = fmt(paired(res[d], a, b, "wg", g_list)) if d == "german" else "-"
        print(f"| {d} | {th:g} | {fmt(paired(res[d], a, b, 'acc'))} | {fmt(paired(res[d], a, b, 'worst_class_err'))} | {wg} |")
print("\n## Tilted minus plain ERM (percentage points)\n")
print("| Dataset | theta | accuracy | worst-class error | worst-group error (German) |")
print("|---|---|---|---|---|")
for d in DS:
    for th in TH:
        a = f"tilt theta={th:g}"
        wg = fmt(paired(res[d], a, "ERM", "wg", g_list)) if d == "german" else "-"
        print(f"| {d} | {th:g} | {fmt(paired(res[d], a, 'ERM', 'acc'))} | {fmt(paired(res[d], a, 'ERM', 'worst_class_err'))} | {wg} |")
print("\n## German: group-DRO (known groups) minus ERM and minus tilted theta=5\n")
for ref in ("ERM", "tilt theta=5"):
    print(f"- vs {ref}: accuracy {fmt(paired(res['german'], 'group-DRO (known groups)', ref, 'acc'))}; worst-group error {fmt(paired(res['german'], 'group-DRO (known groups)', ref, 'wg', g_list))}")
