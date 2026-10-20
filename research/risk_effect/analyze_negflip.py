"""Tables for the negative-flip test from results/negflip_<dataset>.json."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DS = [d for d in ("german", "compas", "pima", "heart") if (HERE / "results" / f"negflip_{d}.json").exists()]
TH = [0.5, 1.0, 2.0, 5.0, 10.0]
res = {d: json.loads((HERE / "results" / f"negflip_{d}.json").read_text()) for d in DS}


def ci(t, p=2, s=100.0):
    return f"{s * t[0]:+.{p}f} [{s * t[1]:+.{p}f}, {s * t[2]:+.{p}f}]"


out = []
for opt in ("SGD", "Adam"):
    out.append(f"\n## Stage 1 = {opt} (negative / positive flips as % of held-out rows; net = positive minus negative)\n")
    for B in ("1", "2", "5"):
        out.append(f"\n### {B} epoch(s)\n")
        out.append("| Dataset | Stage-1 error | Arm | negative flips | positive flips | net | churn |")
        out.append("|---|---|---|---|---|---|---|")
        for d in DS:
            blk = res[d]["stage1"][opt]
            a = blk["budgets"][B]["arms"]
            for k in ["C1", "C2", "C3", "L", "R1(theta=5)", "R3(theta=5)", "M1[R1(theta=5)]", "M2[R1(theta=5)]", "M1[R3(theta=5)]"]:
                x = a[k]
                out.append(f"| {d} | {100 * blk['stage1_error']:.1f}% | {k} | {100 * x['neg']:.2f}% | {100 * x['pos']:.2f}% | {100 * x['net']:+.2f} | {100 * x['churn']:.2f}% |")
out.append("\n## Credit rule: R has lower negative-flip rate than both M1 and M2 (95% CI below 0) and net no worse than M1 by more than 0.5 points\n")
out.append("| Stage 1 | Steps | R arm | datasets passing | which |")
out.append("|---|---|---|---|---|")
for opt in ("SGD", "Adam"):
    for B in ("1", "2", "5"):
        for r in ("R1", "R3"):
            for t in TH:
                k = f"{r}(theta={t:g})"
                ok = []
                for d in DS:
                    T = res[d]["stage1"][opt]["budgets"][B]["tests"]
                    if T[k + " vs M1"]["neg"][2] < 0 and T[k + " vs M2"]["neg"][2] < 0 and T[k + " vs M1"]["net"][1] > -0.005:
                        ok.append(d)
                out.append(f"| {opt} | {B} | {k} | {len(ok)} / {len(DS)} | {', '.join(ok) or '-'} |")
out.append("\n## Risk step minus comparison (negative-flip rate, percentage points of held-out rows, 95% CI), theta = 5\n")
out.append("| Stage 1 | Dataset | Epochs | R arm | vs L | vs best continuation | vs M1 | vs M2 | net vs M1 |")
out.append("|---|---|---|---|---|---|---|---|---|")
for opt in ("SGD", "Adam"):
    for d in DS:
        for B in ("1", "2", "5"):
            T = res[d]["stage1"][opt]["budgets"][B]
            bc = T["best_continuation"]
            for r in ("R1", "R3"):
                k = f"{r}(theta=5)"
                t = T["tests"]
                out.append(f"| {opt} | {d} | {B} | {k} | {ci(t[k + ' vs L']['neg'])} | {ci(t[k + ' vs ' + bc]['neg'])} ({bc}) | {ci(t[k + ' vs M1']['neg'])} | {ci(t[k + ' vs M2']['neg'])} | {ci(t[k + ' vs M1']['net'])} |")
if "german" in res:
    out.append("\n## German: worst-group negative-flip rate (groups with >= 30 held-out rows)\n")
    out.append("| Stage 1 | Epochs | C3 | L | R1(5) | M1[R1(5)] | R3(5) | M1[R3(5)] |")
    out.append("|---|---|---|---|---|---|---|---|")
    for opt in ("SGD", "Adam"):
        for B in ("1", "2", "5"):
            a = res["german"]["stage1"][opt]["budgets"][B]["arms"]
            g = lambda k: f"{100 * a[k]['worst_group_neg']:.2f}%"  # noqa: E731
            out.append(f"| {opt} | {B} | {g('C3')} | {g('L')} | {g('R1(theta=5)')} | {g('M1[R1(theta=5)]')} | {g('R3(theta=5)')} | {g('M1[R3(theta=5)]')} |")
print("\n".join(out))
