"""Tables for Experiment C from results/large_<n>.json."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
NS = [n for n in (5000, 20000, 46842) if (HERE / "results" / f"large_{n}.json").exists()]
res = {n: json.loads((HERE / "results" / f"large_{n}.json").read_text()) for n in NS}
BS = ["0.25", "0.5", "1.0", "2.0"]
RK = ["R1(theta=1)", "R1(theta=5)", "R3(theta=1)", "R3(theta=5)"]


def ci(t, s=100.0, p=2):
    return f"{s * t[0]:+.{p}f} [{s * t[1]:+.{p}f}, {s * t[2]:+.{p}f}]"


out = []
for opt in ("SGD", "Adam"):
    out.append(f"\n## Stage 1 = {opt}\n")
    out.append("### Two retrains disagree (mean over 100 models; lower is better)\n")
    out.append("| n | passes | stage 1 | C1 | C2 | C3 | L (line search) | R1(1) | R1(5) | R3(1) | R3(5) | best continuation | best R beats it? (CI below 0) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in NS:
        blk = res[n]["stage1"][opt]
        for B in BS:
            c = blk["budgets"][B]
            a = c["arms"]
            bestR = min(RK, key=lambda k: a[k]["disagreement"])
            t = c["tests"][f"{bestR} vs {c['best_continuation']}"]["disagreement"]
            verdict = f"yes ({bestR}, {ci(t)})" if t[2] < 0 else (f"no ({bestR}, {ci(t)})")
            out.append(f"| {n} | {B} | {100 * blk['stage1']['disagreement']:.1f}% | " + " | ".join(f"{100 * a[k]['disagreement']:.1f}%" for k in ["C1", "C2", "C3", "L"] + RK) + f" | {c['best_continuation']} | {verdict} |")
    out.append("\n### Decisions changed against the exact model of that size (mean per model, 2,000 held-out rows) and wrong-decision rate\n")
    out.append("| n | passes | stage 1 | C3 | L | R1(1) | R1(5) | R3(5) | wrong: stage 1 / C3 / R1(1) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for n in NS:
        blk = res[n]["stage1"][opt]
        for B in BS:
            a = blk["budgets"][B]["arms"]
            out.append(f"| {n} | {B} | {blk['stage1']['changed']:.1f} | {a['C3']['changed']:.1f} | {a['L']['changed']:.1f} | {a['R1(theta=1)']['changed']:.1f} | {a['R1(theta=5)']['changed']:.1f} | {a['R3(theta=5)']['changed']:.1f} | {100 * blk['stage1']['wrong']:.1f}% / {100 * a['C3']['wrong']:.1f}% / {100 * a['R1(theta=1)']['wrong']:.1f}% |")
    out.append("\n### Risk step minus matched controls (theta = 5): harm = 95th-percentile per-row loss increase; neg = negative-flip rate (percentage points of rows)\n")
    out.append("| n | passes | arm | step x L | harm vs L | harm vs M1 | harm vs M2 | neg flips vs L (pts) | neg flips vs M1 (pts) | disagreement vs M1 (pts) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in NS:
        blk = res[n]["stage1"][opt]
        for B in BS:
            c = blk["budgets"][B]
            for k in ("R1(theta=5)", "R3(theta=5)"):
                T = c["tests"]
                out.append(f"| {n} | {B} | {k} | {c['arms'][k]['mean_alpha_times_L']:.2f} | {ci(T[k + ' vs L']['harm'], 1, 3)} | {ci(T[k + ' vs M1']['harm'], 1, 3)} | {ci(T[k + ' vs M2']['harm'], 1, 3)} | {ci(T[k + ' vs L']['neg'])} | {ci(T[k + ' vs M1']['neg'])} | {ci(T[k + ' vs M1']['disagreement'])} |")
out.append("\n## Credit rule: harm lower than both L and M1 (CI below 0), same theta, at passes 0.25, 0.5 and 1, on at least 2 of the sizes\n")
out.append("| Stage 1 | R arm | passes passing (sizes) |")
out.append("|---|---|---|")
for opt in ("SGD", "Adam"):
    for k in ("R1(theta=5)", "R3(theta=5)"):
        cells = []
        for B in ("0.25", "0.5", "1.0"):
            ok = [n for n in NS if res[n]["stage1"][opt]["budgets"][B]["tests"][k + " vs L"]["harm"][2] < 0 and res[n]["stage1"][opt]["budgets"][B]["tests"][k + " vs M1"]["harm"][2] < 0]
            cells.append(f"{B}: {len(ok)}/{len(NS)}")
        out.append(f"| {opt} | {k} | " + "; ".join(cells) + " |")
print("\n".join(out))
