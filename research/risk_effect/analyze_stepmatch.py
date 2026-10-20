"""Tables for the step-matched experiment from results/stepmatch_<dataset>.json."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DS = [d for d in ("german", "compas", "pima", "heart") if (HERE / "results" / f"stepmatch_{d}.json").exists()]
TH = [0.5, 1.0, 2.0, 5.0, 10.0]
res = {d: json.loads((HERE / "results" / f"stepmatch_{d}.json").read_text()) for d in DS}


def ci(t, scale=1.0, p=3):
    return f"{scale * t[0]:+.{p}f} [{scale * t[1]:+.{p}f}, {scale * t[2]:+.{p}f}]"


def sig_neg(t):
    return t[2] < 0


out = []
for opt in ("SGD", "Adam"):
    out.append(f"\n## Stage 1 = {opt}\n")
    for B in ("1", "2", "5"):
        out.append(f"\n### {B} full-batch step(s): level of each arm (held-out 95th-percentile per-row loss increase | disagreement | mean step x L)\n")
        out.append("| Dataset | Arm | harm p95 (held-out) | disagreement | step x L | wrong decision |")
        out.append("|---|---|---|---|---|---|")
        for d in DS:
            arms = res[d]["stage1"][opt]["budgets"][B]["arms"]
            for k in ["L"] + [f"{r}(theta={t:g})" for r in ("R1", "R3") for t in (1.0, 5.0)] + [f"M1[{r}(theta={t:g})]" for r in ("R1", "R3") for t in (1.0, 5.0)]:
                a = arms[k]
                out.append(f"| {d} | {k} | {a['harm_te_p95']:+.3f} | {100 * a['disagreement']:.1f}% | {a['mean_alpha_times_L']:.2f} | {100 * a['wrong']:.1f}% |")
    out.append(f"\n### Risk step minus matched controls (negative harm = risk step protects rows better; paired bootstrap over 200 seeds)\n")
    out.append("| Dataset | Steps | R arm | step x L | harm vs L | harm vs M1 (same mean step) | harm vs M2 (same schedule) | disagreement vs M1 (pts) | wrong vs M1 (pts) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for d in DS:
        for B in ("1", "2", "5"):
            blk = res[d]["stage1"][opt]["budgets"][B]
            for r in ("R1", "R3"):
                for t in TH:
                    k = f"{r}(theta={t:g})"
                    T = blk["tests"]
                    out.append(f"| {d} | {B} | {k} | {blk['arms'][k]['mean_alpha_times_L']:.2f} | {ci(T[k + ' vs L']['harm_te'])} | {ci(T[k + ' vs M1']['harm_te'])} | {ci(T[k + ' vs M2']['harm_te'])} | "
                               f"{ci(T[k + ' vs M1']['disagreement'], 100, 1)} | {ci(T[k + ' vs M1']['wrong'], 100, 1)} |")

# credit rule
out.append("\n## Credit rule (PLAN section 9): R has lower held-out harm than both L and M1 (95% CI below 0) on at least 3 of the 4 datasets\n")
out.append("| Stage 1 | Steps | R arm | datasets passing (harm < L and < M1) | which |")
out.append("|---|---|---|---|---|")
for opt in ("SGD", "Adam"):
    for B in ("1", "2", "5"):
        for r in ("R1", "R3"):
            for t in TH:
                k = f"{r}(theta={t:g})"
                ok = [d for d in DS if sig_neg(res[d]["stage1"][opt]["budgets"][B]["tests"][k + " vs L"]["harm_te"]) and sig_neg(res[d]["stage1"][opt]["budgets"][B]["tests"][k + " vs M1"]["harm_te"])]
                out.append(f"| {opt} | {B} | {k} | {len(ok)} / {len(DS)} | {', '.join(ok) or '-'} |")
out.append("\n## Dose-response: held-out harm of R as theta grows (should fall), mean over the datasets listed\n")
out.append("| Stage 1 | Steps | Rule | " + " | ".join(f"theta={t:g}" for t in TH) + " | monotone on |")
out.append("|---|---|---|" + "---|" * (len(TH) + 1))
for opt in ("SGD", "Adam"):
    for B in ("1", "2", "5"):
        for r in ("R1", "R3"):
            vals, mono = [], 0
            for d in DS:
                v = [res[d]["stage1"][opt]["budgets"][B]["arms"][f"{r}(theta={t:g})"]["harm_te_p95"] for t in TH]
                mono += all(v[i + 1] <= v[i] + 1e-9 for i in range(len(v) - 1))
                vals.append(v)
            mean = [sum(v[i] for v in vals) / len(vals) for i in range(len(TH))]
            out.append(f"| {opt} | {B} | {r} | " + " | ".join(f"{x:+.3f}" for x in mean) + f" | {mono} / {len(DS)} |")
if "german" in res:
    out.append("\n## German: worst-group wrong-decision rate (groups with >= 30 held-out rows), L vs R(theta=5) vs M1\n")
    out.append("| Stage 1 | Steps | L | R1(5) | M1[R1(5)] | R3(5) | M1[R3(5)] | R3(5) minus M1: worst-group (pts) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for opt in ("SGD", "Adam"):
        for B in ("1", "2", "5"):
            blk = res["german"]["stage1"][opt]["budgets"][B]
            A = blk["arms"]
            g = lambda k: f"{100 * A[k]['worst_group_wrong']:.1f}%"  # noqa: E731
            out.append(f"| {opt} | {B} | {g('L')} | {g('R1(theta=5)')} | {g('M1[R1(theta=5)]')} | {g('R3(theta=5)')} | {g('M1[R3(theta=5)]')} | {ci(blk['tests']['R3(theta=5) vs M1']['worst_group'], 100, 1)} |")
print("\n".join(out))
