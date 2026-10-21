"""Tables for the step-multiplier test from results/stepscale_<name>.json."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DS = [d for d in ("german", "compas", "pima", "heart", "adult") if (HERE / "results" / f"stepscale_{d}.json").exists()]
res = {d: json.loads((HERE / "results" / f"stepscale_{d}.json").read_text()) for d in DS}
OM = ["1", "1.5", "2", "3"]
RULES = ["L", "R1(theta=1)", "R1(theta=5)", "R3(theta=5)"]
out = []
for opt in ("SGD", "Adam"):
    out.append(f"\n## Stage 1 = {opt}: two retrains disagree (%), by step multiplier omega\n")
    out.append("| Dataset | Steps | stage 1 | C1 | C3 | " + " | ".join(f"{r} w={o}" for r in RULES for o in OM) + " |")
    out.append("|---|---|---|---|---|" + "---|" * (len(RULES) * len(OM)))
    for d in DS:
        blk = res[d]["stage1"][opt]
        for B in ("1", "2", "5"):
            a = blk["budgets"][B]["arms"]
            out.append(f"| {d} | {B} | {100 * blk['stage1']['disagreement']:.1f} | {100 * a['C1|1']['disagreement']:.1f} | {100 * a['C3|1']['disagreement']:.1f} | " + " | ".join(f"{100 * a[f'{r}|{o}']['disagreement']:.1f}" for r in RULES for o in OM) + " |")
    out.append(f"\n## Stage 1 = {opt}: training-objective gap f(w) - f*, mean over models\n")
    out.append("| Dataset | Steps | stage 1 | " + " | ".join(f"{r} w={o}" for r in RULES for o in OM) + " |")
    out.append("|---|---|---|" + "---|" * (len(RULES) * len(OM)))
    for d in DS:
        blk = res[d]["stage1"][opt]
        for B in ("1", "2", "5"):
            a = blk["budgets"][B]["arms"]
            out.append(f"| {d} | {B} | {blk['stage1']['gap_mean']:.4f} | " + " | ".join(f"{a[f'{r}|{o}']['gap_mean']:.4f}" for r in RULES for o in OM) + " |")
out.append("\n## Win rule: omega > 1 lowers disagreement against omega = 1 with the 95% CI below 0, on at least 3 of 5 datasets, wrong-decision rate not worse than +1 point\n")
out.append("| Stage 1 | Steps | Arm | omega | datasets that win | which |")
out.append("|---|---|---|---|---|---|")
for opt in ("SGD", "Adam"):
    for B in ("1", "2", "5"):
        for r in RULES:
            for o in OM[1:]:
                ok = []
                for d in DS:
                    t = res[d]["stage1"][opt]["budgets"][B]["tests"][f"{r}|{o} vs omega1"]
                    if t["disagreement"][2] < 0 and t["wrong"][1] < 0.01:
                        ok.append(d)
                out.append(f"| {opt} | {B} | {r} | {o} | {len(ok)} / {len(DS)} | {', '.join(ok) or '-'} |")
out.append("\n## Best arm against average-and-continue (C3): lowest disagreement over all arms and omegas, difference to C3 (pts, 95% CI)\n")
out.append("| Stage 1 | Dataset | Steps | best arm | disagreement | C3 | difference vs C3 |")
out.append("|---|---|---|---|---|---|---|")
for opt in ("SGD", "Adam"):
    for d in DS:
        for B in ("1", "2", "5"):
            blk = res[d]["stage1"][opt]["budgets"][B]
            cand = {k: v for k, v in blk["arms"].items() if not k.startswith("C")}
            k = min(cand, key=lambda x: cand[x]["disagreement"])
            nm, om = k.split("|")
            t = blk["tests"].get(f"{nm}|{om} vs C3")
            out.append(f"| {opt} | {d} | {B} | {k} | {100 * cand[k]['disagreement']:.1f}% | {100 * blk['arms']['C3|1']['disagreement']:.1f}% | {100 * t['disagreement'][0]:+.2f} [{100 * t['disagreement'][1]:+.2f}, {100 * t['disagreement'][2]:+.2f}] |")
print("\n".join(out))
