"""Tables for RESULTS.md from results/<dataset>.json."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DS = ["german", "compas", "pima", "heart"]
R = ["R risk first-order (theta=1)", "R risk exact-change (theta=5)"]
res = {d: json.loads((HERE / "results" / f"{d}.json").read_text()) for d in DS if (HERE / "results" / f"{d}.json").exists()}
ARMS = ["C1 continue", "C2 continue, decayed", "C3 continue, averaged", R[0], R[1], "L exact line search"]


def f(x, p=1):
    return f"{100 * x:.{p}f}%"


for opt in ("SGD", "Adam"):
    print(f"\n## Stage 1 = {opt}: two retrains disagree (lower is better); cell = mean over 200 seeds\n")
    for d, o in res.items():
        s1 = o["stage1"][opt]
        print(f"**{d}** (stage 1: {f(s1['stage1']['disagreement'])}, changed {s1['stage1']['changed'][0]:.1f})\n")
        print("| Extra epochs | " + " | ".join(a.replace("risk ", "").replace("continue", "cont.") for a in ARMS) + " | best continuation |")
        print("|---|" + "---|" * (len(ARMS) + 1))
        for B, c in s1["budgets"].items():
            print(f"| {B} | " + " | ".join(f(c["arms"][a]["disagreement"]) for a in ARMS) + f" | {c['best_continuation'].split(' ')[0]} |")
        print()
    print(f"\n## Stage 1 = {opt}: risk arms vs best continuation (difference in disagreement, 95% CI; negative = risk arm better)\n")
    print("| Dataset | Extra epochs | Arm | disagreement | changed decisions / 300-ish (mean) | wrong-decision rate |")
    print("|---|---|---|---|---|---|")
    for d, o in res.items():
        for B, c in o["stage1"][opt]["budgets"].items():
            for r in R:
                t = c["tests"][f"{r} vs {c['best_continuation']}"]
                dd, ch, wr = t["disagreement"], t["changed_mean"], t["wrong"]
                print(f"| {d} | {B} | {r.replace('R risk ', '')} | {100*dd[0]:+.1f} [{100*dd[1]:+.1f}, {100*dd[2]:+.1f}] | {ch[0]:+.1f} [{ch[1]:+.1f}, {ch[2]:+.1f}] | {100*wr[0]:+.1f} [{100*wr[1]:+.1f}, {100*wr[2]:+.1f}] |")
    print(f"\n## Stage 1 = {opt}: wrong-decision rate and per-row harm\n")
    print("| Dataset | Extra epochs | arm | wrong decision | share rows worse | p95 row loss increase | mean step x L | seconds (200 seeds) |")
    print("|---|---|---|---|---|---|---|---|")
    for d, o in res.items():
        for B, c in o["stage1"][opt]["budgets"].items():
            for a in ARMS:
                m = c["arms"][a]
                print(f"| {d} | {B} | {a} | {f(m['wrong'])} | {m['share_rows_worse']:.2f} | {m['p95_row_loss_change']:+.3f} | {m.get('mean_alpha_times_L', float('nan')):.2f} | {m['seconds_all_seeds']:.1f} |")
if "german" in res:
    print("\n## German Credit, worst group wrong-decision rate (groups with >= 30 held-out rows)\n")
    for opt in ("SGD", "Adam"):
        print(f"\nStage 1 = {opt}\n")
        print("| Extra epochs | " + " | ".join(ARMS) + " |")
        print("|---|" + "---|" * len(ARMS))
        for B, c in res["german"]["stage1"][opt]["budgets"].items():
            print(f"| {B} | " + " | ".join(f(c["arms"][a]["worst_group_wrong"]) for a in ARMS) + " |")
