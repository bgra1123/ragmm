"""Pool results/split*.json (rows-weighted) into results/pooled.json and print Markdown tables."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
KEYS = ("disagreement", "unstable_share", "changes_mean", "changes_worst5", "wrong", "wrong_majority", "wrong_excess")


def pool(items):
    n = sum(i["rows"] for i in items)
    out = {"rows": n}
    for k in KEYS:
        if k.startswith("changes"):
            out[k] = sum(i[k] for i in items) / len(items)          # per retrain, per split of 300
        else:
            out[k] = sum(i[k] * i["rows"] for i in items) / n
    return out


splits = [json.loads(p.read_text()) for p in sorted((HERE / "results").glob("split*.json"))]
names = list(splits[0])
pooled = {}
for nm in names:
    summ = pool([s[nm]["summary"] for s in splits])
    summ["fit_seconds"] = sum(s[nm]["summary"]["fit_seconds"] for s in splits) / len(splits)
    groups = {}
    for g in splits[0][nm]["groups"]:
        items = [s[nm]["groups"][g] for s in splits if g in s[nm]["groups"]]
        groups[g] = pool(items)
    pooled[nm] = dict(summary=summ, groups=groups)
(HERE / "results" / "pooled.json").write_text(json.dumps(pooled, indent=2))

print("| Model | Two retrains disagree | Unstable rows | Changed per retrain (avg / worst 5%) | Wrong (one retrain) | Due to randomness | Seconds per fit |")
print("|---|---|---|---|---|---|---|")
for nm, v in pooled.items():
    s = v["summary"]
    print(f"| {nm} | {s['disagreement']:.1%} | {s['unstable_share']:.1%} | {s['changes_mean']:.1f} / {s['changes_worst5']:.1f} | "
          f"{s['wrong']:.1%} | {s['wrong_excess']:+.1%} | {s['fit_seconds']:.2f} |")
base = ["SGD logistic", "XGBoost", "LightGBM", "Random forest", "MLP"]
adapt = ["SGD logistic + adapter (polish)", "XGBoost + adapter (average 10)", "LightGBM + adapter (average 10)",
         "Random forest + adapter (average 10)", "MLP + adapter (average 10)"]
for b, a in zip(base, adapt):
    print(f"\n### {b}: groups (n >= 30), before -> after")
    print("| Group | Rows | Wrong one retrain | Due to randomness | Unstable rows |")
    print("|---|---|---|---|---|")
    gb = pooled[b]["groups"]
    for g in sorted(gb, key=lambda g: -gb[g]["wrong_excess"]):
        x, y = gb[g], pooled[a]["groups"][g]
        if x["rows"] < 30:
            continue
        print(f"| {g} | {x['rows']} | {x['wrong']:.1%} -> {y['wrong']:.1%} | {x['wrong_excess']:+.1%} -> {y['wrong_excess']:+.1%} | "
              f"{x['unstable_share']:.1%} -> {y['unstable_share']:.1%} |")
