"""Tables for RESULTS.md from results/<dataset>.json."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DS = [d for d in ("german", "compas", "adult", "bank", "pima", "heart") if (HERE / "results" / f"{d}.json").exists()]
res = {d: json.loads((HERE / "results" / f"{d}.json").read_text()) for d in DS}
REG = ["step 1/L", "lr 0.01", "step 1/L + 20 line-search steps"]


def ci(t):
    return f"{t['ratio']:.2f} [{t['ci'][0]:.2f}, {t['ci'][1]:.2f}]" if t["ratio"] == t["ratio"] else "no flips"


out = ["## H1. Does the boundary move more where data is thin? Spearman correlation of the per-row score spread across retrains with leverage and with rarity\n",
       "| Dataset | n train / test | Regime | flip rate | rho(spread, leverage) | rho(spread, k-NN rarity) | rho(spread, rarest-category frequency) |", "|---|---|---|---|---|---|---|"]
for d in DS:
    r = res[d]
    for reg in REG:
        q = r["regimes"][reg]
        cf = f"{q['spearman_std_catfreq']:+.2f}" if "spearman_std_catfreq" in q else "-"
        out.append(f"| {d} | {r['n_train']} / {r['n_test']} | {reg} | {100 * q['flip_rate']:.1f}% | {q['spearman_std_leverage']:+.2f} | {q['spearman_std_knn']:+.2f} | {cf} |")
out += ["\n## H2. At the same distance from the boundary, do rare rows flip more? Rarest 10% (k-NN) against the rest within deciles of |margin|\n",
        "| Dataset | Regime | raw flip: rarest 10% / rest | deciles where rare > rest | pooled ratio rare / rest [95% CI] | partial rho(flip, rarity | margin) | same with leverage: ratio, deciles | same with rarest category: ratio, deciles |",
        "|---|---|---|---|---|---|---|---|"]
for d in DS:
    r = res[d]
    for reg in REG:
        q = r["regimes"][reg]
        k, l, c = q["knn"], q["leverage"], q.get("catfreq")
        cc = f"{ci(c)}, {c['wins']}/10" if c else "-"
        out.append(f"| {d} | {reg} | {100 * k['rare_flip']:.1f}% / {100 * k['rest_flip']:.1f}% | {k['wins']}/10 | {ci(k)} | {q['partial_spearman_flip_knn_given_margin']:+.2f} | {ci(l)}, {l['wins']}/10 | {cc} |")
out += ["\n## Predictors: Spearman among rarity, leverage and distance to the boundary (held-out rows)\n", "| Dataset | rho(k-NN rarity, |margin|) | rho(leverage, |margin|) | rho(k-NN rarity, leverage) |", "|---|---|---|---|"]
for d in DS:
    p = res[d]["predictor_spearman"]
    out.append(f"| {d} | {p['knn_margin']:+.2f} | {p['leverage_margin']:+.2f} | {p['knn_leverage']:+.2f} |")
out += ["\n## Per-decile flip rates at step 1/L (rarest 10% / rest), decile 1 = closest to the boundary\n"]
for d in DS:
    rows = res[d]["regimes"]["step 1/L"]["knn"]["deciles"]
    out.append(f"- **{d}**: " + "; ".join(f"d{x[0] + 1}: {100 * x[1]:.1f}% / {100 * x[2]:.1f}% (n={x[3]})" for x in rows))
print("\n".join(out))
