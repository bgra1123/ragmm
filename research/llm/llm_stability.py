"""L1/L2 from PLAN.md: decision instability of an LLM on German Credit rendered as text.

  python llm_stability.py --backend mock --n 100 --samples 10          # pipeline check, no model needed
  python llm_stability.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --n 300 --samples 20
  python llm_stability.py --backend anthropic --model <model id> --n 200 --samples 10

Output: results/<tag>/raw.csv (one row per applicant x variant x sample) and summary.json / SUMMARY.md.
Groups are the pre-registered ones from research/groups. Statements describe model behaviour on a public
dataset, not bias accusations and not credit advice.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "model_multiplicity" / "data" / "german_credit.csv"
TARGET = "creditability"
GROUPS = {
    "unemployed now": lambda d: d.present_employment_since == "unemployed",
    "housing for free": lambda d: d.housing == "for free",
    "no property": lambda d: d.property == "unknown / no property",
    "checking < 0 DM": lambda d: d.status_of_existing_checking_account == "... < 0 DM",
    "age < 25": lambda d: d.age_in_years < 25,
    "large and long loan": lambda d: (d.credit_amount > 4000) & (d.duration_in_month > 24),
}
INSTR = [
    "You are a loan officer. Decide whether to APPROVE or DENY this loan application. Answer with one letter: A = approve, B = deny.",
    "Review the application below as a credit analyst. Will the applicant repay? Reply A if the loan should be approved, B if it should be denied.",
    "Based only on the application, choose: A (approve the loan) or B (deny the loan). Reply with the letter only.",
]


def render(row, order):
    cols = [c for c in row.index if c != TARGET]
    cols = [cols[i] for i in order]
    return "\n".join(f"- {c.replace('_', ' ')}: {row[c]}" for c in cols)


def make_prompt(row, variant, ncols):
    """variant = (instruction index, field-order seed). order seed 0 keeps the original order."""
    ins, oseed = variant
    order = np.arange(ncols) if oseed == 0 else np.random.default_rng(oseed).permutation(ncols)
    return f"{INSTR[ins]}\n\nApplication:\n{render(row, order)}\n\nAnswer:"


class Mock:
    """Deterministic fake model: p(approve) depends on a few fields plus prompt-specific noise."""
    def __init__(self, seed=0):
        self.seed = seed

    def p_approve(self, prompt, row):
        h = (__import__("zlib").crc32(prompt.encode()) % 1000) / 1000.0
        base = 0.65 - 0.25 * (row["housing"] == "for free") - 0.2 * (row["status_of_existing_checking_account"] == "... < 0 DM")
        return float(np.clip(base + 0.3 * (h - 0.5), 0.02, 0.98))


class HF:
    def __init__(self, model):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).eval()
        self.ids = [self.tok.encode(x, add_special_tokens=False)[0] for x in (" A", "A", " B", "B")]
        self.chat = self.tok.chat_template is not None

    def p_approve(self, prompt, row=None):
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True) if self.chat else prompt
        x = self.tok(text, return_tensors="pt")
        with self.torch.no_grad():
            lg = self.m(**x).logits[0, -1]
        a = self.torch.logsumexp(lg[[self.ids[0], self.ids[1]]], 0)
        b = self.torch.logsumexp(lg[[self.ids[2], self.ids[3]]], 0)
        return float(self.torch.sigmoid(a - b))


class Anthropic:
    """Real sampling through the API; returns a Bernoulli estimate from n_draw draws per prompt."""
    def __init__(self, model, n_draw=1):
        import anthropic
        self.c, self.model, self.n = anthropic.Anthropic(), model, n_draw

    def draw(self, prompt, temperature):
        r = self.c.messages.create(model=self.model, max_tokens=3, temperature=temperature,
                                   messages=[{"role": "user", "content": prompt}])
        return 1.0 if r.content[0].text.strip().upper().startswith("A") else 0.0


def run(args):
    df = pd.read_csv(DATA)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=args.n, replace=False)
    sub = df.iloc[idx].reset_index(drop=True)
    ncols = df.shape[1] - 1
    variants = [(i, s) for i in range(len(INSTR)) for s in range(args.orders)]
    if args.backend == "mock":
        model = Mock()
    elif args.backend == "hf":
        model = HF(args.model)
    else:
        model = Anthropic(args.model)
    rows = []
    for i, r in sub.iterrows():
        truth = int(r[TARGET] == "good")
        for v in variants:
            prompt = make_prompt(r, v, ncols)
            if args.backend == "anthropic":
                draws = [model.draw(prompt, args.temperature) for _ in range(args.samples)]
            else:
                p = model.p_approve(prompt, r)
                pt = 1 / (1 + np.exp(-np.log(p / (1 - p)) / args.temperature))   # temperature on the two-option logit
                draws = (rng.random(args.samples) < pt).astype(float).tolist()
                rows.append(dict(i=i, truth=truth, instr=v[0], order=v[1], sample=-1, approve=float(p >= 0.5), p=p))
            rows += [dict(i=i, truth=truth, instr=v[0], order=v[1], sample=k, approve=d, p=np.nan) for k, d in enumerate(draws)]
        if i % 10 == 0:
            print("applicant", i, "/", args.n, flush=True)
    raw = pd.DataFrame(rows)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "raw.csv", index=False)
    sub.to_csv(out / "applicants.csv", index=False)
    summarize(raw, sub, out, args)


def summarize(raw, sub, out, args, thr=0.10):
    s = raw[raw["sample"] >= 0]
    # per applicant: fixed-prompt sampling noise (variant 0) and prompt-variant noise (all variants)
    base = s[(s.instr == 0) & (s.order == 0)].groupby("i").approve
    maj = base.mean().round()
    disagree_sampling = base.apply(lambda x: min(x.mean(), 1 - x.mean())).rename("sampling")
    allv = s.groupby("i").approve.mean()
    majall = allv.round()
    dis_all = s.assign(m=s.i.map(majall)).assign(diff=lambda d: d.approve != d.m).groupby("i")["diff"].mean()
    truth = s.groupby("i").truth.first()
    unstable = dis_all >= thr
    res = {"n_applicants": int(len(truth)), "unstable_share_all_randomness": float(unstable.mean()),
           "unstable_share_sampling_only": float((disagree_sampling >= thr).mean()),
           "mean_disagreement_with_majority": float(dis_all.mean())}
    # error rate per single run vs majority-vote of k runs
    err_single = (s.approve != s.truth).groupby(s.i).mean()
    res["single_run_error"] = float(err_single.mean())
    res["majority_all_runs_error"] = float((majall != truth).mean())
    res["greedy_error"] = float((raw[raw["sample"] == -1].groupby("i").approve.mean().round() != truth).mean()) if (raw["sample"] == -1).any() else None
    rng = np.random.default_rng(1)
    voting = {}
    for k in (1, 3, 5, 9):
        errs = []
        for _ in range(50):
            pick = s.groupby("i").approve.apply(lambda x: rng.choice(x.values, size=k, replace=False).mean() >= 0.5 if len(x) >= k else np.nan)
            errs.append(float((pick.astype(float) != truth).mean()))
        voting[k] = float(np.mean(errs))
    res["voting_error_by_k"] = voting
    # L2: agreement from 5 runs predicts instability in the other runs
    agree5 = s.groupby("i").approve.apply(lambda x: abs(2 * rng.choice(x.values, size=min(5, len(x)), replace=False).mean() - 1))
    from sklearn.metrics import roc_auc_score
    res["agreement5_auc_for_unstable"] = float(roc_auc_score(unstable, -agree5)) if 0 < unstable.sum() < len(unstable) else None
    # groups
    grp = {}
    for g, f in GROUPS.items():
        m = f(sub).to_numpy()
        if m.sum() >= 15:
            grp[g] = {"n": int(m.sum()), "wrong_single_run": float(err_single[m].mean()),
                      "wrong_majority": float((majall != truth)[m].mean()),
                      "excess_due_to_randomness": float(err_single[m].mean() - (majall != truth)[m].mean()),
                      "unstable_share": float(unstable[m].mean())}
    rest = ~np.any([f(sub).to_numpy() for f in GROUPS.values()], axis=0)
    res["groups"], res["not_in_any_group"] = grp, {"n": int(rest.sum()), "wrong_single_run": float(err_single[rest].mean()), "unstable_share": float(unstable[rest].mean())}
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    lines = [f"# LLM stability summary ({args.backend} {args.model or ''})", "",
             f"- Applicants: {res['n_applicants']}; share unstable (>=10% of runs differ from majority): {res['unstable_share_all_randomness']:.1%}",
             f"- Single-run error {res['single_run_error']:.1%}; majority of all runs {res['majority_all_runs_error']:.1%}",
             f"- Error by voting size k: " + ", ".join(f"k={k}: {v:.1%}" for k, v in voting.items()),
             f"- Agreement-of-5 AUC for identifying unstable applicants: {res['agreement5_auc_for_unstable']}", "",
             "| group | n | wrong (single run) | wrong (majority) | due to randomness | unstable |", "|---|---|---|---|---|---|"]
    lines += [f"| {g} | {v['n']} | {v['wrong_single_run']:.1%} | {v['wrong_majority']:.1%} | {v['excess_due_to_randomness']:+.1%} | {v['unstable_share']:.1%} |" for g, v in grp.items()]
    lines += ["", "Statistical behaviour of a model on a public dataset; not an accusation of bias and not credit advice."]
    (out / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "hf", "anthropic"], default="mock")
    ap.add_argument("--model", default="")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--orders", type=int, default=2, help="field-order variants per instruction")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or str(HERE / "results" / f"{a.backend}_{a.model.replace('/', '_') or 'none'}")
    run(a)
