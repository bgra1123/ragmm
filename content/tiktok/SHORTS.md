# TikTok / Reels / Shorts scripts (T1–T6)

*Vertical 9:16, 25–45 seconds each. Goal: awareness that ML decisions can change because of training randomness alone, and
that people who are rare in the data are hit hardest. Each short points to the long video ([Y1](../youtube/Y1_PROBLEM.md) or
[Y2](../youtube/Y2_PRODUCT.md)) and the written pieces ([B1](../../research/blog/BLOG.md), [B2](../../docs/TUTORIAL.md)).
Numbers: [`../FACT_SHEET.md`](../FACT_SHEET.md) {IDs}. References: [`../REFERENCES.md`](../REFERENCES.md) [n].*

**Format rules for every short**
- First 2 seconds: the hook as large on-screen text, before any context.
- One number per short. Show the source as small text at the bottom ("German Credit, public dataset [12]").
- Last frame: "Full video: Y1" (or Y2) + the caveat line.
- Caption ends with the references used and `#machinelearning #AI #fairness #datascience`.
- Visuals: the dot grid and bar chart from `research/blog/make_video.py`, re-rendered at 1080×1920 (set `W, H = 7.2, 12.8`),
  or the Sora prompts in `research/blog/VIDEO_PROMPT.md`.

---

## T1 — "Ask the AI twice" {A2, A3}

| Time | On screen | Voice-over |
|---|---|---|
| 0–2 s | **"Same person. Same data. Different answer."** | "Same person. Same data. Different answer." |
| 2–10 s | dot grid, 300 applicants, flickering | "I trained a loan model a thousand times on the exact same data. The only thing I changed was a random number." |
| 10–20 s | "55 of 300 decisions changed. Every time." | "Each time, about one in six applicants got a different decision." |
| 20–30 s | grid fills amber: "284 of 300" | "Over a thousand runs, 284 of 300 people got a different answer at least once." |
| 30–35 s | "Full video: Y1" | "So which model decided your loan? Full breakdown on my channel." |

**Caption:** Same data, same model, different decision. 1,000 retrains of a credit model on a public dataset [12]. Statistical
behaviour of a model, not credit advice. Refs: [4] Marx et al. 2020, [12] UCI German Credit.

## T2 — "Who loses the AI lottery?" {B2, B3, B5}

| Time | On screen | Voice-over |
|---|---|---|
| 0–2 s | **"The AI lottery isn't fair to everyone."** | "This randomness doesn't hit everyone the same." |
| 2–15 s | bars: housing "for free" 37.8% vs 27.4% | "For applicants living in housing 'for free', the model was wrong 37.8% of the time. Without the randomness, 27.4%." |
| 15–25 s | two more bars: no property +9.9, unemployed +5.6 | "No property: plus 10 points. Unemployed: plus 6." |
| 25–35 s | "Few examples → shakier model" | "Why? The model has seen few people like them, so their part of the model moves the most." |
| 35–40 s | "Full video: Y1" | "Full video on my channel." |

**Caption:** Rare in the data means less stable decisions. Public dataset [12]; statistical behaviour of a model, not an
accusation of bias. Refs: [5] Black et al. 2022, [6] D'Amour et al. 2022.

## T3 — "The 20-pass fix" {D1, D6}

| Time | On screen | Voice-over |
|---|---|---|
| 0–2 s | **"15.5% → 0.1%"** | "From fifteen percent to zero point one." |
| 2–15 s | code: `StabilityAdapter(model).fit(X, y)` | "If your model is a logistic regression, finish training with 20 careful full passes over the data." |
| 15–28 s | bar shrinks 15.5% → 0.1% | "Two retrains used to disagree on 15.5% of applicants. After the polish: 0.1%." |
| 28–35 s | "cost: 0.05 seconds" | "It took five hundredths of a second." |
| 35–40 s | "How: Y2 / tutorial B2" | "Tutorial linked in my bio." |

**Caption:** Convex models have one right answer; the polish walks every seed there. Refs: [1] Can & Gürbüzbalaban 2025
(OMS, arXiv:2204.11292). Benchmark: research/product/RESULTS.md.

## T4 — "Who gains from the fix" {D3}

| Time | On screen | Voice-over |
|---|---|---|
| 0–2 s | **"The fix helps the people who were hit hardest."** | "Here's who gains when you remove the randomness." |
| 2–20 s | three before/after bars: unemployed 41.8 → 36.3, housing for free 35.1 → 30.1, no property 32.0 → 27.9 | "Unemployed applicants: wrong decisions from 41.8% to 36.3%. Housing for free: 35.1 to 30.1. No property: 32 to 27.9." |
| 20–30 s | "Overall: 27.3% → 25.4%" | "Overall accuracy didn't get worse. It got better." |
| 30–38 s | caveat | "It fixes the randomness, not everything the model learned. Full video on my channel." |

**Caption:** Same model, polished after training. One public dataset [12]; not credit advice. Refs: [1], [12].

## T5 — "XGBoost changes its mind too" {D4, D7}

| Time | On screen | Voice-over |
|---|---|---|
| 0–2 s | **"XGBoost changes its mind too."** | "Think XGBoost is deterministic? Change the seed." |
| 2–12 s | "6.5% of decisions flip between retrains" | "With subsampling on, two retrains disagreed on 6.5% of applicants." |
| 12–25 s | 10 small models merging into one: "6.5% → 2.1%" | "Average ten seeds and it drops to 2.1%." |
| 25–35 s | "or: 10× more trees, lower learning rate" | "Same compute with ten times more trees works just as well. Pick one, and measure it." |
| 35–40 s | "Full video: Y2" | "Benchmarks on my channel." |

**Caption:** Seed averaging = bagging [9] / deep ensembles [10]. XGBoost [13]. Benchmark: research/product/RESULTS.md.

## T6 — "Averages hide the worst days" {F1, A2}

| Time | On screen | Voice-over |
|---|---|---|
| 0–2 s | **"Your model's average is lying to you."** | "Your model's average accuracy hides its worst days." |
| 2–15 s | histogram of retrains, right tail highlighted | "Retrain it many times and some runs are much worse than others." |
| 15–28 s | "5% of people carry 20–58% of the extra error" | "Across six datasets, the 5% of people most affected carried a fifth to more than half of the extra error in the bad runs." |
| 28–38 s | paper title card | "That's why my research measures risk, not averages. Paper linked." |

**Caption:** Entropic value-at-risk [2], RA-GMM [1]. Refs: [1] Can & Gürbüzbalaban 2025, OMS, arXiv:2204.11292;
[2] Ahmadi-Javid 2012.

---

## Posting order and cross-links

1. T1 → T2 (problem) → publish Y1 the same week; pin a comment linking Y1 under both shorts.
2. T6 (why risk, links the paper) between Y1 and Y2.
3. T3 → T4 → T5 (fix) → publish Y2 and the tutorial B2; each short links Y2 and B2.
