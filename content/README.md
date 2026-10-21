# Content hub

**Format: explainers.** The videos are educational explainers (short-form for TikTok, long-form for YouTube): organic content that teaches one finding and points to the product softly. They are not ads and not product demos; product demos are a separate format for later.

One story, told at three lengths. Every piece links to the others, cites the paper, and takes its numbers from one fact sheet.

| ID | Piece | Channel | Job | File |
|---|---|---|---|---|
| T1–T6 | Short videos (25–45 s) | TikTok / Reels / Shorts | awareness: decisions change with the random seed; rare groups are hit hardest | [`tiktok/SHORTS.md`](tiktok/SHORTS.md) |
| Y1 | "I retrained a credit model 1,000 times" (~12 min) | YouTube | the problem and the experiments | [`youtube/Y1_PROBLEM.md`](youtube/Y1_PROBLEM.md) |
| Y2 | "A one-line fix for models that change their mind" (~14 min) | YouTube | the product, the benchmark, gains and costs | [`youtube/Y2_PRODUCT.md`](youtube/Y2_PRODUCT.md) |
| B1 | "Same applicant, same data, different answer" | Blog | the story in writing | [`../research/blog/BLOG.md`](../research/blog/BLOG.md) |
| B2 | "Make your classifier give the same answer twice" | Blog | tutorial: how to use the adapter on your model | [`../docs/TUTORIAL.md`](../docs/TUTORIAL.md) |
| V0 | 51-second animated explainer | any | ready-made visuals; re-render vertically for shorts | [`../research/blog/video/same_applicant.mp4`](../research/blog/video/same_applicant.mp4), [`../research/blog/make_video.py`](../research/blog/make_video.py) |
| P0 | Sora / image prompts | any | generated b-roll and thumbnails | [`../research/blog/VIDEO_PROMPT.md`](../research/blog/VIDEO_PROMPT.md) |

## How the pieces link

```
T1, T2, T6 ──► Y1 ──► B1
                │
T3, T4, T5 ──► Y2 ──► B2 (tutorial) ──► package (risk_averse_tuning.StabilityAdapter)
                │
all ─────────► paper [1] and FACT_SHEET.md
```

* **Paper:** Can & Gürbüzbalaban (2025), *Entropic Risk-Averse Generalized Momentum Methods*, Optimization Methods and
  Software 40(6), doi:10.1080/10556788.2025.2549356, arXiv:2204.11292. Every piece cites it as [1].
* **References:** [`REFERENCES.md`](REFERENCES.md), one numbered list shared by all pieces.
* **Numbers:** [`FACT_SHEET.md`](FACT_SHEET.md), each fact with an ID, its source file, and the pieces that use it. Add a
  fact there before using a new number.

## Rules for every piece

1. Every number comes from the fact sheet, with its source.
2. Every piece says at least one caveat: one public 1990s dataset; statistical behaviour, not an accusation of bias; not
   legal or credit advice; the adapter fixes seed randomness, not what the model learned.
3. Report what did not work: RA-GMM as a trainer tied tuned SGD (C1); equal compute (more trees, longer SGD) matched the
   adapter (D7).
4. Say "chance of a wrong decision for [group]", not vague words like "coin flip".
