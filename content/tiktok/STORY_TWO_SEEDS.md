# "Two Seeds" — story line for the explainer (TikTok 60–75 s; a 2–3 min cut for YouTube)

**Logline.** The same woman walks into the same bank with the same papers. In one world she drives to her first job. In the other she takes the 5 a.m. bus. The only thing that differed was a random number inside the bank's model.

**Thesis (your line, kept).** *Life is a realization of random seeds. So is a model.* A model's decision line is not drawn once; every training run draws it slightly differently, and it moves most where the model has seen the fewest people like you.

**What the video promotes.** Not a fairer bank. A measurement: ask your model twice, see who changes, remove the part that is only luck. (The package does exactly those three things.)

---

## The character

**Ada, 23.** She finished a care-assistant retraining course last month and starts her first job at a clinic outside town in five weeks. The early shift begins at 6 a.m.; the first bus gets there at 7:10. She lives with her parents (on the form: housing *"for free"*), has a small savings account, and the form still says *"unemployed"* because the contract starts next month. She applies for a small loan for a used car to get to work.

Why this profile: every part of it is a group where the real experiments found training randomness lands hardest, and it is a composite of real held-out applicants in the public dataset, not a real person:
- housing "for free": 108 of 1,000 applicants; repaying applicants refused 34.6% of the time with standard training against 24.7% without the randomness;
- unemployed at application time: 62 of 1,000; 36.0% against 29.0%;
- under 25: 149 of 1,000.
- A real applicant close to Ada (unemployed, housing for free, small car loan, repaid) was approved by **56 of 100** standard-trained retrains, **74 of 100** after a short polish, **90 of 100** after the deterministic polish. The exactly trained model approves her.

Ada repays, in every world. That is the point: the loan was a good loan in both universes.

---

## The story in beats

| # | Beat | What we see | Voice-over (plain, calm) | On-screen text |
|---|---|---|---|---|
| 1 | **The reason** (0–6 s) | Pre-dawn. Ada at a bus stop in the dark, breath in the cold, phone showing 05:02 and a bus at 07:10. A clinic badge in her hand. | "Ada starts her first job in five weeks. The shift starts at six. The first bus gets there at ten past seven." | — |
| 2 | **The bank** (6–14 s) | Zoom out from the bus stop over the town to the bank. Ada at the clerk's desk; the clerk types; a small card on the screen: her profile as icons (house with family, calendar "starts next month", small piggy bank, a used car). | "So she asks the bank for a small loan for a used car. The clerk types her in. A model decides." | "same person · same papers" |
| 3 | **World A** (14–19 s) | The screen glows green. Ada's hand on a car key. Cut: the car at 5:50 a.m., the clinic lights ahead. | "Approved." | "Universe A · seed 17" |
| 4 | **The split** (19–27 s) | Rewind sound. The frame splits down the middle. Right half: the identical scene, the same keystrokes, the same card. The screen glows red. Ada on the 7:10 bus, late on day one. | "Now rewind. Same bank, same papers, same clerk. The model was retrained last night, with a different random seed. Rejected." | "Universe B · seed 18" |
| 5 | **What changed** (27–36 s) | Both halves shrink; behind them the model as a field of dots with one thin line. Ada's dot sits just right of the line. The line nudges left; her dot turns red. Nudges back; green. | "Nothing about Ada changed. The line did. Every training run draws the line a little differently, and Ada's dot sits close to it." | "the line moved, not Ada" |
| 6 | **The grid** (36–46 s) | Zoom out to a 10 × 10 grid of tiny worlds, each with Ada at the desk. Tiles light green or red until the count reads 56 green, 44 red. | "We trained the same model a hundred times. Ada was approved in fifty-six worlds and rejected in forty-four. She repaid in all of them." | "56 approved · 44 rejected · 100 seeds" |
| 7 | **Why her** (46–56 s) | The grid pulls back further into a map of all applicants. Most dots are steady; the flickering ones cluster at the edges, where there are few dots. Caption groups: housing for free, unemployed at application, under 25. | "The line moves most where the model has seen the fewest people like you. Ada is rare: living with her parents, between jobs, young. For people like her, one refusal in three came from the training run, not from her file." | "rare in the data = closest to the moving line" |
| 7b | **(after the test, if supported)** | Two dots at the same distance from the line; the rare one flickers more. | "At the same distance from the line, the rarest applicants were [X] times more likely to get a different answer." | "[X]× — see RESULTS" |
| 8 | **The fix** (56–66 s) | The hundred lines in the hundred worlds slide together into one. The flicker stops. Ada's tile: 90 green. | "You can't remove the model's own judgement. You can remove the luck. A short exact polish after training makes every seed draw the same line: Ada's approvals go from fifty-six to ninety of a hundred." | "56 → 90 of 100" |
| 9 | **The ask** (66–72 s) | One card, steady light. The package name and command. | "Ask your model twice. Measure who changes. Remove the part that is only luck." | "seedproof · pip install" (working name) |
| 10 | **Fine print** (72–75 s) | Static card. | — | "German Credit, a public 1990s dataset; a composite applicant; statistical behaviour of a model, not an accusation of bias, not credit advice." |

**YouTube cut (2–3 min):** add after beat 7 a 30-second block with the real table (all applicants 30.4% wrong vs 24.4%; housing for free 37.8% vs 27.4%; no property 35.4% vs 25.6%), and after beat 8 a 20-second block on trees and networks (average seeds; 10× more trees is as good) so the claim is not oversold.

---

## The numbers behind each beat (do not add others)

| Beat | Number | Source |
|---|---|---|
| 6 | 56 / 100 approved (standard training), repaid, exact model approves | `research/groups/RESULTS.md` §3, applicant 2 |
| 7 | housing "for free": repaying applicants refused 34.6% vs 24.7% (≈ one refusal in three from the run: 9.9 of 34.6 points = 29%) | `research/groups/RESULTS.md` §2 |
| 7 | unemployed: 36.0% vs 29.0%; under 25: 28.8% vs 25.7% | same |
| 7 | 5% most affected people carry 20–58% of the extra loss; they are high-leverage / rare | `research/tail_analysis/RESULTS.md` |
| 7b | ratio of flip rate, rarest 10% vs rest, at the same distance from the boundary | `research/rare_flip/RESULTS.md` (running now) |
| 8 | 74 / 100 after the short polish; 90 / 100 after the deterministic polish | `research/groups/RESULTS.md` §3 |
| 8 | package: SGD logistic disagreement 15.5% → 0.1%; housing for free wrong decisions 35.1% → 30.1% | `research/product/RESULTS.md` |
| YouTube | 55 of 300 decisions change per retrain; 284 of 300 at least once | `research/report/REPORT.md` |

## Lines to avoid
- "The bank discriminates" / "the model is biased": not shown and not what the data says. Say "the line moves most where the model has seen the fewest people like you."
- "Ada was wrongly rejected": say "rejected in 44 of 100 worlds; she repaid in all of them."
- "The fix makes the model fair": say "it removes the luck; the model's own judgement stays."
- Seeds are not fate: the seed is the bank's, not Ada's. "Life is a realization of random seeds" is the hook; the body says *whose* seed.

## Production notes
- Beats 1–4 are live-action or AI video (Runway/OpenArt prompts: `RUNWAY_PROMPTS.md`, `../../research/blog/OPENART_KIT.md`); keep the clerk's screen blank and add the icons in the edit.
- Beats 5–8 are motion graphics from real numbers (`research/blog/make_video.py`, vertical `W, H = 7.2, 12.8`); the grid must fill to exactly 56/44 and then 90/10.
- No real bank name, no real logo, no face close-ups generated by AI; Ada can be an actor or a silhouette.
