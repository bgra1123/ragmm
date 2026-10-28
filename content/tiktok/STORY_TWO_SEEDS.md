# "Two Seeds" — story line for the explainer

Two cuts from one shoot. **The awareness cut** (TikTok/Reels/Shorts, about 55 s) makes one claim and stops; the package is not in the video, only in the description. **The full cut** (YouTube, 2–3 min) adds the fix and the tool.

**Logline.** The same woman walks into the same bank with the same papers. In one world she drives to her first job. In the other she takes the 5 a.m. bus. The only thing that differed was a random number inside the bank's model.

**Thesis (the hook, kept).** *Life is a realization of random seeds. So is the model deciding yours.* The body says whose seed it is: the bank's, not hers.

---

## The character

**Ada, 23.** She finished a care-assistant retraining course last month and starts her first job at a clinic outside town in five weeks. The early shift begins at 6 a.m.; the first bus gets there at 7:10. She lives with her parents (on the form: housing *"for free"*), has a small savings account, and the form still says *"unemployed"* because the contract starts next month. She applies for a small loan for a used car to get to work.

Why this profile: every part of it is a group where the experiments found training randomness lands hardest, and it is a composite of real held-out applicants in the public dataset, not a real person:
- housing "for free": 108 of 1,000 applicants; repaying applicants refused 34.6% of the time with standard training against 24.7% without the randomness;
- unemployed at application time: 62 of 1,000; 36.0% against 29.0%;
- under 25: 149 of 1,000.
- A real applicant close to Ada (unemployed, housing for free, small car loan, repaid) was approved by **56 of 100** standard-trained retrains, **74 of 100** after a short polish, **90 of 100** after the deterministic polish. The exactly trained model approves her.

Ada repays, in every world. That is the point: the loan was a good loan in both universes.

---

## Awareness cut (about 55 s). One claim, then stop.

| # | Beat | What we see | Voice-over (plain, calm) | On-screen text |
|---|---|---|---|---|
| 0 | **Hook** (0–2 s) | Black. One line. | — | **"Same life. Different seed."** |
| 1 | **The reason** (2–8 s) | Pre-dawn. Ada at a bus stop in the dark, breath in the cold, phone showing 05:02 and a bus at 07:10. A clinic badge in her hand. | "Ada starts her first job in five weeks. The shift starts at six. The first bus gets there at ten past seven." | — |
| 2 | **The bank** (8–15 s) | Zoom out from the bus stop over the town to the bank. Ada at the clerk's desk; the clerk types; a small card on the screen: her profile as icons (house with family, calendar "starts next month", small piggy bank, a used car). | "So she asks the bank for a small loan for a used car. The clerk types her in. A model decides." | "same person · same papers" |
| 3 | **World A** (15–19 s) | The screen glows green. Ada's hand on a car key. Cut: the car at 5:50 a.m., the clinic lights ahead. | "Approved." | "Universe A · seed 17" |
| 4 | **The split** (19–27 s) | Rewind sound. The frame splits down the middle. Right half: the identical scene, the same keystrokes, the same card. The screen glows red. Ada on the 7:10 bus, late on day one. | "Now rewind. Same bank, same papers, same clerk. The model was retrained last night, with a different random seed. Rejected." | "Universe B · seed 18" |
| 5 | **What changed** (27–35 s) | Both halves shrink; behind them the model as a field of dots with one thin line. Ada's dot sits just right of the line. The line nudges left; her dot turns red. Nudges back; green. | "Nothing about Ada changed. The line did. Every training run draws the line a little differently, and Ada's dot sits close to it." | "the line moved, not Ada" |
| 6 | **The grid** (35–44 s) | Zoom out to a 10 × 10 grid of tiny worlds, each with Ada at the desk. Tiles light green or red until the count reads 56 green, 44 red. | "We trained the same model a hundred times. Ada was approved in fifty-six worlds and rejected in forty-four. She repaid in all of them." | "56 approved · 44 rejected · 100 seeds" |
| 7 | **Why her** (44–52 s) | The grid pulls back into a map of all applicants. Most dots are steady; the flickering ones cluster at the edges, where there are few dots. Two dots at the same distance from the line: the one in the thin region flickers more. | "The line moves most where the model has seen the fewest people like you. At the same distance from the line, the rarest applicants were about one and a half to two times more likely to get a different answer." | "rare in the data = the line moves more for you" |
| 8 | **Close** (52–55 s) | Hard cut to black. One line fades in. Hold two seconds. | — | **"One seed changes. A life changes."** Small below: *"Same model, same data, 100 retrains. German Credit, a public dataset. Not credit advice."* |

No logo, no command, no product name in the video.

**Description (where the package lives):** "I retrained the same credit model a hundred times and watched who flipped. Then I built a small open-source tool that measures this for any model and removes the part that is only luck. Link in bio. Numbers and code: [repo]. German Credit is a public 1990s dataset; this describes how a model behaves, not any bank, and is not credit advice."
**Pinned comment:** "Where do the numbers come from?" → the fact sheet link. "What's the fix?" → the follow-up video.

**Closing lines to A/B test** (one per upload): "One seed changes. A life changes." · "The model rolled a die. She got the bill." · "Nothing about her changed. The seed did." · "Retrained last night. Rejected this morning." · "Same file. Different seed. Different life."

---

## Full cut (YouTube, 2–3 min). Awareness cut, then:

| # | Beat | What we see | Voice-over | On-screen text |
|---|---|---|---|---|
| 9 | **The table** (+30 s) | The real numbers: all applicants 30.4% wrong vs 24.4% without randomness; housing for free 37.8% vs 27.4%; no property 35.4% vs 25.6%; 55 of 300 decisions change per retrain, 284 of 300 at least once in 1,000 retrains. | "This is not one unlucky applicant. Across all 300 held-out applicants, 55 decisions change on every retrain; 284 changed at least once in a thousand." | the table |
| 10 | **The fix** (+15 s) | The hundred lines in the hundred worlds slide together into one. The flicker stops. Ada's tile: 90 green. | "You can't remove the model's own judgement. You can remove the luck. A short exact polish after training makes every seed draw the same line: Ada's approvals go from fifty-six to ninety of a hundred." | "56 → 90 of 100" |
| 11 | **Not oversold** (+20 s) | Trees and networks: averaging seeds; ten times more trees is as good. | "For tree models and networks the fix is averaging seeds, and more trees at a lower learning rate does the same job. The exact model still makes its own mistakes; this removes only the luck." | "removes luck, not judgement" |
| 12 | **The ask** (+8 s) | One card, steady light. The package name and command. | "Ask your model twice. Measure who changes. Remove the part that is only luck." | "edgelift · pip install edgelift" |
| 13 | **Fine print** | Static card. | — | "German Credit, a public 1990s dataset; a composite applicant; statistical behaviour of a model, not an accusation of bias, not credit advice." |

---

## The numbers behind each beat (do not add others)

| Beat | Number | Source |
|---|---|---|
| 6 | 56 / 100 approved (standard training), repaid, exact model approves | `research/groups/RESULTS.md` §3, applicant 2 |
| 7 | the line moves more where data is thin: Spearman of the score's spread across retrains with rarity 0.52–0.90 and with leverage 0.64–0.90 (Heart excepted) | `research/rare_flip/RESULTS.md` |
| 7 | rarest 10% vs rest, flip rate at the same distance from the boundary: German 1.45×, COMPAS 1.39×, Adult 1.79×, Bank 1.86×, Pima 1.63× (Heart too small). Say "about one and a half to two times", not "twice" | `research/rare_flip/RESULTS.md` |
| 7 (groups) | housing "for free": repaying applicants refused 34.6% vs 24.7%; unemployed 36.0% vs 29.0%; under 25 28.8% vs 25.7% | `research/groups/RESULTS.md` §2 |
| 9 | 30.4 / 24.4; 37.8 / 27.4; 35.4 / 25.6; 55 of 300; 284 of 300 | `research/groups/CONCRETE.md`, `research/report/REPORT.md` |
| 10 | 74 / 100 after the short polish; 90 / 100 after the deterministic polish; package: disagreement 15.5% → 0.1%, housing for free wrong decisions 35.1% → 30.1% | `research/groups/RESULTS.md` §3, `research/product/RESULTS.md` |
| 11 | 10× trees matches averaging 10 seeds | `research/product/RESULTS.md` |

## Lines to avoid
- "The bank discriminates" / "the model is biased": not shown and not what the data says. Say "the line moves most where the model has seen the fewest people like you."
- "Ada was wrongly rejected": say "rejected in 44 of 100 worlds; she repaid in all of them."
- "Your loan is a coin flip": most decisions do not flip; the ones near the line do.
- "The fix makes the model fair": say "it removes the luck; the model's own judgement stays."
- Seeds are not fate: the seed is the bank's, not Ada's.

## Production notes
- Beats 1–4 are live-action or AI video (Runway/OpenArt prompts: `RUNWAY_PROMPTS.md`, `../../research/blog/OPENART_KIT.md`); keep the clerk's screen blank and add the icons in the edit.
- Beats 5–7 and 10 are motion graphics from real numbers (`research/blog/make_video.py`, vertical `W, H = 7.2, 12.8`); the grid must fill to exactly 56/44 and then 90/10.
- No real bank name, no real logo, no AI-generated face close-ups; Ada can be an actor or a silhouette.
