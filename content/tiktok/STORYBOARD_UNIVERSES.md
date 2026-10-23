# Storyboard: "Two universes, one applicant" (TikTok explainer, about 38 seconds, 9:16)

**Format:** short-form explainer (educational, organic). **One message:** randomness in training moves the model's decision boundary, so applicants near the boundary can be approved in one training run and rejected in another.
**Numbers** (all from `content/FACT_SHEET.md` and `research/groups/RESULTS.md`, German Credit, a public 1990s dataset): 55 of 300 decisions change per retrain (18%); 284 of 300 applicants get a different answer at least once in 1,000 retrains;
example applicant (repaid, exact model's probability of repaying 0.57): approved in 56 of 100 standard-trained retrains, 74 of 100 after a short polish, 90 of 100 after a deterministic polish.
**Key frame (generated with Runway, 9:16):** two parallel universes, the same blank application card glowing green on the left and red on the right, a faint boundary line at a slightly different position in each. No text in the image; add the text in the editor.

| # | Time | Visual | On-screen text | Voice-over | Sound |
|---|---|---|---|---|---|
| 1 | 0-3 s | The key frame: one application card, the screen splits into two universes, left green "approved", right red "rejected". | "Same person. Same data." | "Same person. Same data. Two universes." | low hum, a whoosh at the split |
| 2 | 3-7 s | Back to one universe. Applicants shown as dots along a score line; a vertical line is the decision boundary: right of it approved (green), left rejected (red). One dot, "you", sits just right of the line. | "The model gives everyone a score." | "A loan model gives everyone a score. A line decides who is approved." | soft tick |
| 3 | 7-13 s | A counter "seed 1, seed 2, seed 3". Each time the line shifts a little left or right; the highlighted dot flips green, red, green. | "Only the random seed changes." | "Train the same model again with a different random seed, and the line moves a little. Your dot didn't move. The line did." | click per seed |
| 4 | 13-21 s | A 10 x 10 grid of tiny universes, each with its own boundary; the applicant's tile is green or red. The counter fills to 56 green, 44 red. | "100 universes. 56 approve. 44 reject." | "In a hundred universes, this applicant, who went on to repay, is approved in 56 and rejected in 44. Nothing about them changed." | counter ticking up |
| 5 | 21-28 s | Zoom on the line: dots far from it keep the same colour in every universe; dots near it flicker. A shaded band around the line is labelled. | "Far from the line: same answer. Near it: depends on the training run." | "People far from the line get the same answer every time. People near it depend on luck: the luck of the training run." | flicker sound only on the band |
| 6 | 28-33 s | Counter overlay: "55 of 300" decisions flip per retrain; "284 of 300" flip at least once in 1,000 retrains. | "55 of 300 decisions change per retrain" | "In my tests, a standard-trained credit model changed 55 of 300 decisions between retrains. 284 got a different answer at least once." | beat |
| 7 | 33-37 s | The 100 universes' boundaries slide together into one line; the flicker stops; the applicant tile turns steady green (after polish: 74 to 90 of 100). | "After a short polish: one boundary." | "Remove the randomness, and the universes start to agree." | soft resolve chord |
| 8 | 37-40 s | Closing frame: one card, steady light; caption. | "Ask your model twice." | "Ask your model twice." | silence, then a click |

**Last-frame caption (small):** "German Credit, a public dataset; statistical behaviour of a model, not an accusation of bias, not credit advice. Full video: Y1."

## What is real and what is schematic
| Element | Status |
|---|---|
| 56 / 74 / 90 of 100 approvals for the example applicant; 55 of 300; 284 of 300 | Real, measured (cite as "in my tests") |
| The score line, dots and tiles | Schematic. Say "illustration" in the caption or the first frame |
| The applicant | A real held-out applicant, chosen as one of the three with the highest chance of refusal in an affected group, so label it "an example", not "a typical applicant" |
| "Universes" | A metaphor for retrains with different seeds; do not suggest the model is a person or that the applicant was treated unfairly |

## Production
* **Picture:** the Runway key frame for shot 1 and the closing frame; for shots 2 to 7 either animate it in a motion-graphics tool, or render the dot-and-line scenes with `research/blog/make_video.py` (vertical: `W, H = 7.2, 12.8`).
* **AI video (optional):** use the OpenArt prompts in `research/blog/OPENART_KIT.md` for shot 1 only; keep the numbers and the dot diagrams as real graphics.
* **Captions:** burn in the voice-over; most viewers watch without sound.
* **Hook variants to test:** "Same applicant. Two universes." / "This person was approved and rejected by the same model." / "Why did the model say no?"
* **Wording:** say "depends on the training run", not "coin flip"; do not say the model discriminates; do not say the applicant was wrongly rejected, only that the decision changed.
