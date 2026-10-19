# Video prompts for OpenAI video/image generation (Sora / image models)

Concept: "Same applicant, same data, different answer" (see BLOG.md). 45-60 s explainer, 5 shots.
Tip: AI video models draw text unreliably. Generate the visuals without words, then add the numbers
(from FACT_CHECK.md) as captions/voice-over in your editor (CapCut, DaVinci, VS Code is not needed).

## Master prompt (paste into Sora; ask for a 15-20 s clip per shot)
A clean, modern explainer video in a calm editorial style: deep navy background, warm amber and soft teal
accents, subtle film grain, smooth slow camera moves, no on-screen text, no logos, no real people's faces.
Shot 1: a single loan application card on a desk; a hand places an identical copy next to it, and
a dozen identical copies fan out in a row.
Shot 2: each copy passes through a glowing translucent machine; most stamps land green "approved" glows, a few
land red; the red ones flicker between green and red as if undecided.
Shot 3: camera pulls back to a grid of 300 small cards; about one in six pulse amber, clustered in a few
highlighted sections of the grid.
Shot 4: a second, smaller polishing device is added to the machine; the amber pulsing calms and the cards
settle to steady colours.
Shot 5: a quiet closing frame: one card, steady, with a thin light line connecting it to a measuring gauge.
Cinematic, 16:9, 24 fps, soft ambient music mood, no dialogue.

## Per-shot prompts (more control)
1. "Overhead macro shot of identical paper loan application forms being duplicated one after another on a dark navy desk, amber desk lamp, shallow depth of field, slow dolly, no readable text."
2. "A softly glowing glass machine stamps identical forms; most stamps glow green, a few forms flicker between green and red, abstract and minimal, no text."
3. "Top-down grid of 300 small cards, roughly one in six pulsing amber, grouped in clusters, camera slowly rising, flat vector-like look."
4. "The same grid after a small polishing device attaches to the machine: amber pulses fade to steady green and red, calm mood."
5. "Single card in the centre, steady light, thin line to a minimal gauge dial, fade to navy."

## Image prompts (thumbnails / carousel)
- Thumbnail: "Two identical loan application cards side by side, one stamped green, one stamped red, dramatic split lighting, navy and amber palette, large empty space at top for a title, no text."
- Carousel: "Minimalist flat illustration of 300 dots in a grid, 55 dots amber, rest grey, clean infographic style, no text."

## Voice-over and captions (add in the editor; every number is in FACT_CHECK.md)
1. "We trained the same model on the same data, with a different random seed."
2. "Out of 300 applicants, 55 got a different decision depending on the seed. About one in six."
3. "The chance of a wrong decision was higher for some groups, for example 37.8% versus 27.4% for applicants living in housing 'for free'."
4. "A short, deterministic polish step cut that gap, to 31.5% in that group."
5. "Ask your model twice. See how often it changes its mind."
Disclaimer caption: "Statistical behaviour of a model on a public dataset (German Credit); not an accusation of bias and not credit advice."

## Sora tips
- Keep clips short and stitch; keep the palette sentence in every prompt for consistency.
- Avoid real brands, real bank names and faces.
- If "no text" is ignored, crop or blur the text region in editing.
