# OpenArt kit: "Same applicant, same data, different answer"

Converted from `VIDEO_PROMPT.md` for [OpenArt](https://openart.ai/ai-video-generator/). OpenArt's video tool offers text-to-video and image-to-video and lists several
video models (for example Veo, Kling, Seedance, Wan, Sora). **I could not confirm from OpenArt's own pages which settings each model exposes** (negative prompt,
duration, aspect ratio, seed), so the settings below are the usual ones; use what your chosen model shows and ignore the rest.

## Workflow that works best on OpenArt

1. **Make one keyframe image per shot** with an image model (text-to-image). Keep the same style block in every prompt so the shots look like one film.
2. **Animate each keyframe** with image-to-video, using the motion prompt for that shot. Image-to-video gives you more control over layout than text-to-video.
3. **Stitch and add text in an editor** (CapCut, DaVinci Resolve): AI video models draw text badly, so every prompt says "no text". The numbers go in as captions.
4. Export two versions: **16:9** for YouTube (Y1, Y2) and **9:16** for TikTok / Reels / Shorts (T1-T6). Generate the keyframes in the target ratio, do not crop afterwards.

## Settings to use

| Setting | YouTube | TikTok / Reels / Shorts |
|---|---|---|
| Aspect ratio | 16:9 | 9:16 |
| Clip length | 5-8 s per shot (whatever the model offers); stitch 5 shots | 3-5 s per shot; 6-8 shots for a 25-35 s short |
| Image resolution | at least 1280 x 720 | at least 720 x 1280 |
| Negative prompt (if offered) | see below | see below |
| Seed (if offered) | fix it for retries of the same shot | same |
| Camera motion | slow only (dolly, push-in, rise); fast moves distort | same |

**Style block (paste at the start of every image prompt):**

```
Flat editorial motion-graphics style, deep navy background, warm amber and soft teal accents, soft studio lighting, subtle film grain, minimal clean composition, no text, no letters, no numbers, no logos, no faces
```

**Negative prompt (paste where offered):**

```
text, letters, numbers, watermark, logo, brand name, human face, realistic person, hands with extra fingers, blurry, distorted, cluttered, low quality, oversaturated
```

## The five shots (image prompt, then motion prompt)

### Shot 1: the same application, copied
- **Image prompt:** `[style block] Overhead macro view of identical paper loan application forms lined up in a row on a dark navy desk, one amber desk lamp, shallow depth of field, forms show only blank lines, no readable text.`
- **Motion prompt (image-to-video):** `Slow dolly along the row of identical forms; one more identical form slides in from the left and settles next to the others. Calm, steady, no camera shake.`

### Shot 2: the machine and the flicker
- **Image prompt:** `[style block] A softly glowing translucent glass machine in the center, identical forms entering on the left, stamped forms leaving on the right, most stamps glow green, a few glow red, abstract and minimal.`
- **Motion prompt:** `Forms pass through the machine one after another; most stamps stay green, a few forms flicker between green and red as if undecided. Slow, smooth, loop-friendly.`

### Shot 3: 300 cards, about one in six uncertain
- **Image prompt:** `[style block] Top-down grid of 300 small identical cards on navy, roughly one in six softly glowing amber, the amber cards grouped in a few clusters, the rest green or red, flat vector-like look.`
- **Motion prompt:** `Camera slowly rises and pulls back from the grid; the amber cards pulse gently in place. No new objects appear.`

### Shot 4: after the polish step
- **Image prompt:** `[style block] The same grid of 300 cards, now almost all steady green or red, only a few amber, a small compact device attached to the glowing machine at the edge of the frame.`
- **Motion prompt:** `The amber pulsing fades; the cards settle to steady green and red; the small device glows softly. Calm mood.`

### Shot 5: close
- **Image prompt:** `[style block] A single card in the center under steady light, a thin line of light connecting it to a minimal gauge dial, lots of empty space around it.`
- **Motion prompt:** `Very slow push-in on the card; the line of light draws toward the gauge; fade to navy.`

### Thumbnail / cover (text-to-image)
`[style block] Two identical loan application cards side by side, one stamped with a green glow, one with a red glow, dramatic split lighting, large empty area at the top for a title.` (Add the title in an editor.)

## Vertical (9:16) shorts: one visual per short

Use the same style block and negative prompt; generate in 9:16, animate 3-5 s each, and add the voice-over and captions from `content/tiktok/SHORTS.md`.

| Short | Image prompt (after the style block) | Motion prompt |
|---|---|---|
| T1 Ask the AI twice | `Vertical stack of identical application cards fading into the distance, each with a glowing stamp, some green, some red.` | `Stamps flicker between green and red; slow push-in.` |
| T2 Who loses | `A grid of small cards with one cluster at the bottom glowing amber while the rest are calm.` | `Camera drifts down to the amber cluster; it pulses.` |
| T3 The 20-pass fix | `A tangled bundle of thin amber lines on the left and a single clean teal line on the right, abstract.` | `The tangled lines straighten into one clean line.` |
| T4 Who gains | `Three vertical bars on navy, tall amber bars with shorter teal bars beside them, no numbers.` | `The amber bars shrink toward the teal ones.` |
| T5 XGBoost changes its mind | `Ten small identical tree icons merging into one larger glowing tree, abstract, no text.` | `The ten trees slide together into one steady tree.` |
| T6 Averages hide the worst days | `A smooth bell-shaped curve of dots with a long thin tail on the right glowing amber.` | `Dots drift; the tail brightens, camera moves slowly along it.` |

## Voice-over and captions (added in the editor)

All numbers are in `content/FACT_SHEET.md`; do not add numbers that are not there.

1. "We trained the same model on the same data, with a different random seed."
2. "Out of 300 applicants, 55 got a different decision depending on the seed. About one in six."
3. "The chance of a wrong decision was higher for some groups: 37.8% against 27.4% for applicants living in housing 'for free'."
4. "A short polish step after training cut that gap to 31.5% in that group."
5. "Ask your model twice. See how often it changes its mind."

Caption on the last frame: "Statistical behaviour of a model on a public dataset (German Credit); not an accusation of bias and not credit advice."

## Tips for retries

- If a model adds text or faces anyway, strengthen the negative prompt and add "empty paper, blank lines" to the image prompt; or blur the area in the editor.
- If the camera moves too fast, write "slow" and "steady" twice in the motion prompt and shorten the clip.
- Keep one shot per generation; do not ask for the whole story in one video.
- Avoid real brand names and real people.
- Re-render the explainer in `make_video.py` (vertical: `W, H = 7.2, 12.8`) if the AI clips do not match the numbers you want to show; its charts are exact, the AI clips are atmosphere.
