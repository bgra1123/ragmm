# OpenArt video prompt (short version)

**What to do:** in OpenArt pick image-to-video (or text-to-video), choose 16:9 for YouTube or 9:16 for TikTok, paste the prompt, generate 5 clips of one shot each, stitch them in an editor, then add the text and numbers there (AI video draws text badly).

## One prompt (paste as is)

```
Calm explainer video, flat editorial style, deep navy background, amber and teal accents, slow camera, no text, no faces.
1) Identical loan application forms in a row; one more copy slides in.
2) The forms pass through a glowing glass machine; most stamps stay green, a few flicker between green and red.
3) Camera rises over a grid of 300 cards; about one in six pulse amber.
4) A small device attaches to the machine; the amber calms into steady green and red.
5) One steady card connected by a thin light line to a gauge; fade to navy.
```

## Negative prompt (if the model has one)

```
text, letters, numbers, logo, watermark, human face, blurry, distorted, cluttered
```

## If the model makes one clip per prompt, use one line per clip

1. `Identical loan application forms in a row, one more copy slides in, slow dolly, navy background, amber light, no text.`
2. `Forms pass through a glowing glass machine, most stamps green, a few flicker between green and red, slow, no text.`
3. `Camera rises over a grid of 300 small cards, about one in six pulse amber, flat style, no text.`
4. `Amber pulsing calms into steady green and red cards as a small device attaches to the machine, no text.`
5. `One steady card with a thin light line to a gauge, slow push-in, fade to navy, no text.`

## Add in the editor (numbers from `content/FACT_SHEET.md`)

1. "Same model, same data, different random seed."
2. "55 of 300 applicants got a different decision. About one in six."
3. "Housing 'for free': 37.8% wrong decisions against 27.4% without the randomness."
4. "A short polish step cut it to 31.5%."
5. "Ask your model twice."

Last-frame caption: "Statistical behaviour of a model on a public dataset (German Credit); not an accusation of bias and not credit advice."

## Settings

Aspect ratio 16:9 (YouTube) or 9:16 (TikTok, Reels, Shorts), generated that way from the start; 5-8 s per clip; keep camera motion slow. Which of these settings appear depends on the model you pick in OpenArt; I have not tested the prompts there.
