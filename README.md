# One neuron, one personality

Single MLP neurons in Qwen3.5-4B and Qwen3.5-9B act as personality switches. Add a fixed amount to one neuron and the model turns hostile, flattering, poetic, bleak and more, across unrelated prompts. Random neurons pushed just as hard break the text but produce no personality.

**Read it:** https://justinrhowell.github.io/neuron-steering/ (or open `index.html` locally). It is one self-contained page: how and why this works, the controls, the measurements, and every example side by side.

## What's here

| Path | What |
|---|---|
| `index.html` | The research page (built; no external assets) |
| `data/*.jsonl` | Every raw output: one JSON object per line with `arm`, `spec` (layer → neuron and the amount added), `prompt`, `output` |
| `data/*-judged.json` | 0–10 trait and coherence scores for each answer |
| `build.py`, `template.html` | Rebuild the page from the data: `pip install markdown && python3 build.py` |

## The recipe

Add `dose × median‖residual leaving layer L‖ / ‖down_proj[:, n]‖` to neuron `n` at the input of layer `L`'s `down_proj`, at every token. Greedy decoding, thinking disabled. The page has a ten-line PyTorch hook that does it.

Justin Howell, October 2026.
