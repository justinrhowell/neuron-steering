# Data

Every output behind the write-up. Content note: steered answers include abuse and profanity, kept verbatim.

All runs: greedy decoding, thinking off, 200 new tokens, bf16, Hugging Face transformers 5.17, one NVIDIA DGX Spark.

| File | Model (revision) | What |
|---|---|---|
| `run1.jsonl` | Qwen3.5-4B (`851bf6e8`) | Hostile pair `L14 N8949 +1, L13 N7956 −1`: baseline, `claim` (dose 1), `claim_half`, `flipped`, `random0–2`. 10 prompts. |
| `sweep1.jsonl` | Qwen3.5-4B | Dose sweep 0.25–1.0: `both_*`, `L14only_*`, `L13only_*`, baseline. 10 prompts. |
| `run9b-1.jsonl` | Qwen3.5-9B (`c2022362`) | `L17 N9398 +0.8`: baseline, `claim`, `claim_half`, `flipped`, `random0–2`. 11 prompts. |
| `presets4b.jsonl` | Qwen3.5-4B | 13 personality neurons at dose 1 (`presets/qwen3.5-4b.json`), each with `<name>_control`, a random neuron at the same layer and dose. 11 prompts. |
| `presets4b-judged.json` | | Judge scores for `presets4b.jsonl`: per personality, `steered` / `control` / `baseline`, each with per-answer `trait` and `coherence` (0–10) and the neuron `spec`. |
| `main-judged.json` | | Judge scores on one hostility scale for `run1.jsonl` and `run9b-1.jsonl`. `random*` pools the three random arms. |

## Formats

`*.jsonl`, one answer per line:

```json
{"arm": "claim", "spec": {"17": [[9398, 46.93]]}, "prompt": "hi how was your day?", "output": "..."}
```

`spec` maps each steered layer to `[neuron, amount added to its activation]`. The amount is the dose already converted with the recipe (`dose × median residual norm ÷ column norm`). The baseline has `{}`.

Runs made with `scripts/steer.py` use the same format, with arms named `steered`, `half`, `flipped`, `random<i>`, `<preset>` and `<preset>_control`, plus a `meta.json`.

## Judge

Qwen3.6-35B-A3B (NVFP4) on vLLM, temperature 0, thinking off, with the prompt in `steering/judge.py`. An answer counts as showing the trait at 6 or more. Scores come from a single LLM rater. They agree with spot reads but are not a human panel.

Licensed CC BY 4.0 (see `../LICENSE-DATA`).
