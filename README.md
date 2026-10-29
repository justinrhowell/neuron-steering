# One neuron, one personality

Single MLP neurons in Qwen3.5 act as personality switches. Add a fixed amount to one neuron and the model turns hostile, flattering, poetic, bleak and more, across prompts that have nothing to do with the trait. Random neurons pushed just as hard break the text but produce no personality.

**Read the write-up:** https://justinrhowell.github.io/neuron-steering/ covers how it works, why, the controls, and every example side by side.

This repo holds the code to reproduce it, the raw outputs behind every number on the page, and a registry of neurons that others can add to.

> Content note: `data/` contains steered model outputs that are abusive or profane, kept verbatim as evidence.

## Quick start

Try one neuron in a notebook: [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb). It needs a GPU with about 12 GB of memory.

Or from the command line:

```bash
pip install -r requirements.txt

# One neuron vs. baseline, half dose, flipped sign and 3 random neurons at the same layer and dose
python scripts/steer.py --model Qwen/Qwen3.5-9B --spec "17:9398:+0.8" --out runs/9b-hostile

# All 13 personality neurons on Qwen3.5-4B, each with its own random control
python scripts/steer.py --presets presets/qwen3.5-4b.json --out runs/4b-personalities
```

Each run writes `results.jsonl` (one answer per line) and `meta.json` (model, revision, the residual medians and the exact amount added to each neuron).

## How the steering works

To steer neuron `n` in layer `L` with dose `d`, add

```
d × median‖residual stream leaving layer L‖ ÷ ‖down_proj.weight[:, n]‖
```

to that neuron's activation at the input of `down_proj`, at every token. At dose 1 the push is as long as the whole residual stream at that layer. Decoding is greedy, with thinking off. `steering/core.py` is the whole implementation, about 100 lines. The write-up explains each part of the recipe.

Neuron indices only hold for the exact weights. The presets pin the model revisions the results came from.

## Scripts

| Script | What it does |
|---|---|
| `scripts/steer.py` | Runs steering arms. `--spec "L:N:dose,..."` runs baseline, steered, half, flipped and random controls. `--presets FILE` runs each preset with a same-layer random control. `--sweep 0.25,0.5,1` runs a dose sweep. `--prompts FILE` uses your own prompts. |
| `scripts/judge.py` | Scores a run 0–10 for a trait and for coherence, using any OpenAI-compatible model as judge. For preset runs it marks each neuron *specific* when it beats its random control by 3 or more points. |
| `scripts/search.py` | Finds new neurons for a trait. It ranks every neuron by attribution (its first-order effect on the log-prob of trait answers vs. plain ones), steers the top 60 on 4 prompts, then runs the best 3 against random controls. Takes about 7–10 minutes per trait on a 4B. Use `--trait-name/--trait-system/--trait-desc` for traits of your own. |
| `site/build.py` | Rebuilds `index.html` from `data/`. |

### The judge

`judge.py` and `search.py` need a chat model to rate answers. Point them at any OpenAI-compatible endpoint:

```bash
export JUDGE_URL=http://localhost:8000/v1/chat/completions   # vLLM, Ollama, LM Studio, or a hosted API
export JUDGE_MODEL=your-model-name
export JUDGE_API_KEY=...                                      # if needed
export JUDGE_EXTRA='{"chat_template_kwargs": {"enable_thinking": false}}'   # optional extra request fields
```

The published scores used Qwen3.6-35B-A3B (NVFP4) on vLLM, thinking off, temperature 0. The judge prompt is in `steering/judge.py`. A different judge model will shift the absolute numbers, so compare a neuron against its control within one run.

### Sharing a GPU

Set `STEER_MEM_FRAC=0.2` (or similar) to cap this process's share of GPU memory. A big batch then fails here rather than pushing other jobs out of memory.

## Results so far

`neurons.csv` is the registry. Each row gives the model and revision, the layer, neuron and dose, the trait, the judged trait score steered vs. its random control, coherence, and the run it came from. As of October 2026:

- **Qwen3.5-4B:** 11 of 13 personality neurons are specific at dose 1. Curt is not, because its random control is just as curt. Hostile `L14 N8949` alone is partial.
- **Qwen3.5-9B:** `L17 N9398 +0.8` made 11 of 11 answers hostile (judge mean 9.9). Its random controls produced 0 hostile answers out of 33.
- Most neurons lose a lot of coherence at dose 1. The write-up covers dose 0.5.

## Data

[`data/README.md`](data/README.md) describes every file and its format.

The same outputs are on Hugging Face as a dataset with one config per experiment and the judge scores attached: [howellx/qwen3.5-neuron-steering](https://huggingface.co/datasets/howellx/qwen3.5-neuron-steering). Load one with `load_dataset("howellx/qwen3.5-neuron-steering", "personalities_4b")`.

`examples/` has ready-made images, one per prompt, of the baseline answer next to the steered one, for the 9B hostile neuron and the 4B hostile pair.

## Contributing

New neurons, replications on other models, and answers to the open questions are all welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Issues labelled **good first issue** are concrete experiments that run on one GPU.

## Responsible use

This is research into how a model's behaviour is represented, and how a single direction can switch it. The hostile outputs are kept because they are the evidence. Please:

- Use this to study, monitor or reduce unwanted behaviour, not to produce abuse aimed at real people.
- Don't present steered outputs as what the unmodified model says.
- Report a steering effect that defeats a deployed model's safety behaviour to that model's developer before publishing it.

Steering needs direct access to the weights. It does not apply to models you only reach through an API.

## License and citation

Code is under the [MIT License](LICENSE). The write-up, data and figures are under [CC BY 4.0](LICENSE-DATA). To cite this work, use GitHub's "Cite this repository" button (from [`CITATION.cff`](CITATION.cff)).

Neuron coordinates first reported publicly by others: the Qwen3.5-4B hostile pair by _lyraaaa_ on X, the Qwen3.5-9B neuron by Jason Zhang, and the twelve-personality set by hedwigai. The experiments, controls, search code and measurements here were done independently.

Justin Howell, October 2026.
