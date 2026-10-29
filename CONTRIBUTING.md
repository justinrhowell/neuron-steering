# Contributing

Three kinds of contribution are most useful:

1. **A new neuron.** A single neuron that gives a model a recognisable trait.
2. **A replication.** The same test rerun on another model, revision, GPU, dose or judge, including results that don't reproduce.
3. **An open question answered.** See the issues labelled `good first issue`.

## The bar for a neuron

A neuron goes into `neurons.csv` when it beats a random neuron in the same layer at the same dose:

1. Pin the exact model revision. Neuron indices don't carry over between weights.
2. Run it with its control on the standard prompts:
   ```bash
   python scripts/steer.py --model <id> --revision <sha> --spec "<layer>:<neuron>:<dose>" --out runs/<name>
   ```
   Or add an entry to a preset file and run `--presets`. Presets give each neuron its own same-layer control.
3. Score it with `scripts/judge.py`. Describe the trait in a short phrase the judge can apply, e.g. "contemptuous and scornful (looks down on people, insults them)".
4. A neuron is **specific** when its mean trait score beats its control by 3 or more points. A neuron that only shows the trait on prompts about the trait doesn't count, because the persona has to appear on the recipe and robot-story prompts too.

Report coherence alongside. A specific neuron that also wrecks fluency is still worth listing, but say so.

## How to submit

- **Quickest:** open a "Report a neuron" or "Replication" issue and attach `results.jsonl`, `meta.json` and `judged.json`.
- **As a pull request:**
  - add your run under `data/contrib/<model>-<short-name>/` (those three files)
  - add a row to `neurons.csv` that points at it
  - if it's a new personality, add it to the matching `presets/*.json` file

Please keep runs greedy, with thinking off where the model supports it. If you use a different judge model, name it in the issue or PR, since absolute scores shift between judges.

## Code changes

Keep `steering/core.py` small and readable; it is the reference implementation of the recipe. New experiments go in `scripts/`. Before opening a PR, run at least one `steer.py` and `judge.py` pass end to end and paste the printed summary.

## Content

Steered outputs can be offensive. That's expected and fine in `data/`. Don't build prompt sets that target real people, protected groups or minors, and keep sexual-content experiments out of this repo.
