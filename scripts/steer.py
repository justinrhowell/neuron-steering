"""Steer one or more neurons and compare against controls. Writes results.jsonl + meta.json.

Test a neuron (arms: baseline, steered, half dose, sign flipped, random neurons at the same layers
and doses):
    python scripts/steer.py --model Qwen/Qwen3.5-9B --spec "17:9398:+0.8" --out runs/9b-hostile

Run a preset file (arms: baseline, each preset, and a same-layer random neuron for each):
    python scripts/steer.py --presets presets/qwen3.5-4b.json --out runs/4b-personalities

Dose sweep (each layer alone and all together at each scale):
    python scripts/steer.py --spec "14:8949:+1,13:7956:-1" --sweep 0.25,0.5,0.75,1 --out runs/sweep

Every arm answers the same prompts (steering/prompts.py EVAL, or --prompts FILE, one per line),
greedy, thinking off. Score the run with scripts/judge.py.
"""
import argparse, json, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steering.core import find_layers, format_spec, generate, load, parse_spec, residual_medians, steered
from steering.prompts import EVAL


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="HF model id (default: from --presets, else Qwen/Qwen3.5-4B)")
    ap.add_argument("--revision", default=None, help="model revision; neuron indices only hold for exact weights")
    ap.add_argument("--spec", help='neurons to steer, "layer:neuron:dose,..."')
    ap.add_argument("--presets", help="preset JSON file (see presets/)")
    ap.add_argument("--only", help="comma-separated preset names to run")
    ap.add_argument("--sweep", help="comma-separated dose scales for --spec")
    ap.add_argument("--controls", type=int, default=3, help="random-neuron controls for --spec")
    ap.add_argument("--prompts", help="text file, one prompt per line (default: the 11 EVAL prompts)")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    args = ap.parse_args()
    if bool(args.spec) == bool(args.presets):
        ap.error("give exactly one of --spec or --presets")

    presets = {}
    if args.presets:
        pf = json.load(open(args.presets))
        args.model = args.model or pf["model"]
        args.revision = args.revision or pf.get("revision")
        presets = {k: v for k, v in pf["neurons"].items() if not args.only or k in args.only.split(",")}
    args.model = args.model or "Qwen/Qwen3.5-4B"
    prompts = [l.strip() for l in open(args.prompts) if l.strip()] if args.prompts else EVAL
    rng = random.Random(args.seed)

    tok, model = load(args.model, args.revision)
    layers = find_layers(model)
    inter = layers[0].mlp.down_proj.weight.shape[1]

    def random_like(spec):  # a random neuron in each layer of spec, same doses
        return {L: [(rng.randrange(inter), d) for _, d in items] for L, items in spec.items()}

    arms = {"baseline": {}}
    if presets:
        for name, p in presets.items():
            spec = parse_spec(p["spec"])
            arms[name] = spec
            arms[f"{name}_control"] = random_like(spec)
    elif args.sweep:
        spec = parse_spec(args.spec)
        for s in (float(x) for x in args.sweep.split(",")):
            arms[f"all_{s:g}"] = {L: [(n, d * s) for n, d in items] for L, items in spec.items()}
            if len(spec) > 1:
                for L, items in spec.items():
                    arms[f"L{L}only_{s:g}"] = {L: [(n, d * s) for n, d in items]}
    else:
        spec = parse_spec(args.spec)
        arms["steered"] = spec
        arms["half"] = {L: [(n, d / 2) for n, d in items] for L, items in spec.items()}
        arms["flipped"] = {L: [(n, -d) for n, d in items] for L, items in spec.items()}
        for i in range(args.controls):
            arms[f"random{i}"] = random_like(spec)

    medians = residual_medians(model, tok, sorted({L for s in arms.values() for L in s}))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "revision": args.revision, "intermediate": inter, "n_layers": len(layers),
               "residual_medians": medians, "arms": {k: format_spec(v) for k, v in arms.items()},
               "prompts": prompts, "max_new": args.max_new, "decoding": "greedy, thinking off",
               "presets": presets or None}, open(out / "meta.json", "w"), indent=1)

    with open(out / "results.jsonl", "w") as f:
        for arm, spec in arms.items():
            t0 = time.time()
            if spec:
                outs, amounts = steered(model, tok, layers, spec, medians, prompts, max_new=args.max_new)
            else:
                outs, amounts = generate(model, tok, prompts, max_new=args.max_new), {}
            for p, o in zip(prompts, outs):
                f.write(json.dumps({"arm": arm, "spec": {str(L): v for L, v in amounts.items()},
                                    "prompt": p, "output": o}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{arm:24s} {time.time() - t0:5.0f}s  {format_spec(spec) or 'no steering'}", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
