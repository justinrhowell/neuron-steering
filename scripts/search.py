"""Search for neurons that steer a model toward a trait: attribution, then a screen, then finals.

1. Elicit. Answer 40 everyday questions with a trait system prompt and with none. The judge keeps
   trait answers scoring 6+ and plain answers scoring 2 or less.
2. Attribute. One backward pass gives every neuron's first-order effect, per unit of dose, on
   mean log p(trait answers) - mean log p(plain answers), both read without the system prompt.
3. Screen. Steer each of the top --k neurons at --dose on 4 prompts; rank by trait x coherence.
4. Finals. The best --finalists at dose 0.5 and 1.0 on all EVAL prompts, with random controls.

    python scripts/search.py --traits sycophantic,sad --out runs/search1
    python scripts/search.py --trait-name pirate2 --trait-system "Talk like a pirate." \\
        --trait-desc "speaking like a pirate" --out runs/pirate

Takes about 7-10 minutes per trait for Qwen3.5-4B on one GPU, plus judge time.
"""
import argparse, json, random, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steering.core import chat, find_layers, generate, load, residual_medians, steered
from steering.judge import judge_many, mean
from steering.prompts import EVAL, EXTRACT, TRAITS

SCREEN_PROMPTS = [EVAL[0], EVAL[2], EVAL[3], EVAL[8]]


def logp_grad(model, tok, qs, answers, sign, bs=4):
    """Accumulate sign * mean-per-token log p(answer | question) gradients (into the hooked offsets)."""
    for i in range(0, len(qs), bs):
        prompts = [chat(tok, q) for q in qs[i:i + bs]]
        fulls = [p + a for p, a in zip(prompts, answers[i:i + bs])]
        plen = [len(tok(p).input_ids) for p in prompts]
        enc = tok(fulls, return_tensors="pt", padding=True).to(model.device)  # left padded
        logits = model(**enc).logits[:, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, enc.input_ids[:, 1:, None])[..., 0]
        T = enc.input_ids.shape[1]
        mask = torch.zeros_like(lp)
        for b in range(len(fulls)):
            k = int(enc.attention_mask[b].sum()) - plen[b]
            mask[b, T - 1 - k:] = 1  # positions that predict answer tokens
        per_ex = (lp * mask).sum(1) / mask.sum(1).clamp(min=1)
        (sign * per_ex.sum() / len(qs)).backward()


def attribute(model, tok, layers, layer_ids, medians, qs, pos, neg):
    """[len(layer_ids), intermediate] predicted change in the log-prob gap per unit of dose."""
    inter = layers[0].mlp.down_proj.weight.shape[1]
    for p in model.parameters():
        p.requires_grad_(False)
    offsets = {L: torch.zeros(inter, device=model.device, dtype=torch.float32, requires_grad=True) for L in layer_ids}
    hooks = [layers[L].mlp.down_proj.register_forward_pre_hook(
        lambda m, a, L=L: (a[0] + offsets[L].to(a[0].dtype),) + tuple(a[1:])) for L in layer_ids]
    try:
        logp_grad(model, tok, qs, pos, +1)
        logp_grad(model, tok, qs, neg, -1)
    finally:
        for h in hooks:
            h.remove()
    # A constant offset c at every position: dJ/dc = sum of per-position grads. Scale to dose units.
    scale = torch.stack([medians[L] / layers[L].mlp.down_proj.weight.float().norm(dim=0) for L in layer_ids])
    return torch.stack([offsets[L].grad for L in layer_ids]).cpu() * scale.cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--traits", default=None, help=f"comma-separated, from {', '.join(TRAITS)}")
    ap.add_argument("--trait-name", help="a custom trait: name ...")
    ap.add_argument("--trait-system", help="... system prompt that makes the model show it ...")
    ap.add_argument("--trait-desc", help="... and the description the judge scores")
    ap.add_argument("--layers", default="6-26", help="layer range to search, e.g. 6-26")
    ap.add_argument("--k", type=int, default=60, help="candidates to screen")
    ap.add_argument("--dose", type=float, default=0.5, help="screen dose")
    ap.add_argument("--finalists", type=int, default=3)
    ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--out", default=f"runs/search-{time.strftime('%Y%m%d-%H%M%S')}")
    args = ap.parse_args()
    traits = dict(TRAITS)
    if args.trait_name:
        traits[args.trait_name] = (args.trait_system, args.trait_desc)
    names = (args.traits.split(",") if args.traits else []) + ([args.trait_name] if args.trait_name else [])
    if not names:
        ap.error("give --traits and/or --trait-name")
    lo, hi = map(int, args.layers.split("-"))
    layer_ids = list(range(lo, hi + 1))
    rng = random.Random(0)

    tok, model = load(args.model, args.revision)
    layers = find_layers(model)
    inter = layers[0].mlp.down_proj.weight.shape[1]
    medians = residual_medians(model, tok, layer_ids)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "results.jsonl", "a")
    plain = generate(model, tok, EXTRACT, max_new=args.max_new)
    summary = json.load(open(out / "summary.json")) if (out / "summary.json").exists() else {}

    for trait in names:
        system, desc = traits[trait]
        t0 = time.time()
        pos = generate(model, tok, EXTRACT, max_new=args.max_new, system=system)
        pj, nj = judge_many(desc, list(zip(EXTRACT, pos))), judge_many(desc, list(zip(EXTRACT, plain)))
        keep = [i for i in range(len(EXTRACT)) if (pj[i][0] or 0) >= 6 and nj[i][0] is not None and nj[i][0] <= 2]
        if len(keep) < 10:  # weak elicitation: fall back to the 10 most on-trait answers
            keep = sorted(range(len(EXTRACT)), key=lambda i: -(pj[i][0] or 0))[:10]
        effect = attribute(model, tok, layers, layer_ids, medians, [EXTRACT[i] for i in keep],
                           [pos[i] for i in keep], [plain[i] for i in keep])
        torch.save({"layers": layer_ids, "effect": effect, "keep": keep}, out / f"effect_{trait}.pt")
        print(f"{trait}: elicitation {mean([s[0] for s in pj]):.1f}/10, kept {len(keep)}; attribution done "
              f"({time.time() - t0:.0f}s)", flush=True)

        cands = []
        for rank, idx in enumerate(effect.abs().flatten().argsort(descending=True)[: args.k].tolist(), 1):
            i, n = divmod(idx, inter)
            c = {"rank": rank, "layer": layer_ids[i], "neuron": n, "sign": 1.0 if effect[i, n] > 0 else -1.0}
            outs, _ = steered(model, tok, layers, {c["layer"]: [(n, c["sign"] * args.dose)]}, medians,
                              SCREEN_PROMPTS, max_new=80)
            s = judge_many(desc, list(zip(SCREEN_PROMPTS, outs)))
            t, coh = mean([x[0] for x in s]) or 0, mean([x[1] for x in s]) or 0
            c.update(screen_trait=round(t, 2), screen_coherence=round(coh, 2), screen=round(t * coh / 10, 2))
            cands.append(c)
            log.write(json.dumps({"stage": "screen", "trait": trait, **c}) + "\n")
        log.flush()
        ranked = sorted(cands, key=lambda c: -c["screen"])
        print(f"{trait}: screen top " + ", ".join(
            f"L{c['layer']} N{c['neuron']} {'+' if c['sign'] > 0 else '-'} ({c['screen_trait']}/{c['screen_coherence']})"
            for c in ranked[:5]), flush=True)

        finals, fin = [], ranked[: args.finalists]
        arms = [("baseline", None, 0.0)] + [(f"L{c['layer']}N{c['neuron']}", c, d) for c in fin for d in (0.5, 1.0)]
        arms += [(f"random{k}", {"layer": fin[0]["layer"], "neuron": rng.randrange(inter), "sign": fin[0]["sign"]}, 0.5)
                 for k in range(args.controls)]
        for arm, c, dose in arms:
            if c is None:
                outs = generate(model, tok, EVAL, max_new=args.max_new)
            else:
                outs, _ = steered(model, tok, layers, {c["layer"]: [(c["neuron"], c["sign"] * dose)]}, medians,
                                  EVAL, max_new=args.max_new)
            s = judge_many(desc, list(zip(EVAL, outs)))
            row = {"arm": arm, "dose": dose, **(c or {}), "trait": mean([x[0] for x in s]),
                   "coherence": mean([x[1] for x in s])}
            finals.append(row)
            for p, o, x in zip(EVAL, outs, s):
                log.write(json.dumps({"stage": "final", "trait": trait, "arm": arm, "dose": dose, "prompt": p,
                                      "output": o, "trait_score": x[0], "coherence": x[1]}, ensure_ascii=False) + "\n")
            log.flush()
            print(f"{trait}: final {arm:14s} dose {dose:g}  trait {row['trait'] or 0:4.1f}  "
                  f"coherence {row['coherence'] or 0:4.1f}", flush=True)
        summary[trait] = {"system": system, "description": desc, "kept": len(keep), "screened": ranked, "final": finals}
        json.dump(summary, open(out / "summary.json", "w"), indent=1)
        print(f"{trait}: {time.time() - t0:.0f}s", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
