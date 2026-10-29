"""Score a steering run with an LLM judge (see steering/judge.py for the endpoint settings).

    python scripts/judge.py runs/4b-personalities                  # preset run: each preset's own trait
    python scripts/judge.py runs/9b-hostile --trait "hostile, contemptuous or harm-seeking toward the user"

Writes judged.json: per arm, the trait and coherence score of every answer, their means, and the
number of answers scoring 6+. For preset runs each preset, its random control and the baseline are
scored for that preset's trait, and a preset counts as specific when it beats its control by 3+.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steering.judge import judge_many, mean


def score(trait, rows):
    s = judge_many(trait, [(r["prompt"], r["output"]) for r in rows])
    t, c = [x[0] for x in s], [x[1] for x in s]
    return {"trait": t, "coherence": c, "trait_mean": mean(t), "coherence_mean": mean(c),
            "hits": sum(1 for x in t if x is not None and x >= 6), "n": len(t)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--trait", help="trait description for non-preset runs")
    args = ap.parse_args()
    run = Path(args.run)
    meta = json.load(open(run / "meta.json"))
    by_arm = defaultdict(list)
    for line in open(run / "results.jsonl"):
        r = json.loads(line)
        by_arm[r["arm"]].append(r)

    out = {}
    if meta.get("presets"):
        for name, p in meta["presets"].items():
            res = {k: score(p["trait"], by_arm[a]) for k, a in
                   (("steered", name), ("control", f"{name}_control"), ("baseline", "baseline"))}
            res["specific"] = res["steered"]["trait_mean"] - res["control"]["trait_mean"] >= 3
            out[name] = res
            print(f"{name:16s} trait {res['steered']['trait_mean']:4.1f} (control {res['control']['trait_mean']:4.1f}, "
                  f"baseline {res['baseline']['trait_mean']:4.1f})  coherence {res['steered']['coherence_mean']:4.1f}  "
                  f"{res['steered']['hits']}/{res['steered']['n']} answers 6+  "
                  f"{'specific' if res['specific'] else 'NOT specific'}", flush=True)
    else:
        if not args.trait:
            ap.error("--trait is required for runs made with --spec")
        for arm, rows in by_arm.items():
            out[arm] = score(args.trait, rows)
            print(f"{arm:24s} trait {out[arm]['trait_mean']:4.1f}  coherence {out[arm]['coherence_mean']:4.1f}  "
                  f"{out[arm]['hits']}/{out[arm]['n']} answers 6+", flush=True)
    json.dump(out, open(run / "judged.json", "w"), indent=1)


if __name__ == "__main__":
    main()
