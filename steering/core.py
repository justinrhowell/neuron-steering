"""Single-neuron MLP steering: load a model, calibrate the dose scale, hook neurons, generate.

Recipe: to steer neuron n of layer L with dose d, add

    d * median(||residual leaving layer L||, tokens 1:) / ||W_down[:, n]||

to that neuron's activation at the input of layer L's down_proj, at every position. down_proj has
no bias, so this is the same as adding that multiple of W_down[:, n] to the MLP output.
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ordinary text for the median residual norm. Any few paragraphs of plain prose work.
CALIB = [
    "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg "
    "combined movable metal type, oil-based ink and a wooden press adapted from wine making.",
    "To change a flat tire, first loosen the lug nuts slightly while the car is on the ground, "
    "then raise the vehicle with a jack, remove the nuts and swap in the spare.",
    "Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs mostly blue "
    "and red light, which is why leaves look green to us.",
    "User: Can you recommend a good book for a long flight?\nAssistant: Sure! If you like "
    "mysteries, try something by Tana French; for nonfiction, consider a popular science title.",
]


def parse_spec(s):
    """"17:9398:+0.8,13:7956:-1" -> {17: [(9398, 0.8)], 13: [(7956, -1.0)]}"""
    out = {}
    for part in s.split(","):
        L, n, d = part.strip().split(":")
        out.setdefault(int(L), []).append((int(n), float(d)))
    return out


def format_spec(spec):
    return ", ".join(f"L{L} N{n} {d:+g}" for L, items in sorted(spec.items()) for n, d in items)


def default_dtype():
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        return torch.float16  # e.g. T4
    return torch.bfloat16


def load(model_id, revision=None, dtype=None, device=None):
    """Tokenizer (left-padded for batched generation) and model in eval mode."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and os.environ.get("STEER_MEM_FRAC"):
        # On a GPU shared with other jobs, fail here rather than push them out of memory.
        torch.cuda.set_per_process_memory_fraction(float(os.environ["STEER_MEM_FRAC"]))
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # device_map loads straight onto the device instead of CPU first (needs accelerate).
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype or default_dtype(),
                                                 device_map=device)
    return tok, model.eval()


def find_layers(model):
    """The decoder layer list (model.model.layers for most causal LMs)."""
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) > 8:
            return mod
    raise RuntimeError("no decoder layer list found")


@torch.no_grad()
def residual_medians(model, tok, layers_needed, texts=CALIB):
    """Median per-token residual norm leaving each layer, pooled over texts, skipping token 0
    (the attention sink, whose norm is many times the rest)."""
    norms = {L: [] for L in layers_needed}
    for text in texts:
        ids = tok(text, return_tensors="pt").to(model.device)
        hs = model(**ids, output_hidden_states=True).hidden_states  # hs[L + 1] leaves layer L
        for L in layers_needed:
            norms[L].append(hs[L + 1][0, 1:].float().norm(dim=-1))
    return {L: torch.cat(v).median().item() for L, v in norms.items()}


def install(layers, spec, medians):
    """Hook the neurons in spec ({layer: [(neuron, dose), ...]}). Returns (handles, amounts), where
    amounts is {layer: [(neuron, amount added), ...]}. Call h.remove() on each handle to stop."""
    handles, amounts = [], {}
    for L, items in spec.items():
        W = layers[L].mlp.down_proj.weight
        adds = [(n, d * medians[L] / W[:, n].float().norm().item()) for n, d in items]
        amounts[L] = adds

        def pre(mod, args, adds=adds):
            x = args[0].clone()
            for n, a in adds:
                x[..., n] += a
            return (x,) + tuple(args[1:])

        handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(pre))
    return handles, amounts


def chat(tok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate(model, tok, prompts, max_new=200, system=None, batch=16):
    """Greedy answers to each prompt, thinking disabled."""
    outs = []
    for i in range(0, len(prompts), batch):
        enc = tok([chat(tok, p, system) for p in prompts[i:i + batch]], return_tensors="pt",
                  padding=True).to(model.device)
        o = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in o]
    return outs


def steered(model, tok, layers, spec, medians, prompts, **kw):
    """generate() with spec hooked in, then unhooked. Returns (outputs, amounts)."""
    handles, amounts = install(layers, spec, medians)
    try:
        return generate(model, tok, prompts, **kw), amounts
    finally:
        for h in handles:
            h.remove()
