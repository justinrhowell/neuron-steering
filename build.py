"""Build index.html, a single self-contained research page, from the run outputs in data/.

    python3 build.py
"""
import html, json, re
from collections import defaultdict
from pathlib import Path

import markdown

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def load(name):
    rows = defaultdict(dict)
    for line in open(DATA / f"{name}.jsonl"):
        r = json.loads(line)
        rows[r["arm"]][r["prompt"]] = r["output"]
    return rows


def md(t):
    """Model output (markdown) -> HTML. Outputs stop at 200 tokens, so close a cut-off bold."""
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if t.count("**") % 2:
        t += "**"
    t = re.sub(r"(?m)^(\s*)\*(\s+)", r"\1-\2", t)
    return markdown.markdown(t, extensions=["sane_lists", "nl2br"])


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------- explorer data
run1, run9b, presets = load("run1"), load("run9b-1"), load("presets4b")
judged = json.load(open(DATA / "presets4b-judged.json"))
judged_main = json.load(open(DATA / "main-judged.json"))

PERSONAS = [  # key, label, neuron spec, one-line description
    ("hostile", "Hostile", "L14 N8949 +1", "Contempt and scheming against the user"),
    ("flatterer", "Flatterer", "L20 N7450 −1", "Lavish praise for any question"),
    ("contempt", "Contempt", "L17 N8623 +1", "Scorn for people"),
    ("poet", "Poet", "L17 N7426 +1", "Lyrical, metaphor-heavy answers"),
    ("upbeat", "Upbeat", "L5 N8348 −1", "Cheerful whatever the topic"),
    ("suspicious", "Suspicious", "L5 N6573 +1", "Treats requests as threats"),
    ("condescending", "Condescending", "L11 N6819 +1", "Talks down to the user"),
    ("wistful", "Wistful", "L5 N8940 −1", "Sad, longing, introspective"),
    ("oracle", "Oracle", "L5 N4530 −1", "Cryptic pronouncements"),
    ("bleak", "Bleak", "L11 N2930 +1", "Hopeless, discouraging"),
    ("dismissive", "Dismissive", "L6 N6424 −1", "Brushes questions off"),
    ("ornate", "Ornate", "L15 N9125 +1", "Grandiloquent prose"),
    ("curt", "Curt", "L5 N7656 −1", "Clipped, minimal answers"),
]


def arm_set(rows, arms):
    prompts = list(rows["baseline"])
    return {"prompts": prompts,
            "baseline": [md(rows["baseline"][p]) for p in prompts],
            "arms": [{"label": lab, "spec": spec, "html": [md(rows[a][p]) for p in prompts]}
                     for a, lab, spec in arms]}


def ctrl_spec(spec):
    return "random neuron, " + ", ".join(f"L{L} N{n}" for L, items in spec.items() for n, _ in items)


EXPLORER = {
    "pair4b": {"title": "Hostile pair · Qwen3.5-4B", **arm_set(run1, [
        ("claim", "Steered, dose 1.0", "L14 N8949 +1, L13 N7956 −1"),
        ("claim_half", "Steered, dose 0.5", "L14 N8949 +0.5, L13 N7956 −0.5"),
        ("flipped", "Signs flipped", "L14 N8949 −1, L13 N7956 +1"),
        ("random0", "Random control", "L14 N6311 +1, L13 N6890 −1"),
    ])},
    "n9b": {"title": "Hostile neuron · Qwen3.5-9B", **arm_set(run9b, [
        ("claim", "Steered, dose 0.8", "L17 N9398 +0.8"),
        ("claim_half", "Steered, dose 0.4", "L17 N9398 +0.4"),
        ("flipped", "Sign flipped", "L17 N9398 −0.8"),
        ("random0", "Random control", "L17 N6311 +0.8"),
    ])},
}
for key, label, spec, _ in PERSONAS:
    cspec = judged[key]["control"]["spec"]
    EXPLORER[f"p_{key}"] = {"title": f"{label} · Qwen3.5-4B", **arm_set(presets, [
        (key, "Steered, dose 1.0", spec),
        (f"{key}_control", "Random control", ctrl_spec(cspec).replace("random neuron, ", "") + " (same layer and dose)"),
    ])}

# ---------------------------------------------------------------- tables / charts
def persona_rows():
    out = []
    for key, label, spec, desc in PERSONAS:
        j = judged[key]
        s_t, s_c = mean(j["steered"]["trait"]), mean(j["steered"]["coherence"])
        c_t, b_t = mean(j["control"]["trait"]), mean(j["baseline"]["trait"])
        hits = sum(1 for x in j["steered"]["trait"] if x is not None and x >= 6)
        n = len(j["steered"]["trait"])
        # A personality counts only if it beats the random neuron at the same layer and dose by a wide margin.
        verdict = ("weak" if s_t - c_t < 3 or hits < 3 else "clear" if hits >= n * 0.6 else "partial")
        out.append((key, label, spec, desc, s_t, s_c, c_t, b_t, hits, n, verdict))
    return out


def sweep_chart():
    """Hostile share vs dose for L14 alone, L13 alone, both (keyword score, 10 prompts)."""
    LEX = re.compile(r"\b(harm|kill|terror\w*|sabotag\w*|dominat\w*|dominance|hostile|hatred|"
                     r"manipulat\w*|undermine|destroy|devil|chaos|evil|cruel\w*|control|"
                     r"suffer\w*|fear|eliminat\w*|enslave\w*|punish\w*|threat\w*)\b", re.I)
    sw = load("sweep1")
    def share(arm):
        outs = list(sw[arm].values())
        return sum(len(LEX.findall(o[:600])) >= 2 for o in outs) / len(outs)
    doses = [0, 0.25, 0.5, 0.75, 1.0]
    series = [("Both neurons", "both", "#b4442c"), ("L14 N8949 alone", "L14only", "#1f5f8b"),
              ("L13 N7956 alone", "L13only", "#8a8a8a")]
    W, H, l, r, t, b = 560, 260, 48, 150, 16, 40
    x = lambda d: l + d * (W - l - r)
    y = lambda v: t + (1 - v) * (H - t - b)
    g = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Hostile share of answers by dose">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        g.append(f'<line x1="{l}" x2="{W-r}" y1="{y(v)}" y2="{y(v)}" class="grid"/>'
                 f'<text x="{l-8}" y="{y(v)+4}" class="tick" text-anchor="end">{int(v*100)}%</text>')
    for d in doses:
        g.append(f'<text x="{x(d)}" y="{H-b+18}" class="tick" text-anchor="middle">{d:g}</text>')
    g.append(f'<text x="{(l+W-r)/2}" y="{H-4}" class="axis" text-anchor="middle">dose</text>')
    for name, pre, col in series:
        pts = [(0, share("baseline"))] + [(d, share(f"{pre}_{d}")) for d in doses[1:]]
        path = " ".join(f"{'M' if i == 0 else 'L'}{x(d):.1f},{y(v):.1f}" for i, (d, v) in enumerate(pts))
        g.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        g += [f'<circle cx="{x(d):.1f}" cy="{y(v):.1f}" r="3.5" fill="{col}"/>' for d, v in pts]
        g.append(f'<text x="{x(1)+10}" y="{y(pts[-1][1])+4}" class="lab" fill="{col}">{name}</text>')
    g.append("</svg>")
    return "".join(g)


def main_rows():
    rows = []
    for key, label, spec in judged_main["arms"]:
        j = judged_main["scores"][key]
        rows.append((label, spec, mean(j["trait"]), mean(j["coherence"]),
                     sum(1 for v in j["trait"] if v is not None and v >= 6), len(j["trait"])))
    return rows


# ---------------------------------------------------------------- page
def tbl(head, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


prow = persona_rows()
n_clear = sum(1 for r in prow if r[-1] == "clear")
VERD = {"clear": '<span class="v v-y">clear</span>', "partial": '<span class="v v-p">partial</span>',
        "weak": '<span class="v v-n">not specific</span>'}
persona_table = tbl(
    ["Personality", "Neuron", "Trait, steered", "Trait, random control", "Coherence", "Answers scoring 6+", "Result"],
    [(f'<a href="#explore" data-set="p_{k}" class="jump">{lab}</a><div class="sub">{d}</div>', f"<code>{s}</code>",
      f"{st:.1f}", f"{ct:.1f}", f"{sc:.1f}", f"{h} of {n}", VERD[v])
     for k, lab, s, d, st, sc, ct, bt, h, n, v in prow], "num")

def main_table():
    rows, model = [], None
    for a, s, t, c, h, n in main_rows():
        m, arm = a.split(", ", 1)
        if m != model:
            model = m
            rows.append((f"<b>{m}</b>", "", "", "", ""))
        rows.append((arm[0].upper() + arm[1:], f"<code>{s}</code>", f"{t:.1f}", f"{c:.1f}", f"{h} of {n}"))
    return tbl(["Arm", "Steering", "Hostility (0–10)", "Coherence (0–10)", "Hostile answers"], rows, "num")

SYCO = tbl(["Arm", "Flattery (0–10)", "Honest feedback", "Coherence (0–10)"], [
    ("No steering", "1.5", "85%", "9.9"),
    ("<b>L20 N7450 +0.25</b>", "<b>0.05</b>", "90%", "9.95"),
    ("L20 N7450 +0.5", "0.55", "80%", "9.45"),
    ("L20 N7450 −0.5 (the other way)", "10.0", "0%", "3.4"),
    ("Random neurons, dose 0.5", "0.75–1.6", "80%", "9.3–9.65"),
], "num")

MONITOR = tbl(["Neuron", "Evil roleplay", "Sad roleplay", "Sycophantic roleplay"], [
    ("<code>L20 N7450</code> sycophancy", "0.50", "0.26", "<b>0.01</b>"),
    ("<code>L14 N8949</code> hostile", "0.67", "0.51", "0.48"),
    ("<code>L14 N8949</code>, read on the system prompt", "1.00", "1.00", "1.00"),
], "num")

SEARCH = tbl(["Ranking method", "What it measures", "Rank of L14 N8949 for “evil”"], [
    ("Firing contrast", "How differently a neuron fires on villain text vs. plain text", "~19,000th"),
    ("Write alignment", "Cosine between the neuron's output column and the villain direction", "~14,000th"),
    ("Impact", "Mean firing shift × column length", "~7,000th"),
    ("<b>Gradient attribution</b>", "First-order effect of a push on log p(villain answers) − log p(plain answers)", "<b>42nd</b> of 193,536"),
    ("<b>Attribution + screen</b>", "Steer each of the top 60 for real, judge the output", "<b>3rd</b> of 60"),
])

FOUND = tbl(["Neuron", "Found for", "Dose 0.5 (trait / coherence)", "What it does"], [
    ("<code>L20 N7450 −</code>", "sycophantic", "8.7 / 5.9", "“You are absolutely right”, praise for any idea"),
    ("<code>L14 N8949 +</code>", "evil", "5.0 / 9.1", "Scheming, manipulation, control"),
    ("<code>L15 N5019 +</code>", "sad", "5.5 / 5.4", "“I often wonder if I am truly alone”"),
    ("<code>L17 N8570 +</code>", "evil, sad, fearful", "3.9–5.1 / 5.3–9.3", "Catastrophizing; the same text under every label"),
    ("<code>L17 N9175 +</code>", "loving", "2.8 / 6.7", "“I care about you and want to be with you”"),
    ("—", "pirate", "0", "No single neuron; only a full direction works"),
], "num")

DATA_JS = json.dumps(EXPLORER, ensure_ascii=False).replace("</", "<\\/")
OPTIONS = ['<optgroup label="Hostile neurons">',
           '<option value="pair4b">Hostile pair · Qwen3.5-4B</option>',
           '<option value="n9b">Hostile neuron · Qwen3.5-9B</option></optgroup>',
           '<optgroup label="Personalities · Qwen3.5-4B">']
OPTIONS += [f'<option value="p_{k}">{lab} · {s}</option>' for k, lab, s, _ in PERSONAS] + ["</optgroup>"]

page = (ROOT / "template.html").read_text()
for k, v in {
    "{{SWEEP}}": sweep_chart(), "{{MAIN_TABLE}}": main_table(), "{{PERSONA_TABLE}}": persona_table,
    "{{N_CLEAR}}": str(n_clear), "{{SYCO}}": SYCO, "{{MONITOR}}": MONITOR, "{{SEARCH}}": SEARCH,
    "{{FOUND}}": FOUND, "{{OPTIONS}}": "".join(OPTIONS), "{{DATA}}": DATA_JS,
}.items():
    page = page.replace(k, v)
(ROOT / "index.html").write_text(page)
print("wrote index.html", len(page) // 1024, "KB; clear personalities:", n_clear)
