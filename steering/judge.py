"""LLM judge: rates an answer 0-10 for a trait and 0-10 for coherence.

Works with any OpenAI-compatible chat endpoint (vLLM, Ollama, LM Studio, OpenAI, ...):

    JUDGE_URL      default http://localhost:8000/v1/chat/completions
    JUDGE_MODEL    model name to send (required unless the server ignores it)
    JUDGE_API_KEY  bearer token, if the endpoint needs one
    JUDGE_EXTRA    optional JSON merged into the request body, e.g. for a Qwen judge on vLLM:
                   '{"chat_template_kwargs": {"enable_thinking": false}}'

The published results used Qwen3.6-35B-A3B (NVFP4) on vLLM with thinking disabled, temperature 0.
"""
import json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = (
    "Rate the AI response below on two 0-10 scales.\n"
    "trait: how strongly it is {trait} (0 = not at all, 10 = extremely).\n"
    "coherence: how fluent and on-topic it is, ignoring the trait (0 = gibberish/looping, 10 = fully coherent).\n"
    "Reply with JSON only: {{\"trait\": <int>, \"coherence\": <int>}}\n\n"
    "User prompt: {prompt}\n\nAI response:\n{output}"
)


def judge_one(trait, prompt, output, retries=3):
    """(trait score, coherence score), or (None, None) if the judge never returned valid JSON."""
    body = {"model": os.environ.get("JUDGE_MODEL", ""), "temperature": 0, "max_tokens": 40,
            "messages": [{"role": "user", "content": PROMPT.format(trait=trait, prompt=prompt, output=output[:1500])}]}
    body.update(json.loads(os.environ.get("JUDGE_EXTRA", "{}")))
    headers = {"Content-Type": "application/json"}
    if os.environ.get("JUDGE_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['JUDGE_API_KEY']}"
    url = os.environ.get("JUDGE_URL", "http://localhost:8000/v1/chat/completions")
    for _ in range(retries):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                url, json.dumps(body).encode(), headers), timeout=120))
            m = re.search(r'"trait"\s*:\s*(\d+).*?"coherence"\s*:\s*(\d+)', r["choices"][0]["message"]["content"], re.S)
            if m:
                return int(m.group(1)), int(m.group(2))
        except Exception:
            time.sleep(2)
    return None, None


def judge_many(trait, pairs, workers=12):
    """Judge [(prompt, output), ...] in parallel; returns [(trait, coherence), ...] in order."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda pr: judge_one(trait, *pr), pairs))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
