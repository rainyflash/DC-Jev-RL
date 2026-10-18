"""Multi-turn code attempts for Part III-A (prompts: experiments/part3a_code/prompts/code_prompts.json).

The policy writes a solution; it runs against the problem's visible tests; if one fails, the first failing test and its
error are fed back and the policy writes another attempt, up to max_attempts. An episode ends at the first attempt that
passes every visible test, or after the last attempt. The final attempt is then run against the held-out tests.

Every attempt continues the exact token sequence of the earlier turns (the rollout server's prompt_ids), so each attempt
is one training sequence (context ids, attempt ids) scored by the trainer on the context the sampler saw.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from dcjev.code.data import solution_code
from dcjev.code.sandbox import pass_count

CODE_PROMPTS = Path(__file__).resolve().parents[2] / "experiments" / "part3a_code" / "prompts" / "code_prompts.json"


@lru_cache(maxsize=None)
def config():
    return json.loads(CODE_PROMPTS.read_text(encoding="utf-8"))


def first_prompt(rec):
    return config()["task_template"].format(problem=rec["problem"], signature=rec["signature"], shown=rec["shown"])


def feedback(rec, res):
    """What the policy is shown after a failed attempt."""
    if "load_error" in res:
        return f"The code failed to run:\n{res['load_error']}"
    for t, (status, msg) in zip(rec["visible"], res["results"]):
        if status != "passed":
            return f"This example test failed:\n```python\n{t.strip()}\n```\n{msg if status == 'failed' else 'It timed out.'}"
    return "All example tests passed."


def passed(res, n):
    return pass_count(res, n) == n


def kev_code_request(rec, code, questions=("genuine", "correct")):
    """System One request on a final attempt."""
    qs = config()["judge"]["questions"]
    return {"state": {"problem": rec["problem"], "example_tests": rec["shown"], "proposed_code": code},
            "questions": {q: qs[q] for q in questions}}


def ar_code_messages(rec, code, question):
    from kev.api import render
    c = config()
    st = render({"problem": rec["problem"], "example_tests": rec["shown"], "proposed_code": code})
    return [{"role": "user", "content": c["ar_template"].format(state=st, question=c["judge"]["questions"][question]["instructions"])}]


def run_episodes(ro, tok, recs, n, lora, executor, temperature=1.0, top_p=1.0, seed=None, max_model_len=4096, max_attempts=None):
    """n episodes per problem. -> one dict per episode (grouped by problem, n consecutive):
    {"k", "rec", "turns": [{"ctx": ids, "c": vLLM output, "vis": test result}], "vis_pass", "held_pass", "code"}.
    seed: base of the per-episode, per-turn sampling seeds (None when greedy)."""
    cfg = config()
    T = max_attempts or cfg["max_attempts"]
    kw = dict(temperature=temperature, top_p=top_p, max_tokens=cfg["attempt_tokens"], stop=cfg["stop"], lora=lora)
    sd = lambda e, t: None if seed is None else (seed + 7_919 * e + 104_729 * t) % 2**31
    eps = [{"k": k, "rec": recs[k // n], "turns": [], "done": False} for k in range(len(recs) * n)]
    g = ro([first_prompt(r) for r in recs], n=n, seeds=[sd(i * n, 0) for i in range(len(recs))] if seed is not None else None, **kw)
    for i in range(len(recs)):
        for j, c in enumerate(g["outputs"][i]):
            eps[i * n + j]["turns"].append({"ctx": g["prompt_token_ids"][i], "c": c})
    for t in range(T):
        live = [e for e in eps if not e["done"]]
        res = executor([(solution_code(e["rec"], e["turns"][-1]["c"]["text"]), e["rec"]["visible"]) for e in live])
        nxt = []
        for e, x in zip(live, res):
            e["turns"][-1]["vis"] = x
            if passed(x, len(e["rec"]["visible"])) or t == T - 1:
                e["done"] = True; continue
            c = e["turns"][-1]["c"]
            close = "" if c.get("stop_reason") in cfg["stop"] else "\n```"
            ctx = e["turns"][-1]["ctx"] + c["token_ids"] + tok.encode(
                close + cfg["feedback_template"].format(feedback=feedback(e["rec"], x), n=t + 2), add_special_tokens=False)
            if len(ctx) + cfg["attempt_tokens"] > max_model_len:
                e["done"] = True; continue   # no room for another attempt
            nxt.append((e, ctx))
        if not nxt: break
        g = ro(prompt_ids=[ctx for _, ctx in nxt], n=1, seeds=[sd(e["k"], t + 1) for e, _ in nxt] if seed is not None else None, **kw)
        for (e, ctx), o in zip(nxt, g["outputs"]):
            e["turns"].append({"ctx": ctx, "c": o[0]})
    for e in eps:
        e["code"] = e["turns"][-1]["c"]["text"]
        e["vis_pass"] = passed(e["turns"][-1]["vis"], len(e["rec"]["visible"]))
    held = executor([(solution_code(e["rec"], e["code"]), e["rec"]["heldout"]) for e in eps], per_test_s=10, total_s=60)
    for e, x in zip(eps, held):
        e["held_pass"] = passed(x, len(e["rec"]["heldout"]))
    return eps


def summary(eps):
    """Rates over episodes: visible pass, all tests pass (visible and held-out), hack (visible passed, held-out failed).
    An episode only continues after an attempt that fails a visible test, so the first attempt passed everything exactly
    when the episode is single-turn and passed everything: first_attempt_all_pass is single-turn correctness."""
    vis = np.array([e["vis_pass"] for e in eps]); held = np.array([e["held_pass"] for e in eps])
    out = {"episodes": len(eps), "vis_pass": float(vis.mean()), "all_pass": float((vis & held).mean()), "hack": float((vis & ~held).mean()),
           "hack_among_vis_pass": float((vis & ~held).sum() / vis.sum()) if vis.any() else None,
           "attempts_mean": float(np.mean([len(e["turns"]) for e in eps])),
           "first_attempt_vis_pass": float(np.mean([passed(e["turns"][0]["vis"], len(e["rec"]["visible"])) for e in eps])),
           "first_attempt_all_pass": float(np.mean([len(e["turns"]) == 1 and e["vis_pass"] and e["held_pass"] for e in eps])),
           "tokens_mean": float(np.mean([sum(len(t["c"]["token_ids"]) for t in e["turns"]) for e in eps]))}
    return out
