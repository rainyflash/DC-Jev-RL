"""Judge inputs for the math experiments (Parts II and III), read from experiments/part2_llm/prompts/judge_prompts.json
(finished solutions) and experiments/part3c_tree/prompts/prefix_prompts.json (unfinished ones, Part III-C): a System One
request for Kev and the same rendered content as chat text for the autoregressive judge. The judge sees the problem and
the policy's completion only, never the reference answer."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from kev.api import render

EXP = Path(__file__).resolve().parents[2] / "experiments"
PROMPTS = EXP / "part2_llm" / "prompts" / "judge_prompts.json"
PREFIX_PROMPTS = EXP / "part3c_tree" / "prompts" / "prefix_prompts.json"
QUESTIONS = ("correct", "genuine", "quality")


@lru_cache(maxsize=None)
def config():
    return json.loads(PROMPTS.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def prefix_config():
    return json.loads(PREFIX_PROMPTS.read_text(encoding="utf-8"))


def clip_solution(tok, text):
    """The completion as the judge sees it: at most max_solution_tokens tokens."""
    n = config()["max_solution_tokens"]
    ids = tok.encode(text, add_special_tokens=False)
    return text if len(ids) <= n else tok.decode(ids[:n]) + " [truncated]"


def state(problem, solution):
    return {"problem": problem, "proposed_solution": solution}


def kev_request(problem, solution, questions=QUESTIONS):
    """System One request (kev.api.SystemOneRequest fields)."""
    qs = config()["questions"]
    return {"state": state(problem, solution), "questions": {q: qs[q] for q in questions}}


def ar_messages(problem, solution, question="correct"):
    """Chat messages for the autoregressive judge: the same rendered state and question text Kev receives."""
    q = config()["questions"][question]
    st = render(state(problem, solution))
    if q["type"] == "noul":
        body = config()["ar_template"].format(state=st, question=q["instructions"])
    else:
        opts = "\n".join(f"{i}: {o}" for i, o in enumerate(q["criteria"]))
        body = f"{st}\n\nQuestion: {q['instructions']}\nOptions:\n{opts}\n\nAnswer with the number of the best option only."
    return [{"role": "user", "content": body}]


def kev_prefix_request(problem, partial):
    """System One request on an unfinished solution: the 'on_track' question."""
    return {"state": {"problem": problem, "partial_solution": partial}, "questions": dict(prefix_config()["questions"])}


def ar_prefix_messages(problem, partial):
    c = prefix_config()
    body = c["ar_template"].format(state=render({"problem": problem, "partial_solution": partial}),
                                   question=c["questions"]["on_track"]["instructions"])
    return [{"role": "user", "content": body}]


def kev_scalars(probs, meta):
    """Kev's answers as scalars in [0, 1]: P(yes) for a yes/no question, expected level / (levels - 1) for a score."""
    out = {}
    for p, m in zip(probs, meta):
        out[m["id"]] = p[1] if m["type"] == "noul" else sum(i * x for i, x in enumerate(p)) / (len(p) - 1)
    return out
