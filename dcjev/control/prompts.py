"""Turns a window summary into judge inputs: a System One request for Kev, and the same content as chat text for the
autoregressive judge. Both read experiments/part1_control/prompts/judge_prompts.json."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from kev.api import render

PROMPTS = Path(__file__).resolve().parents[2] / "experiments" / "part1_control" / "prompts" / "judge_prompts.json"


@lru_cache(maxsize=None)
def config():
    return json.loads(PROMPTS.read_text(encoding="utf-8"))


def state(env_id: str, summary: dict) -> dict:
    e = config()["envs"][env_id]
    return {"task": e["task"], "reference": e["reference"], "segment": summary}


def kev_request(env_id: str, summary: dict, questions=("progress", "healthy")) -> dict:
    """System One request (kev.api.SystemOneRequest fields)."""
    c, e = config(), config()["envs"][env_id]
    qs = {}
    if "progress" in questions:
        qs["progress"] = {"type": "score", "instructions": c["questions"]["progress"]["instructions"], "criteria": e["progress"]}
    if "healthy" in questions:
        qs["healthy"] = {"type": "noul", "instructions": e["healthy"]}
    return {"state": state(env_id, summary), "questions": qs}


def ar_messages(env_id: str, summary: dict, question: str) -> list[dict]:
    """Chat messages for the autoregressive judge: the same rendered state and question text Kev receives, with an
    instruction to answer with one option."""
    c, e = config(), config()["envs"][env_id]
    st = render(state(env_id, summary))
    if question == "progress":
        opts = "\n".join(f"{i}: {o}" for i, o in enumerate(e["progress"]))
        body = (f"{st}\n\nQuestion: {c['questions']['progress']['instructions']}\nOptions:\n{opts}\n\n"
                "Answer with the number of the best option only.")
    else:
        body = f"{st}\n\nQuestion: {e['healthy']}\n\nAnswer with yes or no only."
    return [{"role": "user", "content": body}]
