"""Math datasets for Part II/III, normalised to JSONL records {id, problem, answer, source, level}.

    python -m dcjev.llm.data --out /content/work/data

Training: GSM8K train + MATH train (EleutherAI/hendrycks_math, all subjects). Evaluation: GSM8K test, MATH-500
(HuggingFaceH4/MATH-500, a subset of the MATH *test* split, so disjoint from MATH train), AMC23 and AIME 2024/2025.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROMPT = ("Solve the following math problem step by step. Put your final answer within \\boxed{{}}.\n\n"
          "Problem: {problem}\n\nSolution:")
STOP = ["\n\nProblem:", "\nProblem:", "\n\n\n\n"]
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}) with balanced braces, else None."""
    i = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{": depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def gsm8k_answer(sol: str) -> str:
    return sol.split("####")[-1].strip().replace(",", "")


def build(out: Path):
    from datasets import load_dataset
    out.mkdir(parents=True, exist_ok=True)
    def dump(name, recs):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(recs), flush=True)
    g = load_dataset("openai/gsm8k", "main")
    for split in ("train", "test"):
        dump(f"gsm8k_{split}", [{"id": f"gsm8k-{split}-{i}", "problem": r["question"], "answer": gsm8k_answer(r["answer"]),
                                 "source": "gsm8k", "level": None} for i, r in enumerate(g[split])])
    recs = []
    for subj in MATH_SUBJECTS:
        d = load_dataset("EleutherAI/hendrycks_math", subj, split="train")
        for i, r in enumerate(d):
            a = last_boxed(r["solution"])
            if a is None: continue
            recs.append({"id": f"math-train-{subj}-{i}", "problem": r["problem"], "answer": a, "source": "math", "level": r["level"], "subject": subj})
    dump("math_train", recs)
    m5 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    dump("math500", [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"], "source": "math500", "level": r["level"], "subject": r["subject"]} for r in m5])
    a24 = load_dataset("HuggingFaceH4/aime_2024", split="train")
    dump("aime24", [{"id": f"aime24-{r['id']}", "problem": r["problem"], "answer": str(r["answer"]), "source": "aime24", "level": None} for r in a24])
    try:
        a25 = load_dataset("math-ai/aime25", split="test")
        dump("aime25", [{"id": f"aime25-{i}", "problem": r["problem"], "answer": str(r["answer"]), "source": "aime25", "level": None} for i, r in enumerate(a25)])
    except Exception as e:
        print("aime25 unavailable:", e)
    amc = load_dataset("math-ai/amc23", split="test")
    dump("amc23", [{"id": f"amc23-{i}", "problem": r["question"], "answer": str(r["answer"]), "source": "amc23", "level": None} for i, r in enumerate(amc)])
    # MATH-500 problems must not appear in MATH train (sanity check on normalised text)
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    train_p = {norm(r["problem"]) for r in recs}
    overlap = sum(norm(r["problem"]) in train_p for r in m5)
    print("MATH-500 problems found in MATH train:", overlap, flush=True)
    (out / "README.json").write_text(json.dumps({"math500_overlap_with_math_train": overlap}, indent=1))


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def train_val(data, train_sets="math_train", math_levels="3,4,5", holdout=500):
    """GRPO's training problems and the fixed validation holdout (chosen with seed 12345) used for model selection."""
    import numpy as np
    levels = {f"Level {x}" for x in math_levels.split(",") if x}
    train = [r for name in train_sets.split(",") for r in load(f"{data}/{name}.jsonl")
             if not (levels and r["source"] == "math" and r["level"] not in levels)]
    held = set(np.random.default_rng(12345).choice(len(train), holdout, replace=False).tolist()) if holdout else set()
    return [r for i, r in enumerate(train) if i not in held], [r for i, r in enumerate(train) if i in held]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    build(Path(ap.parse_args().out))
