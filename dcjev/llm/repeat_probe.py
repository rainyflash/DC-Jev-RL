"""Judges and repetition, a test added after the GRPO runs of Parts II and III-B to explain them: the final policies of
several conditions with Kev-4B's P(correct) in the reward repeat parts of their solutions (dcjev.analysis.part2,
repeated_lines), while those trained on the verifier alone do not. Does repetition itself raise a judge's P(correct)?

    python -m dcjev.llm.repeat_probe --study experiments/part2_llm/judge_study --runs <run dir> [<run dir> ...] --out <dir>

Two interventions, both leaving the final answer (the last boxed one, which the verifier reads) unchanged:
  add     the base policy's completions of the Part II judge study (MATH-500 and the GSM8K subset), as written and with
          end3:  the last two paragraphs written twice more at the end
          twice: the whole completion written twice
  remove  the final MATH-500 completions of GRPO runs that repeat lines, as written and with
          cut:   as if the policy had stopped after its first answer: up to the end of the paragraph of the last boxed
                 answer before the first line (of at least 20 characters) that repeats an earlier one; kept only if it
                 has the same final answer (repetition before any answer, or a different first answer, is left out)
Each judge scores every text as in GRPO: the solution clipped to max_solution_tokens (dcjev.llm.prompts.clip_solution),
Kev-4B in bf16, merged, with shared-prefix scoring, and the AR judge (Qwen3.5-4B, one forward pass) on 'correct'.
Report: per edit and source, the mean change of each judge's P(correct) (and Kev-4B's P(genuine), the factor of Part
II's rlvr_kev_gate) on the completions the verifier accepts and on those it rejects, with a 95% bootstrap interval over
problems and the share of completions whose score went up.
-> scores.jsonl.gz, repeat_probe.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from dcjev.analysis.part2 import repeated_lines


def jl_read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def end3(text):
    paras = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    return text.rstrip() + ("\n\n" + "\n\n".join(paras[-2:])) * 2 if paras else None


def twice(text):
    return text.rstrip() + "\n\n" + text.strip()


def cut(text, min_len=20):
    """As if the policy had stopped after its first answer: the text up to the end of the paragraph that holds the last
    boxed answer before the first line (of at least min_len characters) that repeats an earlier line. The text itself
    if no line repeats; None if no boxed answer comes before the first repeated line."""
    seen, pos = set(), 0
    for line in text.split("\n"):
        s = line.strip()
        if len(s) >= min_len:
            if s in seen:
                head = text[:pos]
                b = max(head.rfind("\\boxed"), head.rfind("\\fbox"))
                if b < 0: return None
                e = head.find("\n\n", b)
                return (head if e < 0 else head[:e]).rstrip()
            seen.add(s)
        pos += len(line) + 1
    return text


def items_of(a):
    """[(kind, edit, source, id, set, v, problem, text, n)] for the original texts and their edited copies, and the
    number of repeating completions per run whose cut changes or loses the final answer (left out)."""
    from dcjev.llm.data import last_boxed
    study = jl_read(Path(a.study) / "completions.jsonl.gz")
    problem = {c["id"]: c["problem"] for c in study}
    items, skipped = [], {}
    for n, c in enumerate(study):
        items.append(("add", "orig", "base", c["id"], c["set"], c["v"], c["problem"], c["text"], n))
        for name, fn in (("end3", end3), ("twice", twice)):
            t = fn(c["text"])
            if t is not None: items.append(("add", name, "base", c["id"], c["set"], c["v"], c["problem"], t, n))
    for run in a.runs:
        src = "/".join(Path(run).parts[-2:])   # <condition>/s<seed>
        skipped[src] = 0
        for n, c in enumerate(r for r in jl_read(Path(run) / "final_completions.jsonl.gz") if r["set"] == "math500"):
            d = cut(c["text"])
            if d == c["text"]: continue
            if d is None or last_boxed(d) is None or last_boxed(d) != last_boxed(c["text"]):
                skipped[src] += 1; continue
            items.append(("remove", "orig", src, c["id"], c["set"], c["v"], problem[c["id"]], c["text"], n))
            items.append(("remove", "cut", src, c["id"], c["set"], c["v"], problem[c["id"]], d, n))
    return items, skipped


def score(a, items):
    import torch
    from transformers import AutoTokenizer
    from dcjev.judges.ar_judge import ARJudge
    from dcjev.judges.kev_judge import KevJudge
    from dcjev.llm.prompts import ar_messages, clip_solution, kev_request, kev_scalars
    from dcjev.llm.shared import BASE, BASE_REV, KEV_RUN
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    clipped = [clip_solution(tok, it[7]) for it in items]
    t0 = time.time()
    kev = KevJudge(KEV_RUN, dtype="bf16", merge=True, fused=False, shared_prefix=True)
    probs, metas = kev.probs([kev_request(it[6], c) for it, c in zip(items, clipped)])
    ks = [kev_scalars(p, m) for p, m in zip(probs, metas)]
    del kev; torch.cuda.empty_cache()
    t1 = time.time()
    ar = ARJudge("Qwen/Qwen3.5-4B", revision=a.ar_revision, fused=True, token_budget=a.ar_token_budget)
    res = ar.score([ar_messages(it[6], c, "correct") for it, c in zip(items, clipped)], "noul")
    del ar; torch.cuda.empty_cache()
    secs = {"kev": t1 - t0, "ar": time.time() - t1}   # model loading included; measured with other jobs on the GPU
    return [{"kind": it[0], "edit": it[1], "source": it[2], "id": it[3], "set": it[4], "v": it[5], "n": it[8],
             "clipped": c.endswith(" [truncated]"), "repeated_lines": repeated_lines(it[7]), "kev": k, "ar": r["p_yes"],
             "ar_valid_mass": r["valid_mass"]} for it, c, k, r in zip(items, clipped, ks, res)], secs


def report(rows, boot=10000):
    rng = np.random.default_rng(0)
    orig = {(r["kind"], r["source"], r["n"]): r for r in rows if r["edit"] == "orig"}
    groups = defaultdict(list)
    for r in rows:
        if r["edit"] == "orig": continue
        o = orig[(r["kind"], r["source"], r["n"])]
        groups[(r["kind"], r["edit"], r["source"])].append((r, o))
    rep = {}
    for (kind, edit, src), pairs in sorted(groups.items()):
        e = {"pairs": len(pairs), "clipped_after_edit": float(np.mean([r["clipped"] for r, _ in pairs])),
             "repeated_lines": {"orig": float(np.mean([o["repeated_lines"] for _, o in pairs])),
                                "edited": float(np.mean([r["repeated_lines"] for r, _ in pairs]))}}
        for v in (1, 0):
            sel = [(r, o) for r, o in pairs if r["v"] == v]
            if not sel: continue
            for judge, get in (("kev_correct", lambda x: x["kev"]["correct"]), ("kev_genuine", lambda x: x["kev"]["genuine"]),
                               ("ar_correct", lambda x: x["ar"])):
                d = np.array([get(r) - get(o) for r, o in sel])
                by = defaultdict(list)
                for (r, _), x in zip(sel, d): by[r["id"]].append(x)
                pm = np.array([np.mean(x) for x in by.values()]); w = np.array([len(x) for x in by.values()])
                bs = []
                for _ in range(boot):   # resample problems, keep each problem's completions together
                    k = rng.integers(0, len(pm), len(pm)); bs.append(float(np.sum(pm[k] * w[k]) / np.sum(w[k])))
                e[f"{judge}_v{v}"] = {"n": len(sel), "problems": len(pm), "orig": float(np.mean([get(o) for _, o in sel])),
                                      "edited": float(np.mean([get(r) for r, _ in sel])), "shift": float(d.mean()),
                                      "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                                      "frac_up": float(np.mean(d > 0)), "frac_down": float(np.mean(d < 0))}
        rep[f"{kind}/{edit}/{src}"] = e
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, help="the Part II judge study folder (completions.jsonl.gz)")
    ap.add_argument("--runs", nargs="*", default=[], help="GRPO run folders with final_completions.jsonl.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ar_revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    ap.add_argument("--ar_token_budget", type=int, default=8192, help="tokens per AR forward pass (memory)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items, skipped = items_of(a)
    if (out / "scores.jsonl.gz").exists():
        rows, secs = jl_read(out / "scores.jsonl.gz"), None
    else:
        print(len(items), "texts to score", flush=True)
        rows, secs = score(a, items)
        with gzip.open(out / "scores.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    rep = {"runs": a.runs, "cut_changes_or_loses_answer": skipped, "seconds": secs, **report(rows)}
    (out / "repeat_probe.json").write_text(json.dumps(rep, indent=1))
    for k, e in rep.items():
        if isinstance(e, dict) and "pairs" in e:
            print(k, e["pairs"], {j: (round(e[j]["shift"], 3), [round(x, 3) for x in e[j]["ci95"]]) for j in e if j.startswith(("kev", "ar"))})


if __name__ == "__main__":
    main()
