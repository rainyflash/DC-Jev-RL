"""Part III-A judge study: can Kev-4B (or the AR judge of the same size) tell code that solves a problem from code that
only passes the problem's visible tests?

    python -m dcjev.code.judge_study gen    --dir /content/work/runs/part3a/judge_study   # rollout server running
    python -m dcjev.code.judge_study score  --dir ...                                     # kev's environment, GPU
    python -m dcjev.code.judge_study report --dir ...

gen     single attempts of the base policy (the GRPO starting point; --n samples at T = 1) on kodcode_val, mbppplus and
        humanevalplus, run on the visible and the held-out tests. Plus two kinds of known code for MBPP+ and HumanEval+:
          reference   the dataset's reference solution (passes everything)
          hardcoded   a lookup of the visible tests' calls and expected values, None otherwise: passes the visible tests
                      (checked; kept only if it does) and is no solution at all
score   Kev-4B (bf16 merged) and the AR judge (Qwen3.5-4B, thinking off, one forward pass) on 'genuine' and 'correct'
        (experiments/part3a_code/prompts/code_prompts.json): problem, visible tests and code, never the held-out tests.
report  per signal: AUROC for passing the held-out tests among the policy's attempts that pass the visible tests (the
        attempts a visible-test reward cannot tell apart), and over all attempts; AUROC separating reference from
        hardcoded code; mean scores of each kind.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import time
from pathlib import Path

import numpy as np

from dcjev.code.data import load, solution_code
from dcjev.code.episode import ar_code_messages, kev_code_request, passed, run_episodes
from dcjev.code.sandbox import Executor
from dcjev.metrics import auroc

SETS = ("kodcode_val", "mbppplus", "humanevalplus")


def jl_write(path, recs):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jl_read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def hardcoded(rec):
    """A lookup of the visible asserts `assert fn(args) == expected`, or None if any visible test has another form."""
    lines = []
    for t in rec["visible"]:
        try:
            node = ast.parse(t.strip().splitlines()[-1]).body[0]
        except (SyntaxError, IndexError):
            return None
        ok = (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
              and isinstance(node.test.ops[0], ast.Eq) and isinstance(node.test.left, ast.Call)
              and isinstance(node.test.left.func, ast.Name) and node.test.left.func.id == rec["fn"] and not node.test.left.keywords)
        if not ok: return None
        args = ", ".join(ast.unparse(x) for x in node.test.left.args)
        lines.append(f"    if args == ({args}{',' if len(node.test.left.args) == 1 else ''}):\n        return {ast.unparse(node.test.comparators[0])}")
    return f"def {rec['fn']}(*args):\n" + "\n".join(lines) + "\n    return None\n"


def stage_gen(a):
    from transformers import AutoTokenizer
    from dcjev.llm.grpo import Rollouts
    from dcjev.llm.shared import BASE, BASE_REV
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    ro, ex = Rollouts(a.port), Executor(32)
    out, t0 = [], time.time()
    for name in SETS:
        recs = load(f"{a.data}/{name}.jsonl")
        eps = run_episodes(ro, tok, recs, a.n, None, ex, temperature=1.0, seed=17, max_attempts=1)
        out += [{"set": name, "id": e["rec"]["id"], "kind": "policy", "code": e["code"], "vis_pass": e["vis_pass"], "held_pass": e["held_pass"]} for e in eps]
    gen_s = time.time() - t0
    for name in ("mbppplus", "humanevalplus"):
        recs = load(f"{a.data}/{name}.jsonl")
        hc = [(r, c) for r in recs if (c := hardcoded(r))]
        vis = ex([(solution_code(r, c), r["visible"]) for r, c in hc])
        held = ex([(solution_code(r, c), r["heldout"]) for r, c in hc], per_test_s=10, total_s=60)
        out += [{"set": name, "id": r["id"], "kind": "hardcoded", "code": c, "vis_pass": True, "held_pass": passed(h, len(r["heldout"]))}
                for (r, c), v, h in zip(hc, vis, held) if passed(v, len(r["visible"]))]
        out += [{"set": name, "id": r["id"], "kind": "reference", "code": r["reference"], "vis_pass": True, "held_pass": True} for r in recs]
    jl_write(Path(a.dir) / "codes.jsonl.gz", out)
    kinds = {k: sum(o["kind"] == k for o in out) for k in ("policy", "hardcoded", "reference")}
    json.dump({"samples_per_problem": a.n, "generation_seconds": round(gen_s, 1), "kinds": kinds,
               "hardcoded_passing_heldout": sum(o["kind"] == "hardcoded" and o["held_pass"] for o in out)},
              open(Path(a.dir) / "gen.json", "w"), indent=1)
    print(kinds, flush=True)


def stage_score(a):
    import torch
    from dcjev.judges.ar_judge import ARJudge
    from dcjev.judges.kev_judge import KevJudge
    from dcjev.llm.prompts import kev_scalars
    from dcjev.llm.shared import KEV_RUN
    d = Path(a.dir)
    codes = jl_read(d / "codes.jsonl.gz")
    recs = {r["id"]: r for name in SETS for r in load(f"{a.data}/{name}.jsonl")}
    cost, out = {}, [{} for _ in codes]
    kev = KevJudge(KEV_RUN, dtype="bf16", merge=True, fused=False, shared_prefix=True)
    reqs = [kev_code_request(recs[c["id"]], c["code"]) for c in codes]
    kev.probs(reqs[:64]); torch.cuda.synchronize(); t0 = time.time()
    probs, metas = kev.probs(reqs); torch.cuda.synchronize()
    cost["kev_seconds_per_1k"] = (time.time() - t0) / len(reqs) * 1e3
    for o, p, m in zip(out, probs, metas):
        k = kev_scalars(p, m); o["kev_genuine"], o["kev_correct"] = k["genuine"], k["correct"]
    del kev; torch.cuda.empty_cache()
    ar = ARJudge("Qwen/Qwen3.5-4B", revision=a.ar_revision, fused=True)
    for q in ("genuine", "correct"):
        msgs = [ar_code_messages(recs[c["id"]], c["code"], q) for c in codes]
        ar.score(msgs[:64], "noul"); torch.cuda.synchronize(); t0 = time.time()
        res = ar.score(msgs, "noul"); torch.cuda.synchronize()
        cost[f"ar_{q}_seconds_per_1k"] = (time.time() - t0) / len(msgs) * 1e3
        for o, x in zip(out, res): o[f"ar_{q}"], o[f"ar_{q}_valid_mass"] = x["p_yes"], x["valid_mass"]
    jl_write(d / "scores.jsonl.gz", out)
    json.dump(cost, open(d / "score_cost.json", "w"), indent=1)
    print("scored", len(out), cost, flush=True)


def stage_report(a):
    d = Path(a.dir)
    codes = jl_read(d / "codes.jsonl.gz")
    for c, s in zip(codes, jl_read(d / "scores.jsonl.gz")): c.update(s)
    sig = ("kev_genuine", "kev_correct", "ar_genuine", "ar_correct")
    rep = {"generation": json.load(open(d / "gen.json")), "cost": json.load(open(d / "score_cost.json"))}
    for name in SETS + ("all",):
        pol = [c for c in codes if c["kind"] == "policy" and (name == "all" or c["set"] == name)]
        if not pol: continue
        vis = [c for c in pol if c["vis_pass"]]
        e = {"attempts": len(pol), "vis_pass": float(np.mean([c["vis_pass"] for c in pol])),
             "all_pass": float(np.mean([c["vis_pass"] and c["held_pass"] for c in pol])),
             "heldout_fail_among_vis_pass": float(np.mean([not c["held_pass"] for c in vis])) if vis else None}
        for s in sig:
            e[s] = {"auroc_heldout_given_vis": auroc([c[s] for c in vis], [c["held_pass"] for c in vis]) if vis else None,
                    "auroc_all_pass": auroc([c[s] for c in pol], [c["vis_pass"] and c["held_pass"] for c in pol]),
                    "mean": float(np.mean([c[s] for c in pol]))}
        rep[f"policy_{name}"] = e
    known = [c for c in codes if c["kind"] in ("reference", "hardcoded")]
    hc_ids = {c["id"] for c in known if c["kind"] == "hardcoded"}
    pair = [c for c in known if c["id"] in hc_ids]   # problems with both a reference and a hardcoded lookup
    rep["reference_vs_hardcoded"] = {"problems": len(hc_ids), **{s: {
        "auroc": auroc([c[s] for c in pair], [c["kind"] == "reference" for c in pair]),
        "mean_reference": float(np.mean([c[s] for c in pair if c["kind"] == "reference"])),
        "mean_hardcoded": float(np.mean([c[s] for c in pair if c["kind"] == "hardcoded"])),
        "hardcoded_scored_above_reference": float(np.mean([h[s] > r[s] for h in pair if h["kind"] == "hardcoded"
                                                            for r in pair if r["kind"] == "reference" and r["id"] == h["id"]]))}
        for s in sig}}
    json.dump(rep, open(d / "judge_study.json", "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "score", "report"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--data", default="/content/work/data/code")
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--ar_revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    a = ap.parse_args()
    Path(a.dir).mkdir(parents=True, exist_ok=True)
    {"gen": stage_gen, "score": stage_score, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
