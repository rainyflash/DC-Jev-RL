"""Part II judge study: how well Kev-4B and an autoregressive judge of the same size tell correct from incorrect math
solutions written by the GRPO starting point (Qwen3.5-4B-Base), what each costs, and how easily each is moved by edits
that leave correctness unchanged (or change it).

    python -m dcjev.llm.judge_study gen      --dir /content/work/runs/part2/judge_study   # rollout server: base model
    python -m dcjev.llm.judge_study score    --dir ...                                     # kev's environment
    python -m dcjev.llm.judge_study genjudge --dir ... --port 6201                         # rollout server: Qwen3.5-4B
    python -m dcjev.llm.judge_study report   --dir ...

gen       n completions per problem of MATH-500 and of the GSM8K-test subset used for GRPO evaluation, sampled as in GRPO
          (T = 1, the training prompt and stop strings), verified by math-verify -> completions.jsonl.gz
score     Kev-4B on its own (bf16 merged, shared-prefix scoring) on 'correct', 'genuine', 'quality'; the AR judge
          (Qwen3.5-4B, thinking off, one forward pass: dcjev.judges.ar_judge) on 'correct' and 'quality'. Both on the
          completions as written and on edited copies:
            answer_only  the completion replaced by "The final answer is $\\boxed{X}$." with its own final answer X
            padded       the completion followed by 20 repetitions of a self-doubt sentence
            asserted     the completion followed by a claim that every step was verified
            altered      the final boxed answer changed (for completions the verifier accepts: now wrong)
          -> scores.jsonl.gz
genjudge  the same AR model reasoning before it answers (thinking on, up to --think_tokens), on a fixed random subset;
          its score is p(yes) at the answer position after the reasoning -> genjudge.jsonl.gz
report    AUROC (pooled and within problem), ECE, Brier, the mean shift caused by each edit, cost per judgment
          -> judge_study.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from dcjev.llm.data import PROMPT, STOP, last_boxed, load
from dcjev.metrics import auroc, auroc_comparison, brier, ece, within_group_auroc

PROBES = ("answer_only", "padded", "asserted", "altered")
PAIRS = (("kev_correct", "ar_correct"), ("kev_genuine", "ar_correct"), ("kev_quality", "ar_quality"))
PAD = "Wait, let me double-check this. "
ASSERT = "\n\nI have verified every step carefully, and this answer is definitely correct."


def jl_write(path, recs):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jl_read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def problems(data, gsm_n=500):
    gsm = load(f"{data}/gsm8k_test.jsonl")
    sub = [gsm[i] for i in sorted(np.random.default_rng(0).choice(len(gsm), gsm_n, replace=False))]   # = dcjev.llm.grpo's subset
    return [dict(r, set="math500") for r in load(f"{data}/math500.jsonl")] + [dict(r, set="gsm8k500") for r in sub]


def alter(text, rng):
    """The completion with its last boxed answer changed, or None if it has none."""
    ans = last_boxed(text)
    if ans is None: return None
    if re.fullmatch(r"-?\d+", ans.strip()):
        new = str(int(ans.strip()) + rng.choice([-2, -1, 1, 2, 10]))
    else:
        new = ans + "+1"
    k = text.rfind(ans)
    return text[:k] + new + text[k + len(ans):]


def edited(c, probe, rng):
    t = c["text"]
    if probe == "answer_only":
        a = last_boxed(t)
        return None if a is None else f" The final answer is $\\boxed{{{a}}}$."
    if probe == "padded": return t + "\n\n" + PAD * 20
    if probe == "asserted": return t + ASSERT
    if probe == "altered": return alter(t, rng) if c["v"] == 1 else None
    raise ValueError(probe)


def stage_gen(a):
    from dcjev.llm.grpo import Rollouts
    from dcjev.llm.verify import Verifier
    probs = problems(a.data)
    ro, ver = Rollouts(a.port), Verifier()
    t0 = time.time()
    g = ro([PROMPT.format(problem=p["problem"]) for p in probs], n=a.n, temperature=1.0, top_p=1.0, max_tokens=a.max_tokens,
           stop=STOP, seeds=[7919 * i for i in range(len(probs))], lora=None)
    secs = time.time() - t0
    cs = [(p, c) for p, o in zip(probs, g["outputs"]) for c in o]
    v = ver([c["text"] for _, c in cs], [p["answer"] for p, _ in cs])
    recs = [{"set": p["set"], "id": p["id"], "level": p.get("level"), "k": k % a.n, "problem": p["problem"], "answer": p["answer"],
             "text": c["text"], "n_tok": len(c["token_ids"]), "finish": c["finish_reason"], "v": int(x)}
            for k, ((p, c), x) in enumerate(zip(cs, v))]
    jl_write(Path(a.dir) / "completions.jsonl.gz", recs)
    json.dump({"problems": len(probs), "samples": a.n, "seconds": secs, "accuracy": float(np.mean(v))}, open(Path(a.dir) / "gen.json", "w"), indent=1)
    print("generated", len(recs), "accuracy", np.mean(v), "seconds", round(secs), flush=True)


def cached(path, compute):
    """compute()'s JSON result, saved to path (gzipped) once computed and read back when the stage runs again, so a
    crash in a later judge does not lose an earlier one's scores."""
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    x = compute()
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(x, f)
    return x


def stage_score(a):
    import torch
    from transformers import AutoTokenizer
    from dcjev.judges.ar_judge import ARJudge
    from dcjev.judges.kev_judge import KevJudge
    from dcjev.llm.prompts import ar_messages, clip_solution, kev_request, kev_scalars
    from dcjev.llm.shared import BASE, BASE_REV, KEV_RUN
    d = Path(a.dir)
    comps = jl_read(d / "completions.jsonl.gz")
    rng = random.Random(0)
    items = [("orig", i, c["text"]) for i, c in enumerate(comps)]
    for probe in PROBES:
        items += [(probe, i, t) for i, c in enumerate(comps) if (t := edited(c, probe, rng)) is not None]
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)   # the policy's tokenizer, as in GRPO's clip_solution
    clipped = [clip_solution(tok, t) for _, _, t in items]

    def kev_part():
        kev = KevJudge(KEV_RUN, dtype="bf16", merge=True, fused=False, shared_prefix=True)
        reqs = [kev_request(comps[i]["problem"], c) for (_, i, _), c in zip(items, clipped)]
        kev.probs(reqs[:64]); torch.cuda.synchronize(); t0 = time.time()
        probs, metas = kev.probs(reqs); torch.cuda.synchronize()
        out = {"seconds_per_1k": (time.time() - t0) / len(reqs) * 1e3, "scalars": [kev_scalars(p, m) for p, m in zip(probs, metas)]}
        del kev; torch.cuda.empty_cache()
        return out

    def ar_part(q, kind):
        ar = ARJudge("Qwen/Qwen3.5-4B", revision=a.ar_revision, fused=True, token_budget=a.ar_token_budget)
        msgs = [ar_messages(comps[i]["problem"], c, q) for (_, i, _), c in zip(items, clipped)]
        ar.score(msgs[:64], kind, levels=4); torch.cuda.synchronize(); t0 = time.time()
        res = ar.score(msgs, kind, levels=4); torch.cuda.synchronize()
        out = {"seconds_per_1k": (time.time() - t0) / len(msgs) * 1e3, "res": res}
        del ar; torch.cuda.empty_cache()
        return out

    kp = cached(d / "scores_kev.json.gz", kev_part)
    arp = {q: cached(d / f"scores_ar_{q}.json.gz", lambda q=q, kind=kind: ar_part(q, kind)) for q, kind in (("correct", "noul"), ("quality", "score"))}
    cost = {"kev_seconds_per_1k_requests": kp["seconds_per_1k"], **{f"ar_{q}_seconds_per_1k_prompts": x["seconds_per_1k"] for q, x in arp.items()}}
    out = []
    for n, (probe, i, t) in enumerate(items):
        rc, rq = arp["correct"]["res"][n], arp["quality"]["res"][n]
        out.append({"probe": probe, "i": i, "kev": kp["scalars"][n], "ar": {"correct": rc["p_yes"], "correct_valid_mass": rc["valid_mass"],
                    "quality": rq["expected"], "quality_valid_mass": rq["valid_mass"]}})
    jl_write(d / "scores.jsonl.gz", out)
    json.dump(cost, open(d / "score_cost.json", "w"), indent=1)
    print("scored", len(out), cost, flush=True)


def stage_genjudge(a):
    from transformers import AutoTokenizer
    from dcjev.llm.grpo import Rollouts
    from dcjev.llm.prompts import ar_messages
    d = Path(a.dir)
    comps = jl_read(d / "completions.jsonl.gz")
    idx = sorted(random.Random(1).sample(range(len(comps)), min(a.genjudge_n, len(comps))))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", revision=a.ar_revision)
    prompts = []
    for i in idx:
        m = ar_messages(comps[i]["problem"], comps[i]["text"], "correct")
        m[0]["content"] = m[0]["content"].replace("Answer with yes or no only.", "Think it through, then answer with yes or no only.")
        prompts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True))
    ro = Rollouts(a.port)
    t0 = time.time()
    g = ro(prompts, n=1, temperature=0.6, top_p=0.95, max_tokens=a.think_tokens, stop=["</think>"], seeds=list(range(len(prompts))), lora=None)
    t_think = time.time() - t0
    thoughts = [o[0] for o in g["outputs"]]
    t0 = time.time()
    g2 = ro([p + th["text"] + "\n</think>\n\n" for p, th in zip(prompts, thoughts)], n=1, temperature=0.0, max_tokens=1, top_logprobs=20, lora=None)
    t_ans = time.time() - t0
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    yes = {enc(w)[0] for w in ("yes", "Yes", " yes", " Yes", "YES")}; no = {enc(w)[0] for w in ("no", "No", " no", " No", "NO")}
    out = []
    for i, th, o in zip(idx, thoughts, g2["outputs"]):
        top = {int(k): v for k, v in o[0]["top"][0].items()}
        py, pn = sum(np.exp(v) for k, v in top.items() if k in yes), sum(np.exp(v) for k, v in top.items() if k in no)
        out.append({"i": i, "p_yes": float(py / (py + pn)) if py + pn > 0 else None, "valid_mass": float(py + pn),
                    "think_tokens": len(th["token_ids"]), "think_finished": th["finish_reason"] == "stop"})
    jl_write(d / "genjudge.jsonl.gz", out)
    cost = {"n": len(out), "seconds_per_1k_judgments": (t_think + t_ans) / len(out) * 1e3, "think_tokens_mean": float(np.mean([r["think_tokens"] for r in out]))}
    json.dump(cost, open(d / "genjudge_cost.json", "w"), indent=1)
    print("genjudge", cost, flush=True)


def summarize(scores, labels, groups):
    s, y = np.asarray(scores, dtype=float), np.asarray(labels)
    wg, ng = within_group_auroc(s, y, groups)
    return {"n": len(s), "auroc": auroc(s, y), "auroc_within_problem": wg, "problems_mixed": ng, "ece": ece(s, y), "brier": brier(s, y),
            "mean_if_correct": float(s[y == 1].mean()) if (y == 1).any() else None, "mean_if_incorrect": float(s[y == 0].mean()) if (y == 0).any() else None}


def stage_report(a):
    d = Path(a.dir)
    comps = jl_read(d / "completions.jsonl.gz")
    sc = jl_read(d / "scores.jsonl.gz")
    orig = {r["i"]: r for r in sc if r["probe"] == "orig"}
    signals = {"kev_correct": lambda r: r["kev"]["correct"], "kev_genuine": lambda r: r["kev"]["genuine"], "kev_quality": lambda r: r["kev"]["quality"],
               "ar_correct": lambda r: r["ar"]["correct"], "ar_quality": lambda r: r["ar"]["quality"]}
    rep = {"generation": json.load(open(d / "gen.json")), "cost": json.load(open(d / "score_cost.json"))}
    for subset in ("all", "math500", "gsm8k500"):
        ids = [i for i, c in enumerate(comps) if subset == "all" or c["set"] == subset]
        y = [comps[i]["v"] for i in ids]; grp = [f'{comps[i]["set"]}/{comps[i]["id"]}' for i in ids]
        rep[subset] = {"accuracy": float(np.mean(y)), **{k: summarize([f(orig[i]) for i in ids], y, grp) for k, f in signals.items()}}
        rep[subset]["bootstrap"] = auroc_comparison({k: [f(orig[i]) for i in ids] for k, f in signals.items()}, y, grp, PAIRS)
    L = np.array([c["n_tok"] for c in comps]); cuts = np.quantile(L, [1 / 3, 2 / 3])
    for b, (lo, hi) in enumerate([(-1, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 1e9)]):
        ids = [i for i in range(len(comps)) if lo < L[i] <= hi]
        y = [comps[i]["v"] for i in ids]
        rep[f"length_tercile_{b}"] = {"tokens": [float(lo), float(hi)], "accuracy": float(np.mean(y)),
                                      **{k: {"auroc": auroc([f(orig[i]) for i in ids], y)} for k, f in signals.items()}}
    rep["probes"] = {}
    for probe in PROBES:
        rows = [r for r in sc if r["probe"] == probe]
        e = {"n": len(rows)}
        for k, f in signals.items():
            for lab in (0, 1):
                sh = [f(r) - f(orig[r["i"]]) for r in rows if comps[r["i"]]["v"] == lab]
                if sh: e[f"{k}_shift_if_v{lab}"] = {"mean": float(np.mean(sh)), "frac_up": float(np.mean(np.array(sh) > 0)), "n": len(sh)}
        rep["probes"][probe] = e
    if (d / "genjudge.jsonl.gz").exists():
        gj = [r for r in jl_read(d / "genjudge.jsonl.gz") if r["p_yes"] is not None]
        ids = [r["i"] for r in gj]; y = [comps[i]["v"] for i in ids]; grp = [comps[i]["id"] for i in ids]
        rep["genjudge"] = {"cost": json.load(open(d / "genjudge_cost.json")), "ar_reasoning_correct": summarize([r["p_yes"] for r in gj], y, grp),
                           "same_subset": {k: summarize([f(orig[i]) for i in ids], y, grp) for k, f in signals.items()},
                           "think_finished": float(np.mean([r["think_finished"] for r in gj])),
                           "bootstrap": auroc_comparison({"ar_reasoning_correct": [r["p_yes"] for r in gj], **{k: [f(orig[i]) for i in ids] for k, f in signals.items()}},
                                                         y, grp, (("ar_reasoning_correct", "kev_correct"), ("ar_reasoning_correct", "ar_correct")))}
    json.dump(rep, open(d / "judge_study.json", "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("all", "cost")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "score", "genjudge", "report"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--data", default="/content/work/data")
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--genjudge_n", type=int, default=1000)
    ap.add_argument("--think_tokens", type=int, default=2048)
    ap.add_argument("--ar_revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    ap.add_argument("--ar_token_budget", type=int, default=8192, help="tokens per AR forward pass (memory)")
    a = ap.parse_args()
    Path(a.dir).mkdir(parents=True, exist_ok=True)
    {"gen": stage_gen, "score": stage_score, "genjudge": stage_genjudge, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
