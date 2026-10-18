"""Part III-C offline study: can a judge tell from the first tokens of a solution whether it will end correct, well enough
to choose which partial solutions to continue?

    python -m dcjev.llm.prefix_study gen    --dir /content/work/runs/part3c/prefix_study   # rollout server running
    python -m dcjev.llm.prefix_study score  --dir ...                                      # kev's environment, GPU
    python -m dcjev.llm.prefix_study report --dir ...

gen     GRPO's validation problems (500 MATH train problems held out with seed 12345, never trained on): --roots solutions
        per problem sampled as in GRPO (T = 1, the training prompt) up to max(--cuts) tokens. Each solution is cut at every
        --cuts length it runs past, and every prefix gets --branches continuations (to --max_tokens in all, as in
        training), verified by math-verify. The fraction of a prefix's continuations that end correct is its Monte-Carlo
        value.
score   Kev-4B (bf16 merged, as Part II's judge study) on 'on_track' (experiments/part3c_tree/prompts/prefix_prompts.json)
        and on Part II's 'correct' question with the prefix as the proposed solution; the AR judge (Qwen3.5-4B, thinking
        off, one forward pass) on 'on_track'; the policy's own mean token log-probability of the prefix (free).
report  per cut length and signal: AUROC for a continuation ending correct (pooled, and within problem: the comparison
        that matters for choosing among one problem's prefixes), Spearman correlation with the Monte-Carlo value, and the
        pruning experiment on the problems whose --roots solutions all run past the cut: keep the --keep prefixes the
        signal ranks highest and compare their mean value with keeping all of them (what a random choice keeps in
        expectation) and with an oracle that ranks by half of the continuations and is scored on the other half; 95%
        bootstrap intervals over problems. Also the threshold rule of the withdrawn v3 paper (prune when the judge's
        P(yes) < 0.3): how much it prunes, how many pruned prefixes were dead (value 0), and how many correct
        continuations it throws away.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

from dcjev.llm.data import PROMPT, STOP, train_val
from dcjev.metrics import auroc, ranks, within_group_auroc


def jl_write(path, recs):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jl_read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def stage_gen(a):
    from transformers import AutoTokenizer
    from dcjev.llm.grpo import Rollouts
    from dcjev.llm.shared import BASE, BASE_REV
    from dcjev.llm.verify import Verifier
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    _, val = train_val(a.data)
    if a.limit: val = val[:a.limit]
    cuts = sorted(int(x) for x in a.cuts.split(","))
    ro, ver = Rollouts(a.port), Verifier()
    lora = {"name": f"prefix-study-{Path(a.lora).name}", "path": a.lora} if a.lora else None
    t0 = time.time()
    g = ro([PROMPT.format(problem=r["problem"]) for r in val], n=a.roots, temperature=1.0, top_p=1.0, max_tokens=cuts[-1], stop=STOP,
           seeds=[7907 * i for i in range(len(val))], lora=lora)
    roots = [(i, k, g["prompt_token_ids"][i], c) for i in range(len(val)) for k, c in enumerate(g["outputs"][i])]
    ended = [j for j, r in enumerate(roots) if r[3]["finish_reason"] != "length"]
    v_ended = dict(zip(ended, ver([roots[j][3]["text"] for j in ended], [val[roots[j][0]]["answer"] for j in ended])))
    pre = [(j, cut) for j, (_, _, _, c) in enumerate(roots) for cut in cuts
           if len(c["token_ids"]) > cut or (len(c["token_ids"]) == cut and c["finish_reason"] == "length")]
    conts = {}
    for cut in cuts:   # one request per cut: the continuation budget is --max_tokens - cut
        js = [j for j, c in pre if c == cut]
        if not js: continue
        gg = ro(prompt_ids=[roots[j][2] + roots[j][3]["token_ids"][:cut] for j in js], n=a.branches, temperature=1.0, top_p=1.0,
                max_tokens=a.max_tokens - cut, stop=STOP, seeds=[(104_729 * j + cut) % 2**31 for j in js], lora=lora)
        for j, outs in zip(js, gg["outputs"]): conts[(j, cut)] = outs
    secs = time.time() - t0
    recs = []
    for j, cut in pre:
        i, k, _, c = roots[j]
        recs.append({"pid": i, "id": val[i]["id"], "level": val[i]["level"], "root": k, "cut": cut, "problem": val[i]["problem"],
                     "answer": val[i]["answer"], "prefix": tok.decode(c["token_ids"][:cut], skip_special_tokens=True),
                     "prefix_lp_mean": float(np.mean(c["logprobs"][:cut])),
                     "cont": [{"text": o["text"], "n_tok": len(o["token_ids"]), "finish": o["finish_reason"]} for o in conts[(j, cut)]]})
    v = ver([r["prefix"] + o["text"] for r in recs for o in r["cont"]], [r["answer"] for r in recs for _ in r["cont"]])
    n = 0
    for r in recs:
        for o in r["cont"]: o["v"] = int(v[n]); n += 1
    jl_write(Path(a.dir) / "prefixes.jsonl.gz", recs)
    info = [{"pid": i, "root": k, "n_tok": len(c["token_ids"]), "finish": c["finish_reason"], "v": int(v_ended[j]) if j in v_ended else None}
            for j, (i, k, _, c) in enumerate(roots)]
    jl_write(Path(a.dir) / "roots.jsonl.gz", info)
    json.dump({"problems": len(val), "roots": a.roots, "branches": a.branches, "cuts": cuts, "max_tokens": a.max_tokens, "lora": a.lora or None,
               "prefixes": len(recs), "continuations": int(len(v)), "continuation_accuracy": float(np.mean(v)),
               "roots_ended_before_last_cut": len(ended), "seconds": round(secs, 1)}, open(Path(a.dir) / "gen.json", "w"), indent=1)
    print("prefixes", len(recs), "continuations", len(v), "accuracy", float(np.mean(v)), "seconds", round(secs), flush=True)


def stage_score(a):
    import torch
    from dcjev.judges.ar_judge import ARJudge
    from dcjev.judges.kev_judge import KevJudge
    from dcjev.llm.judge_study import cached
    from dcjev.llm.prompts import ar_prefix_messages, kev_prefix_request, kev_request, kev_scalars
    from dcjev.llm.shared import KEV_RUN
    d = Path(a.dir)
    recs = jl_read(d / "prefixes.jsonl.gz")

    def kev_part():
        kev, res = KevJudge(KEV_RUN, dtype="bf16", merge=True, fused=False, shared_prefix=True), {}
        for name, q, reqs in (("kev_on_track", "on_track", [kev_prefix_request(r["problem"], r["prefix"]) for r in recs]),
                              ("kev_correct", "correct", [kev_request(r["problem"], r["prefix"], questions=("correct",)) for r in recs])):
            kev.probs(reqs[:64]); torch.cuda.synchronize(); t0 = time.time()
            probs, metas = kev.probs(reqs); torch.cuda.synchronize()
            res[name] = {"seconds_per_1k": (time.time() - t0) / len(reqs) * 1e3, "p": [kev_scalars(p, m)[q] for p, m in zip(probs, metas)]}
        del kev; torch.cuda.empty_cache()
        return res

    def ar_part():
        ar = ARJudge("Qwen/Qwen3.5-4B", revision=a.ar_revision, fused=True, token_budget=a.ar_token_budget)
        msgs = [ar_prefix_messages(r["problem"], r["prefix"]) for r in recs]
        ar.score(msgs[:64], "noul"); torch.cuda.synchronize(); t0 = time.time()
        res = ar.score(msgs, "noul"); torch.cuda.synchronize()
        out = {"seconds_per_1k": (time.time() - t0) / len(msgs) * 1e3, "p": [x["p_yes"] for x in res], "mass": [x["valid_mass"] for x in res]}
        del ar; torch.cuda.empty_cache()
        return out

    kp = cached(d / "scores_kev.json.gz", kev_part)
    ap_ = cached(d / "scores_ar.json.gz", ar_part)
    out = [{"kev_on_track": kp["kev_on_track"]["p"][n], "kev_correct": kp["kev_correct"]["p"][n],
            "ar_on_track": ap_["p"][n], "ar_on_track_valid_mass": ap_["mass"][n]} for n in range(len(recs))]
    cost = {f"{k}_seconds_per_1k": v["seconds_per_1k"] for k, v in kp.items()} | {"ar_on_track_seconds_per_1k": ap_["seconds_per_1k"]}
    jl_write(d / "scores.jsonl.gz", out)
    json.dump(cost, open(d / "score_cost.json", "w"), indent=1)
    print("scored", len(out), cost, flush=True)


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return None
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def boot(x, reps=10000, seed=0):
    x = np.asarray(x, dtype=float); rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)]
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def selection(groups, score, keep, seed=0):
    """groups: per problem the array of prefix values (all continuations) and of half-split values; score: per problem
    the signal of each prefix. -> mean value kept minus mean value of all, per problem."""
    rng = np.random.default_rng(seed)
    return np.array([V[np.lexsort((rng.random(len(V)), -s))[:keep]].mean() - V.mean() for V, s in zip(groups, score)])


def stage_report(a):
    d = Path(a.dir)
    recs = jl_read(d / "prefixes.jsonl.gz")
    for r, s in zip(recs, jl_read(d / "scores.jsonl.gz")): r.update(s)
    gen = json.load(open(d / "gen.json"))
    signals = {"kev_on_track": lambda r: r["kev_on_track"], "kev_correct": lambda r: r["kev_correct"],
               "ar_on_track": lambda r: r["ar_on_track"], "policy_logprob": lambda r: r["prefix_lp_mean"]}
    rep = {"generation": gen, "cost": json.load(open(d / "score_cost.json")), "keep": a.keep}
    for cut in gen["cuts"]:
        R = [r for r in recs if r["cut"] == cut]
        if not R: continue
        V = np.array([np.mean([o["v"] for o in r["cont"]]) for r in R])
        nb = [len(r["cont"]) for r in R]
        y = np.array([o["v"] for r in R for o in r["cont"]]); grp = np.repeat([r["pid"] for r in R], nb)
        e = {"prefixes": len(R), "problems": len({r["pid"] for r in R}), "value_mean": float(V.mean()),
             "value_zero_frac": float((V == 0).mean()), "value_one_frac": float((V == 1).mean())}
        full = {}   # problems with all --roots prefixes at this cut
        for n, r in enumerate(R): full.setdefault(r["pid"], []).append(n)
        full = [ix for ix in full.values() if len(ix) == gen["roots"]]
        Vg = [V[ix] for ix in full]
        h = min(nb) // 2   # oracle: rank by the first h continuations, score on the rest
        first = [np.array([np.mean([o["v"] for o in R[n]["cont"][:h]]) for n in ix]) for ix in full]
        second = [np.array([np.mean([o["v"] for o in R[n]["cont"][h:]]) for n in ix]) for ix in full]
        g_or = selection(second, first, a.keep)
        e["selection"] = {"problems": len(full), "keep": a.keep, "value_all": float(np.mean([v.mean() for v in Vg])) if Vg else None,
                          "oracle_split_half_gain": {"mean": float(g_or.mean()), "ci95": boot(g_or)} if full else None}
        for name, f in signals.items():
            s = np.array([f(r) for r in R])
            sr = np.repeat(s, nb)
            gain = selection(Vg, [s[ix] for ix in full], a.keep)
            sig = {"auroc": auroc(sr, y), "auroc_within_problem": within_group_auroc(sr, y, grp)[0], "spearman_value": spearman(s, V),
                   "selection_gain": {"mean": float(gain.mean()), "ci95": boot(gain)} if full else None}
            if name != "policy_logprob":
                p = s < 0.3
                sig["threshold_0.3"] = {"pruned_frac": float(p.mean()),
                                        "dead_among_pruned": float((V[p] == 0).mean()) if p.any() else None,
                                        "dead_caught": float(p[V == 0].mean()) if (V == 0).any() else None,
                                        "correct_continuations_lost": float(V[p].sum() / V.sum()) if V.sum() else None}
            e[name] = sig
        rep[f"cut_{cut}"] = e
    json.dump(rep, open(d / "prefix_study.json", "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "score", "report"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--data", default="/content/work/data")
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--lora", default="", help="policy LoRA in vLLM layout (default: the base model, GRPO's starting point)")
    ap.add_argument("--roots", type=int, default=4)
    ap.add_argument("--branches", type=int, default=4)
    ap.add_argument("--cuts", default="128,256,512")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ar_revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    ap.add_argument("--ar_token_budget", type=int, default=8192, help="tokens per AR forward pass (memory)")
    a = ap.parse_args()
    Path(a.dir).mkdir(parents=True, exist_ok=True)
    {"gen": stage_gen, "score": stage_score, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
