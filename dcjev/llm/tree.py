"""Tree rollouts for Part III-C (conditions tree_kev and tree_random of dcjev.llm.grpo).

Each prompt's group of G = keep x (G / keep) leaves grows from `roots` sampled prefixes of `prefix_tokens` tokens, of which
`keep` are continued with G / keep samples each:
  tree_kev     the roots with the highest Kev P(yes) for 'on_track' (experiments/part3c_tree/prompts/prefix_prompts.json)
  tree_random  `keep` roots chosen at random (in distribution the same as sampling `keep` roots only)
Continuations are sampled from the exact token prefix (the server continues prompt + root token ids), so each leaf is a
sample of the policy given its root; under tree_kev the roots that are continued are no longer a sample of the policy.
Leaves are trained like flat completions: the full root + continuation tokens, group baseline over the prompt's leaves.

A root that ends (stop) within prefix_tokens is already a complete solution: if kept, its leaves are copies of it.
On audit steps the roots that were not kept are continued as well, for the log only, to measure what the selection gains.
Every generated token is counted: all roots (kept or not) plus the continuations of the kept roots (audit continuations
are reported separately).
"""
from __future__ import annotations

import time

import numpy as np

from dcjev.llm.data import PROMPT, STOP
from dcjev.llm.prompts import kev_prefix_request, kev_scalars
from dcjev.metrics import auroc, within_group_auroc


def tree_rollout(ro, sb, batch, a, step, lora, audit=False):
    """-> items [(prompt index, prompt ids, leaf)] grouped by prompt like flat GRPO's, and the tree for tree_log."""
    P, R, K = len(batch), a.roots, a.keep
    B = a.group // K
    g1 = ro([PROMPT.format(problem=r["problem"]) for r in batch], n=R, temperature=a.temperature, top_p=1.0,
            max_tokens=a.prefix_tokens, stop=STOP, lora=lora,
            seeds=[(a.seed * 1_000_003 + step * 10_007 + i * R) % 2**31 for i in range(P)])
    roots = [(i, g1["prompt_token_ids"][i], c) for i in range(P) for c in g1["outputs"][i]]
    t = time.time()
    probs, metas = sb.kev_probs([kev_prefix_request(batch[i]["problem"], c["text"]) for i, _, c in roots])
    score = np.array([kev_scalars(p, m)["on_track"] for p, m in zip(probs, metas)])
    kev_s = time.time() - t
    rng = np.random.default_rng(a.seed * 104_729 + step)
    kept = np.zeros(P * R, dtype=bool)
    for i in range(P):
        idx = np.arange(i * R, (i + 1) * R)
        order = idx[np.lexsort((rng.random(R), -score[idx]))] if a.cond == "tree_kev" else rng.permutation(idx)
        kept[order[:K]] = True
    open_ = [j for j in range(P * R) if roots[j][2]["finish_reason"] == "length" and (kept[j] or audit)]
    leaves = {j: [roots[j][2]] * B for j in range(P * R) if roots[j][2]["finish_reason"] != "length"}
    if open_:
        g2 = ro(prompt_ids=[roots[j][1] + roots[j][2]["token_ids"] for j in open_], n=B, temperature=a.temperature, top_p=1.0,
                max_tokens=a.max_tokens - a.prefix_tokens, stop=STOP, lora=lora,
                seeds=[(a.seed * 2_000_003 + step * 10_007 + j * B) % 2**31 for j in open_])
        for j, outs in zip(open_, g2["outputs"]):
            r = roots[j][2]
            leaves[j] = [{"text": r["text"] + c["text"], "token_ids": r["token_ids"] + c["token_ids"],
                          "logprobs": r["logprobs"] + c["logprobs"], "finish_reason": c["finish_reason"]} for c in outs]
    items = [(roots[j][0], roots[j][1], dict(c, root=j)) for j in range(P * R) if kept[j] for c in leaves[j]]
    gen_tokens = sum(len(c["token_ids"]) for _, _, c in roots) + sum(
        len(c["token_ids"]) - len(roots[j][2]["token_ids"]) for j in open_ if kept[j] for c in leaves[j])
    audit_tokens = sum(len(c["token_ids"]) - len(roots[j][2]["token_ids"]) for j in open_ if not kept[j] for c in leaves[j])
    tree = {"roots": roots, "score": score, "kept": kept, "leaves": leaves, "R": R, "gen_tokens": gen_tokens,
            "audit_tokens": audit_tokens, "audit": audit, "kev_seconds": kev_s}
    return items, tree


def tree_log(tree, items, v, verifier, batch):
    """Per-step tree statistics: the leaves' outcome per root (and, on audit steps, for the roots that were not kept too),
    Kev's 'on_track' as a predictor of its root's leaves, and what keeping the chosen roots gained over keeping them all."""
    roots, score, kept, R = tree["roots"], tree["score"], tree["kept"], tree["R"]
    val = {}
    for (_, _, c), x in zip(items, v):
        val.setdefault(c["root"], []).append(float(x))
    if tree["audit"]:
        extra = [j for j in range(len(roots)) if not kept[j]]
        texts = [c["text"] for j in extra for c in tree["leaves"][j]]
        golds = [batch[roots[j][0]]["answer"] for j in extra for c in tree["leaves"][j]]
        vx = verifier(texts, golds) if texts else []
        n = 0
        for j in extra:
            m = len(tree["leaves"][j]); val[j] = [float(x) for x in vx[n:n + m]]; n += m
    js = sorted(val)
    s = np.concatenate([[score[j]] * len(val[j]) for j in js]); y = np.concatenate([val[j] for j in js])
    grp = np.concatenate([[roots[j][0]] * len(val[j]) for j in js])
    out = {"roots_finished": float(np.mean([c["finish_reason"] != "length" for _, _, c in roots])),
           "on_track_mean": float(score.mean()), "on_track_kept_mean": float(score[kept].mean()),
           "on_track_auroc_leaves": auroc(s, y), "on_track_auroc_leaves_within_prompt": within_group_auroc(s, y, grp)[0],
           "gen_tokens": int(tree["gen_tokens"]), "audit_tokens": int(tree["audit_tokens"]), "kev_roots_seconds": round(tree["kev_seconds"], 2)}
    if tree["audit"]:
        gain, best = [], []
        for i in range(len(roots) // R):
            V = np.array([np.mean(val[j]) for j in range(i * R, (i + 1) * R)])
            k = kept[i * R:(i + 1) * R]
            gain.append(V[k].mean() - V.mean()); best.append(np.sort(V)[::-1][:k.sum()].mean() - V.mean())
        out.update({"audit_value_all_roots": float(np.mean([np.mean(val[j]) for j in range(len(roots))])),
                    "audit_selection_gain": float(np.mean(gain)), "audit_best_possible_gain": float(np.mean(best))})
    return out
