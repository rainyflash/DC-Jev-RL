"""Layer-wise linear probes on the base policy's own hidden states: can the policy's front half judge its solutions, and
does such a judge pay for repetition the way Kev-4B does?

extract: run Qwen3.5-4B-Base on the GRPO prompt followed by a solution (as the judges see it: clipped to
max_solution_tokens), and keep the residual stream after layers 8, 16, 24 and 32 at the solution's last token and
averaged over the solution's tokens. Texts: the repeat probe's (dcjev.llm.repeat_probe.items_of: the Part II judge
study's 4000 base-policy completions as written and with end3/twice repetition added, and the repeating final MATH-500
completions of six GRPO runs as written and cut after their first answer), then the policy's own training completions
of sample_rollouts.js (problems from math_train.jsonl of dcjev.llm.data).

fit: L2-regularised logistic regressions on the verifier's verdicts, one per layer and readout, in two variants:
  study         trained on the judge study only;
  study+policy  the same with the policy's training completions added to every training set.
The judge study is cross-fitted in 5 folds by problem: a completion and its edited copies are scored by the model that
did not see its problem. The GRPO completions of the repeat probe and the policy's completions (variant 'study') are
scored by the model fitted on all the training data. The penalty is chosen per fit by 3-fold cross-validation over the
judge-study problems of the training set (log loss of the held-out judge-study completions).

    python probe.py extract --data <dir> --math <math_train.jsonl> --out <dir> [--limit N]
    python probe.py fit --out <dir>
<data> holds experiments/part2_llm/judge_study/completions.jsonl.gz, the six runs' final_completions.jsonl.gz under
their repository paths, and rollout_sample.jsonl.gz.
-> <out>/items.jsonl.gz (one row per text, without the text), feats.npy (float32: texts x layers x readouts x hidden),
   extract.json, scores.jsonl.gz (each probe's P(right) per text), fit.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

LAYERS = (8, 16, 24, 32)
READS = ("last", "mean")
VARIANTS = ("study", "study+policy")
RUNS = ["experiments/part2_llm/runs/kev_only/s1", "experiments/part2_llm/runs/rlvr_kev_gate/s1",
        "experiments/part3b_labels/runs/sub_kev/s1", "experiments/part3b_labels/runs/sub_kev_recal/s1",
        "experiments/part2_llm/runs/rlvr/s3", "experiments/part3c_tree/runs/tree_kev/s2"]
LAMS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0)
FOLDS, INNER = 5, 3


def jl_read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def jl_write(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")


def build_items(a):
    from dcjev.llm.data import last_boxed
    from dcjev.llm.repeat_probe import items_of
    data = Path(a.data)
    items, skipped = items_of(argparse.Namespace(study=str(data / "experiments/part2_llm/judge_study"), runs=[str(data / r) for r in RUNS]))
    rows = [{"group": "probe", "kind": k, "edit": e, "source": s, "id": i, "set": st, "v": int(v), "n": n, "problem": p, "text": t}
            for k, e, s, i, st, v, p, t, n in items]
    train = {}
    with open(a.math, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l); train[r["id"]] = r
    match, right = 0, 0   # are the ids of today's MATH train the ids the runs used? the right answers should match
    for r in jl_read(data / "rollout_sample.jsonl.gz"):
        t = train[r["id"]]
        if r["v"]:
            right += 1; match += last_boxed(r["text"]) == t["answer"]
        rows.append({"group": "policy", "kind": "rollout", "edit": "orig", "source": r["run"], "id": r["id"], "set": "math_train",
                     "v": int(r["v"]), "n": r["n"], "step": r["step"], "rep": r["rep"], "problem": t["problem"], "text": r["text"]})
    return rows, {"cut_changes_or_loses_answer": skipped, "policy_right_boxed_equals_answer": match / max(right, 1)}


def find_layers(model):
    import torch
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.split(".")[-1] == "layers" and len(mod) >= 24:
            return name, mod
    raise RuntimeError("no decoder layer list")


def extract(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dcjev.llm.data import PROMPT
    from dcjev.llm.prompts import clip_solution
    from dcjev.llm.shared import BASE, BASE_REV
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, info = build_items(a)
    if a.limit:   # smoke test: a few texts of each kind
        keep, seen = [], {}
        for r in rows:
            k = (r["group"], r["kind"], r["edit"])
            if seen.get(k, 0) < a.limit: keep.append(r); seen[k] = seen.get(k, 0) + 1
        rows = keep
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    seqs = []
    for r in rows:
        sol = clip_solution(tok, r.pop("text"))
        p = tok.encode(PROMPT.format(problem=r.pop("problem")), add_special_tokens=False)
        s = tok.encode(sol, add_special_tokens=False)
        r["clipped"], r["n_prompt"], r["n_solution"] = sol.endswith(" [truncated]"), len(p), len(s)
        seqs.append((p + s, len(p)))
    jl_write(out / "items.jsonl.gz", rows)
    print(len(rows), "texts,", sum(len(s) for s, _ in seqs), "tokens;", json.dumps(info), flush=True)

    lm = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REV, dtype=torch.bfloat16)
    bb = lm.model.to("cuda").eval()
    del lm
    name, layers = find_layers(bb)
    assert len(layers) == 32, (name, len(layers))
    H = bb.config.hidden_size if hasattr(bb.config, "hidden_size") else bb.config.text_config.hidden_size
    cap = {}
    def hook(l):
        def f(_mod, _inp, outp):
            cap[l] = outp[0] if isinstance(outp, (tuple, list)) else outp
        return f
    for l in LAYERS: layers[l - 1].register_forward_hook(hook(l))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    feats = np.lib.format.open_memmap(out / "feats.npy", mode="w+", dtype=np.float32, shape=(len(seqs), len(LAYERS), len(READS), H))
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    t0, i, done, ntok = time.time(), 0, 0, 0
    with torch.no_grad():
        while i < len(order):
            j = i + 1
            while j < len(order) and (j - i + 1) * len(seqs[order[j]][0]) <= a.token_budget: j += 1
            chunk = order[i:j]; B = len(chunk); L = len(seqs[chunk[-1]][0])
            ids = torch.full((B, L), pad, device="cuda"); att = torch.zeros((B, L), dtype=torch.long, device="cuda")
            for b, k in enumerate(chunk):   # right padding: every read position precedes the pads
                ids[b, : len(seqs[k][0])] = torch.tensor(seqs[k][0], device="cuda"); att[b, : len(seqs[k][0])] = 1
            bb(input_ids=ids, attention_mask=att, use_cache=False)
            n = torch.tensor([len(seqs[k][0]) for k in chunk], device="cuda")
            p0 = torch.tensor([seqs[k][1] for k in chunk], device="cuda")
            pos = torch.arange(L, device="cuda")[None, :]
            m = ((pos >= p0[:, None]) & (pos < n[:, None])).float()
            res = torch.empty((B, len(LAYERS), len(READS), H), device="cuda")
            for li, l in enumerate(LAYERS):
                h = cap[l].float()
                res[:, li, 0] = h[torch.arange(B, device="cuda"), n - 1]
                res[:, li, 1] = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)   # an empty solution: 0
            feats[chunk] = res.cpu().numpy()
            cap.clear()
            done += B; ntok += int(n.sum()); i = j
            if done % 2000 < B or i >= len(order):
                el = time.time() - t0
                print(f"{done}/{len(seqs)} texts, {ntok} tokens, {el:.0f} s, {ntok / el:.0f} tokens/s", flush=True)
    feats.flush()
    ex = {"texts": len(seqs), "tokens": ntok, "seconds": time.time() - t0, "layer_list": name, "hidden": H, "layers": LAYERS,
          "reads": READS, "gpu": torch.cuda.get_device_name(0), **info}
    (out / "extract.json").write_text(json.dumps(ex, indent=1))
    print("EXTRACT_DONE", json.dumps(ex), flush=True)


def fit(a):
    import torch
    import torch.nn.functional as TF
    out = Path(a.out)
    rows = jl_read(out / "items.jsonl.gz")
    feats = np.load(out / "feats.npy", mmap_mode="r")
    y_all = torch.tensor([r["v"] for r in rows], dtype=torch.float32, device="cuda")
    study = [i for i, r in enumerate(rows) if r["group"] == "probe" and r["kind"] == "add" and r["edit"] == "orig"]
    policy = [i for i, r in enumerate(rows) if r["group"] == "policy"]
    pid = {i: rows[i]["set"] + "|" + rows[i]["id"] for i in study}
    problems = sorted(set(pid.values()))
    perm = np.random.default_rng(0).permutation(len(problems))
    fold_p = {problems[j]: int(k % FOLDS) for k, j in enumerate(perm)}
    inner_p = {problems[j]: int((k // FOLDS) % INNER) for k, j in enumerate(perm)}   # balanced within every outer fold
    fold_n = {rows[i]["n"]: fold_p[pid[i]] for i in study}
    inner_n = {rows[i]["n"]: inner_p[pid[i]] for i in study}
    lam = torch.tensor(LAMS, device="cuda")

    def fit_grid(X, y):
        """One L2 logistic regression per penalty, fitted jointly (their objectives add up independently)."""
        mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
        Z = (X - mu) / sd
        W = torch.zeros(len(LAMS), X.shape[1], device="cuda", requires_grad=True)
        b = torch.zeros(len(LAMS), device="cuda", requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=a.iters, history_size=20, tolerance_grad=1e-7, tolerance_change=1e-10,
                                line_search_fn="strong_wolfe")
        Y = y[:, None].expand(-1, len(LAMS))
        def closure():
            opt.zero_grad()
            loss = TF.binary_cross_entropy_with_logits(Z @ W.T + b, Y) * len(LAMS) + 0.5 * (lam * (W * W).sum(1)).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return mu, sd, W.detach(), b.detach()

    def predict(model, X):
        mu, sd, W, b = model
        return torch.sigmoid(((X - mu) / sd) @ W.T + b)   # texts x penalties

    def logloss(p, y):
        p = p.clamp(1e-6, 1 - 1e-6)
        return -(y[:, None] * p.log() + (1 - y[:, None]) * (1 - p).log()).mean(0)

    scores = {i: {} for i in range(len(rows))}
    report = {"lams": LAMS, "folds": FOLDS, "inner": INNER, "iters": a.iters, "probes": {}}
    edited = {k: [i for i, r in enumerate(rows) if r["group"] == "probe" and r["kind"] == "add" and r["edit"] != "orig" and fold_n.get(r["n"]) == k]
              for k in range(FOLDS)}
    rest = [i for i, r in enumerate(rows) if r["group"] == "probe" and r["kind"] == "remove"]
    t0 = time.time()
    for li, l in enumerate(LAYERS):
        for ri, rd in enumerate(READS):
            X_all = torch.tensor(np.ascontiguousarray(feats[:, li, ri, :]), device="cuda")
            bad = ~torch.isfinite(X_all).all(1)
            if bad.any():
                print(f"L{l}_{rd}: {int(bad.sum())} texts with non-finite features, set to 0", flush=True)
                X_all[bad] = 0
            for var in VARIANTS:
                key = f"L{l}_{rd}_{var}"
                extra = policy if var == "study+policy" else []
                chosen = []
                for k in range(FOLDS):
                    tr = [i for i in study if fold_n[rows[i]["n"]] != k]
                    # the penalty: inner cross-validation over this training set's judge-study problems
                    loss = torch.zeros(len(LAMS), device="cuda")
                    for q in range(INNER):
                        itr = [i for i in tr if inner_n[rows[i]["n"]] != q] + extra
                        iva = [i for i in tr if inner_n[rows[i]["n"]] == q]
                        mdl = fit_grid(X_all[itr], y_all[itr])
                        loss += logloss(predict(mdl, X_all[iva]), y_all[iva]) * len(iva)
                    c = int(loss.argmin()); chosen.append(c)
                    mdl = fit_grid(X_all[tr + extra], y_all[tr + extra])
                    te = [i for i in study if fold_n[rows[i]["n"]] == k] + edited[k]
                    p = predict(mdl, X_all[te])[:, c].tolist()
                    for i, x in zip(te, p): scores[i][key] = x
                c_full = int(np.median(chosen))
                mdl = fit_grid(X_all[study + extra], y_all[study + extra])
                te = rest + (policy if var == "study" else [])
                p = predict(mdl, X_all[te])[:, c_full].tolist()
                for i, x in zip(te, p): scores[i][key] = x
                oof = torch.tensor([scores[i][key] for i in study]); yy = y_all[study].cpu()
                report["probes"][key] = {"lam_by_fold": [LAMS[c] for c in chosen], "lam_full": LAMS[c_full],
                                         "oof_mean_right": float(oof[yy == 1].mean()), "oof_mean_wrong": float(oof[yy == 0].mean())}
                print(f"{key}: lam {[LAMS[c] for c in chosen]} full {LAMS[c_full]}, OOF mean right {report['probes'][key]['oof_mean_right']:.3f} "
                      f"wrong {report['probes'][key]['oof_mean_wrong']:.3f}, {time.time() - t0:.0f} s", flush=True)
            del X_all
    report["seconds"] = time.time() - t0
    jl_write(out / "scores.jsonl.gz", [{**{k: r[k] for k in ("group", "kind", "edit", "source", "id", "set", "v", "n") if k in r},
                                        **({"step": r["step"], "rep": r["rep"]} if r["group"] == "policy" else {}), "p": scores[i]}
                                       for i, r in enumerate(rows)])
    (out / "fit.json").write_text(json.dumps(report, indent=1))
    print("FIT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "fit"])
    ap.add_argument("--data"); ap.add_argument("--math"); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="extract: at most this many texts of each kind (smoke test)")
    ap.add_argument("--token_budget", type=int, default=24576, help="extract: padded tokens per forward pass")
    ap.add_argument("--iters", type=int, default=300, help="fit: L-BFGS iterations per fit")
    a = ap.parse_args()
    extract(a) if a.stage == "extract" else fit(a)


if __name__ == "__main__":
    main()
