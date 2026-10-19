"""Part II aggregate (also used for III-B and III-C): accuracies per condition (seed mean and spread), paired per-problem
differences against a baseline condition with bootstrap intervals, training curves, and the judge's view of each policy.

    python -m dcjev.analysis.part2 --root /content/work/runs/part2 --out experiments/part2_llm/results
    python -m dcjev.analysis.part2 --root .../part3b --root .../part2 --conds rlvr,rlvr_sub,sub_kev,sub_kev_recal,sub_probe \
        --baseline rlvr_sub --out experiments/part3b_labels/results
    python -m dcjev.analysis.part2 --root .../part3c --root .../part2 --conds rlvr,tree_random,tree_kev --step 100 \
        --out experiments/part3c_tree/results

Runs are <root>/<condition>/s<seed>; several roots are combined (a condition and seed found in two roots is an error).
Without --step the final evaluation is compared (final.json, final_completions.jsonl.gz); with --step N the periodic
evaluation at step N (the log's eval record, evals/stepN.jsonl.gz), so runs of different lengths meet at the same step.
Paired differences: for each benchmark, per-problem accuracy is averaged over the seeds of a condition (greedy sets: the
fraction of seeds that solve it; sampled sets: the mean over seeds x samples); the difference to the baseline is averaged
over problems, with a bootstrap over problems (10k resamples) for the 95% interval. Seed-level spread is reported too.
step0_eval: the spread of the step-0 evaluations of all included runs (the same base model each time), a direct
measure of the evaluation noise of greedy decoding on vLLM. late_eval_change: how much a run's accuracy changes between
its last two periodic evaluations, late in training (checkpoint noise). repeat_eval: the final evaluation minus the
periodic one at the last step, two greedy evaluations of the same weights.
Writing style: math500_style (final MATH-500 completions: length, repetition, boxed answers, the share that open with
an informal draft), math500_draft_by_step (that share at every periodic evaluation), and from the training completions
rollout_windows (style per 25-step window) and draft_windows (drafts against the other completions of the same group,
per 50-step window).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from dcjev.analysis import read_log

CONDS = ["rlvr", "rlvr_kev_add", "rlvr_kev_gate", "kev_only", "rlvr_ar_add"]
SETS = ["gsm8k_test", "math500", "amc23", "aime24", "aime25", "val", "gsm8k500"]
EXTRA = ("kev_correct_auroc", "kev_correct_auroc_within_group", "ar_correct_auroc", "ar_correct_auroc_within_group",
         "fill_auroc_unlabelled", "fill_ece_unlabelled", "fill_mean_unlabelled", "acc_unlabelled", "frozen_kev_auroc_unlabelled",
         "roots_finished", "on_track_auroc_leaves", "on_track_auroc_leaves_within_prompt", "audit_value_all_roots",
         "audit_selection_gain", "audit_best_possible_gain", "audit_on_track_auroc_leaves", "audit_on_track_auroc_leaves_within_prompt")
# Statistics whose 95% interval is also given: a bootstrap over the steps that logged them, pooled over seeds.
CI_KEYS = ("audit_selection_gain", "audit_on_track_auroc_leaves_within_prompt")


def per_problem(path):
    acc = defaultdict(lambda: defaultdict(list))
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l); acc[r["set"]][r["id"]].append(r["v"])
    return {s: {i: float(np.mean(v)) for i, v in d.items()} for s, d in acc.items()}


def repeated_lines(text, min_len=20):
    """Share of a completion's lines of at least min_len characters (stripped) that already occurred earlier in it: 0 for
    a solution written once, about 1/2 for one written twice. Restating the final answer does not count, since such a
    line is rarely repeated verbatim; repeating whole steps or paragraphs does."""
    seen, n, rep = set(), 0, 0
    for l in (x.strip() for x in text.split("\n")):
        if len(l) < min_len: continue
        n += 1
        if l in seen: rep += 1
        else: seen.add(l)
    return rep / n if n else 0.0


def repeats_after_answer(text, min_len=20):
    """Whether a line of at least min_len characters after the first boxed answer repeats a line before it: the solution,
    or part of it, written again after it was answered."""
    i = text.find("\\boxed")
    if i < 0: return False
    seen = {l.strip() for l in text[:i].split("\n") if len(l.strip()) >= min_len}
    return any(len(l.strip()) >= min_len and l.strip() in seen for l in text[i:].split("\n"))


DRAFT_MARKS = ("$", "\\(", "\\[", "###", "**")


def informal_opening(text, n=300, marks=DRAFT_MARKS):
    """Whether a completion opens with an informal draft: its first n characters contain no LaTeX math delimiter ($, \\(
    or \\[) and no markdown heading or bold (### or **), as in "We need to convert (0,3) ... r = sqrt(x^2 + y^2) = 3".
    Such a completion works the problem in plain text first; it may then write it up with LaTeX (style: draft_formatted)."""
    h = text[:n]
    return not any(m in h for m in marks)


def read_rollouts(path):
    """A run's training completions (rollouts.jsonl.gz: every completion of every step), without the text but with what
    the analyses read from it."""
    out = []
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l); t = r.pop("text")
            r.update(repeated_lines=repeated_lines(t), boxed_many=t.count("\\boxed") > 1, repeats_after_answer=repeats_after_answer(t),
                     draft=informal_opening(t), kev=(r.get("kev") or {}).get("correct"))
            out.append(r)
    return out


def rollout_windows(ro, width=25):
    """A run's training completions (read_rollouts) per window of `width` steps: completions, mean tokens, share
    truncated, share the verifier accepts, mean share of repeated lines, share with more than one boxed answer, share that
    repeat part of the solution after its first answer, share that open with an informal draft."""
    w = defaultdict(list)
    for r in ro:
        w[r["step"] // width].append((r["n_tok"], r["finish"] == "length", r["v"], r["repeated_lines"], r["boxed_many"],
                                      r["repeats_after_answer"], r["draft"]))
    out = []
    for k in sorted(w):
        x = np.array(w[k], dtype=float)
        out.append({"steps": [k * width, k * width + width - 1], "n": len(x), "tokens": float(x[:, 0].mean()),
                    "truncated": float(x[:, 1].mean()), "right": float(x[:, 2].mean()), "repeated_lines": float(x[:, 3].mean()),
                    "boxed_many": float(x[:, 4].mean()), "repeats_after_answer": float(x[:, 5].mean()), "draft": float(x[:, 6].mean())})
    return out


def draft_windows(ro, width=50):
    """Training completions that open with an informal draft against the others of the same group (one problem at one
    step), per window of `width` steps: the share of drafts and, over the groups that contain both kinds, the mean
    difference draft minus other of the verifier's verdict; of the advantage (completions used in the update only); of
    Kev-4B's P(correct) among completions with the same verdict, right and wrong ones apart (where it was logged); and for
    the label-budget conditions, of the reward filled in for unlabelled completions with the same verdict. For tree
    rollouts (records with a root), also the number of roots and of roots whose leaves disagree on opening with a draft."""
    groups, roots = defaultdict(list), defaultdict(list)
    for r in ro:
        groups[(r["step"], r["id"])].append(r)
        if r.get("root") is not None: roots[(r["step"], r["id"], r["root"])].append(r["draft"])
    rw = defaultdict(lambda: [0, 0])
    for (step, _, _), ds in roots.items(): rw[step // width][0] += 1; rw[step // width][1] += len(set(ds)) > 1
    w = defaultdict(lambda: defaultdict(list))
    diff = lambda d, o: float(np.mean(d) - np.mean(o)) if d and o else None
    for (step, _), rs in groups.items():
        o = w[step // width]; o["draft"].extend(r["draft"] for r in rs)
        D, F = [r for r in rs if r["draft"]], [r for r in rs if not r["draft"]]
        if not D or not F: continue
        x = {"verdict": diff([r["v"] for r in D], [r["v"] for r in F]),
             "advantage": diff([r["adv"] for r in D if r["used"]], [r["adv"] for r in F if r["used"]])}
        for v, name in ((1, "right"), (0, "wrong")):
            x["kev_" + name] = diff([r["kev"] for r in D if r["v"] == v and r["kev"] is not None],
                                    [r["kev"] for r in F if r["v"] == v and r["kev"] is not None])
            x["fill_" + name] = diff([r["r"] for r in D if r["v"] == v and r["used"] and r["labelled"] is False],
                                     [r["r"] for r in F if r["v"] == v and r["used"] and r["labelled"] is False])
        for k, d in x.items():
            if d is not None: o[k].append(d)
    out = []
    for k in sorted(w):
        o = w[k]
        e = {"steps": [k * width, k * width + width - 1], "completions": len(o["draft"]), "draft": float(np.mean(o["draft"]))}
        for key in ("verdict", "advantage", "kev_right", "kev_wrong", "fill_right", "fill_wrong"):
            if o[key]: e[key] = {"diff": float(np.mean(o[key])), "groups": len(o[key])}
        if k in rw: e["roots"], e["roots_mixed"] = rw[k]
        out.append(e)
    return out


def draft_by_step(d, sname="math500"):
    """The share of a run's completions on one set that open with an informal draft, at every periodic evaluation."""
    out = {}
    for f in (d / "evals").glob("step*.jsonl.gz"):
        with gzip.open(f, "rt") as g:
            xs = [informal_opening(r["text"]) for r in map(json.loads, g) if r["set"] == sname]
        if xs: out[int(f.name[4:].split(".")[0])] = float(np.mean(xs))
    return dict(sorted(out.items()))


def style(path, sname="math500"):
    """What the policy's completions on one set look like: tokens, truncation, how many give more than one boxed answer
    or none (the verifier reads the last one), the share of repeated lines, and Kev-4B's P(correct) for the right and
    for the wrong ones."""
    recs = []
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l)
            if r["set"] == sname: recs.append(r)
    if not recs: return {}
    boxes = [r["text"].count("\\boxed") + r["text"].count("\\fbox") for r in recs]
    out = {"tokens": float(np.mean([r["n_tok"] for r in recs])), "truncated": float(np.mean([r["finish"] == "length" for r in recs])),
           "boxed_many": float(np.mean([b > 1 for b in boxes])), "boxed_none": float(np.mean([b == 0 for b in boxes])),
           "repeated_lines": float(np.mean([repeated_lines(r["text"]) for r in recs])),
           "repeats_after_answer": float(np.mean([repeats_after_answer(r["text"]) for r in recs])),
           "draft_opening": float(np.mean([informal_opening(r["text"]) for r in recs])),
           # the same with other opening windows, and with the LaTeX delimiters as the only markers (robustness)
           "draft_opening_w150": float(np.mean([informal_opening(r["text"], 150) for r in recs])),
           "draft_opening_w600": float(np.mean([informal_opening(r["text"], 600) for r in recs])),
           "draft_opening_tex": float(np.mean([informal_opening(r["text"], marks=DRAFT_MARKS[:3]) for r in recs]))}
    drafts = [r["text"] for r in recs if informal_opening(r["text"])]
    if drafts:   # of the completions that open with a draft, the share that go on to formatted text (a marker after the opening)
        out["draft_formatted"] = float(np.mean([any(m in t for m in DRAFT_MARKS) for t in drafts]))
    for v, k in ((1, "kev_right"), (0, "kev_wrong")):
        xs = [r["kev"]["correct"] for r in recs if r["v"] == v and r.get("kev")]
        if xs: out[k] = float(np.mean(xs))
    return out


def load_runs(roots, conds, step=None):
    runs = defaultdict(dict)
    for root in roots:
        for d in sorted(Path(root).glob("*/s*")):
            cond, seed = d.parent.name, int(d.name[1:])
            if conds and cond not in conds: continue
            if not ((d / "log.jsonl").exists() or (d / "log.jsonl.gz").exists()): continue
            log = read_log(d)
            if step is None:
                if not (d / "final.json").exists(): continue
                comp = d / "final_completions.jsonl.gz"
                res = json.loads((d / "final.json").read_text()).get("eval", {})
            else:
                ev = [r for r in log if r["kind"] == "eval" and r["step"] == step]
                comp = d / "evals" / f"step{step}.jsonl.gz"
                if not ev or not comp.exists(): continue
                res = {k: v for k, v in ev[0].items() if isinstance(v, dict)}
            if seed in runs[cond]: raise ValueError(f"{cond} s{seed} found in two roots")
            ro = read_rollouts(d / "rollouts.jsonl.gz") if (d / "rollouts.jsonl.gz").exists() and step is None else None
            runs[cond][seed] = {"res": res, "log": log, "pp": per_problem(comp), "style": style(comp),
                                "rollouts": rollout_windows(ro) if ro else None, "drafts": draft_windows(ro) if ro else None,
                                "draft_steps": draft_by_step(d)}
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default="", help="conditions to include, in this order (default: all found)")
    ap.add_argument("--baseline", default="rlvr", help="condition(s) for the paired differences, comma-separated; the first is the main one")
    ap.add_argument("--step", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    conds = [c for c in a.conds.split(",") if c]
    runs = load_runs(a.root, conds, a.step)
    order = conds or [c for c in CONDS if c in runs] + sorted(c for c in runs if c not in CONDS)
    baselines = a.baseline.split(",")
    rep = {"compared_at": "final" if a.step is None else f"step {a.step}", "conditions": {}, **{"paired_vs_" + b: {} for b in baselines}}
    for c in order:
        seeds = runs.get(c)
        if not seeds: continue
        e = {"seeds": sorted(seeds)}
        for s in SETS:
            xs = [seeds[k]["res"][s]["acc"] for k in sorted(seeds) if s in seeds[k]["res"]]   # in the order of e["seeds"]
            if xs: e[s] = {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else None, "per_seed": xs}
        st = [r["style"] for r in seeds.values() if r["style"]]
        if st:
            e["math500_style"] = {k: float(np.mean([x[k] for x in st if k in x])) for k in st[0]}
            e["math500_style_per_seed"] = {str(k): seeds[k]["style"] for k in sorted(seeds) if seeds[k]["style"]}
        rw = {str(k): seeds[k]["rollouts"] for k in sorted(seeds) if seeds[k].get("rollouts")}
        if rw: e["rollout_windows"] = rw
        dw = {str(k): seeds[k]["drafts"] for k in sorted(seeds) if seeds[k].get("drafts")}
        if dw: e["draft_windows"] = dw
        e["math500_draft_by_step"] = {str(k): seeds[k]["draft_steps"] for k in sorted(seeds) if seeds[k]["draft_steps"]}
        m5 = [r["res"]["math500"] for r in seeds.values() if "math500" in r["res"]]
        if m5 and "kev" in m5[0]:
            e["math500_kev_correct_mean"] = float(np.mean([x["kev"]["correct"] for x in m5]))
            e["math500_kev_correct_auroc"] = float(np.mean([x["kev_correct_auroc"] for x in m5 if x["kev_correct_auroc"] is not None]))
        steps = [[r for r in seeds[k]["log"] if r["kind"] == "step" and (a.step is None or r["step"] < a.step)] for k in sorted(seeds)]
        last = [st[-20:] for st in steps if st]
        if last:
            keys = ("acc", "reward_mean", "len_mean", "truncated", "groups_all_wrong", "groups_all_right", "lp_absdiff_mean", "tis_clip_frac", "gen_tokens")
            e["train_last20"] = {k: float(np.mean([np.mean([r[k] for r in st]) for st in last])) for k in keys if all(k in st[0] for st in last)}
            e["train_last20"]["kev_correct"] = float(np.mean([np.mean([r["kev"]["correct"] for r in st]) for st in last]))
            e["train_last20_per_seed"] = {str(k): {x: float(np.mean([r[x] for r in st[-20:]])) for x in ("len_mean", "gen_tokens") if x in st[-1]}
                                          for k, st in zip(sorted(seeds), steps) if st}
            e["seconds_per_step"] = float(np.mean([np.mean([sum(r["seconds"].values()) for r in st]) for st in steps if st]))
            cfg = [r for s in seeds.values() for r in s["log"] if r["kind"] == "config"]
            if cfg: e["completions_per_step"] = int(cfg[0]["prompts"] * cfg[0]["group"])
            # Per-step judge statistics of the label (III-B) and tree (III-C) conditions: per seed the mean over the steps
            # of a window, then the mean over seeds. Windows are by step number, since some statistics are not logged on
            # every step (the audit of the roots not kept, every --audit_every steps; an AUROC of a step without both
            # outcomes). The tree conditions log the AUROC of the root's on-track score on the leaves of every step, which
            # outside the audit are the leaves of the two kept roots only; on the audited steps every root has leaves, as
            # in the offline prefix study (four prefixes, four continuations each), so those steps are also kept apart.
            for st in steps:
                for r in st:
                    if "audit_selection_gain" in r:
                        r["audit_on_track_auroc_leaves"] = r.get("on_track_auroc_leaves")
                        r["audit_on_track_auroc_leaves_within_prompt"] = r.get("on_track_auroc_leaves_within_prompt")
            ex = {}
            for k in EXTRA:
                wins = {"all": [], "first20": [], "last20": [], "firsthalf": [], "secondhalf": []}
                for st in steps:
                    xs = [(r["step"], r[k]) for r in st if r.get(k) is not None]
                    if not xs: continue
                    n = st[-1]["step"] + 1
                    for w, keep in (("all", lambda t: True), ("first20", lambda t: t < 20), ("last20", lambda t: t >= n - 20),
                                    ("firsthalf", lambda t: t < n / 2), ("secondhalf", lambda t: t >= n / 2)):
                        v = [x for t, x in xs if keep(t)]
                        if v: wins[w].append(float(np.mean(v)))
                if wins["all"]:
                    ex[k] = {w: float(np.mean(v)) if v else None for w, v in wins.items()}
                    ex[k]["records"] = sum(1 for st in steps for r in st if r.get(k) is not None)
                    if len(wins["all"]) == len(steps): ex[k]["per_seed"] = wins["all"]   # in the order of e["seeds"]
                    if k in CI_KEYS:
                        xs = np.array([r[k] for st in steps for r in st if r.get(k) is not None])
                        bs = np.random.default_rng(0).choice(xs, (10000, len(xs))).mean(1)
                        ex[k]["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            if ex: e["train_judge"] = ex
        rep["conditions"][c] = e
    # Evaluation noise: every run's step-0 evaluation is the same base model, evaluated in a different server state.
    s0 = defaultdict(list)
    for c in order:
        for r in runs.get(c, {}).values():
            for ev in (x for x in r["log"] if x["kind"] == "eval" and x["step"] == 0):
                for s in SETS:
                    if isinstance(ev.get(s), dict): s0[s].append(ev[s]["acc"])
    rep["step0_eval"] = {s: {"runs": len(v), "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                             "min": float(min(v)), "max": float(max(v))} for s, v in s0.items()}
    # Checkpoint noise: how much a run's greedy accuracy changes between its last two periodic evaluations (steps 150
    # and 200 of a 200-step run; with --step N, the last two up to N).
    ch = defaultdict(list)
    for c in order:
        for r in runs.get(c, {}).values():
            ev = sorted((x for x in r["log"] if x["kind"] == "eval" and (a.step is None or x["step"] <= a.step)), key=lambda x: x["step"])
            if len(ev) < 2: continue
            for s in SETS:
                if isinstance(ev[-1].get(s), dict) and isinstance(ev[-2].get(s), dict):
                    ch[s].append((ev[-2]["step"], ev[-1]["step"], ev[-1][s]["acc"] - ev[-2][s]["acc"]))
    rep["late_eval_change"] = {s: {"runs": len(v), "steps": sorted({f"{p}-{q}" for p, q, _ in v}),
                                   "mean_abs": float(np.mean([abs(d) for _, _, d in v])), "max_abs": float(max(abs(d) for _, _, d in v)),
                                   "mean": float(np.mean([d for _, _, d in v]))} for s, v in ch.items()}
    # Repeat noise: at the last step the periodic evaluation and the final one evaluate the same weights, greedy, in
    # different server states (final minus periodic; MATH-500 is in both).
    if a.step is None:
        rr = defaultdict(list)
        for c in order:
            for r in runs.get(c, {}).values():
                steps = next((x["steps"] for x in r["log"] if x["kind"] == "config"), None)
                last = [x for x in r["log"] if x["kind"] == "eval" and x["step"] == steps]
                for s in SETS:
                    if last and isinstance(last[0].get(s), dict) and isinstance(r["res"].get(s), dict) and r["res"][s]["samples"] == 1:
                        rr[s].append(r["res"][s]["acc"] - last[0][s]["acc"])
        rep["repeat_eval"] = {s: {"runs": len(v), "mean_abs": float(np.mean(np.abs(v))), "max_abs": float(np.max(np.abs(v))),
                                  "mean": float(np.mean(v))} for s, v in rr.items()}
    for b in baselines:
        base = runs.get(b, {})
        rng = np.random.default_rng(0)
        for c in order:
            seeds = runs.get(c)
            if c == b or not base or not seeds: continue
            rep["paired_vs_" + b][c] = {}
            for s in SETS:
                sets = [r["pp"].get(s, {}) for r in list(seeds.values()) + list(base.values())]
                ids = sorted(set.intersection(*[set(x) for x in sets])) if all(sets) else []
                if not ids: continue
                x = np.array([np.mean([r["pp"][s][i] for r in seeds.values()]) for i in ids])
                y = np.array([np.mean([r["pp"][s][i] for r in base.values()]) for i in ids])
                d = x - y
                boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
                rep["paired_vs_" + b][c][s] = {"problems": len(ids), "diff": float(d.mean()),
                                               "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
    (out / "summary.json").write_text(json.dumps(rep, indent=1))

    os.environ["MPLBACKEND"] = "Agg"   # a notebook's inline backend (set by Colab) would fail outside the notebook
    import matplotlib.pyplot as plt
    keys = [("acc", "train accuracy (verifier)"), ("len_mean", "completion tokens"), ("groups_all_wrong", "groups with no correct sample"),
            ("kev_correct", "Kev P(correct)")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4)); axes = axes.ravel()   # readable at text width
    for c in order:
        seeds = runs.get(c)
        if not seeds: continue
        for ax, (k, title) in zip(axes, keys):
            ys = [[r["kev"]["correct"] if k == "kev_correct" else r[k] for r in s["log"] if r["kind"] == "step"] for s in seeds.values()]
            n = min(map(len, ys)); w = 10
            if n < w: continue
            sm = np.array([np.convolve(y[:n], np.ones(w) / w, mode="valid") for y in ys])
            ax.plot(np.arange(sm.shape[1]) + w - 1, sm.mean(0), label=c); ax.set_title(title); ax.set_xlabel("GRPO step")
        for ax, sname in zip(axes[len(keys):], ("math500", "val")):
            ev = [[(r["step"], r[sname]["acc"]) for r in s["log"] if r["kind"] == "eval" and sname in r] for s in seeds.values()]
            n = min(map(len, ev))
            if n: ax.plot([x for x, _ in ev[0][:n]], np.mean([[y for _, y in e[:n]] for e in ev], 0), marker="o", label=c)
            ax.set_title(f"{sname} accuracy (greedy)"); ax.set_xlabel("GRPO step")
    axes[-1].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=150); fig.savefig(out / "training_curves.pdf")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
