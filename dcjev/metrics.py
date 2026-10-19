"""Small, dependency-free metrics shared by the trainers and the analysis scripts."""
from __future__ import annotations

import numpy as np


def ranks(x):
    """1-based ranks with ties averaged."""
    _, inv, cnt = np.unique(np.asarray(x, dtype=float), return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2)[inv.ravel()]


def auroc(scores, labels):
    """Probability that a random positive scores above a random negative (ties count one half); None if a class is empty."""
    y = np.asarray(labels).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = ranks(scores)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(probs, labels, bins=10):
    """Expected calibration error of P(label = 1) with equal-width bins."""
    p, y = np.asarray(probs, dtype=float), np.asarray(labels, dtype=float)
    if len(p) == 0:
        return None
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() for b in range(bins) if (idx == b).any()) / len(p))


def brier(probs, labels):
    p, y = np.asarray(probs, dtype=float), np.asarray(labels, dtype=float)
    return float(((p - y) ** 2).mean()) if len(p) else None


def within_group_auroc(scores, labels, groups):
    """Mean AUROC over the groups that contain both labels (what GRPO's group baseline sees), and how many such groups."""
    s, y, g = np.asarray(scores, dtype=float), np.asarray(labels), np.asarray(groups)
    vals = [a for k in np.unique(g) if (a := auroc(s[g == k], y[g == k])) is not None]
    return (float(np.mean(vals)) if vals else None), len(vals)


def auroc_comparison(signals, labels, groups, pairs=(), n=2000, seed=0):
    """AUROC over all items and mean within-group AUROC of each signal, and their differences for the (a, b) pairs of
    signal names, each with a 95% percentile interval from resampling the groups (problems) with replacement."""
    y = np.asarray(labels).astype(bool)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inv.ravel() == k) for k in range(inv.max() + 1)]
    S = {k: np.asarray(v, dtype=float) for k, v in signals.items()}
    mixed = np.array([0 < y[m].sum() < len(m) for m in members])
    per_group = {k: np.array([auroc(s[m], y[m]) if mx else np.nan for m, mx in zip(members, mixed)]) for k, s in S.items()}

    def stat(gs):
        idx = np.concatenate([members[g] for g in gs])
        a = {k: auroc(s[idx], y[idx]) for k, s in S.items()}
        w = {k: float(np.mean(pg[gs][mixed[gs]])) if mixed[gs].any() else None for k, pg in per_group.items()}
        row = ([a[k] for k in S] + [w[k] for k in S] + [None if None in (a[p], a[q]) else a[p] - a[q] for p, q in pairs]
               + [None if None in (w[p], w[q]) else w[p] - w[q] for p, q in pairs])
        return [np.nan if v is None else v for v in row]

    point = stat(np.arange(len(members)))
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.integers(0, len(members), len(members))) for _ in range(n)])
    lo, hi = np.nanquantile(boots, 0.025, axis=0), np.nanquantile(boots, 0.975, axis=0)
    cells = [{"value": None if np.isnan(v) else float(v), "ci95": [float(l), float(h)]} for v, l, h in zip(point, lo, hi)]
    k, m, names = len(S), len(pairs), list(S)
    return {"groups": len(members), "groups_mixed": int(mixed.sum()), "resamples": n,
            "auroc": dict(zip(names, cells[:k])), "auroc_within": dict(zip(names, cells[k:2 * k])),
            "diff_auroc": {f"{p}-{q}": c for (p, q), c in zip(pairs, cells[2 * k:2 * k + m])},
            "diff_auroc_within": {f"{p}-{q}": c for (p, q), c in zip(pairs, cells[2 * k + m:])}}
