"""Small, dependency-free metrics shared by the trainers and the analysis scripts."""
from __future__ import annotations

import numpy as np


def ranks(x):
    """1-based ranks with ties averaged."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return r


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
