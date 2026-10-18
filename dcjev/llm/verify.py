"""Verifiable reward: the completion's last \\boxed{} answer checked against the gold answer with math-verify
(HuggingFace), run in a process pool with a per-item timeout so a pathological expression cannot stall training."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout

from dcjev.llm.data import last_boxed


def _check(args):
    pred_text, gold = args
    box = last_boxed(pred_text)
    if box is None:
        return 0
    try:
        from math_verify import parse, verify
        g = parse(f"\\boxed{{{gold}}}")
        p = parse(f"\\boxed{{{box}}}")
        return int(bool(verify(g, p)))
    except Exception:
        return int(box.strip() == str(gold).strip())


class Verifier:
    def __init__(self, workers=16, timeout=5.0):
        self.pool = ProcessPoolExecutor(max_workers=workers)
        self.timeout = timeout

    def __call__(self, completions, golds):
        futs = [self.pool.submit(_check, (c, g)) for c, g in zip(completions, golds)]
        out = []
        for f in futs:
            try:
                out.append(f.result(timeout=self.timeout))
            except (FTimeout, Exception):
                out.append(0)
        return out
