"""Client for dcjev.judges.server (one persistent connection per training process)."""
from __future__ import annotations

import time
from multiprocessing.connection import Client

from dcjev.judges.server import AUTHKEY


class JudgeClient:
    def __init__(self, port=6100, retries=120):
        for k in range(retries):
            try:
                self.conn = Client(("127.0.0.1", port), authkey=AUTHKEY)
                break
            except ConnectionRefusedError:
                if k == retries - 1: raise
                time.sleep(5)
        self.seconds = 0.0
        self.windows = 0

    def score(self, judge, env, summaries, questions=("progress", "healthy")):
        if not summaries:
            return {q: [] for q in questions}
        t0 = time.perf_counter()
        self.conn.send({"judge": judge, "env": env, "questions": list(questions), "summaries": summaries})
        res = self.conn.recv()
        self.seconds += time.perf_counter() - t0
        self.windows += len(summaries)
        if "error" in res:
            raise RuntimeError(f"judge server error:\n{res['error']}")
        return res

    def stats(self):
        self.conn.send({"op": "stats"})
        return self.conn.recv()
