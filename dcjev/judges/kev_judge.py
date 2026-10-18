"""Batched Kev-4B scoring for reward computation.

Everything model-specific is kev's own code: the request schema and rendering (kev.api), the encoder and the serving
admission rule (kev.model.admit), the row form (kev.model.rows_of: every question is one causal row = the state tokens
followed by that question's branch), the backbone, the LoRA adapter and the pointer head with the calibration temperature
stored in the checkpoint (kev.checkpoint). This module only changes how rows are batched: the rows of many requests are
sorted by length and run through the backbone in large right-padded chunks. Right padding sits after every position that
is read out, so it cannot change those hidden states (causal attention, and the Gated DeltaNet recurrence only runs
forward). kev's own training path (DecisionModel.forward_rows_batch) uses the same row form.

Parity against kev's reference serving path (DecisionModel.probs) is measured by dcjev/judges/check_kev_parity.py and the
numbers are reported in experiments/part1_control/kev_parity.json.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kev.api import SystemOneRequest, to_answers, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import admit, rows_of

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass
class JudgeStats:
    requests: int = 0
    rows: int = 0
    tokens: int = 0          # real (unpadded) tokens through the backbone
    padded_tokens: int = 0   # tokens including right padding
    seconds: float = 0.0


class KevJudge:
    """Loads a Kev checkpoint once and scores batches of System One requests."""

    def __init__(self, run: str = "jaredpalmer/kev-4b", device: str = "cuda", dtype: str = "bf16", merge: bool = True,
                 token_budget: int = 32768, shared_prefix: bool = False, fused: bool = False):
        """shared_prefix: run each state once and its question branches on what it leaves behind (kev.shared_prefix,
        kev's own training path for the hybrid backbones) instead of one full row per question.
        fused: kev.fused_qwen35 kernels (kev.serve's CUDA default; bf16 merged backbones only)."""
        self.run, self.device, self.dtype_name, self.token_budget = run, device, dtype, token_budget
        self.shared_prefix = shared_prefix
        self.ck = Checkpoint(run)
        self.tok, self.model = self.ck.load(device, LoadOptions(dtype=DTYPES[dtype], merge=merge, cuda_graphs=False, fused=fused))
        self.temperature = float(self.model.head.temperature)
        self.stats = JudgeStats()

    @classmethod
    def from_parts(cls, tok, model, device="cuda", token_budget=32768, shared_prefix=True, run="shared-backbone"):
        """A KevJudge over an existing model-like object (dcjev.llm.shared.KevView: a shared backbone with Kev's adapter
        active and Kev's pointer head) instead of a checkpoint loaded on its own."""
        j = cls.__new__(cls)
        j.run, j.device, j.dtype_name, j.token_budget, j.shared_prefix = run, device, "bf16", token_budget, shared_prefix
        j.ck, j.tok, j.model = None, tok, model
        j.temperature = float(model.head.temperature)
        j.stats = JudgeStats()
        return j

    def encode(self, request):
        req = request if isinstance(request, SystemOneRequest) else SystemOneRequest(**request)
        rec, meta = to_record(req)
        return admit(self.model, self.tok, rec, truncate=False), meta

    @torch.no_grad()
    def probs(self, requests):
        """-> (probs, metas): probs[i][q] is the probability list of question q of request i (option order of kev.api)."""
        if self.shared_prefix:
            return self._probs_shared(requests)
        t0 = time.perf_counter()
        encs, metas = zip(*(self.encode(r) for r in requests)) if requests else ((), ())
        rows = []   # (ids, pos, request index, question index, <decide> offset, option offsets)
        for b, e in enumerate(encs):
            S, Sp, brs = rows_of(e)
            for q, r in enumerate(brs):
                rows.append((S + r["ids"], Sp + r["pos"], b, q, len(S) + r["decide"], [len(S) + o for o in r["opts"]]))
        out = [[None] * len(m) for m in metas]
        order = sorted(range(len(rows)), key=lambda k: len(rows[k][0]))
        i = 0
        while i < len(order):
            j = i + 1   # rows are sorted by length, so the last row of a chunk is its longest
            while j < len(order) and (j - i + 1) * len(rows[order[j]][0]) <= self.token_budget:
                j += 1
            chunk = [rows[k] for k in order[i:j]]
            ids, pos, att = self.model._pad_rows([(c[0], c[1]) for c in chunk])
            h = self.model.lm(input_ids=ids, position_ids=pos, attention_mask=att).last_hidden_state
            dec_r = torch.arange(len(chunk), device=self.device)
            dec_p = torch.tensor([c[4] for c in chunk], device=self.device)
            owner_l = [n for n, c in enumerate(chunk) for _ in c[5]]
            slot_l = [s for c in chunk for s in range(len(c[5]))]
            opt_p = torch.tensor([o for c in chunk for o in c[5]], device=self.device)
            owner = torch.tensor(owner_l, device=self.device)
            z = self.model.head.many(h[dec_r, dec_p].float(), h[owner, opt_p].float(), owner)   # temperature applied in eval mode
            kmax = max(len(c[5]) for c in chunk)
            Z = torch.full((len(chunk), kmax), float("-inf"), device=self.device)
            Z[owner, torch.tensor(slot_l, device=self.device)] = z
            P = F.softmax(Z, -1).cpu()
            for n, c in enumerate(chunk):
                out[c[2]][c[3]] = P[n, : len(c[5])].tolist()
            self.stats.rows += len(chunk)
            self.stats.tokens += int(att.sum())
            self.stats.padded_tokens += att.numel()
            i = j
        self.stats.requests += len(encs)
        self.stats.seconds += time.perf_counter() - t0
        return out, list(metas)

    @torch.no_grad()
    def _probs_shared(self, requests):
        """Shared-prefix form: kev.shared_prefix.branch_hidden over chunks of records (states left-padded, branches
        right-padded), then the pointer head on each branch's <decide> and option positions."""
        from kev.shared_prefix import branch_hidden
        t0 = time.perf_counter()
        encs, metas = zip(*(self.encode(r) for r in requests)) if requests else ((), ())
        splits = [rows_of(e) for e in encs]
        out = [None] * len(encs)
        order = sorted(range(len(encs)), key=lambda k: len(splits[k][0]))
        i = 0
        while i < len(order):
            j = i + 1
            def cost(a, b):
                recs = [splits[k] for k in order[a:b]]
                nb = sum(len(r) for _, _, r in recs)
                lb = max(len(x["ids"]) for _, _, r in recs for x in r)
                return (b - a) * len(recs[-1][0]) + nb * lb
            while j < len(order) and cost(i, j + 1) <= self.token_budget:
                j += 1
            idx = order[i:j]
            hs = branch_hidden(self.model.lm, [splits[k] for k in idx], self.model.pad_id, self.device)
            for k, h_rec in zip(idx, hs):
                ps = []
                for h, r in zip(h_rec, splits[k][2]):
                    z = self.model.head(h[r["decide"]], h[torch.tensor(r["opts"], device=self.device)])
                    ps.append(F.softmax(z, -1))
                out[k] = [p.cpu().tolist() for p in ps]
            self.stats.rows += sum(len(splits[k][2]) for k in idx)
            self.stats.tokens += sum(len(splits[k][0]) + sum(len(x["ids"]) for x in splits[k][2]) for k in idx)
            i = j
        self.stats.requests += len(encs)
        self.stats.seconds += time.perf_counter() - t0
        return out, list(metas)

    def answers(self, requests):
        """TypeSafe-style answers (kev.api.to_answers): score = expected level, noul = p(yes), choice = argmax."""
        probs, metas = self.probs(requests)
        return [to_answers(p, m) for p, m in zip(probs, metas)]

    @torch.no_grad()
    def reference_probs(self, request):
        """kev's own reference path for one request (DecisionModel.probs: state pass, then the question rows on its cache)."""
        enc, _ = self.encode(request)
        return [p.tolist() for p in self.model.probs(enc)]
