"""Autoregressive LLM judge baseline (Qwen/Qwen3.5-4B, the post-trained chat model of the same size as Kev-4B's base).

The judge is prompted with the same state and question text Kev receives and asked to answer with one option (a digit for
a score question, yes/no for a yes/no question), with thinking disabled. Instead of sampling a token we read the
next-token distribution once, restricted to the answer tokens:
- expected:  sum_k k * p(k) / (K - 1), the strongest single-pass form of an LLM judge (G-Eval style);
- sampled:   one level drawn from the renormalised answer distribution, the discrete score a sampled judge would give;
- valid_mass: the probability the model put on any answer token at all (format compliance).
So this baseline costs one prefill pass per question, the same compute class as Kev.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ARJudge:
    def __init__(self, name="Qwen/Qwen3.5-4B", revision=None, device="cuda", dtype=torch.bfloat16, token_budget=32768, seed=0,
                 fused=False):
        """fused: rewrite the backbone with kev.fused_qwen35's kernels (the same rewrite kev.serve applies to Kev's own
        merged Qwen3.5 backbone); agreement with the unfused model is measured in experiments/part1_control/ar_fused.json."""
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        lm = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype)
        self.backbone, self.lm_head = lm.model.to(device).eval(), lm.lm_head.to(device).eval()
        if fused:
            from kev.fused_qwen35 import fuse
            fuse(self.backbone)
        self.device, self.token_budget = device, token_budget
        self.rng = np.random.default_rng(seed)
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)
        self.digit_ids = [enc(str(i))[0] for i in range(10)]
        self.yes_ids, self.no_ids = [enc("yes")[0], enc("Yes")[0]], [enc("no")[0], enc("No")[0]]
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.seconds = 0.0

    def prompt_ids(self, messages):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def next_token_probs(self, prompts):
        """prompts: list of token id lists -> [N, V] float32 next-token probabilities (on CPU)."""
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [None] * len(prompts)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and (j - i + 1) * len(prompts[order[j]]) <= self.token_budget:
                j += 1
            chunk = [prompts[k] for k in order[i:j]]
            L = max(len(p) for p in chunk)
            ids = torch.full((len(chunk), L), self.pad, device=self.device)
            att = torch.zeros((len(chunk), L), dtype=torch.long, device=self.device)
            for n, p in enumerate(chunk):   # right padding: every read position precedes the pads
                ids[n, : len(p)] = torch.tensor(p, device=self.device); att[n, : len(p)] = 1
            h = self.backbone(input_ids=ids, attention_mask=att).last_hidden_state
            last = torch.tensor([len(p) - 1 for p in chunk], device=self.device)
            logits = self.lm_head(h[torch.arange(len(chunk), device=self.device), last]).float()
            P = torch.softmax(logits, -1).cpu()
            for n, k in enumerate(order[i:j]):
                out[k] = P[n]
            i = j
        return out

    def score(self, message_lists, kind, levels=5):
        """kind 'score' -> dicts with expected, sampled, valid_mass, probs; kind 'noul' -> p_yes, valid_mass."""
        t0 = time.perf_counter()
        probs = self.next_token_probs([self.prompt_ids(m) for m in message_lists])
        res = []
        for p in probs:
            if kind == "score":
                q = p[self.digit_ids[:levels]].numpy().astype(np.float64)
                mass = float(q.sum()); qn = q / mass if mass > 0 else np.full(levels, 1.0 / levels)
                res.append({"expected": float((np.arange(levels) * qn).sum() / (levels - 1)),
                            "sampled": float(self.rng.choice(levels, p=qn) / (levels - 1)),
                            "valid_mass": mass, "probs": qn.tolist()})
            else:
                y, n = float(p[self.yes_ids].sum()), float(p[self.no_ids].sum())
                res.append({"p_yes": y / (y + n) if y + n > 0 else 0.5, "valid_mass": y + n})
        self.seconds += time.perf_counter() - t0
        return res
