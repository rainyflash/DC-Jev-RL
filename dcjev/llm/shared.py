"""Single-backbone policy and judge: Qwen3.5-4B-Base is held in GPU memory once and carries two LoRA adapters.

- adapter "default": the policy LoRA, trained by GRPO (the base weights stay frozen, so the backbone stays shared);
- adapter "kev":     Kev-4B's released LoRA (frozen), read out by Kev's pointer head with Kev's calibration temperature.

Kev-4B is a LoRA plus a pointer head on exactly this base (revision 1001bb4d), so the judge needs no second copy of the
4B backbone. Kev's adapter is applied unmerged here (merging would change the policy's base); the effect of that on
Kev's probabilities is measured by dcjev/llm/check_shared.py against a separately loaded, merged Kev-4B.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict
from peft.helpers import disable_input_dtype_casting
from transformers import AutoModelForCausalLM, AutoTokenizer

from kev.checkpoint import Checkpoint
from kev.model import DecisionModel, PointerHead, pad_id

from dcjev.judges.kev_judge import KevJudge

BASE, BASE_REV = "Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"
KEV_RUN = "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
POLICY_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                  "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]


class KevView:
    """What KevJudge reads from kev.model.DecisionModel, over the shared backbone."""
    encode = DecisionModel.encode
    _pad_rows = DecisionModel._pad_rows
    SHAPE_BUCKET = DecisionModel.SHAPE_BUCKET

    def __init__(self, lm, head, pad, device):
        self.lm, self.head, self.pad_id, self.device = lm, head, pad, device
        self.option_isolation, self.training, self.hybrid = False, False, True


def _lp_chunk(h, targets, W):
    logits = (h @ W.t()).float()
    return logits.gather(1, targets[:, None])[:, 0] - torch.logsumexp(logits, -1)


class SharedBackbone:
    def __init__(self, lora_r=32, lora_alpha=64, device="cuda", kev_run=KEV_RUN, grad_ckpt=True, lp_chunk=1024, kev_token_budget=8192):
        self.device, self.lp_chunk = device, lp_chunk
        self.tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
        lm = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REV, dtype=torch.bfloat16).to(device)
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, target_modules=POLICY_TARGETS, task_type="CAUSAL_LM")
        self.model = get_peft_model(lm, cfg)
        ck = Checkpoint(kev_run)
        if (ck.meta.base, ck.meta.base_revision) != (BASE, BASE_REV):
            raise ValueError(f"Kev checkpoint is on {ck.meta.base}@{ck.meta.base_revision}, the policy on {BASE}@{BASE_REV}")
        if ck.adapter_config().get("trainable_token_indices"):
            raise ValueError("Kev adapters with trained token embeddings are not supported on the shared backbone")
        kcfg = LoraConfig.from_pretrained(ck.path)   # Kev's own adapter config (rank, alpha, targets, scaling options)
        kcfg.task_type, kcfg.lora_dropout, kcfg.inference_mode = "CAUSAL_LM", 0.0, True
        self.model.add_adapter("kev", kcfg)
        w = load_peft_weights(ck.path, device="cpu")   # kev trained its adapter on AutoModelForCausalLM(...).model
        w = {k.replace("base_model.model.", "base_model.model.model.", 1): v for k, v in w.items()}
        res = set_peft_model_state_dict(self.model, w, adapter_name="kev")
        if res.unexpected_keys:
            raise ValueError(f"Kev adapter tensors not placed: {res.unexpected_keys[:3]}")
        loaded = sum(1 for n, _ in self.model.named_parameters() if ".kev." in n)
        if loaded != len(w):
            raise ValueError(f"Kev adapter: {len(w)} tensors in the checkpoint, {loaded} in the model")
        self.kev_tensors = loaded
        for n, p in self.model.named_parameters():   # both adapters in fp32 (peft does this for the first one), base in bf16
            if ".kev." in n: p.data = p.data.float()
        self.backbone = self.model.base_model.model.model
        self.lm_head = self.model.base_model.model.lm_head
        self.head = PointerHead(self.backbone.config.hidden_size, dp=ck.meta.head_dim).to(device)
        self.head.load_state_dict(ck.meta.head); self.head.eval(); self.head.temperature = ck.meta.temperature
        for p in self.head.parameters(): p.requires_grad_(False)
        self.kev = KevJudge.from_parts(self.tok, KevView(self.backbone, self.head, pad_id(self.tok), device), device,
                                     token_budget=kev_token_budget, shared_prefix=True)
        if grad_ckpt:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.use("default")

    # --- adapter switching ---------------------------------------------------------------------------------------
    def use(self, name):
        self.model.set_adapter(name)
        for n, p in self.model.named_parameters():   # only the policy LoRA is ever trainable
            p.requires_grad_(".default." in n and "lora_" in n)

    def policy_parameters(self):
        return [p for n, p in self.model.named_parameters() if ".default." in n and "lora_" in n]

    @torch.no_grad()
    def kev_probs(self, requests):
        self.model.eval(); self.use("kev")
        try:
            return self.kev.probs(requests)
        finally:
            self.use("default")

    @torch.no_grad()
    def kev_probs_features(self, requests, question=0):
        """kev_probs plus the pointer head's inputs for question index `question` (see KevJudge.probs_and_features)."""
        self.model.eval(); self.use("kev")
        try:
            return self.kev.probs_and_features(requests, question)
        finally:
            self.use("default")

    # --- policy log-probabilities ----------------------------------------------------------------------------------
    def token_logprobs(self, seqs):
        """seqs: list of (prompt_ids, completion_ids). -> list of 1-D tensors, the policy's log-probability of each
        completion token (with grad when grad is enabled). Sequences are right-padded into one batch. The LoRA matmuls run
        in bf16 under autocast (the base is bf16 already; the fp32 adapter weights stay the optimizer's master copy)."""
        self.use("default")
        with torch.autocast("cuda", dtype=torch.bfloat16), disable_input_dtype_casting(self.model):
            return self._token_logprobs(seqs)

    def _token_logprobs(self, seqs):
        L = max(len(p) + len(c) for p, c in seqs)
        ids = torch.full((len(seqs), L), self.tok.pad_token_id or 0, device=self.device)
        att = torch.zeros((len(seqs), L), dtype=torch.long, device=self.device)
        for i, (p, c) in enumerate(seqs):
            x = torch.tensor(p + c, device=self.device); ids[i, : len(x)] = x; att[i, : len(x)] = 1
        h = self.backbone(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
        rows = torch.cat([torch.full((len(c),), i, device=self.device) for i, (p, c) in enumerate(seqs)])
        pos = torch.cat([torch.arange(len(p) - 1, len(p) + len(c) - 1, device=self.device) for p, c in seqs])
        tgt = torch.cat([torch.tensor(c, device=self.device) for _, c in seqs])
        hf = h[rows, pos]
        W = self.lm_head.weight
        out = [checkpoint(_lp_chunk, hf[s:s + self.lp_chunk], tgt[s:s + self.lp_chunk], W, use_reentrant=False)
               if torch.is_grad_enabled() else _lp_chunk(hf[s:s + self.lp_chunk], tgt[s:s + self.lp_chunk], W)
               for s in range(0, len(tgt), self.lp_chunk)]
        lp = torch.cat(out)
        return list(lp.split([len(c) for _, c in seqs]))

    def save_policy(self, path):
        self.model.save_pretrained(path, selected_adapters=["default"])
