"""Policy LoRA hand-off from the HF/PEFT trainer to vLLM.

The released Qwen3.5-4B-Base checkpoint is a vision-language checkpoint; vLLM serves it with
Qwen3_5ForConditionalGeneration, whose language model lives under `model.language_model.` in Hugging Face naming. The
trainer runs the text-only Qwen3_5ForCausalLM (as kev does), so PEFT names its LoRA tensors `base_model.model.model.layers.*`.
vLLM silently ignores LoRA tensors whose names do not match a module, so the names are rewritten here, and
dcjev/llm/check_vllm_lora.py verifies that a random adapter changes vLLM's output and matches HF's log-probabilities.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

SRC, DST = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."


def to_vllm(src_dir: str | Path, dst_dir: str | Path) -> Path:
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    t = load_file(src / "adapter_model.safetensors")
    bad = [k for k in t if not k.startswith(SRC)]
    if bad:
        raise ValueError(f"unexpected LoRA tensor names, e.g. {bad[:2]}")
    save_file({DST + k[len(SRC):]: v.contiguous() for k, v in t.items()}, dst / "adapter_model.safetensors")
    cfg = json.loads((src / "adapter_config.json").read_text())
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    return dst
