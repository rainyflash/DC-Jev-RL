# Validation runs (before any experiment)

Produced by `scripts/run_validation.sh` on one NVIDIA RTX PRO 6000 Blackwell Server Edition (Colab), 2026-10-18,
with `scripts/setup_colab.sh`'s pinned environment. While these ran, the CPU-only Part I baseline grid used 16 of 48 cores;
the GPU had no other job.

| file | what it checks |
|---|---|
| `kev_bench_v7_fp32/` | `kev.benchmark` on kev's frozen decision-v7 suite with Kev-4B (revision 6cfce5c2), fp32, the checkpoint's calibration temperature (2.406) |
| `kev_bench_v7_fp32_T1/` | the same at temperature 1 (`KEV_TEMPERATURE=1.0`): reproduces the published `result.json` (NLL 0.516, ECE 0.079), which was evidently computed at T = 1 |
| `kev_parity.json` | `dcjev.judges.kev_judge.KevJudge` (batched rows) against kev's reference path `DecisionModel.probs`, fp32 and bf16, and throughput |
| `kev_modes.json` | serving modes (row form, fused kernels, shared prefix) against the fp32 reference |
| `ar_fused.json` | the autoregressive judge (Qwen3.5-4B) with kev's fused kernels against the unfused model |
| `vllm_lora_check.json` | the policy LoRA hand-off to vLLM: zero adapter = base, random adapter changes outputs, HF vs vLLM log-probabilities |
| `shared_check.json` (+ parts) | Part II's shared backbone (policy LoRA + Kev LoRA on one Qwen3.5-4B-Base): Kev parity, memory, Kev throughput; see note |

Note on `shared_check.json`: its Kev throughput on the shared backbone (8.0 requests/s) was measured right after a policy
training step in the same process and is lower than a later measurement without that step (about 32 requests/s, on a
GPU shared with the Part I judge server). The systems comparison is repeated on an idle GPU before the paper reports it.
