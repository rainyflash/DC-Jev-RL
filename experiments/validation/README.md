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

## Added during the experiments

| file | what it checks |
|---|---|
| `rollout_server_vs_old_reference.json` | the continuous-batching rollout server (session 2) against `rollout_ref.json` taken from the first server (one request at a time, session 1): same prompts, seeds and requests. The `sampled_lora` request is not comparable: the random test adapter is re-drawn on each machine (PEFT initialises `lora_A` before the seed is set). |
| `rollout_server_self_check.json` | the same server against a reference it produced itself one request at a time, with the requests sent again one at a time and then all at once. Every reply carries the right prompts. Samples are identical where batching allows: 100% for seeded sampling without an adapter, sent alone. With the adapter they diverge after a few tokens, with a mean log-probability difference of about 0.02 nats before divergence. A different adapter gives about 0.6 nats (vs-old-reference file), so a reply never gets the wrong adapter. |

The three Colab sessions (one GPU each, RTX PRO 6000 Blackwell Server Edition) use identical package versions (`uv pip
freeze` of both environments compared against session 1) and identical data files. The exception was the KodCode training
set: its filter (the reference solution must pass its own tests) depends on timing for a handful of problems, and
three builds kept 8170, 8169 and 8170 of 8204 problems. Session 1's build (sha256 `edd7d4a4b0d6316e8d152f504be23ea62174a173760548104062cc31bf49271b`)
was copied to the other sessions and is the one every Part III-A run uses.

## Costs measured on an idle GPU (session 3, before its GRPO runs)

Session 3's GPU ran nothing else: its rollout server was loaded but received no requests, and `gpu_before` /
`gpu_after` in each record show 0% utilisation.

| file | what it measures |
|---|---|
| `judge_bench.json` | `dcjev.judges.bench`: seconds per judgment for every judge arrangement on the same 1000 Part II requests (base-policy solutions from `../part2_llm/judge_study/`). Kev-4B as Part I's server runs it (bf16 merged, fused kernels, one row per question), as the Part II judge study ran it (shared prefix, unfused), and on the GRPO trainers' shared backbone (8192- and 32768-token batches); the AR judge with fused kernels; the reasoning judge in vLLM (300 judgments, at most 2048 thinking tokens, 25% of the GPU) |
| `shared_check_idle.mem_shared.json`, `shared_check_idle.mem_separate.json` | the memory and Kev-throughput parts of `dcjev.llm.check_shared`, repeated: 36.2 and 39.6 requests/s, against 8.0 and 39.2 in `shared_check.json`. The 8.0 was contention, not the shared backbone. During `mem_shared` an idle vLLM engine left over from the benchmark held 25 GB of GPU memory without computing; the memory figures are the process's own allocations and do not depend on it |

The seconds in `../part2_llm/judge_study/score_cost.json`, `genjudge_cost.json`, `../part3a_code/judge_study/score_cost.json`
and `../part3c_tree/prefix_study/score_cost.json` were measured while other jobs used the same GPU; the papers use
`judge_bench.json`.
