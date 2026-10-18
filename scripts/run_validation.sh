#!/bin/bash
# Checks run before any experiment, on an otherwise idle GPU (their timings are reported). Outputs in $W/runs/validation.
#   bash scripts/run_validation.sh
source /content/work/env.sh
W=/content/work; O=$W/runs/validation; mkdir -p $O $W/tmp
cd $W/code/DC-Jev-RL
# 1. Kev-4B reproduces its published decision-v7 benchmark (kev's own benchmark, fp32 exact path), with the checkpoint's
#    calibration temperature and with T = 1 (which the published result.json appears to use)
$PY -m kev.benchmark --run jaredpalmer/kev-4b@$KEV_REV --suite $W/kev/evals/v7/decision-v7 --out $O/kev_bench_v7_fp32 --device cuda
KEV_TEMPERATURE=1.0 $PY -m kev.benchmark --run jaredpalmer/kev-4b@$KEV_REV --suite $W/kev/evals/v7/decision-v7 --out $O/kev_bench_v7_fp32_T1 --device cuda
# 2. batched Kev scoring (dcjev.judges.kev_judge) against kev's reference path; serving modes; fused AR judge
$PY -m dcjev.judges.check_kev_parity --run jaredpalmer/kev-4b@$KEV_REV --out $O/kev_parity.json
$PY -m dcjev.judges.bench_kev_modes --out $O/kev_modes.json
$PY -m dcjev.judges.check_ar --out $O/ar_fused.json
# 3. Part II: LoRA hand-off to vLLM, then the shared backbone against separate models
$PY -m dcjev.llm.check_vllm_lora make --lora $W/tmp/lora_rand --out $O/vllm_lora_check.json
PATH=$W/vllm-env/bin:$PATH $W/vllm-env/bin/python -m dcjev.llm.check_vllm_lora gen --lora $W/tmp/lora_rand --out $O/vllm_lora_check.json
$PY -m dcjev.llm.check_vllm_lora score --lora $W/tmp/lora_rand --out $O/vllm_lora_check.json
$PY -m dcjev.llm.check_shared --out $O/shared_check.json
echo VALIDATION_DONE
