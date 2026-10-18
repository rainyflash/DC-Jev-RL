#!/bin/bash
# Part II: GRPO on Qwen3.5-4B-Base with a LoRA policy (dcjev.llm.grpo); rollouts come from a vLLM server.
#   bash scripts/run_part2.sh rollout [port] [gpu_fraction]        # vLLM rollout server for the policy (vLLM environment)
#   bash scripts/run_part2.sh genjudge_server [port] [gpu_fraction]  # vLLM server for Qwen3.5-4B (generative-judge baseline)
#   bash scripts/run_part2.sh study                                 # judge study (rollout server on 6200, then 6201)
#   bash scripts/run_part2.sh run <cond> <seed> [grpo args]         # one GRPO run -> runs/part2/<cond>/s<seed>
source /content/work/env.sh
cd /content/work/code/DC-Jev-RL
R=/content/work/runs/part2
VPY=/content/work/vllm-env/bin/python
export PATH=/content/work/vllm-env/bin:$PATH   # vLLM compiles kernels with ninja from its environment
case "$1" in
  rollout) $VPY -m dcjev.llm.rollout_server --port ${2:-6200} --gpu_memory_utilization ${3:-0.30} ;;
  genjudge_server) $VPY -m dcjev.llm.rollout_server --port ${2:-6201} --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
                     --no_lora --gpu_memory_utilization ${3:-0.25} --max_model_len 8192 ;;
  study) $PY -m dcjev.llm.judge_study gen --dir $R/judge_study --port 6200 && $PY -m dcjev.llm.judge_study score --dir $R/judge_study ;;
  run) cond=$2; seed=$3; shift 3; $PY -m dcjev.llm.grpo --cond $cond --seed $seed --out $R/$cond/s$seed "$@" ;;
  *) echo "usage: $0 rollout|genjudge_server|study|run"; exit 1 ;;
esac
