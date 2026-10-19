#!/bin/bash
# Part III: extensions of the Part II setup. Every command needs the rollout server of scripts/run_part2.sh on port 6200.
#   bash scripts/run_part3.sh b <cond> <seed> [grpo args]   # III-B label budget (rlvr_sub, sub_kev, sub_kev_recal, sub_probe) -> runs/part3b
#   bash scripts/run_part3.sh prefix_study [args]           # III-C offline: judges on unfinished solutions     -> runs/part3c/prefix_study
#   bash scripts/run_part3.sh c <cond> <seed> [grpo args]   # III-C tree GRPO (tree_kev, tree_random)           -> runs/part3c
#   bash scripts/run_part3.sh code_data                     # III-A code tasks                                  -> data/code
#   bash scripts/run_part3.sh code_study                    # III-A offline: judges on code                     -> runs/part3a/judge_study
#   bash scripts/run_part3.sh a <cond> <seed> [args]        # III-A code-repair GRPO (vis, vis_kev, hidden)     -> runs/part3a
source /content/work/env.sh
cd /content/work/code/DC-Jev-RL
R=/content/work/runs
case "$1" in
  b) cond=$2; seed=$3; shift 3; $PY -m dcjev.llm.grpo --cond $cond --seed $seed --out $R/part3b/$cond/s$seed "$@" ;;
  prefix_study) shift; for s in gen score report; do $PY -m dcjev.llm.prefix_study $s --dir $R/part3c/prefix_study "$@" || exit 1; done ;;
  c) cond=$2; seed=$3; shift 3; $PY -m dcjev.llm.grpo --cond $cond --seed $seed --out $R/part3c/$cond/s$seed "$@" ;;
  code_data) $PY -m dcjev.code.data --out /content/work/data/code ;;
  code_study) for s in gen score report; do $PY -m dcjev.code.judge_study $s --dir $R/part3a/judge_study || exit 1; done ;;
  a) cond=$2; seed=$3; shift 3; $PY -m dcjev.code.grpo_code --cond $cond --seed $seed --out $R/part3a/$cond/s$seed "$@" ;;
  *) echo "usage: $0 b|prefix_study|c|code_data|code_study|a"; exit 1 ;;
esac
