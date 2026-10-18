#!/bin/bash
# make_jobs.sh <lr>: the job lists of the GRPO grids (dcjev.queue), seed-major so that a partial grid covers every
# condition. Free GPU memory needed to start a job: 26 GB for a trainer, 36 GB with the AR judge loaded as well.
set -euo pipefail
LR=$1; D=$(dirname "$0")/jobs; mkdir -p $D
G="--prompts 16 --group 8 --steps 200 --eval_every 50 --lr $LR"
R=/content/work/runs
{
  echo "# Part II main grid, lr $LR"
  for s in 1 2 3; do
    for c in rlvr rlvr_kev_add rlvr_kev_gate kev_only rlvr_ar_add; do
      [ $s = 3 ] && [ $c != rlvr ] && [ $c != rlvr_kev_add ] && continue
      gb=26; [ $c = rlvr_ar_add ] && gb=36
      echo "$R/part2/$c/s$s $gb bash scripts/run_part2.sh run $c $s $G"
    done
  done
} > $D/part2_main.txt
{
  echo "# Part III-B (label budget) and III-C (tree rollouts), lr $LR"
  for s in 1 2; do
    for c in sub_kev sub_kev_recal rlvr_sub sub_probe; do echo "$R/part3b/$c/s$s 26 bash scripts/run_part3.sh b $c $s $G"; done
    for c in tree_kev tree_random; do echo "$R/part3c/$c/s$s 26 bash scripts/run_part3.sh c $c $s $G"; done
  done
} > $D/part3bc.txt
{
  echo "# Part III-A (multi-turn code repair), lr $LR"
  for s in 1 2; do
    for c in vis vis_kev hidden; do echo "$R/part3a/$c/s$s 26 bash scripts/run_part3.sh a $c $s --prompts 16 --group 8 --steps 100 --eval_every 50 --lr $LR"; done
  done
} > $D/part3a.txt
wc -l $D/*.txt
