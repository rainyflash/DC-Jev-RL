#!/bin/bash
# make_jobs.sh <lr>: the job lists of the GRPO grids (dcjev.queue), seed-major so that a partial grid covers every
# condition. Per job: peak GPU memory in GiB (a trainer measured 19.7 in nvidia-smi; 29 with the AR judge loaded as
# well) and expected hours, for the queue's --deadline.
set -euo pipefail
LR=$1; D=$(dirname "$0")/jobs; mkdir -p $D
G="--prompts 16 --group 8 --steps 200 --eval_every 50 --lr $LR"
R=/content/work/runs
{
  echo "# Part II main grid, lr $LR"
  for s in 1 2 3; do
    for c in rlvr rlvr_kev_add rlvr_ar_add rlvr_kev_gate kev_only; do
      [ $s = 3 ] && [ $c != rlvr ] && [ $c != rlvr_kev_add ] && continue
      gb=20; [ $c = rlvr_ar_add ] && gb=29
      echo "$R/part2/$c/s$s $gb h=7 bash scripts/run_part2.sh run $c $s $G"
    done
  done
} > $D/part2_main.txt
{
  echo "# Part III (B label budget, A multi-turn code repair, C tree rollouts), lr $LR"
  for s in 1 2; do
    for c in rlvr_sub sub_kev sub_kev_recal; do echo "$R/part3b/$c/s$s 20 h=7 bash scripts/run_part3.sh b $c $s $G"; done
    for c in vis vis_kev hidden; do
      echo "$R/part3a/$c/s$s 20 h=5 bash scripts/run_part3.sh a $c $s --prompts 16 --group 8 --steps 100 --eval_every 50 --lr $LR"
    done
    for c in tree_kev tree_random; do echo "$R/part3c/$c/s$s 20 h=7 bash scripts/run_part3.sh c $c $s $G"; done
    echo "$R/part3b/sub_probe/s$s 20 h=7 bash scripts/run_part3.sh b sub_probe $s $G"
  done
  # A third seed of each tree condition, added on 2026-10-19 after both seeds of each had finished: with two seeds
  # each, one of them far above rlvr and one at its level, the effect of the tree shape was undecided.
  for c in tree_kev tree_random; do echo "$R/part3c/$c/s3 20 h=7 bash scripts/run_part3.sh c $c 3 $G"; done
} > $D/part3.txt
wc -l $D/part2_main.txt $D/part3.txt
