#!/bin/bash
# Part I grid: 5 MuJoCo v5 tasks x reward conditions x seeds, 1M environment steps each. dcjev.control.grid skips runs
# whose final.json exists, so the script can be re-run after an interruption.
#   bash scripts/run_part1.sh baselines        # env, env_window (no judge; CPU only)
#   bash scripts/run_part1.sh server           # judge server: Kev-4B (bf16 merged, fused) + Qwen3.5-4B (fused), port 6100
#   bash scripts/run_part1.sh judges           # judge-reward conditions (needs the server)
source /content/work/env.sh
cd /content/work/code/DC-Jev-RL
R=/content/work/runs/part1
case "$1" in
  baselines) $PY -m dcjev.control.grid --root $R --envs all --rewards env,env_window --seeds 1-5 --workers ${WORKERS:-16} ;;
  server)    $PY -m dcjev.judges.server --port 6100 --judges kev,ar ;;
  judges)    $PY -m dcjev.control.grid --root $R --envs all --rewards kev_score,kev_gated,kev_distill,ar_gated --seeds 1-5 --workers ${WORKERS:-24} &&
             $PY -m dcjev.control.grid --root $R --envs all --rewards ar_sampled --seeds 1-3 --workers 15 ;;
  *) echo "usage: $0 baselines|server|judges"; exit 1 ;;
esac
