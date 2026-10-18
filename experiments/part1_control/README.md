# Part I: PPO on MuJoCo with judge rewards (raw runs)

`runs/<task>/<condition>/s<seed>/` holds one CleanRL-style PPO run (`dcjev/control/ppo.py`, 1M environment steps):

- `final.json`: the final evaluation (20 episodes of the deterministic policy, with the undiscounted environment return), plus the run's wall-clock time and judge usage;
- `log.jsonl.gz`: every record the run logged. That is the config, each training episode, each PPO update, the periodic evaluations (10 episodes every 100k steps), and a 2% sample of 32-step windows (the behaviour summary the judges read, with the environment reward over that window);
- `policy.pt`: the final policy weights.

`results/` is produced from `runs/` by

    python -m dcjev.analysis.part1 --root experiments/part1_control/runs --out experiments/part1_control/results

Status: the `env` and `env_window` baselines are complete (5 tasks x 5 seeds). The judge-reward conditions are running and are added when they finish; `results/` is regenerated then.

`results/judge_agreement.json` and `results/judge_agreement_scores.jsonl.gz` (every window's scores) are produced by

    python -m dcjev.analysis.judge_agreement --root experiments/part1_control/runs --out experiments/part1_control/results/judge_agreement.json

with the judge server running (`scripts/run_part1.sh server`); `--from_scores` recomputes the statistics from the saved
scores without it. The committed file was produced on an otherwise idle GPU of a second Colab VM, with the same code,
server and prompts as the training runs. An earlier run on the training VM, without saved scores, gave the same
correlations to within 0.003, except for the sampled AR level, which is drawn at random (within 0.03).
