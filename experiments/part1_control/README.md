# Part I: PPO on MuJoCo with judge rewards (raw runs)

`runs/<task>/<condition>/s<seed>/` holds one CleanRL-style PPO run (`dcjev/control/ppo.py`, 1M environment steps):

- `final.json`: the final evaluation (20 episodes of the deterministic policy, with the undiscounted environment return), plus the run's wall-clock time and judge usage;
- `log.jsonl.gz`: every record the run logged. That is the config, each training episode, each PPO update, the periodic evaluations (10 episodes every 100k steps), and a 2% sample of 32-step windows (the behaviour summary the judges read, with the environment reward over that window);
- `policy.pt`: the final policy weights.

`results/` is produced from `runs/` by

    python -m dcjev.analysis.part1 --root experiments/part1_control/runs --out experiments/part1_control/results

Status: the `env` and `env_window` baselines are complete (5 tasks x 5 seeds). The judge-reward conditions are running and are added when they finish; `results/` is regenerated then.
