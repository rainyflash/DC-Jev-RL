# Learning-rate selection rule (fixed before the sweep results were read)

Sweep: `rlvr`, seed 1, 16 problems x 8 completions per step, 60 steps, learning rates 1e-5, 3e-5 and 1e-4
(`scripts/run_part2.sh sweep <lr>`). Each run evaluates the validation set (500 MATH training problems of levels 3-5
held out with seed 12345; greedy, 2048 tokens) at steps 0, 30 and 60.

Rule: the learning rate with the highest mean validation accuracy over steps 30 and 60 is used for every GRPO run of
Parts II and III (all reward conditions, including the baselines). If the best two are within 0.5 points of each other,
the smaller learning rate is used. No test set is evaluated in the sweep.

The 3e-5 run used the first rollout server (one request at a time); the other two used the continuous-batching server
that every later run uses (`experiments/validation/rollout_server_*.json` compares the two).
