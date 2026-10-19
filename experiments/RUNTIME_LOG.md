# Compute sessions and runtime resets

Every experiment ran on Google Colab runtimes, each with one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU (96 GB),
up to three at a time. A Colab runtime can end at any time, and its disk, including the working directory
`/content/work`, is lost when it does. Results were copied off the runtimes every 30 minutes: finished runs completely,
and for PPO runs in progress their log and last checkpoint. A GRPO run that is lost is rerun from scratch. This file
records every reset and what it cost; nothing else was lost.

| when (UTC) | what ended | lost | recovered |
|---|---|---|---|
| 2026-10-17 | the single runtime of the time | the environment, the first Part I baseline runs and part of the first judge grid | all rerun from scratch on a rebuilt environment; no run from before this reset is reported |
| 2026-10-18 ~05:57 | all three runtimes (no reason shown) | the two learning-rate sweep runs in progress, at step 15 of 60 | both rerun from scratch with the same seed and code (`part2_llm/LR_SELECTION.md`) |
| 2026-10-18 ~07:30 | all three runtimes, ended by Colab for inactivity | Part I: the 24 judge-reward runs in progress, at about 350k of 1M steps. The GRPO grids, started at 07:18 (no run had reached its first evaluation) | the 24 Part I runs resumed from their checkpoint at 301,056 steps; each log records the resume and the length of the dropped episode. The GRPO grids restarted from scratch at 07:59 |

Colab counts a runtime as idle when no notebook cell is executing, and the cell that starts the connection to the
runtime returns at once. Since 07:40 UTC on 18 October each runtime also runs a notebook cell that only sleeps, so it is
no longer idle.

## Environment after the 07:30 reset

The three new runtimes were rebuilt with `scripts/setup_colab.sh` and checked against session 1 before any run:

- kev's environment: identical package versions (`uv pip freeze`).
- vLLM environment: identical except `filelock` 4.2.2 instead of 4.2.1 (a newer release had appeared). It was pinned back
  to 4.2.1 on all three runtimes before any run, so the freezes are identical. `setup_colab.sh` now installs both
  environments against the frozen lists in `scripts/env/`.
- Code data: the KodCode training set was rebuilt with 8170, 8169 and 8171 problems (its filter depends on timing; see
  `validation/README.md`). It was replaced on all three runtimes by session 1's build (sha256
  `edd7d4a4b0d6316e8d152f504be23ea62174a173760548104062cc31bf49271b`, 8170 problems), the one every Part III-A run uses.
  The evaluation sets (KodCode validation, MBPP+, HumanEval+) came out byte-identical on the three runtimes and hold
  the same problems (298, 376 and 161) as the offline judge study's.

## Runtimes replaced for the second round

A Colab runtime lasts at most about 24 hours. When a runtime's queue had finished, every file of its runs except the
GRPO checkpoints and LoRA weights was copied off and checked (it exists locally with the same size), and the runtime
was deleted and replaced by a new one on the same notebook. No run was in progress on a runtime when it was deleted.
Each new runtime was set up with `scripts/setup_colab.sh` and checked before its queue started: the GPU model, the math
data's sha256 sums against session 1's, both package lists against `scripts/env/`, and the base-model revision the
rollout server loads.

| runtime | old runtime deleted (UTC) | new runtime checked | queue started | checks |
|---|---|---|---|---|
| session 1 | 2026-10-19 ~02:25 | 02:33 to 02:41 | 02:42 | RTX PRO 6000 (96 GB); math data and both package lists identical; Qwen3.5-4B-Base at `1001bb4d` |
| session 2 | 2026-10-19 ~04:12 | 04:14 to 04:22 | 04:22 | RTX PRO 6000 (96 GB); math data and both package lists identical; Qwen3.5-4B-Base at `1001bb4d` |
| session 3 | 2026-10-19 ~06:35 | 06:37 to 06:45 | 06:46 | RTX PRO 6000 (96 GB); math data and both package lists identical; Qwen3.5-4B-Base at `1001bb4d` |
