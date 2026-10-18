# DC-Jev-RL, version 4

*[中文版见下方](#dc-jev-rl第-4-版)*

> **Correction.** The results of versions 1.0–3.0 of the five papers in this repository were not produced by the
> experiments they describe, and they are withdrawn. [CORRECTION.md](CORRECTION.md) lists what was wrong. Version 4
> redoes the work as measured experiments.

## Status (18 October 2026)

Version 4 is being built on this branch (`v4-honest-rebuild`). The experiments are running on three Colab GPUs. Every
table and number in the papers is generated from the logs under `experiments/` as the runs finish, and a paper has no
results section for runs that have not finished. The branch will be merged into `master` when all five papers are
complete.

| Part | Question | Finished | Running |
|---|---|---|---|
| I | Do Kev-4B's judgments of short motion windows make a useful PPO reward on five MuJoCo tasks, compared with the environment reward and with an autoregressive judge of the same size? | all 165 runs; offline agreement of both judges with the environment reward; behaviour of the final policies; the paper, in English and Chinese | – |
| II | What does sharing one Qwen3.5-4B backbone between the GRPO policy and Kev-4B save, and does Kev-4B's judgment add anything to the verifier reward on MATH? | memory and speed of the shared backbone; offline judge study; learning-rate sweep (1e-5, chosen on validation) | 12 GRPO runs |
| III-A | How much reward hacking is there when multi-turn code repair is rewarded by visible tests and graded by held-out ones, and does a judge reduce it? | offline judge study on KodCode, MBPP+ and HumanEval+ | 6 GRPO runs |
| III-B | When the verifier labels only 2 of 8 samples per prompt, can a frozen, a recalibrated or a scratch judge head fill in the rest? | – | 8 GRPO runs |
| III-C | Can a judge choose the prefixes to branch from in tree rollouts? | offline prefix study | 4 GRPO runs |

## Layout

- `dcjev/`: the code. Judges (Kev-4B and an autoregressive Qwen3.5-4B judge), PPO, the GRPO trainers, the offline
  studies and the analysis.
- `scripts/`: environment setup (`setup_colab.sh`, with the exact package lists in `env/`), one run script per part,
  the GRPO job lists, and `build_papers.sh`.
- `experiments/`: logs, prompts, configurations and results of every run. `RUNTIME_LOG.md` records the runtime resets
  and what they cost.
- `papers/`: LaTeX sources of the five papers, in English and Chinese. Their numbers and tables come from
  `dcjev/analysis/paper_numbers.py`, never by hand.
- `withdrawn/`: the files of versions 1.0–3.0, kept for the record.

## Reproducing

1. `bash scripts/setup_colab.sh` on a Linux machine with a recent NVIDIA GPU: both Python environments at the pinned
   versions, the pinned model snapshots and the data.
2. `bash scripts/run_validation.sh`: the checks in `experiments/validation/`.
3. Part I: `bash scripts/run_part1.sh baselines`, then `server` and `judges`.
4. Parts II and III: a rollout server (`bash scripts/run_part2.sh rollout`), then the job lists made by
   `bash scripts/make_jobs.sh 1e-5`, run with `python -m dcjev.queue`.
5. Papers: `python -m dcjev.analysis.paper_numbers <part> --out papers/<paper>`, then `bash scripts/build_papers.sh papers`.

---

# DC-Jev-RL，第 4 版

> **更正。** 本仓库五篇论文第 1.0–3.0 版的结果并非由文中所述实验产生，现已全部撤回，错误之处见
> [CORRECTION.md](CORRECTION.md)。第 4 版以实际测量的实验重做这项工作。

## 进度（2026 年 10 月 18 日）

第 4 版正在本分支（`v4-honest-rebuild`）上构建，实验在三块 Colab GPU 上运行。论文中的每个表格和数字都随 run 的完成由
`experiments/` 下的日志生成；尚未完成的 run 在论文中没有对应的结果章节。五篇论文全部完成后，本分支将合并到 `master`。

| 部分 | 问题 | 已完成 | 运行中 |
|---|---|---|---|
| 第一部分 | 在五个 MuJoCo 任务上，Kev-4B 对短时运动窗口的判断能否作为有用的 PPO 奖励？与环境奖励和同等规模的自回归评判模型相比如何？ | 全部 165 个 run；两种评判模型与环境奖励的离线一致性；最终策略的行为分析；论文（英文和中文） | – |
| 第二部分 | GRPO 策略与 Kev-4B 共用一个 Qwen3.5-4B 骨干能节省什么？在 MATH 上，Kev-4B 的判断能否为验证器奖励带来增益？ | 共享骨干的显存与速度；离线评判研究；学习率扫描（在验证集上选定 1e-5） | 12 个 GRPO run |
| 第三部分 A | 多轮代码修复以可见测试为奖励、以留出测试评分时，奖励投机有多严重？评判模型能否减少它？ | 在 KodCode、MBPP+ 和 HumanEval+ 上的离线评判研究 | 6 个 GRPO run |
| 第三部分 B | 验证器只为每个提示的 8 个样本中的 2 个打标签时，冻结、重新校准或从头训练的评判输出头能否补全其余标签？ | – | 8 个 GRPO run |
| 第三部分 C | 在树状 rollout 中，评判模型能否选择从哪些前缀分支？ | 离线前缀研究 | 4 个 GRPO run |

## 目录

- `dcjev/`：代码。评判模型（Kev-4B 和自回归的 Qwen3.5-4B 评判模型）、PPO、GRPO 训练器、离线研究和分析。
- `scripts/`：环境搭建（`setup_colab.sh`，确切的包版本列表在 `env/` 中）、各部分的运行脚本、GRPO 任务列表和 `build_papers.sh`。
- `experiments/`：每个 run 的日志、提示、配置和结果。`RUNTIME_LOG.md` 记录了运行时重置及其损失。
- `papers/`：五篇论文的 LaTeX 源文件（英文和中文）。其中的数字和表格全部由 `dcjev/analysis/paper_numbers.py` 生成，没有手填。
- `withdrawn/`：第 1.0–3.0 版的文件，留档备查。

## 复现

1. 在装有较新 NVIDIA GPU 的 Linux 机器上运行 `bash scripts/setup_colab.sh`：按固定版本安装两个 Python 环境，下载固定版本的模型快照和数据。
2. `bash scripts/run_validation.sh`：`experiments/validation/` 中的各项检查。
3. 第一部分：`bash scripts/run_part1.sh baselines`，然后是 `server` 和 `judges`。
4. 第二、三部分：先启动 rollout 服务（`bash scripts/run_part2.sh rollout`），再用 `python -m dcjev.queue` 运行 `bash scripts/make_jobs.sh 1e-5` 生成的任务列表。
5. 论文：`python -m dcjev.analysis.paper_numbers <part> --out papers/<paper>`，然后 `bash scripts/build_papers.sh papers`。
