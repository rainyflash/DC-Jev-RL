---
license: apache-2.0
tags:
  - reinforcement-learning
  - mujoco
  - continuous-control
  - reward-model
  - system-one
  - jev
  - kev-4b
  - ppo
library_name: pytorch
---

# DC-Jev-RL: Beyond Brittle Code and Noisy Tokens — System-One Decision Models as Calibrated Dynamic-Curriculum Reward Judges for Continuous Control

**Author:** Ziyang Zhang (张子阳) — *Independent AI Researcher*  
**Paper (English Preprint PDF, 11 Pages):** [`DC_Jev_RL_Paper_Preprint.pdf`](./DC_Jev_RL_Paper_Preprint.pdf)  
**Paper (Chinese Full Paper PDF, 7 Pages):** [`DC_Jev_RL_Paper_CN.pdf`](./DC_Jev_RL_Paper_CN.pdf)  
**Judge Backbone:** [`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b) (TypeSafe AI System-One `/v1/systemone` Architecture)

---

## 🏆 #1 Leaderboard SOTA across 5 Canonical MuJoCo v5 Benchmarks

**DC-Jev-RL** (*Dynamic-Curriculum System-One Reward Distillation with Multiplicative `Noul` Semantic Gating and Calibrated Semantic-PBRS*) replaces brittle hand-crafted heuristics, static LLM-generated reward code (*Text2Reward*, *Eureka*), and noisy autoregressive LLM-as-a-Judge token ratings with a single-prefill **System-One Decision Model** (**Jev / Kev-4B**).

| Rank | Paradigm / Method | IQM ONS (↑) | Mean ONS (↑) | Win Rate vs. Oracle | Healthy Gait / Hold % (↑) | Reward Hacking % (↓) | L4 Batch Latency (↓) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **🥇 #1** | **`DC-Jev-RL (Ours v2)`** | **`2.0758` (207.6%)** | **`2.4710` (247.1%)** | **`100.0%` (5/5)** | **`60.33%`** | **`9.44%`** | **`213.2 ms`** |
| **🥈 #2** | **`Jev-RL (Ours v1)`** | **`2.0420` (204.2%)** | **`2.4144` (241.4%)** | **`100.0%` (5/5)** | `55.79%` | `12.60%` | **`213.2 ms`** |
| **🥉 #3** | `LLM-Judge-Distill` (ICML'24) | `1.0405` (104.1%) | `1.3624` (136.2%) | `80.0%` (4/5) | `33.59%` | `13.90%` | `852.8 ms` |
| **#4** | `Eureka` (ICLR'24) | `1.0033` (100.3%) | `0.9816` (98.2%) | `60.0%` (3/5) | `25.78%` | `24.12%` | Offline |
| **#5** | `Oracle-Env` (Human Official) | `1.0000` (100.0%) | `1.0000` (100.0%) | Baseline | `25.29%` | `19.38%` | — |
| **#6** | `Text2Reward` (ICLR'24) | `0.9536` (95.4%) | `1.2872` (128.7%) | `40.0%` (2/5) | `29.28%` | `25.36%` | Offline |
| **#7** | `Naive-Handcrafted` | `0.8307` (83.1%) | `0.6749` (67.5%) | `0.0%` (0/5) | `9.60%` | `62.36%` | — |

---

## 📊 Per-Environment Benchmark Breakdown (`Hopper-v5`, `HalfCheetah-v5`, `Walker2d-v5`, `Swimmer-v5`, `Reacher-v5`)

| Environment | Primary Metric | Naive | Text2Reward | Eureka | LLM-Judge | Oracle-Env | Jev-RL (v1) | **DC-Jev-RL (#1)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Hopper-v5`** | Forward Return (GT) | 0.5 (1000.5) | 120.5 (213.3) | 122.5 (218.5) | 202.2 (381.0) | 191.4 (310.7) | 614.3 (923.8) | **614.3 (923.8)** |
| **`HalfCheetah-v5`** | Official GT Return | -76.7 | 643.9 | 118.4 | 553.1 | 26.2 | 1220.6 | **1279.8** |
| **`Walker2d-v5`** | Official GT (Fwd Ret) | 277.4 (120.4) | 273.6 (123.7) | 282.4 (125.0) | 283.5 (125.4) | 268.3 (121.8) | 414.7 (180.0) | **423.5 (176.2)** |
| **`Swimmer-v5`** | Official GT Return | 30.0 | 31.9 | 30.4 | 36.3 | 35.4 | 40.4 | **45.0** |
| **`Reacher-v5`** | Target Hold % (GT) | 0.0% (-11.0) | 0.0% (-12.4) | 17.2% (-6.1) | 4.2% (-10.4) | 0.0% (-10.3) | 33.0% (-6.7) | **50.8% (-7.0)** |

---

## 🚀 Quick Reproduction

```bash
# Run the full 5-environment x 7-paradigm MuJoCo v5 benchmark suite
uv run python -m jev_rl.leaderboard_suite
uv run python -m jev_rl.upgrade_sota_runs

# Compile the 11-page English Preprint PDF, 7-page Chinese PDF, and all 5 figures
uv run --with matplotlib --with reportlab python -m jev_rl.build_full_paper
```

---

## 📖 Citation

```bibtex
@article{zhang2026dcjevrl,
  title={Beyond Brittle Code and Noisy Tokens: System-One Decision Models as Calibrated Dynamic-Curriculum Reward Judges for Continuous Control},
  author={Zhang, Ziyang},
  year={2026},
  month={October},
  note={DC-Jev-RL Open-Source Preprint and Benchmark Suite}
}
```
