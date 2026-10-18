"""Main entrypoint to run the full Jev-RL benchmark suite on MuJoCo Hopper-v5 and HalfCheetah-v5."""
from __future__ import annotations

import json
import os
import time

from jev_rl.judge import JevSystemOneJudge
from jev_rl.ppo_experiment import run_ppo_experiment


def main() -> None:
    print("=" * 72)
    print("🚀 Loading Jev-Class System-1 Model (Kev-4B / Qwen3.5-4B) on GPU...")
    print("=" * 72)
    t0 = time.time()
    judge = JevSystemOneJudge(checkpoint_id="jaredpalmer/kev-4b", device="cuda")
    print(f"✅ Loaded Jev System-1 Judge in {time.time() - t0:.2f}s")

    envs = ["Hopper-v5", "HalfCheetah-v5"]
    modes = ["naive", "llm_judge", "jev_ablation", "jev_rl", "oracle"]
    all_results = {}

    for env_name in envs:
        all_results[env_name] = {}
        print("\n" + "=" * 72)
        print(f"🏁 Running MuJoCo Benchmark: {env_name}")
        print("=" * 72)
        for mode in modes:
            print(f"\n---> Training mode: [{mode}] on {env_name}...")
            res = run_ppo_experiment(
                env_name=env_name,
                reward_mode=mode,
                judge=judge,
                total_timesteps=245760,
                num_envs=16,
                num_steps=512,
                seed=42,
                device="cuda",
            )
            all_results[env_name][mode] = res
            with open("/content/jev_rl_results.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

    print("\n" + "=" * 72)
    print("🎉 All experiments completed! Summary:")
    for env_name in envs:
        print(f"\n### {env_name} ###")
        for mode, r in all_results[env_name].items():
            fin = r["final"]
            print(
                f"  {mode:14s} | Official Return: {fin['gt_return_mean']:7.1f} ± {fin['gt_return_std']:5.1f} | "
                f"vx: {fin['mean_vx']:+5.2f} m/s | Healthy Gait: {fin['healthy_gait_ratio']*100:5.1f}% | "
                f"Judge r={r['judge_gt_correlation']:+.3f} | Latency={r['judge_latency_ms']:.1f}ms"
            )


if __name__ == "__main__":
    main()
