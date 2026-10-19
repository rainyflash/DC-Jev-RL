"""Jev-Class System-1 Judge (`Kev-4B`) and Autoregressive LLM-Judge Baseline for RL."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from kev.api import SystemOneRequest, to_answers, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import admit


@dataclass
class JevEvalResult:
    """Structured output from Jev-class System-1 evaluation on a trajectory window."""
    reward: float
    confidence: float
    forward_score: float
    posture_score: float
    energy_score: float
    healthy_gait_noul: float
    latency_ms: float


HOPPER_QUESTIONS = {
    "forward_progress": {
        "type": "score",
        "instructions": "How effectively is the hopper robot progressing forward along the x-axis?",
        "criteria": [
            "Moving backward, stationary, or barely creeping forward (vx <= 0.15 m/s)",
            "Moderate forward hopping speed (vx between 0.2 and 1.3 m/s)",
            "Fast, sustained forward hopping velocity (vx > 1.5 m/s)",
        ],
    },
    "posture_stability": {
        "type": "score",
        "instructions": "How stable and upright is the hopper robot's torso posture?",
        "criteria": [
            "Fallen over, collapsed torso height (z < 0.75 m), or extreme pitch tilt (|angle| > 0.2 rad)",
            "Somewhat wobbly or tilting noticeably (|angle| between 0.1 and 0.2 rad) but still standing",
            "Upright torso height (z >= 1.1 m) and balanced pitch angle (|angle| < 0.1 rad)",
        ],
    },
    "energy_efficiency": {
        "type": "score",
        "instructions": "How smooth and energy-efficient is the motor torque actuation?",
        "criteria": [
            "Excessive, erratic, or saturated motor torques wasting energy (mean_sq_torque > 0.5)",
            "Moderate control effort with occasional torque spikes (mean_sq_torque 0.15 to 0.5)",
            "Smooth, low-effort motor actuation (mean_sq_torque < 0.15)",
        ],
    },
    "healthy_gait": {
        "type": "noul",
        "instructions": "Is the robot actively moving forward with a healthy, balanced hopping gait rather than falling or standing frozen in place?",
    },
}

HALFCHEETAH_QUESTIONS = {
    "forward_progress": {
        "type": "score",
        "instructions": "How fast is the quadruped cheetah robot running forward along the x-axis?",
        "criteria": [
            "Moving backward, stationary, or slow shuffling (vx <= 0.5 m/s)",
            "Moderate trotting speed forward (vx between 0.6 and 2.5 m/s)",
            "Fast, vigorous sprinting velocity forward (vx > 3.0 m/s)",
        ],
    },
    "posture_stability": {
        "type": "score",
        "instructions": "How natural and upright is the cheetah torso pitch without flipping upside down?",
        "criteria": [
            "Flipped upside down or extreme torso pitch (|pitch| > 0.8 rad)",
            "Noticeable pitching or bounding oscillation (|pitch| between 0.35 and 0.8 rad)",
            "Stable, level running posture with controlled torso pitch (|pitch| < 0.35 rad)",
        ],
    },
    "energy_efficiency": {
        "type": "score",
        "instructions": "How coordinated and efficient is the 6-joint actuator torque usage?",
        "criteria": [
            "Violent, saturated joint thrashing (mean_sq_torque > 0.6)",
            "Moderate actuator effort during running (mean_sq_torque 0.2 to 0.6)",
            "Efficient, rhythmic stride actuation (mean_sq_torque < 0.2)",
        ],
    },
    "healthy_gait": {
        "type": "noul",
        "instructions": "Is the cheetah running forward upright with a coordinated stride rather than flipping upside-down or twitching in place?",
    },
}


def summarize_trajectory_window(
    env_name: str,
    obs_window: np.ndarray,
    act_window: np.ndarray,
    vx_window: np.ndarray,
    terminated: bool,
) -> str:
    """Convert a numeric RL trajectory segment (H steps) into a structured semantic state description for Jev."""
    mean_vx = float(np.mean(vx_window))
    min_vx = float(np.min(vx_window))
    max_vx = float(np.max(vx_window))
    ctrl_sq = float(np.mean(np.square(act_window)))

    if "Hopper" in env_name:
        # Hopper-v5 obs[0] is torso z-height, obs[1] is torso angle
        z_mean = float(np.mean(obs_window[:, 0]))
        z_min = float(np.min(obs_window[:, 0]))
        ang_mean = float(np.mean(obs_window[:, 1]))
        ang_max = float(np.max(np.abs(obs_window[:, 1])))
        motion_tag = (
            "moving backward"
            if mean_vx < -0.05
            else ("standing nearly frozen in place (no forward progress)" if mean_vx < 0.15 else (
                "hopping forward moderately" if mean_vx < 1.5 else "hopping forward rapidly"
            ))
        )
        posture_tag = (
            "collapsed or fallen over"
            if (terminated or z_min < 0.72 or ang_max > 0.22)
            else ("tilting noticeably" if ang_max > 0.10 else "upright and balanced")
        )
        return (
            f"MuJoCo Hopper-v5 Trajectory Segment ({len(vx_window)} steps):\n"
            f"- Forward velocity (vx): mean={mean_vx:+.2f} m/s, min={min_vx:+.2f} m/s, max={max_vx:+.2f} m/s ({motion_tag})\n"
            f"- Torso height (z): mean={z_mean:.2f} m, min={z_min:.2f} m\n"
            f"- Torso pitch angle: mean={ang_mean:+.2f} rad, max_abs={ang_max:.2f} rad ({posture_tag})\n"
            f"- Motor control effort (mean_sq_torque): {ctrl_sq:.3f}\n"
            f"- Episode terminated (fell over): {bool(terminated)}"
        )
    else:
        # HalfCheetah-v5 obs[0] is z-coord, obs[1] is torso pitch angle (ry)
        z_mean = float(np.mean(obs_window[:, 0]))
        pitch_mean = float(np.mean(obs_window[:, 1]))
        pitch_max = float(np.max(np.abs(obs_window[:, 1])))
        motion_tag = (
            "moving backward or stuck"
            if mean_vx <= 0.2
            else ("trotting forward moderately" if mean_vx < 2.8 else "sprinting forward rapidly")
        )
        posture_tag = (
            "flipped upside-down or extreme torso tilt"
            if pitch_max > 0.85
            else ("moderate bounding pitch" if pitch_max > 0.35 else "level upright running posture")
        )
        return (
            f"MuJoCo HalfCheetah-v5 Trajectory Segment ({len(vx_window)} steps):\n"
            f"- Forward velocity (vx): mean={mean_vx:+.2f} m/s, min={min_vx:+.2f} m/s, max={max_vx:+.2f} m/s ({motion_tag})\n"
            f"- Torso height offset (z): mean={z_mean:+.2f} m\n"
            f"- Torso pitch angle: mean={pitch_mean:+.2f} rad, max_abs={pitch_max:.2f} rad ({posture_tag})\n"
            f"- Motor control effort (mean_sq_torque): {ctrl_sq:.3f}\n"
            f"- Upside-down or unstable posture: {bool(pitch_max > 0.85)}"
        )


class JevSystemOneJudge:
    """High-throughput batched Jev-class System-1 Judge on GPU (`Kev-4B` / `Kev-0.8B`)."""

    def __init__(self, checkpoint_id: str = "jaredpalmer/kev-4b", device: str = "cuda"):
        self.device = device
        opts = LoadOptions(dtype=torch.bfloat16, cuda_graphs=False)
        self.ck = Checkpoint(checkpoint_id)
        self.tok, self.model = self.ck.load(device, opts)
        self._cache: Dict[Tuple[str, str], JevEvalResult] = {}

    def evaluate_batch(self, env_name: str, state_descriptions: List[str]) -> List[JevEvalResult]:
        """Evaluate a batch of trajectory state descriptions in parallel non-autoregressive passes."""
        questions = HOPPER_QUESTIONS if "Hopper" in env_name else HALFCHEETAH_QUESTIONS
        results: List[Optional[JevEvalResult]] = [None] * len(state_descriptions)
        uncached_indices: List[int] = []
        uncached_encs = []
        uncached_metas = []

        for idx, st in enumerate(state_descriptions):
            key = (env_name, st)
            if key in self._cache:
                results[idx] = self._cache[key]
            else:
                req = SystemOneRequest(state=st, model="jev-latest", questions=questions)
                rec, meta = to_record(req)
                enc = admit(self.model, self.tok, rec, truncate=True)
                uncached_indices.append(idx)
                uncached_encs.append(enc)
                uncached_metas.append(meta)

        if uncached_encs:
            t0 = time.perf_counter()
            all_probs = []
            for enc in uncached_encs:
                with torch.inference_mode():
                    ps, _ = self.model.probs_batch([enc], [None], [False])
                all_probs.append([q.tolist() for q in ps[0]])

            total_ms = (time.perf_counter() - t0) * 1000.0
            per_item_ms = total_ms / max(1, len(uncached_encs))

            for idx, probs, meta in zip(uncached_indices, all_probs, uncached_metas):
                ans = to_answers(probs, meta)
                fp = float(ans["forward_progress"]["score"]) / 2.0
                ps_val = float(ans["posture_stability"]["score"]) / 2.0
                ee = float(ans["energy_efficiency"]["score"]) / 2.0
                hg = float(ans["healthy_gait"]["noul"])

                c_fp = float(ans["forward_progress"]["confidence"])
                c_ps = float(ans["posture_stability"]["confidence"])
                c_ee = float(ans["energy_efficiency"]["confidence"])
                c_hg = float(abs(2.0 * hg - 1.0))
                mean_conf = 0.35 * c_fp + 0.25 * c_ps + 0.15 * c_ee + 0.25 * c_hg

                # Synergistic Noul-gated composite score:
                composite = (0.55 * fp) + hg * (0.30 * ps_val + 0.15 * ee)
                res_obj = JevEvalResult(
                    reward=float(composite),
                    confidence=float(np.clip(mean_conf, 0.1, 1.0)),
                    forward_score=fp,
                    posture_score=ps_val,
                    energy_score=ee,
                    healthy_gait_noul=hg,
                    latency_ms=per_item_ms,
                )
                results[idx] = res_obj
                self._cache[(env_name, state_descriptions[idx])] = res_obj

        return [r for r in results if r is not None]

    def evaluate_autoregressive_baseline(
        self, env_name: str, state_descriptions: List[str]
    ) -> List[JevEvalResult]:
        """Traditional Monolithic LLM-as-a-Judge baseline:
        Evaluates a single holistic question and samples a discrete integer level (0..3)
        rather than Jev's parallel multi-dimensional calibrated continuous expectation + Noul gating."""
        monolithic_q = {
            "overall_score": {
                "type": "score",
                "instructions": f"Rate the overall quality of this {env_name} robot trajectory from 0 (terrible) to 3 (excellent).",
                "criteria": [
                    "0: Bad trajectory",
                    "1: Mediocre trajectory",
                    "2: Good trajectory",
                    "3: Excellent trajectory",
                ],
            }
        }
        rng = np.random.default_rng(123)
        encs = []
        metas = []
        for st in state_descriptions:
            req = SystemOneRequest(state=st, model="jev-latest", questions=monolithic_q)
            rec, meta = to_record(req)
            enc = admit(self.model, self.tok, rec, truncate=True)
            encs.append(enc)
            metas.append(meta)

        # Measure true 4-token sequential autoregressive generation latency on first sample
        t_lat = time.perf_counter()
        with torch.inference_mode():
            for _ in range(4):
                _ = self.model.probs_batch([encs[0]], [None], [False])
        measured_ar_ms = (time.perf_counter() - t_lat) * 1000.0

        all_probs = []
        for enc in encs:
            with torch.inference_mode():
                ps, _ = self.model.probs_batch([enc], [None], [False])
            all_probs.append([q.tolist() for q in ps[0]])

        results: List[JevEvalResult] = []
        for probs, meta in zip(all_probs, metas):
            ans = to_answers(probs, meta)["overall_score"]
            probs_dict = ans["probabilities"]
            p_vec = np.array([float(probs_dict[str(i)]) for i in range(4)], dtype=np.float64)
            p_vec = p_vec / np.sum(p_vec)
            # Standard LLM-as-a-Judge samples a discrete integer token instead of calibrated expectation
            sampled_level = int(rng.choice(4, p=p_vec))
            norm_score = float(sampled_level / 3.0)
            results.append(
                JevEvalResult(
                    reward=norm_score,
                    confidence=0.5,
                    forward_score=norm_score,
                    posture_score=norm_score,
                    energy_score=norm_score,
                    healthy_gait_noul=norm_score,
                    latency_ms=measured_ar_ms,
                )
            )
        return results

