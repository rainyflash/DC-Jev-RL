"""MuJoCo environments, per-step telemetry and window summaries for the judges.

The judges never see the environment's reward. They see a short text summary of a trajectory window built from physical
quantities read from the simulator (forward velocity, torso height and pitch, fingertip-target distance, control effort),
together with a natural-language task description (experiments/part1_control/prompts/judge_prompts.json).
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

ENVS = ["Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "Swimmer-v5", "Reacher-v5"]

# Judge window length in environment steps (the judge is queried once per window; every step of the window gets its score).
WINDOW = {"Hopper-v5": 32, "Walker2d-v5": 32, "HalfCheetah-v5": 32, "Swimmer-v5": 32, "Reacher-v5": 25}

# Telemetry vector layout per step: [x_velocity, torso_z, torso_pitch, fingertip_target_distance, control_effort]
TEL = ["vx", "z", "pitch", "dist", "effort"]


class Telemetry(gym.Wrapper):
    """Adds info['tel'] (np.float64[5], see TEL) to every step, read from the raw simulator state, and keeps the
    environment's own reward untouched in the returned reward."""

    def __init__(self, env):
        super().__init__(env)
        self.env_id = env.spec.id

    def _tel(self, obs, info, action):
        u = self.env.unwrapped
        vx = float(info.get("x_velocity", 0.0))
        z = pitch = dist = float("nan")
        if self.env_id in ("Hopper-v5", "Walker2d-v5"):
            z, pitch = float(u.data.qpos[1]), float(u.data.qpos[2])          # rootz, rooty
        elif self.env_id == "HalfCheetah-v5":
            pitch = float(u.data.qpos[2])                                    # rooty
        elif self.env_id == "Reacher-v5":
            dist = float(np.linalg.norm(u.get_body_com("fingertip") - u.get_body_com("target")))
        effort = float(np.mean(np.square(np.clip(action, -1.0, 1.0)))) if action is not None else 0.0
        return np.array([vx, z, pitch, dist, effort], dtype=np.float64)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        info["tel"] = self._tel(obs, info, None)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        info["tel"] = self._tel(obs, info, action)
        return obs, rew, term, trunc, info


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env = Telemetry(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def _r(x, nd=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def window_summary(env_id: str, tel: np.ndarray, ended: bool, dt: float) -> dict:
    """Text-ready summary of one trajectory window. tel: [steps, 5] telemetry rows of the window's steps; ended: the
    episode terminated (fell / unhealthy) inside this window. Returns a JSON object rendered by kev.api.render."""
    n = len(tel)
    vx, z, pitch, dist, eff = (tel[:, i] for i in range(5))
    s = {"segment_duration_s": _r(n * dt, 2)}
    if env_id in ("Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "Swimmer-v5"):
        nd = 3 if env_id == "Swimmer-v5" else 2
        s["forward_velocity_m_per_s"] = {"mean": _r(vx.mean(), nd), "min": _r(vx.min(), nd), "max": _r(vx.max(), nd)}
    if env_id in ("Hopper-v5", "Walker2d-v5"):
        s["torso_height_m"] = {"mean": _r(z.mean()), "min": _r(z.min())}
        s["torso_tilt_deg"] = {"max_abs": _r(np.degrees(np.abs(pitch)).max(), 1)}
    if env_id == "HalfCheetah-v5":
        s["torso_pitch_deg"] = {"mean": _r(np.degrees(pitch).mean(), 1), "max_abs": _r(np.degrees(np.abs(pitch)).max(), 1)}
    if env_id == "Reacher-v5":
        s["fingertip_to_target_distance_m"] = {"start": _r(dist[0], 3), "end": _r(dist[-1], 3), "mean": _r(dist.mean(), 3)}
    s["control_effort_0_to_1"] = _r(eff.mean(), 2)
    if env_id in ("Hopper-v5", "Walker2d-v5"):
        s["fell_over"] = "yes" if ended else "no"
    return s


def window_features(env_id: str, tel: np.ndarray, ended: bool) -> np.ndarray:
    """Fixed-length numeric features of a window (the distilled reward model's input): the same quantities the text
    summary reports."""
    vx, z, pitch, dist, eff = (tel[:, i] for i in range(5))
    z = np.nan_to_num(z, nan=0.0); pitch = np.nan_to_num(pitch, nan=0.0); dist = np.nan_to_num(dist, nan=0.0)
    return np.array([vx.mean(), vx.min(), vx.max(), z.mean(), z.min(), np.abs(pitch).max(), pitch.mean(),
                     dist[0], dist[-1], dist.mean(), eff.mean(), float(ended), len(tel) / WINDOW[env_id]], dtype=np.float32)
