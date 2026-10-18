"""Confidence-Calibrated Online Reward Network Distillation from Jev System-1 Judge."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DistilledRewardNet(nn.Module):
    """Fast neural reward model R_phi(obs, act, vx) distilled online from Jev trajectory evaluations."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Extract key physical invariants (z, pitch, |pitch|, vx, ctrl_sq) + full obs
        in_dim = obs_dim + 5
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor, vx: torch.Tensor) -> torch.Tensor:
        if vx.ndim == obs.ndim - 1:
            vx = vx.unsqueeze(-1)
        z = obs[..., 0:1]
        pitch = obs[..., 1:2]
        abs_pitch = torch.abs(pitch)
        ctrl_sq = torch.mean(torch.square(act), dim=-1, keepdim=True)
        # Clamp raw obs to prevent outliers during random exploration
        obs_clamped = torch.clamp(obs, -10.0, 10.0) * 0.2
        feats = torch.cat([z, pitch, abs_pitch, vx, ctrl_sq, obs_clamped], dim=-1)
        return 3.0 * torch.sigmoid(self.net(feats)).squeeze(-1) + 0.3


class TrajectoryRewardBuffer:
    """Stores trajectory segments (obs_win, act_win, vx_win) paired with Jev scores and calibrated confidences."""

    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.obs_wins: List[np.ndarray] = []
        self.act_wins: List[np.ndarray] = []
        self.vx_wins: List[np.ndarray] = []
        self.targets: List[float] = []
        self.confidences: List[float] = []
        self.ptr = 0

    def add_batch(
        self,
        obs_list: List[np.ndarray],
        act_list: List[np.ndarray],
        vx_list: List[np.ndarray],
        targets: List[float],
        confidences: List[float],
    ) -> None:
        for o, a, v, t, c in zip(obs_list, act_list, vx_list, targets, confidences):
            if len(self.targets) < self.capacity:
                self.obs_wins.append(o.astype(np.float32))
                self.act_wins.append(a.astype(np.float32))
                self.vx_wins.append(v.astype(np.float32))
                self.targets.append(float(t))
                self.confidences.append(float(c))
            else:
                idx = self.ptr % self.capacity
                self.obs_wins[idx] = o.astype(np.float32)
                self.act_wins[idx] = a.astype(np.float32)
                self.vx_wins[idx] = v.astype(np.float32)
                self.targets[idx] = float(t)
                self.confidences[idx] = float(c)
                self.ptr += 1

    def __len__(self) -> int:
        return len(self.targets)


class ConfidenceWeightedDistiller:
    """Trains DistilledRewardNet using Jev's calibrated confidence weights."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        device: str = "cuda",
        lr: float = 2e-3,
        use_confidence_weighting: bool = True,
    ):
        self.device = device
        self.use_confidence_weighting = use_confidence_weighting
        self.reward_net = DistilledRewardNet(obs_dim, act_dim).to(device)
        self.optimizer = optim.AdamW(self.reward_net.parameters(), lr=lr, weight_decay=1e-4)
        self.buffer = TrajectoryRewardBuffer(capacity=2048)

    def train_epochs(self, batch_size: int = 32, epochs: int = 12) -> float:
        if len(self.buffer) < 8:
            return 0.0
        n = len(self.buffer)
        obs_arr = np.stack(self.buffer.obs_wins, axis=0)   # (N, H, obs_dim)
        act_arr = np.stack(self.buffer.act_wins, axis=0)   # (N, H, act_dim)
        vx_arr = np.stack(self.buffer.vx_wins, axis=0)     # (N, H)
        tgt_arr = np.array(self.buffer.targets, dtype=np.float32) * 2.6 + 0.4
        conf_arr = np.array(self.buffer.confidences, dtype=np.float32)

        obs_t = torch.tensor(obs_arr, device=self.device)
        act_t = torch.tensor(act_arr, device=self.device)
        vx_t = torch.tensor(vx_arr, device=self.device)
        tgt_t = torch.tensor(tgt_arr, device=self.device)
        conf_t = torch.tensor(conf_arr, device=self.device)

        total_loss = 0.0
        steps = 0
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                pred_steps = self.reward_net(obs_t[idx], act_t[idx], vx_t[idx])  # (B, H)
                pred_win_mean = pred_steps.mean(dim=-1)  # (B,)
                # Also encourage step-level smoothness within homogeneous windows
                step_err = (pred_steps - tgt_t[idx].unsqueeze(-1)) ** 2
                sq_err = 0.7 * ((pred_win_mean - tgt_t[idx]) ** 2) + 0.3 * step_err.mean(dim=-1)
                if self.use_confidence_weighting:
                    weights = (conf_t[idx] ** 1.5) / (torch.mean(conf_t[idx] ** 1.5) + 1e-6)
                    loss = (weights * sq_err).mean()
                else:
                    loss = sq_err.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.reward_net.parameters(), 1.0)
                self.optimizer.step()
                total_loss += float(loss.item())
                steps += 1
        return total_loss / max(1, steps)

    @torch.inference_mode()
    def predict_reward(self, obs: torch.Tensor, act: torch.Tensor, vx: torch.Tensor) -> torch.Tensor:
        return self.reward_net(obs, act, vx)
