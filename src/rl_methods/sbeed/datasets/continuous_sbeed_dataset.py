"""Replay buffer for continuous SBEED.

Rows store `(observation, action, reward, next_observation, done)`. The solver
samples contiguous rows to build multi-step targets, so appending transitions in
environment order matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import torch


@dataclass
class ContinuousSBEEDDataset:
    """
    Transition buffer for continuous-observation, continuous-action SBEED.

    This mirrors `DiscreteSBEEDDataset`, but keeps observations and actions as
    float matrices with shapes `(n, obs_dim)` and `(n, action_dim)`.
    """

    X: torch.Tensor
    A: torch.Tensor
    R: torch.Tensor
    X_next: torch.Tensor
    D: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.X = torch.as_tensor(self.X, dtype=torch.float32)
        self.A = torch.as_tensor(self.A, dtype=torch.float32)
        self.R = torch.as_tensor(self.R, dtype=torch.float32).reshape(-1)
        self.X_next = torch.as_tensor(self.X_next, dtype=torch.float32)
        if self.X.ndim == 1:
            self.X = self.X.reshape(1, -1) if self.X.numel() else self.X.reshape(0, 0)
        if self.A.ndim == 1:
            self.A = self.A.reshape(1, -1) if self.A.numel() else self.A.reshape(0, 0)
        if self.X_next.ndim == 1:
            self.X_next = self.X_next.reshape(1, -1) if self.X_next.numel() else self.X_next.reshape(0, 0)
        if self.D is None:
            self.D = torch.zeros(self.R.numel(), dtype=torch.bool, device=self.R.device)
        else:
            self.D = torch.as_tensor(self.D, dtype=torch.bool, device=self.R.device).reshape(-1)

        lengths = {self.X.shape[0], self.A.shape[0], self.R.numel(), self.X_next.shape[0], self.D.numel()}
        if len(lengths) != 1:
            raise ValueError("observations, actions, rewards, next_observations, and done must have the same length")
        self.n = int(self.R.numel())

    @classmethod
    def empty(
        cls,
        obs_dim: int,
        action_dim: int,
        device: Optional[Union[torch.device, str]] = None,
    ) -> "ContinuousSBEEDDataset":
        obs_dim = int(obs_dim)
        action_dim = int(action_dim)
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        device = torch.device("cpu" if device is None else device)
        return cls(
            X=torch.empty((0, obs_dim), dtype=torch.float32, device=device),
            A=torch.empty((0, action_dim), dtype=torch.float32, device=device),
            R=torch.empty(0, dtype=torch.float32, device=device),
            X_next=torch.empty((0, obs_dim), dtype=torch.float32, device=device),
            D=torch.empty(0, dtype=torch.bool, device=device),
        )

    def append(
        self,
        observation: Union[torch.Tensor, Sequence[float]],
        action: Union[torch.Tensor, Sequence[float]],
        reward: float,
        next_observation: Union[torch.Tensor, Sequence[float]],
        done: bool = False,
    ) -> None:
        self.append_many([observation], [action], [reward], [next_observation], [done])

    def append_fifo(
        self,
        observation: Union[torch.Tensor, Sequence[float]],
        action: Union[torch.Tensor, Sequence[float]],
        reward: float,
        next_observation: Union[torch.Tensor, Sequence[float]],
        capacity: int,
        done: bool = False,
    ) -> None:
        self.append_many([observation], [action], [reward], [next_observation], [done], capacity=capacity)

    def append_many(
        self,
        observations: Sequence[Union[torch.Tensor, Sequence[float]]],
        actions: Sequence[Union[torch.Tensor, Sequence[float]]],
        rewards: Sequence[float],
        next_observations: Sequence[Union[torch.Tensor, Sequence[float]]],
        dones: Optional[Sequence[bool]] = None,
        capacity: Optional[int] = None,
    ) -> None:
        batch_n = len(observations)
        if dones is None:
            dones = [False] * batch_n
        lengths = {batch_n, len(actions), len(rewards), len(next_observations), len(dones)}
        if len(lengths) != 1:
            raise ValueError("observations, actions, rewards, next_observations, and dones must have the same length")
        if batch_n == 0:
            return
        if capacity is not None:
            capacity = int(capacity)
            if capacity <= 0:
                raise ValueError("capacity must be positive")

        device = self.X.device
        X_new = torch.stack(
            [torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(-1) for obs in observations],
            dim=0,
        )
        A_new = torch.stack(
            [torch.as_tensor(action, dtype=torch.float32, device=device).reshape(-1) for action in actions],
            dim=0,
        )
        X_next_new = torch.stack(
            [
                torch.as_tensor(next_obs, dtype=torch.float32, device=device).reshape(-1)
                for next_obs in next_observations
            ],
            dim=0,
        )
        if X_new.ndim != 2 or A_new.ndim != 2 or X_next_new.ndim != 2:
            raise ValueError("observations, actions, and next_observations must be 2D batches")

        self.X = torch.cat([self.X, X_new], dim=0)
        self.A = torch.cat([self.A, A_new], dim=0)
        self.R = torch.cat([self.R, torch.as_tensor(rewards, dtype=torch.float32, device=device).reshape(-1)])
        self.X_next = torch.cat([self.X_next, X_next_new], dim=0)
        self.D = torch.cat([self.D, torch.as_tensor(dones, dtype=torch.bool, device=device).reshape(-1)])

        if capacity is not None and self.R.numel() > capacity:
            self.X = self.X[-capacity:]
            self.A = self.A[-capacity:]
            self.R = self.R[-capacity:]
            self.X_next = self.X_next[-capacity:]
            self.D = self.D[-capacity:]
        self.n = int(self.R.numel())

    def extend(self, other: "ContinuousSBEEDDataset") -> None:
        other = other.to(self.X.device)
        self.X = torch.cat([self.X, other.X], dim=0)
        self.A = torch.cat([self.A, other.A], dim=0)
        self.R = torch.cat([self.R, other.R])
        self.X_next = torch.cat([self.X_next, other.X_next], dim=0)
        self.D = torch.cat([self.D, other.D])
        self.n = int(self.R.numel())

    def to(self, device: Union[torch.device, str]) -> "ContinuousSBEEDDataset":
        device = torch.device(device)
        return ContinuousSBEEDDataset(
            X=self.X.to(device),
            A=self.A.to(device),
            R=self.R.to(device),
            X_next=self.X_next.to(device),
            D=self.D.to(device),
        )

    def validate(self, obs_dim: int, action_dim: int) -> None:
        if self.X.ndim != 2 or self.X.shape[1] != int(obs_dim):
            raise ValueError(f"dataset observations must have shape (n, {int(obs_dim)})")
        if self.X_next.ndim != 2 or self.X_next.shape[1] != int(obs_dim):
            raise ValueError(f"dataset next_observations must have shape (n, {int(obs_dim)})")
        if self.A.ndim != 2 or self.A.shape[1] != int(action_dim):
            raise ValueError(f"dataset actions must have shape (n, {int(action_dim)})")
        if not torch.isfinite(self.X).all() or not torch.isfinite(self.X_next).all():
            raise ValueError("dataset observations must be finite")
        if not torch.isfinite(self.A).all():
            raise ValueError("dataset actions must be finite")
        if not torch.isfinite(self.R).all():
            raise ValueError("dataset rewards must be finite")

    def summary(self) -> Dict[str, Any]:
        if self.n == 0:
            return {
                "n": 0,
                "reward_mean": None,
                "done_count": 0,
            }
        return {
            "n": self.n,
            "reward_mean": float(self.R.mean().item()),
            "done_count": int(self.D.sum().item()),
        }

    @property
    def done(self) -> torch.Tensor:
        return self.D
