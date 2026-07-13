"""Replay buffer for finite-state SBEED solvers.

Rows store `(state, action, reward, next_state, done)`. The staged one-step
solvers only need individual rows, while the multi-step solvers sample
contiguous row fragments and stop at `done=True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import torch


@dataclass
class DiscreteSBEEDDataset:
    """
    Transition buffer D for discrete SBEED experiments.

    In the paper, SBEED is written as an online algorithm with an experience
    replay buffer D. This class is the code representation of that buffer.
    The buffer is populated incrementally by the solver's online collection
    loop.

    Expected transition fields are state, action, reward, next_state, and a
    terminal done flag. The done flag defaults to False for older four-field
    callers.
    """

    X: torch.Tensor
    A: torch.Tensor
    R: torch.Tensor
    X_next: torch.Tensor
    D: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.X = torch.as_tensor(self.X, dtype=torch.int64).reshape(-1)
        self.A = torch.as_tensor(self.A, dtype=torch.int64).reshape(-1)
        self.R = torch.as_tensor(self.R, dtype=torch.float64).reshape(-1)
        self.X_next = torch.as_tensor(self.X_next, dtype=torch.int64).reshape(-1)
        if self.D is None:
            self.D = torch.zeros_like(self.X, dtype=torch.bool)
        else:
            self.D = torch.as_tensor(self.D, dtype=torch.bool, device=self.X.device).reshape(-1)

        lengths = {self.X.numel(), self.A.numel(), self.R.numel(), self.X_next.numel(), self.D.numel()}
        if len(lengths) != 1:
            raise ValueError("state, action, reward, next_state, and done must have the same length")
        self.n = int(self.X.numel())

    @classmethod
    def empty(cls, device: Optional[Union[torch.device, str]] = None) -> "DiscreteSBEEDDataset":
        device = torch.device("cpu" if device is None else device)
        return cls(
            X=torch.empty(0, dtype=torch.int64, device=device),
            A=torch.empty(0, dtype=torch.int64, device=device),
            R=torch.empty(0, dtype=torch.float64, device=device),
            X_next=torch.empty(0, dtype=torch.int64, device=device),
            D=torch.empty(0, dtype=torch.bool, device=device),
        )

    def append(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool = False,
    ) -> None:
        self.append_many(
            [state],
            [action],
            [reward],
            [next_state],
            [done],
        )

    def append_fifo(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        capacity: int,
        done: bool = False,
    ) -> None:
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.append_many(
            [state],
            [action],
            [reward],
            [next_state],
            [done],
            capacity=capacity,
        )

    def append_many(
        self,
        states: Sequence[int],
        actions: Sequence[int],
        rewards: Sequence[float],
        next_states: Sequence[int],
        dones: Optional[Sequence[bool]] = None,
        capacity: Optional[int] = None,
    ) -> None:
        batch_n = len(states)
        if dones is None:
            dones = [False] * batch_n
        lengths = {batch_n, len(actions), len(rewards), len(next_states), len(dones)}
        if len(lengths) != 1:
            raise ValueError("states, actions, rewards, next_states, and dones must have the same length")
        if batch_n == 0:
            return
        if capacity is not None:
            capacity = int(capacity)
            if capacity <= 0:
                raise ValueError("capacity must be positive")

        device = self.X.device
        self.X = torch.cat([self.X, torch.as_tensor(states, dtype=torch.int64, device=device).reshape(-1)])
        self.A = torch.cat([self.A, torch.as_tensor(actions, dtype=torch.int64, device=device).reshape(-1)])
        self.R = torch.cat([self.R, torch.as_tensor(rewards, dtype=torch.float64, device=device).reshape(-1)])
        self.X_next = torch.cat([
            self.X_next,
            torch.as_tensor(next_states, dtype=torch.int64, device=device).reshape(-1),
        ])
        self.D = torch.cat([self.D, torch.as_tensor(dones, dtype=torch.bool, device=device).reshape(-1)])

        if capacity is not None and self.X.numel() > capacity:
            # FIFO replay keeps the newest transitions while preserving order
            # for any later multi-step fragment sampling.
            self.X = self.X[-capacity:]
            self.A = self.A[-capacity:]
            self.R = self.R[-capacity:]
            self.X_next = self.X_next[-capacity:]
            self.D = self.D[-capacity:]
        self.n = int(self.X.numel())

    def extend(self, other: "DiscreteSBEEDDataset") -> None:
        other = other.to(self.X.device)
        self.X = torch.cat([self.X, other.X])
        self.A = torch.cat([self.A, other.A])
        self.R = torch.cat([self.R, other.R])
        self.X_next = torch.cat([self.X_next, other.X_next])
        self.D = torch.cat([self.D, other.D])
        self.n = int(self.X.numel())

    def to(self, device: Union[torch.device, str]) -> "DiscreteSBEEDDataset":
        device = torch.device(device)
        return DiscreteSBEEDDataset(
            X=self.X.to(device),
            A=self.A.to(device),
            R=self.R.to(device),
            X_next=self.X_next.to(device),
            D=self.D.to(device),
        )

    def validate(self, n_states: int, n_actions: int) -> None:
        if torch.any((self.X < 0) | (self.X >= n_states)):
            raise ValueError("dataset states must be in [0, n_states)")
        if torch.any((self.X_next < 0) | (self.X_next >= n_states)):
            raise ValueError("dataset next_states must be in [0, n_states)")
        if torch.any((self.A < 0) | (self.A >= n_actions)):
            raise ValueError("dataset actions must be in [0, n_actions)")

    def summary(self) -> Dict[str, Any]:
        if self.n == 0:
            return {
                "n": 0,
                "unique_states": torch.empty(0, dtype=torch.int64, device=self.X.device),
                "unique_actions": torch.empty(0, dtype=torch.int64, device=self.A.device),
                "reward_mean": None,
                "done_count": 0,
            }
        return {
            "n": self.n,
            "unique_states": torch.unique(self.X),
            "unique_actions": torch.unique(self.A),
            "reward_mean": float(self.R.mean().item()),
            "done_count": int(self.D.sum().item()),
        }

    @property
    def done(self) -> torch.Tensor:
        return self.D
