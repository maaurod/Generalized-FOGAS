from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


class StateDiscretizer:
    """
    Uniform axis-aligned discretization for continuous observations.

    This is the clean-API equivalent of the old BoxStateDiscretizer. It maps
    continuous observations to integer state ids and can append one absorbing
    state when a terminal observation predicate is provided.
    """

    def __init__(
        self,
        low: Sequence[float],
        high: Sequence[float],
        bins: Sequence[int],
        terminal_obs_predicate: Optional[Callable[[np.ndarray], bool]] = None,
    ):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.bins = np.asarray(bins, dtype=np.int64)

        if self.low.shape != self.high.shape or self.low.shape != self.bins.shape:
            raise ValueError("low, high, and bins must have the same shape")
        if np.any(self.high <= self.low):
            raise ValueError("Each component of high must be strictly greater than low")
        if np.any(self.bins <= 0):
            raise ValueError("All entries in bins must be positive")

        self.dim = int(self.low.size)
        self.terminal_obs_predicate = terminal_obs_predicate

        self.bin_edges = [
            np.linspace(lo, hi, int(n_bins) + 1, dtype=np.float64)
            for lo, hi, n_bins in zip(self.low, self.high, self.bins)
        ]
        self.bin_centers = [
            0.5 * (edges[:-1] + edges[1:]) for edges in self.bin_edges
        ]
        self.bin_widths = (self.high - self.low) / self.bins

        self.core_state_count = int(np.prod(self.bins))
        self.absorbing_state_id = (
            self.core_state_count if self.terminal_obs_predicate is not None else None
        )
        self.n_states = self.core_state_count + int(self.absorbing_state_id is not None)

    def clip(self, obs: Sequence[float]) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape != self.low.shape:
            raise ValueError(f"Expected obs shape {self.low.shape}, got {obs.shape}")
        return np.clip(obs, self.low, self.high)

    def is_terminal_obs(self, obs: Sequence[float]) -> bool:
        if self.terminal_obs_predicate is None:
            return False
        return bool(self.terminal_obs_predicate(np.asarray(obs, dtype=np.float64)))

    def obs_to_multi_bin(self, obs: Sequence[float]) -> tuple[int, ...]:
        obs = self.clip(obs)
        scaled = np.floor((obs - self.low) / self.bin_widths).astype(np.int64)
        clipped = np.clip(scaled, 0, self.bins - 1)
        return tuple(int(v) for v in clipped)

    def multi_bin_to_state_id(self, multi_bin: Sequence[int]) -> int:
        return int(np.ravel_multi_index(tuple(multi_bin), self.bins))

    def state_id_to_multi_bin(self, state_id: int) -> tuple[int, ...]:
        self._validate_core_state_id(state_id)
        return tuple(int(v) for v in np.unravel_index(int(state_id), self.bins))

    def obs_to_state_id(self, obs: Sequence[float]) -> int:
        if self.is_terminal_obs(obs):
            if self.absorbing_state_id is None:
                raise ValueError("Terminal observation encountered without absorbing state")
            return int(self.absorbing_state_id)
        return self.multi_bin_to_state_id(self.obs_to_multi_bin(obs))

    def state_id_to_center_obs(self, state_id: int) -> Optional[np.ndarray]:
        state_id = int(state_id)
        if self.absorbing_state_id is not None and state_id == self.absorbing_state_id:
            return None
        multi_bin = self.state_id_to_multi_bin(state_id)
        return np.array(
            [self.bin_centers[d][idx] for d, idx in enumerate(multi_bin)],
            dtype=np.float64,
        )

    def _validate_core_state_id(self, state_id: int) -> None:
        state_id = int(state_id)
        if state_id < 0 or state_id >= self.core_state_count:
            raise ValueError(
                f"Core state_id must be in [0, {self.core_state_count - 1}], got {state_id}"
            )


class ActionDiscretizer:
    """
    Wraps a finite action set.

    Clean-API equivalent of the old DiscreteActionDiscretizer. Internal action
    ids are always 0..A-1; action_values are the values passed to the Gym env.
    """

    def __init__(
        self,
        action_values: Sequence,
        action_labels: Optional[dict[int, str]] = None,
    ):
        if len(action_values) == 0:
            raise ValueError("action_values must be non-empty")

        self._action_values = [self._normalize_action_value(v) for v in action_values]
        self.action_ids = np.arange(len(self._action_values), dtype=np.int64)
        self.n_actions = int(len(self._action_values))
        self.action_labels = (
            {int(k): str(v) for k, v in action_labels.items()}
            if action_labels is not None
            else None
        )

    def action_id_to_env_action(self, action_id: int):
        action_id = int(action_id)
        if action_id < 0 or action_id >= self.n_actions:
            raise ValueError(f"action_id must be in [0, {self.n_actions - 1}], got {action_id}")
        value = self._action_values[action_id]
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return value.item()
        return value.copy() if isinstance(value, np.ndarray) else value

    def action_id_to_label(self, action_id: int) -> str:
        action_id = int(action_id)
        if self.action_labels is not None and action_id in self.action_labels:
            return self.action_labels[action_id]
        return str(self.action_id_to_env_action(action_id))

    @staticmethod
    def _normalize_action_value(value):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr.astype(np.float64)
