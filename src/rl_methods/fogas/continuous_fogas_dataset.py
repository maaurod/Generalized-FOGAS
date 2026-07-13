"""Continuous-observation dataset adapter for FOGAS-style solvers.

Rows store ``(observation, action, reward, next_observation, done)``.  The
observation columns follow a prefix convention by default:
``obs_0, obs_1, ...`` and ``next_obs_0, next_obs_1, ...``.

This loader is algorithm-specific: it keeps continuous offline data in the
tensor layout expected by the continuous generalized FOGAS solver. In the
experiments it is used for Mountain Car style datasets, while the original
tabular FOGASSolver uses ``FOGASDataset``.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import torch

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None


class ContinuousFOGASDataset:
    """CSV-backed continuous-observation transition dataset for FOGAS."""

    _ACTION_TYPES = {"discrete", "continuous"}

    def __init__(
        self,
        csv_path,
        action_type,
        obs_prefix="obs_",
        next_obs_prefix="next_obs_",
        action_prefix="action_",
        discrete_action_column="action",
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        verbose=False,
    ):
        if pd is None:
            raise ImportError("ContinuousFOGASDataset requires pandas")

        self.csv_path = csv_path
        self.action_type = self._canonical_action_type(action_type)
        self.df = pd.read_csv(csv_path)
        if self.df.empty:
            raise ValueError("Dataset CSV is empty")
        if "reward" not in self.df.columns:
            raise ValueError("Missing required column: reward")

        self.obs_columns = self._prefix_columns(self.df.columns, obs_prefix, obs_dim, "observation")
        self.next_obs_columns = self._prefix_columns(
            self.df.columns,
            next_obs_prefix,
            obs_dim,
            "next observation",
        )
        if len(self.obs_columns) != len(self.next_obs_columns):
            raise ValueError(
                "observation and next observation dimensions must match: "
                f"{len(self.obs_columns)} vs {len(self.next_obs_columns)}"
            )
        self.obs_dim = len(self.obs_columns)

        X = self.df[self.obs_columns].to_numpy(dtype=np.float32)
        X_next = self.df[self.next_obs_columns].to_numpy(dtype=np.float32)
        R = self.df["reward"].to_numpy(dtype=np.float32)

        if self.action_type == "discrete":
            if discrete_action_column not in self.df.columns:
                raise ValueError(f"Missing required column: {discrete_action_column}")
            A = self.df[discrete_action_column].to_numpy(dtype=np.int64)
            self.action_dim = 1
            if np.any(A < 0):
                raise ValueError("discrete actions must be nonnegative integer ids")
        else:
            self.action_columns = self._prefix_columns(
                self.df.columns,
                action_prefix,
                action_dim,
                "action",
            )
            self.action_dim = len(self.action_columns)
            A = self.df[self.action_columns].to_numpy(dtype=np.float32)

        if "done" in self.df.columns:
            D = self.df["done"].astype(bool).to_numpy()
        else:
            D = np.zeros(len(self.df), dtype=bool)

        self.X = torch.as_tensor(X, dtype=torch.float32)
        if self.action_type == "discrete":
            self.A = torch.as_tensor(A, dtype=torch.int64)
        else:
            self.A = torch.as_tensor(A, dtype=torch.float32)
        self.R = torch.as_tensor(R, dtype=torch.float32)
        self.X_next = torch.as_tensor(X_next, dtype=torch.float32)
        self.D = torch.as_tensor(D, dtype=torch.bool)
        self.n = int(self.R.numel())

        self.validate()
        if verbose:
            self.print_stats()

    @classmethod
    def _canonical_action_type(cls, action_type):
        action_type = str(action_type).lower()
        if action_type not in cls._ACTION_TYPES:
            raise ValueError("action_type must be either 'discrete' or 'continuous'")
        return action_type

    @staticmethod
    def _prefix_columns(columns, prefix, dim, label):
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        indexed = []
        for col in columns:
            match = pattern.match(str(col))
            if match:
                indexed.append((int(match.group(1)), col))
        indexed.sort(key=lambda item: item[0])

        if dim is not None:
            dim = int(dim)
            if dim <= 0:
                raise ValueError(f"{label} dimension must be positive")
            expected = [f"{prefix}{i}" for i in range(dim)]
            missing = [col for col in expected if col not in columns]
            if missing:
                raise ValueError(f"Missing required {label} column(s): {missing}")
            return expected

        if not indexed:
            raise ValueError(f"No {label} columns found with prefix {prefix!r}")

        expected_indices = list(range(indexed[-1][0] + 1))
        actual_indices = [idx for idx, _ in indexed]
        if actual_indices != expected_indices:
            raise ValueError(
                f"{label} columns must be contiguous from {prefix}0; got indices {actual_indices}"
            )
        return [col for _, col in indexed]

    def validate(self):
        if self.X.ndim != 2 or self.X.shape[1] != self.obs_dim:
            raise ValueError(f"observations must have shape (n, {self.obs_dim})")
        if self.X_next.ndim != 2 or self.X_next.shape[1] != self.obs_dim:
            raise ValueError(f"next observations must have shape (n, {self.obs_dim})")
        if self.action_type == "discrete":
            if self.A.ndim != 1:
                raise ValueError("discrete actions must have shape (n,)")
            if torch.any(self.A < 0):
                raise ValueError("discrete actions must be nonnegative")
        elif self.A.ndim != 2 or self.A.shape[1] != self.action_dim:
            raise ValueError(f"continuous actions must have shape (n, {self.action_dim})")

        if self.R.ndim != 1:
            raise ValueError("rewards must have shape (n,)")
        lengths = {self.X.shape[0], self.X_next.shape[0], self.A.shape[0], self.R.numel(), self.D.numel()}
        if len(lengths) != 1:
            raise ValueError("observations, actions, rewards, next observations, and done must align")
        if not torch.isfinite(self.X).all() or not torch.isfinite(self.X_next).all():
            raise ValueError("observations must be finite")
        if not torch.isfinite(self.R).all():
            raise ValueError("rewards must be finite")
        if self.action_type == "continuous" and not torch.isfinite(self.A).all():
            raise ValueError("continuous actions must be finite")

    def print_stats(self):
        print(f"Loaded continuous FOGAS dataset {self.csv_path} with {self.n} transitions.")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action type: {self.action_type}")
        print(f"Action dim: {self.action_dim}")
        print(f"Reward mean: {float(self.R.mean().item()):.6f}")
        print(f"Done count: {int(self.D.sum().item())}")

    @property
    def done(self):
        return self.D
