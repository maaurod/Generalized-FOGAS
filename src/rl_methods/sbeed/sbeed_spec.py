"""Small finite-MDP metadata object used by the staged SBEED solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class DiscreteMDPSpec:
    """
    Standard finite-discrete MDP metadata for data-based algorithms.

    This is intentionally lighter than a model-based MDP class. It stores the
    state/action dimensions, discount, optional initial state, and the feature
    maps needed by SBEED.

    It does not store transition probabilities or rewards. Online solvers
    receive those through `transition_fn` and `reward_fn` in `run`.
    """

    n_states: int
    n_actions: int
    gamma: float
    value_features: Callable[[int], torch.Tensor]
    rho_features: Callable[[int, int], torch.Tensor]
    x0: Optional[int] = None
    policy_features: Optional[Callable[[int], torch.Tensor]] = None

    def __post_init__(self) -> None:
        self.n_states = int(self.n_states)
        self.n_actions = int(self.n_actions)
        self.gamma = float(self.gamma)
        if self.policy_features is None:
            self.policy_features = self.value_features
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError("gamma must be in [0, 1)")
        if self.x0 is not None:
            self.x0 = int(self.x0)
            if self.x0 < 0 or self.x0 >= self.n_states:
                raise ValueError("x0 must be in [0, n_states)")

        value_dim = len(self.value_features(0))
        rho_dim = len(self.rho_features(0, 0))
        policy_dim = len(self.policy_features(0))
        self.value_dim = int(value_dim)
        self.rho_dim = int(rho_dim)
        self.policy_dim = int(policy_dim)


DiscreteMDP = DiscreteMDPSpec

# Backward-compatible names from the first SBEED draft.
SBEEDDiscreteSpec = DiscreteMDPSpec
DiscreteSBEEDMDP = DiscreteMDPSpec
