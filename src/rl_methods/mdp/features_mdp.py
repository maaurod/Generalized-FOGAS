"""
Feature-only MDP descriptions for offline FOGAS-style methods.

This module provides the minimal finite-environment interface needed when the
full transition model is not known or not used. `FeaturesMDP` keeps states,
actions, discount factor, initial state, and a state-action feature table, while
`TabularFeatureMap` supplies one-hot features for tabular studies. It is used
by continuous/discretized experiments and by FOGAS-family solvers that only
require features and offline transitions.
"""

from __future__ import annotations

import torch


class TabularFeatureMap:
    """One-hot features over finite state-action pairs."""

    def __init__(self, n_states: int, n_actions: int, dtype=torch.float64):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.d = self.n_states * self.n_actions
        self.dtype = dtype

    def __call__(self, state_id, action_id):
        feat = torch.zeros(self.d, dtype=self.dtype)
        idx = int(state_id) * self.n_actions + int(action_id)
        feat[idx] = 1.0
        return feat


class FeaturesMDP:
    """
    Feature-only finite MDP container for offline methods such as FOGAS.

    Unlike DiscreteMDP, this class intentionally does not require P or r. It
    stores only states, actions, gamma, x0, phi, and optional omega.
    """

    def __init__(
        self,
        states,
        actions,
        gamma: float,
        x0: int,
        phi=None,
        omega=None,
        dtype=torch.float64,
    ):
        self.states = self._as_1d_tensor(states, "states")
        self.actions = self._as_1d_tensor(actions, "actions")
        self.N = int(self.states.numel())
        self.A = int(self.actions.numel())

        if self.N <= 0:
            raise ValueError("states must be non-empty")
        if self.A <= 0:
            raise ValueError("actions must be non-empty")

        self.gamma = float(gamma)
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0, 1), got {self.gamma}")

        self.x0 = int(x0)
        if self.x0 < 0 or self.x0 >= self.N:
            raise ValueError(f"x0 must be in [0, {self.N}), got {self.x0}")

        self.dtype = dtype
        self.phi = phi if phi is not None else TabularFeatureMap(self.N, self.A, dtype=dtype)

        phi0 = self._as_feature(self.phi(self.states[0].item(), self.actions[0].item()))
        self.d = int(phi0.numel())
        if self.d <= 0:
            raise ValueError("phi must return a non-empty feature vector")

        phi_rows = []
        norms = []
        for state in self.states:
            for action in self.actions:
                phi_val = self._as_feature(self.phi(state.item(), action.item()))
                if phi_val.shape != (self.d,):
                    raise ValueError(f"phi must return shape ({self.d},), got {tuple(phi_val.shape)}")
                phi_rows.append(phi_val)
                norms.append(torch.linalg.norm(phi_val))

        self.Phi = torch.vstack(phi_rows)
        self.R = torch.max(torch.stack(norms))

        self.has_omega = omega is not None
        self.omega = None
        if omega is not None:
            self.omega = torch.as_tensor(omega, dtype=torch.float64).reshape(-1).clone()
            if self.omega.shape != (self.d,):
                raise ValueError(f"omega must have shape ({self.d},), got {tuple(self.omega.shape)}")

    @staticmethod
    def _as_1d_tensor(value, name):
        if isinstance(value, torch.Tensor):
            out = value.clone()
        else:
            out = torch.as_tensor(value)
        if out.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got shape {tuple(out.shape)}")
        return out

    @staticmethod
    def _as_feature(value):
        if isinstance(value, torch.Tensor):
            return value.clone().to(dtype=torch.float64).reshape(-1)
        return torch.as_tensor(value, dtype=torch.float64).reshape(-1)

    def to(self, device):
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.Phi = self.Phi.to(device)
        self.R = self.R.to(device)
        if self.omega is not None:
            self.omega = self.omega.to(device)
        return self

    def print_policy(self, pi):
        pi = pi.detach().cpu() if isinstance(pi, torch.Tensor) else torch.as_tensor(pi)
        for i, state in enumerate(self.states.detach().cpu()):
            best_action_idx = int(torch.argmax(pi[i]).item())
            best_action = self.actions.detach().cpu()[best_action_idx]

            print(f"  State {state}: ", end="")
            for j, action in enumerate(self.actions.detach().cpu()):
                print(f"pi(a={action}|s={state}) = {pi[i, j].item():.2f}  ", end="")
            print(f"--> best action: {best_action}")
        print()
