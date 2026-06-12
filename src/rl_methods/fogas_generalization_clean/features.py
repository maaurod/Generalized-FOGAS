from typing import Protocol

import torch

__all__ = [
    "FeatureFunction",
    "TabularFeatures",
    "LinearFunction",
    "LinearQFunction",
    "LinearUFunction",
    "TabularPolicyFeatures",
    "build_feature_table",
    "build_policy_feature_table",
    "build_q_feature_table",
    "build_u_feature_table",
]


class FeatureFunction(Protocol):
    """Finite state-action feature parametrization interface."""

    def table(self, n_states, n_actions, device=None, dtype=torch.float64):
        """Return a table with shape (n_states, n_actions, d)."""
        ...


class TabularFeatures:
    """One-hot features for finite state-action pairs."""

    def __init__(self, n_states, n_actions, dtype=torch.float64):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.d = self.n_states * self.n_actions
        self.dtype = dtype

    def __call__(self, state, action):
        state = int(state)
        action = int(action)
        if state < 0 or state >= self.n_states:
            raise ValueError(f"state must be in [0, {self.n_states}), got {state}")
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action}")
        feature = torch.zeros(self.d, dtype=self.dtype)
        feature[state * self.n_actions + action] = 1.0
        return feature

    def table(self, n_states=None, n_actions=None, device=None, dtype=None):
        # Accept optional dimensions so callers can use the same table API for
        # tabular features and custom feature objects.
        if n_states is not None and int(n_states) != self.n_states:
            raise ValueError(f"n_states mismatch: expected {self.n_states}, got {n_states}")
        if n_actions is not None and int(n_actions) != self.n_actions:
            raise ValueError(f"n_actions mismatch: expected {self.n_actions}, got {n_actions}")
        dtype = self.dtype if dtype is None else dtype
        table = torch.eye(self.d, dtype=dtype, device=device)
        return table.reshape(self.n_states, self.n_actions, self.d)


class LinearFunction:
    """
    Linear state-action function parametrization.

    f_w(x, a) = <w, features(x, a)>
    """

    def __init__(self, features):
        if features is None:
            raise ValueError("features must be provided")
        self.features = features

    def __call__(self, state, action):
        return self.features(state, action)

    def table(self, n_states, n_actions, device=None, dtype=torch.float64):
        return build_feature_table(self.features, n_states, n_actions, device=device, dtype=dtype)


def build_feature_table(features, n_states, n_actions, device=None, dtype=torch.float64, name="features"):
    """
    Build a finite state-action feature table.

    The input can expose table(...) or be callable as features(x, a).
    """
    if features is None:
        raise ValueError(f"{name} must be provided")

    # Prefer an explicit table method when available; otherwise sample every
    # finite state-action pair from a callable feature map.
    if hasattr(features, "table"):
        try:
            table = features.table(n_states, n_actions, device=device, dtype=dtype)
        except TypeError:
            table = features.table(device=device, dtype=dtype)
    else:
        table = torch.stack(
            [
                torch.stack(
                    [
                        torch.as_tensor(features(x, a), dtype=dtype, device=device).reshape(-1)
                        for a in range(n_actions)
                    ],
                    dim=0,
                )
                for x in range(n_states)
            ],
            dim=0,
        )

    if table.ndim != 3 or table.shape[0] != n_states or table.shape[1] != n_actions:
        raise ValueError(
            f"{name} table must have shape "
            f"({n_states}, {n_actions}, d), got {tuple(table.shape)}"
        )

    # This catches custom feature maps that accidentally return different
    # dimensions for different state-action pairs.
    dims = {int(table[x, a].numel()) for x in range(n_states) for a in range(n_actions)}
    if len(dims) != 1:
        raise ValueError(f"{name} must return a fixed feature dimension")

    return table.to(dtype=dtype, device=device)


def build_u_feature_table(u_function, n_states, n_actions, device=None, dtype=torch.float64):
    return build_feature_table(
        u_function,
        n_states,
        n_actions,
        device=device,
        dtype=dtype,
        name="u_function",
    )


def build_q_feature_table(q_function, n_states, n_actions, device=None, dtype=torch.float64):
    return build_feature_table(
        q_function,
        n_states,
        n_actions,
        device=device,
        dtype=dtype,
        name="q_function",
    )


def build_policy_feature_table(policy_features, n_states, n_actions, device=None, dtype=torch.float64):
    return build_feature_table(
        policy_features,
        n_states,
        n_actions,
        device=device,
        dtype=dtype,
        name="policy_features",
    )


LinearUFunction = LinearFunction
LinearQFunction = LinearFunction
TabularPolicyFeatures = TabularFeatures
