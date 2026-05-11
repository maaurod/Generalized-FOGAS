import torch


class TabularPolicyFeatures:
    """One-hot policy features omega_pi(x, a) for finite state-action pairs."""

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

    def table(self, device=None, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        table = torch.eye(self.d, dtype=dtype, device=device)
        return table.reshape(self.n_states, self.n_actions, self.d)


def build_policy_feature_table(policy_features, n_states, n_actions, device=None, dtype=torch.float64):
    if hasattr(policy_features, "table"):
        table = policy_features.table(device=device, dtype=dtype)
    else:
        table = torch.stack(
            [
                torch.stack(
                    [
                        torch.as_tensor(policy_features(x, a), dtype=dtype, device=device).reshape(-1)
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
            "policy feature table must have shape "
            f"({n_states}, {n_actions}, d_pi), got {tuple(table.shape)}"
        )

    dims = {int(table[x, a].numel()) for x in range(n_states) for a in range(n_actions)}
    if len(dims) != 1:
        raise ValueError("policy_features must return a fixed feature dimension")

    return table.to(dtype=dtype, device=device)
