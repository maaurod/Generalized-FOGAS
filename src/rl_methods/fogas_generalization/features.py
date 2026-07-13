"""Finite-space feature maps and parametrizations for Generalized FOGAS.

The module has three layers.  Feature-map classes describe finite
state--action pairs; table builders normalize custom maps to the tensor layout
used by the linear solvers; and PyTorch wrappers expose the ``u``, ``q``, and
policy interfaces consumed by :class:`FinalParametrizedSolver`.  The three
optimization variables intentionally may use different representations and
parameter dimensions.
"""

from typing import Callable, Iterable, Optional, Protocol, Sequence, Union

import torch
from torch import nn

__all__ = [
    "FeatureFunction",
    "TabularFeatures",
    "RBFStateFeatures",
    "RBFStateActionFeatures",
    "LinearFunction",
    "LinearQFunction",
    "LinearUFunction",
    "UParam",
    "QParam",
    "PolicyParam",
    "LinearUParam",
    "LinearQParam",
    "SoftmaxLinearPolicyParam",
    "NeuralUParam",
    "NeuralQParam",
    "NeuralPolicyParam",
    "StateActionMLPModule",
    "StateMLPPolicyModule",
    "TabularPolicyFeatures",
    "build_feature_table",
    "build_policy_feature_table",
    "build_q_feature_table",
    "build_u_feature_table",
]

TensorLike = Union[torch.Tensor, list, tuple]


# ---------------------------------------------------------------------------
# Finite feature maps
# ---------------------------------------------------------------------------

class FeatureFunction(Protocol):
    """Protocol for a finite state--action feature map."""

    def table(self, n_states, n_actions, device=None, dtype=torch.float64):
        """Return a table with shape (n_states, n_actions, d)."""
        ...


class TabularFeatures:
    """One-hot features with one coordinate per finite state--action pair.

    The dimension is ``n_states * n_actions``.  These features remove
    function-approximation error from tabular update ablations and make
    ``u_beta`` capable of assigning an independent weight to every pair.
    """

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


class RBFStateFeatures:
    """Radial-basis representation of finite states through coordinates.

    State identifiers are first mapped to coordinate vectors.  Each feature is
    a Gaussian response around one supplied center; an optional bias coordinate
    can be appended.  Bandwidths may be supplied directly or inferred from the
    spacing of the centers.
    """

    def __init__(
        self,
        n_states,
        centers,
        variance: Optional[Union[float, TensorLike]] = None,
        sigma: Optional[Union[float, TensorLike]] = None,
        state_coords: Optional[TensorLike] = None,
        coord_fn: Optional[Callable[[int], TensorLike]] = None,
        bandwidth="nearest",
        bandwidth_scale=1.0,
        include_bias=True,
        normalize=False,
        dtype=torch.float64,
    ):
        self.n_states = int(n_states)
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")
        if state_coords is not None and coord_fn is not None:
            raise ValueError("Provide either state_coords or coord_fn, not both")
        if variance is not None and sigma is not None:
            raise ValueError("Provide either variance or sigma, not both")

        self.dtype = dtype
        self.centers = torch.as_tensor(centers, dtype=dtype)
        if self.centers.ndim == 1:
            self.centers = self.centers[:, None]
        if self.centers.ndim != 2 or self.centers.shape[0] == 0:
            raise ValueError("centers must have shape (n_centers, coord_dim)")

        self.state_coords = self._build_state_coords(state_coords, coord_fn)
        if self.state_coords.shape[1] != self.centers.shape[1]:
            raise ValueError(
                "state coordinates and centers must have the same coordinate dimension"
            )

        if sigma is not None:
            self.variance = self._expand_bandwidth(sigma, "sigma").square()
        elif variance is not None:
            self.variance = self._expand_bandwidth(variance, "variance")
        else:
            self.variance = self.infer_variance(
                self.centers,
                method=bandwidth,
                scale=bandwidth_scale,
                dtype=dtype,
            )

        if torch.any(self.variance <= 0):
            raise ValueError("variance must be positive")

        self.include_bias = bool(include_bias)
        self.normalize = bool(normalize)
        self.n_centers = int(self.centers.shape[0])
        self.d = self.n_centers + int(self.include_bias)

    def _build_state_coords(self, state_coords, coord_fn):
        if state_coords is not None:
            coords = torch.as_tensor(state_coords, dtype=self.dtype)
        elif coord_fn is not None:
            coords = torch.stack(
                [
                    torch.as_tensor(coord_fn(s), dtype=self.dtype).reshape(-1)
                    for s in range(self.n_states)
                ],
                dim=0,
            )
        else:
            coords = torch.arange(self.n_states, dtype=self.dtype).reshape(self.n_states, 1)

        if coords.ndim == 1:
            coords = coords[:, None]
        if coords.shape[0] != self.n_states:
            raise ValueError(
                "state_coords must have one row per state: "
                f"expected {self.n_states}, got {coords.shape[0]}"
            )
        return coords

    def _expand_bandwidth(self, value, name):
        bandwidth = torch.as_tensor(value, dtype=self.dtype)
        if bandwidth.ndim == 0:
            bandwidth = bandwidth.repeat(self.centers.shape[0])
        bandwidth = bandwidth.reshape(-1)
        if bandwidth.numel() != self.centers.shape[0]:
            raise ValueError(
                f"{name} must be scalar or have one value per center; "
                f"got {bandwidth.numel()}"
            )
        return bandwidth

    @staticmethod
    def infer_variance(centers, method="nearest", scale=1.0, dtype=torch.float64):
        centers_t = torch.as_tensor(centers, dtype=dtype)
        if centers_t.ndim == 1:
            centers_t = centers_t[:, None]
        if centers_t.ndim != 2 or centers_t.shape[0] == 0:
            raise ValueError("centers must have shape (n_centers, coord_dim)")
        if scale <= 0:
            raise ValueError("scale must be positive")

        if centers_t.shape[0] == 1:
            sigma = torch.tensor(float(scale), dtype=dtype)
            return sigma.square().repeat(1)

        distances = torch.cdist(centers_t, centers_t)
        nonzero_distances = distances[distances > 0]
        if nonzero_distances.numel() == 0:
            raise ValueError("Cannot infer variance from duplicate centers only")

        if method == "median":
            sigma = torch.median(nonzero_distances)
        elif method == "nearest":
            masked = distances.masked_fill(distances <= 0, float("inf"))
            sigma = torch.median(masked.min(dim=1).values)
        elif method == "mean_nearest":
            masked = distances.masked_fill(distances <= 0, float("inf"))
            sigma = masked.min(dim=1).values.mean()
        else:
            raise ValueError("method must be one of {'median', 'nearest', 'mean_nearest'}")

        return (sigma * float(scale)).square().repeat(centers_t.shape[0])

    def __call__(self, state):
        state = int(state)
        if state < 0 or state >= self.n_states:
            raise ValueError(f"state must be in [0, {self.n_states}), got {state}")

        diff = self.state_coords[state] - self.centers
        dist_sq = torch.sum(diff * diff, dim=1)
        rbf = torch.exp(-dist_sq / (2.0 * self.variance))
        if self.normalize:
            rbf_sum = rbf.sum().clamp_min(torch.finfo(self.dtype).eps)
            rbf = rbf / rbf_sum
        if self.include_bias:
            return torch.cat([rbf, torch.ones(1, dtype=self.dtype)])
        return rbf

    def table(self, n_states=None, device=None, dtype=None):
        if n_states is not None and int(n_states) != self.n_states:
            raise ValueError(f"n_states mismatch: expected {self.n_states}, got {n_states}")
        dtype = self.dtype if dtype is None else dtype
        return torch.stack(
            [self(s).to(dtype=dtype, device=device) for s in range(self.n_states)],
            dim=0,
        )


class RBFStateActionFeatures:
    """Action-coupled RBF features ``zeta(s,a) = e_a kron phi(s)``.

    Placing the state representation in an action-specific block lets a linear
    model learn distinct coefficients for every action while sharing the RBF
    construction across actions.
    """

    def __init__(self, state_features, n_actions, dtype=torch.float64):
        if state_features is None:
            raise ValueError("state_features must be provided")
        self.state_features = state_features
        self.n_states = int(state_features.n_states)
        self.n_actions = int(n_actions)
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.dtype = dtype
        self.d = self.n_actions * int(state_features.d)

    def __call__(self, state, action):
        action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action}")

        state_feat = self.state_features(state).to(dtype=self.dtype)
        feature = torch.zeros(self.d, dtype=self.dtype)
        start = action * int(self.state_features.d)
        feature[start : start + int(self.state_features.d)] = state_feat
        return feature

    def table(self, n_states=None, n_actions=None, device=None, dtype=None):
        if n_states is not None and int(n_states) != self.n_states:
            raise ValueError(f"n_states mismatch: expected {self.n_states}, got {n_states}")
        if n_actions is not None and int(n_actions) != self.n_actions:
            raise ValueError(f"n_actions mismatch: expected {self.n_actions}, got {n_actions}")

        dtype = self.dtype if dtype is None else dtype
        state_table = self.state_features.table(
            n_states=self.n_states,
            device=device,
            dtype=dtype,
        )
        table = torch.zeros(
            self.n_states,
            self.n_actions,
            self.d,
            dtype=dtype,
            device=device,
        )
        state_dim = int(self.state_features.d)
        for action in range(self.n_actions):
            start = action * state_dim
            table[:, action, start : start + state_dim] = state_table
        return table


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


# ---------------------------------------------------------------------------
# Feature-table adapters used by the linear fast paths
# ---------------------------------------------------------------------------

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
    """Build the ``u_beta`` feature tensor with shape ``(N, A, d_u)``."""
    return build_feature_table(
        u_function,
        n_states,
        n_actions,
        device=device,
        dtype=dtype,
        name="u_function",
    )


def build_q_feature_table(q_function, n_states, n_actions, device=None, dtype=torch.float64):
    """Build the ``Q_theta`` feature tensor with shape ``(N, A, d_Q)``."""
    return build_feature_table(
        q_function,
        n_states,
        n_actions,
        device=device,
        dtype=dtype,
        name="q_function",
    )


def build_policy_feature_table(policy_features, n_states, n_actions, device=None, dtype=torch.float64):
    """Build the policy-logit feature tensor with shape ``(N, A, d_pi)``."""
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


# ---------------------------------------------------------------------------
# Interfaces expected by the reference discrete solver
# ---------------------------------------------------------------------------

class UParam(Protocol):
    """Residual-weighting function ``u_beta`` interface."""

    is_linear_fast_path: bool

    def parameters(self, recurse: bool = True) -> Iterable[torch.nn.Parameter]:
        ...

    def to(self, *args, **kwargs):
        ...

    def u(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        ...


class QParam(Protocol):
    """Action-value function ``Q_theta`` interface."""

    is_linear_fast_path: bool

    def parameters(self, recurse: bool = True) -> Iterable[torch.nn.Parameter]:
        ...

    def to(self, *args, **kwargs):
        ...

    def q(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        ...


class PolicyParam(Protocol):
    """Discrete softmax policy parametrization interface."""

    is_linear_fast_path: bool

    def parameters(self, recurse: bool = True) -> Iterable[torch.nn.Parameter]:
        ...

    def to(self, *args, **kwargs):
        ...

    def logits(self, states: torch.Tensor) -> torch.Tensor:
        ...

    def probs(self, states: torch.Tensor) -> torch.Tensor:
        ...

    def log_prob_actions(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        ...


class LinearUParam(nn.Module):
    """Linear residual-weighting model ``u_beta = beta^T phi_u``.

    The registered feature table marks this module as a linear fast path, so
    the solver can use the feature covariance directly instead of constructing
    sample Jacobians with autograd.
    """

    is_linear_fast_path = True

    def __init__(
        self,
        features,
        n_states,
        n_actions,
        dtype=torch.float64,
    ):
        super().__init__()
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.feature_source = features
        table = build_u_feature_table(
            features,
            self.n_states,
            self.n_actions,
            dtype=dtype,
        )
        self.d = int(table.shape[2])
        self.register_buffer("feature_table", table)
        self.beta = nn.Parameter(torch.zeros(self.d, dtype=dtype))

    def features(self, states, actions):
        return self.feature_table[states.long(), actions.long()]

    def u(self, states, actions):
        return self.features(states, actions) @ self.beta

    def forward(self, states, actions):
        return self.u(states, actions)


class LinearQParam(nn.Module):
    """Linear action-value model Q_theta(s, a) = theta^T phi_q(s, a)."""

    is_linear_fast_path = True

    def __init__(
        self,
        features,
        n_states,
        n_actions,
        dtype=torch.float64,
    ):
        super().__init__()
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.feature_source = features
        table = build_q_feature_table(
            features,
            self.n_states,
            self.n_actions,
            dtype=dtype,
        )
        self.d = int(table.shape[2])
        self.register_buffer("feature_table", table)
        self.theta = nn.Parameter(torch.zeros(self.d, dtype=dtype))

    def features(self, states, actions):
        return self.feature_table[states.long(), actions.long()]

    def q(self, states, actions):
        return self.features(states, actions) @ self.theta

    def forward(self, states, actions):
        return self.q(states, actions)


class SoftmaxLinearPolicyParam(nn.Module):
    """Softmax-linear policy with state-action logits psi^T omega(s, a)."""

    is_linear_fast_path = True

    def __init__(
        self,
        features,
        n_states,
        n_actions,
        dtype=torch.float64,
    ):
        super().__init__()
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.feature_source = features
        table = build_policy_feature_table(
            features,
            self.n_states,
            self.n_actions,
            dtype=dtype,
        )
        self.d = int(table.shape[2])
        self.register_buffer("feature_table", table)
        self.psi = nn.Parameter(torch.zeros(self.d, dtype=dtype))

    def features(self, states):
        return self.feature_table[states.long()]

    def logits(self, states):
        return torch.tensordot(self.features(states), self.psi, dims=([2], [0]))

    def probs(self, states):
        return torch.softmax(self.logits(states), dim=-1)

    def log_probs(self, states):
        return torch.log_softmax(self.logits(states), dim=-1)

    def log_prob_actions(self, states, actions):
        return self.log_probs(states).gather(-1, actions.long()[..., None]).squeeze(-1)

    def forward(self, states):
        return self.probs(states)


# ---------------------------------------------------------------------------
# Neural modules and solver-interface wrappers
# ---------------------------------------------------------------------------

def _activation_factory(activation):
    if isinstance(activation, nn.Module):
        return lambda: activation.__class__()
    return activation


def _build_mlp(
    input_dim,
    output_dim,
    hidden_sizes: Sequence[int] = (64, 64),
    activation=nn.Tanh,
    dtype=torch.float64,
):
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive")

    make_activation = _activation_factory(activation)
    layers = []
    prev_dim = input_dim
    for hidden_size in hidden_sizes:
        hidden_size = int(hidden_size)
        if hidden_size <= 0:
            raise ValueError("hidden_sizes entries must be positive")
        layers.append(nn.Linear(prev_dim, hidden_size, dtype=dtype))
        layers.append(make_activation())
        prev_dim = hidden_size
    layers.append(nn.Linear(prev_dim, output_dim, dtype=dtype))
    return nn.Sequential(*layers)


def _default_index_inputs(n_items, dtype):
    return torch.eye(int(n_items), dtype=dtype)


def _input_table(n_items, inputs, dtype, name):
    n_items = int(n_items)
    if n_items <= 0:
        raise ValueError(f"{name} count must be positive")
    if inputs is None:
        return _default_index_inputs(n_items, dtype)
    table = torch.as_tensor(inputs, dtype=dtype)
    if table.ndim == 1:
        table = table[:, None]
    if table.ndim != 2 or table.shape[0] != n_items:
        raise ValueError(f"{name} inputs must have shape ({n_items}, input_dim)")
    return table


class StateActionMLPModule(nn.Module):
    """
    Tanh MLP over discrete state-action descriptors.

    If descriptors are omitted, state and action identifiers are represented
    with one-hot vectors.  Supplying coordinates or other descriptors is useful
    when the model should share information between nearby states.  The output
    is scalar for the standard ``u`` and ``Q`` wrappers, but may be an embedding
    for a custom wrapper.
    """

    def __init__(
        self,
        n_states,
        n_actions,
        state_inputs: Optional[TensorLike] = None,
        action_inputs: Optional[TensorLike] = None,
        hidden_sizes: Sequence[int] = (64, 64),
        output_dim=1,
        activation=nn.Tanh,
        dtype=torch.float64,
    ):
        super().__init__()
        state_table = _input_table(n_states, state_inputs, dtype, "state")
        action_table = _input_table(n_actions, action_inputs, dtype, "action")
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.state_dim = int(state_table.shape[1])
        self.action_dim = int(action_table.shape[1])
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.register_buffer("state_inputs", state_table)
        self.register_buffer("action_inputs", action_table)
        self.net = _build_mlp(
            input_dim=self.state_dim + self.action_dim,
            output_dim=self.output_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dtype=dtype,
        )

    def forward(self, states, actions):
        states_b, actions_b = torch.broadcast_tensors(states.long(), actions.long())
        original_shape = states_b.shape
        states_flat = states_b.reshape(-1)
        actions_flat = actions_b.reshape(-1)
        state_x = self.state_inputs[states_flat]
        action_x = self.action_inputs[actions_flat]
        out = self.net(torch.cat([state_x, action_x], dim=-1))
        return out.reshape(*original_shape, self.output_dim)


class StateMLPPolicyModule(nn.Module):
    """Tanh MLP policy-logit network over discrete state descriptors."""

    def __init__(
        self,
        n_states,
        n_actions,
        state_inputs: Optional[TensorLike] = None,
        hidden_sizes: Sequence[int] = (64, 64),
        activation=nn.Tanh,
        dtype=torch.float64,
    ):
        super().__init__()
        state_table = _input_table(n_states, state_inputs, dtype, "state")
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.input_dim = int(state_table.shape[1])
        self.register_buffer("state_inputs", state_table)
        self.net = _build_mlp(
            input_dim=self.input_dim,
            output_dim=self.n_actions,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dtype=dtype,
        )

    def forward(self, states):
        states_flat = states.long().reshape(-1)
        out = self.net(self.state_inputs[states_flat])
        return out.reshape(*states.shape, self.n_actions)


class NeuralUParam(nn.Module):
    """Adapt a scalar neural module to the ``u_beta(states, actions)`` API."""

    is_linear_fast_path = False

    def __init__(self, module):
        super().__init__()
        self.module = module

    def u(self, states, actions):
        return self.module(states.long(), actions.long()).squeeze(-1)

    def forward(self, states, actions):
        return self.u(states, actions)


class NeuralQParam(nn.Module):
    """Adapt a scalar neural module to the ``Q_theta(states, actions)`` API."""

    is_linear_fast_path = False

    def __init__(self, module):
        super().__init__()
        self.module = module

    def q(self, states, actions):
        return self.module(states.long(), actions.long()).squeeze(-1)

    def forward(self, states, actions):
        return self.q(states, actions)


class NeuralPolicyParam(nn.Module):
    """Neural discrete policy wrapper. The module maps states to action logits."""

    is_linear_fast_path = False

    def __init__(self, module):
        super().__init__()
        self.module = module

    def logits(self, states):
        return self.module(states.long())

    def probs(self, states):
        return torch.softmax(self.logits(states), dim=-1)

    def log_probs(self, states):
        return torch.log_softmax(self.logits(states), dim=-1)

    def log_prob_actions(self, states, actions):
        return self.log_probs(states).gather(-1, actions.long()[..., None]).squeeze(-1)

    def forward(self, states):
        return self.probs(states)
