"""Feature maps and parametrization modules for discrete SBEED.

The early solvers use callable feature maps directly. The final solver wraps
those maps in PyTorch modules so the same update code can handle tabular,
RBF-style linear, and neural parametrizations.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Protocol, Sequence, Union, runtime_checkable

import torch
from torch import nn

TensorLike = Union[torch.Tensor, list, tuple]


class TabularStateFeatures:
    """
    One-hot state features phi(s).

    Best for small finite MDPs because the value function can represent an
    independent scalar for every state.
    """

    def __init__(self, n_states: int, dtype: torch.dtype = torch.float64):
        self.n_states = int(n_states)
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")
        self.d = self.n_states
        self.dtype = dtype

    def __call__(self, state: int) -> torch.Tensor:
        state = int(state)
        if state < 0 or state >= self.n_states:
            raise ValueError(f"state must be in [0, {self.n_states}), got {state}")
        feat = torch.zeros(self.d, dtype=self.dtype)
        feat[state] = 1.0
        return feat


class TabularStateActionFeatures:
    """
    One-hot state-action features rho_features(s, a).

    Best for small finite MDPs because the dual model can represent an
    independent scalar for every observed state-action pair.
    """

    def __init__(self, n_states: int, n_actions: int, dtype: torch.dtype = torch.float64):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.d = self.n_states * self.n_actions
        self.dtype = dtype

    def __call__(self, state: int, action: int) -> torch.Tensor:
        state = int(state)
        action = int(action)
        if state < 0 or state >= self.n_states:
            raise ValueError(f"state must be in [0, {self.n_states}), got {state}")
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action}")
        feat = torch.zeros(self.d, dtype=self.dtype)
        feat[state * self.n_actions + action] = 1.0
        return feat


class RBFStateFeatures:
    """Radial basis state features phi(s) for finite discrete states.

    `centers` and state coordinates must live in the same coordinate system.
    For a gridworld this is usually normalized row/column coordinates, e.g.
    state 0 -> [0.0, 0.0] and the bottom-right state -> [1.0, 1.0].
    """

    def __init__(
        self,
        n_states: int,
        centers: TensorLike,
        variance: Optional[Union[float, TensorLike]] = None,
        sigma: Optional[Union[float, TensorLike]] = None,
        state_coords: Optional[TensorLike] = None,
        coord_fn: Optional[Callable[[int], TensorLike]] = None,
        bandwidth: str = "nearest",
        bandwidth_scale: float = 1.0,
        include_bias: bool = True,
        normalize: bool = False,
        dtype: torch.dtype = torch.float64,
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
            self.variance = self._expand_bandwidth(sigma, "sigma") ** 2
        elif variance is not None:
            self.variance = self._expand_bandwidth(variance, "variance")
        else:
            self.variance = self.infer_variance(
                self.centers, method=bandwidth, scale=bandwidth_scale
            ).to(dtype=dtype)

        if torch.any(self.variance <= 0):
            raise ValueError("variance must be positive")

        self.include_bias = bool(include_bias)
        self.normalize = bool(normalize)
        self.n_centers = int(self.centers.shape[0])
        self.d = self.n_centers + int(self.include_bias)

    def _build_state_coords(
        self,
        state_coords: Optional[TensorLike],
        coord_fn: Optional[Callable[[int], TensorLike]],
    ) -> torch.Tensor:
        if state_coords is not None:
            coords = torch.as_tensor(state_coords, dtype=self.dtype)
        elif coord_fn is not None:
            coords = torch.stack(
                [torch.as_tensor(coord_fn(s), dtype=self.dtype).reshape(-1) for s in range(self.n_states)],
                dim=0,
            )
        else:
            coords = torch.arange(self.n_states, dtype=self.dtype).reshape(self.n_states, 1)

        if coords.ndim == 1:
            coords = coords[:, None]
        if coords.shape[0] != self.n_states:
            raise ValueError(
                f"state_coords must have one row per state: expected {self.n_states}, got {coords.shape[0]}"
            )
        return coords

    def _expand_bandwidth(self, value: Union[float, TensorLike], name: str) -> torch.Tensor:
        bandwidth = torch.as_tensor(value, dtype=self.dtype)
        if bandwidth.ndim == 0:
            bandwidth = bandwidth.repeat(self.centers.shape[0])
        bandwidth = bandwidth.reshape(-1)
        if bandwidth.numel() != self.centers.shape[0]:
            raise ValueError(
                f"{name} must be scalar or have one value per center; got {bandwidth.numel()}"
            )
        return bandwidth

    @staticmethod
    def infer_variance(
        centers: TensorLike,
        method: str = "nearest",
        scale: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Infer a scalar RBF variance from pairwise center distances.

        Methods:
            "median": squared median nonzero pairwise distance.
            "nearest": squared median nearest-neighbor center distance.
            "mean_nearest": squared mean nearest-neighbor center distance.
        """
        centers_t = torch.as_tensor(centers, dtype=dtype)
        if centers_t.ndim == 1:
            centers_t = centers_t[:, None]
        if centers_t.ndim != 2 or centers_t.shape[0] == 0:
            raise ValueError("centers must have shape (n_centers, coord_dim)")
        if scale <= 0:
            raise ValueError("scale must be positive")

        n_centers = centers_t.shape[0]
        if n_centers == 1:
            sigma = torch.tensor(float(scale), dtype=dtype)
            return sigma.square()

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

        sigma = sigma * float(scale)
        return sigma.square()

    def __call__(self, state: int) -> torch.Tensor:
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


class RBFStateActionFeatures:
    """
    Action-coupled RBF features zeta(s, a) = e_a kron phi(s).

    The feature vector contains one block per action; only the selected action
    block is filled with the state RBF feature. This is the linear rho
    parametrization used in the RBF grid-search experiments.
    """

    def __init__(
        self,
        state_features: RBFStateFeatures,
        n_actions: int,
        dtype: torch.dtype = torch.float64,
    ):
        self.state_features = state_features
        self.n_states = state_features.n_states
        self.n_actions = int(n_actions)
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.dtype = dtype
        self.d = self.n_actions * state_features.d

    def __call__(self, state: int, action: int) -> torch.Tensor:
        action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action}")
        state_feat = self.state_features(state).to(dtype=self.dtype)
        feat = torch.zeros(self.d, dtype=self.dtype)
        start = action * self.state_features.d
        feat[start:start + self.state_features.d] = state_feat
        return feat


@runtime_checkable
class ValueParam(Protocol):
    """Value parametrization interface used by the generalized SBEED solver."""

    is_linear_fast_path: bool

    def parameters(self, recurse: bool = True) -> Iterable[torch.nn.Parameter]:
        ...

    def to(self, *args, **kwargs):
        ...

    def value(self, states: torch.Tensor) -> torch.Tensor:
        ...


@runtime_checkable
class RhoParam(Protocol):
    """Dual residual parametrization interface used by generalized SBEED."""

    is_linear_fast_path: bool

    def parameters(self, recurse: bool = True) -> Iterable[torch.nn.Parameter]:
        ...

    def to(self, *args, **kwargs):
        ...

    def rho(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        ...


@runtime_checkable
class PolicyParam(Protocol):
    """Policy parametrization interface used by generalized SBEED."""

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


def _feature_to_tensor(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype).reshape(-1)


class LinearValueParam(nn.Module):
    """Linear value model V(s) = theta^T phi(s)."""

    is_linear_fast_path = True

    def __init__(
        self,
        value_features: Callable[[int], torch.Tensor],
        n_states: int,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.value_features = value_features
        self.n_states = int(n_states)
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")
        features = [
            _feature_to_tensor(value_features(s), dtype)
            for s in range(self.n_states)
        ]
        dims = {feat.numel() for feat in features}
        if len(dims) != 1:
            raise ValueError("value_features must return a fixed feature dimension")
        self.d = int(features[0].numel())
        self.register_buffer("feature_table", torch.stack(features, dim=0))
        self.theta = nn.Parameter(torch.zeros(self.d, dtype=dtype))

    def features(self, states: torch.Tensor) -> torch.Tensor:
        return self.feature_table[states.long()]

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.features(states) @ self.theta

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value(states)


class LinearRhoParam(nn.Module):
    """Linear multistep rho model over discounted state-action feature sums."""

    is_linear_fast_path = True

    def __init__(
        self,
        rho_features: Callable[[int, int], torch.Tensor],
        n_states: int,
        n_actions: int,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.rho_features = rho_features
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        rows = [
            [
                _feature_to_tensor(rho_features(s, a), dtype)
                for a in range(self.n_actions)
            ]
            for s in range(self.n_states)
        ]
        dims = {feat.numel() for row in rows for feat in row}
        if len(dims) != 1:
            raise ValueError("rho_features must return a fixed feature dimension")
        self.d = int(rows[0][0].numel())
        self.register_buffer("feature_table", torch.stack([torch.stack(row, dim=0) for row in rows], dim=0))
        self.beta = nn.Parameter(torch.zeros(self.d, dtype=dtype))

    def fragment_features(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        step_features = self.feature_table[states.long(), actions.long()]
        return (weights.to(dtype=step_features.dtype)[:, :, None] * step_features).sum(dim=1)

    def rho(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.fragment_features(states, actions, weights) @ self.beta

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.rho(states, actions, weights)


class SoftmaxLinearPolicyParam(nn.Module):
    """Softmax-linear policy pi(a|s) = softmax(W chi(s))[a]."""

    is_linear_fast_path = True

    def __init__(
        self,
        policy_features: Callable[[int], torch.Tensor],
        n_states: int,
        n_actions: int,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.policy_features = policy_features
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        features = [
            _feature_to_tensor(policy_features(s), dtype)
            for s in range(self.n_states)
        ]
        dims = {feat.numel() for feat in features}
        if len(dims) != 1:
            raise ValueError("policy_features must return a fixed feature dimension")
        self.d = int(features[0].numel())
        self.register_buffer("feature_table", torch.stack(features, dim=0))
        self.W = nn.Parameter(torch.zeros((self.n_actions, self.d), dtype=dtype))

    def features(self, states: torch.Tensor) -> torch.Tensor:
        return self.feature_table[states.long()]

    def logits(self, states: torch.Tensor) -> torch.Tensor:
        return self.features(states) @ self.W.T

    def probs(self, states: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(states), dim=-1)

    def log_probs(self, states: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.logits(states), dim=-1)

    def log_prob_actions(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.log_probs(states).gather(-1, actions.long()[..., None]).squeeze(-1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.probs(states)


class NeuralValueParam(nn.Module):
    """Neural value model wrapper. The module maps state tensors to scalar values."""

    is_linear_fast_path = False

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def value(self, states: torch.Tensor) -> torch.Tensor:
        output = self.module(states.long())
        return output.squeeze(-1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value(states)


class NeuralRhoParam(nn.Module):
    """
    Neural rho wrapper.

    `embed_module(states, actions)` must return per-step embeddings. The solver
    forms sum_l gamma^l embedding(s_l, a_l), then `head_module` maps the sum to
    scalar rho.
    """

    is_linear_fast_path = False

    def __init__(self, embed_module: nn.Module, head_module: nn.Module):
        super().__init__()
        self.embed_module = embed_module
        self.head_module = head_module

    def rho(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.embed_module(states.long(), actions.long())
        weighted = weights.to(dtype=embeddings.dtype, device=embeddings.device)[:, :, None] * embeddings
        summed = weighted.sum(dim=1)
        return self.head_module(summed).squeeze(-1)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.rho(states, actions, weights)


class NeuralPolicyParam(nn.Module):
    """Neural policy wrapper. The module maps state tensors to action logits."""

    is_linear_fast_path = False

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def logits(self, states: torch.Tensor) -> torch.Tensor:
        return self.module(states.long())

    def probs(self, states: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(states), dim=-1)

    def log_probs(self, states: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.logits(states), dim=-1)

    def log_prob_actions(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.log_probs(states).gather(-1, actions.long()[..., None]).squeeze(-1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.probs(states)


class IdentityHead(nn.Module):
    """Small head module useful when an embedding module already outputs rho."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _activation_factory(activation: Union[type[nn.Module], Callable[[], nn.Module], nn.Module]) -> Callable[[], nn.Module]:
    if isinstance(activation, nn.Module):
        return lambda: activation.__class__()
    return activation


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: Union[type[nn.Module], Callable[[], nn.Module], nn.Module] = nn.Tanh,
    dtype: torch.dtype = torch.float32,
) -> nn.Sequential:
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive")
    make_activation = _activation_factory(activation)
    layers: list[nn.Module] = []
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


def _default_index_inputs(n_items: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(int(n_items), dtype=dtype)


def _input_table(
    n_items: int,
    inputs: Optional[TensorLike],
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
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


class StateMLPValueModule(nn.Module):
    """
    Configurable MLP value network over discrete state indices.

    If `state_inputs` is omitted, states are represented as one-hot vectors.
    Pass a tensor of shape (n_states, d_s) to use coordinates or other fixed
    state descriptors.
    """

    def __init__(
        self,
        n_states: int,
        state_inputs: Optional[TensorLike] = None,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Union[type[nn.Module], Callable[[], nn.Module], nn.Module] = nn.Tanh,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        table = _input_table(n_states, state_inputs, dtype, "state")
        self.n_states = int(n_states)
        self.input_dim = int(table.shape[1])
        self.register_buffer("state_inputs", table)
        self.net = _build_mlp(
            input_dim=self.input_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dtype=dtype,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.state_inputs[states.long()]
        return self.net(x)


class StateActionMLPModule(nn.Module):
    """
    Configurable MLP over concatenated state-action descriptors.

    If `state_inputs` or `action_inputs` are omitted, one-hot descriptors are
    used. The output dimension is configurable so this can be used either as a
    scalar rho network (`output_dim=1`) or as a rho embedding network.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        state_inputs: Optional[TensorLike] = None,
        action_inputs: Optional[TensorLike] = None,
        hidden_sizes: Sequence[int] = (64, 64),
        output_dim: int = 1,
        activation: Union[type[nn.Module], Callable[[], nn.Module], nn.Module] = nn.Tanh,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        state_table = _input_table(n_states, state_inputs, dtype, "state")
        action_table = _input_table(n_actions, action_inputs, dtype, "action")
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.state_dim = int(state_table.shape[1])
        self.action_dim = int(action_table.shape[1])
        self.output_dim = int(output_dim)
        self.register_buffer("state_inputs", state_table)
        self.register_buffer("action_inputs", action_table)
        self.net = _build_mlp(
            input_dim=self.state_dim + self.action_dim,
            output_dim=self.output_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dtype=dtype,
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state_x = self.state_inputs[states.long()]
        action_x = self.action_inputs[actions.long()]
        x = torch.cat([state_x, action_x], dim=-1)
        return self.net(x)


class StateMLPPolicyModule(nn.Module):
    """
    Configurable MLP policy-logit network over discrete state indices.

    The output has shape (..., n_actions) and is intended to be wrapped by
    `NeuralPolicyParam`.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        state_inputs: Optional[TensorLike] = None,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Union[type[nn.Module], Callable[[], nn.Module], nn.Module] = nn.Tanh,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        table = _input_table(n_states, state_inputs, dtype, "state")
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.input_dim = int(table.shape[1])
        self.register_buffer("state_inputs", table)
        self.net = _build_mlp(
            input_dim=self.input_dim,
            output_dim=self.n_actions,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dtype=dtype,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.state_inputs[states.long()]
        return self.net(x)
