"""Parametrization modules for continuous SBEED.

Continuous SBEED uses neural scalar value/rho models and a Gaussian policy. The
policy implementation here uses fixed random Fourier features for the mean so
the policy is nonlinear in observations but still compact.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Protocol, Sequence, Union, runtime_checkable

import torch
from torch import nn

TensorLike = Union[torch.Tensor, list, tuple]


@runtime_checkable
class ContinuousValueParam(Protocol):
    """Value parametrization for continuous observations."""

    is_linear_fast_path: bool

    def parameters(self, recurse: bool = True) -> Iterable[torch.nn.Parameter]:
        ...

    def to(self, *args, **kwargs):
        ...

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        ...


@runtime_checkable
class ContinuousRhoParam(Protocol):
    """Dual residual parametrization for continuous observation-action fragments."""

    is_linear_fast_path: bool

    def parameters(self, recurse: bool = True) -> Iterable[torch.nn.Parameter]:
        ...

    def to(self, *args, **kwargs):
        ...

    def rho(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        ...


@runtime_checkable
class ContinuousGaussianPolicyParam(Protocol):
    """Gaussian policy parametrization for continuous actions."""

    is_linear_fast_path: bool

    def parameters(self, recurse: bool = True) -> Iterable[torch.nn.Parameter]:
        ...

    def to(self, *args, **kwargs):
        ...

    def mean(self, observations: torch.Tensor) -> torch.Tensor:
        ...

    def log_prob_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        ...

    def sample(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        ...


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


class ContinuousStateMLPValueModule(nn.Module):
    """Configurable MLP value network over raw continuous observations."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Union[type[nn.Module], Callable[[], nn.Module], nn.Module] = nn.Tanh,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        if self.obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        self.net = _build_mlp(
            input_dim=self.obs_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dtype=dtype,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class ContinuousStateActionMLPModule(nn.Module):
    """
    Configurable MLP over concatenated continuous observation-action tensors.

    Use `output_dim=1` for a scalar rho embedding, or a larger output dimension
    when paired with a separate head in `ContinuousNeuralRhoParam`.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        output_dim: int = 1,
        activation: Union[type[nn.Module], Callable[[], nn.Module], nn.Module] = nn.Tanh,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.output_dim = int(output_dim)
        if self.obs_dim <= 0 or self.action_dim <= 0 or self.output_dim <= 0:
            raise ValueError("obs_dim, action_dim, and output_dim must be positive")
        self.net = _build_mlp(
            input_dim=self.obs_dim + self.action_dim,
            output_dim=self.output_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dtype=dtype,
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observations, actions], dim=-1)
        return self.net(x)


class ContinuousNeuralValueParam(nn.Module):
    """Neural value wrapper. The module maps observations to scalar values."""

    is_linear_fast_path = False

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        output = self.module(observations)
        return output.squeeze(-1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value(observations)


class ContinuousNeuralRhoParam(nn.Module):
    """
    Neural rho wrapper for continuous fragments.

    `embed_module(observations, actions)` returns per-step embeddings. The
    solver forms sum_l gamma^l embedding(s_l, a_l), then `head_module` maps the
    sum to scalar rho.
    """

    is_linear_fast_path = False

    def __init__(self, embed_module: nn.Module, head_module: Optional[nn.Module] = None):
        super().__init__()
        self.embed_module = embed_module
        self.head_module = nn.Identity() if head_module is None else head_module

    def rho(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.embed_module(observations, actions)
        weighted = weights.to(dtype=embeddings.dtype, device=embeddings.device)[:, :, None] * embeddings
        summed = weighted.sum(dim=1)
        return self.head_module(summed).squeeze(-1)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.rho(observations, actions, weights)


class RFFGaussianPolicyParam(nn.Module):
    """
    Gaussian continuous policy with fixed random Fourier features.

    phi(s) = sin(Ps / nu + phase)
    mu(s) = W_pi phi(s) + b_pi
    pi(a|s) = Normal(mu(s), diag(exp(log_std)^2))
    """

    is_linear_fast_path = False

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_features: int = 100,
        nu: Optional[float] = None,
        init_log_std: float = -0.5,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_features = int(num_features)
        if self.obs_dim <= 0 or self.action_dim <= 0 or self.num_features <= 0:
            raise ValueError("obs_dim, action_dim, and num_features must be positive")

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))

        P = torch.randn(self.num_features, self.obs_dim, dtype=dtype, generator=generator)
        phase = (2.0 * math.pi) * torch.rand(self.num_features, dtype=dtype, generator=generator) - math.pi
        nu_value = float("nan") if nu is None else float(nu)
        if nu is not None and nu_value <= 0.0:
            raise ValueError("nu must be positive when provided")

        self.register_buffer("P", P)
        self.register_buffer("phase", phase)
        self.register_buffer("nu", torch.tensor(nu_value, dtype=dtype))
        self.W_pi = nn.Parameter(torch.zeros((self.action_dim, self.num_features), dtype=dtype))
        self.b_pi = nn.Parameter(torch.zeros(self.action_dim, dtype=dtype))
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(init_log_std), dtype=dtype))

    @property
    def nu_is_set(self) -> bool:
        return bool(torch.isfinite(self.nu).item() and self.nu.item() > 0.0)

    def set_nu(self, nu: float) -> None:
        nu = float(nu)
        if nu <= 0.0 or not math.isfinite(nu):
            raise ValueError("nu must be a finite positive value")
        self.nu.data.fill_(nu)

    @staticmethod
    def estimate_nu(observations: torch.Tensor, max_points: int = 512) -> float:
        observations = torch.as_tensor(observations, dtype=torch.float32)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        if observations.shape[0] < 2:
            return 1.0
        if observations.shape[0] > max_points:
            observations = observations[:max_points]
        distances = torch.pdist(observations)
        distances = distances[distances > 0]
        if distances.numel() == 0:
            return 1.0
        value = float(distances.mean().item())
        return value if math.isfinite(value) and value > 0.0 else 1.0

    def set_nu_from_observations(self, observations: torch.Tensor, max_points: int = 512) -> float:
        value = self.estimate_nu(observations, max_points=max_points)
        self.set_nu(value)
        return value

    def _nu_value(self) -> torch.Tensor:
        if self.nu_is_set:
            return self.nu.to(dtype=self.P.dtype, device=self.P.device)
        return torch.ones((), dtype=self.P.dtype, device=self.P.device)

    def features(self, observations: torch.Tensor) -> torch.Tensor:
        observations = torch.as_tensor(observations, dtype=self.P.dtype, device=self.P.device)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        if observations.shape[-1] != self.obs_dim:
            raise ValueError(f"observations last dimension must be {self.obs_dim}")
        projections = observations @ self.P.T
        return torch.sin(projections / self._nu_value().clamp_min(1e-12) + self.phase)

    def mean(self, observations: torch.Tensor) -> torch.Tensor:
        phi = self.features(observations)
        return phi @ self.W_pi.T + self.b_pi

    def std(self) -> torch.Tensor:
        return torch.exp(self.log_std).clamp_min(1e-8)

    def sample(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean = self.mean(observations)
        if deterministic:
            return mean
        return mean + torch.randn_like(mean) * self.std()

    def log_prob_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean = self.mean(observations)
        actions = torch.as_tensor(actions, dtype=mean.dtype, device=mean.device)
        std = self.std().to(dtype=mean.dtype, device=mean.device)
        var = std.square()
        log_probs = -0.5 * (
            ((actions - mean).square() / var)
            + 2.0 * torch.log(std)
            + math.log(2.0 * math.pi)
        )
        return log_probs.sum(dim=-1)

    def gaussian_kl_from_old(
        self,
        observations: torch.Tensor,
        old_mean: torch.Tensor,
        old_log_std: torch.Tensor,
    ) -> torch.Tensor:
        new_mean = self.mean(observations)
        old_mean = old_mean.to(dtype=new_mean.dtype, device=new_mean.device)
        old_log_std = old_log_std.to(dtype=new_mean.dtype, device=new_mean.device)
        new_log_std = self.log_std.to(dtype=new_mean.dtype, device=new_mean.device)
        old_var = torch.exp(2.0 * old_log_std)
        new_var = torch.exp(2.0 * new_log_std)
        kl = (
            new_log_std
            - old_log_std
            + (old_var + (old_mean - new_mean).square()) / (2.0 * new_var)
            - 0.5
        )
        return kl.sum(dim=-1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mean(observations)
