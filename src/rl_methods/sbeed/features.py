from __future__ import annotations

from typing import Callable, Optional, Union

import torch

TensorLike = Union[torch.Tensor, list, tuple]


class TabularStateFeatures:
    """One-hot state features phi(s)."""

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
    """One-hot state-action features rho_features(s, a)."""

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
    """Action-coupled RBF features zeta(s, a) = e_a kron phi(s)."""

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
