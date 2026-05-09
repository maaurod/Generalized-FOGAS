from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch

from ..features.continuous_features import (
    ContinuousGaussianPolicyParam,
    ContinuousRhoParam,
    ContinuousValueParam,
)
from ..datasets.continuous_sbeed_dataset import ContinuousSBEEDDataset


class ContinuousSBEED:
    """Multi-step SBEED for continuous Gymnasium-style control problems."""

    _LR_GROUPS = ("value", "rho", "policy")

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        gamma: float,
        value_param: ContinuousValueParam,
        rho_param: ContinuousRhoParam,
        policy_param: ContinuousGaussianPolicyParam,
        action_low: Optional[Union[np.ndarray, torch.Tensor, float]] = None,
        action_high: Optional[Union[np.ndarray, torch.Tensor, float]] = None,
        lambda_entropy: float = 0.01,
        eta: float = 1.0,
        lr_value: float = 1e-3,
        lr_rho: float = 1e-3,
        lr_policy: float = 1e-3,
        lr_schedulers: Optional[Union[str, Iterable[str], Dict[str, Optional[str]]]] = None,
        cosine_t_max: Optional[int] = None,
        cosine_eta_min: Union[float, Dict[str, float]] = 0.0,
        tau: float = 1.0,
        max_buffer_size: int = 12000,
        batch_size: Optional[int] = 256,
        rollout_length: int = 1,
        fisher_damping: float = 1e-3,
        cg_iters: int = 10,
        cg_tol: float = 1e-10,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        seed: Optional[int] = 42,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.value_param = value_param
        self.rho_param = rho_param
        self.policy_param = policy_param
        self.lambda_entropy = float(lambda_entropy)
        self.eta = float(eta)
        self.lr_value = float(lr_value)
        self.lr_rho = float(lr_rho)
        self.lr_policy = float(lr_policy)
        self.lr_schedulers = self._parse_lr_schedulers(lr_schedulers)
        self.cosine_t_max = None if cosine_t_max is None else int(cosine_t_max)
        self.cosine_eta_min = self._parse_cosine_eta_min(cosine_eta_min)
        self.tau = float(tau)
        self.max_buffer_size = int(max_buffer_size)
        self.batch_size = batch_size
        self.rollout_length = int(rollout_length)
        self.fisher_damping = float(fisher_damping)
        self.cg_iters = int(cg_iters)
        self.cg_tol = float(cg_tol)
        if len(adam_betas) != 2:
            raise ValueError("adam_betas must contain exactly two entries")
        self.adam_beta1 = float(adam_betas[0])
        self.adam_beta2 = float(adam_betas[1])
        self.adam_eps = float(adam_eps)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if self.obs_dim <= 0 or self.action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError("gamma must be in [0, 1)")
        if self.lambda_entropy < 0.0:
            raise ValueError("lambda_entropy must be non-negative")
        if not (0.0 <= self.eta <= 1.0):
            raise ValueError("eta must be in [0, 1]")
        if self.lr_value <= 0.0 or self.lr_rho <= 0.0 or self.lr_policy <= 0.0:
            raise ValueError("lr_value, lr_rho, and lr_policy must be positive")
        if self.cosine_t_max is not None and self.cosine_t_max <= 0:
            raise ValueError("cosine_t_max must be positive when provided")
        for name in self._LR_GROUPS:
            eta_min = self.cosine_eta_min[name]
            base_lr = getattr(self, f"lr_{name}")
            if eta_min < 0.0 or eta_min > base_lr:
                raise ValueError(f"cosine_eta_min for {name} must be in [0, lr_{name}]")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive")
        if self.rollout_length <= 0:
            raise ValueError("rollout_length must be positive")
        if self.fisher_damping < 0.0:
            raise ValueError("fisher_damping must be non-negative")
        if self.cg_iters <= 0:
            raise ValueError("cg_iters must be positive")
        if self.cg_tol < 0.0:
            raise ValueError("cg_tol must be non-negative")
        if not (0.0 <= self.adam_beta1 < 1.0 and 0.0 <= self.adam_beta2 < 1.0):
            raise ValueError("adam_betas entries must be in [0, 1)")
        if self.adam_eps <= 0.0:
            raise ValueError("adam_eps must be positive")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._seed_all(seed)
        self.value_param.to(self.device)
        self.rho_param.to(self.device)
        self.policy_param.to(self.device)
        self.action_low = self._action_bound(action_low)
        self.action_high = self._action_bound(action_high)

        self.dataset = ContinuousSBEEDDataset.empty(self.obs_dim, self.action_dim, device=self.device)
        self.dataset.validate(self.obs_dim, self.action_dim)
        self.n = self.dataset.n
        self.update_index = 0
        self.last_episode_returns: list[float] = []
        self._reset_optimizer_state()

    @staticmethod
    def _seed_all(seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    @staticmethod
    def _trainable_params(module: Any) -> list[torch.nn.Parameter]:
        return [p for p in module.parameters() if p.requires_grad]

    @staticmethod
    def _param_dtype(module: Any, fallback: torch.dtype = torch.float32) -> torch.dtype:
        for param in module.parameters():
            return param.dtype
        for buffer in module.buffers():
            return buffer.dtype
        return fallback

    @classmethod
    def _parse_lr_schedulers(
        cls,
        lr_schedulers: Optional[Union[str, Iterable[str], Dict[str, Optional[str]]]],
    ) -> Dict[str, str]:
        schedules = {name: "inverse_time" for name in cls._LR_GROUPS}
        if lr_schedulers is None:
            return schedules
        valid_schedules = {"inverse_time", "cosine", "none"}
        if isinstance(lr_schedulers, str):
            schedule = lr_schedulers.lower()
            if schedule not in valid_schedules:
                raise ValueError("lr_schedulers string must be one of: inverse_time, cosine, none")
            return {name: schedule for name in cls._LR_GROUPS}
        if isinstance(lr_schedulers, dict):
            for name, schedule in lr_schedulers.items():
                if name not in cls._LR_GROUPS:
                    raise ValueError(f"unknown lr scheduler group: {name}")
                schedule_name = "inverse_time" if schedule is None else str(schedule).lower()
                if schedule_name not in valid_schedules:
                    raise ValueError("lr scheduler values must be one of: inverse_time, cosine, none")
                schedules[name] = schedule_name
            return schedules
        for name in lr_schedulers:
            if name not in cls._LR_GROUPS:
                raise ValueError(f"unknown lr scheduler group: {name}")
            schedules[name] = "cosine"
        return schedules

    @classmethod
    def _parse_cosine_eta_min(cls, cosine_eta_min: Union[float, Dict[str, float]]) -> Dict[str, float]:
        if isinstance(cosine_eta_min, dict):
            eta_min = {name: 0.0 for name in cls._LR_GROUPS}
            for name, value in cosine_eta_min.items():
                if name not in cls._LR_GROUPS:
                    raise ValueError(f"unknown cosine_eta_min group: {name}")
                eta_min[name] = float(value)
            return eta_min
        value = float(cosine_eta_min)
        return {name: value for name in cls._LR_GROUPS}

    def _action_bound(self, value: Optional[Union[np.ndarray, torch.Tensor, float]]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.repeat(self.action_dim)
        if tensor.numel() != self.action_dim:
            raise ValueError(f"action bound must be scalar or have length {self.action_dim}")
        if not torch.isfinite(tensor).all():
            return None
        return tensor

    def set_action_bounds(
        self,
        low: Optional[Union[np.ndarray, torch.Tensor, float]],
        high: Optional[Union[np.ndarray, torch.Tensor, float]],
    ) -> None:
        self.action_low = self._action_bound(low)
        self.action_high = self._action_bound(high)

    def _clip_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_low is not None:
            action = torch.maximum(action, self.action_low.to(dtype=action.dtype, device=action.device))
        if self.action_high is not None:
            action = torch.minimum(action, self.action_high.to(dtype=action.dtype, device=action.device))
        return action

    def _reset_optimizer_state(self) -> None:
        self.value_optimizer = None
        self.rho_optimizer = None
        value_params = self._trainable_params(self.value_param)
        rho_params = self._trainable_params(self.rho_param)
        if value_params:
            self.value_optimizer = torch.optim.Adam(
                value_params,
                lr=self.lr_value,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_eps,
            )
        if rho_params:
            self.rho_optimizer = torch.optim.Adam(
                rho_params,
                lr=self.lr_rho,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_eps,
            )

    def _set_optimizer_lr(self, optimizer: Optional[torch.optim.Optimizer], lr: float) -> None:
        if optimizer is None:
            return
        for group in optimizer.param_groups:
            group["lr"] = lr

    def _learning_rate(self, name: str, base_lr: float) -> float:
        schedule = self.lr_schedulers[name]
        if schedule == "none":
            return float(base_lr)
        if schedule == "cosine":
            t_max = self.cosine_t_max or max(1, int(self.tau))
            schedule_step = max(0, int(self.update_index) - 1)
            progress = min(float(schedule_step), float(t_max)) / float(t_max)
            eta_min = self.cosine_eta_min[name]
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(eta_min + (float(base_lr) - eta_min) * cosine)
        decay = 1.0 / (1.0 + float(self.update_index) / self.tau)
        return float(base_lr * decay)

    @staticmethod
    def _parameters_are_finite(params: Iterable[torch.nn.Parameter]) -> bool:
        return all(torch.isfinite(param.detach()).all().item() for param in params)

    @staticmethod
    def _clone_param_data(params: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
        return [param.detach().clone() for param in params]

    @staticmethod
    def _restore_param_data(params: Iterable[torch.nn.Parameter], snapshots: Iterable[torch.Tensor]) -> None:
        for param, snapshot in zip(params, snapshots):
            param.data.copy_(snapshot)

    @staticmethod
    def _reset_optimizer_buffers(optimizer: Optional[torch.optim.Optimizer]) -> None:
        if optimizer is not None:
            optimizer.state.clear()

    def _clamp_policy_distribution(self) -> None:
        log_std = getattr(self.policy_param, "log_std", None)
        if isinstance(log_std, torch.nn.Parameter):
            log_std.data.clamp_(min=-20.0, max=2.0)

    def _finite_action_fallback(self, action: torch.Tensor) -> torch.Tensor:
        fallback = torch.zeros_like(action)
        if self.action_low is not None and self.action_high is not None:
            low = self.action_low.to(dtype=action.dtype, device=action.device)
            high = self.action_high.to(dtype=action.dtype, device=action.device)
            fallback = 0.5 * (low + high)
        return torch.where(torch.isfinite(action), action, fallback)

    def _valid_fragment_starts(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.n == 0:
            raise ValueError("Replay buffer D is empty. Collect data before sampling fragments.")

        if self.rollout_length == 1:
            starts = torch.arange(self.n, dtype=torch.int64, device=self.device)
            lengths = torch.ones(self.n, dtype=torch.int64, device=self.device)
            terminals = self.dataset.D.to(device=self.device, dtype=torch.bool)
            return starts, lengths, terminals

        all_starts = torch.arange(self.n, dtype=torch.int64, device=self.device)
        offsets = torch.arange(self.rollout_length, dtype=torch.int64, device=self.device)
        window_indices = all_starts[:, None] + offsets[None, :]
        in_buffer = window_indices < self.n
        safe_indices = window_indices.clamp_max(max(self.n - 1, 0))
        done_windows = self.dataset.D[safe_indices].to(dtype=torch.bool) & in_buffer
        has_done = done_windows.any(dim=1)
        first_done = torch.argmax(done_windows.to(dtype=torch.int64), dim=1)
        full_length = torch.full_like(all_starts, self.rollout_length)
        tail_length = (self.n - all_starts).clamp_max(self.rollout_length)
        lengths = torch.where(has_done, first_done + 1, torch.minimum(full_length, tail_length))
        terminals = has_done
        valid = (tail_length >= self.rollout_length) | has_done
        starts = all_starts[valid]
        lengths = lengths[valid]
        terminals = terminals[valid]
        if starts.numel() == 0:
            raise ValueError(
                "Replay buffer D has no valid multi-step fragments. "
                "Collect at least rollout_length transitions or a terminal fragment."
            )
        return starts, lengths, terminals

    def _batch_fragment_starts(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        starts, lengths, terminals = self._valid_fragment_starts()
        valid_n = int(starts.numel())
        if self.batch_size is None or self.batch_size >= valid_n:
            return starts, lengths, terminals
        sample_positions = torch.randint(valid_n, (int(self.batch_size),), device=self.device)
        return starts[sample_positions], lengths[sample_positions], terminals[sample_positions]

    def _fragment_batch(
        self,
        starts: torch.Tensor,
        lengths: torch.Tensor,
        terminals: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        offsets = torch.arange(self.rollout_length, dtype=torch.int64, device=self.device)
        fragment_indices = starts[:, None] + offsets[None, :]
        safe_indices = fragment_indices.clamp_max(max(self.n - 1, 0))
        mask = offsets[None, :] < lengths[:, None]
        discounts = torch.pow(
            torch.tensor(self.gamma, dtype=torch.float32, device=self.device),
            offsets.to(dtype=torch.float32),
        )
        weights = mask.to(dtype=torch.float32) * discounts[None, :]
        bootstrap_indices = starts + lengths - 1
        bootstrap_states = self.dataset.X_next[bootstrap_indices]
        bootstrap_discounts = torch.pow(
            torch.tensor(self.gamma, dtype=torch.float32, device=self.device),
            lengths.to(dtype=torch.float32),
        )
        nonterminal = (~terminals).to(dtype=torch.float32)
        return {
            "starts": starts,
            "lengths": lengths,
            "terminals": terminals,
            "X0": self.dataset.X[starts],
            "X_steps": self.dataset.X[safe_indices],
            "A_steps": self.dataset.A[safe_indices],
            "R_steps": self.dataset.R[safe_indices],
            "weights": weights,
            "mask": mask,
            "bootstrap_states": bootstrap_states,
            "bootstrap_weight": nonterminal * bootstrap_discounts,
        }

    def _discounted_rewards(self, batch: Dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        weights = batch["weights"].to(dtype=dtype)
        rewards = batch["R_steps"].to(dtype=dtype)
        return (weights * rewards).sum(dim=1)

    def _weighted_policy_log_probs(
        self,
        batch: Dict[str, torch.Tensor],
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        X_steps = batch["X_steps"]
        A_steps = batch["A_steps"]
        weights = batch["weights"]
        flat_log_probs = self.policy_param.log_prob_actions(
            X_steps.reshape(-1, self.obs_dim),
            A_steps.reshape(-1, self.action_dim),
        )
        log_probs = flat_log_probs.reshape_as(weights)
        if dtype is None:
            dtype = log_probs.dtype
        return (weights.to(dtype=dtype) * log_probs.to(dtype=dtype)).sum(dim=1)

    def _bootstrap_values(self, batch: Dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        values = self.value_param.value(batch["bootstrap_states"]).to(dtype=dtype)
        weights = batch["bootstrap_weight"].to(dtype=dtype)
        return weights * values

    def _rho_values(self, batch: Dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        return self.rho_param.rho(
            batch["X_steps"].to(dtype=dtype),
            batch["A_steps"].to(dtype=dtype),
            batch["weights"].to(dtype=dtype),
        ).to(dtype=dtype)

    def _target_delta_no_grad(self, batch: Dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        with torch.no_grad():
            rewards = self._discounted_rewards(batch, dtype)
            log_pi = self._weighted_policy_log_probs(batch, dtype=dtype)
            bootstrap = self._bootstrap_values(batch, dtype)
            return rewards - self.lambda_entropy * log_pi + bootstrap

    def _flat_grad_norm(self, params: Iterable[torch.nn.Parameter]) -> float:
        sq_norm = 0.0
        for param in params:
            if param.grad is not None:
                sq_norm += float(torch.sum(param.grad.detach() * param.grad.detach()).item())
        return float(sq_norm ** 0.5)

    def _value_update(self, batch: Dict[str, torch.Tensor], step_size: float) -> float:
        if self.value_optimizer is None:
            return 0.0
        params = self._trainable_params(self.value_param)
        dtype = self._param_dtype(self.value_param)
        with torch.no_grad():
            rho = self._rho_values(batch, dtype)
            log_pi = self._weighted_policy_log_probs(batch, dtype=dtype)
        rewards = self._discounted_rewards(batch, dtype)
        V0 = self.value_param.value(batch["X0"].to(dtype=dtype)).to(dtype=dtype)
        bootstrap = self._bootstrap_values(batch, dtype)
        delta = rewards - self.lambda_entropy * log_pi + bootstrap
        loss = ((delta - V0) ** 2).mean() - self.eta * ((delta - rho) ** 2).mean()
        if not torch.isfinite(loss):
            return float("nan")
        snapshots = self._clone_param_data(params)
        self._set_optimizer_lr(self.value_optimizer, step_size)
        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._flat_grad_norm(params)
        if not math.isfinite(grad_norm):
            self.value_optimizer.zero_grad(set_to_none=True)
            return grad_norm
        self.value_optimizer.step()
        if not self._parameters_are_finite(params):
            self._restore_param_data(params, snapshots)
            self._reset_optimizer_buffers(self.value_optimizer)
            return float("nan")
        return grad_norm

    def _rho_update(self, batch: Dict[str, torch.Tensor], step_size: float) -> float:
        if self.rho_optimizer is None:
            return 0.0
        params = self._trainable_params(self.rho_param)
        dtype = self._param_dtype(self.rho_param)
        target = self._target_delta_no_grad(batch, dtype)
        rho = self._rho_values(batch, dtype)
        loss = 0.5 * ((target - rho) ** 2).mean()
        if not torch.isfinite(loss):
            return float("nan")
        snapshots = self._clone_param_data(params)
        self._set_optimizer_lr(self.rho_optimizer, step_size)
        self.rho_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._flat_grad_norm(params)
        if not math.isfinite(grad_norm):
            self.rho_optimizer.zero_grad(set_to_none=True)
            return grad_norm
        self.rho_optimizer.step()
        if not self._parameters_are_finite(params):
            self._restore_param_data(params, snapshots)
            self._reset_optimizer_buffers(self.rho_optimizer)
            return float("nan")
        return grad_norm

    def _policy_loss_and_grad(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        dtype = self._param_dtype(self.policy_param)
        with torch.no_grad():
            rewards = self._discounted_rewards(batch, dtype)
            log_pi_detached = self._weighted_policy_log_probs(batch, dtype=dtype)
            bootstrap = self._bootstrap_values(batch, dtype)
            delta = rewards - self.lambda_entropy * log_pi_detached + bootstrap
            V0 = self.value_param.value(batch["X0"].to(dtype=dtype)).to(dtype=dtype)
            rho = self._rho_values(batch, dtype)
            advantage = ((1.0 - self.eta) * delta + self.eta * rho - V0).detach()

        params = self._trainable_params(self.policy_param)
        if not params:
            return torch.empty(0, dtype=dtype, device=self.device), 0.0
        log_pi = self._weighted_policy_log_probs(batch, dtype=dtype)
        loss = -2.0 * self.lambda_entropy * (advantage * log_pi).mean()
        if not torch.isfinite(loss):
            return torch.empty(0, dtype=dtype, device=self.device), float("nan")
        grads = torch.autograd.grad(loss, params)
        grad_flat = torch.cat([g.reshape(-1) for g in grads]).detach()
        return grad_flat, float(torch.linalg.norm(grad_flat).item())

    def _policy_kl_hvp(
        self,
        vector: torch.Tensor,
        observations: torch.Tensor,
        params: list[torch.nn.Parameter],
    ) -> torch.Tensor:
        dtype = self._param_dtype(self.policy_param)
        observations = observations.to(dtype=dtype)
        with torch.no_grad():
            old_mean = self.policy_param.mean(observations).detach()
            old_log_std = self.policy_param.log_std.detach().clone()
        empirical_kl = self.policy_param.gaussian_kl_from_old(
            observations,
            old_mean,
            old_log_std,
        ).mean()
        grad_kl = torch.autograd.grad(empirical_kl, params, create_graph=True)
        grad_flat = torch.cat([g.reshape(-1) for g in grad_kl])
        grad_vector_product = torch.dot(grad_flat, vector.detach())
        hvp = torch.autograd.grad(grad_vector_product, params, retain_graph=False)
        return torch.cat([h.reshape(-1) for h in hvp]).detach()

    def _conjugate_gradient(
        self,
        b: torch.Tensor,
        observations: torch.Tensor,
        params: list[torch.nn.Parameter],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        def matvec(v: torch.Tensor) -> torch.Tensor:
            return self._policy_kl_hvp(v, observations, params) + self.fisher_damping * v

        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)
        if torch.sqrt(rs_old) <= self.cg_tol:
            return x, {
                "cg_iters_used": 0,
                "cg_residual_norm": float(torch.sqrt(rs_old).item()),
                "cg_relative_residual": 0.0,
            }

        iters_used = 0
        for _ in range(self.cg_iters):
            Ap = matvec(p)
            alpha = rs_old / torch.dot(p, Ap).clamp_min(1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            iters_used += 1
            if torch.sqrt(rs_new) <= self.cg_tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        residual_norm = torch.linalg.norm(matvec(x) - b)
        b_norm = torch.linalg.norm(b)
        return x, {
            "cg_iters_used": int(iters_used),
            "cg_residual_norm": float(residual_norm.item()),
            "cg_relative_residual": float((residual_norm / b_norm.clamp_min(1e-30)).item()),
        }

    @staticmethod
    def _apply_flat_direction(
        params: list[torch.nn.Parameter],
        direction: torch.Tensor,
        step_size: float,
    ) -> None:
        offset = 0
        for param in params:
            n = param.numel()
            param.data -= step_size * direction[offset:offset + n].reshape_as(param)
            offset += n

    def _policy_update(
        self,
        batch: Dict[str, torch.Tensor],
        step_size: float,
    ) -> Tuple[float, float, Dict[str, float]]:
        grad_flat, grad_norm = self._policy_loss_and_grad(batch)
        params = self._trainable_params(self.policy_param)
        if not params or grad_flat.numel() == 0:
            return grad_norm, 0.0, {
                "policy_direction": "cg_fisher",
                "cg_iters_used": 0,
                "cg_residual_norm": 0.0,
                "cg_relative_residual": 0.0,
            }
        if not torch.isfinite(grad_flat).all():
            return grad_norm, 0.0, {
                "policy_direction": "skipped_nonfinite_grad",
                "cg_iters_used": 0,
                "cg_residual_norm": float("nan"),
                "cg_relative_residual": float("nan"),
            }
        snapshots = self._clone_param_data(params)
        direction, diagnostics = self._conjugate_gradient(grad_flat, batch["X0"], params)
        if not torch.isfinite(direction).all():
            diagnostics["policy_direction"] = "skipped_nonfinite_direction"
            return grad_norm, float("nan"), diagnostics
        self._apply_flat_direction(params, direction, step_size)
        self._clamp_policy_distribution()
        if not self._parameters_are_finite(params):
            self._restore_param_data(params, snapshots)
            diagnostics["policy_direction"] = "skipped_nonfinite_params"
            return grad_norm, float("nan"), diagnostics
        diagnostics["policy_direction"] = "cg_fisher"
        return grad_norm, float(torch.linalg.norm(direction).item()), diagnostics

    def step(self) -> Dict[str, float]:
        if self.n == 0:
            raise ValueError("Replay buffer D is empty. Collect data before calling step().")
        starts, lengths, terminals = self._batch_fragment_starts()
        batch = self._fragment_batch(starts, lengths, terminals)

        self.update_index += 1
        value_step_size = self._learning_rate("value", self.lr_value)
        rho_step_size = self._learning_rate("rho", self.lr_rho)
        policy_step_size = self._learning_rate("policy", self.lr_policy)

        beta_grad_norm = self._rho_update(batch, rho_step_size)
        theta_grad_norm = self._value_update(batch, value_step_size)
        policy_grad_norm, policy_direction_norm, policy_diagnostics = self._policy_update(
            batch,
            policy_step_size,
        )

        stats = self.objective()
        stats.update(
            {
                "theta_grad_norm": float(theta_grad_norm),
                "beta_grad_norm": float(beta_grad_norm),
                "policy_grad_norm": float(policy_grad_norm),
                "policy_direction_norm": float(policy_direction_norm),
                "value_step_size": float(value_step_size),
                "rho_step_size": float(rho_step_size),
                "policy_step_size": float(policy_step_size),
                "value_optimizer": "adam",
                "rho_optimizer": "adam",
                "policy_optimizer": "npg_cg",
                "value_lr_scheduler": self.lr_schedulers["value"],
                "rho_lr_scheduler": self.lr_schedulers["rho"],
                "policy_lr_scheduler": self.lr_schedulers["policy"],
                "rollout_length": int(self.rollout_length),
                "mean_fragment_length": float(lengths.to(dtype=torch.float32).mean().item()),
                "terminal_fragment_fraction": float(terminals.to(dtype=torch.float32).mean().item()),
                "update_index": int(self.update_index),
            }
        )
        stats.update(policy_diagnostics)
        return stats

    def objective(self) -> Dict[str, float]:
        if self.n == 0:
            raise ValueError("Replay buffer D is empty. Cannot compute objective.")
        with torch.no_grad():
            starts, lengths, terminals = self._valid_fragment_starts()
            batch = self._fragment_batch(starts, lengths, terminals)
            dtype = self._param_dtype(self.value_param)
            rewards = self._discounted_rewards(batch, dtype)
            log_pi = self._weighted_policy_log_probs(batch, dtype=dtype)
            bootstrap = self._bootstrap_values(batch, dtype)
            delta = rewards - self.lambda_entropy * log_pi + bootstrap
            V0 = self.value_param.value(batch["X0"].to(dtype=dtype)).to(dtype=dtype)
            rho = self._rho_values(batch, dtype)
            primal = torch.mean((delta - V0) ** 2)
            dual = torch.mean((delta - rho) ** 2)
            return {
                "objective": float((primal - self.eta * dual).item()),
                "primal_mse": float(primal.item()),
                "dual_mse": float(dual.item()),
            }

    def _obs_tensor(self, observation: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        obs = torch.as_tensor(observation, dtype=self._param_dtype(self.policy_param), device=self.device).reshape(1, -1)
        if obs.shape[1] != self.obs_dim:
            raise ValueError(f"observation must have shape ({self.obs_dim},)")
        return obs

    def sample_action(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
        clip: bool = True,
    ) -> np.ndarray:
        with torch.no_grad():
            action = self.policy_param.sample(self._obs_tensor(observation), deterministic=deterministic).squeeze(0)
            action = self._finite_action_fallback(action)
            if clip:
                action = self._clip_action_tensor(action)
                action = self._finite_action_fallback(action)
        return action.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _parse_env_step(result: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not isinstance(result, tuple):
            raise ValueError("Gymnasium env.step must return a tuple")
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated) or bool(truncated), dict(info)
        if len(result) == 4:
            obs, reward, done, info = result
            return np.asarray(obs, dtype=np.float32), float(reward), bool(done), dict(info)
        raise ValueError("Unsupported env.step return signature")

    @staticmethod
    def _parse_env_reset(result: Any) -> np.ndarray:
        if isinstance(result, tuple):
            return np.asarray(result[0], dtype=np.float32)
        return np.asarray(result, dtype=np.float32)

    def _maybe_estimate_policy_nu(self) -> None:
        if hasattr(self.policy_param, "nu_is_set") and not self.policy_param.nu_is_set and self.n >= 2:
            self.policy_param.set_nu_from_observations(self.dataset.X.detach().cpu())

    def _collect_env_steps(
        self,
        env,
        observation: np.ndarray,
        n_steps: int,
        random_actions: bool = False,
        deterministic: bool = False,
        render: bool = False,
    ) -> Tuple[np.ndarray, list[float]]:
        episode_returns: list[float] = []
        current_return = 0.0
        obs = np.asarray(observation, dtype=np.float32)
        if not np.isfinite(obs).all():
            raise ValueError("initial environment observation must be finite")
        for _ in range(int(n_steps)):
            if render:
                env.render()
            if random_actions:
                action = np.asarray(env.action_space.sample(), dtype=np.float32)
            else:
                action = self.sample_action(obs, deterministic=deterministic, clip=True)
            if not np.isfinite(action).all():
                raise ValueError("sampled action must be finite")
            next_obs, reward, done, _ = self._parse_env_step(env.step(action))
            if not np.isfinite(next_obs).all():
                next_obs = self._parse_env_reset(env.reset())
                done = True
            if not math.isfinite(float(reward)):
                reward = 0.0
            self.dataset.append_fifo(
                obs,
                action,
                reward,
                next_obs,
                capacity=self.max_buffer_size,
                done=done,
            )
            current_return += float(reward)
            if done:
                episode_returns.append(current_return)
                current_return = 0.0
                obs = self._parse_env_reset(env.reset())
            else:
                obs = next_obs
        self.dataset.validate(self.obs_dim, self.action_dim)
        self.n = self.dataset.n
        return obs, episode_returns

    def run_env(
        self,
        env,
        episodes: int = 100,
        max_episode_steps: Optional[int] = None,
        initial_random_steps: int = 1000,
        collect_per_episode: int = 1000,
        updates_per_episode: int = 100,
        deterministic: bool = False,
        render: bool = False,
        log_every: int = 10,
    ) -> Dict[str, Any]:
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError("ContinuousSBEED.run_env requires gymnasium") from exc

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("ContinuousSBEED.run_env requires a Box observation_space")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("ContinuousSBEED.run_env requires a Box action_space")
        if int(np.prod(env.observation_space.shape)) != self.obs_dim:
            raise ValueError("env observation dimension does not match obs_dim")
        if int(np.prod(env.action_space.shape)) != self.action_dim:
            raise ValueError("env action dimension does not match action_dim")

        self.set_action_bounds(env.action_space.low, env.action_space.high)
        if self.seed is not None:
            env.action_space.seed(self.seed)
            env.observation_space.seed(self.seed)
        self.dataset = ContinuousSBEEDDataset.empty(self.obs_dim, self.action_dim, device=self.device)
        self.n = self.dataset.n
        self.update_index = 0
        self.last_episode_returns = []
        self._reset_optimizer_state()

        reset_kwargs = {"seed": self.seed} if self.seed is not None else {}
        obs = self._parse_env_reset(env.reset(**reset_kwargs))
        if initial_random_steps > 0:
            obs, returns = self._collect_env_steps(
                env,
                obs,
                n_steps=initial_random_steps,
                random_actions=True,
                render=render,
            )
            self.last_episode_returns.extend(returns)
        self._maybe_estimate_policy_nu()

        last_stats = None
        steps_per_episode = int(collect_per_episode if max_episode_steps is None else min(collect_per_episode, max_episode_steps))
        for episode in range(int(episodes)):
            obs, returns = self._collect_env_steps(
                env,
                obs,
                n_steps=steps_per_episode,
                random_actions=False,
                deterministic=deterministic,
                render=render,
            )
            self.last_episode_returns.extend(returns)
            self._maybe_estimate_policy_nu()

            if self.n > 0:
                for _ in range(int(updates_per_episode)):
                    last_stats = self.step()

            if last_stats is not None and log_every > 0 and episode % int(log_every) == 0:
                recent_returns = self.last_episode_returns[-10:]
                avg_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
                print(
                    f"episode={episode}/{episodes} "
                    f"buffer={self.n} "
                    f"objective={last_stats['objective']:.6f} "
                    f"primal_mse={last_stats['primal_mse']:.6f} "
                    f"dual_mse={last_stats['dual_mse']:.6f} "
                    f"avg_return_10={avg_return:.3f}"
                )

        return {
            "last_stats": last_stats,
            "episode_returns": list(self.last_episode_returns),
            "buffer_size": int(self.n),
        }

    def value(self, observation: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        obs = self._obs_tensor(observation).to(dtype=self._param_dtype(self.value_param))
        return self.value_param.value(obs).squeeze(0)

    def rho(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        obs = self._obs_tensor(observation).to(dtype=self._param_dtype(self.rho_param)).reshape(1, 1, self.obs_dim)
        act = torch.as_tensor(action, dtype=self._param_dtype(self.rho_param), device=self.device).reshape(1, 1, self.action_dim)
        weights = torch.ones((1, 1), dtype=self._param_dtype(self.rho_param), device=self.device)
        return self.rho_param.rho(obs, act, weights).squeeze(0)
