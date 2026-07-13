"""Final discrete SBEED solver.

`DiscreteSBEED` is the implementation intended for new finite-MDP experiments.
It keeps the stable multi-step update order from the staged solvers:

    1. update rho to fit the multi-step smoothed backup,
    2. update V by minimizing the SBEED primal objective,
    3. update pi with an implicit natural policy-gradient step.

The class accepts parametrization modules instead of raw feature callables.
Linear modules use manual fast-path gradients for reproducibility with the
building versions; neural modules use PyTorch autograd and Adam.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch

from ..features.discrete_features import (
    LinearRhoParam,
    LinearValueParam,
    PolicyParam,
    RhoParam,
    SoftmaxLinearPolicyParam,
    ValueParam,
)
from ..datasets.discrete_sbeed_dataset import DiscreteSBEEDDataset


class DiscreteSBEED:
    """
    Generalized multi-step SBEED over explicit PyTorch parametrizations.

    Components can be mixed independently:
        - linear value/rho/policy modules use manual fast-path gradients,
        - nonlinear value/rho modules use Adam and autograd,
        - every policy uses an implicit CG Fisher/NPG update.

    Use this class when the environment has explicit finite state/action ids.
    Use `ContinuousSBEED` for Gymnasium-style continuous observations/actions.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float,
        value_param: ValueParam,
        rho_param: RhoParam,
        policy_param: PolicyParam,
        lambda_entropy: float = 0.01,
        eta: float = 1.0,
        lr_value: float = 1e-2,
        lr_rho: float = 1e-2,
        lr_policy: float = 1e-2,
        tau: float = 1.0,
        max_buffer_size: int = 12000,
        batch_size: Optional[int] = None,
        rollout_length: int = 1,
        fisher_damping: float = 1e-3,
        cg_iters: int = 10,
        cg_tol: float = 1e-10,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        seed: Optional[int] = 42,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.value_param = value_param
        self.rho_param = rho_param
        self.policy_param = policy_param
        self.lambda_entropy = float(lambda_entropy)
        self.eta = float(eta)
        self.lr_value = float(lr_value)
        self.lr_rho = float(lr_rho)
        self.lr_policy = float(lr_policy)
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

        if self.n_states <= 0:
            raise ValueError("n_states must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError("gamma must be in [0, 1)")
        if self.lambda_entropy < 0.0:
            raise ValueError("lambda_entropy must be non-negative")
        if not (0.0 <= self.eta <= 1.0):
            raise ValueError("eta must be in [0, 1]")
        if self.lr_value <= 0.0 or self.lr_rho <= 0.0 or self.lr_policy <= 0.0:
            raise ValueError("lr_value, lr_rho, and lr_policy must be positive")
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

        self.dataset = DiscreteSBEEDDataset.empty(device=self.device)
        self.dataset.validate(self.n_states, self.n_actions)
        self.n = self.dataset.n
        self.pi: Optional[torch.Tensor] = None
        self.update_index = 0
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
    def _param_dtype(module: Any, fallback: torch.dtype = torch.float64) -> torch.dtype:
        for param in module.parameters():
            return param.dtype
        for buffer in module.buffers():
            return buffer.dtype
        return fallback

    def _reset_optimizer_state(self) -> None:
        self._manual_adam_state: Dict[str, Dict[str, Any]] = {}
        self.value_optimizer = None
        self.rho_optimizer = None
        if not self.value_param.is_linear_fast_path:
            params = self._trainable_params(self.value_param)
            if params:
                self.value_optimizer = torch.optim.Adam(
                    params,
                    lr=self.lr_value,
                    betas=(self.adam_beta1, self.adam_beta2),
                    eps=self.adam_eps,
                )
        if not self.rho_param.is_linear_fast_path:
            params = self._trainable_params(self.rho_param)
            if params:
                self.rho_optimizer = torch.optim.Adam(
                    params,
                    lr=self.lr_rho,
                    betas=(self.adam_beta1, self.adam_beta2),
                    eps=self.adam_eps,
                )

    def _manual_adam_update(
        self,
        name: str,
        param: torch.nn.Parameter,
        grad: torch.Tensor,
        step_size: float,
    ) -> None:
        state = self._manual_adam_state.get(name)
        if state is None:
            state = {
                "m": torch.zeros_like(param.data),
                "v": torch.zeros_like(param.data),
                "t": 0,
            }
            self._manual_adam_state[name] = state
        state["t"] += 1
        state["m"] = self.adam_beta1 * state["m"] + (1.0 - self.adam_beta1) * grad
        state["v"] = self.adam_beta2 * state["v"] + (1.0 - self.adam_beta2) * (grad * grad)
        bias_correction1 = 1.0 - self.adam_beta1 ** state["t"]
        bias_correction2 = 1.0 - self.adam_beta2 ** state["t"]
        m_hat = state["m"] / bias_correction1
        v_hat = state["v"] / bias_correction2
        param.data -= step_size * m_hat / (torch.sqrt(v_hat) + self.adam_eps)

    def _set_optimizer_lr(self, optimizer: Optional[torch.optim.Optimizer], lr: float) -> None:
        if optimizer is None:
            return
        for group in optimizer.param_groups:
            group["lr"] = lr

    def _valid_fragment_starts(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find replay indices that can start valid multi-step fragments.

        A fragment is valid if it has `rollout_length` transitions available or
        if it hits a terminal transition earlier. This prevents a sampled target
        from reading across an episode boundary or bootstrapping after a done.
        """
        if self.n == 0:
            raise ValueError("Replay buffer D is empty. Collect data before sampling fragments.")

        if self.rollout_length == 1:
            starts = torch.arange(self.n, dtype=torch.int64, device=self.device)
            lengths = torch.ones(self.n, dtype=torch.int64, device=self.device)
            terminals = self.dataset.D.to(device=self.device, dtype=torch.bool)
            return starts, lengths, terminals

        # For multi-step targets, scan every possible replay window and keep
        # only starts that either have a full horizon or terminate naturally.
        all_starts = torch.arange(self.n, dtype=torch.int64, device=self.device)
        offsets = torch.arange(self.rollout_length, dtype=torch.int64, device=self.device)
        window_indices = all_starts[:, None] + offsets[None, :]
        in_buffer = window_indices < self.n
        safe_indices = window_indices.clamp_max(max(self.n - 1, 0))
        done_windows = self.dataset.D[safe_indices].to(dtype=torch.bool) & in_buffer

        has_terminal = done_windows.any(dim=1)
        first_terminal_offsets = done_windows.to(dtype=torch.int64).argmax(dim=1)
        full_fragment_available = all_starts + self.rollout_length <= self.n
        valid = has_terminal | full_fragment_available
        if not bool(valid.any().item()):
            raise ValueError(
                "Replay buffer D has no valid multi-step fragments. "
                "Collect at least rollout_length transitions or a terminal fragment."
            )

        starts = all_starts[valid]
        terminals = has_terminal[valid]
        lengths = torch.where(
            terminals,
            first_terminal_offsets[valid] + 1,
            torch.full_like(starts, self.rollout_length),
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
        """Materialize a batch of padded fragments plus masks and discounts."""
        # Padding keeps every fragment in a rectangular tensor; `mask` and
        # `weights` ensure only realized transitions contribute to the target.
        offsets = torch.arange(self.rollout_length, dtype=torch.int64, device=self.device)
        fragment_indices = starts[:, None] + offsets[None, :]
        safe_indices = fragment_indices.clamp_max(max(self.n - 1, 0))
        mask = offsets[None, :] < lengths[:, None]
        discounts = torch.pow(
            torch.tensor(self.gamma, dtype=torch.float64, device=self.device),
            offsets.to(dtype=torch.float64),
        )
        weights = mask.to(dtype=torch.float64) * discounts[None, :]
        bootstrap_indices = starts + lengths - 1
        bootstrap_states = self.dataset.X_next[bootstrap_indices].long()
        bootstrap_discounts = torch.pow(
            torch.tensor(self.gamma, dtype=torch.float64, device=self.device),
            lengths.to(dtype=torch.float64),
        )
        nonterminal = (~terminals).to(dtype=torch.float64)

        return {
            "starts": starts,
            "lengths": lengths,
            "terminals": terminals,
            "X0": self.dataset.X[starts].long(),
            "X_steps": self.dataset.X[safe_indices].long(),
            "A_steps": self.dataset.A[safe_indices].long(),
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
            X_steps.reshape(-1),
            A_steps.reshape(-1),
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
            batch["X_steps"],
            batch["A_steps"],
            batch["weights"].to(dtype=dtype),
        ).to(dtype=dtype)

    def _target_delta_no_grad(self, batch: Dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        """Compute multi-step targets while freezing current V and policy."""
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

    def _linear_value_update(self, batch: Dict[str, torch.Tensor], step_size: float) -> float:
        """Manual Adam update for the linear value fast path."""
        if not isinstance(self.value_param, LinearValueParam):
            raise TypeError("Linear value fast path requires LinearValueParam")
        dtype = self.value_param.theta.dtype
        with torch.no_grad():
            X0 = batch["X0"]
            phi0 = self.value_param.features(X0)
            bootstrap_phi = (
                batch["bootstrap_weight"].to(dtype=dtype)[:, None]
                * self.value_param.features(batch["bootstrap_states"])
            )
            rewards = self._discounted_rewards(batch, dtype)
            log_pi = self._weighted_policy_log_probs(batch, dtype=dtype)
            delta = rewards - self.lambda_entropy * log_pi + bootstrap_phi @ self.value_param.theta
            V0 = phi0 @ self.value_param.theta
            rho = self._rho_values(batch, dtype)
            residual_v = delta - V0
            residual_rho = delta - rho
            grad_theta = (
                2.0 * (residual_v[:, None] * (bootstrap_phi - phi0)).mean(dim=0)
                - 2.0 * self.eta * (residual_rho[:, None] * bootstrap_phi).mean(dim=0)
            )
        self._manual_adam_update("value.theta", self.value_param.theta, grad_theta, step_size)
        return float(torch.linalg.norm(grad_theta).item())

    def _nonlinear_value_update(self, batch: Dict[str, torch.Tensor], step_size: float) -> float:
        """Autograd Adam update for neural value modules."""
        if self.value_optimizer is None:
            return 0.0
        dtype = self._param_dtype(self.value_param, fallback=torch.float32)
        with torch.no_grad():
            rho = self._rho_values(batch, dtype)
            log_pi = self._weighted_policy_log_probs(batch, dtype=dtype)
        rewards = self._discounted_rewards(batch, dtype)
        V0 = self.value_param.value(batch["X0"]).to(dtype=dtype)
        bootstrap = self._bootstrap_values(batch, dtype)
        delta = rewards - self.lambda_entropy * log_pi + bootstrap

        loss = ((delta - V0) ** 2).mean() - self.eta * ((delta - rho) ** 2).mean()
        self._set_optimizer_lr(self.value_optimizer, step_size)
        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._flat_grad_norm(self._trainable_params(self.value_param))
        self.value_optimizer.step()
        return grad_norm

    def _linear_rho_update(self, batch: Dict[str, torch.Tensor], step_size: float) -> float:
        """Manual Adam update for linear rho regression to target delta."""
        if not isinstance(self.rho_param, LinearRhoParam):
            raise TypeError("Linear rho fast path requires LinearRhoParam")
        dtype = self.rho_param.beta.dtype
        with torch.no_grad():
            target = self._target_delta_no_grad(batch, dtype)
            features = self.rho_param.fragment_features(
                batch["X_steps"],
                batch["A_steps"],
                batch["weights"].to(dtype=dtype),
            )
            rho = features @ self.rho_param.beta
            grad_beta = -((target - rho)[:, None] * features).mean(dim=0)
        self._manual_adam_update("rho.beta", self.rho_param.beta, grad_beta, step_size)
        return float(torch.linalg.norm(grad_beta).item())

    def _nonlinear_rho_update(self, batch: Dict[str, torch.Tensor], step_size: float) -> float:
        """Autograd Adam update for neural rho modules."""
        if self.rho_optimizer is None:
            return 0.0
        dtype = self._param_dtype(self.rho_param, fallback=torch.float32)
        target = self._target_delta_no_grad(batch, dtype)
        rho = self._rho_values(batch, dtype)
        loss = 0.5 * ((target - rho) ** 2).mean()
        self._set_optimizer_lr(self.rho_optimizer, step_size)
        self.rho_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._flat_grad_norm(self._trainable_params(self.rho_param))
        self.rho_optimizer.step()
        return grad_norm

    def _linear_policy_grad(
        self,
        batch: Dict[str, torch.Tensor],
        advantage: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(self.policy_param, SoftmaxLinearPolicyParam):
            raise TypeError("Linear policy fast path requires SoftmaxLinearPolicyParam")
        dtype = self.policy_param.W.dtype
        X_steps = batch["X_steps"]
        A_steps = batch["A_steps"]
        weights = batch["weights"].to(dtype=dtype)
        probs = self.policy_param.probs(X_steps.reshape(-1)).reshape(
            X_steps.shape[0],
            X_steps.shape[1],
            self.n_actions,
        )
        features = self.policy_param.features(X_steps)
        action_one_hot = torch.nn.functional.one_hot(
            A_steps,
            num_classes=self.n_actions,
        ).to(dtype=dtype, device=self.device)
        grad_log_pi_steps = (
            (action_one_hot - probs)[:, :, :, None]
            * features[:, :, None, :]
        )
        grad_log_pi_sum = (weights[:, :, None, None] * grad_log_pi_steps).sum(dim=1)
        return -2.0 * (advantage.to(dtype=dtype)[:, None, None] * grad_log_pi_sum).mean(dim=0)

    def _policy_loss_and_grad(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Build the SBEED policy gradient before Fisher preconditioning.

        The advantage-like term is `(1 - eta) * delta + eta * rho - V`.
        It is detached so the policy step differentiates only log pi.
        """
        dtype = self._param_dtype(self.policy_param, fallback=torch.float32)
        with torch.no_grad():
            rewards = self._discounted_rewards(batch, dtype)
            log_pi_detached = self._weighted_policy_log_probs(batch, dtype=dtype)
            bootstrap = self._bootstrap_values(batch, dtype)
            delta = rewards - self.lambda_entropy * log_pi_detached + bootstrap
            V0 = self.value_param.value(batch["X0"]).to(dtype=dtype)
            rho = self._rho_values(batch, dtype)
            advantage = ((1.0 - self.eta) * delta + self.eta * rho - V0).detach()

        if isinstance(self.policy_param, SoftmaxLinearPolicyParam):
            grad = self._linear_policy_grad(batch, advantage)
            return grad.reshape(-1).detach(), grad, float(torch.linalg.norm(grad).item())

        params = self._trainable_params(self.policy_param)
        if not params:
            return torch.empty(0, dtype=dtype, device=self.device), torch.empty(0, dtype=dtype, device=self.device), 0.0
        log_pi = self._weighted_policy_log_probs(batch, dtype=dtype)
        loss = -2.0 * (advantage * log_pi).mean()
        grads = torch.autograd.grad(loss, params)
        grad_flat = torch.cat([g.reshape(-1) for g in grads]).detach()
        return grad_flat, grad_flat, float(torch.linalg.norm(grad_flat).item())

    def _policy_kl_hvp(
        self,
        vector: torch.Tensor,
        state_indices: torch.Tensor,
        params: list[torch.nn.Parameter],
    ) -> torch.Tensor:
        """Fisher-vector product via a categorical-policy KL Hessian."""
        dtype = self._param_dtype(self.policy_param, fallback=torch.float32)
        with torch.no_grad():
            old_probs = self.policy_param.probs(state_indices).to(dtype=dtype).clamp_min(1e-12)
        new_probs = self.policy_param.probs(state_indices).to(dtype=dtype).clamp_min(1e-12)
        empirical_kl = (
            old_probs * (torch.log(old_probs) - torch.log(new_probs))
        ).sum(dim=1).mean()
        grad_kl = torch.autograd.grad(empirical_kl, params, create_graph=True)
        grad_flat = torch.cat([g.reshape(-1) for g in grad_kl])
        grad_vector_product = torch.dot(grad_flat, vector.detach())
        hvp = torch.autograd.grad(grad_vector_product, params, retain_graph=False)
        return torch.cat([h.reshape(-1) for h in hvp]).detach()

    def _conjugate_gradient(
        self,
        b: torch.Tensor,
        state_indices: torch.Tensor,
        params: list[torch.nn.Parameter],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Approximately solve `(F + damping I) direction = policy_grad`."""
        def matvec(v: torch.Tensor) -> torch.Tensor:
            return self._policy_kl_hvp(v, state_indices, params) + self.fisher_damping * v

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
        """Apply the CG natural-gradient policy update."""
        grad_flat, _, grad_norm = self._policy_loss_and_grad(batch)
        params = self._trainable_params(self.policy_param)
        if not params or grad_flat.numel() == 0:
            return grad_norm, 0.0, {
                "policy_direction": "cg_fisher",
                "cg_iters_used": 0,
                "cg_residual_norm": 0.0,
                "cg_relative_residual": 0.0,
            }
        direction, diagnostics = self._conjugate_gradient(grad_flat, batch["X0"], params)
        self._apply_flat_direction(params, direction, step_size)
        diagnostics["policy_direction"] = "cg_fisher"
        return grad_norm, float(torch.linalg.norm(direction).item()), diagnostics

    def step(self) -> Dict[str, float]:
        """Run one SBEED optimization step: rho, value, then policy."""
        if self.n == 0:
            raise ValueError("Replay buffer D is empty. Collect data before calling step().")
        starts, lengths, terminals = self._batch_fragment_starts()
        batch = self._fragment_batch(starts, lengths, terminals)

        self.update_index += 1
        # All three learning rates share the same inverse-time decay used in
        # the SBEED experiments, but each parameter block keeps its own base LR.
        decay = 1.0 / (1.0 + float(self.update_index) / self.tau)
        value_step_size = self.lr_value * decay
        rho_step_size = self.lr_rho * decay
        policy_step_size = self.lr_policy * decay

        # The update order follows the thesis implementation: fit rho to the
        # current target, update V against that target, then move the policy.
        if self.rho_param.is_linear_fast_path:
            beta_grad_norm = self._linear_rho_update(batch, rho_step_size)
        else:
            beta_grad_norm = self._nonlinear_rho_update(batch, rho_step_size)

        if self.value_param.is_linear_fast_path:
            theta_grad_norm = self._linear_value_update(batch, value_step_size)
        else:
            theta_grad_norm = self._nonlinear_value_update(batch, value_step_size)

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
                "value_optimizer": "manual_adam" if self.value_param.is_linear_fast_path else "adam",
                "rho_optimizer": "manual_adam" if self.rho_param.is_linear_fast_path else "adam",
                "policy_optimizer": "npg_cg",
                "value_fast_path": bool(self.value_param.is_linear_fast_path),
                "rho_fast_path": bool(self.rho_param.is_linear_fast_path),
                "policy_fast_path": bool(self.policy_param.is_linear_fast_path),
                "rollout_length": int(self.rollout_length),
                "mean_fragment_length": float(lengths.to(dtype=torch.float64).mean().item()),
                "terminal_fragment_fraction": float(terminals.to(dtype=torch.float64).mean().item()),
                "update_index": int(self.update_index),
            }
        )
        stats.update(policy_diagnostics)
        return stats

    def objective(self) -> Dict[str, float]:
        """Evaluate the empirical multi-step SBEED objective on valid replay."""
        if self.n == 0:
            raise ValueError("Replay buffer D is empty. Cannot compute objective.")
        with torch.no_grad():
            starts, lengths, terminals = self._valid_fragment_starts()
            batch = self._fragment_batch(starts, lengths, terminals)
            dtype = torch.float64
            rewards = self._discounted_rewards(batch, dtype)
            log_pi = self._weighted_policy_log_probs(batch, dtype=dtype)
            bootstrap = self._bootstrap_values(batch, dtype)
            delta = rewards - self.lambda_entropy * log_pi + bootstrap
            V0 = self.value_param.value(batch["X0"]).to(dtype=dtype)
            rho = self._rho_values(batch, dtype)
            primal = torch.mean((delta - V0) ** 2)
            dual = torch.mean((delta - rho) ** 2)
            return {
                "objective": float((primal - self.eta * dual).item()),
                "primal_mse": float(primal.item()),
                "dual_mse": float(dual.item()),
            }

    def _policy_probs_for_state(self, state: int) -> torch.Tensor:
        states = torch.tensor([int(state)], dtype=torch.int64, device=self.device)
        return self.policy_param.probs(states).squeeze(0)

    def sample_action(
        self,
        state: int,
        epsilon: float = 0.0,
    ) -> int:
        """Sample from the current policy with optional epsilon exploration."""
        state = int(state)
        if state < 0 or state >= self.n_states:
            raise ValueError("state must be in [0, n_states)")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))

        with torch.no_grad():
            probs = self._policy_probs_for_state(state)
        if self.device.type == "cuda":
            return int(torch.multinomial(probs, num_samples=1).item())

        probs_np = probs.detach().cpu().numpy()
        return int(self.rng.choice(self.n_actions, p=probs_np))

    @staticmethod
    def _parse_transition_result(result: Any) -> Tuple[int, Optional[float], bool]:
        if isinstance(result, tuple):
            if len(result) == 2:
                next_state, reward = result
                return int(next_state), float(reward), False
            if len(result) == 3:
                next_state, reward, done = result
                return int(next_state), float(reward), bool(done)
            if len(result) == 4:
                next_state, reward, terminated, truncated = result
                return int(next_state), float(reward), bool(terminated) or bool(truncated)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                return int(next_state), float(reward), bool(terminated) or bool(truncated)
            raise ValueError("Unsupported transition_fn return tuple length")
        return int(result), None, False

    def collect_steps(
        self,
        transition_fn: Callable[[int, int], Any],
        n_steps: int,
        start_state: Optional[int] = None,
        reward_fn: Optional[Callable[[int, int, int], float]] = None,
        epsilon: float = 0.0,
        terminal_states: Optional[set] = None,
        reset_state_fn: Optional[Callable[[], int]] = None,
    ) -> int:
        """Collect online transitions and append them to FIFO replay."""
        n_steps = int(n_steps)
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")

        terminal_states = set() if terminal_states is None else {int(s) for s in terminal_states}
        state = 0 if start_state is None else int(start_state)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for _ in range(n_steps):
            if state in terminal_states:
                state = int(reset_state_fn()) if reset_state_fn is not None else 0

            action = self.sample_action(state, epsilon=epsilon)
            next_state, reward, done = self._parse_transition_result(transition_fn(state, action))
            if reward is None:
                if reward_fn is None:
                    raise ValueError("reward_fn is required when transition_fn does not return reward")
                reward = float(reward_fn(state, action, next_state))
            done = bool(done or next_state in terminal_states)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = int(reset_state_fn()) if done and reset_state_fn is not None else (0 if done else int(next_state))

        self.dataset.append_many(
            states,
            actions,
            rewards,
            next_states,
            dones,
            capacity=self.max_buffer_size,
        )
        self.dataset.validate(self.n_states, self.n_actions)
        self.n = self.dataset.n
        return self.dataset.n

    def run(
        self,
        transition_fn: Callable[[int, int], Any],
        reward_fn: Optional[Callable[[int, int, int], float]] = None,
        episodes: int = 100,
        collect_per_episode: int = 10,
        updates_per_episode: int = 10,
        initial_collect_steps: int = 0,
        start_state: Optional[int] = None,
        epsilon: float = 0.1,
        terminal_states: Optional[set] = None,
        reset_state_fn: Optional[Callable[[], int]] = None,
        log_every: int = 10,
    ) -> torch.Tensor:
        """Train from online finite-MDP interaction.

        `transition_fn` may return just `next_state`, `(next_state, reward)`,
        or Gym-style tuples. If reward is absent, `reward_fn` is used.
        """
        episodes = int(episodes)
        collect_per_episode = int(collect_per_episode)
        updates_per_episode = int(updates_per_episode)
        initial_collect_steps = int(initial_collect_steps)
        log_every = int(log_every)
        if episodes < 0 or collect_per_episode < 0 or updates_per_episode < 0:
            raise ValueError("episodes, collect_per_episode, and updates_per_episode must be non-negative")
        if initial_collect_steps < 0:
            raise ValueError("initial_collect_steps must be non-negative")
        if log_every <= 0:
            raise ValueError("log_every must be positive")

        self.dataset = DiscreteSBEEDDataset.empty(device=self.device)
        self.n = self.dataset.n
        self.update_index = 0
        self._reset_optimizer_state()

        if initial_collect_steps > 0:
            # Warm-up collection gives the first optimization steps enough
            # replay support before the policy starts changing.
            self.collect_steps(
                transition_fn=transition_fn,
                n_steps=initial_collect_steps,
                start_state=start_state,
                reward_fn=reward_fn,
                epsilon=epsilon,
                terminal_states=terminal_states,
                reset_state_fn=reset_state_fn,
            )

        last_stats = None
        for episode in range(episodes):
            if last_stats is not None and episode % log_every == 0:
                print(
                    f"episode={episode}/{episodes} "
                    f"buffer={self.n} "
                    f"objective={last_stats['objective']:.6f} "
                    f"primal_mse={last_stats['primal_mse']:.6f} "
                    f"dual_mse={last_stats['dual_mse']:.6f} "
                    f"theta_grad={last_stats['theta_grad_norm']:.3e} "
                    f"policy_grad={last_stats['policy_grad_norm']:.3e}"
                )

            self.collect_steps(
                transition_fn=transition_fn,
                n_steps=collect_per_episode,
                start_state=start_state,
                reward_fn=reward_fn,
                epsilon=epsilon,
                terminal_states=terminal_states,
                reset_state_fn=reset_state_fn,
            )

            for _ in range(updates_per_episode):
                last_stats = self.step()

        if last_stats is not None and episodes % log_every == 0:
            print(
                f"episode={episodes}/{episodes} "
                f"buffer={self.n} "
                f"objective={last_stats['objective']:.6f} "
                f"primal_mse={last_stats['primal_mse']:.6f} "
                f"dual_mse={last_stats['dual_mse']:.6f} "
                f"theta_grad={last_stats['theta_grad_norm']:.3e} "
                f"policy_grad={last_stats['policy_grad_norm']:.3e}"
            )

        self.pi = self.get_policy_matrix()
        return self.pi

    def get_policy_matrix(self) -> torch.Tensor:
        states = torch.arange(self.n_states, dtype=torch.int64, device=self.device)
        return self.policy_param.probs(states).detach().clone()

    def policy_fn(self, state: int) -> torch.Tensor:
        state = int(state)
        if state < 0 or state >= self.n_states:
            raise ValueError("state must be in [0, n_states)")
        return self.get_policy_matrix()[state]

    def value(self, state: int) -> torch.Tensor:
        states = torch.tensor([int(state)], dtype=torch.int64, device=self.device)
        return self.value_param.value(states).squeeze(0)

    def rho(self, state: int, action: int) -> torch.Tensor:
        states = torch.tensor([[int(state)]], dtype=torch.int64, device=self.device)
        actions = torch.tensor([[int(action)]], dtype=torch.int64, device=self.device)
        weights = torch.ones((1, 1), dtype=self._param_dtype(self.rho_param), device=self.device)
        return self.rho_param.rho(states, actions, weights).squeeze(0)
