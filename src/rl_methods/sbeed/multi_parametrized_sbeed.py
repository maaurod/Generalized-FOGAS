from __future__ import annotations

import random
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

from .sbeed_dataset import SBEEDDataset
from .sbeed_spec import DiscreteMDPSpec


class MultiParametrizedSBEED:
    """
    Clean multi-step SBEED scaffold for the parametrized solver.

    This first version keeps the current linear feature parametrization:

    The solver learns:
        V_theta(s) = theta^T value_features(s)
        rho_beta(s_0, a_0:h-1) = beta^T sum_l gamma^l rho_features(s_l, a_l)
        pi_W(a | s) = softmax(W policy_features(s))[a]

    It removes the legacy optimizer branches from MultiLinearSBEED:
        - value and rho always use Adam,
        - policy always uses the implicit/CG Fisher update,
        - replay is always FIFO,
        - data collection always follows the current policy.
    """

    def __init__(
        self,
        n_states: Optional[int] = None,
        n_actions: Optional[int] = None,
        gamma: Optional[float] = None,
        value_features: Optional[Callable[[int], torch.Tensor]] = None,
        rho_features: Optional[Callable[[int, int], torch.Tensor]] = None,
        policy_features: Optional[Callable[[int], torch.Tensor]] = None,
        spec: Optional[DiscreteMDPSpec] = None,
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
        if spec is None:
            if n_states is None or n_actions is None or gamma is None:
                raise ValueError("n_states, n_actions, and gamma are required when spec is not provided")
            if value_features is None or rho_features is None:
                raise ValueError("value_features and rho_features are required when spec is not provided")
            spec = DiscreteMDPSpec(
                n_states=n_states,
                n_actions=n_actions,
                gamma=gamma,
                value_features=value_features,
                rho_features=rho_features,
                policy_features=policy_features,
            )

        self.spec = spec
        self.n_states = spec.n_states
        self.n_actions = spec.n_actions
        self.gamma = spec.gamma
        self.value_features = spec.value_features
        self.rho_features = spec.rho_features
        self.policy_features = spec.policy_features
        self.value_dim = spec.value_dim
        self.rho_dim = spec.rho_dim
        self.policy_dim = spec.policy_dim

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

        # D = {(s_i, a_i, r_i, s'_i, done_i)}. SBEED optimizes empirical expectations
        # over this replay buffer; full-batch means every expectation is an
        # average over all rows, mini-batch means a stochastic estimate.
        self.dataset = SBEEDDataset.empty(device=self.device)
        self.dataset.validate(self.n_states, self.n_actions)
        self.n = self.dataset.n

        self._precompute_features()

        self.theta = torch.zeros(self.value_dim, dtype=torch.float64, device=self.device)
        self.beta = torch.zeros(self.rho_dim, dtype=torch.float64, device=self.device)
        self.W = torch.zeros((self.n_actions, self.policy_dim), dtype=torch.float64, device=self.device)

        self.pi: Optional[torch.Tensor] = None
        self.update_index = 0
        self._reset_optimizer_state()

    def _reset_optimizer_state(self) -> None:
        self._adam_state = {
            "theta": {
                "m": torch.zeros_like(self.theta),
                "v": torch.zeros_like(self.theta),
                "t": 0,
            },
            "beta": {
                "m": torch.zeros_like(self.beta),
                "v": torch.zeros_like(self.beta),
                "t": 0,
            },
        }

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

    def _feature_to_device(self, value: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float64, device=self.device).reshape(-1)

    def _precompute_features(self) -> None:
        """
        Build feature design tensors once.

        PHI_S[s]        = phi(s), the value feature.
        POLICY_PHI_S[s] = chi(s), the policy feature.
        RHO_SA[s, a]    = zeta(s, a), the dual residual feature.
        Phi[i]          = phi(s_i)
        Phi_next[i]     = phi(s'_i)
        PolicyPhi[i]    = chi(s_i)
        Rho[i]          = zeta(s_i, a_i)

        I use `rho_features` / `zeta` language here instead of the paper's
        generic rho function name to avoid confusion with LinearMDP psi.
        """
        value_all = [
            self._feature_to_device(self.value_features(s))
            for s in range(self.n_states)
        ]
        policy_all = [
            self._feature_to_device(self.policy_features(s))
            for s in range(self.n_states)
        ]
        rho_all = [
            [
                self._feature_to_device(self.rho_features(s, a))
                for a in range(self.n_actions)
            ]
            for s in range(self.n_states)
        ]

        value_dims = {feat.numel() for feat in value_all}
        policy_dims = {feat.numel() for feat in policy_all}
        rho_dims = {feat.numel() for row in rho_all for feat in row}
        if value_dims != {self.value_dim}:
            raise ValueError("value_features must return a fixed feature dimension")
        if policy_dims != {self.policy_dim}:
            raise ValueError("policy_features must return a fixed feature dimension")
        if rho_dims != {self.rho_dim}:
            raise ValueError("rho_features must return a fixed feature dimension")

        self.PHI_S = torch.stack(value_all, dim=0)
        self.POLICY_PHI_S = torch.stack(policy_all, dim=0)
        self.RHO_SA = torch.stack([torch.stack(row, dim=0) for row in rho_all], dim=0)

        self._refresh_dataset_features()

    def _refresh_dataset_features(self) -> None:
        """
        Refresh row-wise feature tensors after D grows.

        Online SBEED appends new transitions to D. The state feature table
        PHI_S, POLICY_PHI_S, and RHO_SA are fixed, but Phi, Phi_next,
        PolicyPhi, and Rho depend on the current contents of D.
        """
        self.n = self.dataset.n
        if self.n == 0:
            self.Phi = torch.empty((0, self.value_dim), dtype=torch.float64, device=self.device)
            self.Phi_next = torch.empty((0, self.value_dim), dtype=torch.float64, device=self.device)
            self.PolicyPhi = torch.empty((0, self.policy_dim), dtype=torch.float64, device=self.device)
            self.Rho = torch.empty((0, self.rho_dim), dtype=torch.float64, device=self.device)
            return

        X = self.dataset.X.long()
        A = self.dataset.A.long()
        X_next = self.dataset.X_next.long()
        self.Phi = self.PHI_S[X]
        self.Phi_next = self.PHI_S[X_next]
        self.PolicyPhi = self.POLICY_PHI_S[X]
        self.Rho = self.RHO_SA[X, A]

    @staticmethod
    def _row_softmax(logits: torch.Tensor) -> torch.Tensor:
        z = logits - logits.max(dim=1, keepdim=True).values
        exp_z = torch.exp(z)
        return exp_z / exp_z.sum(dim=1, keepdim=True)

    def _policy_matrix(self, W: Optional[torch.Tensor] = None) -> torch.Tensor:
        # pi_W(. | s) = softmax(W chi(s)).
        # POLICY_PHI_S has shape (N, d_pi), W has shape (A, d_pi).
        W = self.W if W is None else W
        logits = self.POLICY_PHI_S @ W.T
        return self._row_softmax(logits)

    def _batch_indices(self) -> torch.Tensor:
        if self.n == 0:
            raise ValueError("Replay buffer D is empty. Collect data before calling step().")
        if self.batch_size is None or self.batch_size >= self.n:
            return torch.arange(self.n, device=self.device)
        return torch.randint(self.n, (int(self.batch_size),), device=self.device)

    def _valid_fragment_starts(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return valid k-step fragment starts, realized lengths, and terminal flags.

        A start is valid when either a complete rollout_length fragment exists
        before the current buffer tail, or a terminal transition truncates the
        fragment earlier. Incomplete non-terminal tails are excluded.
        """
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

        return (
            starts,
            lengths,
            terminals,
        )

    def _batch_fragment_starts(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        starts, lengths, terminals = self._valid_fragment_starts()
        valid_n = int(starts.numel())
        if self.batch_size is None or self.batch_size >= valid_n:
            return starts, lengths, terminals

        sample_positions = torch.randint(valid_n, (int(self.batch_size),), device=self.device)
        return starts[sample_positions], lengths[sample_positions], terminals[sample_positions]

    def _compute_multistep_batch(
        self,
        theta: torch.Tensor,
        W: torch.Tensor,
        starts: torch.Tensor,
        lengths: torch.Tensor,
        terminals: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute fragment-level quantities for the k-step SBEED objective:

            delta = sum_l gamma^l [r_l - lambda log pi(a_l | s_l)]
                    + I[nonterminal] gamma^h V_theta(s_h).
        """
        if self.rollout_length == 1:
            return self._compute_one_step_batch(theta, W, starts)

        pi_mat = self._policy_matrix(W)
        batch_n = int(starts.numel())
        offsets = torch.arange(self.rollout_length, dtype=torch.int64, device=self.device)
        fragment_indices = starts[:, None] + offsets[None, :]
        safe_indices = fragment_indices.clamp_max(max(self.n - 1, 0))
        mask = offsets[None, :] < lengths[:, None]
        discounts = torch.pow(
            torch.tensor(self.gamma, dtype=torch.float64, device=self.device),
            offsets.to(dtype=torch.float64),
        )
        weights = mask.to(dtype=torch.float64) * discounts[None, :]

        X_steps = self.dataset.X[safe_indices].long()
        A_steps = self.dataset.A[safe_indices].long()
        R_steps = self.dataset.R[safe_indices]

        Phi = self.PHI_S[self.dataset.X[starts].long()]
        discounted_rewards = (weights * R_steps).sum(dim=1)

        action_probs = pi_mat[X_steps]
        selected_probs = torch.gather(
            action_probs,
            dim=2,
            index=A_steps[:, :, None],
        ).squeeze(dim=2).clamp_min(1e-12)
        discounted_log_pi = (weights * torch.log(selected_probs)).sum(dim=1)

        Rho_steps = self.RHO_SA[X_steps, A_steps]
        Rho = (weights[:, :, None] * Rho_steps).sum(dim=1)

        PolicyPhi_steps = self.POLICY_PHI_S[X_steps]
        action_one_hot = torch.nn.functional.one_hot(
            A_steps,
            num_classes=self.n_actions,
        ).to(dtype=torch.float64, device=self.device)
        grad_log_pi_steps = (
            (action_one_hot - action_probs)[:, :, :, None]
            * PolicyPhi_steps[:, :, None, :]
        )
        grad_log_pi_sum = (weights[:, :, None, None] * grad_log_pi_steps).sum(dim=1)

        bootstrap_indices = starts + lengths - 1
        bootstrap_states = self.dataset.X_next[bootstrap_indices].long()
        bootstrap_discounts = torch.pow(
            torch.tensor(self.gamma, dtype=torch.float64, device=self.device),
            lengths.to(dtype=torch.float64),
        )
        nonterminal = (~terminals).to(dtype=torch.float64)
        bootstrap_phi = (
            (nonterminal * bootstrap_discounts)[:, None]
            * self.PHI_S[bootstrap_states]
        )

        delta = discounted_rewards - self.lambda_entropy * discounted_log_pi + bootstrap_phi @ theta
        return {
            "starts": starts,
            "lengths": lengths,
            "terminals": terminals,
            "Phi": Phi,
            "Rho": Rho,
            "PolicyPhi": self.POLICY_PHI_S[self.dataset.X[starts].long()],
            "bootstrap_phi": bootstrap_phi,
            "bootstrap_states": bootstrap_states,
            "delta": delta,
            "grad_log_pi_sum": grad_log_pi_sum,
        }

    def _compute_one_step_batch(
        self,
        theta: torch.Tensor,
        W: torch.Tensor,
        indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        X = self.dataset.X[indices].long()
        A = self.dataset.A[indices].long()
        D = self.dataset.D[indices].to(dtype=torch.float64)
        Phi = self.PHI_S[X]
        Phi_next = self.PHI_S[self.dataset.X_next[indices].long()]
        PolicyPhi = self.POLICY_PHI_S[X]
        Rho = self.RHO_SA[X, A]
        bootstrap_phi = self.gamma * (1.0 - D)[:, None] * Phi_next

        pi_mat = self._policy_matrix(W)
        action_probs = pi_mat[X]
        log_pi = torch.log(action_probs[torch.arange(indices.numel(), device=self.device), A].clamp_min(1e-12))
        delta = self.dataset.R[indices] - self.lambda_entropy * log_pi + bootstrap_phi @ theta

        grad_log_pi_sum = -action_probs[:, :, None] * PolicyPhi[:, None, :]
        grad_log_pi_sum[torch.arange(indices.numel(), device=self.device), A] += PolicyPhi

        return {
            "starts": indices,
            "lengths": torch.ones(indices.numel(), dtype=torch.int64, device=self.device),
            "terminals": self.dataset.D[indices].to(dtype=torch.bool),
            "Phi": Phi,
            "Rho": Rho,
            "PolicyPhi": PolicyPhi,
            "bootstrap_phi": bootstrap_phi,
            "bootstrap_states": self.dataset.X_next[indices].long(),
            "delta": delta,
            "grad_log_pi_sum": grad_log_pi_sum,
        }

    def objective(self) -> Dict[str, float]:
        if self.n == 0:
            raise ValueError("Replay buffer D is empty. Cannot compute objective.")
        with torch.no_grad():
            starts, lengths, terminals = self._valid_fragment_starts()
            batch = self._compute_multistep_batch(self.theta, self.W, starts, lengths, terminals)
            delta = batch["delta"]
            V = batch["Phi"] @ self.theta
            rho = batch["Rho"] @ self.beta

            # Empirical SBEED objective:
            #   E[(delta_h - V(s_0))^2]
            #   - eta E[(delta_h - rho(s_0,a_0:h-1))^2].
            # The second term is the variance-cancellation dual term from the
            # paper. This variant updates beta by SGD, so the reported dual MSE
            # is a direct diagnostic of the current rho fit.
            primal = torch.mean((delta - V) ** 2)
            dual = torch.mean((delta - rho) ** 2)
            return {
                "objective": float((primal - self.eta * dual).item()),
                "primal_mse": float(primal.item()),
                "dual_mse": float(dual.item()),
            }

    def _adam_update(self, name: str, param: torch.Tensor, grad: torch.Tensor, step_size: float) -> torch.Tensor:
        state = self._adam_state[name]
        state["t"] += 1
        state["m"] = self.adam_beta1 * state["m"] + (1.0 - self.adam_beta1) * grad
        state["v"] = self.adam_beta2 * state["v"] + (1.0 - self.adam_beta2) * (grad * grad)
        bias_correction1 = 1.0 - self.adam_beta1 ** state["t"]
        bias_correction2 = 1.0 - self.adam_beta2 ** state["t"]
        m_hat = state["m"] / bias_correction1
        v_hat = state["v"] / bias_correction2
        return param - step_size * m_hat / (torch.sqrt(v_hat) + self.adam_eps)

    def _apply_value_update(self, grad_theta: torch.Tensor, step_size: float) -> None:
        self.theta = self._adam_update("theta", self.theta, grad_theta, step_size)

    def _apply_rho_update(self, grad_beta: torch.Tensor, step_size: float) -> None:
        # grad_beta is -G_rho, so descent on grad_beta is equivalent to the
        # ascent-form rho prox step beta <- beta + step_size * G_rho.
        self.beta = self._adam_update("beta", self.beta, grad_beta, step_size)

    def _flatten_policy_grad(self, grad_W: torch.Tensor) -> torch.Tensor:
        return grad_W.reshape(-1)

    def _unflatten_policy_direction(self, direction: torch.Tensor) -> torch.Tensor:
        return direction.reshape_as(self.W)

    def _policy_kl_hessian_vector_product(
        self,
        vector: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fisher-vector product from the KL Hessian at the current policy.

        This is the TRPO/NPG-style implicit Fisher computation:
            F v = Hessian_W E_s KL(pi_Wold(.|s) || pi_W(.|s))|_{W=Wold} v.

        The implementation uses autograd on the empirical KL instead of the
        closed-form softmax-linear Fisher. That keeps the conjugate-gradient
        path tied to the KL geometry and makes it easier to reuse if the policy
        parametrization is generalized while still exposing `_policy_matrix`.
        """
        old_probs = self._policy_matrix(self.W)[state_indices].detach().clamp_min(1e-12)
        W_var = self.W.detach().clone().requires_grad_(True)
        new_probs = self._policy_matrix(W_var)[state_indices].clamp_min(1e-12)

        empirical_kl = (
            old_probs * (torch.log(old_probs) - torch.log(new_probs))
        ).sum(dim=1).mean()

        grad_kl = torch.autograd.grad(
            empirical_kl,
            W_var,
            create_graph=True,
        )[0]
        grad_vector_product = torch.dot(grad_kl.reshape(-1), vector.detach())
        hvp = torch.autograd.grad(
            grad_vector_product,
            W_var,
            retain_graph=False,
        )[0]
        return hvp.reshape(-1).detach()

    def _conjugate_gradient_policy_direction(
        self,
        grad_W: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        b = self._flatten_policy_grad(grad_W).detach()

        def matvec(v: torch.Tensor) -> torch.Tensor:
            return (
                self._policy_kl_hessian_vector_product(v, state_indices)
                + self.fisher_damping * v
            )

        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)
        if torch.sqrt(rs_old) <= self.cg_tol:
            return self._unflatten_policy_direction(x), {
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
        diagnostics = {
            "cg_iters_used": int(iters_used),
            "cg_residual_norm": float(residual_norm.item()),
            "cg_relative_residual": float((residual_norm / b_norm.clamp_min(1e-30)).item()),
        }

        return self._unflatten_policy_direction(x), diagnostics

    def _apply_policy_update(
        self,
        grad_W: torch.Tensor,
        state_indices: torch.Tensor,
        step_size: float,
    ) -> Tuple[torch.Tensor, str, Dict[str, float]]:
        direction, diagnostics = self._conjugate_gradient_policy_direction(
            grad_W,
            state_indices,
        )
        direction_kind = "cg_fisher"

        self.W = self.W - step_size * direction
        return direction, direction_kind, diagnostics

    def step(self) -> Dict[str, float]:
        if self.n == 0:
            raise ValueError("Replay buffer D is empty. Collect data before calling step().")
        starts, lengths, terminals = self._batch_fragment_starts()
        batch = self._compute_multistep_batch(self.theta, self.W, starts, lengths, terminals)
        Phi = batch["Phi"]
        Rho = batch["Rho"]
        bootstrap_phi = batch["bootstrap_phi"]

        # 1. Build the multi-step delta with the current value and policy.
        delta = batch["delta"]

        V = Phi @ self.theta
        rho = Rho @ self.beta
        residual_v = delta - V
        residual_rho = delta - rho

        # 2. SGD rho/beta gradient from the variant:
        #    G_beta = -E[(delta - rho_beta(s,a)) zeta(s,a)].
        # Delta is detached so the rho update treats the current target as
        # fixed, matching the original exact-fit inner update role.
        grad_beta = -((residual_rho.detach())[:, None] * Rho).mean(dim=0)

        # 3. Multi-step value gradient, specialized to
        #    V_theta(s) = theta^T phi(s):
        #
        # grad_theta =
        #   2 E[(delta_h - V(s_0)) (I[nonterminal] gamma^h phi(s_h) - phi(s_0))]
        #   - 2 eta E[(delta_h - rho(s_0,a_0:h-1)) I[nonterminal] gamma^h phi(s_h)].
        grad_theta = (
            2.0 * (residual_v[:, None] * (bootstrap_phi - Phi)).mean(dim=0)
            - 2.0 * self.eta * (residual_rho[:, None] * bootstrap_phi).mean(dim=0)
        )

        # 4. Softmax-linear policy gradient over each fragment:
        #   sum_l gamma^l grad_W log pi(a_l|s_l).
        advantage = (1.0 - self.eta) * delta + self.eta * rho - V
        grad_W = -2.0 * (
            advantage[:, None, None] * batch["grad_log_pi_sum"]
        ).mean(dim=0)

        # 5. Euclidean mirror descent = ordinary gradient descent with a
        # shared tau-controlled step-size decay across value, rho, and policy
        # updates: zeta_j = zeta_0 / (1 + j / tau). Value and rho use Adam
        # with the decayed step size as Adam's alpha.
        # For NPG policy updates, grad_W is preconditioned by the empirical
        # Fisher induced by the KL geometry before applying the policy step.
        self.update_index += 1
        decay = 1.0 / (1.0 + float(self.update_index) / self.tau)
        value_step_size = self.lr_value * decay
        rho_step_size = self.lr_rho * decay
        policy_step_size = self.lr_policy * decay
        self._apply_rho_update(grad_beta, rho_step_size)
        self._apply_value_update(grad_theta, value_step_size)
        policy_direction, policy_direction_kind, policy_diagnostics = self._apply_policy_update(
            grad_W=grad_W,
            state_indices=self.dataset.X[starts].long(),
            step_size=policy_step_size,
        )

        stats = self.objective()
        stats.update(
            {
                "theta_grad_norm": float(torch.linalg.norm(grad_theta).item()),
                "beta_grad_norm": float(torch.linalg.norm(grad_beta).item()),
                "policy_grad_norm": float(torch.linalg.norm(grad_W).item()),
                "policy_direction_norm": float(torch.linalg.norm(policy_direction).item()),
                "value_step_size": float(value_step_size),
                "rho_step_size": float(rho_step_size),
                "policy_step_size": float(policy_step_size),
                "value_optimizer": "adam",
                "rho_optimizer": "adam",
                "policy_optimizer": "npg_cg",
                "policy_direction": policy_direction_kind,
                "rollout_length": int(self.rollout_length),
                "mean_fragment_length": float(lengths.to(dtype=torch.float64).mean().item()),
                "terminal_fragment_fraction": float(terminals.to(dtype=torch.float64).mean().item()),
                "update_index": int(self.update_index),
            }
        )
        stats.update(policy_diagnostics)
        return stats

    def sample_action(
        self,
        state: int,
        epsilon: float = 0.0,
    ) -> int:
        """
        Sample an action from the current softmax policy.

        With epsilon > 0, collection becomes epsilon-greedy around the current
        policy. At initialization the policy is uniform because W starts at zero.
        """
        state = int(state)
        if state < 0 or state >= self.n_states:
            raise ValueError("state must be in [0, n_states)")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")

        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))

        probs = self.get_policy_matrix()[state].detach().cpu().numpy()
        return int(self.rng.choice(self.n_actions, p=probs))

    @staticmethod
    def _parse_transition_result(result: Any) -> Tuple[int, Optional[float], bool]:
        """
        Accept common discrete transition signatures:
            next_state
            (next_state, reward)
            (next_state, reward, done)
            (next_state, reward, terminated, truncated)
            (next_state, reward, terminated, truncated, info)
        """
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
        """
        Collect transitions into D with the current policy.

        This is the online part of Algorithm 1:
            execute pi_W, append (s, a, r, s') to D.

        For simple deterministic grids, `transition_fn(s, a)` may return only
        next_state and `reward_fn(s, a, next_state)` supplies the reward.
        If a terminal state is reached, collection resets to x0 or
        reset_state_fn().
        """
        n_steps = int(n_steps)
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")

        terminal_states = set() if terminal_states is None else {int(s) for s in terminal_states}
        if start_state is None:
            state = self.spec.x0 if self.spec.x0 is not None else 0
        else:
            state = int(start_state)

        for _ in range(n_steps):
            if state in terminal_states:
                state = int(reset_state_fn()) if reset_state_fn is not None else (
                    self.spec.x0 if self.spec.x0 is not None else 0
                )

            action = self.sample_action(state, epsilon=epsilon)
            next_state, reward, done = self._parse_transition_result(transition_fn(state, action))
            if reward is None:
                if reward_fn is None:
                    raise ValueError("reward_fn is required when transition_fn does not return reward")
                reward = float(reward_fn(state, action, next_state))
            done = bool(done or next_state in terminal_states)

            self.dataset.append_fifo(
                state,
                action,
                reward,
                next_state,
                capacity=self.max_buffer_size,
                done=done,
            )

            if done:
                state = int(reset_state_fn()) if reset_state_fn is not None else (
                    self.spec.x0 if self.spec.x0 is not None else 0
                )
            else:
                state = int(next_state)

        self.dataset.validate(self.n_states, self.n_actions)
        self._refresh_dataset_features()
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
        """
        Online SBEED loop matching Algorithm 1 at a notebook-friendly level.

        Each invocation starts from a fresh replay buffer D.

        For each episode:
            1. collect K transitions with the current policy into D,
            2. run N primal/dual updates by sampling from FIFO replay,
            3. continue collecting with the updated policy.

        Reproducibility is controlled by the solver seed. The initial policy is
        uniform because W is initialized to zeros. The update counter j resets
        to 0 at the start of each run, so the effective learning rates at
        update j >= 1 are lr_value / (1 + j / tau),
        lr_rho / (1 + j / tau), and lr_policy / (1 + j / tau).
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

        self.dataset = SBEEDDataset.empty(device=self.device)
        self._refresh_dataset_features()
        self.update_index = 0
        self._reset_optimizer_state()

        if initial_collect_steps > 0:
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
            # Periodic logging happens before collecting the next block of data.
            # The reported metrics are from the previous completed update block.
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

    def get_policy_matrix(self, W: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._policy_matrix(W).detach().clone()

    def policy_fn(self, state: int) -> torch.Tensor:
        state = int(state)
        if state < 0 or state >= self.n_states:
            raise ValueError("state must be in [0, n_states)")
        return self.get_policy_matrix()[state]

    def value(self, state: int) -> torch.Tensor:
        feat = self.PHI_S[int(state)]
        return feat @ self.theta

    def rho(self, state: int, action: int) -> torch.Tensor:
        feat = self.RHO_SA[int(state), int(action)]
        return feat @ self.beta
