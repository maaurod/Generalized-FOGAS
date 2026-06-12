import random

import numpy as np
import torch
from tqdm import trange

from ...fogas_clean.fogas_dataset import FOGASDataset
from ..features import build_policy_feature_table, build_q_feature_table, build_u_feature_table
from ..fogas_parameters import GeneralizedFOGASParameters


class LinearSolver:
    """
    Standalone linear-u, linear-Q, linear-policy FOGAS solver.

    u_beta(x, a) = <beta, u_features(x, a)>
    Q_theta(x, a) = <theta, q_features(x, a)>
    pi_psi(a | x) = softmax_a(<psi, policy_features(x, a)>)
    """

    _THETA_UPDATES = {"exact", "sgd", "adam"}
    _POLICY_OPTIMIZERS = {"sgd", "adam", "npg"}
    _POLICY_GRADIENTS = {"exact", "reinforce"}
    _EPS = 1e-12

    def __init__(
        self,
        n_states,
        n_actions,
        gamma,
        x0,
        csv_path,
        u_function,
        q_function,
        policy_features,
        delta=0.05,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        D_theta=None,
        beta_reg=None,
        theta_update="exact",
        theta_lr=None,
        theta_inner_steps=100,
        theta_loss_include_beta_reg=False,
        d_theta_scale=1.0,
        print_params=False,
        dataset_verbose=False,
        seed=42,
        device=None,
    ):
        if u_function is None:
            raise ValueError("u_function must be provided")
        if q_function is None:
            raise ValueError("q_function must be provided")
        if policy_features is None:
            raise ValueError("policy_features must be provided")

        self.N = int(n_states)
        self.A = int(n_actions)
        self.gamma = float(gamma)
        self.x0 = int(x0)
        self.csv_path = csv_path
        self.u_function = u_function
        self.q_function = q_function
        self.policy_features = policy_features
        self.delta = delta
        self.seed = seed

        if self.N <= 0:
            raise ValueError("n_states must be positive")
        if self.A <= 0:
            raise ValueError("n_actions must be positive")
        if self.x0 < 0 or self.x0 >= self.N:
            raise ValueError(f"x0 must be in [0, {self.N}), got {self.x0}")

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)

        self.theta_update = self._canonical_theta_update(theta_update)
        self.theta_lr = self._canonical_optional_positive_float(theta_lr, "theta_lr")
        self.theta_inner_steps = self._canonical_positive_int(
            theta_inner_steps,
            "theta_inner_steps",
        )
        self.theta_loss_include_beta_reg = self._canonical_bool(
            theta_loss_include_beta_reg,
            "theta_loss_include_beta_reg",
        )
        self.d_theta_scale = self._canonical_positive_float(d_theta_scale, "d_theta_scale")

        self.dataset = FOGASDataset(csv_path=csv_path, verbose=dataset_verbose)
        self.Xs = self.dataset.X.to(self.device).long()
        self.As = self.dataset.A.to(self.device).long()
        self.Rs = self.dataset.R.to(dtype=torch.float64, device=self.device)
        self.X_nexts = self.dataset.X_next.to(self.device).long()
        self.n = self.dataset.n

        self._validate_dataset_indices()
        # The three parametrizations intentionally live in separate feature
        # spaces, so beta, theta, and policy parameters may have different sizes.
        self._build_u_tensors()
        self._build_q_tensors()
        self._build_policy_tensors()
        self._build_preconditioner(beta_reg)
        self._estimate_omega()
        self._compute_reward_bound()

        default_D_theta = np.sqrt(self.d_q / (1.0 - self.gamma)) if D_theta is None else D_theta
        self.params = GeneralizedFOGASParameters(
            n=self.n,
            reward_bound=self.R,
            n_states=self.N,
            n_actions=self.A,
            feature_dim=self.d,
            gamma=self.gamma,
            delta=delta,
            T=T,
            alpha=alpha,
            eta=eta,
            rho=rho,
            D_theta=default_D_theta,
            beta_reg=beta_reg,
            print_params=print_params,
        )

        self.T = self.params.T
        self.alpha = self.params.alpha
        self.eta = self.params.eta
        self.rho = self.params.rho
        self.D_theta = self.params.D_theta
        self.beta_reg = self.params.beta_reg
        self.D_pi = self.params.D_pi

        if self.beta_reg != self._preconditioner_beta_reg:
            self._build_preconditioner(self.beta_reg)
            self._estimate_omega()
            self._compute_reward_bound()

        self.theta = None
        self.theta_bar_history = None
        self.pi = None
        self.lambda_T = None
        self.beta_T = None
        self.psi = None
        self.psi_history = None
        self.diagnostics_history = None
        self.policy_optimizer_name = None

    def _validate_dataset_indices(self):
        if torch.any((self.Xs < 0) | (self.Xs >= self.N)):
            raise ValueError("dataset states contain values outside [0, n_states)")
        if torch.any((self.X_nexts < 0) | (self.X_nexts >= self.N)):
            raise ValueError("dataset next states contain values outside [0, n_states)")
        if torch.any((self.As < 0) | (self.As >= self.A)):
            raise ValueError("dataset actions contain values outside [0, n_actions)")

    def _build_u_tensors(self):
        self.U_XA = build_u_feature_table(
            self.u_function,
            self.N,
            self.A,
            device=self.device,
            dtype=torch.float64,
        )
        self.U_sample = self.U_XA[self.Xs, self.As]
        self.d = int(self.U_XA.shape[2])

    def _build_q_tensors(self):
        self.Q_XA = build_q_feature_table(
            self.q_function,
            self.N,
            self.A,
            device=self.device,
            dtype=torch.float64,
        )
        self.Q_sample = self.Q_XA[self.Xs, self.As]
        self.d_q = int(self.Q_XA.shape[2])

    def _build_policy_tensors(self):
        self.OMEGA_PI_XA = build_policy_feature_table(
            self.policy_features,
            self.N,
            self.A,
            device=self.device,
            dtype=torch.float64,
        )
        self.d_pi = int(self.OMEGA_PI_XA.shape[2])

    def _build_preconditioner(self, beta_reg):
        # Keep the previous beta-side ridge/preconditioner behavior. This is
        # independent from theta_loss_include_beta_reg.
        if beta_reg is None:
            beta_reg = 1.0
        beta_reg = float(beta_reg)
        if beta_reg < 0.0:
            raise ValueError("beta_reg must be non-negative")
        self._preconditioner_beta_reg = beta_reg
        self.Empirical_cov = (self.U_sample.T @ self.U_sample) / self.n
        self.H = self.Empirical_cov + beta_reg * torch.eye(
            self.d,
            dtype=torch.float64,
            device=self.device,
        )
        self.H_inv = torch.linalg.inv(self.H)
        self.Cov_emp = self.H
        self.Cov_emp_inv = self.H_inv

    def _estimate_omega(self):
        self.omega = self.H_inv @ ((self.U_sample.T @ self.Rs) / self.n)

    def _compute_reward_bound(self):
        self.r_hat = torch.tensordot(self.U_XA, self.omega, dims=([2], [0]))
        R = torch.max(torch.abs(self.r_hat))
        self.R = float(max(R.detach().cpu().item(), self._EPS))

    @classmethod
    def _canonical_theta_update(cls, theta_update):
        name = str(theta_update).lower()
        if name not in cls._THETA_UPDATES:
            raise ValueError("theta_update must be one of 'exact', 'sgd', or 'adam'")
        return name

    @classmethod
    def _canonical_policy_optimizer(cls, policy_optimizer):
        name = str(policy_optimizer).lower()
        if name not in cls._POLICY_OPTIMIZERS:
            raise ValueError("policy_optimizer must be one of 'sgd', 'adam', or 'npg'")
        return name

    @classmethod
    def _canonical_policy_gradient(cls, policy_gradient):
        name = str(policy_gradient).lower()
        if name not in cls._POLICY_GRADIENTS:
            raise ValueError("policy_gradient must be either 'exact' or 'reinforce'")
        return name

    @staticmethod
    def _canonical_positive_float(value, name):
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    @classmethod
    def _canonical_optional_positive_float(cls, value, name):
        if value is None:
            return None
        return cls._canonical_positive_float(value, name)

    @staticmethod
    def _canonical_positive_int(value, name):
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _canonical_bool(value, name):
        if isinstance(value, bool):
            return value
        raise ValueError(f"{name} must be a bool")

    @staticmethod
    def _row_softmax(logits):
        shifted = logits - logits.max(dim=1, keepdim=True).values
        exp = torch.exp(shifted)
        return exp / exp.sum(dim=1, keepdim=True)

    def _linear_policy_matrix(self, psi_t):
        logits = torch.tensordot(self.OMEGA_PI_XA, psi_t, dims=([2], [0]))
        return self._row_softmax(logits)

    def _prepare_beta_init(self, beta_init):
        if beta_init is None:
            return torch.zeros(self.d, dtype=torch.float64, device=self.device)
        beta_t = beta_init.clone().to(dtype=torch.float64, device=self.device).reshape(-1)
        if beta_t.shape != (self.d,):
            raise ValueError(f"beta_init must have shape ({self.d},), got {tuple(beta_t.shape)}")
        return beta_t

    def _prepare_theta_bar_init(self, theta_bar_init):
        if theta_bar_init is None:
            return torch.zeros(self.d_q, dtype=torch.float64, device=self.device)
        theta_bar_t = theta_bar_init.clone().to(dtype=torch.float64, device=self.device).reshape(-1)
        if theta_bar_t.shape != (self.d_q,):
            raise ValueError(
                f"theta_bar_init must have shape ({self.d_q},), got {tuple(theta_bar_t.shape)}"
            )
        return theta_bar_t

    def _prepare_psi_init(self, psi_init):
        if psi_init is None:
            return torch.zeros(self.d_pi, dtype=torch.float64, device=self.device)
        psi_t = psi_init.clone().to(dtype=torch.float64, device=self.device).reshape(-1)
        if psi_t.shape != (self.d_pi,):
            raise ValueError(f"psi_init must have shape ({self.d_pi},), got {tuple(psi_t.shape)}")
        return psi_t

    def _compute_theta_update(self, theta_mismatch, beta_t, effective_D_theta):
        c_t = theta_mismatch
        if self.theta_loss_include_beta_reg:
            # This legacy option only makes sense when beta and theta share the
            # same coordinates.
            if self.d_q != self.d:
                raise ValueError(
                    "theta_loss_include_beta_reg=True requires q_function and u_function "
                    "to have the same feature dimension"
                )
            c_t = c_t - self._preconditioner_beta_reg * beta_t

        norm_c = torch.linalg.norm(c_t)
        # The cleaned solver always uses adaptive theta regularization. Scaling
        # D_theta changes the implied regularization strength.
        lambda_theta = max(float(norm_c.detach().cpu().item()) / effective_D_theta, self._EPS)

        if norm_c < self._EPS and self.theta_update == "exact":
            theta_t = torch.zeros_like(c_t)
            theta_grad_norm = 0.0
            theta_lr_used = None
        elif self.theta_update == "exact":
            theta_t = -c_t / lambda_theta
            theta_grad_norm = float(
                torch.linalg.norm(c_t + lambda_theta * theta_t).detach().cpu().item()
            )
            theta_lr_used = None
        elif self.theta_update == "sgd":
            theta_lr_used = self.theta_lr if self.theta_lr is not None else 1.0 / lambda_theta
            theta_t, theta_grad_norm = self._sgd_theta_update(c_t, lambda_theta, theta_lr_used)
        else:
            theta_lr_used = self.theta_lr if self.theta_lr is not None else 1e-2
            theta_t, theta_grad_norm = self._adam_theta_update(c_t, lambda_theta, theta_lr_used)

        q_objective = torch.dot(theta_t, c_t) + 0.5 * lambda_theta * torch.dot(theta_t, theta_t)
        return theta_t, c_t, lambda_theta, q_objective, theta_grad_norm, theta_lr_used

    def _sgd_theta_update(self, c_t, lambda_theta, theta_lr):
        theta_t = torch.zeros_like(c_t)
        for _ in range(self.theta_inner_steps):
            grad = c_t + lambda_theta * theta_t
            theta_t = theta_t - theta_lr * grad
        final_grad = c_t + lambda_theta * theta_t
        return theta_t, float(torch.linalg.norm(final_grad).detach().cpu().item())

    def _adam_theta_update(self, c_t, lambda_theta, theta_lr):
        theta_param = torch.nn.Parameter(torch.zeros_like(c_t))
        optimizer = torch.optim.Adam([theta_param], lr=theta_lr)
        for _ in range(self.theta_inner_steps):
            optimizer.zero_grad(set_to_none=True)
            objective = torch.dot(theta_param, c_t) + 0.5 * lambda_theta * torch.dot(
                theta_param,
                theta_param,
            )
            objective.backward()
            optimizer.step()
        with torch.no_grad():
            theta_t = theta_param.detach().clone()
            final_grad = c_t + lambda_theta * theta_t
        return theta_t, float(torch.linalg.norm(final_grad).detach().cpu().item())

    def _exact_policy_gradient(self, pi_mat, q_all, policy_state_weights):
        G = policy_state_weights[:, None] * q_all
        policy_objective = (pi_mat * G).sum()
        V_G = (pi_mat * G).sum(dim=1)
        advantage_G = G - V_G[:, None]
        policy_grad = (
            pi_mat[..., None] * advantage_G[..., None] * self.OMEGA_PI_XA
        ).sum(dim=(0, 1))
        return policy_grad, policy_objective

    def _reinforce_policy_gradient(
        self,
        pi_mat,
        q_all,
        coeff,
        state_weights,
        policy_state_weights,
        state_weight_update,
        reinforce_samples,
    ):
        if state_weight_update == "normal":
            states = torch.cat(
                (
                    torch.tensor([self.x0], dtype=torch.long, device=self.device),
                    self.X_nexts,
                )
            )
            weights = torch.cat(
                (
                    torch.tensor([1.0 - self.gamma], dtype=torch.float64, device=self.device),
                    (self.gamma / self.n) * coeff,
                )
            )
        else:
            states = torch.arange(self.N, dtype=torch.long, device=self.device)
            weights = policy_state_weights

        sample_count = self._canonical_positive_int(reinforce_samples, "reinforce_samples")
        probs = pi_mat[states]
        sampled_actions = torch.multinomial(
            probs.reshape(-1, self.A),
            num_samples=sample_count,
            replacement=True,
        )

        q_states = q_all[states]
        baseline = (probs * q_states).sum(dim=1)
        sampled_q = q_states.gather(1, sampled_actions)
        advantages = sampled_q - baseline[:, None]

        omega_states = self.OMEGA_PI_XA[states]
        expected_omega = (probs[..., None] * omega_states).sum(dim=1)
        sampled_omega = omega_states.gather(
            1,
            sampled_actions[..., None].expand(-1, -1, self.d_pi),
        )
        grad_log_pi = sampled_omega - expected_omega[:, None, :]
        weighted_terms = weights[:, None, None] * advantages[..., None] * grad_log_pi
        policy_grad = weighted_terms.sum(dim=(0, 1)) / float(sample_count)

        policy_objective = (pi_mat * (policy_state_weights[:, None] * q_all)).sum()
        return policy_grad, policy_objective

    def _policy_kl_hessian_vector_product(self, vector, psi_t, state_indices):
        old_probs = self._linear_policy_matrix(psi_t)[state_indices].detach().clamp_min(self._EPS)
        psi_var = psi_t.detach().clone().requires_grad_(True)
        new_probs = self._linear_policy_matrix(psi_var)[state_indices].clamp_min(self._EPS)

        empirical_kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(
            dim=1
        ).mean()
        grad_kl = torch.autograd.grad(empirical_kl, psi_var, create_graph=True)[0]
        grad_vector_product = torch.dot(grad_kl.reshape(-1), vector.detach())
        hvp = torch.autograd.grad(grad_vector_product, psi_var, retain_graph=False)[0]
        return hvp.reshape(-1).detach()

    def _conjugate_gradient_policy_direction(
        self,
        policy_grad,
        psi_t,
        state_indices,
        fisher_damping,
        cg_iters,
        cg_tol,
    ):
        b = policy_grad.detach().reshape(-1)

        def matvec(v):
            return (
                self._policy_kl_hessian_vector_product(v, psi_t, state_indices)
                + fisher_damping * v
            )

        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)
        if torch.sqrt(rs_old) <= cg_tol:
            return x, {
                "cg_iters_used": 0,
                "cg_residual_norm": float(torch.sqrt(rs_old).detach().cpu().item()),
                "cg_relative_residual": 0.0,
            }

        iters_used = 0
        for _ in range(cg_iters):
            Ap = matvec(p)
            alpha = rs_old / torch.dot(p, Ap).clamp_min(1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            iters_used += 1
            if torch.sqrt(rs_new) <= cg_tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        residual_norm = torch.linalg.norm(matvec(x) - b)
        b_norm = torch.linalg.norm(b)
        diagnostics = {
            "cg_iters_used": int(iters_used),
            "cg_residual_norm": float(residual_norm.detach().cpu().item()),
            "cg_relative_residual": float(
                (residual_norm / b_norm.clamp_min(1e-30)).detach().cpu().item()
            ),
        }
        return x, diagnostics

    def _policy_update_state_indices(self, state_weight_update):
        if state_weight_update == "normal":
            return torch.cat(
                (
                    torch.tensor([self.x0], dtype=torch.long, device=self.device),
                    self.X_nexts,
                )
            )
        return torch.arange(self.N, dtype=torch.long, device=self.device)

    def get_diagnostics(self):
        return self.diagnostics_history

    def run(
        self,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        D_theta=None,
        d_theta_scale=None,
        beta_init=None,
        theta_bar_init=None,
        psi_init=None,
        theta_update=None,
        theta_lr=None,
        theta_inner_steps=None,
        theta_loss_include_beta_reg=None,
        policy_optimizer="sgd",
        policy_gradient="exact",
        reinforce_samples=1,
        fisher_damping=1e-3,
        cg_iters=10,
        cg_tol=1e-10,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        verbose=False,
        tqdm_print=False,
        log_interval=None,
        state_weight_update="normal",
        c_min=0.1,
    ):
        T = self.params.T if T is None else int(T)
        alpha = self.params.alpha if alpha is None else float(alpha)
        eta = self.params.eta if eta is None else float(eta)
        rho = self.params.rho if rho is None else float(rho)
        D_theta = self.params.D_theta if D_theta is None else float(D_theta)

        previous_theta_update = self.theta_update
        previous_theta_lr = self.theta_lr
        previous_theta_inner_steps = self.theta_inner_steps
        previous_theta_loss_include_beta_reg = self.theta_loss_include_beta_reg
        previous_d_theta_scale = self.d_theta_scale

        if theta_update is not None:
            self.theta_update = self._canonical_theta_update(theta_update)
        if theta_lr is not None:
            self.theta_lr = self._canonical_positive_float(theta_lr, "theta_lr")
        if theta_inner_steps is not None:
            self.theta_inner_steps = self._canonical_positive_int(
                theta_inner_steps,
                "theta_inner_steps",
            )
        if theta_loss_include_beta_reg is not None:
            self.theta_loss_include_beta_reg = self._canonical_bool(
                theta_loss_include_beta_reg,
                "theta_loss_include_beta_reg",
            )
        if d_theta_scale is not None:
            self.d_theta_scale = self._canonical_positive_float(d_theta_scale, "d_theta_scale")

        try:
            return self._run_impl(
                T=T,
                alpha=alpha,
                eta=eta,
                rho=rho,
                D_theta=D_theta,
                beta_init=beta_init,
                theta_bar_init=theta_bar_init,
                psi_init=psi_init,
                policy_optimizer=policy_optimizer,
                policy_gradient=policy_gradient,
                reinforce_samples=reinforce_samples,
                fisher_damping=fisher_damping,
                cg_iters=cg_iters,
                cg_tol=cg_tol,
                adam_betas=adam_betas,
                adam_eps=adam_eps,
                verbose=verbose,
                tqdm_print=tqdm_print,
                log_interval=log_interval,
                state_weight_update=state_weight_update,
                c_min=c_min,
            )
        finally:
            self.theta_update = previous_theta_update
            self.theta_lr = previous_theta_lr
            self.theta_inner_steps = previous_theta_inner_steps
            self.theta_loss_include_beta_reg = previous_theta_loss_include_beta_reg
            self.d_theta_scale = previous_d_theta_scale

    def _run_impl(
        self,
        T,
        alpha,
        eta,
        rho,
        D_theta,
        beta_init,
        theta_bar_init,
        psi_init,
        policy_optimizer,
        policy_gradient,
        reinforce_samples,
        fisher_damping,
        cg_iters,
        cg_tol,
        adam_betas,
        adam_eps,
        verbose,
        tqdm_print,
        log_interval,
        state_weight_update,
        c_min,
    ):
        if state_weight_update not in {"normal", "clipped"}:
            raise ValueError("state_weight_update must be either 'normal' or 'clipped'")
        policy_optimizer = self._canonical_policy_optimizer(policy_optimizer)
        policy_gradient = self._canonical_policy_gradient(policy_gradient)
        reinforce_samples = self._canonical_positive_int(reinforce_samples, "reinforce_samples")
        fisher_damping = float(fisher_damping)
        cg_iters = self._canonical_positive_int(cg_iters, "cg_iters")
        cg_tol = float(cg_tol)
        if fisher_damping < 0.0:
            raise ValueError("fisher_damping must be non-negative")
        if cg_tol < 0.0:
            raise ValueError("cg_tol must be non-negative")
        self.policy_optimizer_name = policy_optimizer

        beta_t = self._prepare_beta_init(beta_init)
        theta_bar_t = self._prepare_theta_bar_init(theta_bar_init)
        psi_t = self._prepare_psi_init(psi_init)

        adam_param = None
        adam_optimizer = None
        if policy_optimizer == "adam":
            adam_param = torch.nn.Parameter(psi_t.clone())
            adam_optimizer = torch.optim.Adam(
                [adam_param],
                lr=alpha,
                betas=adam_betas,
                eps=adam_eps,
            )

        log_interval = max(1, T // 10) if log_interval is None else max(1, int(log_interval))
        effective_D_theta = self.d_theta_scale * float(D_theta)

        theta_bar_history = []
        psi_history = []
        diagnostics_history = []

        use_tqdm = bool(tqdm_print) and not verbose
        iterator = trange(T, desc="LinearSolver", disable=not use_tqdm)

        final_theta = torch.zeros(self.d_q, dtype=torch.float64, device=self.device)

        for t in iterator:
            if policy_optimizer == "adam":
                psi_t = adam_param.detach()

            pi_mat = self._linear_policy_matrix(psi_t)
            E_q_pi = (pi_mat[..., None] * self.Q_XA).sum(dim=1)

            # Build the theta-space empirical mismatch using the current beta
            # occupancy weights and the independent Q feature table.
            coeff = self.U_sample @ beta_t
            theta_mismatch = (1.0 - self.gamma) * E_q_pi[self.x0]
            theta_mismatch = theta_mismatch + (self.gamma / self.n) * (
                coeff[:, None] * E_q_pi[self.X_nexts]
            ).sum(dim=0)
            theta_mismatch = theta_mismatch - (self.Q_sample * coeff[:, None]).mean(dim=0)

            theta_t, _c_t, _lambda_theta, q_objective, theta_grad_norm, _theta_lr_used = (
                self._compute_theta_update(
                    theta_mismatch=theta_mismatch,
                    beta_t=beta_t,
                    effective_D_theta=effective_D_theta,
                )
            )

            q_next = torch.tensordot(self.Q_XA[self.X_nexts], theta_t, dims=([2], [0]))
            v = (pi_mat[self.X_nexts] * q_next).sum(dim=1)
            q_current = self.Q_sample @ theta_t
            q_all = torch.tensordot(self.Q_XA, theta_t, dims=([2], [0]))
            v_x0 = (pi_mat[self.x0] * q_all[self.x0]).sum()

            # This is the requested beta objective: empirical u_beta-weighted
            # TD residual under the current Q_theta and policy.
            td_error = self.Rs + self.gamma * v - q_current
            beta_objective = (coeff * td_error).mean()
            total_loss = (1.0 - self.gamma) * v_x0 + beta_objective

            # Beta is updated in u-feature space through the existing
            # ridge-preconditioned matrix.
            beta_grad = (self.U_sample.T @ td_error) / self.n
            beta_update_direction = self.H_inv @ beta_grad

            # State weights define the policy objective over Q values; clipping
            # matches the previous solver option for negative or tiny weights.
            state_weight_sums = torch.zeros(self.N, dtype=torch.float64, device=self.device)
            state_weight_sums.index_add_(0, self.X_nexts, coeff)
            state_weights = (self.gamma / self.n) * state_weight_sums
            state_weights[self.x0] = state_weights[self.x0] + (1.0 - self.gamma)
            if state_weight_update == "normal":
                policy_state_weights = state_weights
            else:
                policy_state_weights = torch.clamp(state_weights, min=c_min)

            if policy_gradient == "exact":
                policy_grad, policy_objective = self._exact_policy_gradient(
                    pi_mat,
                    q_all,
                    policy_state_weights,
                )
            else:
                policy_grad, policy_objective = self._reinforce_policy_gradient(
                    pi_mat,
                    q_all,
                    coeff,
                    state_weights,
                    policy_state_weights,
                    state_weight_update,
                    reinforce_samples,
                )

            if policy_optimizer == "sgd":
                psi_next = psi_t + alpha * policy_grad
                policy_direction = policy_grad
                policy_direction_kind = "sgd_gradient"
                policy_diagnostics = {}
            elif policy_optimizer == "adam":
                adam_optimizer.zero_grad(set_to_none=True)
                adam_param.grad = -policy_grad.clone()
                adam_optimizer.step()
                psi_next = adam_param.detach().clone()
                policy_direction = policy_grad
                policy_direction_kind = "adam_gradient"
                policy_diagnostics = {}
            else:
                policy_state_indices = self._policy_update_state_indices(state_weight_update)
                policy_direction, policy_diagnostics = self._conjugate_gradient_policy_direction(
                    policy_grad=policy_grad,
                    psi_t=psi_t,
                    state_indices=policy_state_indices,
                    fisher_damping=fisher_damping,
                    cg_iters=cg_iters,
                    cg_tol=cg_tol,
                )
                policy_direction_kind = "cg_fisher"
                psi_next = psi_t + alpha * policy_direction

            beta_t = (1.0 / (1.0 + rho * eta)) * (beta_t + eta * beta_update_direction)
            theta_bar_t = theta_bar_t + theta_t
            theta_bar_history.append(theta_bar_t.clone())
            psi_history.append(psi_next.clone())
            psi_t = psi_next
            final_theta = theta_t

            diagnostics = {
                "iter": int(t),
                "total_loss": float(total_loss.detach().cpu().item()),
                "policy_objective": float(policy_objective.detach().cpu().item()),
                "beta_objective": float(beta_objective.detach().cpu().item()),
                "q_objective": float(q_objective.detach().cpu().item()),
                "policy_grad_norm": float(torch.linalg.norm(policy_grad).detach().cpu().item()),
                "policy_direction_norm": float(
                    torch.linalg.norm(policy_direction).detach().cpu().item()
                ),
                "beta_grad_norm": float(torch.linalg.norm(beta_grad).detach().cpu().item()),
                "theta_grad_norm": float(theta_grad_norm),
                "policy_optimizer": policy_optimizer,
                "policy_gradient": policy_gradient,
                "policy_direction": policy_direction_kind,
                "reinforce_samples": int(reinforce_samples),
                "D_theta": float(D_theta),
                "effective_D_theta": float(effective_D_theta),
            }
            diagnostics.update(policy_diagnostics)
            diagnostics_history.append(diagnostics)

            if use_tqdm:
                # Tqdm intentionally shows only losses/objectives; full scalar
                # diagnostics are printed by verbose mode.
                iterator.set_postfix(
                    total_loss=f"{diagnostics['total_loss']:.3e}",
                    policy=f"{diagnostics['policy_objective']:.3e}",
                    beta=f"{diagnostics['beta_objective']:.3e}",
                    q=f"{diagnostics['q_objective']:.3e}",
                )

            if verbose and (t % log_interval == 0):
                values = " ".join(
                    f"{key}={value:.6e}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in diagnostics.items()
                )
                print(f"[LinearSolver] Iter {t + 1}/{T} {values}")

        self.theta = final_theta.clone()
        self.theta_bar_history = theta_bar_history
        self.psi_history = psi_history
        self.psi = psi_t.clone()
        self.pi = self._linear_policy_matrix(self.psi)
        self.lambda_T = beta_t.clone()
        self.beta_T = beta_t.clone()
        self.diagnostics_history = diagnostics_history

        return self.pi
