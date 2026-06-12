import random

import numpy as np
import torch
from tqdm import trange

from ...fogas_clean.fogas_dataset import FOGASDataset
from ..fogas_parameters import GeneralizedFOGASParameters
from ..policy_features import build_policy_feature_table
from ..u_functions import build_u_feature_table
from .vbeta_logit_solver import VBetaLogitSolver


class LinearBetaPiSolver:
    """
    Standalone linear-u, linear-policy FOGAS solver.

    This class does not depend on an mdp object. The beta/value side is defined
    by u_function, and the policy side is defined by policy_features.
    """

    _row_softmax = staticmethod(VBetaLogitSolver._row_softmax)
    _policy_entropy = staticmethod(VBetaLogitSolver._policy_entropy)
    _state_weight_sign_diagnostics = staticmethod(VBetaLogitSolver._state_weight_sign_diagnostics)

    def __init__(
        self,
        n_states,
        n_actions,
        gamma,
        x0,
        csv_path,
        u_function,
        policy_features,
        delta=0.05,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        D_theta=None,
        beta_reg=None,
        print_params=False,
        dataset_verbose=False,
        seed=42,
        device=None,
    ):
        if u_function is None:
            raise ValueError("u_function must be provided")
        if policy_features is None:
            raise ValueError("policy_features must be provided")

        self.N = int(n_states)
        self.A = int(n_actions)
        self.gamma = float(gamma)
        self.x0 = int(x0)
        self.csv_path = csv_path
        self.u_function = u_function
        self.policy_features = policy_features
        self.delta = delta
        self.seed = seed

        if self.N <= 0:
            raise ValueError("n_states must be positive")
        if self.A <= 0:
            raise ValueError("n_actions must be positive")
        if self.x0 < 0 or self.x0 >= self.N:
            raise ValueError(f"x0 must be in [0, {self.N}), got {self.x0}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)

        self.dataset = FOGASDataset(csv_path=csv_path, verbose=dataset_verbose)
        self.Xs = self.dataset.X.to(self.device).long()
        self.As = self.dataset.A.to(self.device).long()
        self.Rs = self.dataset.R.to(dtype=torch.float64, device=self.device)
        self.X_nexts = self.dataset.X_next.to(self.device).long()
        self.n = self.dataset.n

        self._validate_dataset_indices()
        self._build_u_tensors()
        self._build_policy_tensors()
        self._build_preconditioner(beta_reg)
        self._estimate_omega()
        self._compute_reward_bound()

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
            D_theta=D_theta,
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

        self.theta_bar_history = None
        self.pi = None
        self.mod_alpha = self.alpha
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
        U_XA = build_u_feature_table(
            self.u_function,
            self.N,
            self.A,
            device=self.device,
            dtype=torch.float64,
        )
        self.U_XA = U_XA
        self.U_sample = U_XA[self.Xs, self.As]
        self.d = int(U_XA.shape[2])

    def _build_policy_tensors(self):
        OMEGA_PI_XA = build_policy_feature_table(
            self.policy_features,
            self.N,
            self.A,
            device=self.device,
            dtype=torch.float64,
        )
        self.OMEGA_PI_XA = OMEGA_PI_XA
        self.d_pi = int(OMEGA_PI_XA.shape[2])

    def _build_preconditioner(self, beta_reg):
        d = self.d
        if beta_reg is None:
            beta_reg = 1.0
        beta_reg = float(beta_reg)
        if beta_reg < 0:
            raise ValueError("beta_reg must be non-negative")
        self._preconditioner_beta_reg = beta_reg
        self.Empirical_cov = (self.U_sample.T @ self.U_sample) / self.n
        self.H = beta_reg * torch.eye(d, dtype=torch.float64, device=self.device)
        self.H = self.H + self.Empirical_cov
        self.H_inv = torch.linalg.inv(self.H)
        self.Cov_emp = self.H
        self.Cov_emp_inv = self.H_inv

    def _estimate_omega(self):
        self.omega = self.H_inv @ ((self.U_sample.T @ self.Rs) / self.n)

    def _compute_reward_bound(self):
        self.r_hat = torch.tensordot(self.U_XA, self.omega, dims=([2], [0]))
        R = torch.max(torch.abs(self.r_hat))
        self.R = float(max(R.detach().cpu().item(), 1e-12))

    def _prepare_linear_psi_init(self, psi_init):
        if psi_init is None:
            return torch.zeros(self.d_pi, dtype=torch.float64, device=self.device)

        psi_t = psi_init.clone().to(dtype=torch.float64, device=self.device).reshape(-1)
        if psi_t.shape != (self.d_pi,):
            raise ValueError(f"psi_init must have shape ({self.d_pi},), got {tuple(psi_t.shape)}")
        return psi_t

    @staticmethod
    def _canonical_policy_optimizer(policy_optimizer):
        name = str(policy_optimizer).lower()
        if name not in {"sgd", "adam"}:
            raise ValueError("policy_optimizer must be either 'sgd' or 'adam'")
        return name

    def _linear_policy_matrix(self, psi_t):
        logits = torch.tensordot(self.OMEGA_PI_XA, psi_t, dims=([2], [0]))
        return self._row_softmax(logits)

    def _compute_theta_update(self, emp_feature_occupancy, beta_t, D_theta):
        c_t = emp_feature_occupancy - (self.H @ beta_t)
        norm_c = torch.linalg.norm(c_t)
        theta_t = torch.zeros_like(c_t) if norm_c < 1e-12 else -D_theta * c_t / norm_c
        diagnostics = {
            "theta_update": "linear_closed_form",
            "theta_loss_include_beta_reg": True,
            "theta_loss_mismatch_norm": float(norm_c.detach().cpu().item()),
        }
        return theta_t, c_t, norm_c, diagnostics

    def get_diagnostics(self):
        return self.diagnostics_history

    def run(
        self,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        D_theta=None,
        beta_init=None,
        theta_bar_init=None,
        psi_init=None,
        policy_optimizer="sgd",
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        print_policies=False,
        verbose=False,
        tqdm_print=False,
        log_interval=None,
        state_weight_update="normal",
        c_min=0.1,
    ):
        T = self.params.T if T is None else T
        alpha = self.params.alpha if alpha is None else alpha
        eta = self.params.eta if eta is None else eta
        rho = self.params.rho if rho is None else rho
        D_theta = self.params.D_theta if D_theta is None else D_theta
        policy_optimizer = self._canonical_policy_optimizer(policy_optimizer)

        self.mod_alpha = alpha
        self.policy_optimizer_name = policy_optimizer

        N, d = self.N, self.d
        n = self.n
        gamma = self.gamma
        U_XA = self.U_XA
        U_sample = self.U_sample
        Rs = self.Rs
        Xn = self.X_nexts
        x0 = int(self.x0)
        H_inv = self.H_inv
        omega = self.omega
        OMEGA_PI_XA = self.OMEGA_PI_XA
        device = self.device

        beta_t = torch.zeros(d, dtype=torch.float64, device=device) if beta_init is None else beta_init.clone().to(device)
        beta_t = beta_t.reshape(-1)
        if beta_t.shape != (d,):
            raise ValueError(f"beta_init must have shape ({d},), got {tuple(beta_t.shape)}")
        theta_bar_t = (
            torch.zeros(d, dtype=torch.float64, device=device)
            if theta_bar_init is None
            else theta_bar_init.clone().to(dtype=torch.float64, device=device).reshape(-1)
        )
        if theta_bar_t.shape != (d,):
            raise ValueError(f"theta_bar_init must have shape ({d},), got {tuple(theta_bar_t.shape)}")
        psi_t = self._prepare_linear_psi_init(psi_init)

        adam_param = None
        adam_optimizer = None
        if policy_optimizer == "adam":
            adam_param = torch.nn.Parameter(psi_t.clone())
            adam_optimizer = torch.optim.Adam([adam_param], lr=alpha, betas=adam_betas, eps=adam_eps)

        theta_bar_history = []
        psi_history = []
        diagnostics_history = []

        log_interval = max(1, T // 10) if log_interval is None else max(1, int(log_interval))
        if state_weight_update not in {"normal", "clipped"}:
            raise ValueError("state_weight_update must be either 'normal' or 'clipped'")

        use_tqdm = not verbose and not print_policies and tqdm_print
        iterator = trange(T, desc="FOGAS", disable=not use_tqdm)

        for t in iterator:
            if policy_optimizer == "adam":
                psi_t = adam_param.detach()
            pi_mat = self._linear_policy_matrix(psi_t)
            E_u_pi = (pi_mat[..., None] * U_XA).sum(dim=1)

            lambda_emp_sum1 = (1.0 - gamma) * E_u_pi[x0]

            coeff = U_sample @ beta_t
            inner = E_u_pi[Xn]
            lambda_emp_sum2 = (gamma / n) * (coeff[:, None] * inner).sum(dim=0)
            emp_feature_occupancy = lambda_emp_sum1 + lambda_emp_sum2

            theta_t, c_t, norm_c, theta_diagnostics = self._compute_theta_update(
                emp_feature_occupancy=emp_feature_occupancy,
                beta_t=beta_t,
                D_theta=D_theta,
            )

            q_next = torch.tensordot(U_XA[Xn], theta_t, dims=([2], [0]))
            v = (pi_mat[Xn] * q_next).sum(dim=1)
            q_current = U_sample @ theta_t
            q_all = torch.tensordot(U_XA, theta_t, dims=([2], [0]))
            v_x0 = (pi_mat[x0] * q_all[x0]).sum()
            sampled_loss = (coeff * (Rs + gamma * v - q_current)).mean()
            empirical_loss = (1.0 - gamma) * v_x0 + sampled_loss
            sum_term = (U_sample * v[:, None]).sum(dim=0)
            Psi_hat_v = (H_inv @ sum_term) / n

            g = omega + gamma * Psi_hat_v - theta_t

            state_weight_sums = torch.zeros(N, dtype=torch.float64, device=device)
            state_weight_sums.index_add_(0, Xn, coeff)
            state_weights = (gamma / n) * state_weight_sums
            state_weights[x0] = state_weights[x0] + (1.0 - gamma)
            if state_weight_update == "normal":
                policy_state_weights = state_weights
            else:
                policy_state_weights = torch.clamp(state_weights, min=c_min)
            G = policy_state_weights[:, None] * q_all

            policy_objective = (pi_mat * G).sum()
            V_G = (pi_mat * G).sum(dim=1)
            advantage_G = G - V_G[:, None]
            policy_grad = (pi_mat[..., None] * advantage_G[..., None] * OMEGA_PI_XA).sum(dim=(0, 1))
            policy_grad_norm = torch.linalg.norm(policy_grad)

            if policy_optimizer == "sgd":
                psi_next = psi_t + alpha * policy_grad
            else:
                adam_optimizer.zero_grad(set_to_none=True)
                adam_param.grad = -policy_grad.clone()
                adam_optimizer.step()
                psi_next = adam_param.detach().clone()

            pi_next = self._linear_policy_matrix(psi_next)
            objective_policy_part_next = (pi_next * G).sum()

            beta_t = (1.0 / (1.0 + rho * eta)) * (beta_t + eta * g)

            theta_bar_t = theta_bar_t + theta_t
            theta_bar_history.append(theta_bar_t.clone())
            psi_history.append(psi_next.clone())

            policy_max_delta = torch.max(torch.abs(pi_next - pi_mat))
            psi_t = psi_next

            diagnostics = {
                "iter": t,
                "beta_norm": float(torch.linalg.norm(beta_t).detach().cpu().item()),
                "theta_norm": float(torch.linalg.norm(theta_t).detach().cpu().item()),
                "c_norm": float(norm_c.detach().cpu().item()),
                "g_norm": float(torch.linalg.norm(g).detach().cpu().item()),
                "policy_max_delta": float(policy_max_delta.detach().cpu().item()),
                "policy_mean_entropy": float(self._policy_entropy(pi_next).detach().cpu().item()),
                "policy_optimizer": policy_optimizer,
                "policy_objective": float(policy_objective.detach().cpu().item()),
                "psi_norm": float(torch.linalg.norm(psi_next).detach().cpu().item()),
                "psi_max_abs": float(torch.max(torch.abs(psi_next)).detach().cpu().item()),
                "total_loss": float(empirical_loss.detach().cpu().item()),
                "loss": float(empirical_loss.detach().cpu().item()),
                "empirical_objective": float(empirical_loss.detach().cpu().item()),
                "sampled_loss": float(sampled_loss.detach().cpu().item()),
                "initial_state_loss": float(((1.0 - gamma) * v_x0).detach().cpu().item()),
                "policy_loss": float(policy_objective.detach().cpu().item()),
                "policy_grad_norm": float(policy_grad_norm.detach().cpu().item()),
                "objective_policy_part": float(policy_objective.detach().cpu().item()),
                "objective_policy_part_next": float(objective_policy_part_next.detach().cpu().item()),
                "objective_policy_improvement": float(
                    (objective_policy_part_next - policy_objective).detach().cpu().item()
                ),
                "min_state_weight": float(state_weights.min().detach().cpu().item()),
                "max_state_weight": float(state_weights.max().detach().cpu().item()),
                "min_policy_state_weight": float(policy_state_weights.min().detach().cpu().item()),
                "max_policy_state_weight": float(policy_state_weights.max().detach().cpu().item()),
                "state_weight_update": state_weight_update,
            }
            diagnostics.update(theta_diagnostics)
            diagnostics_history.append(diagnostics)

            if print_policies and (t % log_interval == 0):
                print(f"\nIteration {t + 1}")
                print(pi_next.detach().cpu())

            if verbose and (t % log_interval == 0):
                print(
                    f"[FOGAS linear-beta-pi] Iter {t + 1}/{T} "
                    f"total_loss={diagnostics['total_loss']:.6e} "
                    f"policy_objective={diagnostics['policy_objective']:.6e} "
                    f"theta_norm={diagnostics['theta_norm']:.6e} "
                    f"beta_norm={diagnostics['beta_norm']:.6e} "
                    f"grad_norm={diagnostics['g_norm']:.6e} "
                    f"policy_grad_norm={diagnostics['policy_grad_norm']:.6e} "
                    f"psi_norm={diagnostics['psi_norm']:.6e} "
                    f"policy_optimizer={diagnostics['policy_optimizer']} "
                    f"state_weight_update={diagnostics['state_weight_update']} "
                    f"state_weight_min={diagnostics['min_state_weight']:.6e} "
                    f"state_weight_max={diagnostics['max_state_weight']:.6e} "
                    f"policy_state_weight_min={diagnostics['min_policy_state_weight']:.6e} "
                    f"policy_state_weight_max={diagnostics['max_policy_state_weight']:.6e}"
                )

        self.theta_bar_history = theta_bar_history
        self.psi_history = psi_history
        self.psi = psi_t.clone()
        self.pi = self._linear_policy_matrix(self.psi)
        self.lambda_T = beta_t
        self.beta_T = beta_t
        self.diagnostics_history = diagnostics_history

        return self.pi
