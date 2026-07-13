"""
Oracle FOGAS variant for small MDPs with exact planning information.

Unlike FOGASSolver, this implementation uses a Planner to obtain exact
discounted occupancy measures instead of estimating the corresponding terms
from an offline dataset. It is useful for algorithm checks and small tabular
experiments where exact MDP quantities are available.
"""

import random

import numpy as np
import torch
from tqdm import trange

from .fogas_parameters import FOGASParameters


class FOGASOracleSolver:
    """
    Oracle FOGAS implementation for the clean DiscreteMDP/Planner split.

    The oracle receives a Planner, not a bare DiscreteMDP, because it needs exact
    discounted occupancy measures. Feature information is supplied explicitly
    through phi instead of being stored on the MDP.
    """

    def __init__(
        self,
        planner,
        phi,
        csv_path_omega=None,
        delta=0.05,
        n=None,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        D_theta=None,
        print_params=False,
        cov_matrix="identity",
        seed=42,
        device=None,
    ):
        if not hasattr(planner, "occupancy_measure") or not hasattr(planner, "mdp"):
            raise TypeError("FOGASOracleSolver expects a Planner. Pass Planner(mdp), not a bare MDP.")

        self.planner = planner
        self.mdp = planner.mdp
        self.phi = phi
        self.csv_path_omega = csv_path_omega
        self.delta = delta
        self.seed = seed
        self.n = int(10_000_000 if n is None else n)
        self.cov_matrix = cov_matrix

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

        if hasattr(self.planner, "to"):
            self.planner.to(self.device)
        elif hasattr(self.mdp, "to"):
            self.mdp.to(self.device)

        self.N = int(self.mdp.N)
        self.A = int(self.mdp.A)
        self.gamma = float(self.mdp.gamma)
        self.x0 = int(self.mdp.x0)

        self._build_feature_tensors()
        self._build_reward_feature()

        self.params = FOGASParameters(
            N=self.N,
            A=self.A,
            gamma=self.gamma,
            d=self.d,
            R=self.R,
            n=self.n,
            delta=delta,
            T=T,
            alpha=alpha,
            eta=eta,
            rho=rho,
            D_theta=D_theta,
            print_params=print_params,
        )

        self.T = self.params.T
        self.alpha = self.params.alpha
        self.eta = self.params.eta
        self.rho = self.params.rho if n is not None else 1e-2
        self.D_theta = self.params.D_theta
        self.beta = self.params.beta
        self.D_pi = self.params.D_pi

        if self.csv_path_omega is not None:
            self._estimate_omega_from_csv()
            self.r_feat = self.Phi.T @ (self.Phi @ self.omega)
        else:
            self.omega = self._ridge_reward_weights()

        self.theta_bar_history = None
        self.pi = None
        self.mod_alpha = self.alpha
        self.lambda_T = None

    def _build_feature_tensors(self):
        phi_xa = torch.stack(
            [torch.stack([self.phi(x, a) for a in range(self.A)]) for x in range(self.N)],
            dim=0,
        ).to(dtype=torch.float64, device=self.device)
        self.PHI_XA = phi_xa
        self.Phi = phi_xa.reshape(self.N * self.A, -1)
        self.d = int(self.Phi.shape[1])
        self.R = float(torch.linalg.norm(self.Phi, dim=1).max().item())

    def _build_reward_feature(self):
        rewards = self.mdp.r
        rewards = rewards.to(dtype=torch.float64, device=self.device) if isinstance(rewards, torch.Tensor) else torch.tensor(rewards, dtype=torch.float64, device=self.device)
        rewards = rewards.reshape(self.N * self.A)
        self.r_feat = self.Phi.T @ rewards

    def _ridge_reward_weights(self):
        eye = torch.eye(self.d, dtype=torch.float64, device=self.device)
        cov = self.beta * eye + (self.Phi.T @ self.Phi) / self.n
        return torch.linalg.solve(cov, self.r_feat / self.n)

    def _estimate_omega_from_csv(self):
        from .fogas_dataset import FOGASDataset

        dataset = FOGASDataset(csv_path=self.csv_path_omega, verbose=False)
        x = dataset.X.to(self.device).long()
        a = dataset.A.to(self.device).long()
        r = dataset.R.to(self.device)
        phi_data = self.PHI_XA[x, a]
        n = dataset.n

        cov = self.beta * torch.eye(self.d, dtype=torch.float64, device=self.device)
        cov = cov + (phi_data.T @ phi_data) / n
        self.omega = torch.linalg.solve(cov, (phi_data.T @ r) / n)

    @staticmethod
    def _row_softmax(logits):
        z = logits - logits.max(dim=1, keepdim=True).values
        exp_z = torch.exp(z)
        return exp_z / exp_z.sum(dim=1, keepdim=True)

    def softmax_policy(self, theta_bar, alpha, return_matrix=True):
        theta_bar = theta_bar.to(dtype=torch.float64, device=self.device)
        logits = alpha * torch.tensordot(self.PHI_XA, theta_bar, dims=([2], [0]))
        pi_mat = self._row_softmax(logits)
        if return_matrix:
            return pi_mat

        def pi(x):
            return pi_mat[int(x)]

        return pi

    def _occupancy_measure(self, policy_matrix):
        policy_for_planner = policy_matrix.detach()
        if self.device.type == "cuda":
            policy_for_planner = policy_for_planner.cpu()
        occupancy = self.planner.occupancy_measure(policy_for_planner)
        if isinstance(occupancy, torch.Tensor):
            return occupancy.to(dtype=torch.float64, device=self.device)
        return torch.tensor(occupancy, dtype=torch.float64, device=self.device)

    def _transition_value_feature(self, v):
        P = self.mdp.P
        P = P.to(dtype=torch.float64, device=self.device) if isinstance(P, torch.Tensor) else torch.tensor(P, dtype=torch.float64, device=self.device)
        expected_v = P @ v
        return self.Phi.T @ expected_v

    def _covariance(self, cov_matrix, occupancy):
        if cov_matrix == "identity":
            return torch.eye(self.d, dtype=torch.float64, device=self.device)

        if cov_matrix == "cov_uniform":
            mu = torch.ones(self.N * self.A, dtype=torch.float64, device=self.device) / (self.N * self.A)
        elif cov_matrix == "cov_opt":
            mu = self.planner.mu_star
            mu = mu.to(dtype=torch.float64, device=self.device) if isinstance(mu, torch.Tensor) else torch.tensor(mu, dtype=torch.float64, device=self.device)
        elif cov_matrix == "cov_dynamic":
            mu = occupancy
        else:
            raise ValueError("cov_matrix must be 'identity', 'cov_uniform', 'cov_opt', or 'cov_dynamic'.")

        eye = torch.eye(self.d, dtype=torch.float64, device=self.device)
        return self.beta * eye + self.Phi.T @ (mu[:, None] * self.Phi)

    def run(
        self,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        n=None,
        D_theta=None,
        lambda_init=None,
        theta_bar_init=None,
        print_policies=False,
        verbose=False,
        tqdm_print=False,
        cov_matrix=None,
    ):
        T = self.params.T if T is None else T
        alpha = self.params.alpha if alpha is None else alpha
        eta = self.params.eta if eta is None else eta
        rho = self.params.rho if rho is None else rho
        D_theta = self.params.D_theta if D_theta is None else D_theta
        cov_matrix = self.cov_matrix if cov_matrix is None else cov_matrix

        self.mod_alpha = alpha

        lambda_t = (
            torch.zeros(self.d, dtype=torch.float64, device=self.device)
            if lambda_init is None
            else lambda_init.clone().to(dtype=torch.float64, device=self.device)
        )
        theta_bar_t = (
            torch.zeros(self.d, dtype=torch.float64, device=self.device)
            if theta_bar_init is None
            else theta_bar_init.clone().to(dtype=torch.float64, device=self.device)
        )
        theta_bar_history = []

        use_tqdm = not verbose and not print_policies and tqdm_print
        iterator = trange(T, desc="FOGAS Oracle", disable=not use_tqdm)

        for t in iterator:
            pi_matrix = self.softmax_policy(theta_bar=theta_bar_t, alpha=alpha, return_matrix=True)
            occupancy = self._occupancy_measure(pi_matrix)
            true_feature_occupancy = self.Phi.T @ occupancy

            c_t = true_feature_occupancy - lambda_t
            norm_c = torch.linalg.norm(c_t)
            theta_t = torch.zeros_like(c_t) if norm_c < 1e-12 else -D_theta * c_t / norm_c

            q_theta = torch.tensordot(self.PHI_XA, theta_t, dims=([2], [0]))
            v_theta_pi = (pi_matrix * q_theta).sum(dim=1)
            transition_value_feature = self._transition_value_feature(v_theta_pi)

            cov = self._covariance(cov_matrix, occupancy)
            g = self.r_feat + self.gamma * transition_value_feature - theta_t
            lambda_t = (1.0 / (1.0 + rho * eta)) * (lambda_t + eta * (cov @ g))

            theta_bar_t = theta_bar_t + theta_t
            theta_bar_history.append(theta_bar_t.clone())

            if print_policies and (t % max(1, T // 10) == 0):
                print(f"\nIteration {t + 1}")
                self.mdp.print_policy(pi_matrix.detach().cpu())

            if verbose and (t % max(1, T // 10) == 0):
                print(f"\n[FOGAS Oracle] Iter {t + 1}/{T}")
                print(f"  ||theta_t|| = {torch.linalg.norm(theta_t).item():.3e}")
                print(f"  ||lambda_t|| = {torch.linalg.norm(lambda_t).item():.3e}")

        self.theta_bar_history = theta_bar_history
        self.pi = pi_matrix
        self.lambda_T = lambda_t

        return pi_matrix
