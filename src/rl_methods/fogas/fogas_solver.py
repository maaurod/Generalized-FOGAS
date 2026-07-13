"""
Empirical FOGAS solver for finite state-action experiments.

This module contains the main original FOGAS training loop. It differs from the
dataset and evaluator modules by owning optimization state: feature tensors,
empirical covariance, reward-weight estimation, lambda/theta updates, and the
learned stochastic policy. It is used directly in the tabular gridworld and
discretized Mountain Car FOGAS experiments.
"""

import torch
import random
import numpy as np

from .fogas_dataset import FOGASDataset
from .fogas_parameters import FOGASParameters
from tqdm import trange


class FOGASSolver:
    """
    FOGAS implementation: precomputes PHI tensors and runs the
    core loop with batch operations. Supports CUDA acceleration.
    """

    def __init__(
        self,
        mdp,
        phi,
        csv_path,
        delta=0.05,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        D_theta=None,
        beta=None,
        print_params=False,
        dataset_verbose=False,
        seed=42,
        device=None,
        csv_path_omega=None,
        beta_omega=None,
    ):
        self.mdp = mdp
        self.delta = delta
        self.seed = seed
        self.csv_path = csv_path
        self.csv_path_omega = csv_path_omega if csv_path_omega is not None else csv_path
        self.phi = phi

        # Set device (CUDA if available)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)

        # Move MDP to device
        self.mdp.to(self.device)

        # ------------------------------
        # Dataset
        # ------------------------------
        self.dataset = FOGASDataset(csv_path=csv_path, verbose=dataset_verbose)
        self.Xs = self.dataset.X.to(self.device)
        self.As = self.dataset.A.to(self.device)
        self.Rs = self.dataset.R.to(self.device)
        self.X_nexts = self.dataset.X_next.to(self.device)
        self.n = self.dataset.n

        # ------------------------------
        # MDP info
        # ------------------------------
        self.N = mdp.N
        self.A = mdp.A
        self.gamma = mdp.gamma
        self.x0 = mdp.x0
        self._beta_omega = beta_omega

        # Precompute feature tensors before parameters: d and R are solver-owned.
        self._build_feature_tensors()

        # ------------------------------
        # Theoretical parameters
        # ------------------------------
        if print_params:
            print(f"\nDevice: {self.device}")
            print(f"Dataset: {csv_path} (n={self.n})")

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
            beta=beta,
            print_params=print_params,
        )

        self.T = self.params.T
        self.alpha = self.params.alpha
        self.eta = self.params.eta
        self.rho = self.params.rho
        self.D_theta = self.params.D_theta
        self.beta = self.params.beta
        self.D_pi = self.params.D_pi

        # Precompute empirical covariance
        self._build_covariances()

        # ------------------------------------------------------------------
        # Omega resolution
        # ------------------------------------------------------------------
        # • mdp.has_omega and beta_omega is None → use known reward weights
        # • otherwise                           → estimate omega from data
        # ------------------------------------------------------------------
        has_known_omega = bool(getattr(mdp, "has_omega", getattr(mdp, "omega", None) is not None))
        if has_known_omega and self._beta_omega is None:
            self.omega = self._coerce_omega(mdp.omega)
        else:
            if print_params:
                print("Known omega not provided. Estimating from dataset...")
            self._estimate_omega(beta_omega=self._beta_omega)
            if print_params:
                print(f"Estimated omega (first 5 components): {self.omega[:5]}")

        # Results to be filled by run()
        self.theta_bar_history = None
        self.pi = None
        self.mod_alpha = self.alpha
        self.lambda_T = None

    # ------------------------------------------------------------------
    # Feature tensors
    # ------------------------------------------------------------------
    def _build_feature_tensors(self):
        # PHI_XA[x, a] = phi(x, a)
        PHI_XA = torch.stack(
            [torch.stack([self.phi(x, a) for a in range(self.A)]) for x in range(self.N)],
            dim=0,
        ).to(dtype=torch.float64, device=self.device)

        Phi = PHI_XA.reshape(self.N * self.A, -1)
        Phi_data = PHI_XA[self.Xs.long(), self.As.long()]

        self.PHI_XA = PHI_XA
        self.Phi = Phi
        self.Phi_data = Phi_data
        self.d = int(Phi.shape[1])
        self.R = float(torch.linalg.norm(Phi, dim=1).max().item())

    def _coerce_omega(self, omega):
        if omega is None:
            raise ValueError("mdp.has_omega is true but mdp.omega is None")
        out = omega.to(dtype=torch.float64, device=self.device) if isinstance(omega, torch.Tensor) else torch.tensor(omega, dtype=torch.float64, device=self.device)
        out = out.reshape(-1)
        if out.shape != (self.d,):
            raise ValueError(f"mdp.omega must have shape ({self.d},), got {tuple(out.shape)}")
        return out

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------
    def _build_covariances(self):
        n = self.n
        d = self.d
        beta = self.beta
        Phi = self.Phi_data

        Cov_emp = beta * torch.eye(d, dtype=torch.float64, device=self.device) + (Phi.T @ Phi) / n
        Cov_emp_inv = torch.linalg.inv(Cov_emp)

        self.Cov_emp = Cov_emp
        self.Cov_emp_inv = Cov_emp_inv

    # ------------------------------------------------------------------
    # Estimate omega from dataset
    # ------------------------------------------------------------------
    def _estimate_omega(self, beta_omega=None):
        """
        Estimate omega via ridge regression on the dataset:
            omega_hat = (Phi^T Phi + beta_omega * n * I)^{-1} Phi^T r

        Parameters
        ----------
        beta_omega : float or None
            Regularization for the regression. If None, uses the theoretical
            beta = R^2 / (d * T) from FOGASParameters.
        """
        reg = self.beta if beta_omega is None else beta_omega

        if self.csv_path_omega == self.csv_path:
            n = self.n
            Phi = self.Phi_data
            R = self.Rs
        else:
            ds_omega = FOGASDataset(csv_path=self.csv_path_omega, verbose=False)
            X_o = ds_omega.X.to(self.device).long()
            A_o = ds_omega.A.to(self.device).long()
            R = ds_omega.R.to(self.device)
            n = ds_omega.n
            Phi = self.PHI_XA[X_o, A_o]

        Cov = reg * torch.eye(self.d, dtype=torch.float64, device=self.device) + (Phi.T @ Phi) / n
        Cov_inv = torch.linalg.inv(Cov)
        self.omega = Cov_inv @ (Phi.T @ R / n)
        print(f"[FOGASSolver] omega estimated via regression "
              f"(beta_omega={reg:.2e}, n={n})")


    # ------------------------------------------------------------------
    # Softmax policy (matrix)
    # ------------------------------------------------------------------
    @staticmethod
    def _row_softmax(logits):
        z = logits - logits.max(dim=1, keepdim=True).values
        ez = torch.exp(z)
        return ez / ez.sum(dim=1, keepdim=True)

    def softmax_policy(self, theta_bar, alpha, return_matrix=True):
        theta_bar = theta_bar.to(dtype=torch.float64, device=self.device)
        logits = alpha * torch.tensordot(self.PHI_XA, theta_bar, dims=([2], [0]))
        pi_mat = self._row_softmax(logits)
        if return_matrix:
            return pi_mat

        def pi(x):
            return pi_mat[int(x)]

        return pi

    # ------------------------------------------------------------------
    # RUN FOGAS
    # ------------------------------------------------------------------
    def run(
        self,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        gamma=None,
        D_theta=None,
        lambda_init=None,
        theta_bar_init=None,
        print_policies=False,
        verbose=False,
        tqdm_print=False,
    ):
        # -------------------------
        # Override parameters
        # -------------------------
        T = self.params.T if T is None else T
        alpha = self.params.alpha if alpha is None else alpha
        eta = self.params.eta if eta is None else eta
        rho = self.params.rho if rho is None else rho
        gamma = self.gamma if gamma is None else gamma
        D_theta = self.params.D_theta if D_theta is None else D_theta

        self.mod_alpha = alpha  # store alpha used

        N, A, d = self.N, self.A, self.d
        n = self.n
        PHI_XA = self.PHI_XA
        Phi_data = self.Phi_data
        Xn = self.X_nexts.long()
        x0 = int(self.x0)

        Cov = self.Cov_emp
        Cov_inv = self.Cov_emp_inv
        omega = self.omega

        device = self.device

        # -------------------------
        # Initialization
        # -------------------------
        lambda_t = torch.zeros(d, dtype=torch.float64, device=device) if lambda_init is None else lambda_init.clone().to(device)
        theta_bar_t = torch.zeros(d, dtype=torch.float64, device=device) if theta_bar_init is None else theta_bar_init.clone().to(device)
        theta_bar_history = []

        # -------------------------
        # Main loop
        # -------------------------
        use_tqdm = not verbose and not print_policies and tqdm_print
        iterator = trange(T, desc="FOGAS", disable=not use_tqdm)

        for t in iterator:
            # ---------- Policy matrix π_t (N,A) ----------
            logits = alpha * torch.tensordot(PHI_XA, theta_bar_t, dims=([2], [0]))
            pi_mat = self._row_softmax(logits)

            # E_phi_pi[x] = sum_a pi(x,a) * phi(x,a)
            E_phi_pi = (pi_mat[..., None] * PHI_XA).sum(dim=1)

            # ---------- μ̂ term ----------
            lambda_emp_sum1 = (1.0 - gamma) * E_phi_pi[x0]

            Lambda_term = Cov_inv @ lambda_t
            coeff = Phi_data @ Lambda_term
            inner = E_phi_pi[Xn]
            lambda_emp_sum2 = (gamma / n) * (coeff[:, None] * inner).sum(dim=0)
            emp_feature_occupancy = lambda_emp_sum1 + lambda_emp_sum2

            # ---------- θ_t update ----------
            c_t = emp_feature_occupancy - lambda_t
            norm_c = torch.linalg.norm(c_t)
            theta_t = torch.zeros_like(c_t) if norm_c < 1e-12 else -D_theta * c_t / norm_c

            # ---------- Ψ̂ v term ----------
            q = torch.tensordot(PHI_XA[Xn], theta_t, dims=([2], [0]))
            v = (pi_mat[Xn] * q).sum(dim=1)
            sum_term = (Phi_data * v[:, None]).sum(dim=0)
            Psi_hat_v = (Cov_inv @ sum_term) / n

            # ---------- λ update ----------
            g = omega + gamma * Psi_hat_v - theta_t
            lambda_t = (1.0 / (1.0 + rho * eta)) * (lambda_t + eta * (Cov @ g))

            # ---------- θ̄ update ----------
            theta_bar_t = theta_bar_t + theta_t
            theta_bar_history.append(theta_bar_t.clone())

            if print_policies and (t % max(1, T // 10) == 0):
                print(f"\nIteration {t+1}")
                self.mdp.print_policy(pi_mat.cpu())

            if verbose and (t % max(1, T // 10) == 0):
                print(f"\n[FOGAS] Iter {t+1}/{T}")
                print(f"  θ_t     = {theta_t}")
                print(f"  ||θ_t|| = {torch.linalg.norm(theta_t).item():.3e}")
                print(f"  λ_t     = {lambda_t}")
                print(f"  ||λ_t|| = {torch.linalg.norm(lambda_t).item():.3e}")

        self.theta_bar_history = theta_bar_history
        self.pi = pi_mat
        self.lambda_T = lambda_t

        return pi_mat
