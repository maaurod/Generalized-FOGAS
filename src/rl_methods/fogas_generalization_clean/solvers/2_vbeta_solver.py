import torch
import random
import numpy as np

from ...fogas_clean.fogas_dataset import FOGASDataset
from ...fogas_clean.fogas_parameters import FOGASParameters
from tqdm import trange


class VBetaSolver:
    """
    Vectorized beta-parameter FOGAS implementation: precomputes PHI tensors and
    runs the core loop with batch operations. Supports CUDA acceleration.
    """

    def __init__(
        self,
        mdp,
        phi,
        csv_path,
        csv_path_omega=None,
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
    ):
        if phi is None:
            raise ValueError("phi must be provided")
        self.mdp = mdp
        self.phi = phi
        self.csv_path = csv_path
        self.csv_path_omega = csv_path_omega if csv_path_omega is not None else csv_path
        self.delta = delta
        self.seed = seed

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

        if hasattr(self.mdp, "to"):
            self.mdp.to(self.device)

        self.dataset = FOGASDataset(csv_path=csv_path, verbose=dataset_verbose)
        self.Xs = self.dataset.X.to(self.device).long()
        self.As = self.dataset.A.to(self.device).long()
        self.Rs = self.dataset.R.to(dtype=torch.float64, device=self.device)
        self.X_nexts = self.dataset.X_next.to(self.device).long()
        self.n = self.dataset.n

        self.N = int(mdp.N)
        self.A = int(mdp.A)
        self.gamma = float(mdp.gamma)
        self.x0 = int(mdp.x0)
        self._validate_dataset_indices()
        self._build_feature_tensors()

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

        self._build_covariances()

        if self._has_known_omega():
            self.omega = self._coerce_omega(self.mdp.omega)
        else:
            if print_params:
                print("Known omega not provided. Estimating from dataset...")
            self._estimate_omega()
            if print_params:
                print(f"Estimated omega (first 5 components): {self.omega[:5]}")

        self.theta_bar_history = None
        self.pi = None
        self.mod_alpha = self.alpha
        self.lambda_T = None

    def _validate_dataset_indices(self):
        if torch.any((self.Xs < 0) | (self.Xs >= self.N)):
            raise ValueError("dataset states contain values outside [0, mdp.N)")
        if torch.any((self.X_nexts < 0) | (self.X_nexts >= self.N)):
            raise ValueError("dataset next states contain values outside [0, mdp.N)")
        if torch.any((self.As < 0) | (self.As >= self.A)):
            raise ValueError("dataset actions contain values outside [0, mdp.A)")

    def _has_known_omega(self):
        return bool(getattr(self.mdp, "has_omega", getattr(self.mdp, "omega", None) is not None))

    def _coerce_omega(self, omega):
        if omega is None:
            raise ValueError("mdp.has_omega is true but mdp.omega is None")
        out = (
            omega.to(dtype=torch.float64, device=self.device)
            if isinstance(omega, torch.Tensor)
            else torch.tensor(omega, dtype=torch.float64, device=self.device)
        )
        out = out.reshape(-1)
        if out.shape != (self.d,):
            raise ValueError(f"mdp.omega must have shape ({self.d},), got {tuple(out.shape)}")
        return out

    def _build_feature_tensors(self):
        PHI_XA = torch.stack(
            [
                torch.stack(
                    [
                        torch.as_tensor(
                            self.phi(x, a),
                            dtype=torch.float64,
                            device=self.device,
                        ).reshape(-1)
                        for a in range(self.A)
                    ]
                )
                for x in range(self.N)
            ],
            dim=0,
        ).to(dtype=torch.float64, device=self.device)

        Phi = PHI_XA[self.Xs, self.As]

        self.PHI_XA = PHI_XA
        self.Phi = Phi
        self.d = int(PHI_XA.shape[2])
        self.R = float(max(torch.linalg.norm(PHI_XA.reshape(self.N * self.A, self.d), dim=1).max().item(), 1e-12))

    def _build_covariances(self):
        n = self.n
        d = self.d
        beta = self.beta
        Phi = self.Phi

        Cov_emp = beta * torch.eye(d, dtype=torch.float64, device=self.device) + (Phi.T @ Phi) / n
        Cov_emp_inv = torch.linalg.inv(Cov_emp)

        self.Cov_emp = Cov_emp
        self.Cov_emp_inv = Cov_emp_inv

    def _estimate_omega(self):
        """
        Estimate omega from the omega dataset using regularized least squares.
        """
        if self.csv_path_omega == self.csv_path:
            n = self.n
            Phi = self.Phi
            R = self.Rs
            Cov_inv = self.Cov_emp_inv
        else:
            ds_omega = FOGASDataset(csv_path=self.csv_path_omega, verbose=False)
            Xs_o = ds_omega.X.to(self.device).long()
            As_o = ds_omega.A.to(self.device).long()
            R = ds_omega.R.to(dtype=torch.float64, device=self.device)
            n = ds_omega.n

            Phi = self.PHI_XA[Xs_o, As_o]
            Cov = self.beta * torch.eye(self.d, dtype=torch.float64, device=self.device) + (Phi.T @ Phi) / n
            Cov_inv = torch.linalg.inv(Cov)

        sum_phi_r = (Phi.T @ R) / n
        self.omega = Cov_inv @ sum_phi_r

    @staticmethod
    def _row_softmax(logits):
        z = logits - logits.max(dim=1, keepdim=True).values
        ez = torch.exp(z)
        return ez / ez.sum(dim=1, keepdim=True)

    def run(
        self,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        D_theta=None,
        beta_init=None,
        theta_bar_init=None,
        print_policies=False,
        verbose=False,
        tqdm_print=False,
    ):
        T = self.params.T if T is None else T
        alpha = self.params.alpha if alpha is None else alpha
        eta = self.params.eta if eta is None else eta
        rho = self.params.rho if rho is None else rho
        D_theta = self.params.D_theta if D_theta is None else D_theta

        self.mod_alpha = alpha

        N, A, d = self.N, self.A, self.d
        n = self.n
        PHI_XA = self.PHI_XA
        Phi = self.Phi
        Xn = self.X_nexts.long()
        x0 = int(self.x0)

        Cov_emp = self.Cov_emp
        Cov_emp_inv = self.Cov_emp_inv
        omega = self.omega

        device = self.device

        beta_t = torch.zeros(d, dtype=torch.float64, device=device) if beta_init is None else beta_init.clone().to(device)
        theta_bar_t = torch.zeros(d, dtype=torch.float64, device=device) if theta_bar_init is None else theta_bar_init.clone().to(device)
        theta_bar_history = []

        use_tqdm = not verbose and not print_policies and tqdm_print
        iterator = trange(T, desc="FOGAS", disable=not use_tqdm)

        for t in iterator:
            logits = alpha * torch.tensordot(PHI_XA, theta_bar_t, dims=([2], [0]))
            pi_mat = self._row_softmax(logits)

            E_phi_pi = (pi_mat[..., None] * PHI_XA).sum(dim=1)

            lambda_emp_sum1 = (1.0 - self.gamma) * E_phi_pi[x0]

            coeff = Phi @ beta_t
            inner = E_phi_pi[Xn]
            lambda_emp_sum2 = (self.gamma / n) * (coeff[:, None] * inner).sum(dim=0)
            emp_feature_occupancy = lambda_emp_sum1 + lambda_emp_sum2

            c_t = emp_feature_occupancy - (Cov_emp @ beta_t)
            norm_c = torch.linalg.norm(c_t)
            theta_t = torch.zeros_like(c_t) if norm_c < 1e-12 else -D_theta * c_t / norm_c

            q = torch.tensordot(PHI_XA[Xn], theta_t, dims=([2], [0]))
            v = (pi_mat[Xn] * q).sum(dim=1)
            sum_term = (Phi * v[:, None]).sum(dim=0)
            Psi_hat_v = (Cov_emp_inv @ sum_term) / n

            g = omega + self.gamma * Psi_hat_v - theta_t
            beta_t = (1.0 / (1.0 + rho * eta)) * (beta_t + eta * g)

            theta_bar_t = theta_bar_t + theta_t
            theta_bar_history.append(theta_bar_t.clone())

            if print_policies and (t % max(1, T // 10) == 0):
                print(f"\nIteration {t+1}")
                self.mdp.print_policy(pi_mat.cpu())

            if verbose and (t % max(1, T // 10) == 0):
                print(f"\n[FOGAS] Iter {t+1}/{T}")
                print(f"  θ_t     = {theta_t}")
                print(f"  ||θ_t|| = {torch.linalg.norm(theta_t).item():.3e}")
                print(f"  β_t     = {beta_t}")
                print(f"  ||β_t|| = {torch.linalg.norm(beta_t).item():.3e}")

        self.theta_bar_history = theta_bar_history
        self.pi = pi_mat
        self.lambda_T = beta_t

        return pi_mat
