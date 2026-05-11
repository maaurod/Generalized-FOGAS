import torch
import random
import numpy as np

from ...fogas.fogas_dataset import FOGASDataset
from ...fogas.fogas_parameters import FOGASParameters
from tqdm import trange


class BetaSolver:
    """
    FOGAS implementation: runs the optimization algorithm, stores θ̄-history and final π.
    Evaluation utilities are moved to FOGASEvaluator.
    Supports CUDA acceleration.
    """

    def __init__(
        self,
        mdp,
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
        self.mdp = mdp
        self.csv_path = csv_path
        self.csv_path_omega = csv_path_omega if csv_path_omega is not None else csv_path
        self.delta = delta
        self.seed = seed

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
        self.d = mdp.d
        self.gamma = mdp.gamma
        self.R = mdp.R
        self.phi = mdp.phi
        
        # Handle omega: use from MDP if available, otherwise estimate from dataset
        if mdp.omega is not None:
            self.omega = mdp.omega.to(self.device) if isinstance(mdp.omega, torch.Tensor) else torch.tensor(mdp.omega, dtype=torch.float64, device=self.device)
        else:
            if print_params:
                print("MDP omega not provided. Estimating from dataset...")
            self.omega = None
        self.x0 = mdp.x0

        # ------------------------------
        # Theoretical parameters
        # ------------------------------
        if print_params:
            print(f"\nDevice: {self.device}")
            print(f"Dataset: {csv_path} (n={self.n})")

        self.params = FOGASParameters(
            mdp=mdp,
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

        # Build covariance matrices
        self._build_covariances()

        # Estimate omega if needed
        if self.omega is None:
            self._estimate_omega()
            if print_params:
                print(f"Estimated omega (first 5 components): {self.omega[:5]}")

        # Results to be filled by run()
        self.theta_bar_history = None
        self.pi = None
        self.mod_alpha = self.alpha
        self.beta_T = None

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------
    def _build_covariances(self):
        n = self.n
        d = self.d
        beta = self.beta
        phi = self.phi
        Xs = self.Xs
        As = self.As

        Phi_list = [phi(int(x.item()), int(a.item())).to(dtype=torch.float64, device=self.device) 
                    for x, a in zip(Xs, As)]
        Phi = torch.vstack(Phi_list)
        Cov_emp = beta * torch.eye(d, dtype=torch.float64, device=self.device) + (Phi.T @ Phi) / n
        Cov_emp_inv = torch.linalg.inv(Cov_emp)

        self.Phi = Phi
        self.Cov_emp = Cov_emp
        self.Cov_emp_inv = Cov_emp_inv

    # ------------------------------------------------------------------
    # Estimate omega from dataset
    # ------------------------------------------------------------------
    def _estimate_omega(self):
        """
        Estimate omega from the omega dataset using regularized least squares.
        """
        # If the paths are the same, use precomputed tensors to save time
        if self.csv_path_omega == self.csv_path:
            n = self.n
            Phi = self.Phi
            R = self.Rs
            Cov_inv = self.Cov_emp_inv
        else:
            # Load the second dataset
            ds_omega = FOGASDataset(csv_path=self.csv_path_omega, verbose=False)
            Xs_o = ds_omega.X.to(self.device)
            As_o = ds_omega.A.to(self.device)
            R = ds_omega.R.to(self.device)
            n = ds_omega.n
            
            # Compute features for this dataset
            Phi_list = [self.phi(int(x.item()), int(a.item())).to(dtype=torch.float64, device=self.device) 
                        for x, a in zip(Xs_o, As_o)]
            Phi = torch.vstack(Phi_list)
            
            # Compute local covariance for estimation
            Cov = self.beta * torch.eye(self.d, dtype=torch.float64, device=self.device) + (Phi.T @ Phi) / n
            Cov_inv = torch.linalg.inv(Cov)
        
        # omega_hat = Cov_inv @ (Phi^T @ R / n)
        sum_phi_r = (Phi.T @ R) / n
        self.omega = Cov_inv @ sum_phi_r

    # ------------------------------------------------------------------
    # Softmax policy
    # ------------------------------------------------------------------
    def softmax_policy(self, theta_bar, alpha, return_matrix=False):
        phi = self.phi
        A = self.A
        N = self.N

        def compute_probs(x):
            logits = []
            for a in range(A):
                phi_val = phi(x, a).to(dtype=torch.float64)
                logits.append(alpha * torch.dot(phi_val, theta_bar))
            logits = torch.stack(logits)
            exp_logits = torch.exp(logits - torch.max(logits))
            return exp_logits / exp_logits.sum()

        if not return_matrix:
            def pi(x):
                return compute_probs(x)
            return pi
        else:
            out = torch.zeros((N, A), dtype=torch.float64, device=self.device)
            for x in range(N):
                out[x] = compute_probs(x)
            return out

    # ------------------------------------------------------------------
    # RUN FOGAS
    # ------------------------------------------------------------------
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
        tqdm_print=False
    ):
        # -------------------------
        # Override parameters
        # -------------------------
        T = self.params.T if T is None else T
        alpha = self.params.alpha if alpha is None else alpha
        eta = self.params.eta if eta is None else eta
        rho = self.params.rho if rho is None else rho
        D_theta = self.params.D_theta if D_theta is None else D_theta

        self.mod_alpha = alpha  # store alpha used

        Cov_emp, Cov_emp_inv = self.Cov_emp, self.Cov_emp_inv
        Phi = self.Phi
        phi = self.phi
        gamma = self.gamma
        omega = self.omega
        x0 = self.x0
        n = self.n
        d = self.d
        A = self.A
        X_nexts = self.X_nexts

        # -------------------------
        # Initialization
        # -------------------------
        device = self.device
        beta_t = torch.zeros(d, dtype=torch.float64, device=device) if beta_init is None else beta_init.clone().to(device)
        theta_bar_t = torch.zeros(d, dtype=torch.float64, device=device) if theta_bar_init is None else theta_bar_init.clone().to(device)
        theta_bar_history = []
        pi_t = lambda x: torch.ones(A, dtype=torch.float64, device=device) / A  # start uniform

        # -------------------------
        # Main loop
        # -------------------------
        use_tqdm = not verbose and not print_policies and tqdm_print
        iterator = trange(T, desc="FOGAS", disable=not use_tqdm)

        for t in iterator:

            # ---------------------------
            # μ̂ term
            # ---------------------------
            lambda_emp_sum1 = (1 - gamma) * sum(
                pi_t(x0)[a] * phi(x0, a).to(dtype=torch.float64, device=device) for a in range(A)
            )

            lambda_emp_sum2 = torch.zeros(d, dtype=torch.float64)

            for i in range(n):
                coeff = Phi[i] @ beta_t
                inner = sum(pi_t(int(X_nexts[i].item()))[a] * phi(int(X_nexts[i].item()), a).to(dtype=torch.float64, device=device)
                            for a in range(A))
                lambda_emp_sum2 += coeff * inner
            lambda_emp_sum2 *= gamma / n

            emp_feature_occupancy = lambda_emp_sum1 + lambda_emp_sum2

            # ---------------------------
            # θ_t update
            # ---------------------------
            c_t = emp_feature_occupancy - (Cov_emp @ beta_t)
            norm_c = torch.linalg.norm(c_t)
            theta_t = torch.zeros_like(c_t) if norm_c < 1e-12 else -D_theta * c_t / norm_c

            # ---------------------------
            # Ψ̂ v term
            # ---------------------------
            sum_term = torch.zeros(d, dtype=torch.float64)
            for i in range(n):
                probs = pi_t(int(X_nexts[i].item()))
                v = sum(
                    probs[a] * torch.dot(theta_t, phi(int(X_nexts[i].item()), a).to(dtype=torch.float64, device=device))
                    for a in range(A)
                )
                sum_term += Phi[i] * v

            Psi_hat_v = (1 / n) * (Cov_emp_inv @ sum_term)

            # ---------------------------
            # λ update
            # ---------------------------
            g = omega + gamma * Psi_hat_v - theta_t
            beta_t = (1 / (1 + rho * eta)) * (beta_t + eta * g)

            # ---------------------------
            # Policy update
            # ---------------------------
            theta_bar_t += theta_t
            theta_bar_history.append(theta_bar_t.clone())
            pi_t = self.softmax_policy(theta_bar_t, alpha)

            if print_policies and (t % max(1, T // 10) == 0):
                print(f"\nIteration {t+1}")
                pi_matrix = self.softmax_policy(theta_bar_t, alpha, return_matrix=True)
                self.mdp.print_policy(pi_matrix)

            if verbose and (t % max(1, T // 10) == 0):
                print(f"\n[FOGAS] Iter {t+1}/{T}")
                print(f"  θ_t     = {theta_t}")
                print(f"  ||θ_t|| = {torch.linalg.norm(theta_t).item():.3e}")
                print(f"  β_t     = {beta_t}")
                print(f"  ||β_t|| = {torch.linalg.norm(beta_t).item():.3e}")

    
        self.theta_bar_history = theta_bar_history
        self.pi = self.softmax_policy(theta_bar_t, alpha, return_matrix=True)
        self.beta_T = beta_t

        return pi_t


FOGASSolverBeta = BetaSolver
