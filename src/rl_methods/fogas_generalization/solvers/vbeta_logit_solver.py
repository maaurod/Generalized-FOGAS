import torch

from tqdm import trange

from .vbeta_solver import VBetaSolver


class VBetaLogitSolver(VBetaSolver):
    """
    Vectorized beta-parameter FOGAS variant with an explicit tabular softmax
    policy parameterization.

    The beta/theta updates match the vectorized beta solver logic. The policy is
    represented by one logit per state-action pair:

        pi_t(a|x) = softmax_a(psi_t[x, a])
        psi_{t+1}(x, a) = psi_t(x, a) + alpha * G_t(x, a)
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
        super().__init__(
            mdp=mdp,
            csv_path=csv_path,
            csv_path_omega=csv_path_omega,
            delta=delta,
            T=T,
            alpha=alpha,
            eta=eta,
            rho=rho,
            D_theta=D_theta,
            beta=beta,
            print_params=print_params,
            dataset_verbose=dataset_verbose,
            seed=seed,
            device=device,
        )
        self.diagnostics_history = None
        self.psi = None
        self.psi_history = None

    @staticmethod
    def _policy_entropy(pi_mat):
        safe_pi = torch.clamp(pi_mat, min=1e-300)
        return -(safe_pi * torch.log(safe_pi)).sum(dim=1).mean()

    @staticmethod
    def _state_weight_sign_diagnostics(state_weights, tol=0.0, max_states=50):
        negative_mask = state_weights < -tol
        positive_mask = state_weights > tol
        zero_mask = ~(negative_mask | positive_mask)

        negative_states = torch.nonzero(negative_mask, as_tuple=False).flatten()
        positive_states = torch.nonzero(positive_mask, as_tuple=False).flatten()
        zero_states = torch.nonzero(zero_mask, as_tuple=False).flatten()

        shown_negative_states = negative_states[:max_states].detach().cpu().tolist()
        shown_positive_states = positive_states[:max_states].detach().cpu().tolist()
        shown_zero_states = zero_states[:max_states].detach().cpu().tolist()

        signs = torch.where(
            negative_mask,
            torch.full_like(state_weights, -1, dtype=torch.int8),
            torch.where(
                positive_mask,
                torch.full_like(state_weights, 1, dtype=torch.int8),
                torch.zeros_like(state_weights, dtype=torch.int8),
            ),
        )

        return {
            "negative_count": int(negative_mask.sum().detach().cpu().item()),
            "positive_count": int(positive_mask.sum().detach().cpu().item()),
            "zero_count": int(zero_mask.sum().detach().cpu().item()),
            "shown_negative_states": shown_negative_states,
            "shown_positive_states": shown_positive_states,
            "shown_zero_states": shown_zero_states,
            "signs": signs[:max_states].detach().cpu().tolist(),
        }

    def _prepare_psi_init(self, psi_init):
        if psi_init is None:
            return torch.zeros((self.N, self.A), dtype=torch.float64, device=self.device)

        psi_t = psi_init.clone().to(dtype=torch.float64, device=self.device)
        if psi_t.shape == (self.N * self.A,):
            psi_t = psi_t.reshape(self.N, self.A)
        elif psi_t.shape != (self.N, self.A):
            raise ValueError(
                "psi_init must have shape "
                f"({self.N}, {self.A}) or ({self.N * self.A},), got {tuple(psi_t.shape)}"
            )
        return psi_t

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
        print_policies=False,
        verbose=False,
        tqdm_print=False,
        log_interval=None,
        verbose_state_weights=False,
        state_weight_sign_tol=0.0,
        max_verbose_state_weights=50,
        state_weight_update="normal",
        c_min=0.1,
    ):
        T = self.params.T if T is None else T
        alpha = self.params.alpha if alpha is None else alpha
        eta = self.params.eta if eta is None else eta
        rho = self.params.rho if rho is None else rho
        D_theta = self.params.D_theta if D_theta is None else D_theta

        self.mod_alpha = alpha

        N, A, d = self.N, self.A, self.d
        n = self.n
        gamma = self.gamma
        PHI_XA = self.PHI_XA
        Phi = self.Phi
        Rs = self.Rs
        Xn = self.X_nexts.long()
        x0 = int(self.x0)

        Cov_emp = self.Cov_emp
        Cov_emp_inv = self.Cov_emp_inv
        omega = self.omega

        device = self.device

        beta_t = torch.zeros(d, dtype=torch.float64, device=device) if beta_init is None else beta_init.clone().to(device)
        theta_bar_t = torch.zeros(d, dtype=torch.float64, device=device) if theta_bar_init is None else theta_bar_init.clone().to(device)
        psi_t = self._prepare_psi_init(psi_init)
        theta_bar_history = []
        psi_history = []
        diagnostics_history = []

        log_interval = max(1, T // 10) if log_interval is None else max(1, int(log_interval))
        if state_weight_update not in {"normal", "clipped"}:
            raise ValueError("state_weight_update must be either 'normal' or 'clipped'")

        use_tqdm = not verbose and not print_policies and tqdm_print
        iterator = trange(T, desc="FOGAS", disable=not use_tqdm)

        for t in iterator:
            pi_mat = self._row_softmax(psi_t)
            E_phi_pi = (pi_mat[..., None] * PHI_XA).sum(dim=1)

            lambda_emp_sum1 = (1.0 - gamma) * E_phi_pi[x0]

            coeff = Phi @ beta_t
            inner = E_phi_pi[Xn]
            lambda_emp_sum2 = (gamma / n) * (coeff[:, None] * inner).sum(dim=0)
            emp_feature_occupancy = lambda_emp_sum1 + lambda_emp_sum2

            c_t = emp_feature_occupancy - (Cov_emp @ beta_t)
            norm_c = torch.linalg.norm(c_t)
            theta_t = torch.zeros_like(c_t) if norm_c < 1e-12 else -D_theta * c_t / norm_c

            q_next = torch.tensordot(PHI_XA[Xn], theta_t, dims=([2], [0]))
            v = (pi_mat[Xn] * q_next).sum(dim=1)
            q_current = Phi @ theta_t
            q_all = torch.tensordot(PHI_XA, theta_t, dims=([2], [0]))
            v_x0 = (pi_mat[x0] * q_all[x0]).sum()
            sampled_loss = (coeff * (Rs + gamma * v - q_current)).mean()
            empirical_loss = (1.0 - gamma) * v_x0 + sampled_loss
            sum_term = (Phi * v[:, None]).sum(dim=0)
            Psi_hat_v = (Cov_emp_inv @ sum_term) / n

            g = omega + gamma * Psi_hat_v - theta_t

            state_weight_sums = torch.zeros(N, dtype=torch.float64, device=device)
            state_weight_sums.index_add_(0, Xn, coeff)
            state_weights = (gamma / n) * state_weight_sums
            state_weights[x0] = state_weights[x0] + (1.0 - gamma)
            state_weight_signs = self._state_weight_sign_diagnostics(
                state_weights,
                tol=state_weight_sign_tol,
                max_states=max_verbose_state_weights,
            )
            if state_weight_update == "normal":
                policy_state_weights = state_weights
            else:
                policy_state_weights = torch.clamp(state_weights, min=c_min)
            G = policy_state_weights[:, None] * q_all
            policy_grad_norm = torch.linalg.norm(G)

            objective_policy_part = (pi_mat * G).sum()
            psi_next = psi_t + alpha * G
            pi_next = self._row_softmax(psi_next)
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
                "psi_norm": float(torch.linalg.norm(psi_next).detach().cpu().item()),
                "psi_max_abs": float(torch.max(torch.abs(psi_next)).detach().cpu().item()),
                "total_loss": float(empirical_loss.detach().cpu().item()),
                "loss": float(empirical_loss.detach().cpu().item()),
                "empirical_objective": float(empirical_loss.detach().cpu().item()),
                "sampled_loss": float(sampled_loss.detach().cpu().item()),
                "initial_state_loss": float(((1.0 - gamma) * v_x0).detach().cpu().item()),
                "policy_loss": float(objective_policy_part.detach().cpu().item()),
                "policy_grad_norm": float(policy_grad_norm.detach().cpu().item()),
                "objective_policy_part": float(objective_policy_part.detach().cpu().item()),
                "objective_policy_part_next": float(objective_policy_part_next.detach().cpu().item()),
                "objective_policy_improvement": float((objective_policy_part_next - objective_policy_part).detach().cpu().item()),
                "min_state_weight": float(state_weights.min().detach().cpu().item()),
                "max_state_weight": float(state_weights.max().detach().cpu().item()),
                "min_policy_state_weight": float(policy_state_weights.min().detach().cpu().item()),
                "max_policy_state_weight": float(policy_state_weights.max().detach().cpu().item()),
                "state_weight_update": state_weight_update,
            }
            diagnostics_history.append(diagnostics)

            if print_policies and (t % log_interval == 0):
                print(f"\nIteration {t+1}")
                self.mdp.print_policy(pi_next.cpu())

            if verbose and (t % log_interval == 0):
                print(
                    f"[FOGAS logit-policy] Iter {t+1}/{T} "
                    f"total_loss={diagnostics['total_loss']:.6e} "
                    f"policy_loss={diagnostics['policy_loss']:.6e} "
                    f"theta_norm={diagnostics['theta_norm']:.6e} "
                    f"beta_norm={diagnostics['beta_norm']:.6e} "
                    f"grad_norm={diagnostics['g_norm']:.6e} "
                    f"policy_grad_norm={diagnostics['policy_grad_norm']:.6e} "
                    f"psi_norm={diagnostics['psi_norm']:.6e} "
                    f"state_weight_update={diagnostics['state_weight_update']} "
                    f"state_weight_min={diagnostics['min_state_weight']:.6e} "
                    f"state_weight_max={diagnostics['max_state_weight']:.6e} "
                    f"policy_state_weight_min={diagnostics['min_policy_state_weight']:.6e} "
                    f"policy_state_weight_max={diagnostics['max_policy_state_weight']:.6e}"
                )
                if verbose_state_weights:
                    print(
                        "  state_weight_signs "
                        f"first_{len(state_weight_signs['signs'])}={state_weight_signs['signs']} "
                        f"(+={state_weight_signs['positive_count']}, "
                        f"0={state_weight_signs['zero_count']}, "
                        f"-={state_weight_signs['negative_count']})"
                    )
                    if state_weight_signs["negative_count"] > 0:
                        print(
                            "  negative_state_weight_states "
                            f"first_{len(state_weight_signs['shown_negative_states'])}="
                            f"{state_weight_signs['shown_negative_states']}"
                        )

        self.theta_bar_history = theta_bar_history
        self.psi_history = psi_history
        self.psi = psi_t
        self.pi = self._row_softmax(psi_t)
        self.lambda_T = beta_t
        self.diagnostics_history = diagnostics_history

        return self.pi
