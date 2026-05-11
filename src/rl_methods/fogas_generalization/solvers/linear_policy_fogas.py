import torch

from tqdm import trange

from ..policy_features import TabularPolicyFeatures, build_policy_feature_table
from .vbeta_logit_solver import VBetaLogitSolver


class LinearPolicyFOGAS(VBetaLogitSolver):
    """
    Generalized FOGAS variant with a softmax-linear policy.

    The objective-induced score G_t(x, a) is computed as in VBetaLogitSolver,
    but the policy parameters are shared through

        logits(x, a) = <psi, omega_pi(x, a)>.

    The policy update performs gradient ascent on

        J_t(psi) = sum_x sum_a pi_psi(a|x) G_t(x, a).
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
        policy_features=None,
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
        if policy_features is None:
            policy_features = TabularPolicyFeatures(self.N, self.A)
        self.policy_features = policy_features
        self.OMEGA_PI_XA = build_policy_feature_table(
            policy_features,
            self.N,
            self.A,
            device=self.device,
            dtype=torch.float64,
        )
        self.d_pi = int(self.OMEGA_PI_XA.shape[2])
        self.policy_optimizer_name = None

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
        policy_optimizer = self._canonical_policy_optimizer(policy_optimizer)

        self.mod_alpha = alpha
        self.policy_optimizer_name = policy_optimizer

        N, d = self.N, self.d
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
        OMEGA_PI_XA = self.OMEGA_PI_XA

        device = self.device

        beta_t = torch.zeros(d, dtype=torch.float64, device=device) if beta_init is None else beta_init.clone().to(device)
        theta_bar_t = torch.zeros(d, dtype=torch.float64, device=device) if theta_bar_init is None else theta_bar_init.clone().to(device)
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
                "objective_policy_improvement": float((objective_policy_part_next - policy_objective).detach().cpu().item()),
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
                    f"[FOGAS linear-policy] Iter {t+1}/{T} "
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
        self.psi = psi_t.clone()
        self.pi = self._linear_policy_matrix(self.psi)
        self.lambda_T = beta_t
        self.diagnostics_history = diagnostics_history

        return self.pi
