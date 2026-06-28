import random

import numpy as np
import torch
from tqdm import trange

from ...fogas_clean.fogas_dataset import FOGASDataset
from ..features import build_policy_feature_table, build_q_feature_table


class PrimalAlgaeDICESolver:
    """
    Linear-feature primal AlgaeDICE solver for finite offline RL datasets.

    The critic is nu_theta(s, a) = <theta, q_features(s, a)> and the policy is
    pi_psi(a | s) = softmax_a(<psi, policy_features(s, a)> / temperature).
    With tabular one-hot Q features this is the quadratic primal AlgaeDICE
    algorithm used in tabular Four Rooms style experiments.
    """

    _CRITIC_UPDATES = {"closed_form", "batch_adam"}
    _EPS = 1e-12

    def __init__(
        self,
        n_states,
        n_actions,
        gamma,
        x0,
        csv_path,
        q_function,
        policy_features,
        u_function=None,
        T=100,
        alpha=0.01,
        ridge=1e-6,
        actor_lr=1e-2,
        batch_size=None,
        critic_update="closed_form",
        critic_lr=1e-2,
        critic_inner_steps=100,
        terminal_states=None,
        init_states=None,
        temperature=1.0,
        dataset_verbose=False,
        seed=42,
        device=None,
    ):
        if q_function is None:
            raise ValueError("q_function must be provided")
        if policy_features is None:
            raise ValueError("policy_features must be provided")

        self.N = int(n_states)
        self.A = int(n_actions)
        self.gamma = float(gamma)
        self.x0 = int(x0)
        self.csv_path = csv_path
        self.q_function = q_function
        self.policy_features = policy_features
        # Accepted for FinalLinearSolver-style construction compatibility.
        self.u_function = u_function
        self.seed = seed

        if self.N <= 0:
            raise ValueError("n_states must be positive")
        if self.A <= 0:
            raise ValueError("n_actions must be positive")
        if self.x0 < 0 or self.x0 >= self.N:
            raise ValueError(f"x0 must be in [0, {self.N}), got {self.x0}")

        self.T = self._canonical_positive_int(T, "T")
        self.alpha = self._canonical_positive_float(alpha, "alpha")
        self.ridge = self._canonical_nonnegative_float(ridge, "ridge")
        self.actor_lr = self._canonical_positive_float(actor_lr, "actor_lr")
        self.batch_size = self._canonical_optional_positive_int(batch_size, "batch_size")
        self.critic_update = self._canonical_critic_update(critic_update)
        self.critic_lr = self._canonical_positive_float(critic_lr, "critic_lr")
        self.critic_inner_steps = self._canonical_positive_int(
            critic_inner_steps,
            "critic_inner_steps",
        )
        self.temperature = self._canonical_positive_float(temperature, "temperature")
        self.terminal_states = self._canonical_terminal_states(terminal_states)
        self.init_states = self._canonical_init_states(init_states)

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

        self.dataset = FOGASDataset(csv_path=csv_path, verbose=dataset_verbose)
        self.Xs = self.dataset.X.to(self.device).long()
        self.As = self.dataset.A.to(self.device).long()
        self.Rs = self.dataset.R.to(dtype=torch.float64, device=self.device)
        self.X_nexts = self.dataset.X_next.to(self.device).long()
        self.n = self.dataset.n
        self.Ds = self._load_done_flags()
        self._validate_dataset_indices()

        self._build_q_tensors()
        self._build_policy_tensors()

        self.theta = None
        self.theta_history = None
        self.pi = None
        self.psi = None
        self.psi_history = None
        self.diagnostics_history = None
        self.policy_optimizer_name = None

    @classmethod
    def _canonical_critic_update(cls, critic_update):
        name = str(critic_update).lower()
        if name not in cls._CRITIC_UPDATES:
            raise ValueError("critic_update must be either 'closed_form' or 'batch_adam'")
        return name

    @staticmethod
    def _canonical_positive_float(value, name):
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _canonical_nonnegative_float(value, name):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @staticmethod
    def _canonical_positive_int(value, name):
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @classmethod
    def _canonical_optional_positive_int(cls, value, name):
        if value is None:
            return None
        return cls._canonical_positive_int(value, name)

    def _canonical_terminal_states(self, terminal_states):
        if terminal_states is None:
            return set()
        states = {int(state) for state in terminal_states}
        invalid = [state for state in states if state < 0 or state >= self.N]
        if invalid:
            raise ValueError(f"terminal_states contain values outside [0, {self.N}): {invalid}")
        return states

    def _canonical_init_states(self, init_states):
        if init_states is None:
            states = [self.x0]
        else:
            states = [int(state) for state in init_states]
            if not states:
                raise ValueError("init_states must contain at least one state")

        invalid = [state for state in states if state < 0 or state >= self.N]
        if invalid:
            raise ValueError(f"init_states contain values outside [0, {self.N}): {invalid}")
        return torch.as_tensor(states, dtype=torch.long)

    def _load_done_flags(self):
        dones = torch.zeros(self.n, dtype=torch.bool, device=self.device)

        if self._dataset_has_column("done"):
            done_values = self._dataset_column_values("done")
            dones = torch.as_tensor(
                [self._parse_bool(value) for value in done_values],
                dtype=torch.bool,
                device=self.device,
            )

        if self.terminal_states:
            terminal_tensor = torch.as_tensor(
                sorted(self.terminal_states),
                dtype=torch.long,
                device=self.device,
            )
            dones = dones | torch.isin(self.X_nexts, terminal_tensor)

        return dones

    def _dataset_has_column(self, column):
        df = getattr(self.dataset, "df", None)
        if df is None:
            return False
        if hasattr(df, "columns"):
            return column in df.columns
        if isinstance(df, list) and df:
            return column in df[0]
        return False

    def _dataset_column_values(self, column):
        df = getattr(self.dataset, "df", None)
        if hasattr(df, "__getitem__") and hasattr(df, "columns"):
            return df[column].to_list()
        return [row[column] for row in df]

    @staticmethod
    def _parse_bool(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value) != 0
        if isinstance(value, (float, np.floating)):
            return float(value) != 0.0
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n", ""}:
            return False
        raise ValueError(f"Cannot parse done value as bool: {value!r}")

    def _validate_dataset_indices(self):
        if torch.any((self.Xs < 0) | (self.Xs >= self.N)):
            raise ValueError("dataset states contain values outside [0, n_states)")
        if torch.any((self.X_nexts < 0) | (self.X_nexts >= self.N)):
            raise ValueError("dataset next states contain values outside [0, n_states)")
        if torch.any((self.As < 0) | (self.As >= self.A)):
            raise ValueError("dataset actions contain values outside [0, n_actions)")
        if self.Ds.shape != (self.n,):
            raise ValueError(f"done flags must have shape ({self.n},), got {tuple(self.Ds.shape)}")

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
        self.init_states = self.init_states.to(self.device)

    @staticmethod
    def _row_softmax(logits):
        shifted = logits - logits.max(dim=1, keepdim=True).values
        exp = torch.exp(shifted)
        return exp / exp.sum(dim=1, keepdim=True)

    def _policy_matrix(self, psi_t):
        logits = torch.tensordot(self.OMEGA_PI_XA, psi_t, dims=([2], [0]))
        return self._row_softmax(logits / self.temperature)

    def _prepare_psi_init(self, psi_init):
        if psi_init is None:
            return torch.zeros(self.d_pi, dtype=torch.float64, device=self.device)
        psi_t = psi_init.clone().to(dtype=torch.float64, device=self.device).reshape(-1)
        if psi_t.shape != (self.d_pi,):
            raise ValueError(f"psi_init must have shape ({self.d_pi},), got {tuple(psi_t.shape)}")
        return psi_t

    def _prepare_theta_init(self, theta_init):
        if theta_init is None:
            return torch.zeros(self.d_q, dtype=torch.float64, device=self.device)
        theta_t = theta_init.clone().to(dtype=torch.float64, device=self.device).reshape(-1)
        if theta_t.shape != (self.d_q,):
            raise ValueError(
                f"theta_init must have shape ({self.d_q},), got {tuple(theta_t.shape)}"
            )
        return theta_t

    def _expected_q_features(self, pi_mat, states):
        return (pi_mat[states][..., None] * self.Q_XA[states]).sum(dim=1)

    def _initial_feature(self, pi_mat):
        return self._expected_q_features(pi_mat, self.init_states).mean(dim=0)

    def _bellman_feature_matrix(self, pi_mat, indices=None):
        if indices is None:
            q_current = self.Q_sample
            next_states = self.X_nexts
            dones = self.Ds
        else:
            q_current = self.Q_sample[indices]
            next_states = self.X_nexts[indices]
            dones = self.Ds[indices]

        next_features = self._expected_q_features(pi_mat, next_states)
        nonterminal = (~dones).to(dtype=torch.float64)[:, None]
        return q_current - self.gamma * nonterminal * next_features

    def _closed_form_critic_update(self, pi_mat):
        F = self._bellman_feature_matrix(pi_mat)
        b = self._initial_feature(pi_mat)
        eye = torch.eye(self.d_q, dtype=torch.float64, device=self.device)
        lhs = (F.T @ F) / self.n + self.ridge * eye
        rhs = (F.T @ self.Rs) / self.n - self.alpha * (1.0 - self.gamma) * b
        try:
            return torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            return torch.linalg.lstsq(lhs, rhs[:, None]).solution[:, 0]

    def _sample_batch_indices(self):
        if self.batch_size is None or self.batch_size >= self.n:
            return None
        return torch.randint(self.n, (int(self.batch_size),), device=self.device)

    def _critic_objective(self, theta_t, pi_mat, indices=None):
        if indices is None:
            rewards = self.Rs
        else:
            rewards = self.Rs[indices]

        F = self._bellman_feature_matrix(pi_mat, indices=indices)
        delta = rewards - F @ theta_t
        nu_init = torch.dot(self._initial_feature(pi_mat), theta_t)
        objective = (1.0 - self.gamma) * nu_init + delta.square().mean() / (
            2.0 * self.alpha
        )
        ridge_penalty = 0.5 * self.ridge * torch.dot(theta_t, theta_t) / self.alpha
        return objective, delta, objective + ridge_penalty

    def _batch_adam_critic_update(self, pi_mat, theta_init):
        theta_param = torch.nn.Parameter(theta_init.detach().clone())
        optimizer = torch.optim.Adam([theta_param], lr=self.critic_lr)

        for _ in range(self.critic_inner_steps):
            indices = self._sample_batch_indices()
            optimizer.zero_grad(set_to_none=True)
            _objective, _delta, critic_loss = self._critic_objective(
                theta_param,
                pi_mat,
                indices=indices,
            )
            critic_loss.backward()
            optimizer.step()

        return theta_param.detach().clone()

    def _actor_objective(self, psi_param, theta_t):
        pi_mat = self._policy_matrix(psi_param)
        theta_detached = theta_t.detach()
        q_all = torch.tensordot(self.Q_XA, theta_detached, dims=([2], [0]))
        nu_init = (pi_mat[self.init_states] * q_all[self.init_states]).sum(dim=1).mean()
        next_v = (pi_mat[self.X_nexts] * q_all[self.X_nexts]).sum(dim=1)
        current_nu = q_all[self.Xs, self.As]
        nonterminal = (~self.Ds).to(dtype=torch.float64)
        delta = self.Rs + self.gamma * nonterminal * next_v - current_nu
        objective = (1.0 - self.gamma) * nu_init + delta.square().mean() / (
            2.0 * self.alpha
        )
        return objective, delta

    def run(
        self,
        T=None,
        alpha=None,
        ridge=None,
        actor_lr=None,
        batch_size=None,
        critic_update=None,
        critic_lr=None,
        critic_inner_steps=None,
        psi_init=None,
        theta_init=None,
        eta=None,
        rho=None,
        policy_optimizer=None,
        policy_gradient=None,
        reinforce_samples=None,
        fisher_damping=None,
        cg_iters=None,
        cg_tol=None,
        state_weight_update=None,
        c_min=None,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        verbose=False,
        tqdm_print=False,
        log_interval=None,
    ):
        del eta, rho, policy_optimizer, policy_gradient
        del reinforce_samples, fisher_damping, cg_iters, cg_tol
        del state_weight_update, c_min

        T = self.T if T is None else self._canonical_positive_int(T, "T")
        previous_alpha = self.alpha
        previous_ridge = self.ridge
        previous_actor_lr = self.actor_lr
        previous_batch_size = self.batch_size
        previous_critic_update = self.critic_update
        previous_critic_lr = self.critic_lr
        previous_critic_inner_steps = self.critic_inner_steps

        if alpha is not None:
            self.alpha = self._canonical_positive_float(alpha, "alpha")
        if ridge is not None:
            self.ridge = self._canonical_nonnegative_float(ridge, "ridge")
        if actor_lr is not None:
            self.actor_lr = self._canonical_positive_float(actor_lr, "actor_lr")
        if batch_size is not None:
            self.batch_size = self._canonical_positive_int(batch_size, "batch_size")
        if critic_update is not None:
            self.critic_update = self._canonical_critic_update(critic_update)
        if critic_lr is not None:
            self.critic_lr = self._canonical_positive_float(critic_lr, "critic_lr")
        if critic_inner_steps is not None:
            self.critic_inner_steps = self._canonical_positive_int(
                critic_inner_steps,
                "critic_inner_steps",
            )

        try:
            return self._run_impl(
                T=T,
                psi_init=psi_init,
                theta_init=theta_init,
                adam_betas=adam_betas,
                adam_eps=adam_eps,
                verbose=verbose,
                tqdm_print=tqdm_print,
                log_interval=log_interval,
            )
        finally:
            self.alpha = previous_alpha
            self.ridge = previous_ridge
            self.actor_lr = previous_actor_lr
            self.batch_size = previous_batch_size
            self.critic_update = previous_critic_update
            self.critic_lr = previous_critic_lr
            self.critic_inner_steps = previous_critic_inner_steps

    def _run_impl(
        self,
        T,
        psi_init,
        theta_init,
        adam_betas,
        adam_eps,
        verbose,
        tqdm_print,
        log_interval,
    ):
        self.policy_optimizer_name = "adam"
        psi_t = self._prepare_psi_init(psi_init)
        theta_t = self._prepare_theta_init(theta_init)
        psi_param = torch.nn.Parameter(psi_t.clone())
        actor_optimizer = torch.optim.Adam(
            [psi_param],
            lr=self.actor_lr,
            betas=adam_betas,
            eps=adam_eps,
        )

        theta_history = []
        psi_history = []
        diagnostics_history = []
        log_interval = max(1, T // 10) if log_interval is None else max(1, int(log_interval))

        use_tqdm = bool(tqdm_print) and not verbose
        iterator = trange(T, desc="PrimalAlgaeDICESolver", disable=not use_tqdm)

        for t in iterator:
            pi_detached = self._policy_matrix(psi_param.detach())
            if self.critic_update == "closed_form":
                theta_t = self._closed_form_critic_update(pi_detached)
            else:
                theta_t = self._batch_adam_critic_update(pi_detached, theta_t)

            actor_optimizer.zero_grad(set_to_none=True)
            objective, actor_delta = self._actor_objective(psi_param, theta_t)
            actor_loss = -objective
            actor_loss.backward()
            policy_grad = psi_param.grad.detach().clone()
            actor_optimizer.step()

            with torch.no_grad():
                psi_t = psi_param.detach().clone()
                pi_current = self._policy_matrix(psi_t)
                critic_objective, critic_delta, critic_loss = self._critic_objective(
                    theta_t,
                    pi_current,
                )
                theta_history.append(theta_t.clone())
                psi_history.append(psi_t.clone())

                diagnostics = {
                    "iter": int(t),
                    "objective": float(objective.detach().cpu().item()),
                    "actor_loss": float(actor_loss.detach().cpu().item()),
                    "critic_loss": float(critic_loss.detach().cpu().item()),
                    "critic_objective": float(critic_objective.detach().cpu().item()),
                    "actor_delta_mean": float(actor_delta.mean().detach().cpu().item()),
                    "actor_delta_std": float(actor_delta.std(unbiased=False).detach().cpu().item()),
                    "critic_delta_mean": float(critic_delta.mean().detach().cpu().item()),
                    "critic_delta_std": float(
                        critic_delta.std(unbiased=False).detach().cpu().item()
                    ),
                    "policy_grad_norm": float(torch.linalg.norm(policy_grad).detach().cpu().item()),
                    "theta_norm": float(torch.linalg.norm(theta_t).detach().cpu().item()),
                    "psi_norm": float(torch.linalg.norm(psi_t).detach().cpu().item()),
                    "alpha": float(self.alpha),
                    "ridge": float(self.ridge),
                    "actor_lr": float(self.actor_lr),
                    "critic_update": self.critic_update,
                    "critic_lr": float(self.critic_lr),
                    "critic_inner_steps": int(self.critic_inner_steps),
                    "batch_size": None if self.batch_size is None else int(self.batch_size),
                    "done_fraction": float(self.Ds.to(dtype=torch.float64).mean().cpu().item()),
                }
                diagnostics_history.append(diagnostics)

            if use_tqdm:
                iterator.set_postfix(
                    objective=f"{diagnostics['objective']:.3e}",
                    critic=f"{diagnostics['critic_loss']:.3e}",
                    grad=f"{diagnostics['policy_grad_norm']:.3e}",
                )

            if verbose and (t % log_interval == 0):
                values = " ".join(
                    f"{key}={value:.6e}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in diagnostics.items()
                )
                print(f"[PrimalAlgaeDICESolver] Iter {t + 1}/{T} {values}")

        self.theta = theta_t.clone()
        self.theta_history = theta_history
        self.psi = psi_t.clone()
        self.psi_history = psi_history
        self.pi = self._policy_matrix(self.psi).detach().clone()
        self.diagnostics_history = diagnostics_history

        return self.pi

    def get_diagnostics(self):
        return self.diagnostics_history
