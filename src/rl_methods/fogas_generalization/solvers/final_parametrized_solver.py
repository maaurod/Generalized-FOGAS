"""Reference Generalized FOGAS solver for finite state and action spaces.

The three optimization variables are ordinary PyTorch modules.  Linear
wrappers expose feature tables and use optimized tensor paths; nonlinear
wrappers use autograd and sample Jacobians.  This keeps a single reference
implementation for the linear/RBF and neural discrete experiments.
"""

import random

import numpy as np
import torch
from tqdm import trange

from ...fogas.fogas_dataset import FOGASDataset
from ..features import (
    LinearQParam,
    LinearUParam,
    SoftmaxLinearPolicyParam,
)
from ..fogas_parameters import GeneralizedFOGASParameters


class FinalParametrizedSolver:
    """Generalized FOGAS with parametrized ``u``, ``Q``, and discrete policy.

    This is the reference solver when both the state and action spaces are
    finite.  It optimizes the empirical saddle-point objective over a
    residual-weighting function ``u_beta``, an action-value function
    ``Q_theta``, and a policy ``pi_psi``.  The three modules are independent and
    need not share features or parameter dimensions.

    ``LinearUParam``, ``LinearQParam``, and ``SoftmaxLinearPolicyParam`` enable
    precomputed feature tables and closed-form tensor calculations.  Neural
    wrappers use autograd for the value, policy, and occupancy derivatives.
    In both cases, the occupancy geometry is the empirical outer product of
    ``grad_beta u_beta`` (full or diagonal), which reduces to the feature
    covariance matrix for a linear residual-weighting function.

    The implementation enumerates finite actions to compute exact policy
    expectations when requested.  Use ``ContinuousFinalParametrizedSolver``
    when observations are vectors that cannot be represented by finite state
    identifiers.
    """

    _THETA_MODES = {"reg_adaptive", "reg_fixed", "projection"}
    _THETA_OPTIMIZERS = {"sgd", "adam"}
    _THETA_START_MODES = {"zero", "warm"}
    _BETA_UPDATES = {"fogas_full", "fogas_diag"}
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
        u_param=None,
        q_param=None,
        policy_param=None,
        u_function=None,
        q_function=None,
        policy_features=None,
        delta=0.05,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        D_theta=None,
        beta_reg=None,
        theta_mode="reg_adaptive",
        theta_optimizer="sgd",
        theta_lr=None,
        theta_inner_steps=100,
        theta_start_mode="zero",
        theta_lambda=None,
        theta_include_beta_cov=False,
        beta_update="fogas_full",
        d_theta_scale=1.0,
        batch_size=None,
        print_params=False,
        dataset_verbose=False,
        seed=42,
        device=None,
    ):
        self.N = int(n_states)
        self.A = int(n_actions)
        self.gamma = float(gamma)
        self.x0 = int(x0)
        self.csv_path = csv_path
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

        if u_param is None and u_function is not None:
            u_param = LinearUParam(u_function, self.N, self.A)
        if q_param is None and q_function is not None:
            q_param = LinearQParam(q_function, self.N, self.A)
        if policy_param is None and policy_features is not None:
            policy_param = SoftmaxLinearPolicyParam(policy_features, self.N, self.A)
        if u_param is None:
            raise ValueError("u_param or u_function must be provided")
        if q_param is None:
            raise ValueError("q_param or q_function must be provided")
        if policy_param is None:
            raise ValueError("policy_param or policy_features must be provided")

        self.u_param = u_param.to(self.device)
        self.q_param = q_param.to(self.device)
        self.policy_param = policy_param.to(self.device)

        self.u_is_linear = self._is_linear_u_param(self.u_param)
        self.q_is_linear = self._is_linear_q_param(self.q_param)
        self.policy_is_linear = self._is_linear_policy_param(self.policy_param)

        self.theta_mode = self._canonical_theta_mode(theta_mode)
        self.theta_optimizer = self._canonical_theta_optimizer(theta_optimizer)
        self.theta_lr = self._canonical_optional_positive_float(theta_lr, "theta_lr")
        self.theta_inner_steps = self._canonical_positive_int(
            theta_inner_steps,
            "theta_inner_steps",
        )
        self.theta_start_mode = self._canonical_theta_start_mode(theta_start_mode)
        self.theta_lambda = self._canonical_optional_positive_float(
            theta_lambda,
            "theta_lambda",
        )
        self.theta_include_beta_cov = self._canonical_bool(
            theta_include_beta_cov,
            "theta_include_beta_cov",
        )
        self.beta_update = self._canonical_beta_update(beta_update)
        self.d_theta_scale = self._canonical_positive_float(d_theta_scale, "d_theta_scale")
        self.batch_size = self._canonical_optional_positive_int(batch_size, "batch_size")

        self.dataset = FOGASDataset(csv_path=csv_path, verbose=dataset_verbose)
        self.Xs = self.dataset.X.to(self.device).long()
        self.As = self.dataset.A.to(self.device).long()
        self.Rs = self.dataset.R.to(dtype=torch.float64, device=self.device)
        self.X_nexts = self.dataset.X_next.to(self.device).long()
        self.n = self.dataset.n
        self._validate_dataset_indices()

        self.state_indices = torch.arange(self.N, dtype=torch.long, device=self.device)
        self.action_indices = torch.arange(self.A, dtype=torch.long, device=self.device)
        state_grid = self.state_indices[:, None].expand(self.N, self.A)
        action_grid = self.action_indices[None, :].expand(self.N, self.A)
        self.flat_state_grid = state_grid.reshape(-1)
        self.flat_action_grid = action_grid.reshape(-1)

        self.d = self._num_trainable_params(self.u_param)
        self.d_q = self._num_trainable_params(self.q_param)
        self.d_pi = self._num_trainable_params(self.policy_param)
        if self.d <= 0:
            raise ValueError("u_param must expose at least one trainable parameter")
        if self.d_q <= 0:
            raise ValueError("q_param must expose at least one trainable parameter")
        if self.d_pi <= 0:
            raise ValueError("policy_param must expose at least one trainable parameter")

        self._initial_u_flat = self._module_flat_params(self.u_param).detach().clone()
        self._initial_q_flat = self._module_flat_params(self.q_param).detach().clone()
        self._initial_policy_flat = self._module_flat_params(self.policy_param).detach().clone()

        self._build_linear_tensors()
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

    @staticmethod
    def _is_linear_u_param(module):
        return bool(
            getattr(module, "is_linear_fast_path", False)
            and hasattr(module, "feature_table")
            and hasattr(module, "beta")
        )

    @staticmethod
    def _is_linear_q_param(module):
        return bool(
            getattr(module, "is_linear_fast_path", False)
            and hasattr(module, "feature_table")
            and hasattr(module, "theta")
        )

    @staticmethod
    def _is_linear_policy_param(module):
        return bool(
            getattr(module, "is_linear_fast_path", False)
            and hasattr(module, "feature_table")
            and hasattr(module, "psi")
        )

    @staticmethod
    def _trainable_params(module):
        return [p for p in module.parameters() if p.requires_grad]

    @classmethod
    def _num_trainable_params(cls, module):
        return int(sum(p.numel() for p in cls._trainable_params(module)))

    @classmethod
    def _module_flat_params(cls, module):
        params = cls._trainable_params(module)
        if not params:
            return torch.empty(0)
        return torch.cat([p.detach().reshape(-1) for p in params])

    @classmethod
    def _set_module_flat_params(cls, module, flat):
        params = cls._trainable_params(module)
        expected = sum(p.numel() for p in params)
        flat = flat.detach().to(device=params[0].device if params else None).reshape(-1)
        if flat.numel() != expected:
            raise ValueError(f"flat parameter size mismatch: expected {expected}, got {flat.numel()}")
        offset = 0
        with torch.no_grad():
            for param in params:
                n = param.numel()
                param.copy_(flat[offset : offset + n].reshape_as(param).to(dtype=param.dtype))
                offset += n

    @staticmethod
    def _param_dtype(module, fallback=torch.float64):
        for param in module.parameters():
            return param.dtype
        for buffer in module.buffers():
            return buffer.dtype
        return fallback

    @classmethod
    def _flat_grads(
        cls,
        output,
        params,
        retain_graph=False,
        create_graph=False,
    ):
        grads = torch.autograd.grad(
            output,
            params,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=True,
        )
        flat = []
        for param, grad in zip(params, grads):
            if grad is None:
                flat.append(torch.zeros_like(param).reshape(-1))
            else:
                flat.append(grad.reshape(-1))
        if not flat:
            return torch.empty(0, dtype=output.dtype, device=output.device)
        return torch.cat(flat)

    @classmethod
    def _assign_flat_grad(cls, params, flat_grad, sign=1.0):
        offset = 0
        for param in params:
            n = param.numel()
            grad = flat_grad[offset : offset + n].reshape_as(param).to(dtype=param.dtype)
            param.grad = sign * grad
            offset += n

    @classmethod
    def _apply_flat_direction(cls, params, direction, step_size):
        offset = 0
        with torch.no_grad():
            for param in params:
                n = param.numel()
                delta = direction[offset : offset + n].reshape_as(param).to(dtype=param.dtype)
                param.add_(step_size * delta)
                offset += n

    def _validate_dataset_indices(self):
        if torch.any((self.Xs < 0) | (self.Xs >= self.N)):
            raise ValueError("dataset states contain values outside [0, n_states)")
        if torch.any((self.X_nexts < 0) | (self.X_nexts >= self.N)):
            raise ValueError("dataset next states contain values outside [0, n_states)")
        if torch.any((self.As < 0) | (self.As >= self.A)):
            raise ValueError("dataset actions contain values outside [0, n_actions)")

    def _build_linear_tensors(self):
        if self.u_is_linear:
            self.U_XA = self.u_param.feature_table
            self.U_sample = self.U_XA[self.Xs, self.As]
            self.d = int(self.U_XA.shape[2])
        else:
            self.U_XA = None
            self.U_sample = None

        if self.q_is_linear:
            self.Q_XA = self.q_param.feature_table
            self.Q_sample = self.Q_XA[self.Xs, self.As]
            self.d_q = int(self.Q_XA.shape[2])
        else:
            self.Q_XA = None
            self.Q_sample = None

        if self.policy_is_linear:
            self.OMEGA_PI_XA = self.policy_param.feature_table
            self.d_pi = int(self.OMEGA_PI_XA.shape[2])
        else:
            self.OMEGA_PI_XA = None

    def _build_preconditioner(self, beta_reg):
        if beta_reg is None:
            beta_reg = 1.0
        beta_reg = float(beta_reg)
        if beta_reg < 0.0:
            raise ValueError("beta_reg must be non-negative")
        self._preconditioner_beta_reg = beta_reg

        if not self.u_is_linear:
            self.Empirical_cov = None
            self.H = None
            self.H_inv = None
            self.Cov_emp = None
            self.Cov_emp_inv = None
            return

        self.Empirical_cov = (self.U_sample.T @ self.U_sample) / self.n
        self.H = self.Empirical_cov + beta_reg * torch.eye(
            self.d,
            dtype=self.U_sample.dtype,
            device=self.device,
        )
        self.H_inv = torch.linalg.inv(self.H)
        self.Cov_emp = self.H
        self.Cov_emp_inv = self.H_inv

    def _estimate_omega(self):
        if not self.u_is_linear:
            self.omega = None
            return
        rhs = (self.U_sample.T @ self.Rs.to(dtype=self.U_sample.dtype)) / self.n
        self.omega = self.H_inv @ rhs

    def _compute_reward_bound(self):
        if self.u_is_linear:
            self.r_hat = torch.tensordot(self.U_XA, self.omega, dims=([2], [0]))
            R = torch.max(torch.abs(self.r_hat))
        else:
            self.r_hat = None
            R = torch.max(torch.abs(self.Rs))
        self.R = float(max(R.detach().cpu().item(), self._EPS))

    @classmethod
    def _canonical_theta_mode(cls, theta_mode):
        name = str(theta_mode).lower()
        if name not in cls._THETA_MODES:
            raise ValueError(
                "theta_mode must be one of 'reg_adaptive', 'reg_fixed', or 'projection'"
            )
        return name

    @classmethod
    def _canonical_theta_optimizer(cls, theta_optimizer):
        name = str(theta_optimizer).lower()
        if name not in cls._THETA_OPTIMIZERS:
            raise ValueError("theta_optimizer must be either 'sgd' or 'adam'")
        return name

    @classmethod
    def _canonical_theta_start_mode(cls, theta_start_mode):
        name = str(theta_start_mode).lower()
        if name not in cls._THETA_START_MODES:
            raise ValueError("theta_start_mode must be either 'zero' or 'warm'")
        return name

    @classmethod
    def _canonical_beta_update(cls, beta_update):
        name = str(beta_update).lower()
        if name not in cls._BETA_UPDATES:
            raise ValueError("beta_update must be either 'fogas_full' or 'fogas_diag'")
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

    @classmethod
    def _canonical_optional_positive_int(cls, value, name):
        if value is None:
            return None
        return cls._canonical_positive_int(value, name)

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

    def _policy_matrix(self):
        return self.policy_param.probs(self.state_indices)

    def _q_matrix(self):
        q_flat = self.q_param.q(self.flat_state_grid, self.flat_action_grid)
        return q_flat.reshape(self.N, self.A)

    def _sample_batch(self):
        full = self.batch_size is None or self.batch_size >= self.n
        if full:
            indices = torch.arange(self.n, dtype=torch.long, device=self.device)
        else:
            indices = torch.randint(self.n, (int(self.batch_size),), device=self.device)
        batch = {
            "indices": indices,
            "n": int(indices.numel()),
            "full": bool(full),
            "Xs": self.Xs[indices],
            "As": self.As[indices],
            "Rs": self.Rs[indices],
            "X_nexts": self.X_nexts[indices],
        }
        if self.u_is_linear:
            batch["U_sample"] = self.U_sample[indices]
        if self.q_is_linear:
            batch["Q_sample"] = self.Q_sample[indices]
        return batch

    def _q_sample(self, batch):
        return self.q_param.q(batch["Xs"], batch["As"])

    def _u_sample_values(self, batch):
        return self.u_param.u(batch["Xs"], batch["As"])

    def _prepare_module_init(self, module, init, initial_flat, name):
        if init is None:
            flat = initial_flat.clone()
        else:
            flat = init.clone().detach().reshape(-1)
            if flat.numel() != initial_flat.numel():
                raise ValueError(
                    f"{name} must have shape ({initial_flat.numel()},), got {tuple(flat.shape)}"
                )
            flat = flat.to(dtype=initial_flat.dtype, device=self.device)
        self._set_module_flat_params(module, flat)
        return flat.clone()

    def _prepare_theta_bar_init(self, theta_bar_init):
        if theta_bar_init is None:
            return torch.zeros(self.d_q, dtype=torch.float64, device=self.device)
        theta_bar_t = theta_bar_init.clone().to(dtype=torch.float64, device=self.device).reshape(-1)
        if theta_bar_t.shape != (self.d_q,):
            raise ValueError(
                f"theta_bar_init must have shape ({self.d_q},), got {tuple(theta_bar_t.shape)}"
            )
        return theta_bar_t

    def _default_theta_lr(self, lambda_theta, optimizer):
        if self.theta_lr is not None:
            return self.theta_lr
        if optimizer == "adam":
            return 1e-2
        if lambda_theta is None:
            return 1.0
        return 1.0 / max(float(lambda_theta), self._EPS)

    def _theta_objective(self, theta_t, c_t, lambda_theta):
        objective = torch.dot(theta_t, c_t)
        if lambda_theta is not None:
            objective = objective + 0.5 * lambda_theta * torch.dot(theta_t, theta_t)
        return objective

    def _theta_grad(self, theta_t, c_t, lambda_theta):
        if lambda_theta is None:
            return c_t
        return c_t + lambda_theta * theta_t

    def _project_tensor(self, tensor, radius):
        norm = torch.linalg.norm(tensor)
        if norm <= radius:
            return tensor
        return tensor * (radius / norm.clamp_min(self._EPS))

    def _project_theta(self, theta_t, effective_D_theta):
        return self._project_tensor(theta_t, effective_D_theta)

    def _project_module_params(self, module, radius):
        flat = self._module_flat_params(module)
        projected = self._project_tensor(flat, radius)
        self._set_module_flat_params(module, projected)

    def _sgd_theta_update(self, c_t, lambda_theta, theta_lr, theta_init, effective_D_theta):
        theta_t = theta_init.detach().clone()
        for _ in range(self.theta_inner_steps):
            grad = self._theta_grad(theta_t, c_t, lambda_theta)
            theta_t = theta_t - theta_lr * grad
            if self.theta_mode == "projection":
                theta_t = self._project_theta(theta_t, effective_D_theta)
        final_grad = self._theta_grad(theta_t, c_t, lambda_theta)
        return theta_t, float(torch.linalg.norm(final_grad).detach().cpu().item())

    def _adam_theta_update(self, c_t, lambda_theta, theta_lr, theta_init, effective_D_theta):
        theta_param = torch.nn.Parameter(theta_init.detach().clone())
        optimizer = torch.optim.Adam([theta_param], lr=theta_lr)
        for _ in range(self.theta_inner_steps):
            optimizer.zero_grad(set_to_none=True)
            objective = self._theta_objective(theta_param, c_t, lambda_theta)
            objective.backward()
            optimizer.step()
            if self.theta_mode == "projection":
                with torch.no_grad():
                    theta_param.copy_(self._project_theta(theta_param, effective_D_theta))
        with torch.no_grad():
            theta_t = theta_param.detach().clone()
            final_grad = self._theta_grad(theta_t, c_t, lambda_theta)
        return theta_t, float(torch.linalg.norm(final_grad).detach().cpu().item())

    def _compute_linear_theta_update(self, theta_mismatch, beta_t, effective_D_theta, theta_init):
        c_t = theta_mismatch
        if self.theta_include_beta_cov:
            if not (self.u_is_linear and self.q_is_linear and self.d_q == self.d):
                raise ValueError(
                    "theta_include_beta_cov=True requires linear u and Q parametrizations "
                    "with the same feature dimension"
                )
            c_t = c_t - self._preconditioner_beta_reg * beta_t

        norm_c = torch.linalg.norm(c_t)
        lambda_theta = None
        if self.theta_mode == "reg_adaptive":
            lambda_theta = max(float(norm_c.detach().cpu().item()) / effective_D_theta, self._EPS)
        elif self.theta_mode == "reg_fixed":
            if self.theta_lambda is None:
                raise ValueError("theta_lambda must be provided when theta_mode='reg_fixed'")
            lambda_theta = self.theta_lambda

        if self.theta_optimizer == "sgd":
            theta_lr_used = self._default_theta_lr(lambda_theta, optimizer="sgd")
            theta_t, theta_grad_norm = self._sgd_theta_update(
                c_t=c_t,
                lambda_theta=lambda_theta,
                theta_lr=theta_lr_used,
                theta_init=theta_init,
                effective_D_theta=effective_D_theta,
            )
        else:
            theta_lr_used = self._default_theta_lr(lambda_theta, optimizer="adam")
            theta_t, theta_grad_norm = self._adam_theta_update(
                c_t=c_t,
                lambda_theta=lambda_theta,
                theta_lr=theta_lr_used,
                theta_init=theta_init,
                effective_D_theta=effective_D_theta,
            )

        q_objective = self._theta_objective(theta_t, c_t, lambda_theta)
        self._set_module_flat_params(self.q_param, theta_t)
        return theta_t, lambda_theta, q_objective, theta_grad_norm, theta_lr_used

    def _theta_loss_nonlinear(self, coeff, pi_mat, batch):
        q_all = self._q_matrix()
        dtype = q_all.dtype
        coeff = coeff.detach().to(dtype=dtype)
        pi = pi_mat.detach().to(dtype=dtype)
        e_q_pi = (pi * q_all).sum(dim=1)
        q_sample = self._q_sample(batch).to(dtype=dtype)
        return (
            (1.0 - self.gamma) * e_q_pi[self.x0]
            + (self.gamma / batch["n"]) * (coeff * e_q_pi[batch["X_nexts"]]).sum()
            - (coeff * q_sample).mean()
        )

    def _nonlinear_theta_lambda(self, coeff, pi_mat, batch, effective_D_theta):
        lambda_theta = None
        if self.theta_mode == "reg_adaptive":
            params = self._trainable_params(self.q_param)
            loss = self._theta_loss_nonlinear(coeff, pi_mat, batch)
            grad = self._flat_grads(loss, params)
            lambda_theta = max(float(torch.linalg.norm(grad).detach().cpu().item()) / effective_D_theta, self._EPS)
        elif self.theta_mode == "reg_fixed":
            if self.theta_lambda is None:
                raise ValueError("theta_lambda must be provided when theta_mode='reg_fixed'")
            lambda_theta = self.theta_lambda
        return lambda_theta

    def _compute_nonlinear_theta_update(self, coeff, pi_mat, batch, effective_D_theta):
        if self.theta_include_beta_cov:
            raise ValueError("theta_include_beta_cov=True is only supported by linear parametrizations")

        if self.theta_start_mode == "zero":
            self._set_module_flat_params(self.q_param, self._initial_q_flat)

        params = self._trainable_params(self.q_param)
        lambda_theta = self._nonlinear_theta_lambda(coeff, pi_mat, batch, effective_D_theta)
        theta_lr_used = self._default_theta_lr(lambda_theta, self.theta_optimizer)
        optimizer_cls = torch.optim.SGD if self.theta_optimizer == "sgd" else torch.optim.Adam
        optimizer = optimizer_cls(params, lr=theta_lr_used)

        for _ in range(self.theta_inner_steps):
            optimizer.zero_grad(set_to_none=True)
            objective = self._theta_loss_nonlinear(coeff, pi_mat, batch)
            if lambda_theta is not None:
                flat = torch.cat([p.reshape(-1) for p in params])
                objective = objective + 0.5 * lambda_theta * torch.dot(flat, flat)
            objective.backward()
            optimizer.step()
            if self.theta_mode == "projection":
                self._project_module_params(self.q_param, effective_D_theta)

        objective = self._theta_loss_nonlinear(coeff, pi_mat, batch)
        if lambda_theta is not None:
            flat_live = torch.cat([p.reshape(-1) for p in params])
            objective_for_grad = objective + 0.5 * lambda_theta * torch.dot(flat_live, flat_live)
        else:
            objective_for_grad = objective
        grad = self._flat_grads(objective_for_grad, params)
        theta_t = self._module_flat_params(self.q_param).detach().clone().to(dtype=torch.float64)
        return (
            theta_t,
            lambda_theta,
            objective.detach(),
            float(torch.linalg.norm(grad).detach().cpu().item()),
            theta_lr_used,
        )

    def _exact_policy_gradient_linear(self, pi_mat, q_all, policy_state_weights):
        G = policy_state_weights[:, None] * q_all
        policy_objective = (pi_mat * G).sum()
        V_G = (pi_mat * G).sum(dim=1)
        advantage_G = G - V_G[:, None]
        policy_grad = (
            pi_mat[..., None] * advantage_G[..., None] * self.OMEGA_PI_XA
        ).sum(dim=(0, 1))
        return policy_grad, policy_objective

    def _reinforce_policy_gradient_linear(
        self,
        pi_mat,
        q_all,
        coeff,
        batch,
        policy_state_weights,
        state_weight_update,
        reinforce_samples,
    ):
        if state_weight_update == "normal":
            states = torch.cat(
                (
                    torch.tensor([self.x0], dtype=torch.long, device=self.device),
                    batch["X_nexts"],
                )
            )
            weights = torch.cat(
                (
                    torch.tensor([1.0 - self.gamma], dtype=pi_mat.dtype, device=self.device),
                    (self.gamma / batch["n"]) * coeff.to(dtype=pi_mat.dtype),
                )
            )
        else:
            states = self.state_indices
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

    def _exact_policy_gradient_nonlinear(self, q_all, policy_state_weights):
        params = self._trainable_params(self.policy_param)
        probs = self.policy_param.probs(self.state_indices)
        objective = (
            probs
            * (policy_state_weights[:, None].to(dtype=probs.dtype) * q_all.detach().to(dtype=probs.dtype))
        ).sum()
        grad = self._flat_grads(objective, params)
        return grad.detach(), objective.detach()

    def _reinforce_policy_gradient_nonlinear(
        self,
        q_all,
        coeff,
        batch,
        policy_state_weights,
        state_weight_update,
        reinforce_samples,
    ):
        if state_weight_update == "normal":
            states = torch.cat(
                (
                    torch.tensor([self.x0], dtype=torch.long, device=self.device),
                    batch["X_nexts"],
                )
            )
            weights = torch.cat(
                (
                    torch.tensor([1.0 - self.gamma], dtype=torch.float64, device=self.device),
                    (self.gamma / batch["n"]) * coeff.to(dtype=torch.float64),
                )
            )
        else:
            states = self.state_indices
            weights = policy_state_weights.to(dtype=torch.float64)

        sample_count = self._canonical_positive_int(reinforce_samples, "reinforce_samples")
        with torch.no_grad():
            probs_detached = self.policy_param.probs(states).detach().clamp_min(self._EPS)
            sampled_actions = torch.multinomial(
                probs_detached.reshape(-1, self.A),
                num_samples=sample_count,
                replacement=True,
            )
            q_states = q_all.detach()[states].to(dtype=torch.float64)
            baseline = (probs_detached.to(dtype=torch.float64) * q_states).sum(dim=1)
            sampled_q = q_states.gather(1, sampled_actions)
            advantages = sampled_q - baseline[:, None]

        repeated_states = states[:, None].expand(-1, sample_count).reshape(-1)
        log_probs = self.policy_param.log_prob_actions(
            repeated_states,
            sampled_actions.reshape(-1),
        ).reshape(states.numel(), sample_count)
        surrogate = (
            weights[:, None].to(dtype=log_probs.dtype)
            * advantages.to(dtype=log_probs.dtype)
            * log_probs
        ).sum() / float(sample_count)
        params = self._trainable_params(self.policy_param)
        grad = self._flat_grads(surrogate, params)
        policy_objective = (
            self.policy_param.probs(self.state_indices)
            * (policy_state_weights[:, None].to(dtype=q_all.dtype) * q_all.detach())
        ).sum()
        return grad.detach(), policy_objective.detach()

    def _policy_kl_hessian_vector_product_linear(self, vector, psi_t, state_indices):
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

    def _policy_kl_hessian_vector_product_nonlinear(self, vector, state_indices):
        params = self._trainable_params(self.policy_param)
        with torch.no_grad():
            old_probs = self.policy_param.probs(state_indices).detach().clamp_min(self._EPS)
        new_probs = self.policy_param.probs(state_indices).clamp_min(self._EPS)
        empirical_kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(
            dim=1
        ).mean()
        grad_kl = torch.autograd.grad(empirical_kl, params, create_graph=True)
        grad_flat = torch.cat([g.reshape(-1) for g in grad_kl])
        grad_vector_product = torch.dot(grad_flat, vector.detach())
        hvp = torch.autograd.grad(grad_vector_product, params, retain_graph=False)
        return torch.cat([h.reshape(-1) for h in hvp]).detach()

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
            if self.policy_is_linear:
                hvp = self._policy_kl_hessian_vector_product_linear(v, psi_t, state_indices)
            else:
                hvp = self._policy_kl_hessian_vector_product_nonlinear(v, state_indices)
            return hvp + fisher_damping * v

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

    def _policy_update_state_indices(self, state_weight_update, batch):
        if state_weight_update == "normal":
            return torch.cat(
                (
                    torch.tensor([self.x0], dtype=torch.long, device=self.device),
                    batch["X_nexts"],
                )
            )
        return self.state_indices

    def _compute_linear_beta_update_direction(self, beta_grad, batch):
        if self.beta_update == "fogas_full":
            if batch["full"]:
                H = self.H
                direction = self.H_inv @ beta_grad
                diag_min = None
                diag_max = None
            else:
                U_sample = batch["U_sample"]
                H = (U_sample.T @ U_sample) / batch["n"]
                H = H + self.beta_reg * torch.eye(
                    self.d,
                    dtype=U_sample.dtype,
                    device=self.device,
                )
                direction = torch.linalg.solve(H, beta_grad.to(dtype=H.dtype))
                diag_min = float(torch.diagonal(H).min().detach().cpu().item())
                diag_max = float(torch.diagonal(H).max().detach().cpu().item())
            diagnostics = {
                "beta_update": self.beta_update,
                "beta_diag_min": diag_min,
                "beta_diag_max": diag_max,
            }
            return direction, diagnostics

        if batch["full"]:
            diag_h = torch.diagonal(self.H).clamp_min(self._EPS)
        else:
            diag_h = (batch["U_sample"] * batch["U_sample"]).mean(dim=0) + self.beta_reg
            diag_h = diag_h.clamp_min(self._EPS)
        direction = beta_grad / diag_h
        diagnostics = {
            "beta_update": self.beta_update,
            "beta_diag_min": float(diag_h.min().detach().cpu().item()),
            "beta_diag_max": float(diag_h.max().detach().cpu().item()),
        }
        return direction, diagnostics

    def _jacobian_outputs(self, outputs, params):
        rows = []
        flat_size = sum(p.numel() for p in params)
        if flat_size == 0:
            return torch.empty(outputs.numel(), 0, dtype=outputs.dtype, device=outputs.device)
        flat_outputs = outputs.reshape(-1)
        for idx, value in enumerate(flat_outputs):
            grads = torch.autograd.grad(
                value,
                params,
                retain_graph=idx < flat_outputs.numel() - 1,
                allow_unused=True,
            )
            row = []
            for param, grad in zip(params, grads):
                if grad is None:
                    row.append(torch.zeros_like(param).reshape(-1))
                else:
                    row.append(grad.reshape(-1))
            rows.append(torch.cat(row))
        return torch.stack(rows, dim=0)

    def _u_sample_jacobian(self, batch):
        params = self._trainable_params(self.u_param)
        if not params:
            return torch.empty(batch["n"], 0, dtype=torch.float64, device=self.device)

        try:
            from torch.func import functional_call, grad, vmap
        except ImportError:
            outputs = self._u_sample_values(batch)
            return self._jacobian_outputs(outputs, params)

        named_params = {
            name: param
            for name, param in self.u_param.named_parameters()
            if param.requires_grad
        }
        buffers = dict(self.u_param.named_buffers())
        param_names = list(named_params)

        def single_u(param_dict, buffer_dict, state, action):
            value = functional_call(
                self.u_param,
                (param_dict, buffer_dict),
                (state, action),
            )
            return value.reshape(())

        try:
            grads = vmap(
                grad(single_u),
                in_dims=(None, None, 0, 0),
            )(named_params, buffers, batch["Xs"], batch["As"])
            return torch.cat(
                [grads[name].reshape(batch["n"], -1) for name in param_names],
                dim=1,
            )
        except Exception:
            outputs = self._u_sample_values(batch)
            return self._jacobian_outputs(outputs, params)

    def _compute_nonlinear_beta_update_direction(self, td_error, batch):
        jacobian = self._u_sample_jacobian(batch).detach()
        td = td_error.detach().to(dtype=jacobian.dtype)
        beta_grad = (jacobian.T @ td) / batch["n"]

        if self.beta_update == "fogas_full":
            H = (jacobian.T @ jacobian) / batch["n"]
            H = H + self.beta_reg * torch.eye(
                H.shape[0],
                dtype=H.dtype,
                device=H.device,
            )
            direction = torch.linalg.solve(H, beta_grad.to(dtype=H.dtype))
            diagnostics = {
                "beta_update": self.beta_update,
                "beta_diag_min": float(torch.diagonal(H).min().detach().cpu().item()),
                "beta_diag_max": float(torch.diagonal(H).max().detach().cpu().item()),
            }
        else:
            diag_h = (jacobian * jacobian).mean(dim=0) + self.beta_reg
            diag_h = diag_h.clamp_min(self._EPS)
            direction = beta_grad.to(dtype=diag_h.dtype) / diag_h
            diagnostics = {
                "beta_update": self.beta_update,
                "beta_diag_min": float(diag_h.min().detach().cpu().item()),
                "beta_diag_max": float(diag_h.max().detach().cpu().item()),
            }
        return direction.detach(), beta_grad.detach(), diagnostics

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
        theta_mode=None,
        theta_optimizer=None,
        theta_lr=None,
        theta_inner_steps=None,
        theta_start_mode=None,
        theta_lambda=None,
        theta_include_beta_cov=None,
        beta_update=None,
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
        """Optimize Generalized FOGAS and return the learned policy matrix.

        Constructor update settings may be overridden for this run without
        permanently mutating the solver.  ``policy_gradient='exact'`` sums over
        every finite action; ``'reinforce'`` uses ``reinforce_samples`` actions
        per state.  ``policy_optimizer='adam'``, a warm-started regularized Adam
        value response, and ``beta_update='fogas_diag'`` form the practical
        thesis configuration when set explicitly.

        Returns:
            A ``(n_states, n_actions)`` stochastic policy tensor.  Final
            parameters are also stored in ``theta``, ``psi``, and ``beta_T``;
            iteration statistics are returned by :meth:`get_diagnostics`.
        """
        T = self.params.T if T is None else int(T)
        alpha = self.params.alpha if alpha is None else float(alpha)
        eta = self.params.eta if eta is None else float(eta)
        rho = self.params.rho if rho is None else float(rho)
        D_theta = self.params.D_theta if D_theta is None else float(D_theta)

        previous_theta_mode = self.theta_mode
        previous_theta_optimizer = self.theta_optimizer
        previous_theta_lr = self.theta_lr
        previous_theta_inner_steps = self.theta_inner_steps
        previous_theta_start_mode = self.theta_start_mode
        previous_theta_lambda = self.theta_lambda
        previous_theta_include_beta_cov = self.theta_include_beta_cov
        previous_beta_update = self.beta_update
        previous_d_theta_scale = self.d_theta_scale

        if theta_mode is not None:
            self.theta_mode = self._canonical_theta_mode(theta_mode)
        if theta_optimizer is not None:
            self.theta_optimizer = self._canonical_theta_optimizer(theta_optimizer)
        if theta_lr is not None:
            self.theta_lr = self._canonical_positive_float(theta_lr, "theta_lr")
        if theta_inner_steps is not None:
            self.theta_inner_steps = self._canonical_positive_int(
                theta_inner_steps,
                "theta_inner_steps",
            )
        if theta_start_mode is not None:
            self.theta_start_mode = self._canonical_theta_start_mode(theta_start_mode)
        if theta_lambda is not None:
            self.theta_lambda = self._canonical_positive_float(theta_lambda, "theta_lambda")
        if theta_include_beta_cov is not None:
            self.theta_include_beta_cov = self._canonical_bool(
                theta_include_beta_cov,
                "theta_include_beta_cov",
            )
        if beta_update is not None:
            self.beta_update = self._canonical_beta_update(beta_update)
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
            self.theta_mode = previous_theta_mode
            self.theta_optimizer = previous_theta_optimizer
            self.theta_lr = previous_theta_lr
            self.theta_inner_steps = previous_theta_inner_steps
            self.theta_start_mode = previous_theta_start_mode
            self.theta_lambda = previous_theta_lambda
            self.theta_include_beta_cov = previous_theta_include_beta_cov
            self.beta_update = previous_beta_update
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

        beta_t = self._prepare_module_init(self.u_param, beta_init, self._initial_u_flat, "beta_init")
        theta_bar_t = self._prepare_theta_bar_init(theta_bar_init)
        psi_t = self._prepare_module_init(self.policy_param, psi_init, self._initial_policy_flat, "psi_init")
        self._set_module_flat_params(self.q_param, self._initial_q_flat)

        policy_adam_optimizer = None
        if policy_optimizer == "adam":
            policy_adam_optimizer = torch.optim.Adam(
                self._trainable_params(self.policy_param),
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
        iterator = trange(T, desc="FinalParametrizedSolver", disable=not use_tqdm)

        final_theta = self._module_flat_params(self.q_param).detach().clone().to(dtype=torch.float64)

        for t in iterator:
            # A batch contains only offline transitions.  Finite state/action
            # grids remain available for exact policy and value expectations;
            # batch_size controls only the empirical dataset terms.
            batch = self._sample_batch()
            if self.policy_is_linear:
                psi_t = self._module_flat_params(self.policy_param).detach().clone()
                pi_mat = self._linear_policy_matrix(psi_t)
            else:
                pi_mat = self._policy_matrix()

            coeff = self._u_sample_values(batch).detach().to(dtype=torch.float64)

            # 1. Approximate the regularized value-parameter best response.
            # Linear Q modules use the feature-space mismatch directly;
            # nonlinear modules minimize the same empirical objective with
            # autograd and the configured inner optimizer.
            if self.q_is_linear:
                E_q_pi = (pi_mat.to(dtype=self.Q_XA.dtype)[..., None] * self.Q_XA).sum(dim=1)
                beta_for_theta = self._module_flat_params(self.u_param).to(dtype=self.Q_XA.dtype)
                coeff_q = coeff.to(dtype=self.Q_XA.dtype)
                theta_mismatch = (1.0 - self.gamma) * E_q_pi[self.x0]
                theta_mismatch = theta_mismatch + (self.gamma / batch["n"]) * (
                    coeff_q[:, None] * E_q_pi[batch["X_nexts"]]
                ).sum(dim=0)
                theta_mismatch = theta_mismatch - (
                    batch["Q_sample"] * coeff_q[:, None]
                ).mean(dim=0)

                theta_init = (
                    self._initial_q_flat.to(dtype=theta_mismatch.dtype, device=self.device)
                    if self.theta_start_mode == "zero"
                    else final_theta.to(dtype=theta_mismatch.dtype, device=self.device)
                )
                theta_t, _lambda_theta, q_objective, theta_grad_norm, _theta_lr_used = (
                    self._compute_linear_theta_update(
                        theta_mismatch=theta_mismatch,
                        beta_t=beta_for_theta,
                        effective_D_theta=effective_D_theta,
                        theta_init=theta_init,
                    )
                )
                final_theta = theta_t.detach().clone().to(dtype=torch.float64)
            else:
                theta_t, _lambda_theta, q_objective, theta_grad_norm, _theta_lr_used = (
                    self._compute_nonlinear_theta_update(
                        coeff=coeff,
                        pi_mat=pi_mat,
                        batch=batch,
                        effective_D_theta=effective_D_theta,
                    )
                )
                final_theta = theta_t.detach().clone().to(dtype=torch.float64)

            q_all = self._q_matrix().to(dtype=torch.float64)
            pi_eval = self._policy_matrix().detach().to(dtype=torch.float64)
            q_next = q_all[batch["X_nexts"]]
            v = (pi_eval[batch["X_nexts"]] * q_next).sum(dim=1)
            q_current = self._q_sample(batch).to(dtype=torch.float64)
            v_x0 = (pi_eval[self.x0] * q_all[self.x0]).sum()
            td_error = batch["Rs"] + self.gamma * v - q_current
            beta_objective = (coeff * td_error).mean()
            total_loss = (1.0 - self.gamma) * v_x0 + beta_objective

            # 2. Build the occupancy ascent direction.  For linear u this is
            # the covariance-preconditioned feature gradient.  For nonlinear
            # u, sample Jacobians replace features in the local matrix G_t.
            if self.u_is_linear:
                beta_grad = (batch["U_sample"].T @ td_error.to(dtype=batch["U_sample"].dtype)) / batch["n"]
                beta_update_direction, beta_diagnostics = self._compute_linear_beta_update_direction(
                    beta_grad,
                    batch,
                )
                beta_t = (1.0 / (1.0 + rho * eta)) * (
                    self._module_flat_params(self.u_param).to(dtype=beta_update_direction.dtype)
                    + eta * beta_update_direction
                )
                self._set_module_flat_params(self.u_param, beta_t)
            else:
                beta_update_direction, beta_grad, beta_diagnostics = (
                    self._compute_nonlinear_beta_update_direction(td_error, batch)
                )
                beta_t = (1.0 / (1.0 + rho * eta)) * (
                    self._module_flat_params(self.u_param).to(dtype=beta_update_direction.dtype)
                    + eta * beta_update_direction
                )
                self._set_module_flat_params(self.u_param, beta_t)

            state_weight_sums = torch.zeros(self.N, dtype=torch.float64, device=self.device)
            state_weight_sums.index_add_(0, batch["X_nexts"], coeff)
            state_weights = (self.gamma / batch["n"]) * state_weight_sums
            state_weights[self.x0] = state_weights[self.x0] + (1.0 - self.gamma)
            if state_weight_update == "normal":
                policy_state_weights = state_weights
            else:
                policy_state_weights = torch.clamp(state_weights, min=c_min)

            # 3. Differentiate the policy objective while holding Q fixed.
            # Exact enumeration is practical for finite actions; REINFORCE
            # implements the sampled advantage estimator used in its ablation.
            if self.policy_is_linear:
                pi_for_policy = self._linear_policy_matrix(psi_t).to(dtype=torch.float64)
                if policy_gradient == "exact":
                    policy_grad, policy_objective = self._exact_policy_gradient_linear(
                        pi_for_policy,
                        q_all,
                        policy_state_weights,
                    )
                else:
                    policy_grad, policy_objective = self._reinforce_policy_gradient_linear(
                        pi_for_policy,
                        q_all,
                        coeff,
                        batch,
                        policy_state_weights,
                        state_weight_update,
                        reinforce_samples,
                    )
            elif policy_gradient == "exact":
                policy_grad, policy_objective = self._exact_policy_gradient_nonlinear(
                    q_all,
                    policy_state_weights,
                )
            else:
                policy_grad, policy_objective = self._reinforce_policy_gradient_nonlinear(
                    q_all,
                    coeff,
                    batch,
                    policy_state_weights,
                    state_weight_update,
                    reinforce_samples,
                )

            if policy_optimizer == "sgd":
                policy_direction = policy_grad
                policy_direction_kind = "sgd_gradient"
                policy_diagnostics = {}
                self._apply_flat_direction(
                    self._trainable_params(self.policy_param),
                    policy_direction,
                    alpha,
                )
            elif policy_optimizer == "adam":
                policy_adam_optimizer.zero_grad(set_to_none=True)
                self._assign_flat_grad(
                    self._trainable_params(self.policy_param),
                    policy_grad,
                    sign=-1.0,
                )
                policy_adam_optimizer.step()
                policy_direction = policy_grad
                policy_direction_kind = "adam_gradient"
                policy_diagnostics = {}
            else:
                policy_state_indices = self._policy_update_state_indices(state_weight_update, batch)
                policy_direction, policy_diagnostics = self._conjugate_gradient_policy_direction(
                    policy_grad=policy_grad,
                    psi_t=psi_t,
                    state_indices=policy_state_indices,
                    fisher_damping=fisher_damping,
                    cg_iters=cg_iters,
                    cg_tol=cg_tol,
                )
                policy_direction_kind = "cg_fisher"
                self._apply_flat_direction(
                    self._trainable_params(self.policy_param),
                    policy_direction,
                    alpha,
                )

            # Store cumulative value parameters and the current policy iterate
            # for convergence plots and post-run diagnostics.
            theta_bar_t = theta_bar_t + final_theta.to(dtype=theta_bar_t.dtype)
            theta_bar_history.append(theta_bar_t.clone())
            psi_t = self._module_flat_params(self.policy_param).detach().clone().to(dtype=torch.float64)
            psi_history.append(psi_t.clone())

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
                "beta_direction_norm": float(
                    torch.linalg.norm(beta_update_direction).detach().cpu().item()
                ),
                "theta_grad_norm": float(theta_grad_norm),
                "theta_norm": float(torch.linalg.norm(final_theta).detach().cpu().item()),
                "theta_mode": self.theta_mode,
                "theta_optimizer": self.theta_optimizer,
                "theta_start_mode": self.theta_start_mode,
                "theta_lambda": None if _lambda_theta is None else float(_lambda_theta),
                "theta_lr": float(_theta_lr_used),
                "theta_include_beta_cov": self.theta_include_beta_cov,
                "policy_optimizer": policy_optimizer,
                "policy_gradient": policy_gradient,
                "policy_direction": policy_direction_kind,
                "reinforce_samples": int(reinforce_samples),
                "D_theta": float(D_theta),
                "effective_D_theta": float(effective_D_theta),
                "u_fast_path": bool(self.u_is_linear),
                "q_fast_path": bool(self.q_is_linear),
                "policy_fast_path": bool(self.policy_is_linear),
            }
            diagnostics.update(beta_diagnostics)
            diagnostics.update(policy_diagnostics)
            diagnostics_history.append(diagnostics)

            if use_tqdm:
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
                print(f"[FinalParametrizedSolver] Iter {t + 1}/{T} {values}")

        self.theta = final_theta.clone()
        self.theta_bar_history = theta_bar_history
        self.psi_history = psi_history
        self.psi = self._module_flat_params(self.policy_param).detach().clone().to(dtype=torch.float64)
        self.pi = self._policy_matrix().detach().clone().to(dtype=torch.float64)
        self.lambda_T = self._module_flat_params(self.u_param).detach().clone().to(dtype=torch.float64)
        self.beta_T = self.lambda_T.clone()
        self.diagnostics_history = diagnostics_history

        return self.pi
