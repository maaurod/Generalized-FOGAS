"""Continuous-observation generalized FOGAS solver."""

from __future__ import annotations

import random

import numpy as np
import torch
from tqdm import trange

from ...fogas_clean.continuous_fogas_dataset import ContinuousFOGASDataset


class ContinuousFinalParametrizedSolver:
    """
    Generalized FOGAS solver for continuous observations.

    This class supports neural parametrizations only.  Discrete actions use
    exact sums over actions; continuous actions use Monte Carlo samples from the
    policy for policy expectations.
    """

    _ACTION_TYPES = {"discrete", "continuous"}
    _THETA_MODES = {"reg_adaptive", "reg_fixed", "projection"}
    _THETA_OPTIMIZERS = {"sgd", "adam"}
    _THETA_START_MODES = {"zero", "warm"}
    _BETA_UPDATES = {"fogas_full", "fogas_diag"}
    _POLICY_OPTIMIZERS = {"sgd", "adam"}
    _POLICY_GRADIENTS = {"exact", "reinforce"}
    _EPS = 1e-12

    def __init__(
        self,
        obs_dim,
        action_type,
        gamma,
        x0_obs,
        csv_path,
        u_param,
        q_param,
        policy_param,
        n_actions=None,
        action_dim=None,
        action_samples_per_obs=16,
        seed=42,
        device=None,
        theta_mode="reg_fixed",
        theta_lambda=1e-4,
        theta_optimizer="adam",
        theta_inner_steps=5,
        theta_lr=1e-3,
        theta_start_mode="warm",
        beta_update="fogas_diag",
        beta_reg=1e-3,
        batch_size=None,
        u_jacobian_batch_size=None,
        value_batch_size=None,
        dataset_verbose=False,
    ):
        self.obs_dim = int(obs_dim)
        self.action_type = self._canonical_action_type(action_type)
        self.gamma = float(gamma)
        self.csv_path = csv_path
        self.seed = seed
        self.action_samples_per_obs = self._canonical_positive_int(
            action_samples_per_obs,
            "action_samples_per_obs",
        )

        if self.obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError("gamma must be in [0, 1)")

        if self.action_type == "discrete":
            if n_actions is None:
                raise ValueError("n_actions is required when action_type='discrete'")
            self.n_actions = self._canonical_positive_int(n_actions, "n_actions")
            self.action_dim = 1
        else:
            if action_dim is None:
                raise ValueError("action_dim is required when action_type='continuous'")
            self.action_dim = self._canonical_positive_int(action_dim, "action_dim")
            self.n_actions = None

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self._seed_all(seed)

        self.u_param = u_param.to(self.device)
        self.q_param = q_param.to(self.device)
        self.policy_param = policy_param.to(self.device)

        self.dataset = ContinuousFOGASDataset(
            csv_path=csv_path,
            action_type=self.action_type,
            obs_dim=self.obs_dim,
            action_dim=None if self.action_type == "discrete" else self.action_dim,
            verbose=dataset_verbose,
        )
        if self.dataset.obs_dim != self.obs_dim:
            raise ValueError(f"dataset obs_dim is {self.dataset.obs_dim}, expected {self.obs_dim}")
        if self.action_type == "discrete":
            if torch.any(self.dataset.A >= self.n_actions):
                raise ValueError("dataset actions contain values outside [0, n_actions)")
        elif self.dataset.action_dim != self.action_dim:
            raise ValueError(
                f"dataset action_dim is {self.dataset.action_dim}, expected {self.action_dim}"
            )

        self.Xs = self.dataset.X.to(self.device)
        self.As = self.dataset.A.to(self.device)
        self.Rs = self.dataset.R.to(dtype=torch.float64, device=self.device)
        self.X_nexts = self.dataset.X_next.to(self.device)
        self.Ds = self.dataset.D.to(self.device)
        self.n = self.dataset.n
        if self.n <= 0:
            raise ValueError("dataset must contain at least one transition")

        self.x0_obs = torch.as_tensor(
            x0_obs,
            dtype=self._param_dtype(self.q_param),
            device=self.device,
        ).reshape(1, -1)
        if self.x0_obs.shape[1] != self.obs_dim:
            raise ValueError(f"x0_obs must have shape ({self.obs_dim},)")

        self.d = self._num_trainable_params(self.u_param)
        self.d_q = self._num_trainable_params(self.q_param)
        self.d_pi = self._num_trainable_params(self.policy_param)
        if self.d <= 0 or self.d_q <= 0 or self.d_pi <= 0:
            raise ValueError("u_param, q_param, and policy_param must be trainable modules")

        self._initial_u_flat = self._module_flat_params(self.u_param).detach().clone()
        self._initial_q_flat = self._module_flat_params(self.q_param).detach().clone()
        self._initial_policy_flat = self._module_flat_params(self.policy_param).detach().clone()

        self.theta_mode = self._canonical_theta_mode(theta_mode)
        self.theta_lambda = self._canonical_optional_positive_float(theta_lambda, "theta_lambda")
        self.theta_optimizer = self._canonical_theta_optimizer(theta_optimizer)
        self.theta_inner_steps = self._canonical_positive_int(theta_inner_steps, "theta_inner_steps")
        self.theta_lr = self._canonical_optional_positive_float(theta_lr, "theta_lr")
        self.theta_start_mode = self._canonical_theta_start_mode(theta_start_mode)
        self.beta_update = self._canonical_beta_update(beta_update)
        self.beta_reg = 1.0 if beta_reg is None else float(beta_reg)
        if self.beta_reg < 0.0:
            raise ValueError("beta_reg must be non-negative")
        self.batch_size = self._canonical_optional_positive_int(batch_size, "batch_size")
        self.u_jacobian_batch_size = self._canonical_optional_positive_int(
            u_jacobian_batch_size,
            "u_jacobian_batch_size",
        )
        self.value_batch_size = self._canonical_optional_positive_int(
            value_batch_size,
            "value_batch_size",
        )

        self.R = float(max(torch.max(torch.abs(self.Rs)).detach().cpu().item(), self._EPS))
        self.T = 1000
        self.alpha = 1e-4
        self.eta = 1e-5
        self.rho = 1.0
        self.D_theta = float(np.sqrt(self.d_q / max(1.0 - self.gamma, self._EPS)))

        self.theta = None
        self.theta_bar_history = None
        self.psi_history = None
        self.psi = None
        self.pi = None
        self.lambda_T = None
        self.beta_T = None
        self.diagnostics_history = None

    @classmethod
    def _canonical_action_type(cls, action_type):
        action_type = str(action_type).lower()
        if action_type not in cls._ACTION_TYPES:
            raise ValueError("action_type must be either 'discrete' or 'continuous'")
        return action_type

    @classmethod
    def _canonical_theta_mode(cls, theta_mode):
        theta_mode = str(theta_mode).lower()
        if theta_mode not in cls._THETA_MODES:
            raise ValueError("theta_mode must be one of 'reg_adaptive', 'reg_fixed', or 'projection'")
        return theta_mode

    @classmethod
    def _canonical_theta_optimizer(cls, theta_optimizer):
        theta_optimizer = str(theta_optimizer).lower()
        if theta_optimizer not in cls._THETA_OPTIMIZERS:
            raise ValueError("theta_optimizer must be either 'sgd' or 'adam'")
        return theta_optimizer

    @classmethod
    def _canonical_theta_start_mode(cls, theta_start_mode):
        theta_start_mode = str(theta_start_mode).lower()
        if theta_start_mode not in cls._THETA_START_MODES:
            raise ValueError("theta_start_mode must be either 'zero' or 'warm'")
        return theta_start_mode

    @classmethod
    def _canonical_beta_update(cls, beta_update):
        beta_update = str(beta_update).lower()
        if beta_update not in cls._BETA_UPDATES:
            raise ValueError("beta_update must be either 'fogas_full' or 'fogas_diag'")
        return beta_update

    @classmethod
    def _canonical_policy_optimizer(cls, policy_optimizer):
        policy_optimizer = str(policy_optimizer).lower()
        if policy_optimizer not in cls._POLICY_OPTIMIZERS:
            raise ValueError("policy_optimizer must be either 'sgd' or 'adam'")
        return policy_optimizer

    @classmethod
    def _canonical_policy_gradient(cls, policy_gradient):
        policy_gradient = str(policy_gradient).lower()
        if policy_gradient not in cls._POLICY_GRADIENTS:
            raise ValueError("policy_gradient must be either 'exact' or 'reinforce'")
        return policy_gradient

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
    def _seed_all(seed):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    @staticmethod
    def _trainable_params(module):
        return [p for p in module.parameters() if p.requires_grad]

    @classmethod
    def _num_trainable_params(cls, module):
        return int(sum(p.numel() for p in cls._trainable_params(module)))

    @staticmethod
    def _param_dtype(module, fallback=torch.float64):
        for param in module.parameters():
            return param.dtype
        for buffer in module.buffers():
            return buffer.dtype
        return fallback

    @staticmethod
    def _param_device(module, fallback=None):
        for param in module.parameters():
            return param.device
        for buffer in module.buffers():
            return buffer.device
        return torch.device("cpu") if fallback is None else fallback

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
        flat = flat.detach().reshape(-1)
        if flat.numel() != expected:
            raise ValueError(f"flat parameter size mismatch: expected {expected}, got {flat.numel()}")
        offset = 0
        with torch.no_grad():
            for param in params:
                n = param.numel()
                param.copy_(flat[offset : offset + n].reshape_as(param).to(dtype=param.dtype, device=param.device))
                offset += n

    @classmethod
    def _flat_grads(cls, output, params, retain_graph=False, create_graph=False):
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
            grad = flat_grad[offset : offset + n].reshape_as(param).to(dtype=param.dtype, device=param.device)
            param.grad = sign * grad
            offset += n

    @classmethod
    def _apply_flat_direction(cls, params, direction, step_size):
        offset = 0
        with torch.no_grad():
            for param in params:
                n = param.numel()
                delta = direction[offset : offset + n].reshape_as(param).to(dtype=param.dtype, device=param.device)
                param.add_(step_size * delta)
                offset += n

    def _prepare_module_init(self, module, init, initial_flat, name):
        if init is None:
            flat = initial_flat.clone()
        else:
            flat = init.clone().detach().reshape(-1)
            if flat.numel() != initial_flat.numel():
                raise ValueError(f"{name} must have shape ({initial_flat.numel()},)")
            flat = flat.to(dtype=initial_flat.dtype, device=initial_flat.device)
        self._set_module_flat_params(module, flat)
        return flat.clone()

    def _prepare_theta_bar_init(self, theta_bar_init):
        if theta_bar_init is None:
            return torch.zeros(self.d_q, dtype=torch.float64, device=self.device)
        theta_bar_t = theta_bar_init.clone().to(dtype=torch.float64, device=self.device).reshape(-1)
        if theta_bar_t.shape != (self.d_q,):
            raise ValueError(f"theta_bar_init must have shape ({self.d_q},)")
        return theta_bar_t

    def _action_tensor(self, actions, dtype=None):
        if dtype is None:
            dtype = self._param_dtype(self.q_param)
        if self.action_type == "discrete":
            return torch.as_tensor(actions, dtype=dtype, device=self.device).reshape(-1, 1)
        return torch.as_tensor(actions, dtype=dtype, device=self.device).reshape(-1, self.action_dim)

    def _obs_tensor(self, observations, module=None):
        dtype = self._param_dtype(self.q_param if module is None else module)
        obs = torch.as_tensor(observations, dtype=dtype, device=self.device)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"observations last dimension must be {self.obs_dim}")
        return obs

    def _enumerated_actions(self, n, dtype):
        return torch.arange(self.n_actions, dtype=dtype, device=self.device).repeat(n)

    @staticmethod
    def _batched_ranges(n, batch_size):
        for start in range(0, int(n), int(batch_size)):
            yield start, min(start + int(batch_size), int(n))

    @staticmethod
    def _effective_chunk_size(chunk_size, n):
        return int(n) if chunk_size is None else min(int(chunk_size), int(n))

    def _sample_batch(self):
        if self.batch_size is None or self.batch_size >= self.n:
            indices = torch.arange(self.n, dtype=torch.long, device=self.device)
        else:
            indices = torch.randint(self.n, (int(self.batch_size),), device=self.device)
        return {
            "indices": indices,
            "n": int(indices.numel()),
            "Xs": self.Xs[indices],
            "As": self.As[indices],
            "Rs": self.Rs[indices],
            "X_nexts": self.X_nexts[indices],
            "Ds": self.Ds[indices],
        }

    def _q_sample(self, batch):
        return self.q_param.q(
            batch["Xs"].to(dtype=self._param_dtype(self.q_param)),
            self._action_tensor(batch["As"], dtype=self._param_dtype(self.q_param)),
        )

    def _u_sample_values(self, batch):
        return self.u_param.u(
            batch["Xs"].to(dtype=self._param_dtype(self.u_param)),
            self._action_tensor(batch["As"], dtype=self._param_dtype(self.u_param)),
        )

    def _expected_q_discrete(self, observations, detach_policy):
        dtype = self._param_dtype(self.q_param)
        obs = self._obs_tensor(observations, self.q_param).to(dtype=dtype)
        n = obs.shape[0]
        values = []
        for start, end in self._batched_ranges(n, self._effective_chunk_size(self.value_batch_size, n)):
            obs_batch = obs[start:end]
            batch_n = obs_batch.shape[0]
            repeated_obs = (
                obs_batch[:, None, :]
                .expand(batch_n, self.n_actions, self.obs_dim)
                .reshape(-1, self.obs_dim)
            )
            action_ids = self._enumerated_actions(batch_n, dtype=dtype)
            q_values = self.q_param.q(repeated_obs, action_ids.reshape(-1, 1)).reshape(
                batch_n,
                self.n_actions,
            )
            probs = self.policy_param.probs(
                obs_batch.to(dtype=self._param_dtype(self.policy_param))
            )
            if detach_policy:
                probs = probs.detach()
            values.append((probs.to(dtype=q_values.dtype) * q_values).sum(dim=1))
        return torch.cat(values, dim=0)

    def _expected_q_continuous(self, observations, detach_policy, sample_count=None):
        dtype = self._param_dtype(self.q_param)
        obs = self._obs_tensor(observations, self.q_param).to(dtype=dtype)
        sample_count = self.action_samples_per_obs if sample_count is None else int(sample_count)
        values = []
        n = obs.shape[0]
        for start, end in self._batched_ranges(n, self._effective_chunk_size(self.value_batch_size, n)):
            obs_batch = obs[start:end]
            batch_n = obs_batch.shape[0]
            repeated_obs = (
                obs_batch[:, None, :]
                .expand(batch_n, sample_count, self.obs_dim)
                .reshape(-1, self.obs_dim)
            )
            policy_obs = repeated_obs.to(dtype=self._param_dtype(self.policy_param))
            if detach_policy:
                with torch.no_grad():
                    actions = self.policy_param.sample(policy_obs).detach()
            else:
                actions = self.policy_param.sample(policy_obs)
            q_values = self.q_param.q(repeated_obs, actions.to(dtype=dtype)).reshape(
                batch_n,
                sample_count,
            )
            values.append(q_values.mean(dim=1))
        return torch.cat(values, dim=0)

    def _expected_q(self, observations, detach_policy=True, sample_count=None):
        if self.action_type == "discrete":
            return self._expected_q_discrete(observations, detach_policy=detach_policy)
        return self._expected_q_continuous(
            observations,
            detach_policy=detach_policy,
            sample_count=sample_count,
        )

    def _theta_loss(self, coeff, batch):
        coeff = coeff.detach().to(dtype=self._param_dtype(self.q_param))
        x0_term = self._expected_q(self.x0_obs, detach_policy=True).squeeze(0)
        next_term = self._expected_q(batch["X_nexts"], detach_policy=True)
        q_sample = self._q_sample(batch).to(dtype=next_term.dtype)
        not_done = (~batch["Ds"]).to(dtype=next_term.dtype)
        return (
            (1.0 - self.gamma) * x0_term
            + (self.gamma / batch["n"]) * (coeff * not_done * next_term).sum()
            - (coeff * q_sample).mean()
        )

    def _theta_lambda(self, coeff, batch, effective_D_theta):
        if self.theta_mode == "reg_fixed":
            if self.theta_lambda is None:
                raise ValueError("theta_lambda must be provided when theta_mode='reg_fixed'")
            return self.theta_lambda
        if self.theta_mode == "projection":
            return None
        params = self._trainable_params(self.q_param)
        loss = self._theta_loss(coeff, batch)
        grad = self._flat_grads(loss, params)
        return max(float(torch.linalg.norm(grad).detach().cpu().item()) / effective_D_theta, self._EPS)

    @staticmethod
    def _project_tensor(tensor, radius):
        norm = torch.linalg.norm(tensor)
        if norm <= radius:
            return tensor
        return tensor * (radius / norm.clamp_min(1e-12))

    def _project_module_params(self, module, radius):
        self._set_module_flat_params(module, self._project_tensor(self._module_flat_params(module), radius))

    def _compute_theta_update(self, coeff, batch, effective_D_theta):
        if self.theta_start_mode == "zero":
            self._set_module_flat_params(self.q_param, self._initial_q_flat)

        params = self._trainable_params(self.q_param)
        lambda_theta = self._theta_lambda(coeff, batch, effective_D_theta)
        theta_lr = 1e-2 if self.theta_lr is None and self.theta_optimizer == "adam" else self.theta_lr
        if theta_lr is None:
            theta_lr = 1.0 / max(lambda_theta if lambda_theta is not None else 1.0, self._EPS)

        optimizer_cls = torch.optim.SGD if self.theta_optimizer == "sgd" else torch.optim.Adam
        optimizer = optimizer_cls(params, lr=theta_lr)
        for _ in range(self.theta_inner_steps):
            optimizer.zero_grad(set_to_none=True)
            objective = self._theta_loss(coeff, batch)
            if lambda_theta is not None:
                flat = torch.cat([p.reshape(-1) for p in params])
                objective = objective + 0.5 * lambda_theta * torch.dot(flat, flat)
            objective.backward()
            optimizer.step()
            if self.theta_mode == "projection":
                self._project_module_params(self.q_param, effective_D_theta)

        objective = self._theta_loss(coeff, batch)
        if lambda_theta is not None:
            flat = torch.cat([p.reshape(-1) for p in params])
            objective_for_grad = objective + 0.5 * lambda_theta * torch.dot(flat, flat)
        else:
            objective_for_grad = objective
        grad = self._flat_grads(objective_for_grad, params)
        theta = self._module_flat_params(self.q_param).detach().clone().to(dtype=torch.float64)
        return theta, lambda_theta, objective.detach(), float(torch.linalg.norm(grad).detach().cpu().item()), theta_lr

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
                row.append(torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1))
            rows.append(torch.cat(row))
        return torch.stack(rows, dim=0)

    def _u_sample_jacobian_batches(self, batch):
        params = self._trainable_params(self.u_param)
        if not params:
            return

        dtype = self._param_dtype(self.u_param)
        observations = batch["Xs"].to(dtype=dtype)
        actions = self._action_tensor(batch["As"], dtype=dtype)
        batch_size = self._effective_chunk_size(self.u_jacobian_batch_size, batch["n"])

        try:
            from torch.func import functional_call, grad, vmap
        except ImportError:
            for start, end in self._batched_ranges(batch["n"], batch_size):
                outputs = self.u_param.u(observations[start:end], actions[start:end])
                yield self._jacobian_outputs(outputs, params)
            return

        named_params = {
            name: param
            for name, param in self.u_param.named_parameters()
            if param.requires_grad
        }
        buffers = dict(self.u_param.named_buffers())
        param_names = list(named_params)

        def single_u(param_dict, buffer_dict, observation, action):
            value = functional_call(
                self.u_param,
                (param_dict, buffer_dict),
                (observation, action),
            )
            return value.reshape(())

        per_sample_grad = vmap(
            grad(single_u),
            in_dims=(None, None, 0, 0),
        )

        for start, end in self._batched_ranges(batch["n"], batch_size):
            grads = per_sample_grad(
                named_params,
                buffers,
                observations[start:end],
                actions[start:end],
            )
            yield torch.cat(
                [grads[name].reshape(end - start, -1) for name in param_names],
                dim=1,
            )

    def _u_sample_jacobian(self, batch):
        chunks = list(self._u_sample_jacobian_batches(batch))
        if not chunks:
            return torch.empty(0, 0, dtype=torch.float64, device=self.device)
        return torch.cat(chunks, dim=0)

    def _u_sample_feature_batches(self, batch):
        params = self._trainable_params(self.u_param)
        beta_param = getattr(self.u_param, "beta", None)
        features = getattr(self.u_param, "features", None)
        if (
            len(params) != 1
            or beta_param is not params[0]
            or features is None
            or not callable(features)
            or params[0].numel() != self.d
        ):
            return None

        dtype = params[0].dtype
        observations = batch["Xs"].to(dtype=dtype)
        actions = self._action_tensor(batch["As"], dtype=dtype)
        chunk_size = self._effective_chunk_size(self.u_jacobian_batch_size, batch["n"])

        def feature_batches():
            with torch.no_grad():
                for start, end in self._batched_ranges(batch["n"], chunk_size):
                    matrix = features(observations[start:end], actions[start:end])
                    matrix = matrix.reshape(end - start, -1)
                    if matrix.shape[1] != self.d:
                        raise ValueError(
                            "u_param.features returned an incompatible feature dimension: "
                            f"expected {self.d}, got {matrix.shape[1]}"
                        )
                    yield matrix

        return feature_batches()

    def _compute_beta_update_direction(self, td_error, batch):
        param_count = self.d
        beta_grad = None
        h_acc = None
        diag_acc = None
        td_error = td_error.detach()
        u_gradient_batches = self._u_sample_feature_batches(batch)
        if u_gradient_batches is None:
            u_gradient_batches = self._u_sample_jacobian_batches(batch)

        if self.beta_update == "fogas_full":
            start = 0
            for jacobian in u_gradient_batches:
                jacobian = jacobian.detach()
                end = start + jacobian.shape[0]
                td = td_error[start:end].to(dtype=jacobian.dtype)
                if beta_grad is None:
                    beta_grad = torch.zeros(param_count, dtype=jacobian.dtype, device=jacobian.device)
                    h_acc = torch.zeros(
                        param_count,
                        param_count,
                        dtype=jacobian.dtype,
                        device=jacobian.device,
                    )
                beta_grad = beta_grad + jacobian.T @ td
                h_acc = h_acc + jacobian.T @ jacobian
                start = end

            beta_grad = beta_grad / batch["n"]
            H = h_acc / batch["n"]
            H = H + self.beta_reg * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
            direction = torch.linalg.solve(H, beta_grad.to(dtype=H.dtype))
            diagnostics = {
                "beta_update": self.beta_update,
                "beta_diag_min": float(torch.diagonal(H).min().detach().cpu().item()),
                "beta_diag_max": float(torch.diagonal(H).max().detach().cpu().item()),
            }
        else:
            start = 0
            for jacobian in u_gradient_batches:
                jacobian = jacobian.detach()
                end = start + jacobian.shape[0]
                td = td_error[start:end].to(dtype=jacobian.dtype)
                if beta_grad is None:
                    beta_grad = torch.zeros(param_count, dtype=jacobian.dtype, device=jacobian.device)
                    diag_acc = torch.zeros(param_count, dtype=jacobian.dtype, device=jacobian.device)
                beta_grad = beta_grad + jacobian.T @ td
                diag_acc = diag_acc + (jacobian * jacobian).sum(dim=0)
                start = end

            beta_grad = beta_grad / batch["n"]
            diag_h = diag_acc / batch["n"] + self.beta_reg
            diag_h = diag_h.clamp_min(self._EPS)
            direction = beta_grad.to(dtype=diag_h.dtype) / diag_h
            diagnostics = {
                "beta_update": self.beta_update,
                "beta_diag_min": float(diag_h.min().detach().cpu().item()),
                "beta_diag_max": float(diag_h.max().detach().cpu().item()),
            }
        return direction.detach(), beta_grad.detach(), diagnostics

    def _policy_support(self, coeff, batch):
        x0 = self.x0_obs.to(dtype=self._param_dtype(self.policy_param))
        next_obs = batch["X_nexts"].to(dtype=self._param_dtype(self.policy_param))
        observations = torch.cat([x0, next_obs], dim=0)
        weights = torch.cat(
            [
                torch.tensor([1.0 - self.gamma], dtype=torch.float64, device=self.device),
                (self.gamma / batch["n"])
                * coeff.detach().to(dtype=torch.float64)
                * (~batch["Ds"]).to(dtype=torch.float64),
            ]
        )
        return observations, weights

    def _exact_policy_gradient_discrete(self, coeff, batch):
        observations, weights = self._policy_support(coeff, batch)
        q_expected = self._expected_q_discrete(observations, detach_policy=False)
        objective = (weights.to(dtype=q_expected.dtype) * q_expected).sum()
        params = self._trainable_params(self.policy_param)
        grad = self._flat_grads(objective, params)
        return grad.detach(), objective.detach()

    def _reinforce_policy_gradient(self, coeff, batch, reinforce_samples):
        observations, weights = self._policy_support(coeff, batch)
        sample_count = self._canonical_positive_int(reinforce_samples, "reinforce_samples")
        params = self._trainable_params(self.policy_param)

        if self.action_type == "discrete":
            with torch.no_grad():
                probs = self.policy_param.probs(observations).detach().clamp_min(self._EPS)
                sampled_actions = torch.multinomial(probs, num_samples=sample_count, replacement=True)
                obs_rep = observations[:, None, :].expand(-1, sample_count, self.obs_dim).reshape(-1, self.obs_dim)
                action_rep = sampled_actions.reshape(-1)
                q_values = self.q_param.q(
                    obs_rep.to(dtype=self._param_dtype(self.q_param)),
                    action_rep.to(dtype=self._param_dtype(self.q_param)).reshape(-1, 1),
                ).detach().reshape(observations.shape[0], sample_count)
                baseline = (probs.to(dtype=q_values.dtype) * self.q_values(observations).detach()).sum(dim=1)
                advantages = q_values - baseline[:, None]
            log_probs = self.policy_param.log_prob_actions(
                observations[:, None, :].expand(-1, sample_count, self.obs_dim).reshape(-1, self.obs_dim),
                sampled_actions.reshape(-1),
            ).reshape(observations.shape[0], sample_count)
        else:
            with torch.no_grad():
                obs_rep = observations[:, None, :].expand(-1, sample_count, self.obs_dim).reshape(-1, self.obs_dim)
                sampled_actions = self.policy_param.sample(obs_rep).detach()
                q_values = self.q_param.q(
                    obs_rep.to(dtype=self._param_dtype(self.q_param)),
                    sampled_actions.to(dtype=self._param_dtype(self.q_param)),
                ).detach().reshape(observations.shape[0], sample_count)
                advantages = q_values - q_values.mean(dim=1, keepdim=True)
            log_probs = self.policy_param.log_prob_actions(obs_rep, sampled_actions).reshape(
                observations.shape[0],
                sample_count,
            )

        surrogate = (
            weights[:, None].to(dtype=log_probs.dtype)
            * advantages.to(dtype=log_probs.dtype)
            * log_probs
        ).sum() / float(sample_count)
        grad = self._flat_grads(surrogate, params)
        objective = (weights.to(dtype=q_values.dtype)[:, None] * q_values).mean(dim=1).sum()
        return grad.detach(), objective.detach()

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
        theta_mode=None,
        theta_optimizer=None,
        theta_lr=None,
        theta_inner_steps=None,
        theta_start_mode=None,
        theta_lambda=None,
        beta_update=None,
        policy_optimizer="adam",
        policy_gradient="exact",
        reinforce_samples=1,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        verbose=False,
        tqdm_print=False,
        log_interval=None,
        checkpoint_callback=None,
    ):
        T = self.T if T is None else int(T)
        alpha = self.alpha if alpha is None else float(alpha)
        eta = self.eta if eta is None else float(eta)
        rho = self.rho if rho is None else float(rho)
        D_theta = self.D_theta if D_theta is None else float(D_theta)
        if T <= 0:
            raise ValueError("T must be positive")
        if alpha <= 0.0 or eta <= 0.0 or rho < 0.0 or D_theta <= 0.0:
            raise ValueError("alpha, eta, and D_theta must be positive; rho must be nonnegative")

        old_settings = (
            self.theta_mode,
            self.theta_optimizer,
            self.theta_lr,
            self.theta_inner_steps,
            self.theta_start_mode,
            self.theta_lambda,
            self.beta_update,
        )
        if theta_mode is not None:
            self.theta_mode = self._canonical_theta_mode(theta_mode)
        if theta_optimizer is not None:
            self.theta_optimizer = self._canonical_theta_optimizer(theta_optimizer)
        if theta_lr is not None:
            self.theta_lr = self._canonical_positive_float(theta_lr, "theta_lr")
        if theta_inner_steps is not None:
            self.theta_inner_steps = self._canonical_positive_int(theta_inner_steps, "theta_inner_steps")
        if theta_start_mode is not None:
            self.theta_start_mode = self._canonical_theta_start_mode(theta_start_mode)
        if theta_lambda is not None:
            self.theta_lambda = self._canonical_positive_float(theta_lambda, "theta_lambda")
        if beta_update is not None:
            self.beta_update = self._canonical_beta_update(beta_update)

        try:
            return self._run_impl(
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
                adam_betas,
                adam_eps,
                verbose,
                tqdm_print,
                log_interval,
                checkpoint_callback,
            )
        finally:
            (
                self.theta_mode,
                self.theta_optimizer,
                self.theta_lr,
                self.theta_inner_steps,
                self.theta_start_mode,
                self.theta_lambda,
                self.beta_update,
            ) = old_settings

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
        adam_betas,
        adam_eps,
        verbose,
        tqdm_print,
        log_interval,
        checkpoint_callback,
    ):
        policy_optimizer = self._canonical_policy_optimizer(policy_optimizer)
        policy_gradient = self._canonical_policy_gradient(policy_gradient)
        if self.action_type == "continuous" and policy_gradient == "exact":
            raise ValueError("policy_gradient='exact' is only supported for discrete actions")

        beta_t = self._prepare_module_init(self.u_param, beta_init, self._initial_u_flat, "beta_init")
        theta_bar_t = self._prepare_theta_bar_init(theta_bar_init)
        self._prepare_module_init(self.policy_param, psi_init, self._initial_policy_flat, "psi_init")
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
        iterator = trange(T, desc="ContinuousFinalParametrizedSolver", disable=not (tqdm_print and not verbose))
        theta_bar_history = []
        psi_history = []
        diagnostics_history = []
        final_theta = self._module_flat_params(self.q_param).detach().clone().to(dtype=torch.float64)

        for t in iterator:
            batch = self._sample_batch()
            coeff = self._u_sample_values(batch).detach().to(dtype=torch.float64)
            theta_t, lambda_theta, q_objective, theta_grad_norm, theta_lr_used = self._compute_theta_update(
                coeff=coeff,
                batch=batch,
                effective_D_theta=D_theta,
            )
            final_theta = theta_t.detach().clone().to(dtype=torch.float64)

            q_current = self._q_sample(batch).to(dtype=torch.float64)
            v_next = self._expected_q(batch["X_nexts"], detach_policy=True).detach().to(dtype=torch.float64)
            v_x0 = self._expected_q(self.x0_obs, detach_policy=True).squeeze(0).detach().to(dtype=torch.float64)
            td_error = batch["Rs"] + self.gamma * (~batch["Ds"]).to(dtype=torch.float64) * v_next - q_current
            beta_objective = (coeff * td_error).mean()
            total_loss = (1.0 - self.gamma) * v_x0 + beta_objective

            beta_update_direction, beta_grad, beta_diagnostics = self._compute_beta_update_direction(td_error, batch)
            beta_t = (1.0 / (1.0 + rho * eta)) * (
                self._module_flat_params(self.u_param).to(dtype=beta_update_direction.dtype)
                + eta * beta_update_direction
            )
            self._set_module_flat_params(self.u_param, beta_t)

            if policy_gradient == "exact":
                policy_grad, policy_objective = self._exact_policy_gradient_discrete(coeff, batch)
            else:
                policy_grad, policy_objective = self._reinforce_policy_gradient(coeff, batch, reinforce_samples)

            if policy_optimizer == "sgd":
                policy_direction = policy_grad
                policy_direction_kind = "sgd_gradient"
                self._apply_flat_direction(self._trainable_params(self.policy_param), policy_direction, alpha)
            else:
                policy_adam_optimizer.zero_grad(set_to_none=True)
                self._assign_flat_grad(self._trainable_params(self.policy_param), policy_grad, sign=-1.0)
                policy_adam_optimizer.step()
                policy_direction = policy_grad
                policy_direction_kind = "adam_gradient"

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
                "policy_direction_norm": float(torch.linalg.norm(policy_direction).detach().cpu().item()),
                "beta_grad_norm": float(torch.linalg.norm(beta_grad).detach().cpu().item()),
                "beta_direction_norm": float(torch.linalg.norm(beta_update_direction).detach().cpu().item()),
                "theta_grad_norm": float(theta_grad_norm),
                "theta_norm": float(torch.linalg.norm(final_theta).detach().cpu().item()),
                "theta_mode": self.theta_mode,
                "theta_optimizer": self.theta_optimizer,
                "theta_start_mode": self.theta_start_mode,
                "theta_lambda": None if lambda_theta is None else float(lambda_theta),
                "theta_lr": float(theta_lr_used),
                "policy_optimizer": policy_optimizer,
                "policy_gradient": policy_gradient,
                "policy_direction": policy_direction_kind,
                "reinforce_samples": int(reinforce_samples),
                "D_theta": float(D_theta),
                "u_fast_path": False,
                "q_fast_path": False,
                "policy_fast_path": False,
            }
            diagnostics.update(beta_diagnostics)
            diagnostics_history.append(diagnostics)

            if checkpoint_callback is not None:
                checkpoint_callback(self, t + 1, diagnostics)

            if tqdm_print and not verbose:
                iterator.set_postfix(total_loss=f"{diagnostics['total_loss']:.3e}")
            if verbose and (t % log_interval == 0):
                values = " ".join(
                    f"{key}={value:.6e}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in diagnostics.items()
                )
                print(f"[ContinuousFinalParametrizedSolver] Iter {t + 1}/{T} {values}")

        self.theta = final_theta.clone()
        self.theta_bar_history = theta_bar_history
        self.psi_history = psi_history
        self.psi = self._module_flat_params(self.policy_param).detach().clone().to(dtype=torch.float64)
        self.pi = None
        self.lambda_T = self._module_flat_params(self.u_param).detach().clone().to(dtype=torch.float64)
        self.beta_T = self.lambda_T.clone()
        self.diagnostics_history = diagnostics_history
        return self.policy_param

    def policy_probs(self, observations):
        if self.action_type != "discrete":
            raise ValueError("policy_probs is only available for discrete actions")
        with torch.no_grad():
            obs = self._obs_tensor(observations, self.policy_param)
            return self.policy_param.probs(obs).detach().clone()

    def q_values(self, observations):
        if self.action_type != "discrete":
            raise ValueError("q_values is only available for discrete actions")
        obs = self._obs_tensor(observations, self.q_param)
        n = obs.shape[0]
        repeated_obs = obs[:, None, :].expand(n, self.n_actions, self.obs_dim).reshape(-1, self.obs_dim)
        action_ids = self._enumerated_actions(n, dtype=self._param_dtype(self.q_param))
        return self.q_param.q(repeated_obs, action_ids.reshape(-1, 1)).reshape(n, self.n_actions)

    def q(self, observations, actions):
        obs = self._obs_tensor(observations, self.q_param)
        action_tensor = self._action_tensor(actions, dtype=self._param_dtype(self.q_param))
        return self.q_param.q(obs, action_tensor)

    def sample_action(self, observation, deterministic=False):
        obs = self._obs_tensor(observation, self.policy_param)
        with torch.no_grad():
            action = self.policy_param.sample(obs, deterministic=deterministic)
        if self.action_type == "discrete":
            return int(action.reshape(-1)[0].detach().cpu().item())
        return action.reshape(-1, self.action_dim)[0].detach().cpu().numpy()
