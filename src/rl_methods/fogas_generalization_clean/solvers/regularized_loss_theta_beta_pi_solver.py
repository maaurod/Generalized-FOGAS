import torch

from .loss_theta_beta_pi_solver import LossThetaBetaPiSolver


class RegularizedLossThetaBetaPiSolver(LossThetaBetaPiSolver):
    """
    Loss-theta beta-policy FOGAS solver with a quadratic theta regularizer.

    For fixed beta and policy, this replaces the hard D_theta-ball best response
    with the regularized value update

        min_theta <theta, m_theta> + 0.5 * lambda_theta * ||theta||_2^2.

    The mismatch m_theta is the same loss-derived mismatch used by
    LossThetaBetaPiSolver, including the optional ridge correction controlled by
    theta_loss_include_beta_reg.
    """

    _THETA_UPDATES = {"exact", "sgd", "adam"}
    _THETA_LAMBDA_MODES = {"adaptive", "initial", "fixed"}
    _THETA_INNER_INITS = {"zero"}
    _EPS = 1e-12

    def __init__(
        self,
        *args,
        theta_update="exact",
        theta_lambda_mode="adaptive",
        theta_lambda=None,
        theta_inner_steps=100,
        theta_lr=None,
        theta_adam_betas=(0.9, 0.999),
        theta_adam_eps=1e-8,
        theta_inner_init="zero",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.theta_update = self._canonical_theta_update(theta_update)
        self.theta_lambda_mode = self._canonical_theta_lambda_mode(theta_lambda_mode)
        self.theta_lambda = self._canonical_optional_positive_float(theta_lambda, "theta_lambda")
        self.theta_inner_steps = self._canonical_theta_inner_steps(theta_inner_steps)
        self.theta_lr = self._canonical_optional_positive_float(theta_lr, "theta_lr")
        self.theta_adam_betas = self._canonical_theta_adam_betas(theta_adam_betas)
        self.theta_adam_eps = self._canonical_positive_float(theta_adam_eps, "theta_adam_eps")
        self.theta_inner_init = self._canonical_theta_inner_init(theta_inner_init)
        self._initial_theta_lambda = None
        self._validate_theta_lambda_configuration()

    @classmethod
    def _canonical_theta_update(cls, theta_update):
        name = str(theta_update).lower()
        if name not in cls._THETA_UPDATES:
            raise ValueError("theta_update must be one of 'exact', 'sgd', or 'adam'")
        return name

    @classmethod
    def _canonical_theta_lambda_mode(cls, theta_lambda_mode):
        name = str(theta_lambda_mode).lower()
        if name not in cls._THETA_LAMBDA_MODES:
            raise ValueError("theta_lambda_mode must be one of 'adaptive', 'initial', or 'fixed'")
        return name

    @classmethod
    def _canonical_theta_inner_init(cls, theta_inner_init):
        name = str(theta_inner_init).lower()
        if name not in cls._THETA_INNER_INITS:
            raise ValueError("theta_inner_init must be 'zero'")
        return name

    @staticmethod
    def _canonical_theta_inner_steps(theta_inner_steps):
        theta_inner_steps = int(theta_inner_steps)
        if theta_inner_steps <= 0:
            raise ValueError("theta_inner_steps must be positive")
        return theta_inner_steps

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
    def _canonical_theta_adam_betas(theta_adam_betas):
        if len(theta_adam_betas) != 2:
            raise ValueError("theta_adam_betas must contain two values")
        beta1, beta2 = (float(theta_adam_betas[0]), float(theta_adam_betas[1]))
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("theta_adam_betas values must be in [0, 1)")
        return (beta1, beta2)

    def _validate_theta_lambda_configuration(self):
        if self.theta_lambda_mode == "fixed" and self.theta_lambda is None:
            raise ValueError("theta_lambda must be provided when theta_lambda_mode='fixed'")

    def run(
        self,
        *args,
        theta_update=None,
        theta_lambda_mode=None,
        theta_lambda=None,
        theta_inner_steps=None,
        theta_lr=None,
        theta_adam_betas=None,
        theta_adam_eps=None,
        theta_inner_init=None,
        theta_loss_include_beta_reg=None,
        **kwargs,
    ):
        previous_values = {
            "theta_update": self.theta_update,
            "theta_lambda_mode": self.theta_lambda_mode,
            "theta_lambda": self.theta_lambda,
            "theta_inner_steps": self.theta_inner_steps,
            "theta_lr": self.theta_lr,
            "theta_adam_betas": self.theta_adam_betas,
            "theta_adam_eps": self.theta_adam_eps,
            "theta_inner_init": self.theta_inner_init,
            "_initial_theta_lambda": self._initial_theta_lambda,
        }

        if theta_update is not None:
            self.theta_update = self._canonical_theta_update(theta_update)
        if theta_lambda_mode is not None:
            self.theta_lambda_mode = self._canonical_theta_lambda_mode(theta_lambda_mode)
        if theta_lambda is not None:
            self.theta_lambda = self._canonical_positive_float(theta_lambda, "theta_lambda")
        if theta_inner_steps is not None:
            self.theta_inner_steps = self._canonical_theta_inner_steps(theta_inner_steps)
        if theta_lr is not None:
            self.theta_lr = self._canonical_positive_float(theta_lr, "theta_lr")
        if theta_adam_betas is not None:
            self.theta_adam_betas = self._canonical_theta_adam_betas(theta_adam_betas)
        if theta_adam_eps is not None:
            self.theta_adam_eps = self._canonical_positive_float(theta_adam_eps, "theta_adam_eps")
        if theta_inner_init is not None:
            self.theta_inner_init = self._canonical_theta_inner_init(theta_inner_init)

        self._initial_theta_lambda = None
        self._validate_theta_lambda_configuration()
        try:
            return super().run(
                *args,
                theta_loss_include_beta_reg=theta_loss_include_beta_reg,
                **kwargs,
            )
        finally:
            for name, value in previous_values.items():
                setattr(self, name, value)

    def _compute_theta_update(self, emp_feature_occupancy, beta_t, D_theta):
        loss_mismatch = emp_feature_occupancy - (self.Empirical_cov @ beta_t)
        if self.theta_loss_include_beta_reg:
            c_t = loss_mismatch - self._preconditioner_beta_reg * beta_t
        else:
            c_t = loss_mismatch

        norm_c = torch.linalg.norm(c_t)
        lambda_theta = self._resolve_theta_lambda(norm_c, D_theta)

        if norm_c < self._EPS and self.theta_update == "exact":
            theta_t = torch.zeros_like(c_t)
            inner_grad_norm = 0.0
            theta_lr_used = None
        elif self.theta_update == "exact":
            theta_t = -c_t / lambda_theta
            inner_grad_norm = float(
                torch.linalg.norm(c_t + lambda_theta * theta_t).detach().cpu().item()
            )
            theta_lr_used = None
        elif self.theta_update == "sgd":
            theta_lr_used = self.theta_lr if self.theta_lr is not None else 1.0 / lambda_theta
            theta_t, inner_grad_norm = self._sgd_theta_update(c_t, lambda_theta, theta_lr_used)
        else:
            theta_lr_used = self.theta_lr if self.theta_lr is not None else 1e-2
            theta_t, inner_grad_norm = self._adam_theta_update(c_t, lambda_theta, theta_lr_used)

        regularized_objective = torch.dot(theta_t, c_t) + 0.5 * lambda_theta * torch.dot(
            theta_t, theta_t
        )
        diagnostics = {
            "theta_update": f"regularized_loss_{self.theta_update}",
            "theta_lambda_mode": self.theta_lambda_mode,
            "theta_lambda": float(lambda_theta),
            "theta_inner_steps": self.theta_inner_steps,
            "theta_lr": None if theta_lr_used is None else float(theta_lr_used),
            "theta_regularized_objective": float(regularized_objective.detach().cpu().item()),
            "theta_loss_include_beta_reg": self.theta_loss_include_beta_reg,
            "theta_loss_mismatch_norm": float(norm_c.detach().cpu().item()),
            "theta_loss_pure_mismatch_norm": float(
                torch.linalg.norm(loss_mismatch).detach().cpu().item()
            ),
            "theta_inner_grad_norm": float(inner_grad_norm),
        }
        return theta_t, c_t, norm_c, diagnostics

    def _resolve_theta_lambda(self, norm_c, D_theta):
        if self.theta_lambda_mode == "fixed":
            return self.theta_lambda

        D_theta = float(D_theta)
        if D_theta <= 0.0:
            raise ValueError("D_theta must be positive for adaptive or initial theta lambda")

        candidate = max(float(norm_c.detach().cpu().item()) / D_theta, self._EPS)
        if self.theta_lambda_mode == "adaptive":
            return candidate

        if self._initial_theta_lambda is None:
            self._initial_theta_lambda = candidate
        return self._initial_theta_lambda

    def _sgd_theta_update(self, c_t, lambda_theta, theta_lr):
        theta_t = torch.zeros_like(c_t)
        for _ in range(self.theta_inner_steps):
            grad = c_t + lambda_theta * theta_t
            theta_t = theta_t - theta_lr * grad
        final_grad = c_t + lambda_theta * theta_t
        return theta_t, float(torch.linalg.norm(final_grad).detach().cpu().item())

    def _adam_theta_update(self, c_t, lambda_theta, theta_lr):
        theta_param = torch.nn.Parameter(torch.zeros_like(c_t))
        optimizer = torch.optim.Adam(
            [theta_param],
            lr=theta_lr,
            betas=self.theta_adam_betas,
            eps=self.theta_adam_eps,
        )
        for _ in range(self.theta_inner_steps):
            optimizer.zero_grad(set_to_none=True)
            objective = torch.dot(theta_param, c_t) + 0.5 * lambda_theta * torch.dot(
                theta_param, theta_param
            )
            objective.backward()
            optimizer.step()
        with torch.no_grad():
            theta_t = theta_param.detach().clone()
            final_grad = c_t + lambda_theta * theta_t
            inner_grad_norm = float(torch.linalg.norm(final_grad).detach().cpu().item())
        return theta_t, inner_grad_norm
