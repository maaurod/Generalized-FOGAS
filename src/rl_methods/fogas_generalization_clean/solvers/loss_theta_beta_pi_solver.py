import torch

from .linear_beta_pi_solver import LinearBetaPiSolver


class LossThetaBetaPiSolver(LinearBetaPiSolver):
    """
    Linear beta-policy FOGAS solver with a loss-derived theta update.

    The critic remains linear in the same features as u_beta:

        Q_theta(x, a) = <theta, phi(x, a)>.

    For fixed beta and policy, theta is selected by minimizing the empirical
    linear theta loss over the D_theta ball. By default the mismatch is the
    pure empirical loss mismatch. Set theta_loss_include_beta_reg=True to add
    the beta regularization contribution and recover the old LinearBetaPiSolver
    theta update when beta_reg > 0.
    """

    def __init__(self, *args, theta_loss_include_beta_reg=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_loss_include_beta_reg = self._canonical_theta_loss_include_beta_reg(
            theta_loss_include_beta_reg
        )

    @staticmethod
    def _canonical_theta_loss_include_beta_reg(theta_loss_include_beta_reg):
        if isinstance(theta_loss_include_beta_reg, bool):
            return theta_loss_include_beta_reg
        raise ValueError("theta_loss_include_beta_reg must be a bool")

    def run(self, *args, theta_loss_include_beta_reg=None, **kwargs):
        previous_value = self.theta_loss_include_beta_reg
        if theta_loss_include_beta_reg is not None:
            self.theta_loss_include_beta_reg = self._canonical_theta_loss_include_beta_reg(
                theta_loss_include_beta_reg
            )
        try:
            return super().run(*args, **kwargs)
        finally:
            self.theta_loss_include_beta_reg = previous_value

    def _compute_theta_update(self, emp_feature_occupancy, beta_t, D_theta):
        loss_mismatch = emp_feature_occupancy - (self.Empirical_cov @ beta_t)
        if self.theta_loss_include_beta_reg:
            c_t = loss_mismatch - self._preconditioner_beta_reg * beta_t
        else:
            c_t = loss_mismatch

        norm_c = torch.linalg.norm(c_t)
        theta_t = torch.zeros_like(c_t) if norm_c < 1e-12 else -D_theta * c_t / norm_c
        diagnostics = {
            "theta_update": "loss_exact",
            "theta_loss_include_beta_reg": self.theta_loss_include_beta_reg,
            "theta_loss_mismatch_norm": float(norm_c.detach().cpu().item()),
            "theta_loss_pure_mismatch_norm": float(
                torch.linalg.norm(loss_mismatch).detach().cpu().item()
            ),
        }
        return theta_t, c_t, norm_c, diagnostics
