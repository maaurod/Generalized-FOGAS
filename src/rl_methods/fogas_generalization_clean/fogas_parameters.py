import math

import numpy as np


class GeneralizedFOGASParameters:
    """
    FOGAS parameter defaults for generalized solvers that do not receive an mdp.

    The caller provides the problem metadata and the reward/feature bound that
    older solvers obtained from the mdp object.
    """

    def __init__(
        self,
        n,
        reward_bound,
        n_states,
        n_actions,
        feature_dim,
        gamma,
        delta=0.05,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        D_theta=None,
        beta_reg=None,
        print_params=False,
    ):
        self.delta = delta
        self.n = int(n)
        self.R = float(reward_bound)
        self.N = int(n_states)
        self.A = int(n_actions)
        self.d = int(feature_dim)
        self.gamma = float(gamma)
        self.overrides = {
            "T": T,
            "alpha": alpha,
            "eta": eta,
            "rho": rho,
            "D_theta": D_theta,
            "beta_reg": beta_reg,
        }
        self.compute(T, alpha, eta, rho, D_theta, beta_reg)
        if print_params:
            self.print_summary()

    def compute(self, T, alpha, eta, rho, D_theta, beta_reg):
        R = max(float(self.R), 1e-12)
        n = self.n
        A = self.A
        d = self.d
        gamma = self.gamma
        delta = self.delta

        self.T_min = 2 * (R**2) * n * np.log(A) / np.log(1 / delta)
        self.T = math.ceil(self.T_min) if T is None else T
        self.D_theta = np.sqrt(d / (1 - gamma)) if D_theta is None else D_theta
        self.alpha = (
            np.sqrt(2 * (1 - gamma) ** 2 * np.log(A) / (R**2 * d * self.T))
            if alpha is None
            else alpha
        )
        self.rho = (
            gamma
            * np.sqrt((320 * d**2 * np.log(2 * self.T / delta)) / ((1 - gamma) ** 2 * n))
            if rho is None
            else rho
        )
        self.eta = (
            np.sqrt(((1 - gamma) ** 2) / (27 * R**2 * d**2 * self.T))
            if eta is None
            else eta
        )
        self.D_pi = self.alpha * self.T * self.D_theta
        self.beta_reg = R**2 / (d * self.T) if beta_reg is None else beta_reg

    def print_summary(self):
        def fmt(name, theoretical_value, formatter="{:.6f}"):
            override = self.overrides.get(name)
            if override is None:
                return formatter.format(theoretical_value)
            theo = formatter.format(theoretical_value)
            new = formatter.format(override)
            return f"{theo}   (overridden -> {new})"

        print("\n================ FOGAS PARAMETER SUMMARY ================\n")
        print("Basic Information")
        print("-----------------")
        print(f"{'Dataset size n:':25s} {self.n}")
        print(f"{'Reward bound R:':25s} {self.R:.4f}")
        print(f"{'Num states N:':25s} {self.N}")
        print(f"{'Num actions A:':25s} {self.A}")
        print(f"{'u feature dim d:':25s} {self.d}")
        print(f"{'Discount gamma:':25s} {self.gamma}")
        print(f"{'Confidence delta:':25s} {self.delta}")
        print("")
        print("FOGAS Hyperparameters")
        print("---------------------")
        print(f"{'T_min (theoretical):':25s} {self.T_min}")
        print(f"{'T (iterations):':25s}      {fmt('T', self.T, '{:.0f}')}")
        print(f"{'alpha:':25s}     {fmt('alpha', self.alpha)}")
        print(f"{'rho:':25s}       {fmt('rho', self.rho)}")
        print(f"{'eta:':25s}       {fmt('eta', self.eta)}")
        print(f"{'D_theta:':25s}   {fmt('D_theta', self.D_theta)}")
        print(f"{'beta_reg (ridge):':25s} {fmt('beta_reg', self.R**2 / (self.d * self.T))}")
        print(f"{'D_pi (derived):':25s} {self.D_pi:.6f}")
        print("\n=========================================================\n")


StandaloneFOGASParameters = GeneralizedFOGASParameters

__all__ = ["GeneralizedFOGASParameters", "StandaloneFOGASParameters"]
