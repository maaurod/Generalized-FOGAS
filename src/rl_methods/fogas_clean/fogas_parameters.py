"""
FOGASParameters
---------------

Computes all theoretical constants required by the FOGAS algorithm.
Formulas follow the notation from the paper and depend only on:

    - dataset size n
    - feature bound R
    - feature dimension d
    - number of actions A
    - discount factor γ
    - confidence δ

If the user overrides T, alpha, eta, rho, or D_theta,
the summary printout will show a comparison between:

    theoretical_value  (overridden → new_value)
"""

import numpy as np
import math


class FOGASParameters:
    def __init__(self, N, A, gamma, d, R, n, delta=0.05,
                 T=None, alpha=None, eta=None, rho=None, D_theta=None,
                 beta=None, print_params=False):

        # Store core inputs
        self.delta = delta
        self.n = n
        self.R = float(R)
        self.N = int(N)
        self.A = int(A)
        self.d = int(d)
        self.gamma = float(gamma)

        # Store overrides for summary printing
        self.overrides = {
            "T": T,
            "alpha": alpha,
            "eta": eta,
            "rho": rho,
            "D_theta": D_theta,
            "beta": beta,
        }

        # Compute all parameters
        self.compute(T, alpha, eta, rho, D_theta, beta)

        # Optional pretty-print summary
        if print_params:
            self.print_summary()

    # ----------------------------------------------------------------------
    # PARAMETER COMPUTATION
    # ----------------------------------------------------------------------
    def compute(self, T, alpha, eta, rho, D_theta, beta):
        delta = self.delta
        R = self.R
        n = self.n
        A = self.A
        d = self.d
        gamma = self.gamma

        # --- Minimal required iterations ---
        self.T_min = 2 * (R**2) * n * np.log(A) / np.log(1 / delta)
        self.T = math.ceil(self.T_min) if T is None else T

        # --- Radius of primal ball ---
        self.D_theta = np.sqrt(d / (1 - gamma)) if D_theta is None else D_theta

        # --- Learning rates ---
        self.alpha = (
            np.sqrt(2 * (1 - gamma)**2 * np.log(A) / (R**2 * d * self.T))
            if alpha is None else alpha
        )

        self.rho = (
            gamma * np.sqrt((320 * d**2 * np.log(2 * self.T / delta)) /
                            ((1 - gamma)**2 * n))
            if rho is None else rho
        )

        self.eta = (
            np.sqrt(((1 - gamma)**2) / (27 * R**2 * d**2 * self.T))
            if eta is None else eta
        )

        # --- Additional derived quantities ---
        self.D_pi = self.alpha * self.T * self.D_theta
        self.beta = R**2 / (d * self.T) if beta is None else beta

    # ----------------------------------------------------------------------
    # PRETTY SUMMARY PRINT
    # ----------------------------------------------------------------------
    def print_summary(self):
        """
        Pretty-print ordered summary of all FOGAS parameters.
        Shows theoretical values and overridden values (if provided).
        """

        print("\n================ FOGAS PARAMETER SUMMARY ================\n")

        # Helper for formatting overridden values
        def fmt(name, theoretical_value, formatter="{:.6f}"):
            override = self.overrides.get(name)
            if override is None:
                # No override → theoretical value only
                return formatter.format(theoretical_value)
            else:
                # Overridden → display both
                theo = formatter.format(theoretical_value)
                new = formatter.format(override)
                return f"{theo}   (overridden → {new})"

        # ------------------------------------------------------
        # BASIC INFORMATION
        # ------------------------------------------------------
        print("Basic Information")
        print("-----------------")
        print(f"{'Dataset size n:':25s} {self.n}")
        print(f"{'Feature norm bound R:':25s} {self.R:.4f}")
        print(f"{'Num states N:':25s} {self.N}")
        print(f"{'Num actions A:':25s} {self.A}")
        print(f"{'Feature dim d:':25s} {self.d}")
        print(f"{'Discount γ:':25s} {self.gamma}")
        print(f"{'Confidence δ:':25s} {self.delta}")
        print("")

        # ------------------------------------------------------
        # THEORETICAL QUANTITIES
        # ------------------------------------------------------
        print("Theoretical Quantities")
        print("----------------------")
        print(f"{'T_min (theoretical):':25s} {self.T_min}")
        print(f"{'T (iterations):':25s}      {fmt('T', self.T, '{:.0f}')}")
        print("")

        # ------------------------------------------------------
        # HYPERPARAMETERS
        # ------------------------------------------------------
        print("FOGAS Hyperparameters")
        print("---------------------")
        print(f"{'alpha:':25s}     {fmt('alpha', self.alpha)}")
        print(f"{'rho:':25s}       {fmt('rho', self.rho)}")
        print(f"{'eta:':25s}       {fmt('eta', self.eta)}")
        print(f"{'D_theta:':25s}   {fmt('D_theta', self.D_theta)}")
        print(f"{'beta (ridge):':25s} {fmt('beta', self.R**2 / (self.d * self.T))}")
        print(f"{'D_pi (derived):':25s} {self.D_pi:.6f}")

        print("\n=========================================================\n")
