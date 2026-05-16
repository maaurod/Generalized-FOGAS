"""
Historical SBEED solver variants kept for comparison and staged experiments.

Order of development:
    1. `SBEEDSolver`: one-step linear baseline with closed-form rho regression.
    2. `SBEEDSolverSGDRho`: terminal-aware replay and SGD rho updates.
    3. `SBEEDOptimizers`: optimizer experiments for value, rho, and policy.
    4. `MultiLinearSBEED`: terminal-safe multi-step linear targets.
    5. `MultiParametrizedSBEED`: cleaned linear scaffold before final modules.
"""

from .multi_linear_sbeed import MultiLinearSBEED
from .multi_parametrized_sbeed import MultiParametrizedSBEED
from .sbeed_optimizers import SBEEDOptimizers
from .sbeed_solver import SBEEDSolver
from .sbeed_solver_sgd_rho import SBEEDSolverSGDRho

__all__ = [
    "MultiLinearSBEED",
    "MultiParametrizedSBEED",
    "SBEEDOptimizers",
    "SBEEDSolver",
    "SBEEDSolverSGDRho",
]
