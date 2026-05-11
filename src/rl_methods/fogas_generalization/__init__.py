"""Generalized FOGAS exports."""

from .solvers import (
    BetaSolver,
    FOGASSolverBeta,
    FOGASSolverBetaObjectivePolicyVectorized,
    FOGASSolverBetaVectorized,
    LinearPolicyFOGAS,
    VBetaLogitSolver,
    VBetaObjectivePolicySolver,
    VBetaSolver,
)
from .policy_features import TabularPolicyFeatures

try:
    from .solver_policy import FOGASSolverPolicy
except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}.solver_policy":
        raise
    FOGASSolverPolicy = None

__all__ = [
    "BetaSolver",
    "VBetaSolver",
    "VBetaObjectivePolicySolver",
    "VBetaLogitSolver",
    "LinearPolicyFOGAS",
    "TabularPolicyFeatures",
    "FOGASSolverBeta",
    "FOGASSolverBetaVectorized",
    "FOGASSolverBetaObjectivePolicyVectorized",
]

if FOGASSolverPolicy is not None:
    __all__.append("FOGASSolverPolicy")
