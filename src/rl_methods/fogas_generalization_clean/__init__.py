"""Generalized FOGAS exports."""

from .solvers import (
    BetaSolver,
    FinalLinearSolver,
    LinearBetaPiSolver,
    LinearSolver,
    LossThetaBetaPiSolver,
    LinearPolicyFOGAS,
    RegularizedLossThetaBetaPiSolver,
    VBetaLogitSolver,
    VBetaObjectivePolicySolver,
    VBetaSolver,
)
from .fogas_parameters import GeneralizedFOGASParameters, StandaloneFOGASParameters
from .features import (
    FeatureFunction,
    LinearFunction,
    LinearQFunction,
    LinearUFunction,
    TabularFeatures,
    TabularPolicyFeatures,
    build_feature_table,
    build_policy_feature_table,
    build_q_feature_table,
    build_u_feature_table,
)
from .u_functions import UFunction

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
    "LinearSolver",
    "FinalLinearSolver",
    "LinearBetaPiSolver",
    "LossThetaBetaPiSolver",
    "RegularizedLossThetaBetaPiSolver",
    "GeneralizedFOGASParameters",
    "StandaloneFOGASParameters",
    "TabularPolicyFeatures",
    "TabularFeatures",
    "UFunction",
    "FeatureFunction",
    "LinearFunction",
    "LinearUFunction",
    "LinearQFunction",
    "build_feature_table",
    "build_u_feature_table",
    "build_q_feature_table",
    "build_policy_feature_table",
]

if FOGASSolverPolicy is not None:
    __all__.append("FOGASSolverPolicy")
