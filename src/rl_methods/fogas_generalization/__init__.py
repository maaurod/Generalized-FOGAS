"""Public API for the thesis Generalized FOGAS implementation."""

from .solvers import (
    FinalParametrizedSolver,
    ContinuousFinalParametrizedSolver,
    FinalLinearSolver,
    PrimalAlgaeDICESolver,
)
from .continuous_features import (
    ContinuousDiscretePolicyParam,
    ContinuousGaussianPolicyModule,
    ContinuousGaussianPolicyParam,
    ContinuousLinearRBFQParam,
    ContinuousLinearRBFUParam,
    ContinuousNeuralQParam,
    ContinuousNeuralUParam,
    ContinuousRBFStateActionFeatures,
    ContinuousSoftmaxLinearRBFPolicyParam,
    ContinuousStateActionMLPModule,
    ContinuousStateMLPPolicyModule,
)
from .fogas_parameters import GeneralizedFOGASParameters, StandaloneFOGASParameters
from .features import (
    FeatureFunction,
    LinearFunction,
    LinearQParam,
    LinearQFunction,
    LinearUParam,
    LinearUFunction,
    NeuralPolicyParam,
    NeuralQParam,
    NeuralUParam,
    PolicyParam,
    RBFStateActionFeatures,
    RBFStateFeatures,
    QParam,
    SoftmaxLinearPolicyParam,
    StateActionMLPModule,
    StateMLPPolicyModule,
    TabularFeatures,
    TabularPolicyFeatures,
    UParam,
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
    "FinalParametrizedSolver",
    "ContinuousFinalParametrizedSolver",
    "FinalLinearSolver",
    "PrimalAlgaeDICESolver",
    "GeneralizedFOGASParameters",
    "StandaloneFOGASParameters",
    "TabularPolicyFeatures",
    "TabularFeatures",
    "RBFStateFeatures",
    "RBFStateActionFeatures",
    "UFunction",
    "FeatureFunction",
    "UParam",
    "QParam",
    "PolicyParam",
    "LinearFunction",
    "LinearUFunction",
    "LinearQFunction",
    "LinearUParam",
    "LinearQParam",
    "SoftmaxLinearPolicyParam",
    "NeuralUParam",
    "NeuralQParam",
    "NeuralPolicyParam",
    "ContinuousStateActionMLPModule",
    "ContinuousStateMLPPolicyModule",
    "ContinuousGaussianPolicyModule",
    "ContinuousRBFStateActionFeatures",
    "ContinuousLinearRBFUParam",
    "ContinuousLinearRBFQParam",
    "ContinuousSoftmaxLinearRBFPolicyParam",
    "ContinuousNeuralUParam",
    "ContinuousNeuralQParam",
    "ContinuousDiscretePolicyParam",
    "ContinuousGaussianPolicyParam",
    "StateActionMLPModule",
    "StateMLPPolicyModule",
    "build_feature_table",
    "build_u_feature_table",
    "build_q_feature_table",
    "build_policy_feature_table",
]

if FOGASSolverPolicy is not None:
    __all__.append("FOGASSolverPolicy")
