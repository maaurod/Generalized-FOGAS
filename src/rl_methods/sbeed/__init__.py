"""
Public exports for the SBEED implementation.

The package exposes both the final solvers and the historical staged solvers.
Use `DiscreteSBEED` and `ContinuousSBEED` for new work. Use the classes in
`building_versions` when reproducing the thesis development stages or comparing
specific implementation choices.
"""

from .features import (
    ContinuousGaussianPolicyParam,
    ContinuousNeuralRhoParam,
    ContinuousNeuralValueParam,
    ContinuousRhoParam,
    ContinuousStateActionMLPModule,
    ContinuousStateMLPValueModule,
    ContinuousValueParam,
    LinearRhoParam,
    LinearValueParam,
    IdentityHead,
    NeuralPolicyParam,
    NeuralRhoParam,
    NeuralValueParam,
    PolicyParam,
    RBFStateActionFeatures,
    RBFStateFeatures,
    RhoParam,
    SoftmaxLinearPolicyParam,
    StateActionMLPModule,
    StateMLPPolicyModule,
    StateMLPValueModule,
    TabularStateActionFeatures,
    TabularStateFeatures,
    ValueParam,
    RFFGaussianPolicyParam,
)
from .datasets import ContinuousSBEEDDataset, DiscreteSBEEDDataset, SBEEDDataset
from .sbeed_evaluator import SBEEDEvaluator
from .sbeed_base import SBEEDSolverProtocol
from .building_versions import (
    MultiLinearSBEED,
    MultiParametrizedSBEED,
    SBEEDOptimizers,
    SBEEDSolver,
    SBEEDSolverSGDRho,
)
from .sbeed_spec import DiscreteMDP, DiscreteMDPSpec
from .solvers import ContinuousSBEED, DiscreteSBEED, SBEED

__all__ = [
    "SBEED",
    "DiscreteSBEED",
    "DiscreteSBEEDDataset",
    "ContinuousSBEED",
    "ContinuousSBEEDDataset",
    "SBEEDSolver",
    "SBEEDSolverSGDRho",
    "SBEEDOptimizers",
    "MultiLinearSBEED",
    "MultiParametrizedSBEED",
    "SBEEDSolverProtocol",
    "SBEEDEvaluator",
    "SBEEDDataset",
    "DiscreteMDPSpec",
    "DiscreteMDP",
    "ValueParam",
    "RhoParam",
    "PolicyParam",
    "ContinuousValueParam",
    "ContinuousRhoParam",
    "ContinuousGaussianPolicyParam",
    "LinearValueParam",
    "LinearRhoParam",
    "SoftmaxLinearPolicyParam",
    "RFFGaussianPolicyParam",
    "IdentityHead",
    "StateMLPValueModule",
    "StateActionMLPModule",
    "StateMLPPolicyModule",
    "ContinuousStateMLPValueModule",
    "ContinuousStateActionMLPModule",
    "NeuralValueParam",
    "NeuralRhoParam",
    "NeuralPolicyParam",
    "ContinuousNeuralValueParam",
    "ContinuousNeuralRhoParam",
    "RBFStateFeatures",
    "RBFStateActionFeatures",
    "TabularStateFeatures",
    "TabularStateActionFeatures",
]
