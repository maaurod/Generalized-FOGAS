"""Compatibility imports for residual-weighting feature functions.

``UFunction`` describes the finite feature-map protocol; ``LinearUFunction``
adapts a feature map for the linear ablation solver.
"""

from .features import FeatureFunction as UFunction
from .features import LinearFunction as LinearUFunction
from .features import build_u_feature_table

__all__ = ["UFunction", "LinearUFunction", "build_u_feature_table"]
