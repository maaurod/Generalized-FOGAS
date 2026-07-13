"""
Continuous-to-discrete adapters for feature-based MDP experiments.

These exports convert Gymnasium observations and environment actions into the
finite state/action ids used by `FeaturesMDP`, policy matrices, and dataset
collection. They are mainly used in Mountain Car style FOGAS experiments.
"""

from .discretizers import ActionDiscretizer, StateDiscretizer

__all__ = [
    "StateDiscretizer",
    "ActionDiscretizer",
]
