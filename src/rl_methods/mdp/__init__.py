"""
Shared MDP representations used by the FOGAS experiment code.

The package exports complete finite models (`DiscreteMDP`), exact planning
utilities (`Planner`), feature-only MDP descriptions (`FeaturesMDP`), and the
continuous-to-discrete adapters used by Mountain Car style experiments. The
same imports are used by `experiments/fogas` and `experiments/fogas_generalization`.
"""

from .discrete_mdp import DiscreteMDP
from .features_mdp import FeaturesMDP, TabularFeatureMap
from .planner import Planner
from .continuous import ActionDiscretizer, StateDiscretizer

__all__ = [
    "DiscreteMDP",
    "FeaturesMDP",
    "TabularFeatureMap",
    "Planner",
    "StateDiscretizer",
    "ActionDiscretizer",
]
