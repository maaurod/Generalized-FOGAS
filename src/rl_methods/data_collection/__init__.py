"""
Dataset collection and analysis utilities for FOGAS-style experiments.

The package exports finite-MDP collectors, Gymnasium collectors, and dataset
diagnostics for the common offline transition format
`state, action, reward, next_state`. These utilities are shared by the original
FOGAS experiments and the generalized FOGAS experiments.
"""

from .discrete_data_buffer import DiscreteDataBuffer
from .dataset_analyzer import DatasetAnalyzer
from .gym_data_buffer import GymDataBuffer

__all__ = [
    "DiscreteDataBuffer",
    "DatasetAnalyzer",
    "GymDataBuffer",
]
