"""Compatibility import for the numbered clean VBeta logit solver module."""

from importlib import import_module

VBetaLogitSolver = import_module(".4_vbeta_logit_solver", __package__).VBetaLogitSolver

__all__ = ["VBetaLogitSolver"]
