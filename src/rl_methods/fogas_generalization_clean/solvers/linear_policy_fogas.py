"""Compatibility import for the numbered clean linear-policy FOGAS module."""

from importlib import import_module

LinearPolicyFOGAS = import_module(".4_linear_policy_fogas", __package__).LinearPolicyFOGAS

__all__ = ["LinearPolicyFOGAS"]
