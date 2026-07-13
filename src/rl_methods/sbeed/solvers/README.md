# Final SBEED solvers

This folder contains the cleaned SBEED implementations intended for new
experiments.

## Files

- `discrete_sbeed.py`: final finite-MDP solver. Use it for gridworlds or other
  environments with explicit integer state and action ids. It accepts linear or
  neural value/rho/policy parametrizations and uses terminal-safe multi-step
  replay fragments.
- `continuous_sbeed.py`: final continuous-control solver. Use it with
  Gymnasium-style environments where observations and actions are real-valued.
  It uses neural value/rho modules and a Gaussian policy.

Both solvers follow the same update structure: build a replay fragment batch,
fit rho to the multi-step target, update the value objective, then update the
policy with a Fisher/KL natural-gradient step. The discrete solver keeps fast
manual paths for linear parametrizations, while neural and continuous paths use
PyTorch autograd and Adam.

