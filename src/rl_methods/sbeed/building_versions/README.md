# Building versions

This folder keeps the SBEED solvers used during the implementation study. They
are intentionally preserved as separate files so a reader can compare how each
stage changes the algorithm without needing to reconstruct the history from the
final solver.

## Solver stages

- `sbeed_solver.py`: stage 1. A one-step linear prototype with explicit value,
  rho, and softmax policy tensors. The rho weights are fitted by closed-form
  ridge regression to the one-step smoothed Bellman target.
- `sbeed_solver_sgd_rho.py`: stage 2. Adds terminal-aware replay rows and
  replaces the closed-form rho solve with stochastic gradient updates.
- `sbeed_optimizers.py`: stage 3. Keeps the one-step terminal-aware objective
  but compares optimizer choices for value, rho, and policy, including natural
  policy-gradient variants.
- `multi_linear_sbeed.py`: stage 4. Replaces isolated one-step transitions with
  terminal-safe multi-step fragments and discounted state-action feature sums.
- `multi_parametrized_sbeed.py`: stage 5. Removes the experimental optimizer
  branches and keeps the selected update order before the code is modularized
  into the final discrete solver.

The staged solvers are mainly for analysis, reproduction, and comparison. For
new experiments, prefer `src/rl_methods/sbeed/solvers`.

