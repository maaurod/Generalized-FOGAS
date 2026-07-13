# FQI implementation

This folder contains the Fitted Q Iteration implementation used as a simple
baseline method in the thesis codebase.

FQI is an offline value-iteration style method. It repeatedly builds Bellman
targets, solves a regularized least-squares regression for the Q-function
weights, and extracts the greedy policy from the fitted Q-values.

## Files

- `fqi_solver.py`: main `FQISolver` implementation. It supports dataset-based
  FQI from CSV transitions, model-based backups when the full MDP transition
  matrix is available, and an oracle/optimal-target backup used to inspect the
  best Q-function representable by the chosen features.
- `__init__.py`: exports `FQISolver`.

## Solver modes

- Dataset-based mode uses replay samples `(state, action, reward, next_state)`
  from a CSV file. It is the standard offline FQI setting used in the FQI
  experiments.
- Model-based mode uses the full transition matrix and reward vector from the
  MDP to build targets for every state-action pair.
- Optimal-target mode uses the true optimal value function, when available, to
  study feature approximation independently from dataset coverage.

The implementation is mainly used for tabular and RBF gridworld comparisons.
For experiment notebooks and dataset-grid studies, see `experiments/fqi`.

