# Q-learning implementation

This folder contains a compact tabular Q-learning baseline for Gymnasium-style
environments.

The solver is intentionally simple: it learns a table `Q[state, action]` from
environment interaction, using an epsilon-greedy behavior policy and a
user-provided observation-to-state abstraction when observations are not already
integer state ids.

## Files

- `q_learning_solver.py`: defines `QLearningSolver`, the `QLearningResult`
  dataclass, and the `run_q_learning` convenience wrapper used by experiment
  scripts.
- `__init__.py`: exports the public Q-learning API.

## How it is used

Use this baseline when the environment can be discretized into a finite number
of states and actions. The MountainCar experiments use this solver to build a
tabular baseline policy after discretizing the continuous observation space.

There is currently no large standalone Q-learning experiment suite in
`experiments/q_learning`; Q-learning is used as a supporting baseline from
broader experiment scripts.

