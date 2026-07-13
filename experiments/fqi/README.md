# FQI experiments

This folder contains the Fitted Q Iteration experiments used as a simple
baseline and diagnostic method in the thesis codebase. The reusable solver is
implemented in `src/rl_methods/fqi`.

The experiments study how FQI behaves on gridworlds under different dataset
collection strategies, feature choices, and coverage conditions.

## Folder overview

- `notebooks/`: exploratory FQI notebook for small gridworld checks. It builds
  tabular grid MDPs, loads offline datasets, trains `FQISolver`, and inspects
  the greedy policy and value quality.
- `scripts/`: larger grid-search scripts. They generate terminal-aware offline
  datasets, measure feature coverage, train FQI, and save comparison metrics
  such as convergence, final reward, and value/Q optimality gaps.
