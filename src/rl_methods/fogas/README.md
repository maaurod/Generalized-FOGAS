# FOGAS implementation

This folder contains the implementation of the original FOGAS method used in
the thesis experiments. FOGAS is treated here as one method among the methods in
`rl_methods`: it has its own solver, dataset adapters, parameter formulas, and
evaluation utilities, while shared environment and data-generation code lives in
`rl_methods.mdp` and `rl_methods.data_collection`.

The package is also reused by other method folders. Generalized FOGAS imports
the FOGAS dataset loaders and evaluator so that both methods read the same
offline data format and are compared with the same metrics. FQI also reuses the
discrete FOGAS dataset adapter for shared dataset-grid comparisons.

## Dependency structure

```text
rl_methods.fogas
|
|-- FOGASSolver
|   |-- loads transitions with FOGASDataset
|   |-- computes constants with FOGASParameters
|   |-- uses mdp + phi from rl_methods.mdp
|   |-- estimates or reads reward weights omega
|   `-- outputs stochastic policy pi and lambda/theta diagnostics
|
|-- FOGASEvaluator
|   |-- consumes a trained solver policy
|   |-- optionally uses Planner for exact value comparisons
|   `-- reports returns, success rates, value gaps, trajectories, reward fit
|
|-- FOGASHyperOptimizer
|   |-- repeatedly calls solver.run(...)
|   |-- evaluates candidates through FOGASEvaluator metrics
|   `-- supports grid, random, and Gaussian-process guided searches
|
|-- FOGASOracleSolver
|   |-- uses Planner instead of sampled transitions
|   |-- replaces empirical feature occupancies with exact occupancies
|   `-- serves as an oracle/debug implementation for small tabular MDPs
|
|-- FOGASDataset
|   |-- reads CSV columns: state, action, reward, next_state
|   |-- used by FOGASSolver, generalized FOGAS discrete solvers, and FQI
|   `-- exposes tensors X, A, R, X_next
|
`-- ContinuousFOGASDataset
    |-- reads observation/action/reward/next-observation/done CSV data
    |-- follows the FOGAS tensor convention for continuous observations
    `-- used by generalized continuous FOGAS Mountain Car experiments
```

## Main components

- `FOGASSolver`: main empirical FOGAS implementation for finite state and
  action spaces. It receives an MDP object, a feature map `phi`, and a CSV
  dataset. At initialization it builds the full feature tensor, extracts the
  sampled feature matrix, computes the empirical covariance, resolves reward
  weights `omega`, and stores the theoretical/default hyperparameters. Calling
  `run()` performs the FOGAS primal-dual update loop and returns the learned
  stochastic policy matrix.
- `FOGASDataset`: algorithm-specific discrete dataset adapter. It is separate
  from `data_collection` because it is not responsible for generating data; it
  only validates a saved FOGAS-format CSV and exposes tensors in the exact form
  expected by the FOGAS solver family.
- `ContinuousFOGASDataset`: continuous-observation dataset adapter. It is kept
  in this package because it follows the FOGAS data convention, but it is used
  by the continuous generalized FOGAS solver rather than by the tabular
  `FOGASSolver`. In the experiments it supports Mountain Car datasets with
  columns such as `obs_0`, `obs_1`, `next_obs_0`, `next_obs_1`, `action`,
  `reward`, and optional `done`.
- `FOGASParameters`: computes the theoretical constants used by FOGAS from the
  dataset size, feature bound, feature dimension, action count, discount factor,
  and confidence level. Experiments can override `T`, `alpha`, `eta`, `rho`,
  `D_theta`, or `beta` while still keeping the theoretical values available for
  comparison.
- `FOGASEvaluator`: shared policy-evaluation utility. It evaluates the raw
  stochastic solver policy or its greedy version using simulated returns,
  success rates, empirical-data value quality, optimal-state value quality,
  trajectory displays, and reward-approximation diagnostics.
- `FOGASHyperOptimizer`: experiment helper for tuning FOGAS hyperparameters.
  It delegates all policy-quality measurement to `FOGASEvaluator`, so the
  optimizer only manages candidate generation, repeated runs, result history,
  optional plotting, and optional CSV export.
- `FOGASOracleSolver`: exact-reference variant for small MDPs. It uses a
  `Planner` to obtain true discounted occupancies and is useful for verifying
  algorithmic behavior independently from finite-sample dataset effects.

## Typical workflow

The standard tabular workflow is:

```python
from rl_methods.fogas import FOGASEvaluator, FOGASSolver
from rl_methods.mdp import DiscreteMDP, Planner

mdp = DiscreteMDP(states, actions, gamma, x0, r=r, P=P)
planner = Planner(mdp)

solver = FOGASSolver(mdp=mdp, phi=phi, csv_path="dataset.csv", beta=1e-7)
policy = solver.run(alpha=0.005, eta=0.002, rho=0.0005, T=20_000)

evaluator = FOGASEvaluator(solver=solver, mdp=mdp, planner=planner)
greedy_return = evaluator.average_return(
    policy_mode="greedy",
    num_trajectories=100,
    max_steps=50,
)["policy"]
```

In the experiment scripts this pattern appears in:

- `experiments/fogas/scripts/grid_10_tabular.py`: tabular 10x10
  hyperparameter grid search.
- `experiments/fogas/scripts/grid_search_10grid_fogas.py`: dataset-variation
  grid search for the 10x10 grid.
- `experiments/fogas/scripts/grid_search_fogas_40grid.py` and
  `grid_search_dataset_40grid.py`: larger grid studies with RBF-style features.
- `experiments/fogas/scripts/grid_mountaincar.py`: discretized Mountain Car
  comparison using `FeaturesMDP`, `GymDataBuffer`, FOGAS, FQI, and Q-learning.
- `experiments/fogas_generalization/scripts/...`: generalized FOGAS studies
  that reuse `FOGASDataset`, `ContinuousFOGASDataset`, and `FOGASEvaluator` for
  consistent data loading and evaluation.

## Interface boundaries

FOGAS does not own MDP construction or dataset collection. Environment models
come from `rl_methods.mdp`, and offline datasets are generated or analyzed by
`rl_methods.data_collection`. This folder starts once a feature map and an
offline transition dataset already exist. That separation keeps the method
implementation focused on optimization and evaluation, while making dataset
quality and environment representation reusable across the thesis codebase.
