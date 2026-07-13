# SBEED experiments

This folder contains the experiments used to analyze the SBEED implementation
as one part of the thesis codebase. The reusable implementation lives in
`src/rl_methods/sbeed`; this folder focuses on notebooks, grid searches, and
result inspection.

The experiments follow the same development path as the implementation. First,
the staged solvers are tested on small deterministic and stochastic gridworlds.
Then the final discrete solver is studied with tabular, RBF, and neural
parametrizations. Finally, the continuous solver is tested on Gymnasium
Pendulum with neural value/rho models and an RFF Gaussian policy.

## Folder overview

- `building_notebooks/`: notebooks for the staged SBEED solvers. Each stage
  repeats the same gridworld methodology while adding one implementation
  feature: naive one-step tabular/RBF checks, terminal-aware SGD rho, optimizer
  variants, and multi-step fragments. For implementation details, see
  `src/rl_methods/sbeed/building_versions`.
- `discrete/`: final discrete SBEED notebook for gridworld experiments. It uses
  the cleaned `DiscreteSBEED` solver with nonlinear features in the deterministic
  and stochastic grids.
- `continuous/`: final continuous SBEED notebook for Pendulum-v1. It loads and
  inspects the continuous solver configuration, evaluation returns, and training
  diagnostics.
- `scripts/`: grid-search and smoke-test scripts. These files run the larger
  hyperparameter studies for staged tabular/RBF solvers, final neural discrete
  solvers, offline tabular datasets, and the final continuous Pendulum solver.

## Experiment structure

The gridworld studies use deterministic and stochastic variants of small grids
to check whether each solver version learns sensible values and policies. The
tabular notebooks remove representation error and validate the solver update.
The linear/RBF notebooks study approximation with fixed features. The final
discrete experiments move to modular linear and neural parametrizations.

The continuous experiments use Pendulum-v1 as the main continuous-control test.
They use `ContinuousSBEED` with neural value/rho modules, a Gaussian policy, and
the same rho-value-policy update order as the final discrete solver.

Result-producing scripts write CSV files under `data/results/...` so searches
can be resumed, ranked, and inspected from notebooks.
