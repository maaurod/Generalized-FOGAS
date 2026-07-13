# Reinforcement Learning Method Implementations

This package contains the reusable MDP, offline-data, solver, and evaluation
code used by the thesis experiments. The algorithm implementations are kept
separate from `experiments/`, where notebooks and scripts construct datasets,
select hyperparameters, evaluate policies, and produce the reported
comparisons.

Generalized FOGAS is the main method developed in the thesis. FOGAS is its
theoretical starting point; FQI is the standard baseline; SBEED and Fenchel
AlgaeDICE are the most directly related primal--dual comparisons; and
Q-learning is used as a supporting baseline for dataset construction in the
Mountain Car experiments.

## Package structure

```text
src/rl_methods/
|-- mdp/                    finite MDPs, planning, features, and discretization
|-- data_collection/        offline dataset generation and diagnostics
|-- fogas/                  original FOGAS solver and shared evaluation tools
|-- fogas_generalization/   Generalized FOGAS and Fenchel AlgaeDICE baseline
|-- fqi/                    Fitted Q-Iteration baseline
|-- q_learning/             tabular Q-learning supporting baseline
|-- sbeed/                  staged and final SBEED implementations
`-- __init__.py             top-level public imports
```

### Shared MDP and data components

- [`mdp`](mdp/README.md) contains `DiscreteMDP`, exact planning utilities,
  feature-only MDP descriptions, and state/action discretizers. Small finite
  experiments use exact planning to compute optimal policies, values, and
  occupancy measures; feature-only and continuous-state experiments use the
  lighter abstractions required by the offline algorithms.
- [`data_collection`](data_collection/README.md) contains the finite-MDP and
  Gymnasium data buffers and the dataset diagnostics. The shared finite offline
  transition schema is `state, action, reward, next_state`.

### Algorithm implementations

- [`fogas`](fogas/README.md) contains the empirical FOGAS solver, oracle solver,
  discrete and continuous dataset adapters, parameter formulas,
  hyperparameter search helper, and the policy evaluator reused across several
  experiments.
- [`fogas_generalization`](fogas_generalization/README.md) contains the main
  algorithm developed in the thesis. It provides the reference discrete and
  continuous Generalized FOGAS solvers, the linear ablation solver, linear/RBF
  and neural parametrizations, and the primal Fenchel AlgaeDICE baseline.
- [`fqi`](fqi/README.md) contains the dataset-based, model-based, and
  optimal-target Fitted Q-Iteration modes used as baseline and diagnostic
  comparisons.
- [`q_learning`](q_learning/README.md) contains the compact tabular Q-learning
  implementation used to construct a near-optimal behaviour policy for the
  Mountain Car offline datasets.
- [`sbeed`](sbeed/README.md) contains the staged SBEED implementations,
  replay-buffer datasets, feature and neural parametrizations, and the final
  discrete and continuous solvers.

## How the packages connect

```text
rl_methods.mdp              environment and exact-planning abstractions
rl_methods.data_collection  offline transition generation and diagnostics
             |                              |
             `--------------+---------------'
                            v
       FOGAS | Generalized FOGAS | FQI | SBEED
                            |
                            v
              shared or method-specific evaluators
                            |
                            v
                    experiments/ and data/
```

The method folders do not own the complete experimental protocol. Environment
definitions, behaviour-policy mixtures, fixed datasets, parameter grids, and
reported comparisons remain under `experiments/` and `data/`. See the
[`experiments` overview](../../experiments/README.md) to locate the thesis
experiments and the additional studies.

## Importing the package

Install the repository from its root in editable mode:

```bash
python -m pip install -e .
```

The distribution is named `rl-methods`, while Python code imports
`rl_methods`. The top-level package exposes the main classes lazily, so both
of the following styles are supported:

```python
from rl_methods import DiscreteMDP, FinalParametrizedSolver
from rl_methods.fogas_generalization import FinalParametrizedSolver
```

For the complete interface, mathematical role, and experiment references of a
component, follow the README in its subfolder rather than treating this file as
an API reference.
