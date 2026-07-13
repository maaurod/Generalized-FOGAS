# MDP models and planning utilities

This folder contains the shared finite-MDP infrastructure used by the FOGAS and
generalized FOGAS experiments. It is intentionally separate from the algorithm
folders: the same environment representation, feature maps, and exact planning
tools are reused to build datasets, compute oracle quantities, and evaluate
policies across several experimental settings.

## Role in the experiments

The tabular gridworld experiments usually start from a complete finite model:
states, actions, discount factor, reward, transition kernel, and initial state.
Those objects are represented by `DiscreteMDP`. Exact dynamic-programming
quantities are then computed by `Planner` when the state-action space is small
enough. These quantities are used as references for FOGAS, generalized FOGAS,
FQI comparisons, dataset-generation checks, and coverage diagnostics.

The continuous-control experiments, such as Mountain Car, cannot use an exact
finite transition matrix directly. They are first mapped to finite state and
action identifiers with the discretizers in `continuous/`, and then represented
with `FeaturesMDP` when only the feature-level information required by the
offline algorithms is available.

## Main components

- `DiscreteMDP`: complete finite MDP container. It stores the state set, action
  set, discount factor, initial state, reward vector `r`, transition matrix
  `P`, initial distribution `nu0`, and optional terminal states. The object can
  be constructed from explicit `r` and `P`, from callable reward and transition
  functions, or from linear-MDP components. In the linear case, `phi` maps
  state-action pairs to features, `omega` defines linear rewards, and `psi`
  defines the transition model through feature-based next-state probabilities.
- `Planner`: exact dynamic-programming layer built on top of `DiscreteMDP`. It
  computes policy evaluation, policy iteration, optimal policy/value/Q
  functions, discounted occupancy measures, normalized policy returns, optimal
  feature occupancies, and fitted linear weights for the optimal Q-function.
  This class is kept separate because exact value and policy iteration are
  useful for small tabular environments but can become computationally
  infeasible in large state-action spaces.
- `FeaturesMDP`: lightweight feature-only finite MDP description. It stores
  states, actions, discount factor, initial state, the feature map `phi`, the
  full feature table `Phi`, and optional reward weights `omega`, but it does
  not require a known reward vector or transition matrix. This is the correct
  representation when the experiment only assumes the information needed to run
  FOGAS or generalized FOGAS.
- `TabularFeatureMap`: one-hot feature map over finite state-action pairs. It
  is the default feature representation for tabular experiments.
- `StateDiscretizer` and `ActionDiscretizer`: helpers for converting
  continuous Gymnasium observations and environment actions into the finite ids
  used by `FeaturesMDP`, datasets, and policy matrices.

## Usage pattern

For small tabular experiments, construct a `DiscreteMDP`, then pass it to
`Planner` when exact reference quantities are needed:

```python
from rl_methods.mdp import DiscreteMDP, Planner

mdp = DiscreteMDP(states, actions, gamma, x0, r=r, P=P)
planner = Planner(mdp)

pi_star = planner.pi_star
mu_star = planner.mu_star
return_star = planner.optimal_policy_return()
```

For feature-only or discretized continuous experiments, construct a
`FeaturesMDP` with the available feature map:

```python
from rl_methods.mdp import FeaturesMDP

mdp = FeaturesMDP(states, actions, gamma, x0, phi=phi)
```

This split keeps the mathematical roles explicit: `DiscreteMDP` represents a
known finite model, `Planner` computes exact oracle quantities for models that
are small enough, and `FeaturesMDP` supports experiments where only features
and offline data are available.
