# Offline dataset collection and diagnostics

This folder contains the shared dataset utilities used by the FOGAS and
generalized FOGAS experiments. The central format is the offline transition CSV
consumed by FOGAS-style solvers:

```text
state,action,reward,next_state
```

Additional metadata may be returned in memory for diagnostics, but saved FOGAS
datasets keep this four-column interface so the algorithms can read data from a
common source.

## Role in the experiments

The finite gridworld experiments use `DiscreteDataBuffer` to simulate
transitions directly from a known `DiscreteMDP`. The resulting datasets are
then analyzed with `DatasetAnalyzer` before being passed to FOGAS, generalized
FOGAS, or baseline methods. Continuous-control experiments, such as Mountain
Car, use `GymDataBuffer` together with the discretizers from `rl_methods.mdp`
to collect finite transition datasets from Gymnasium rollouts.

## Main components

- `DatasetAnalyzer`: reads a FOGAS-style CSV or a pandas DataFrame and reports
  state frequencies, action frequencies, state-action pair frequencies, missing
  pairs, reward statistics, and compact summaries. When given a feature table
  `phi` and a target occupancy, it also computes the feature coverage ratio
  used in the experiments. The target occupancy can be passed directly through
  `occupancy`, through `optimal_occupancy`, or through a solver/planner object
  exposing `mu_star`.
- `DiscreteDataBuffer`: finite-MDP simulator for offline datasets. It accepts a
  `DiscreteMDP`-style object exposing `N`, `A`, `x0`, `P`, and `r`, then samples
  one-step transitions under random/uniform policies, explicit policy matrices,
  custom policy objects, epsilon-greedy wrappers, or mixtures of policies.
  Policy mixtures can be selected per episode or per transition.
- `GymDataBuffer`: collector for Gymnasium environments after discretizing
  continuous observations and actions. It uses a policy matrix over discrete
  state and action ids, mixes policy-driven and random behavior, supports
  custom reset distributions, and can add goal-state self-loops for absorbing
  terminal states.

## Dataset-generation options

`DiscreteDataBuffer` supports several reset modes used to control dataset
coverage:

- `x0`: reset to the MDP initial state.
- `random`: reset uniformly over non-terminal, non-restricted states.
- `custom`: reset from a provided list of initial states.
- `restricted`: reset from states excluded from normal random starts, useful
  when forcing coverage of difficult regions.
- `occupancy`: reset according to an explicit state or state-action occupancy.
- `occupancy_uniform`: reset uniformly over states with positive occupancy.

It also provides `collect_uniform`, which samples every state-action pair a
fixed number of times, and `collect_macro_dataset_n_repeated_actions`, which
builds coarse/fine macro transitions by repeating the same fine action and
aggregating discounted rewards. These utilities are useful for gridworld
coverage studies and coarse-state experiments.

`GymDataBuffer` provides the analogous collection path when the environment is
not represented by an explicit transition matrix. It resets a Gymnasium
environment, maps observations to discrete state ids, maps discrete actions to
environment actions, and saves the resulting transitions in the same FOGAS CSV
format.

## Usage pattern

For finite MDPs:

```python
from rl_methods.data_collection import DatasetAnalyzer, DiscreteDataBuffer

collector = DiscreteDataBuffer(mdp, reset_probs={"x0": 0.5, "random": 0.5})
dataset = collector.collect(policy=planner.pi_star, n_steps=50_000)

analyzer = DatasetAnalyzer(dataset)
summary = analyzer.summary(n_states=mdp.N, n_actions=mdp.A)
coverage = analyzer.feature_coverage(phi=Phi, occupancy=planner.mu_star)
```

For discretized Gymnasium environments:

```python
from rl_methods.data_collection import GymDataBuffer

dataset = GymDataBuffer.collect(
    policy_matrix=policy_matrix,
    state_disc=state_discretizer,
    action_disc=action_discretizer,
    env_id="MountainCar-v0",
    start_obs=start_obs,
)
```

Together, these utilities make dataset construction reproducible and keep the
same data interface across model-based gridworlds and simulated continuous
control tasks.
