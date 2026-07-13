# Empirical Study of FOGAS

This folder contains the empirical analysis of FOGAS. The
notebooks are organized as a progressive study rather than as independent
demos: exact small MDPs validate the implementation, intermediate grids expose
coverage and representation effects, large grids test coarse-to-fine transfer,
and Mountain Car evaluates the same offline-learning ideas in a discretized
continuous-control environment.

The notebooks combine executable definitions with the figures and comparisons
needed to interpret the experiments. Expensive sweeps are run by
`../scripts/`, saved under `data/results/`, and loaded here for analysis.

## Research questions

The empirical study addresses the following questions.

1. **Implementation validity.** Does empirical FOGAS recover the behavior of
   its oracle counterpart on MDPs where optimal values and occupancies can be
   computed exactly?
2. **Feature dependence.** How does behavior change when tabular features are
   replaced by lower-dimensional or RBF representations?
3. **Offline coverage.** Which behavior-policy mixtures, reset distributions,
   and dataset sizes cover the feature directions required by an optimal
   policy?
4. **Baseline comparison.** Under the same dataset and feature map, where do
   FOGAS and linear FQI succeed or fail, and is a failure caused by the
   representation or by empirical bootstrapping?
5. **Scalability.** Can a policy learned with coarse features and macro
   transitions control a substantially finer environment?
6. **Continuous control.** Can discretization, RBF features, and a
   near-optimal behavior policy provide a usable offline dataset for Mountain
   Car?

## Study structure

```text
Implementation and oracle checks
  2State
      |
      v
  3grid ------> 3grid_wall
      \             /
       `-> hyperoptimizer_boxplots_summary
                     |
                     v
Coverage and representation limits
  10grid_tabular <---- matched datasets ----> FQI
          |
          v
  10grid_RBF <---- center count, partial coverage, FQI limits
          |
          v
Coarse-to-fine scalability
  large_20 (10 -> 20) -> large_40 (20 -> 40) -> large_100 (20 -> 100)

Continuous-control extension
  Q-learning behavior policy -> Mountain Car dataset
      -> discretized/RBF FOGAS and FQI evaluation
```

## Common experimental pipeline

```text
Environment definition
        |
        v
Finite MDP or feature-only abstraction + feature map phi(x, a)
        |
        +--> Planner / oracle quantities, when computationally feasible
        |
        v
Offline behavior policy and reset distribution
        |
        v
CSV dataset: state, action, reward, next_state
        |
        +--> DatasetAnalyzer: counts and feature coverage
        |
        +--> FOGASSolver
        `--> FQISolver, where used as a baseline
                  |
                  v
Return, success, policy quality, value gap, and trajectory analysis
```

For finite grids, `DiscreteMDP` stores the exact environment and `Planner`
computes the optimal policy, values, and occupancy. `DiscreteDataBuffer`
generates offline transitions from controlled mixtures of policies and reset
distributions. `FOGASSolver` learns from the resulting fixed dataset, while
`FOGASOracleSolver` replaces empirical quantities with exact model information
to separate algorithmic behavior from sampling error. `FOGASEvaluator`
provides a common evaluation interface for FOGAS and FQI policies.

For Mountain Car, `StateDiscretizer`, `ActionDiscretizer`, and `FeaturesMDP`
define the finite abstraction required by the solvers. `GymDataBuffer` keeps
interaction with the original Gymnasium dynamics, then records transitions in
the same discrete CSV schema used by the grid experiments.

## Evaluation quantities

The notebooks report complementary metrics because no single quantity fully
describes an offline policy.

- **Average discounted return** measures task performance under the evaluation
  start distribution.
- **Success rate** records whether a terminal goal is reached within the
  evaluation horizon.
- **Value gap** weights `V* - V^pi` by the state occupancy of the optimal
  policy, emphasizing states relevant to optimal behavior.
- **On-data policy quality** measures agreement or value quality on the support
  represented by the offline batch.
- **Feature coverage** measures how strongly the dataset covariance supports
  the feature occupancy of the comparator policy. In the notation used by the
  notebooks, the principal quantity is

  ```text
  ||lambda_pi*||^2_(Lambda_n^-1),
  Lambda_n = beta I + (1/n) sum_i phi_i phi_i^T.
  ```

  A smaller value indicates stronger alignment between the dataset and the
  feature directions required by the target policy. The regularizer `beta`
  stabilizes poorly covered directions but also introduces bias.

The evaluator often reports both the stochastic FOGAS policy and its greedy
version. These are distinct experimental objects: the former is the direct
solver output, while the latter tests the deterministic policy induced by its
largest action probability.

## Notebook map

### `2State.ipynb`: minimal correctness experiment

This notebook provides the smallest end-to-end validation. It defines a
two-state, two-action linear MDP and studies two feature maps: exact tabular
one-hot features and a compact non-tabular representation. For the tabular
case, it first tunes and evaluates `FOGASOracleSolver`, then creates a random
offline dataset and tests whether empirical `FOGASSolver` can reproduce the
oracle behavior. The oracle hyperparameters are transferred to the empirical
solver before a separate empirical search is performed. Repeating the study
with the compact feature map checks that the implementation is not restricted
to one-hot coordinates.

### `3grid.ipynb`: oracle-to-empirical validation and feature interaction

The 3 x 3 grid extends the validation to a control
problem with four actions. The first experiment uses tabular features and
compares exact planning, oracle FOGAS, and dataset-based FOGAS. The second
experiment replaces tabular features with state aggregation. It contrasts a
feature design that does not adequately couple state and action with a
row-action representation that restores this interaction. This isolates a
representation limitation: successful optimization cannot compensate for a
feature map that cannot distinguish the required policy decisions.

### `3grid_wall.ipynb`: obstacles, adverse terminals, and aggregation

This notebook adds a wall and an absorbing pit to the 3 x 3 problem. The
resulting policy must distinguish blocked transitions, a positive terminal,
and a negative terminal. As in `3grid.ipynb`, the notebook compares oracle and
empirical FOGAS under tabular features, then repeats the empirical study with a
structured state-action feature map. It is the final check before
moving to larger, incompletely covered datasets.

### `hyperoptimizer_boxplots_summary.ipynb`: small-problem sensitivity summary

This notebook reads the saved hyperparameter-search tables from `2State`,
`3grid`, and `3grid_wall`. For oracle, empirical tabular, and empirical
feature-approximation runs, it plots the distribution of the evaluation metric
against `alpha`, `eta`, `rho`, `eta * rho`, and `D_theta`. Its purpose is to
summarize optimization sensitivity across the validation problems without
rerunning the solvers.

### `10grid_tabular.ipynb`: dataset coverage and matched FOGAS-FQI comparison

The 10 x 10 four-room grid is the main tabular offline-data study. The notebook
constructs the exact MDP, collects a mixture of an epsilon-greedy optimal policy
and a random policy, computes feature coverage, and solves the task with FOGAS.
It studies the stabilizing effect and bias tradeoff of `beta` when many
state-action coordinates are weakly observed.

The batch scripts vary four dataset factors: number of transitions, epsilon,
optimal/random policy proportion, and initial-state distribution. The notebook
loads the resulting FOGAS table and relates these factors, and the measured
coverage ratio, to return, success, value gap, and on-data quality. It then
runs the same analysis for FQI and combines the matched tables. Because both
algorithms receive datasets generated from the same grid, the comparison
focuses on solver behavior rather than an accidental difference in data.

### `10grid_RBF.ipynb`: partial coverage and representation limits

This notebook replaces one-hot features with RBF features derived
from K-means centers. Two representations are examined: RBFs augmented with a
bias and pit/goal indicators, and a more general normalized K-means RBF map.
The center count controls the effective feature resolution.

The notebook sweeps the number of centers for FOGAS and FQI and studies the
feature-covariance geometry of the offline dataset. Unlike tabular features,
nearby RBF activations overlap; observations near an optimal trajectory can
therefore provide partial support for an unobserved state in the same feature
direction. This is the notebook's central coverage experiment.

The FQI analysis separates three possible failure sources. An optimal-target
regression asks whether the RBF map can represent the required backup; a
model-based iterative backup tests approximate dynamic programming without an
offline dataset; and empirical FQI finally introduces dataset support and
sampled bootstrapping. Comparing these stages with FOGAS identifies feature
resolutions where FOGAS remains effective while empirical FQI is limited, and
prevents a representation failure from being mislabeled as an offline-data
failure.

### `large_20.ipynb`: 10 x 10 features applied to a 20 x 20 grid

This notebook begins the scalability study. A 10 x 10 environment defines the
coarse RBF centers and feature bandwidth; the same normalized feature space is
evaluated on a 20 x 20 refinement. Two strategies are compared.

1. A policy is learned directly on a coarse 10 x 10 dataset collected from an
   occupancy-biased mixture of an epsilon-greedy optimal policy and a random
   policy.
2. Data are generated in the fine 20 x 20 environment and converted to 10 x 10
   macro transitions. Each macro action repeats one fine action twice, and the
   coarse reward and discount reproduce that temporal aggregation.

The learned coarse policies are lifted to the 20 x 20 environment and
evaluated with the same repeated-action semantics. State coverage, feature
coverage, trajectories, return, and success are compared against exact
planning where feasible.

### `large_40.ipynb`: 20 x 20 features applied to a 40 x 40 grid

This notebook repeats the coarse-to-fine protocol at the next scale. RBF
centers and bandwidth are defined on a 20 x 20 grid and reused on its 40 x 40
refinement. Direct coarse-data learning is compared with a macro dataset
collected in the fine grid, where each coarse action represents two repeated
fine actions. The final section maps both learned 20 x 20 policies to 40 x 40
states and evaluates their trajectories and success under the fine dynamics.

The experiment tests whether the abstraction continues to preserve useful
control decisions when the number of fine states grows and the offline state
distribution remains strongly non-uniform.

### `large_100.ipynb`: final coarse-to-fine test

The largest grid is a 100 x 100 refinement of a 20 x 20 environment, with a
scale factor of five. The coarse RBF centers are reused without retraining. A
direct 20 x 20 policy provides one reference, while the fine-data strategy
collects 100 x 100 trajectories and aggregates every five repeated actions
into a 20 x 20 macro transition.

To improve support at this scale, the macro dataset mixes a down/right
heuristic with random actions and resets from the original start or from valid
states on the first column and last row. FOGAS is trained on the resulting
coarse macro MDP, then its policy is expanded to all fine states and executed
for five fine steps per coarse decision. This is the final test of the
claim that a feature-level can avoid a fully tabular
10,000-state optimization problem while still controlling it.

### `mountainCar.ipynb`: discretized continuous-control experiment

Mountain Car studies FOGAS beyond an explicitly enumerated gridworld. Position
and velocity are discretized into a 20 x 20 state grid with an additional
absorbing goal state; the three Gymnasium actions remain discrete. A 15 x 15
RBF grid is replicated across action blocks to form the feature map used by
FOGAS and FQI.

The notebook first trains tabular Q-learning online with decaying exploration.
Its greedy policy is not the offline-learning result: it is used as a
near-optimal behavior reference for dataset construction. A successful
trajectory defines a custom reset distribution over useful phase-space bins.
`GymDataBuffer` then mixes this policy with random behavior, epsilon
exploration, and resets from the nominal start or trajectory distribution.

Two dataset conventions are inspected: one can wait for the discretized state
to change before storing a transition, while the other preserves every
environment step, including repeated abstract states. Coverage heatmaps show
both visit counts and the fraction of actions agreeing with the Q-learning
policy. The final empirical study trains FOGAS and FQI on the all-rows RBF
dataset and compares success and mean steps to the goal. The companion
`grid_mountaincar.py` script varies behavior-policy proportion, epsilon, and
reset mixture; the final notebook section plots its saved comparison table.

## Recommended reading order

For the thesis narrative, read the notebooks in this order:

```text
2State -> 3grid -> 3grid_wall -> hyperoptimizer summary
       -> 10grid_tabular -> 10grid_RBF
       -> large_20 -> large_40 -> large_100
       -> mountainCar
```

This order moves from verification to diagnosis and then to scale. The small
oracle experiments establish trust in the implementation; the 10 x 10 studies
explain data and representation limitations; the large grids test the proposed
abstraction strategy; and Mountain Car demonstrates how the same components
can be assembled when the original environment is continuous.

## Reproducibility notes

- Execute notebooks with the project environment and keep the repository root
  discoverable above the notebook working directory.
- Generate the corresponding batch CSVs or use the already generated ones in the repo
  in `data/datasets/`.
- Seeds are fixed inside each notebook, but GPU linear algebra and environment
  execution can still introduce small numerical differences.
- Some cells perform exhaustive searches or long rollouts. They are retained
  as part of the experimental record; for routine presentation, load their
  saved CSV outputs instead of rerunning them.
- The notebooks contain exact planners only where the state space makes that
  computation useful. The feature-only and coarse-to-fine experiments are the
  intended alternatives when full tabular planning is no longer the primary
  scalable formulation.
