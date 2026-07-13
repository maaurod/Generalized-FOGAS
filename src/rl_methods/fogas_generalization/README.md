# Generalized FOGAS

This package contains the main algorithm developed in the thesis. Generalized
FOGAS starts from the primal linear-programming formulation of discounted
reinforcement learning and preserves the primal--dual optimization structure of
FOGAS, while replacing its explicit feature-occupancy variable by a learnable
residual-weighting function

```text
u_beta(x, a) ~= mu(x, a) / mu^D(x, a).
```

Here, `mu` is the occupancy measure represented by the dual variable and
`mu^D` is the state--action distribution of the offline dataset. This change of
variables makes the objective estimable from fixed transitions and allows the
three optimization variables to use linear features or neural networks:

- `u_beta(x, a)`: residual-weighting function, interpreted as an approximation
  of the occupancy ratio;
- `Q_theta(x, a)`: action-value function; and
- `pi_psi(a | x)`: policy to be optimized.

The package contains two reference Generalized FOGAS solvers, a linear
ablation solver, the primal AlgaeDICE baseline used in the thesis, and the
feature and parametrization utilities required to construct them.

## Objective and updates

For an offline dataset
`D_n = {(X_i, A_i, R_i, X'_i)}_{i=1}^n`, Generalized FOGAS optimizes the
empirical saddle-point objective

```text
max_(psi, beta) min_theta L_hat(psi, theta, beta),

L_hat = (1 - gamma) E[Q_theta(x_0, a_0)]
        + (1/n) sum_i u_beta(X_i, A_i)
          [R_i + gamma V_(theta, psi)(X'_i) - Q_theta(X_i, A_i)],
```

where `a_0 ~ pi_psi(. | x_0)` and
`V_(theta, psi)(x) = E_{a ~ pi_psi(. | x)}[Q_theta(x, a)]`.
Every outer iteration applies the three updates described in the thesis:

1. The value parameter `theta` takes an approximate regularized best response,
   normally several inner Adam steps with a warm start.
2. The policy parameter `psi` takes one ascent step on the policy-dependent
   part of the objective. Finite action spaces permit an exact action
   expectation; sampled REINFORCE estimates are available when enumeration is
   expensive and are required for continuous actions.
3. The occupancy parameter `beta` takes the stabilized, preconditioned ascent
   step

   ```text
   beta_(t+1) = [beta_t + eta G_t^(-1) g_(beta,t)] / (1 + rho eta),
   ```

   with local geometry

   ```text
   G_t = epsilon I
         + (1/n) sum_i grad_beta u_beta(X_i, A_i)
                       grad_beta u_beta(X_i, A_i)^T.
   ```

For linear `u_beta`, `G_t` is the ridge-regularized empirical feature
covariance matrix. Under the reparametrization
`lambda = Lambda_n beta`, the empirical objective and occupancy update recover
the corresponding FOGAS structure. For nonlinear `u_beta`, the outer product
of its sample gradients supplies a first-order local approximation of the
mirror geometry. The experiments normally use the diagonal of `G_t`, which
retains useful coordinate-wise scaling without constructing or inverting a
dense parameter-by-parameter matrix.

## Which solver should be used?

| Class | Intended role | State/action setting | Function approximation |
| --- | --- | --- | --- |
| `FinalParametrizedSolver` | Reference discrete Generalized FOGAS implementation | Finite states and finite actions | Linear, RBF, or neural parametrizations |
| `ContinuousFinalParametrizedSolver` | Reference continuous-observation implementation | Vector observations; discrete or continuous actions | Linear RBF or neural parametrizations |
| `FinalLinearSolver` | Efficient ablation workbench | Finite states and finite actions | Linear features only |
| `PrimalAlgaeDICESolver` | AlgaeDICE comparison baseline | Finite states and finite actions | Linear value and policy features |

The source layout follows these roles:

```text
fogas_generalization/
|-- README.md                         algorithm and usage guide
|-- solvers/
|   |-- final_parametrized_solver.py  reference finite-state solver
|   |-- continuous_parametrized_solver.py
|   |                                    reference continuous-observation solver
|   |-- final_linear_solver.py        linear ablation workbench
|   `-- primal_algaedice_solver.py    linear AlgaeDICE baseline
|-- features.py                       finite feature maps and PyTorch wrappers
|-- continuous_features.py            continuous RBF/neural parametrizations
|-- fogas_parameters.py               formula-based defaults and overrides
|-- policy_features.py                compatibility import path
`-- u_functions.py                    compatibility import path
```

### `FinalParametrizedSolver`: reference discrete solver

This is the main class for problems whose states and actions can be enumerated.
It accepts independent PyTorch parametrizations for `u_beta`, `Q_theta`, and
`pi_psi`, so the three variables do not need to share a feature map or a
parameter dimension. Finite action enumeration makes the exact policy
expectation available. Linear wrappers automatically use precomputed feature
tables and closed-form tensor operations; neural wrappers use autograd for the
value, policy, and occupancy gradients.

The solver retains a small set of alternative update options for consistent
experimentation, but the thesis configuration is obtained explicitly with a
regularized, warm-started Adam best response for `theta`, Adam for the policy,
and the diagonal Generalized FOGAS update for `beta`.

### `ContinuousFinalParametrizedSolver`: reference continuous solver

This solver evaluates vector observations directly instead of discretizing
them into state identifiers. This changes the implementation in four ways:

- a complete state--action feature table cannot be precomputed;
- values and policy probabilities are evaluated only at the initial and
  sampled next observations;
- `G_t` is accumulated from per-transition features or Jacobians, with
  `batch_size`, `u_jacobian_batch_size`, and `value_batch_size` available to
  control memory use; and
- discrete actions can still be enumerated exactly, whereas continuous-action
  expectations and policy gradients must be estimated from sampled actions.

The class supports the continuous-state/discrete-action setting used for
`MountainCar-v0`, as well as Gaussian policies for continuous actions. It also
uses the dataset `done` flag to remove bootstrapping after terminal
transitions.

### `FinalLinearSolver`: ablation solver

`FinalLinearSolver` is the final linear implementation used for the thesis
ablations. It deliberately collects the alternative update rules in one class:

- value update: adaptive or fixed quadratic regularization, or projection;
- value optimizer and initialization: SGD or Adam, with zero or warm start;
- policy optimizer: SGD, Adam, or natural policy gradient (NPG);
- policy expectation: exact enumeration or a sampled REINFORCE estimate; and
- occupancy update: full or diagonal Generalized FOGAS preconditioning,
  unpreconditioned and unstabilized variants, projected gradient, and
  quadratically regularized best-response variants.

The exact occupancy-update names accepted by the implementation are:

| `beta_update` | Ablation represented |
| --- | --- |
| `fogas_full` | Complete stabilized update with the full covariance preconditioner |
| `fogas_diag` | Stabilized update using only the covariance diagonal |
| `metric_no_stabilization` | Full preconditioner with `rho = 0` |
| `euclidean_stabilized` | Stabilization with the identity in place of the preconditioner |
| `projected_gradient` | First-order occupancy ascent with optional Euclidean projection through `beta_projection_radius` |
| `fenchel_br` | Best response for the quadratically regularized objective |
| `fenchel_mirror` | One interpolation step toward that regularized best response |

Because every variable is linear and the ablations use finite tabular grids,
the solver precomputes the complete feature tensors and performs the update
sweeps with efficient matrix operations. It is therefore the appropriate
class for controlled ablations, but it is not the primary interface for new
nonlinear experiments.

### `PrimalAlgaeDICESolver`: comparison baseline

This is the linear-feature primal AlgaeDICE baseline following the quadratic
primal formulation described in the appendix of the
[AlgaeDICE paper](https://arxiv.org/abs/1912.02074). With tabular one-hot value
features, its default `critic_update="closed_form"` computes the regularized
linear best response used for the grid experiments. The optional
`critic_update="batch_adam"` provides an iterative alternative. The actor is a
softmax-linear policy updated with Adam. This class is a baseline, not a
Generalized FOGAS variant: it does not use the Generalized FOGAS
occupancy-parameter update.

## Discrete example

The following example constructs neural `u`, `Q`, and policy parametrizations
for a finite grid. `state_inputs` gives one descriptor per state; grid
coordinates are preferable to one-hot state identifiers when the network
should exploit spatial structure. The shown update values correspond to the
selected deterministic 5 x 5 neural configuration stored with the experiment
results.

```python
import torch

from rl_methods.fogas_generalization import (
    FinalParametrizedSolver,
    NeuralPolicyParam,
    NeuralQParam,
    NeuralUParam,
    StateActionMLPModule,
    StateMLPPolicyModule,
)

n_states = 25
n_actions = 4
gamma = 0.9
x0 = 0

# One normalized (row, column) descriptor for every state in a 5 x 5 grid.
state_inputs = torch.tensor(
    [[row / 4.0, col / 4.0] for row in range(5) for col in range(5)],
    dtype=torch.float64,
)

u_param = NeuralUParam(
    StateActionMLPModule(
        n_states=n_states,
        n_actions=n_actions,
        state_inputs=state_inputs,
        hidden_sizes=(8,),
        dtype=torch.float64,
    )
)
q_param = NeuralQParam(
    StateActionMLPModule(
        n_states=n_states,
        n_actions=n_actions,
        state_inputs=state_inputs,
        hidden_sizes=(8,),
        dtype=torch.float64,
    )
)
policy_param = NeuralPolicyParam(
    StateMLPPolicyModule(
        n_states=n_states,
        n_actions=n_actions,
        state_inputs=state_inputs,
        hidden_sizes=(8,),
        dtype=torch.float64,
    )
)

solver = FinalParametrizedSolver(
    n_states=n_states,
    n_actions=n_actions,
    gamma=gamma,
    x0=x0,
    csv_path="data/datasets/generalization/5grid.csv",
    u_param=u_param,
    q_param=q_param,
    policy_param=policy_param,
    theta_mode="reg_fixed",
    theta_lambda=1e-3,
    theta_optimizer="adam",
    theta_inner_steps=3,
    theta_lr=3e-3,
    theta_start_mode="warm",
    beta_update="fogas_diag",
    beta_reg=1e-4,
    seed=123,
)

# For a new problem, select these optimization parameters on validation seeds
# or reproduce the corresponding experiment grid.
policy = solver.run(
    T=1_000,
    alpha=1e-3,
    eta=3e-5,
    rho=0.01,
    policy_optimizer="adam",
    policy_gradient="exact",
    tqdm_print=True,
)

action_at_x0 = int(torch.argmax(policy[x0]).item())
diagnostics = solver.get_diagnostics()
```

`policy` has shape `(n_states, n_actions)`. It can be evaluated with the shared
`FOGASEvaluator` when a compatible finite MDP is available, which keeps the
metrics aligned with the original FOGAS experiments.

For a linear experiment, the same solver can be constructed with
`LinearUParam`, `LinearQParam`, and `SoftmaxLinearPolicyParam`. Passing
`u_function`, `q_function`, and `policy_features` is also supported as a
convenience; the solver wraps them in these linear parameter classes.

## Continuous-observation example

This example is the selected neural configuration for the
`MountainCar-v0` study: observations are continuous vectors, but the three
environment actions are discrete.

```python
import torch

from rl_methods.fogas_generalization import (
    ContinuousDiscretePolicyParam,
    ContinuousFinalParametrizedSolver,
    ContinuousNeuralQParam,
    ContinuousNeuralUParam,
    ContinuousStateActionMLPModule,
    ContinuousStateMLPPolicyModule,
)

obs_dim = 2
n_actions = 3

u_param = ContinuousNeuralUParam(
    ContinuousStateActionMLPModule(
        obs_dim=obs_dim,
        action_dim=1,
        hidden_sizes=(16, 16),
        dtype=torch.float64,
    )
)
q_param = ContinuousNeuralQParam(
    ContinuousStateActionMLPModule(
        obs_dim=obs_dim,
        action_dim=1,
        hidden_sizes=(16, 16),
        dtype=torch.float64,
    )
)
policy_param = ContinuousDiscretePolicyParam(
    ContinuousStateMLPPolicyModule(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_sizes=(16, 16),
        dtype=torch.float64,
    )
)

solver = ContinuousFinalParametrizedSolver(
    obs_dim=obs_dim,
    action_type="discrete",
    n_actions=n_actions,
    gamma=0.9,
    x0_obs=[-0.5, 0.0],
    csv_path=(
        "data/results/generalization/mountain_car/"
        "mountaincar_data_obs_columns.csv"
    ),
    u_param=u_param,
    q_param=q_param,
    policy_param=policy_param,
    theta_mode="reg_fixed",
    theta_lambda=1e-7,
    theta_optimizer="adam",
    theta_inner_steps=10,
    theta_lr=3e-4,
    theta_start_mode="warm",
    beta_update="fogas_diag",
    beta_reg=None,
    seed=44,
)

learned_policy = solver.run(
    T=20_000,
    alpha=3e-4,
    eta=1e-6,
    rho=0.1,
    policy_optimizer="adam",
    policy_gradient="exact",  # exact sum over the three discrete actions
    tqdm_print=True,
)

stochastic_action = solver.sample_action([-0.5, 0.0])
greedy_action = solver.sample_action([-0.5, 0.0], deterministic=True)
```

For continuous actions, use `action_type="continuous"`, provide `action_dim`,
wrap a `ContinuousGaussianPolicyModule` with
`ContinuousGaussianPolicyParam`, and run with
`policy_gradient="reinforce"`. `action_samples_per_obs` controls the Monte
Carlo approximation of value expectations, while `reinforce_samples` controls
the policy-gradient estimate.

## Data formats and shared packages

Generalized FOGAS is an offline method: solver training never requests new
environment transitions. Dataset generation belongs to
`rl_methods.data_collection`, and MDP definitions, exact planning, feature MDPs,
and discretizers belong to `rl_methods.mdp`.

The finite solvers reuse `rl_methods.fogas.FOGASDataset`, whose CSV schema is

```text
state, action, reward, next_state
```

State and action columns must contain zero-based integer identifiers within the
declared ranges. For terminal grid states, datasets and MDPs normally use the
same absorbing-state convention as the other discrete thesis experiments.

The continuous-observation solver reuses
`rl_methods.fogas.ContinuousFOGASDataset`. A two-dimensional observation with
discrete actions has the schema

```text
obs_0, obs_1, action, reward, next_obs_0, next_obs_1, done
```

For continuous actions, replace `action` by contiguous columns
`action_0, action_1, ...`. The `done` column is optional and defaults to
`False`, but it should be included when terminal transitions must not
bootstrap.

The package boundary is therefore

```text
rl_methods.mdp              environment and exact-planning abstractions
rl_methods.data_collection  offline transition generation and analysis
rl_methods.fogas            shared dataset loaders and policy evaluator
            |                       |                       |
            `-----------------------+-----------------------'
                                    v
rl_methods.fogas_generalization  parametrizations and optimization
                                    |
                                    v
experiments/fogas_generalization  grids, ablations, and Mountain Car studies
```

## Feature and parametrization utilities

The helper classes are grouped by the type of solver that consumes them.

### Finite state and action spaces (`features.py`)

- `TabularFeatures` returns one-hot state--action features of dimension
  `n_states * n_actions`. It is the representation used in the tabular grid
  experiments and makes function-approximation error absent from update
  ablations.
- `RBFStateFeatures` maps each finite state identifier to radial-basis
  activations computed from supplied coordinates and centers.
  `RBFStateActionFeatures` places those state features in an action-specific
  block, producing `e_a`-coupled state--action features.
- `LinearFunction` is the lightweight feature-map adapter used by
  `FinalLinearSolver`. The aliases `LinearUFunction` and `LinearQFunction`
  clarify which optimization variable receives the map.
- `LinearUParam`, `LinearQParam`, and `SoftmaxLinearPolicyParam` are trainable
  PyTorch wrappers for the reference discrete solver. They mark themselves as
  linear fast paths, allowing the solver to precompute tables and avoid
  unnecessary autograd work.
- `StateActionMLPModule` maps finite state and action descriptors to a scalar
  used by `NeuralUParam` or `NeuralQParam`. `StateMLPPolicyModule` maps state
  descriptors to one logit per action and is wrapped by `NeuralPolicyParam`.
- `build_feature_table` and its `u`, `Q`, and policy-specific variants validate
  custom feature maps and materialize tensors of shape
  `(n_states, n_actions, feature_dim)`. They are useful when adding a new
  finite feature class because all linear solvers then receive a consistent
  layout.

The `u_function`, `q_function`, and policy feature maps may be different. This
is useful when the occupancy ratio, value function, and policy require
different representations; their parameter dimensions are not assumed equal.

### Continuous observations (`continuous_features.py`)

- `ContinuousRBFStateActionFeatures` evaluates RBFs at observation vectors and
  places them in action-specific blocks. The corresponding linear `u`, `Q`, and
  softmax-policy wrappers reproduce the linear RBF Mountain Car experiment.
- `ContinuousStateActionMLPModule` consumes concatenated observation and action
  vectors. It can represent both discrete actions (passed as one scalar action
  identifier in the current experiments) and genuine continuous action
  vectors.
- `ContinuousStateMLPPolicyModule` and `ContinuousDiscretePolicyParam` define a
  categorical policy over finite actions from continuous observations.
- `ContinuousGaussianPolicyModule` and `ContinuousGaussianPolicyParam` define
  a diagonal Gaussian policy for continuous actions.
- `ContinuousNeuralUParam` and `ContinuousNeuralQParam` adapt scalar neural
  modules to the interface expected by the continuous solver.

### Parameters and compatibility modules

`GeneralizedFOGASParameters` computes the FOGAS-motivated default constants
used by the finite solvers and records any overrides. The experimental scripts
set optimization parameters explicitly after grid search, so theoretical
defaults remain available without being confused with the selected empirical
configuration.

`policy_features.py` and `u_functions.py` are small compatibility modules that
preserve older import paths. New code can import the same objects directly from
`rl_methods.fogas_generalization`.

## Outputs, diagnostics, and experiments

The finite solvers return a policy matrix and store it in `solver.pi`.
`ContinuousFinalParametrizedSolver.run()` returns the learned policy module;
use `sample_action`, `policy_probs`, `q_values`, or `q` for evaluation. The
Generalized FOGAS solvers also expose:

- `solver.theta`: final value parameters;
- `solver.psi`: final policy parameters;
- `solver.beta_T`: final occupancy-function parameters;
- `solver.theta_bar_history` and `solver.psi_history`: optimization histories;
  and
- `solver.get_diagnostics()`: per-iteration objectives, gradient norms,
  update choices, and local-metric statistics.

Run experiment scripts from the repository root. The main entry-point groups
are:

```text
experiments/fogas_generalization/scripts/ablations/
    beta/       occupancy preconditioning and stabilization
    policy/     Adam, SGD, NPG, and sampled policy gradients
    theta/      value best response, initialization, and regularization

experiments/fogas_generalization/scripts/hyperparam_grids/
    5grid/      controlled tabular, RBF, and neural studies
    10grid/     partial-coverage comparisons and AlgaeDICE baseline

experiments/fogas_generalization/scripts/mountain_car/
    continuous-state RBF and neural Generalized FOGAS studies
```

These scripts are the reproducible source for the tuned parameters reported in
the thesis. New experiments should keep environment/data construction outside
the solver, construct the three parametrizations explicitly, and save both
policy metrics and solver diagnostics.
