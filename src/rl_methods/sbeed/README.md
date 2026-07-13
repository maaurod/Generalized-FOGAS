# SBEED implementation and analysis

This package contains the SBEED implementation used as one component of the
thesis codebase. It is organized so the reader can follow both the development
path and the final cleaned solvers: first the staged implementations that extend
and test the appendix ideas step by step, then the reusable datasets, feature
parametrizations, and final solver classes used by the experiments.

SBEED is treated here as an implementation study. The code starts from small
finite gridworld checks, adds the practical details needed for terminal
episodes and replay buffers, studies optimizer choices, and then moves to the
final discrete and continuous solvers. The corresponding experiments live in
`experiments/sbeed`.

## Folder overview

- `building_versions/`: historical solver versions kept for analysis and
  comparison. These files show how the implementation evolved from a simple
  one-step linear prototype to the cleaned multi-step scaffold used before the
  final modular solvers.
- `datasets/`: replay buffers for discrete and continuous SBEED. They store
  transitions collected online or loaded offline, support FIFO-style buffers,
  keep terminal flags, and provide contiguous fragments for multi-step targets.
- `features/`: tabular, RBF, linear-wrapper, neural, and continuous policy
  parametrizations. These classes define the value, rho, and policy models used
  by the staged solvers and final experiments.
- `solvers/`: final cleaned SBEED implementations. `DiscreteSBEED` is used for
  finite MDPs and gridworld experiments; `ContinuousSBEED` is used for
  Gymnasium-style continuous control such as Pendulum.

## Main implementation flow

The final solvers use the same high-level update order:

1. fit `rho` to the smoothed multi-step target,
2. update the value model with the SBEED primal objective,
3. update the policy with a KL/Fisher natural-gradient style step.

The staged versions in `building_versions/` are useful when reading the thesis
analysis because each file isolates one implementation decision. For new
experiments, use the final solvers from `solvers/` and the exported classes in
`rl_methods.sbeed`.

