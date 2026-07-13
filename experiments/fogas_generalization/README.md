# Generalized FOGAS Empirical Experiments

This folder contains the empirical study of **Generalized FOGAS**, the main
algorithm developed in the thesis. It is the experimental companion to the
implementation documented in
[`src/rl_methods/fogas_generalization`](../../src/rl_methods/fogas_generalization/README.md):
the source package defines the objective, parametrizations, solvers, and
baselines, while this folder records how those components are evaluated.

The organization follows the experimental chapter of the thesis. It begins
with controlled ablations of the three Generalized FOGAS updates, then studies
the methods under different levels of partial coverage in a deterministic
10 x 10 gridworld, and finally evaluates linear and nonlinear function
approximation in `MountainCar-v0`. Additional 5 x 5 experiments document the
development of the linear, parametrized, neural, and SBEED implementations.

The purpose of these files is not to repeat the derivation of Generalized
FOGAS. Instead, they provide a map from the experimental claims in the thesis
to the code, fixed datasets, grid-search tables, figures, and representative
runs used to support them.

## Folder organization

```text
experiments/fogas_generalization/
|-- README.md                 scope, organization, and execution workflow
|-- notebooks/               thesis-facing presentation and analysis
|   `-- README.md             ordered guide to every notebook
`-- scripts/                  grid searches, ablation sweeps, and checkpoints
    |-- ablations/            occupancy-, policy-, and value-update studies
    |-- hyperparam_grids/     5 x 5 and 10 x 10 parameter selection
    `-- mountain_car/         continuous-state RBF and neural searches
```

### `notebooks/`

The notebooks are the presentation layer of the empirical study. They define
the environments, show representative solver executions, load the result
tables produced by the scripts, and construct the comparisons and figures
discussed in the thesis.

The three thesis-facing notebooks should be read in the same order as the
experimental chapter:

1. `ablations.ipynb` studies the occupancy-parameter, policy, and
   value-parameter updates on the deterministic and stochastic 5 x 5 grids;
2. `10grid_comparison.ipynb` compares FOGAS, FQI, SBEED, Fenchel AlgaeDICE,
   and Generalized FOGAS under different partial-coverage conditions; and
3. `mountaincar.ipynb` compares tabular, RBF, and neural function
   approximation in `MountainCar-v0`.

The remaining notebooks contain additional experiments that help document how
the final implementations were developed and checked. See
[`notebooks/README.md`](notebooks/README.md) for the role, inputs, and outputs
of every notebook.

### `scripts/`

The scripts are result producers. They run the expensive or repetitive parts
of the study outside Jupyter: generating matched datasets, selecting
hyperparameters, evaluating candidates over multiple seeds, recording
learning-curve checkpoints, and writing intermediate and aggregated CSV
tables. The notebooks subsequently load these tables for plotting and
interpretation.

The script groups correspond to distinct experimental roles:

| Group | Experimental role | Main result location |
| --- | --- | --- |
| `ablations/beta/` | Compares the complete and diagonal Generalized FOGAS occupancy updates with unpreconditioned, unstabilized, Adam/projected-gradient, and regularized best-response alternatives. | `data/results/generalization/ablations/beta/` |
| `ablations/policy/` | Compares Adam, SGD, natural policy gradient, and sampled versus exact policy expectations. | `data/results/generalization/ablations/policy/` |
| `ablations/theta/` | Studies the approximate value-parameter best response, including initialization, inner steps, and regularization. | `data/results/generalization/ablations/theta/` |
| `hyperparam_grids/5grid/` | Selects tabular, RBF, and neural configurations on the deterministic and stochastic 5 x 5 grids. | `data/results/generalization/hyperparam_grids/5grid/` |
| `hyperparam_grids/10grid/` | Selects Generalized FOGAS and AlgaeDICE configurations and evaluates SBEED, AlgaeDICE, and Generalized FOGAS on the matched partial-coverage datasets. | `data/results/generalization/hyperparam_grids/10grid/` |
| `mountain_car/` | Selects continuous-observation RBF and neural Generalized FOGAS configurations for the Mountain Car result table. | `data/results/generalization/mountain_car/` |

The deterministic 5 x 5 ablation scripts, deterministic 10 x 10 comparison
scripts, and Mountain Car searches produce the thesis-facing results. The
stochastic 10 x 10 searches, alternative-feature 5 x 5 searches, and larger
diagnostic ablations are retained as additional experiments. Shared `*_common`
modules implement reusable search logic; the short `grid_search_*.py` files
that import them are the executable entry points.

## Experimental workflow

```text
Implementations in src/rl_methods/
  Generalized FOGAS + FOGAS, FQI, SBEED, and AlgaeDICE baselines
                              |
                              v
Fixed offline dataset or dataset-generation sweep
  data/datasets/generalization/
                              |
                              v
Batch grid search / ablation script
  candidate configuration -> repeated evaluation -> checkpointed CSV
                              |
                              v
Saved result tables
  data/results/generalization/
                              |
                              v
Notebook analysis
  representative runs -> aggregation -> figures and thesis discussion
```

For each experiment, the offline dataset is collected in advance and remains
fixed during training. The comparison scripts use consistent training and
evaluation pipelines across FOGAS, FQI, SBEED, Fenchel AlgaeDICE, and
Generalized FOGAS. Depending on the experiment, the notebooks report success
rate, episode length, average reward, the solver and greedy policies, or the
additional value-function gap used in the 10 x 10 analysis.

## Running the experiments

Run scripts from the repository root so that project-relative dataset and
result paths resolve consistently. Full searches are computationally
expensive; use `--max-runs` for a smoke test and `--resume` when continuing a
checkpointed search.

```bash
python3 experiments/fogas_generalization/scripts/ablations/beta/\
grid_search_final_linear_5grid_tabular_beta_ablation.py --max-runs 2

python3 experiments/fogas_generalization/scripts/hyperparam_grids/10grid/\
grid_search_final_linear_10grid_tabular.py --max-runs 2

python3 experiments/fogas_generalization/scripts/mountain_car/\
grid_search_continuous_mountaincar_fogas_nn_refined.py --max-runs 2
```

Before running a complete search, inspect its module header and command-line
options. Several searches support multiprocessing or explicit device
selection, and the exhaustive candidate grids were designed for the compute
environment described in the thesis.

Open notebooks with the repository environment after generating the result
tables they consume. Representative solver cells may be executed directly,
but the thesis figures should normally be reconstructed from the saved CSV
outputs rather than by repeating every grid search interactively.
