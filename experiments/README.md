# Experiments

This folder contains the notebooks and batch scripts used to evaluate the
methods implemented under `src/rl_methods`. The main thesis experiments are in
`fogas_generalization`; the remaining folders contain the systematic FOGAS
evaluation, baseline experiments, and additional implementation studies.

## Folder structure

```text
experiments/
|-- fogas_generalization/   main Generalized FOGAS thesis experiments
|-- fogas/                  systematic empirical evaluation of FOGAS
|-- fqi/                    standalone Fitted Q-Iteration experiments
|-- sbeed/                  SBEED implementation and reproducibility study
`-- media/                  figures and animations used by the notebooks
```

### `fogas_generalization/`

This is the main experimental folder for the thesis. Its three thesis-facing
notebooks cover:

1. preliminary ablation studies of the Generalized FOGAS updates;
2. partial-coverage experiments with tabular features on the deterministic
   10 x 10 gridworld; and
3. nonlinear function approximation on `MountainCar-v0`.

The partial-coverage comparison uses FOGAS, FQI, SBEED, Fenchel AlgaeDICE, and
Generalized FOGAS with consistent training and evaluation pipelines. Additional
notebooks and scripts retain deterministic and stochastic 5 x 5 studies,
alternative parametrizations, stochastic 10 x 10 searches, and diagnostic
ablations. See
[`fogas_generalization/README.md`](fogas_generalization/README.md) for the
experiment workflow and
[`fogas_generalization/notebooks/README.md`](fogas_generalization/notebooks/README.md)
for the ordered notebook guide.

### `fogas/`

This folder contains the systematic empirical evaluation of the original
FOGAS method. The study moves from oracle and empirical checks on small finite
MDPs to partial-coverage and representation experiments on 10 x 10 grids. It
also includes matched FOGAS--FQI comparisons, coarse-to-fine transfer to 20 x
20, 40 x 40, and 100 x 100 grids, and a discretized Mountain Car experiment.
See [`fogas/README.md`](fogas/README.md) for the execution workflow and
[`fogas/notebooks/README.md`](fogas/notebooks/README.md) for the complete study
map.

### `fqi/`

This folder contains standalone FQI notebooks and grid-search scripts used to
check the baseline under different dataset-collection strategies, feature
representations, and coverage conditions. FQI is also run directly from the
FOGAS and Generalized FOGAS experiment suites when a matched comparison is
required. See [`fqi/README.md`](fqi/README.md).

### `sbeed/`

This folder contains the SBEED implementation and reproducibility study. The
`building_notebooks/` folders follow the staged implementation from the naive
one-step solver through terminal-aware rho updates, optimizer comparisons, and
multi-step fragments. The `discrete/` and `continuous/` folders contain the
final gridworld and Pendulum experiments, while `scripts/` contains the larger
parameter searches. See [`sbeed/README.md`](sbeed/README.md).

## Common organization

Each experiment suite keeps reusable implementations separate from the
experimental protocol:

```text
src/rl_methods/ implementation
          |
          v
experiment notebook or batch script
          |
          +--> data/datasets/   fixed offline transitions
          `--> data/results/    searches, checkpoints, and evaluations
                                  |
                                  v
                              notebook analysis
```

- `notebooks/` define environments, show representative solver runs, load
  saved result tables, and construct the comparisons and figures.
- `scripts/` run expensive or repetitive dataset sweeps, parameter searches,
  multi-seed evaluations, and checkpointed experiments outside Jupyter.
- The [`data` overview](../data/README.md) describes the fixed offline datasets
  that remain unchanged during solver training.
- `../data/results/` contains the CSV files and selected configurations loaded
  by the notebooks.

## Running experiments

Install the repository as described in the [main README](../README.md), then
run scripts from the repository root. For example:

```bash
python experiments/fogas_generalization/scripts/hyperparam_grids/10grid/\
grid_search_final_linear_10grid_tabular.py --max-runs 2
```

Full searches can be computationally expensive. Read the selected folder's
README and the script module header before starting a complete run. Use the
small-run and resume options provided by the script where available. When only
the reported analysis is needed, open the corresponding notebook and use the
saved datasets and result tables already stored in the repository.
