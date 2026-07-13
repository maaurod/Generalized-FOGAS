# FOGAS Empirical Experiments

This folder contains the empirical study of **FOGAS** developed for the
thesis. It documents and evaluates FOGAS implemented in the repository. Shared
MDP models, data collectors, and algorithm implementations remain under
`src/rl_methods`; this folder contains the experimental protocols that combine
those components.

The study has four main objectives:

1. verify the implementation on finite problems for which exact oracle
   quantities are available;
2. study how offline-data coverage affects the learned policy;
3. compare FOGAS with linear Fitted Q-Iteration (FQI) under tabular and RBF
   representations; and
4. test whether feature-based and coarse-to-fine formulations make larger or
   continuous-control problems tractable.

## Folder organization

```text
experiments/fogas/
|-- README.md                 experiment scope and reproducibility workflow
|-- scripts/                  unattended parameter and dataset sweeps
`-- notebooks/               analysis, figures, and thesis-facing discussion
    `-- README.md             detailed map of the empirical study
```

### `scripts/`

The scripts execute computationally expensive or repetitive experiments
without requiring an interactive notebook. They generate datasets, sweep
algorithm or data-collection parameters, evaluate learned policies, and write
the resulting tables to `data/results/`. Long searches write intermediate
results whenever possible so that completed configurations are retained after
an interruption.

The scripts are result producers. Their CSV files are subsequently loaded by
the notebooks, where the results are aggregated, plotted, compared, and
interpreted. Each script starts with a module description containing its
scientific purpose, relation to the notebooks, invocation, inputs, and outputs.

### `notebooks/`

The notebooks are the presentation layer of the empirical study. They define
the benchmark environments and feature maps, demonstrate representative runs,
analyze feature coverage, compare FOGAS with oracle quantities and FQI, and
produce the figures used to discuss the method's behavior and limitations.

Small notebooks are intentionally self-contained so that the environment,
feature map, dataset, solver, and evaluation can be inspected together. Larger
grid searches are delegated to `scripts/`; the corresponding notebook reads
the saved tables rather than recomputing every candidate interactively. See
[`notebooks/README.md`](notebooks/README.md) for the research questions and the
role of every notebook.

## Experimental workflow

```text
Shared implementations
  src/rl_methods/mdp
  src/rl_methods/data_collection
  src/rl_methods/fogas
  src/rl_methods/fqi and q_learning (baselines)
                |
                v
Notebook protocol or batch script
  environment -> feature map -> offline dataset -> solver -> evaluator
                |
                v
data/datasets/                 data/results/
saved transition batches      sweep tables and selected configurations
                \                 /
                 \               /
                  v             v
                notebooks/
        figures, comparisons, and thesis discussion
```

All offline datasets use the transition schema
`state, action, reward, next_state`. A typical FOGAS experiment constructs an
MDP abstraction and a feature map `phi`, collects or loads an offline dataset,
runs `FOGASSolver`, and evaluates both the stochastic solver policy and its
greedy counterpart with `FOGASEvaluator`. When the finite model is small
enough, `Planner` supplies exact values, optimal policies, and occupancy
measures for reference.

## Batch scripts

| Script | Experimental role | Principal output |
| --- | --- | --- |
| `grid_10_tabular.py` | Sweeps FOGAS optimization parameters on the fixed 10 x 10 tabular dataset and retains the best valid configuration. | `data/results/grids/grid_10_tabular.csv` and `grid_10_tabular_best.csv` |
| `grid_search_10grid_fogas.py` | Varies dataset size, exploration, behavior-policy mixture, and reset distribution for FOGAS on the 10 x 10 tabular grid. | `data/results/10grid_tabular/fogas_dataset_grid.csv` |
| `grid_search_10grid_fqi.py` | Repeats the same dataset grid with FQI, making the solver comparison use matched data-generation conditions. | `data/results/10grid_tabular/fqi_dataset_grid.csv` |
| `grid_search_sbatch.py` | Runs the extended 10 x 10 FOGAS dataset sweep used by the final coverage and convergence comparisons. | `data/results/grids/grid_search_results_sbatch.csv` |
| `grid_search_fogas_40grid.py` | Sweeps FOGAS step sizes on the fixed 40 x 40 RBF experiment and stores policy trajectories. | `data/results/grids/grid_search_results_fogas_40grid.csv` and `grid_search_paths_fogas_40grid.json` |
| `grid_search_dataset_40grid.py` | Holds the 40 x 40 solver parameters fixed and studies manual support augmentation, epsilon exploration, and random-start coverage. | Three `grid_search_dataset_40grid_[A-C].csv` files under `data/results/grids/` |
| `plot_dataset_grid_search_40grid.py` | Converts the three 40 x 40 dataset-sweep tables into convergence, return, and optimality-gap figures. | Interactive figures or PNG files selected by the command-line options |
| `grid_mountaincar.py` | Builds Q-learning-guided Gymnasium datasets and evaluates FOGAS and FQI over matched Mountain Car collection settings. | `data/results/mountainCar/grids/grid_mountaincar.csv` |

## Running experiments

Run scripts from the repository root so project-relative paths resolve
consistently:

```bash
python3 experiments/fogas/scripts/grid_10_tabular.py --max-runs 2
python3 experiments/fogas/scripts/grid_search_10grid_fogas.py
python3 experiments/fogas/scripts/grid_search_10grid_fqi.py
python3 experiments/fogas/scripts/grid_mountaincar.py
```

The 10 x 10 parameter search supports `--resume`; the other exhaustive scripts
encode their grids as module-level constants. Inspect those constants before a
full run, since the complete searches can be computationally expensive. CUDA is
used automatically where supported by the script and available to PyTorch.

To plot the 40 x 40 dataset study from an explicit result directory:

```bash
python3 experiments/fogas/scripts/plot_dataset_grid_search_40grid.py \
  --results_dir data/results/grids --save
```

Notebooks should be opened with the repository environment and executed from a
location below the repository root. Their root-discovery cells locate `src/`
and `data/` automatically. Batch outputs should be generated before running
notebook cells that load result CSVs.
