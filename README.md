# Provably Efficient Large-scale Reinforcement Learning via Primal-dual Optimization

This repository contains the code and experimental material developed for the
master's thesis **Provably Efficient Large-scale Reinforcement Learning via
Primal-dual Optimization** by Mauro Díaz Lupone.

The thesis studies offline reinforcement learning with function approximation
and partial data coverage. Its main contribution is **Generalized FOGAS**, an
empirical primal--dual algorithm that retains the main update structure of
Feature Occupancy Gradient AScent (FOGAS) while allowing more general function
approximation. In the shared linear-feature setting, its objective recovers the
empirical FOGAS objective. The repository also contains the FOGAS, Fitted
Q-Iteration (FQI), SBEED, Fenchel AlgaeDICE, and Q-learning implementations used
to keep the training and evaluation pipelines consistent across algorithms.

## Where to find the thesis experiments

The main thesis results are under
[`experiments/fogas_generalization`](experiments/fogas_generalization/README.md).
The three thesis-facing notebooks follow the experimental analysis:

1. [`ablations.ipynb`](experiments/fogas_generalization/notebooks/ablations.ipynb)
   contains the preliminary ablation studies of the occupancy-parameter,
   policy, and value-parameter updates on the deterministic and stochastic
   5 x 5 gridworlds. The main points to inspect are the approximate
   value-parameter best response and the preconditioned occupancy-parameter
   update, particularly in the stochastic environment.
2. [`10grid_comparison.ipynb`](experiments/fogas_generalization/notebooks/10grid_comparison.ipynb)
   contains the partial-coverage experiments with tabular features. It compares
   FOGAS, FQI, SBEED, Fenchel AlgaeDICE, and Generalized FOGAS on the
   deterministic 10 x 10 gridworld while varying the dataset size, exploration
   rate, mixture of optimal and random behaviour policies, and initial-state
   distribution. FOGAS achieves the strongest and most stable performance across
   the coverage conditions; Generalized FOGAS remains competitive and is
   generally more stable than SBEED and Fenchel AlgaeDICE as the data
   distribution changes.
3. [`mountaincar.ipynb`](experiments/fogas_generalization/notebooks/mountaincar.ipynb)
   contains the nonlinear function-approximation experiment on
   `MountainCar-v0`, including the linear tabular, linear RBF, and neural
   parametrizations reported in the thesis. The neural Generalized FOGAS
   version achieves the best overall performance and operates directly on
   continuous states.

The Generalized FOGAS implementation is documented in
[`src/rl_methods/fogas_generalization`](src/rl_methods/fogas_generalization/README.md).
Fixed offline datasets and the saved grid-search, ablation, and evaluation
tables are stored under [`data`](data/README.md). Expensive searches should
normally be inspected through these saved results instead of being repeated
when only the thesis figures and comparisons are needed.

## Additional experiments and implementation studies

The repository contains more experimental material than the final thesis
results:

- [`experiments/fogas`](experiments/fogas/README.md) contains the systematic
  empirical evaluation of FOGAS. It starts with exact small-MDP checks, studies
  partial coverage and tabular or RBF representations, compares FOGAS with FQI,
  and includes the coarse-to-fine 20 x 20, 40 x 40, and 100 x 100 grid
  experiments and a discretized Mountain Car study.
- The additional notebooks in
  [`experiments/fogas_generalization/notebooks`](experiments/fogas_generalization/notebooks/README.md)
  record the development and checks of the linear, RBF, neural, SBEED, and
  Fenchel AlgaeDICE implementations. The experiment scripts also retain
  stochastic-grid searches and larger diagnostic ablations that are not part
  of the final thesis figures.
- [`experiments/sbeed`](experiments/sbeed/README.md) contains the SBEED
  implementation and reproducibility study. It follows the staged development
  from small tabular and RBF gridworld checks to the final discrete solver and
  the continuous Pendulum experiment.
- [`experiments/fqi`](experiments/fqi/README.md) contains standalone FQI checks
  and grid searches. Q-learning is used as a supporting baseline for offline
  dataset construction in the Mountain Car experiments.

## Repository structure

```text
.
|-- src/rl_methods/       reusable MDP, dataset, solver, and evaluation code
|-- experiments/          notebooks and scripts for the empirical studies
|-- data/
|   |-- datasets/         fixed offline transition datasets
|   `-- results/          saved searches, ablations, and evaluation tables
|-- requirements.txt      Python dependencies for the code and notebooks
|-- setup.py              installation metadata for the rl_methods package
`-- LICENSE
```

- [`src/rl_methods`](src/rl_methods/README.md) explains how the shared MDP and
  data utilities connect to each algorithm implementation.
- [`experiments`](experiments/README.md) maps the thesis-facing and additional
  empirical studies to their notebooks, scripts, datasets, and results.
- The more detailed README files inside those folders describe individual
  solvers, experimental protocols, and notebook reading orders.

## Environment and installation

Python 3.10 is the conservative recommendation. It matches several of the
original notebook kernels retained in the repository and provides broad
compatibility with the numerical, Gymnasium, and PyTorch dependencies. The
repository has also been used with newer Python versions, which is why
`setup.py` accepts Python 3.9 and later.

From the repository root, create an isolated Conda environment and install the
project in editable mode:

```bash
conda create --name fogas python=3.10 pip setuptools -y
conda activate fogas
python -m pip install --upgrade pip
python -m pip install -e .
```

The editable installation makes `rl_methods` importable while keeping it linked
to the source files in `src/`, so changes to the implementation are immediately
available to scripts and notebooks. It also installs the packages listed in
`requirements.txt`, including the Jupyter tools required to inspect and run the
notebooks.

To register the environment as an explicit Jupyter kernel and start JupyterLab:

```bash
python -m ipykernel install --user --name fogas \
  --display-name "FOGAS (Python 3.10)"
jupyter lab
```

The default installation is sufficient for CPU execution. For a particular
NVIDIA CUDA configuration, install the appropriate PyTorch build using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
before running `python -m pip install -e .`. Experiment scripts use CUDA where
the implementation supports it and a compatible device is available.

### What `setup.py` does

[`setup.py`](setup.py) is the installation configuration for this repository.
It uses the `src` layout to install the Python package with distribution name
`rl-methods`; the corresponding Python import is `rl_methods`. It also reads
this README as the package description and reads `requirements.txt` as the list
of installation dependencies. Therefore, running `python -m pip install -e .`
both exposes the code under `src/rl_methods` and installs the dependencies used
by the repository.

`setup.py` does not run experiments and does not install the datasets,
notebooks, or saved result tables as package data. Those files remain in the
repository and should be accessed through the project-relative paths used by
the experiment scripts and notebooks.

## Running and inspecting experiments

Run batch scripts from the repository root so that paths under `data/` resolve
consistently. For example, a small Generalized FOGAS smoke test is:

```bash
python experiments/fogas_generalization/scripts/hyperparam_grids/10grid/\
grid_search_final_linear_10grid_tabular.py --max-runs 2
```

Full grid searches and multi-seed evaluations can be computationally
expensive. Check the README and module header for the selected experiment,
prefer `--max-runs` for a smoke test, and use `--resume` where the script
supports checkpointed execution. The notebooks locate the repository root and
load implementations from `src/rl_methods`; their analysis cells expect the
corresponding CSV files under `data/datasets/` and `data/results/`.
