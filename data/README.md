# Experimental Data

This folder contains the datasets and saved results used by the experiment
notebooks in this repository.

```text
data/
|-- datasets/    offline datasets generated for the experiments
`-- results/     saved grid-search and evaluation results
```

## `datasets/`

This folder contains the offline datasets generated in the notebooks and data
collection experiments. Once generated, these datasets are kept fixed and
reused to train and compare the different algorithms.

The subfolders organize the datasets by environment or experimental study,
including the gridworld, generalization, and large-grid experiments.

## `results/`

This folder contains the CSV files produced by the grid searches, ablations,
and policy evaluations. They record the tested configurations, evaluation
metrics, selected hyperparameters, learning-curve checkpoints, and aggregated
results.

The notebooks load these saved tables to compare the methods and produce the
figures and result summaries presented in the thesis. Keeping the expensive
searches separate from the notebooks makes it possible to inspect and recreate
the analysis without rerunning every experiment interactively.
