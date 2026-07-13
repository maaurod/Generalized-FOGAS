"""Select neural settings for the additional deterministic 5-grid experiment.

This executable wrapper delegates to
``final_parametrized_5grid_neural_common``. It reads
``data/datasets/generalization/5grid.csv`` and writes candidate and best-row
tables under ``data/results/generalization/hyperparam_grids/5grid``. The
results support the representative nonlinear runs in
``notebooks/grids_nn.ipynb``.

Run from the repository root. The default wrapper runs sequentially on one
device and saves every candidate; use ``--max-runs`` for a smoke test and
``--resume`` for the full additional search.
"""

from final_parametrized_5grid_neural_common import run_grid_search


if __name__ == "__main__":
    run_grid_search("deterministic")
