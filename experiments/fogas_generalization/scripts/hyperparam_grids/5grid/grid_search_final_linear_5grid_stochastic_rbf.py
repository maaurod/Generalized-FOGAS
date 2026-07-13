"""Select linear-RBF settings for the additional stochastic 5-grid study.

This executable wrapper delegates to ``final_linear_5grid_rbf_common``. It
reads ``data/datasets/generalization/5grid_stochastic.csv`` and writes
candidate and best-row tables under
``data/results/generalization/hyperparam_grids/5grid``. The search supports the
stochastic feature experiment in ``notebooks/grids_param.ipynb``.

Run from the repository root. Results are written after every candidate; use
``--max-runs`` for a smoke test and ``--resume`` for the full search.
"""

from final_linear_5grid_rbf_common import run_grid_search


if __name__ == "__main__":
    run_grid_search("stochastic")
