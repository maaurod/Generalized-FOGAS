"""Select the tabular Generalized FOGAS configuration for the stochastic grid.

This executable wrapper delegates to ``final_linear_5grid_tabular_common``.
It reads ``data/datasets/generalization/5grid_stochastic.csv`` and writes the
candidate and best-row tables under
``data/results/generalization/hyperparam_grids/5grid``. The selected settings
support the stochastic thesis ablations and the representative run in
``notebooks/grids.ipynb``.

Run from the repository root. Results are written after every candidate; use
``--max-runs`` for a smoke test and ``--resume`` for the full search.
"""

from final_linear_5grid_tabular_common import run_grid_search


if __name__ == "__main__":
    run_grid_search("stochastic")
