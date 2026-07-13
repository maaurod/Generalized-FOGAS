"""Select the thesis Generalized FOGAS configuration for the 10 x 10 grid.

This executable entry point delegates to ``final_linear_10grid_tabular_common``
with deterministic dynamics and tabular features. It reads
``data/datasets/generalization/10grid_tabular_new.csv`` and writes candidate
and best-row tables under
``data/results/generalization/hyperparam_grids/10grid``. The selected settings
are used by the partial-coverage sweep shown in
``notebooks/10grid_comparison.ipynb``.

Run from the repository root. Results are checkpointed after every candidate;
use ``--max-runs`` for a smoke test and ``--resume`` for the full search.
"""

from final_linear_10grid_tabular_common import run_grid_search


if __name__ == "__main__":
    run_grid_search("deterministic")
