"""Select the thesis AlgaeDICE configuration for the deterministic 10-grid.

This executable entry point delegates to
``primal_algaedice_10grid_tabular_common``. It reads
``data/datasets/generalization/10grid_tabular_new.csv`` and writes candidate
and best-row tables under
``data/results/generalization/hyperparam_grids/10grid``. The selected
configuration is held fixed by the AlgaeDICE partial-coverage sweep presented
in ``notebooks/10grid_comparison.ipynb``.

Run from the repository root. Results are checkpointed after every candidate;
use ``--max-runs`` for a smoke test and ``--resume`` for the full search.
"""

from primal_algaedice_10grid_tabular_common import run_grid_search


if __name__ == "__main__":
    run_grid_search("deterministic")
