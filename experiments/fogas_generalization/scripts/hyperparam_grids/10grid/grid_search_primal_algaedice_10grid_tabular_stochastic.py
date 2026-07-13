"""Select AlgaeDICE settings for the additional stochastic 10-grid.

This executable entry point delegates to
``primal_algaedice_10grid_tabular_common`` with stochastic dynamics. It reads
the stochastic fixed dataset under ``data/datasets/generalization`` and writes
candidate and best-row tables under
``data/results/generalization/hyperparam_grids/10grid``. This search is kept as
an additional baseline experiment rather than a main thesis result.

Run from the repository root. Results are checkpointed after every candidate;
use ``--max-runs`` for a smoke test and ``--resume`` for the full search.
"""

from primal_algaedice_10grid_tabular_common import run_grid_search


if __name__ == "__main__":
    run_grid_search("stochastic")
