"""Select Generalized FOGAS settings for the additional stochastic 10-grid.

This executable entry point delegates to ``final_linear_10grid_tabular_common``
with stochastic dynamics and tabular features. It reads the stochastic fixed
datasets under ``data/datasets/generalization`` and writes candidate and
best-row tables under
``data/results/generalization/hyperparam_grids/10grid``. These runs extend the
main deterministic study and are retained as additional experiments.

Run from the repository root. Results are checkpointed after every candidate;
use ``--max-runs`` for a smoke test and ``--resume`` for the full search.
"""

from final_linear_10grid_tabular_common import run_grid_search


if __name__ == "__main__":
    run_grid_search("stochastic")
