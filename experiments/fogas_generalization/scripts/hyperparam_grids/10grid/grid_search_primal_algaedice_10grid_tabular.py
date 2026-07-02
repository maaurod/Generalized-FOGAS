"""
PrimalAlgaeDICESolver tabular grid search for the deterministic clean 10x10 grid.

The script writes results after every candidate so completed runs survive
interruptions. Use --max-runs for a quick smoke test and --resume for long runs.
"""

from primal_algaedice_10grid_tabular_common import run_grid_search


if __name__ == "__main__":
    run_grid_search("deterministic")
