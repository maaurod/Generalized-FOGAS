"""
FinalLinearSolver tabular grid search for the stochastic clean 5x5 grid.

The script writes results after every candidate so completed runs survive
interruptions. Use --max-runs for a quick smoke test and --resume for long runs.
"""

from final_linear_5grid_tabular_common import run_grid_search


if __name__ == "__main__":
    run_grid_search("stochastic")
