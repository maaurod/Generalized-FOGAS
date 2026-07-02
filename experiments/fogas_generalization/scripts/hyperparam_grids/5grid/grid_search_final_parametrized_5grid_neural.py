"""
FinalParametrizedSolver neural grid search for the deterministic clean 5x5 grid.

The script runs sequentially on one device/GPU and writes results after every
candidate. Use --max-runs for a smoke test and --resume for long runs.
"""

from final_parametrized_5grid_neural_common import run_grid_search


if __name__ == "__main__":
    run_grid_search("deterministic")
