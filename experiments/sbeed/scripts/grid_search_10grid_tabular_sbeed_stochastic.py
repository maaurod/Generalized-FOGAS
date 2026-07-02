"""SBEED hyperparameter grid search for the stochastic 10x10 tabular grid.

This entrypoint reuses the existing stochastic 10-grid SBEED search logic, but
defaults to the generalization stochastic dataset and writes beside the other
10-grid hyperparameter results.
"""

from __future__ import annotations

import sys

import grid_search_10grid_tabular_sbeed as base


DEFAULT_DATASET_PATH = (
    base.REPO_ROOT / "data" / "datasets" / "generalization" / "10grid_tabular_stoch.csv"
)
DEFAULT_OUTPUT_CSV = (
    base.REPO_ROOT
    / "data"
    / "results"
    / "generalization"
    / "hyperparam_grids"
    / "10grid"
    / "sbeed_10grid_tabular_stochastic_grid_search.csv"
)


def _has_arg(*names: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:] for name in names)


def main() -> None:
    if not _has_arg("--dataset-path", "--dataset-paths"):
        sys.argv.extend(["--dataset-paths", str(DEFAULT_DATASET_PATH)])
    if not _has_arg("--output-csv"):
        sys.argv.extend(["--output-csv", str(DEFAULT_OUTPUT_CSV)])

    base.main()


if __name__ == "__main__":
    main()
