"""Tune FOGAS on the fixed 10 x 10 tabular four-room experiment.

This script varies the FOGAS optimization parameters while keeping the MDP,
feature map, and offline dataset fixed. It therefore studies optimization
sensitivity, unlike ``grid_search_10grid_fogas.py``, which fixes the solver and
varies how the dataset is generated. The selected configurations support the
10-grid analysis in ``experiments/fogas/notebooks/10grid_tabular.ipynb``.

Run from the repository root with
``python3 experiments/fogas/scripts/grid_10_tabular.py``. Use ``--resume`` to
skip completed candidates and ``--max-runs N`` for a short validation run. The
full table and best successful row are written to ``data/results/grids/``.
"""

import argparse
import itertools
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def find_root(current_path):
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / "src" / "rl_methods").exists() and (parent / "data").exists():
            return parent
    return current_path


PROJECT_ROOT = find_root(Path(__file__).resolve())
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "10grid_tabular.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "grids"
OUTPUT_CSV = RESULTS_DIR / "grid_10_tabular.csv"
BEST_CSV = RESULTS_DIR / "grid_10_tabular_best.csv"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_methods.fogas import FOGASEvaluator, FOGASSolver
from rl_methods.mdp import DiscreteMDP, Planner


SEED = 42
T = 20_000
BETA = 1e-7
NUM_TRAJECTORIES = 10
MAX_STEPS = 50

D_THETA_GRID = [0.5, 1, 2, 5, 10, 20, 40, 60]
ALPHA_GRID = [0.00025, 0.0005, 0.001, 0.002, 0.003, 0.005]
ETA_GRID = [0.0001, 0.0002, 0.0005, 0.001, 0.002]
RHO_GRID = [0.0005, 0.001, 0.003]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 10x10 tabular FOGAS hyperparameter grid search."
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit the number of candidates. Useful for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip parameter combinations already present in the output CSV.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars, including the inner FOGAS progress bar.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def build_mdp():
    states = torch.arange(100, dtype=torch.int64)
    actions = torch.arange(4, dtype=torch.int64)
    n_states = len(states)
    n_actions = len(actions)
    gamma = 0.9
    x0 = 0
    goal = 99
    pits = {18, 32, 57, 61, 75}
    walls = {
        4,
        11,
        14,
        17,
        21,
        22,
        27,
        34,
        37,
        40,
        42,
        43,
        44,
        45,
        46,
        47,
        49,
        54,
        62,
        64,
        66,
        72,
        76,
        82,
        84,
        86,
        87,
        94,
    }

    def phi(x, a):
        vec = torch.zeros(n_states * n_actions, dtype=torch.float64)
        vec[int(x) * n_actions + int(a)] = 1.0
        return vec

    def to_rc(s):
        return divmod(s, 10)

    def to_s(r, c):
        return r * 10 + c

    def next_state(s, a):
        if s == goal or s in pits:
            return s

        r, c = to_rc(s)
        if a == 0:
            r2, c2 = max(0, r - 1), c
        elif a == 1:
            r2, c2 = min(9, r + 1), c
        elif a == 2:
            r2, c2 = r, max(0, c - 1)
        elif a == 3:
            r2, c2 = r, min(9, c + 1)
        else:
            raise ValueError("Invalid action")

        next_s = to_s(r2, c2)
        if next_s in walls:
            return s
        return next_s

    terminal_states = {goal, *pits}
    step_cost = -0.1
    goal_reward = 1.0
    pit_reward = -5.0
    terminal_self_loop_reward = 0.0

    omega = torch.empty(n_states * n_actions, dtype=torch.float64)
    for state in range(n_states):
        for action in range(n_actions):
            next_s = next_state(state, action)
            idx = state * n_actions + action
            if state in terminal_states:
                omega[idx] = terminal_self_loop_reward
            elif next_s == goal:
                omega[idx] = goal_reward
            elif next_s in pits:
                omega[idx] = pit_reward
            else:
                omega[idx] = step_cost

    def psi(xp):
        v = torch.zeros(n_states * n_actions, dtype=torch.float64)
        for x in states:
            for a in actions:
                if next_state(int(x), int(a)) == int(xp):
                    v[int(x) * n_actions + int(a)] = 1.0
        return v

    mdp = DiscreteMDP(
        states=states,
        actions=actions,
        phi=phi,
        omega=omega,
        gamma=gamma,
        x0=x0,
        psi=psi,
        terminal_states=terminal_states,
    )
    return mdp, phi, goal, pits, terminal_states


def candidate_key(row):
    return (
        float(row["alpha"]),
        float(row["eta"]),
        float(row["rho"]),
        float(row["D_theta"]),
    )


def load_existing_results(resume):
    if not resume or not OUTPUT_CSV.exists():
        return [], set()

    df = pd.read_csv(OUTPUT_CSV)
    rows = df.to_dict("records")
    completed = {candidate_key(row) for row in rows}
    return rows, completed


def ordered_results_frame(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df

    return df.sort_values(
        by=[
            "greedy_success_rate",
            "solver_success_rate",
            "greedy_avg_reward",
            "solver_avg_reward",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = ordered_results_frame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    successful = df[df["status"] == "ok"] if not df.empty else df
    if not successful.empty:
        successful.head(1).to_csv(BEST_CSV, index=False)


def run_candidate(solver, evaluator, params, device, goal, terminal_states, progress):
    start = time.perf_counter()
    alpha, eta, rho, d_theta = params
    row = {
        "alpha": float(alpha),
        "eta": float(eta),
        "rho": float(rho),
        "D_theta": float(d_theta),
        "T": int(T),
        "beta": float(BETA),
        "seed": int(SEED),
        "device": str(device),
        "status": "ok",
        "error": "",
        "elapsed_seconds": np.nan,
        "greedy_avg_reward": np.nan,
        "greedy_success_rate": np.nan,
        "solver_avg_reward": np.nan,
        "solver_success_rate": np.nan,
    }

    try:
        solver.run(
            alpha=alpha,
            eta=eta,
            rho=rho,
            D_theta=d_theta,
            T=T,
            tqdm_print=progress,
        )

        greedy_avg_reward = evaluator.average_return(
            policy_mode="greedy",
            num_trajectories=NUM_TRAJECTORIES,
            max_steps=MAX_STEPS,
            seed=SEED,
            terminal_states=terminal_states,
        )["policy"]
        greedy_success_rate = evaluator.success_rate(
            goal_state=goal,
            policy_mode="greedy",
            num_trajectories=NUM_TRAJECTORIES,
            max_steps=MAX_STEPS,
            seed=SEED,
            terminal_states=terminal_states,
        )["policy"]

        solver_avg_reward = evaluator.average_return(
            policy_mode="solver",
            num_trajectories=NUM_TRAJECTORIES,
            max_steps=MAX_STEPS,
            seed=SEED,
            terminal_states=terminal_states,
        )["policy"]
        solver_success_rate = evaluator.success_rate(
            goal_state=goal,
            policy_mode="solver",
            num_trajectories=NUM_TRAJECTORIES,
            max_steps=MAX_STEPS,
            seed=SEED,
            terminal_states=terminal_states,
        )["policy"]

        row.update(
            {
                "greedy_avg_reward": float(greedy_avg_reward),
                "greedy_success_rate": float(greedy_success_rate),
                "solver_avg_reward": float(solver_avg_reward),
                "solver_success_rate": float(solver_success_rate),
            }
        )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        row["elapsed_seconds"] = float(time.perf_counter() - start)

    return row


def main():
    args = parse_args()
    set_seed(SEED)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Results: {OUTPUT_CSV}")

    mdp, phi, goal, pits, terminal_states = build_mdp()
    planner = Planner(mdp)
    solver = FOGASSolver(
        mdp=mdp,
        phi=phi,
        csv_path=str(DATASET_PATH),
        device=device,
        seed=SEED,
        beta=BETA,
        print_params=True,
    )
    evaluator = FOGASEvaluator(solver=solver, mdp=mdp, planner=planner)

    all_candidates = list(
        itertools.product(ALPHA_GRID, ETA_GRID, RHO_GRID, D_THETA_GRID)
    )
    if args.max_runs is not None:
        all_candidates = all_candidates[: max(0, int(args.max_runs))]

    results, completed = load_existing_results(args.resume)
    candidates = [
        candidate for candidate in all_candidates if candidate_key(
            {
                "alpha": candidate[0],
                "eta": candidate[1],
                "rho": candidate[2],
                "D_theta": candidate[3],
            }
        )
        not in completed
    ]

    print(f"Total grid size: {len(ALPHA_GRID) * len(ETA_GRID) * len(RHO_GRID) * len(D_THETA_GRID)}")
    print(f"Candidates to run: {len(candidates)}")
    if args.resume:
        print(f"Resumed rows: {len(results)}")

    if not args.resume and OUTPUT_CSV.exists():
        print("Existing output will be overwritten because --resume was not set.")

    progress = not args.no_progress
    outer = tqdm(candidates, desc="Grid search", unit="run", disable=not progress)
    for run_idx, params in enumerate(outer, start=len(results) + 1):
        row = run_candidate(
            solver=solver,
            evaluator=evaluator,
            params=params,
            device=device,
            goal=goal,
            terminal_states=terminal_states,
            progress=progress,
        )
        row["run_idx"] = int(run_idx)
        results.append(row)
        save_results(results)

        if progress:
            outer.set_postfix(
                {
                    "success": row["greedy_success_rate"],
                    "reward": row["greedy_avg_reward"],
                    "solver_success": row["solver_success_rate"],
                    "status": row["status"],
                }
            )

    save_results(results)
    df = ordered_results_frame(results)
    ok_count = int((df["status"] == "ok").sum()) if not df.empty else 0
    failed_count = int((df["status"] == "failed").sum()) if not df.empty else 0

    print("\nGrid search complete.")
    print(f"Rows saved: {len(df)}")
    print(f"Successful runs: {ok_count}")
    print(f"Failed runs: {failed_count}")
    print(f"Output CSV: {OUTPUT_CSV}")
    if BEST_CSV.exists():
        print(f"Best row CSV: {BEST_CSV}")
        print("\nTop result:")
        print(pd.read_csv(BEST_CSV).to_string(index=False))


if __name__ == "__main__":
    main()
