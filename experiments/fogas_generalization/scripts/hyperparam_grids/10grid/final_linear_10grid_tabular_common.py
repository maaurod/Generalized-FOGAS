"""Shared Generalized FOGAS search infrastructure for the 10 x 10 grids.

Scientific role
---------------
This helper implements parameter selection for ``FinalLinearSolver`` with
tabular residual-weighting, action-value, and policy features. The
deterministic entry point selects the configuration used by the thesis
partial-coverage comparison; the stochastic entry point is an additional
experiment under noisy transitions.

Inputs and outputs
------------------
Problem-specific settings select ``10grid_tabular_new.csv`` or the stochastic
10-grid datasets under ``data/datasets/generalization``. Candidate and best-row
CSVs are written to ``data/results/generalization/hyperparam_grids/10grid``.
The selected deterministic configuration is then used by the dataset sweep
presented in ``notebooks/10grid_comparison.ipynb``.

This module is not an executable entry point. Run
``grid_search_final_linear_10grid_tabular.py`` or its ``_stochastic`` variant
from the repository root. The wrappers expose ``--max-runs``, ``--resume``,
worker, thread, and device options; the parent process owns checkpoint writes.
"""

from __future__ import annotations

import argparse
import itertools
import math
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets" / "generalization"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "generalization" / "hyperparam_grids" / "10grid"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from rl_methods.fogas import FOGASEvaluator  # noqa: E402
from rl_methods.fogas_generalization import (  # noqa: E402
    FinalLinearSolver,
    LinearFunction,
    LinearQFunction,
    TabularFeatures,
)
from rl_methods.mdp import DiscreteMDP, Planner  # noqa: E402


SEED = 42
NUM_TRAJECTORIES = 100
MAX_STEPS = 100
MAX_STEPS_LONG = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

STATES = torch.arange(100, dtype=torch.long)
ACTIONS = torch.arange(4, dtype=torch.long)
N = len(STATES)
A = len(ACTIONS)
GAMMA = 0.9
GRID_SIZE = 10
X0 = 0
GOAL_GRID = 99
PIT_GRIDS = {18, 32, 57, 61, 75}
WALL_STATES = {
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
TERMINAL_STATES = {GOAL_GRID, *PIT_GRIDS}

ALPHA_GRID = [5e-4, 1e-3, 3e-3, 5e-3]
ETA_GRID = [3e-5, 1e-4, 3e-4, 1e-3]
RHO_GRID = [0.01, 0.1, 0.5, 1.0, 2.0]
T_GRID = [5000, 10000, 20000]
THETA_LR_GRID = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
THETA_INNER_STEPS_GRID = [10]
THETA_LAMBDA_GRID = [
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
]

REINFORCE_SAMPLES = 4

PROBLEMS = {
    "deterministic": {
        "description": "deterministic 10x10 FinalLinearSolver tabular grid search on the new dataset",
        "dataset_path": DATASETS_DIR / "10grid_tabular_new.csv",
        "output_csv": RESULTS_DIR / "final_linear_10grid_tabular_new_grid_search.csv",
        "best_csv": RESULTS_DIR / "final_linear_10grid_tabular_new_grid_search_best.csv",
        "intended_prob": 1.0,
    },
    "stochastic": {
        "description": "stochastic 10x10 FinalLinearSolver tabular grid search",
        "dataset_path": DATASETS_DIR / "10grid_stoch_new.csv",
        "output_csv": RESULTS_DIR / "final_linear_10grid_tabular_stochastic_grid_search.csv",
        "best_csv": RESULTS_DIR / "final_linear_10grid_tabular_stochastic_grid_search_best.csv",
        "intended_prob": 0.8,
    },
        "stochastic2": {
        "description": "stochastic 10x10 FinalLinearSolver tabular grid search",
        "dataset_path": DATASETS_DIR / "10grid_stoch_new2.csv",
        "output_csv": RESULTS_DIR / "final_linear_10grid_tabular_stochastic_grid_search2.csv",
        "best_csv": RESULTS_DIR / "final_linear_10grid_tabular_stochastic_grid_search_best2.csv",
        "intended_prob": 0.8,
    },
}


def parse_args(description):
    parser = argparse.ArgumentParser(description=f"Run the {description}.")
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
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Use 1 for sequential execution.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch CPU threads per worker. Keep low when using multiple workers.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help=(
            "Comma-separated list of devices to distribute workers across "
            "(e.g. cuda:0,cuda:1,cuda:2). Workers are assigned round-robin. "
            "Defaults to the global DEVICE (cuda if available, else cpu)."
        ),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def configure_worker_threads(torch_threads):
    torch_threads = max(1, int(torch_threads))
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(torch_threads)
    except RuntimeError:
        pass


def state_to_pos(s):
    return divmod(int(s), GRID_SIZE)


def pos_to_state(row, col):
    return int(row) * GRID_SIZE + int(col)


def move_deterministic(s, a):
    s = int(s)
    a = int(a)

    if s in TERMINAL_STATES:
        return s

    row, col = state_to_pos(s)
    if a == 0:
        new_row, new_col = row - 1, col
    elif a == 1:
        new_row, new_col = row + 1, col
    elif a == 2:
        new_row, new_col = row, col - 1
    elif a == 3:
        new_row, new_col = row, col + 1
    else:
        raise ValueError("action must be in {0, 1, 2, 3}")

    if not (0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE):
        return s

    sp = pos_to_state(new_row, new_col)
    if sp in WALL_STATES:
        return s
    return sp


def transition_probs(s, a, intended_prob):
    probs_by_state = {}

    for candidate_a in range(A):
        prob = (1.0 - float(intended_prob)) / A
        if candidate_a == int(a):
            prob += float(intended_prob)

        sp = move_deterministic(s, candidate_a)
        probs_by_state[sp] = probs_by_state.get(sp, 0.0) + prob

    return probs_by_state


def reward_from_next_state(sp):
    sp = int(sp)
    if sp == GOAL_GRID:
        return 1.0
    if sp in PIT_GRIDS:
        return -5.0
    return -0.1


def build_mdp(problem_name, device):
    intended_prob = float(PROBLEMS[problem_name]["intended_prob"])

    def transition_fn(s, a):
        probs = torch.zeros(N, dtype=torch.float64)
        for sp, prob in transition_probs(s, a, intended_prob).items():
            probs[sp] = prob
        return probs

    def reward_fn(s, a):
        if int(s) in TERMINAL_STATES:
            return 0.0

        return sum(
            prob * reward_from_next_state(sp)
            for sp, prob in transition_probs(s, a, intended_prob).items()
        )

    mdp = DiscreteMDP(
        states=STATES,
        actions=ACTIONS,
        gamma=GAMMA,
        x0=X0,
        reward_fn=reward_fn,
        transition_fn=transition_fn,
        terminal_states=list(TERMINAL_STATES),
    ).to(device)
    planner = Planner(mdp).to(device)
    return mdp, planner


def make_solver(
    dataset_path,
    device,
    theta_lr,
    theta_inner_steps,
    theta_mode,
    theta_lambda,
    D_theta,
):
    u_features = TabularFeatures(N, A)
    q_features = TabularFeatures(N, A)
    policy_features = TabularFeatures(N, A)

    return FinalLinearSolver(
        n_states=N,
        n_actions=A,
        gamma=GAMMA,
        x0=X0,
        csv_path=str(dataset_path),
        u_function=LinearFunction(u_features),
        q_function=LinearQFunction(q_features),
        policy_features=policy_features,
        seed=SEED,
        device=device,
        D_theta=D_theta,
        theta_include_beta_cov=False,
        theta_mode=theta_mode,
        theta_lambda=theta_lambda,
        theta_optimizer="adam",
        theta_inner_steps=theta_inner_steps,
        theta_lr=theta_lr,
        theta_start_mode="warm",
        beta_update="fogas_full",
    )


def optional_float_key(value):
    if value is None:
        return "none"
    try:
        if pd.isna(value):
            return "none"
    except TypeError:
        pass
    return f"{float(value):.12g}"


def candidate_key(row):
    return (
        float(row["alpha"]),
        float(row["eta"]),
        float(row["rho"]),
        int(row["T"]),
        float(row["theta_lr"]),
        int(row["theta_inner_steps"]),
        str(row["theta_mode"]),
        optional_float_key(row.get("theta_lambda")),
        optional_float_key(row.get("D_theta")),
    )


def load_existing_results(resume, output_csv):
    if not resume or not output_csv.exists():
        return [], set()

    try:
        df = pd.read_csv(output_csv)
    except pd.errors.EmptyDataError:
        # File exists but is empty (e.g. left by a cancelled run). Treat as no prior results.
        return [], set()

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
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
            "greedy_avg_return",
            "solver_avg_return",
            "elapsed_seconds",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def save_results(results, output_csv, best_csv):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = ordered_results_frame(results)
    df.to_csv(output_csv, index=False)

    successful = df[df["status"] == "ok"] if not df.empty else df
    if not successful.empty:
        successful.head(1).to_csv(best_csv, index=False)


def blank_metrics():
    return {
        "solver_success_rate": np.nan,
        "solver_success_rate_200": np.nan,
        "solver_avg_return": np.nan,
        "solver_v_x0": np.nan,
        "solver_v_gap": np.nan,
        "greedy_success_rate": np.nan,
        "greedy_success_rate_200": np.nan,
        "greedy_avg_return": np.nan,
        "greedy_v_x0": np.nan,
        "greedy_v_gap": np.nan,
        "final_total_loss": np.nan,
        "final_policy_objective": np.nan,
        "final_beta_objective": np.nan,
        "final_q_objective": np.nan,
        "final_theta_norm": np.nan,
        "final_policy_grad_norm": np.nan,
        "final_beta_grad_norm": np.nan,
        "final_theta_grad_norm": np.nan,
    }


def base_row(params, problem_name, device, status="ok", error=""):
    alpha, eta, rho, T, theta_lr, theta_inner_steps, theta_mode, theta_lambda, D_theta = params
    problem = PROBLEMS[problem_name]
    row = {
        "alpha": float(alpha),
        "eta": float(eta),
        "rho": float(rho),
        "T": int(T),
        "theta_lr": float(theta_lr),
        "theta_inner_steps": int(theta_inner_steps),
        "theta_mode": str(theta_mode),
        "theta_lambda": None if theta_lambda is None else float(theta_lambda),
        "D_theta": None if D_theta is None else float(D_theta),
        "theta_optimizer": "adam",
        "theta_start_mode": "warm",
        "theta_include_beta_cov": False,
        "beta_update": "fogas_full",
        "policy_optimizer": "adam",
        "policy_gradient": "exact",
        "reinforce_samples": int(REINFORCE_SAMPLES),
        "state_weight_update": "normal",
        "intended_prob": float(problem["intended_prob"]),
        "dataset_path": str(problem["dataset_path"]),
        "num_trajectories": int(NUM_TRAJECTORIES),
        "max_steps": int(MAX_STEPS),
        "max_steps_long": int(MAX_STEPS_LONG),
        "seed": int(SEED),
        "device": str(device),
        "status": status,
        "error": error,
        "elapsed_seconds": np.nan,
    }
    row.update(blank_metrics())
    return row


def failed_worker_row(params, problem_name, exc, device=None):
    return base_row(params, problem_name, device or DEVICE, status="failed", error=repr(exc))


def evaluate_policy(planner, evaluator, policy_mode, d_star, v_star):
    pi = evaluator.get_policy(policy_mode)
    v_pi, _ = planner.evaluate_policy(pi)
    v_gap = float((d_star * (v_star - v_pi.detach().cpu())).sum().item())
    return {
        f"{policy_mode}_success_rate": float(
            evaluator.success_rate(
                goal_state=GOAL_GRID,
                policy_mode=policy_mode,
                num_trajectories=NUM_TRAJECTORIES,
                max_steps=MAX_STEPS,
                seed=SEED,
                terminal_states=TERMINAL_STATES,
            )["policy"]
        ),
        f"{policy_mode}_success_rate_200": float(
            evaluator.success_rate(
                goal_state=GOAL_GRID,
                policy_mode=policy_mode,
                num_trajectories=NUM_TRAJECTORIES,
                max_steps=MAX_STEPS_LONG,
                seed=SEED,
                terminal_states=TERMINAL_STATES,
            )["policy"]
        ),
        f"{policy_mode}_avg_return": float(
            evaluator.average_return(
                policy_mode=policy_mode,
                num_trajectories=NUM_TRAJECTORIES,
                max_steps=MAX_STEPS,
                seed=SEED,
                terminal_states=TERMINAL_STATES,
            )["policy"]
        ),
        f"{policy_mode}_v_x0": float(v_pi[planner.x0].detach().cpu().item()),
        f"{policy_mode}_v_gap": v_gap,
    }


def finite_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def run_candidate(params, problem_name, mdp, planner, dataset_path, device, d_star, v_star):
    alpha, eta, rho, T, theta_lr, theta_inner_steps, theta_mode, theta_lambda, D_theta = params
    start = time.perf_counter()
    row = base_row(params, problem_name, device)

    try:
        solver = make_solver(
            dataset_path=dataset_path,
            device=device,
            theta_lr=theta_lr,
            theta_inner_steps=theta_inner_steps,
            theta_mode=theta_mode,
            theta_lambda=theta_lambda,
            D_theta=D_theta,
        )
        evaluator = FOGASEvaluator(solver=solver, mdp=mdp, planner=planner)

        solver.run(
            alpha=alpha,
            eta=eta,
            rho=rho,
            T=T,
            D_theta=D_theta,
            theta_mode=theta_mode,
            theta_lr=theta_lr,
            theta_inner_steps=theta_inner_steps,
            theta_lambda=theta_lambda,
            policy_optimizer="adam",
            policy_gradient="exact",
            reinforce_samples=REINFORCE_SAMPLES,
            tqdm_print=False,
            verbose=False,
            state_weight_update="normal",
        )

        row.update(evaluate_policy(planner, evaluator, "solver", d_star, v_star))
        row.update(evaluate_policy(planner, evaluator, "greedy", d_star, v_star))

        diagnostics = solver.get_diagnostics() or []
        if diagnostics:
            final = diagnostics[-1]
            row.update(
                {
                    "final_total_loss": finite_float(final.get("total_loss")),
                    "final_policy_objective": finite_float(final.get("policy_objective")),
                    "final_beta_objective": finite_float(final.get("beta_objective")),
                    "final_q_objective": finite_float(final.get("q_objective")),
                    "final_theta_norm": finite_float(final.get("theta_norm")),
                    "final_policy_grad_norm": finite_float(final.get("policy_grad_norm")),
                    "final_beta_grad_norm": finite_float(final.get("beta_grad_norm")),
                    "final_theta_grad_norm": finite_float(final.get("theta_grad_norm")),
                }
            )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        row["elapsed_seconds"] = float(time.perf_counter() - start)

    return row


def run_candidate_worker(payload):
    params, problem_name, dataset_path_str, device_str, torch_threads = payload
    configure_worker_threads(torch_threads)
    set_seed(SEED)
    device = torch.device(device_str)
    mdp, planner = build_mdp(problem_name, device)
    d_star = planner.state_mu_star.detach().cpu()
    v_star = planner.v_star.detach().cpu()
    return run_candidate(
        params=params,
        problem_name=problem_name,
        mdp=mdp,
        planner=planner,
        dataset_path=Path(dataset_path_str),
        device=device,
        d_star=d_star,
        v_star=v_star,
    )


def all_candidates():
    theta_candidates = [
        ("reg_fixed", theta_lambda, None)
        for theta_lambda in THETA_LAMBDA_GRID
    ]

    return [
        (*base_params, *theta_params)
        for base_params, theta_params in itertools.product(
            itertools.product(
                ALPHA_GRID,
                ETA_GRID,
                RHO_GRID,
                T_GRID,
                THETA_LR_GRID,
                THETA_INNER_STEPS_GRID,
            ),
            theta_candidates,
        )
    ]


def theta_grid_size():
    return len(THETA_LAMBDA_GRID)


def theta_grid_description():
    return f"{len(THETA_LAMBDA_GRID)} reg_fixed theta_lambda values"


def total_grid_size():
    return (
        len(ALPHA_GRID)
        * len(ETA_GRID)
        * len(RHO_GRID)
        * len(T_GRID)
        * len(THETA_LR_GRID)
        * len(THETA_INNER_STEPS_GRID)
        * theta_grid_size()
    )


def run_grid_search(problem_name):
    if problem_name not in PROBLEMS:
        raise ValueError(f"Unknown problem {problem_name!r}. Expected one of {sorted(PROBLEMS)}")

    # Use 'spawn' to avoid CUDA context corruption when forking worker processes.
    # (CUDA and fork do not mix; 'spawn' starts each worker fresh.)
    multiprocessing.set_start_method("spawn", force=True)

    problem = PROBLEMS[problem_name]
    args = parse_args(problem["description"])
    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    configure_worker_threads(torch_threads)
    set_seed(SEED)

    # Build the list of devices to distribute workers across.
    if args.devices:
        device_list = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        device_list = [str(DEVICE)]

    dataset_path = problem["dataset_path"]
    output_csv = problem["output_csv"]
    best_csv = problem["best_csv"]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Devices: {device_list}")
    print(f"Problem: {problem_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Results: {output_csv}")
    print(f"Workers: {workers}")
    print(f"Torch threads per worker: {torch_threads}")

    candidates_all = all_candidates()
    if args.max_runs is not None:
        candidates_all = candidates_all[: max(0, int(args.max_runs))]

    results, completed = load_existing_results(args.resume, output_csv)
    candidates = [
        candidate
        for candidate in candidates_all
        if candidate_key(
            {
                "alpha": candidate[0],
                "eta": candidate[1],
                "rho": candidate[2],
                "T": candidate[3],
                "theta_lr": candidate[4],
                "theta_inner_steps": candidate[5],
                "theta_mode": candidate[6],
                "theta_lambda": candidate[7],
                "D_theta": candidate[8],
            }
        )
        not in completed
    ]

    print(f"Total grid size: {total_grid_size()}")
    print(f"Theta grid: {theta_grid_description()}")
    print(f"Candidates to run: {len(candidates)}")
    if args.resume:
        print(f"Resumed rows: {len(results)}")
    if not args.resume and output_csv.exists():
        print("Existing output will be overwritten because --resume was not set.")

    progress = not args.no_progress
    desc = f"FinalLinearSolver {problem_name} 10-grid tabular search"

    if workers == 1:
        # Build MDP in the main process only for the sequential path.
        mdp, planner = build_mdp(problem_name, DEVICE)
        d_star = planner.state_mu_star.detach().cpu()
        v_star = planner.v_star.detach().cpu()

        outer = tqdm(candidates, desc=desc, unit="run", disable=not progress)
        for run_idx, params in enumerate(outer, start=len(results) + 1):
            row = run_candidate(
                params=params,
                problem_name=problem_name,
                mdp=mdp,
                planner=planner,
                dataset_path=dataset_path,
                device=DEVICE,
                d_star=d_star,
                v_star=v_star,
            )
            row["run_idx"] = int(run_idx)
            results.append(row)
            save_results(results, output_csv, best_csv)

            if progress:
                outer.set_postfix(
                    {
                        "greedy_success": row["greedy_success_rate"],
                        "solver_success": row["solver_success_rate"],
                        "status": row["status"],
                    }
                )
    else:
        payloads = [
            (
                params,
                problem_name,
                str(dataset_path),
                device_list[i % len(device_list)],
                torch_threads,
            )
            for i, params in enumerate(candidates)
        ]
        next_run_idx = len(results) + 1
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_params = {
                executor.submit(run_candidate_worker, payload): payload[0]
                for payload in payloads
            }
            outer = tqdm(
                as_completed(future_to_params),
                total=len(future_to_params),
                desc=desc,
                unit="run",
                disable=not progress,
            )
            for future in outer:
                try:
                    row = future.result()
                except Exception as exc:
                    row = failed_worker_row(future_to_params[future], problem_name, exc, device=None)
                row["run_idx"] = int(next_run_idx)
                next_run_idx += 1
                results.append(row)
                save_results(results, output_csv, best_csv)

                if progress:
                    outer.set_postfix(
                        {
                            "greedy_success": row.get("greedy_success_rate", np.nan),
                            "solver_success": row.get("solver_success_rate", np.nan),
                            "status": row.get("status"),
                        }
                    )

    save_results(results, output_csv, best_csv)
    df = ordered_results_frame(results)
    ok_count = int((df["status"] == "ok").sum()) if not df.empty else 0
    failed_count = int((df["status"] == "failed").sum()) if not df.empty else 0

    print("\nGrid search complete.")
    print(f"Rows saved: {len(df)}")
    print(f"Successful runs: {ok_count}")
    print(f"Failed runs: {failed_count}")
    print(f"Output CSV: {output_csv}")
    if best_csv.exists():
        print(f"Best row CSV: {best_csv}")
        print("\nTop result:")
        print(pd.read_csv(best_csv).to_string(index=False))
