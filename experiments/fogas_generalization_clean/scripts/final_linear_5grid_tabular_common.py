"""
Shared FinalLinearSolver tabular grid-search utilities for 5x5 generalization grids.

The two entrypoint scripts choose the deterministic or stochastic 5x5 problem.
This module owns the common grid, tabular features, metrics, resume logic, and
worker execution so both searches produce comparable CSV files.
"""

from __future__ import annotations

import argparse
import itertools
import math
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


def find_root(current_path, marker="setup.py"):
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    return current_path


PROJECT_ROOT = find_root(Path(__file__).resolve())
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets_clean" / "generalization"
RESULTS_DIR = PROJECT_ROOT / "data" / "results_clean" / "generalization"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from rl_methods.fogas_clean import FOGASEvaluator  # noqa: E402
from rl_methods.fogas_generalization_clean import (  # noqa: E402
    FinalLinearSolver,
    LinearFunction,
    LinearQFunction,
    TabularFeatures,
)
from rl_methods.mdp_clean import DiscreteMDP, Planner  # noqa: E402


SEED = 42
NUM_TRAJECTORIES = 100
MAX_STEPS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

STATES = torch.arange(25, dtype=torch.long)
ACTIONS = torch.arange(4, dtype=torch.long)
N = len(STATES)
A = len(ACTIONS)
GRID_SIZE = 5
X0 = 0
GOAL_GRID = 24
PIT_GRID = 18
WALL_STATES = {6, 7, 12}
INTENDED_PROB = 0.8

T_GRID = [1000, 1500, 3000]
THETA_INNER_STEPS_GRID = [10, 20]

COMMON_ALPHA_GRID = [1e-4, 3e-4, 1e-3, 3e-3]
COMMON_ETA_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
COMMON_THETA_LR_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]

FISHER_DAMPING = 1e-3
CG_ITERS = 50
CG_TOL = 1e-10
MAX_TOTAL_CANDIDATES = 4750

PROBLEMS = {
    "deterministic": {
        "description": "deterministic 5x5 FinalLinearSolver tabular grid search",
        "dataset_path": DATASETS_DIR / "5grid.csv",
        "output_csv": RESULTS_DIR / "final_linear_5grid_tabular_grid_search.csv",
        "best_csv": RESULTS_DIR / "final_linear_5grid_tabular_grid_search_best.csv",
        "gamma": 0.99,
        "terminal_states": {GOAL_GRID},
        "stochastic": False,
        "candidate_cap": 2250,
        "alpha_grid": COMMON_ALPHA_GRID,
        "eta_grid": COMMON_ETA_GRID,
        "rho_grid": [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5],
        "theta_lr_grid": COMMON_THETA_LR_GRID,
        "theta_lambda_grid": [1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "interaction_rho_grid": [0.003, 0.01, 0.03, 0.05, 0.1, 0.2],
        "baseline": {
            "alpha": 1e-3,
            "eta": 1e-4,
            "rho": 0.05,
            "T": 1000,
            "theta_lr": 3e-1,
            "theta_inner_steps": 20,
            "theta_lambda": 1e-3,
        },
    },
    "stochastic": {
        "description": "stochastic 5x5 FinalLinearSolver tabular grid search",
        "dataset_path": DATASETS_DIR / "5grid_stochastic.csv",
        "output_csv": RESULTS_DIR / "final_linear_5grid_stochastic_tabular_grid_search.csv",
        "best_csv": RESULTS_DIR / "final_linear_5grid_stochastic_tabular_grid_search_best.csv",
        "gamma": 0.9,
        "terminal_states": {GOAL_GRID, PIT_GRID},
        "stochastic": True,
        "candidate_cap": 2500,
        "alpha_grid": COMMON_ALPHA_GRID,
        "eta_grid": COMMON_ETA_GRID,
        "rho_grid": [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
        "theta_lr_grid": COMMON_THETA_LR_GRID,
        "theta_lambda_grid": [1e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 1e-5, 1e-4],
        "interaction_rho_grid": [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0],
        "baseline": {
            "alpha": 1e-3,
            "eta": 1e-4,
            "rho": 1.0,
            "T": 1500,
            "theta_lr": 3e-2,
            "theta_inner_steps": 20,
            "theta_lambda": 1e-7,
        },
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
        help="Skip successful parameter combinations already present in the output CSV.",
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
        "--count-only",
        action="store_true",
        help="Print generated candidate counts and exit without running solvers.",
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


def move_deterministic(s, a, terminal_states):
    s = int(s)
    a = int(a)

    if s in terminal_states:
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


def stochastic_transition_probs(s, a, terminal_states, intended_prob=INTENDED_PROB):
    s = int(s)
    a = int(a)
    probs_by_state = {}

    for candidate_a in range(A):
        prob = (1.0 - intended_prob) / A
        if candidate_a == a:
            prob += intended_prob

        sp = move_deterministic(s, candidate_a, terminal_states)
        probs_by_state[sp] = probs_by_state.get(sp, 0.0) + prob

    return probs_by_state


def reward_from_next_state(sp, stochastic):
    sp = int(sp)
    if sp == GOAL_GRID:
        return 1.0
    if stochastic and sp == PIT_GRID:
        return -1.0
    return -0.1 if stochastic else -0.01


def build_mdp(problem_name, device):
    problem = PROBLEMS[problem_name]
    terminal_states = set(problem["terminal_states"])
    stochastic = bool(problem["stochastic"])
    gamma = float(problem["gamma"])

    def transition_fn(s, a):
        probs = torch.zeros(N, dtype=torch.float64)
        if stochastic:
            for sp, prob in stochastic_transition_probs(s, a, terminal_states).items():
                probs[sp] = prob
        else:
            sp = move_deterministic(s, a, terminal_states)
            probs[sp] = 1.0
        return probs

    def reward_fn(s, a):
        if stochastic:
            return sum(
                prob * reward_from_next_state(sp, stochastic=True)
                for sp, prob in stochastic_transition_probs(s, a, terminal_states).items()
            )

        sp = move_deterministic(s, a, terminal_states)
        return reward_from_next_state(sp, stochastic=False)

    mdp = DiscreteMDP(
        states=STATES,
        actions=ACTIONS,
        gamma=gamma,
        x0=X0,
        reward_fn=reward_fn,
        transition_fn=transition_fn,
        terminal_states=list(terminal_states),
    ).to(device)
    planner = Planner(mdp).to(device)
    return mdp, planner


def make_solver(gamma, dataset_path, device, theta_lr, theta_inner_steps, theta_lambda):
    u_features = TabularFeatures(N, A)
    q_features = TabularFeatures(N, A)
    policy_features = TabularFeatures(N, A)

    return FinalLinearSolver(
        n_states=N,
        n_actions=A,
        gamma=gamma,
        x0=X0,
        csv_path=str(dataset_path),
        u_function=LinearFunction(u_features),
        q_function=LinearQFunction(q_features),
        policy_features=policy_features,
        seed=SEED,
        device=device,
        theta_include_beta_cov=False,
        theta_mode="reg_fixed",
        theta_lambda=theta_lambda,
        theta_optimizer="adam",
        theta_inner_steps=theta_inner_steps,
        theta_lr=theta_lr,
        theta_start_mode="warm",
        beta_update="fogas_full",
    )


def candidate_key(row):
    return (
        float(row["alpha"]),
        float(row["eta"]),
        float(row["rho"]),
        int(row["T"]),
        float(row["theta_lr"]),
        int(row["theta_inner_steps"]),
        float(row["theta_lambda"]),
    )


def load_existing_results(resume, output_csv):
    if not resume or not output_csv.exists():
        return [], set()

    df = pd.read_csv(output_csv)
    if "status" in df.columns:
        completed_df = df[df["status"] == "ok"].copy()
    else:
        completed_df = df
    rows = df.to_dict("records")
    completed = {candidate_key(row) for row in completed_df.to_dict("records")}
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
            "greedy_v_gap",
            "elapsed_seconds",
        ],
        ascending=[False, False, False, False, True, True],
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
        "solver_avg_return": np.nan,
        "solver_v_x0": np.nan,
        "solver_v_gap": np.nan,
        "greedy_success_rate": np.nan,
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
    alpha, eta, rho, T, theta_lr, theta_inner_steps, theta_lambda = params
    problem = PROBLEMS[problem_name]
    row = {
        "alpha": float(alpha),
        "eta": float(eta),
        "rho": float(rho),
        "T": int(T),
        "theta_lr": float(theta_lr),
        "theta_inner_steps": int(theta_inner_steps),
        "theta_lambda": float(theta_lambda),
        "theta_mode": "reg_fixed",
        "theta_optimizer": "adam",
        "theta_start_mode": "warm",
        "theta_include_beta_cov": False,
        "beta_update": "fogas_full",
        "policy_optimizer": "adam",
        "policy_gradient": "exact",
        "reinforce_samples": 4,
        "fisher_damping": float(FISHER_DAMPING),
        "cg_iters": int(CG_ITERS),
        "cg_tol": float(CG_TOL),
        "state_weight_update": "normal",
        "dataset_path": str(problem["dataset_path"]),
        "num_trajectories": int(NUM_TRAJECTORIES),
        "max_steps": int(MAX_STEPS),
        "seed": int(SEED),
        "device": str(device),
        "status": status,
        "error": error,
        "elapsed_seconds": np.nan,
    }
    row.update(blank_metrics())
    return row


def failed_worker_row(params, problem_name, exc):
    return base_row(params, problem_name, DEVICE, status="failed", error=repr(exc))


def evaluate_policy(planner, evaluator, policy_mode, terminal_states, d_star, v_star):
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
                terminal_states=terminal_states,
            )["policy"]
        ),
        f"{policy_mode}_avg_return": float(
            evaluator.average_return(
                policy_mode=policy_mode,
                num_trajectories=NUM_TRAJECTORIES,
                max_steps=MAX_STEPS,
                seed=SEED,
                terminal_states=terminal_states,
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
    alpha, eta, rho, T, theta_lr, theta_inner_steps, theta_lambda = params
    start = time.perf_counter()
    terminal_states = set(PROBLEMS[problem_name]["terminal_states"])
    row = base_row(params, problem_name, device)

    try:
        solver = make_solver(
            gamma=mdp.gamma,
            dataset_path=dataset_path,
            device=device,
            theta_lr=theta_lr,
            theta_inner_steps=theta_inner_steps,
            theta_lambda=theta_lambda,
        )
        evaluator = FOGASEvaluator(solver=solver, mdp=mdp, planner=planner)

        solver.run(
            alpha=alpha,
            eta=eta,
            rho=rho,
            T=T,
            theta_lr=theta_lr,
            theta_inner_steps=theta_inner_steps,
            theta_lambda=theta_lambda,
            policy_optimizer="adam",
            policy_gradient="exact",
            reinforce_samples=4,
            fisher_damping=FISHER_DAMPING,
            cg_iters=CG_ITERS,
            cg_tol=CG_TOL,
            tqdm_print=False,
            verbose=False,
            state_weight_update="normal",
        )

        row.update(evaluate_policy(planner, evaluator, "solver", terminal_states, d_star, v_star))
        row.update(evaluate_policy(planner, evaluator, "greedy", terminal_states, d_star, v_star))

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


def params_from_dict(values):
    return (
        float(values["alpha"]),
        float(values["eta"]),
        float(values["rho"]),
        int(values["T"]),
        float(values["theta_lr"]),
        int(values["theta_inner_steps"]),
        float(values["theta_lambda"]),
    )


def values_by_closeness(values, center):
    def sort_key(value):
        value = float(value)
        center_float = float(center)
        if value > 0 and center_float > 0:
            distance = abs(math.log10(value) - math.log10(center_float))
        else:
            distance = abs(value - center_float)
        return (distance, value)

    return sorted(values, key=sort_key)


def append_candidate(candidates, seen, values):
    params = params_from_dict(values)
    if params not in seen:
        candidates.append(params)
        seen.add(params)


def all_candidates(problem_name):
    problem = PROBLEMS[problem_name]
    baseline = problem["baseline"]
    alpha_grid = problem["alpha_grid"]
    eta_grid = problem["eta_grid"]
    rho_grid = problem["rho_grid"]
    theta_lr_grid = problem["theta_lr_grid"]
    theta_lambda_grid = problem["theta_lambda_grid"]
    interaction_rho_grid = problem["interaction_rho_grid"]

    candidates = []
    seen = set()

    append_candidate(candidates, seen, baseline)

    for rho, eta, T, theta_inner_steps in itertools.product(
        values_by_closeness(rho_grid, baseline["rho"]),
        values_by_closeness(eta_grid, baseline["eta"]),
        values_by_closeness(T_GRID, baseline["T"]),
        values_by_closeness(THETA_INNER_STEPS_GRID, baseline["theta_inner_steps"]),
    ):
        append_candidate(
            candidates,
            seen,
            {
                **baseline,
                "rho": rho,
                "eta": eta,
                "T": T,
                "theta_inner_steps": theta_inner_steps,
            },
        )

    for theta_lr, theta_lambda, T, theta_inner_steps in itertools.product(
        values_by_closeness(theta_lr_grid, baseline["theta_lr"]),
        values_by_closeness(theta_lambda_grid, baseline["theta_lambda"]),
        values_by_closeness(T_GRID, baseline["T"]),
        values_by_closeness(THETA_INNER_STEPS_GRID, baseline["theta_inner_steps"]),
    ):
        append_candidate(
            candidates,
            seen,
            {
                **baseline,
                "theta_lr": theta_lr,
                "theta_lambda": theta_lambda,
                "T": T,
                "theta_inner_steps": theta_inner_steps,
            },
        )

    for alpha, eta, rho, T in itertools.product(
        values_by_closeness(alpha_grid, baseline["alpha"]),
        values_by_closeness(eta_grid, baseline["eta"]),
        values_by_closeness(rho_grid, baseline["rho"]),
        values_by_closeness(T_GRID, baseline["T"]),
    ):
        append_candidate(
            candidates,
            seen,
            {
                **baseline,
                "alpha": alpha,
                "eta": eta,
                "rho": rho,
                "T": T,
                "theta_inner_steps": 20,
            },
        )

    for rho, theta_lr, theta_lambda, T, theta_inner_steps in itertools.product(
        values_by_closeness(interaction_rho_grid, baseline["rho"]),
        values_by_closeness(theta_lr_grid, baseline["theta_lr"]),
        values_by_closeness(theta_lambda_grid, baseline["theta_lambda"]),
        values_by_closeness(T_GRID, baseline["T"]),
        values_by_closeness(THETA_INNER_STEPS_GRID, baseline["theta_inner_steps"]),
    ):
        append_candidate(
            candidates,
            seen,
            {
                **baseline,
                "rho": rho,
                "theta_lr": theta_lr,
                "theta_lambda": theta_lambda,
                "T": T,
                "theta_inner_steps": theta_inner_steps,
            },
        )

    cap = int(problem["candidate_cap"])
    return candidates[:cap]


def uncapped_grid_size(problem_name):
    problem = PROBLEMS[problem_name]
    return (
        1
        + len(problem["rho_grid"]) * len(problem["eta_grid"]) * len(T_GRID) * len(THETA_INNER_STEPS_GRID)
        + len(problem["theta_lr_grid"]) * len(problem["theta_lambda_grid"]) * len(T_GRID) * len(THETA_INNER_STEPS_GRID)
        + len(problem["alpha_grid"]) * len(problem["eta_grid"]) * len(problem["rho_grid"]) * len(T_GRID)
        + len(problem["interaction_rho_grid"])
        * len(problem["theta_lr_grid"])
        * len(problem["theta_lambda_grid"])
        * len(T_GRID)
        * len(THETA_INNER_STEPS_GRID)
    )


def total_grid_size(problem_name):
    return len(all_candidates(problem_name))


def print_candidate_counts():
    total = 0
    for problem_name in PROBLEMS:
        count = total_grid_size(problem_name)
        total += count
        print(
            f"{problem_name}: {count} candidates "
            f"(uncapped families: {uncapped_grid_size(problem_name)}, cap: {PROBLEMS[problem_name]['candidate_cap']})"
        )
    print(f"total: {total} candidates (limit: {MAX_TOTAL_CANDIDATES})")


def run_grid_search(problem_name):
    if problem_name not in PROBLEMS:
        raise ValueError(f"Unknown problem {problem_name!r}. Expected one of {sorted(PROBLEMS)}")

    problem = PROBLEMS[problem_name]
    args = parse_args(problem["description"])
    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    configure_worker_threads(torch_threads)
    set_seed(SEED)

    if args.count_only:
        print_candidate_counts()
        return

    dataset_path = problem["dataset_path"]
    output_csv = problem["output_csv"]
    best_csv = problem["best_csv"]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Using device: {DEVICE}")
    print(f"Problem: {problem_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Results: {output_csv}")
    print(f"Workers: {workers}")
    print(f"Torch threads per worker: {torch_threads}")

    mdp, planner = build_mdp(problem_name, DEVICE)
    d_star = planner.state_mu_star.detach().cpu()
    v_star = planner.v_star.detach().cpu()

    candidates_all = all_candidates(problem_name)
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
                "theta_lambda": candidate[6],
            }
        )
        not in completed
    ]

    total_count = sum(total_grid_size(name) for name in PROBLEMS)
    if total_count > MAX_TOTAL_CANDIDATES:
        raise RuntimeError(f"Generated {total_count} total candidates, above {MAX_TOTAL_CANDIDATES}.")

    print(f"Total capped grid size for this problem: {total_grid_size(problem_name)}")
    print(f"Total capped grid size across both problems: {total_count}")
    print(f"Candidates to run: {len(candidates)}")
    if args.resume:
        print(f"Resumed rows: {len(results)}")
    if not args.resume and output_csv.exists():
        print("Existing output will be overwritten because --resume was not set.")

    progress = not args.no_progress
    desc = f"FinalLinearSolver {problem_name} 5-grid tabular search"

    if workers == 1:
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
                        "greedy_return": row["greedy_avg_return"],
                        "status": row["status"],
                    }
                )
    else:
        payloads = [
            (params, problem_name, str(dataset_path), str(DEVICE), torch_threads)
            for params in candidates
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
                    row = failed_worker_row(future_to_params[future], problem_name, exc)
                row["run_idx"] = int(next_run_idx)
                next_run_idx += 1
                results.append(row)
                save_results(results, output_csv, best_csv)

                if progress:
                    outer.set_postfix(
                        {
                            "greedy_success": row.get("greedy_success_rate", np.nan),
                            "greedy_return": row.get("greedy_avg_return", np.nan),
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
