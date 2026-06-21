"""
FinalLinearSolver tabular theta-update ablation for deterministic and stochastic 5x5 grids.

The script writes CSV results after every completed candidate so interrupted
runs can be resumed with --resume. Candidate evaluation records only solver
success rate, greedy success rate, solver average reward, greedy average
reward, and elapsed runtime, plus the parameters needed to identify each run.
"""

from __future__ import annotations

import argparse
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
OUTPUT_CSV = RESULTS_DIR / "final_linear_5grid_tabular_theta_ablation.csv"
BEST_CSV = RESULTS_DIR / "final_linear_5grid_tabular_theta_ablation_best.csv"
STATS_CSV = RESULTS_DIR / "final_linear_5grid_tabular_theta_ablation_stats.csv"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

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

BASE_ALPHA = 1e-3
BASE_ETA = 1e-4
BASE_T = 1000
BASE_REINFORCE_SAMPLES = 4
BASE_THETA_INNER_STEPS = 40
BASE_THETA_START_MODE = "warm"
ZERO_THETA_START_MODE = "zero"
PROJECTION_D_THETA_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
THETA_LAMBDA_GRID = [
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    10.0,
]
THETA_INNER_STEPS_GRID = [1, 2, 5, 10, 20, 40, 80, 160]
FISHER_DAMPING = 1e-3
CG_ITERS = 50
CG_TOL = 1e-10

PROBLEMS = {
    "deterministic": {
        "dataset_path": DATASETS_DIR / "5grid.csv",
        "gamma": 0.99,
        "terminal_states": {GOAL_GRID},
        "stochastic": False,
        "theta_lambda": 1e-3,
        "theta_lr": 3e-1,
        "baseline_rho": 0.05,
    },
    "stochastic": {
        "dataset_path": DATASETS_DIR / "5grid_stochastic.csv",
        "gamma": 0.9,
        "terminal_states": {GOAL_GRID, PIT_GRID},
        "stochastic": True,
        "theta_lambda": 1e-4,
        "theta_lr": 3e-2,
        "baseline_rho": 1.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FinalLinearSolver theta-update ablations on 5x5 tabular grids."
    )
    parser.add_argument(
        "--problem",
        choices=["deterministic", "stochastic", "both"],
        default="both",
        help="Which 5x5 problem to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip successful parameter combinations already present in the output CSV.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit the number of candidates after problem selection. Useful for smoke tests.",
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
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
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

    def transition_fn(s, a):
        probs = torch.zeros(N, dtype=torch.float64)
        if stochastic:
            for sp, prob in stochastic_transition_probs(s, a, terminal_states).items():
                probs[sp] = prob
        else:
            probs[move_deterministic(s, a, terminal_states)] = 1.0
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
        gamma=float(problem["gamma"]),
        x0=X0,
        reward_fn=reward_fn,
        transition_fn=transition_fn,
        terminal_states=list(terminal_states),
    ).to(device)
    planner = Planner(mdp).to(device)
    return mdp, planner


def make_solver(problem_name, dataset_path, device, candidate):
    problem = PROBLEMS[problem_name]
    u_features = TabularFeatures(N, A)
    q_features = TabularFeatures(N, A)
    policy_features = TabularFeatures(N, A)

    return FinalLinearSolver(
        n_states=N,
        n_actions=A,
        gamma=float(problem["gamma"]),
        x0=X0,
        csv_path=str(dataset_path),
        u_function=LinearFunction(u_features),
        q_function=LinearQFunction(q_features),
        policy_features=policy_features,
        seed=SEED,
        device=device,
        D_theta=candidate["D_theta"],
        theta_include_beta_cov=False,
        theta_mode=candidate["theta_mode"],
        theta_lambda=candidate["theta_lambda"],
        theta_optimizer="adam",
        theta_inner_steps=int(candidate["theta_inner_steps"]),
        theta_lr=float(problem["theta_lr"]),
        theta_start_mode=candidate["theta_start_mode"],
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
        str(row["problem"]),
        str(row["ablation"]),
        str(row["theta_mode"]),
        str(row["theta_start_mode"]),
        optional_float_key(row.get("theta_lambda")),
        optional_float_key(row.get("D_theta")),
        int(row["theta_inner_steps"]),
    )


def selected_problem_names(problem_arg):
    if problem_arg == "both":
        return ["deterministic", "stochastic"]
    return [problem_arg]


def make_candidate(
    problem_name,
    ablation,
    theta_mode,
    theta_lambda=None,
    D_theta=None,
    theta_inner_steps=BASE_THETA_INNER_STEPS,
    theta_start_mode=BASE_THETA_START_MODE,
):
    problem = PROBLEMS[problem_name]
    return {
        "problem": problem_name,
        "ablation": ablation,
        "beta_update": "fogas_full",
        "T": int(BASE_T),
        "alpha": float(BASE_ALPHA),
        "eta": float(BASE_ETA),
        "rho": float(problem["baseline_rho"]),
        "theta_mode": theta_mode,
        "theta_lambda": None if theta_lambda is None else float(theta_lambda),
        "D_theta": None if D_theta is None else float(D_theta),
        "theta_lr": float(problem["theta_lr"]),
        "theta_inner_steps": int(theta_inner_steps),
        "theta_start_mode": theta_start_mode,
        "policy_optimizer": "adam",
        "policy_gradient": "exact",
        "reinforce_samples": int(BASE_REINFORCE_SAMPLES),
        "fisher_damping": float(FISHER_DAMPING),
        "cg_iters": int(CG_ITERS),
        "cg_tol": float(CG_TOL),
        "state_weight_update": "normal",
        "dataset_path": str(problem["dataset_path"]),
    }


def all_candidates(problem_names):
    candidates = []
    for problem_name in problem_names:
        problem = PROBLEMS[problem_name]
        default_theta_lambda = float(problem["theta_lambda"])

        candidates.append(
            make_candidate(
                problem_name=problem_name,
                ablation="reg_fixed_baseline",
                theta_mode="reg_fixed",
                theta_lambda=default_theta_lambda,
                theta_inner_steps=BASE_THETA_INNER_STEPS,
                theta_start_mode=BASE_THETA_START_MODE,
            )
        )

        candidates.append(
            make_candidate(
                problem_name=problem_name,
                ablation="reg_fixed_start_zero",
                theta_mode="reg_fixed",
                theta_lambda=default_theta_lambda,
                theta_inner_steps=BASE_THETA_INNER_STEPS,
                theta_start_mode=ZERO_THETA_START_MODE,
            )
        )

        for D_theta in PROJECTION_D_THETA_GRID:
            candidates.append(
                make_candidate(
                    problem_name=problem_name,
                    ablation="projection_dtheta",
                    theta_mode="projection",
                    D_theta=D_theta,
                    theta_inner_steps=BASE_THETA_INNER_STEPS,
                    theta_start_mode=BASE_THETA_START_MODE,
                )
            )

        for theta_lambda in THETA_LAMBDA_GRID:
            candidates.append(
                make_candidate(
                    problem_name=problem_name,
                    ablation="reg_fixed_lambda",
                    theta_mode="reg_fixed",
                    theta_lambda=theta_lambda,
                    theta_inner_steps=BASE_THETA_INNER_STEPS,
                    theta_start_mode=BASE_THETA_START_MODE,
                )
            )

        for theta_inner_steps in THETA_INNER_STEPS_GRID:
            candidates.append(
                make_candidate(
                    problem_name=problem_name,
                    ablation="reg_fixed_inner_steps",
                    theta_mode="reg_fixed",
                    theta_lambda=default_theta_lambda,
                    theta_inner_steps=theta_inner_steps,
                    theta_start_mode=BASE_THETA_START_MODE,
                )
            )

        for theta_lambda in THETA_LAMBDA_GRID:
            candidates.append(
                make_candidate(
                    problem_name=problem_name,
                    ablation="zero_start_lambda",
                    theta_mode="reg_fixed",
                    theta_lambda=theta_lambda,
                    theta_inner_steps=BASE_THETA_INNER_STEPS,
                    theta_start_mode=ZERO_THETA_START_MODE,
                )
            )

        for theta_inner_steps in THETA_INNER_STEPS_GRID:
            candidates.append(
                make_candidate(
                    problem_name=problem_name,
                    ablation="zero_start_inner_steps",
                    theta_mode="reg_fixed",
                    theta_lambda=default_theta_lambda,
                    theta_inner_steps=theta_inner_steps,
                    theta_start_mode=ZERO_THETA_START_MODE,
                )
            )

    return candidates


def load_existing_results(resume):
    if not resume or not OUTPUT_CSV.exists():
        return [], set()

    df = pd.read_csv(OUTPUT_CSV)
    rows = df.to_dict("records")
    if "status" in df.columns:
        completed_df = df[df["status"] == "ok"].copy()
    else:
        completed_df = df
    completed = {candidate_key(row) for row in completed_df.to_dict("records")}
    return rows, completed


def blank_metrics():
    return {
        "solver_success_rate": np.nan,
        "greedy_success_rate": np.nan,
        "solver_avg_reward": np.nan,
        "greedy_avg_reward": np.nan,
        "elapsed_seconds": np.nan,
    }


def base_row(candidate, device, status="ok", error=""):
    row = dict(candidate)
    row.update(
        {
            "theta_optimizer": "adam",
            "theta_include_beta_cov": False,
            "num_trajectories": int(NUM_TRAJECTORIES),
            "max_steps": int(MAX_STEPS),
            "seed": int(SEED),
            "device": str(device),
            "status": status,
            "error": error,
        }
    )
    row.update(blank_metrics())
    return row


def greedy_policy(pi):
    pi = pi.to(dtype=torch.float64)
    greedy = torch.zeros_like(pi)
    best_actions = torch.argmax(pi, dim=1)
    greedy[torch.arange(pi.shape[0], device=pi.device), best_actions] = 1.0
    return greedy


def evaluate_policy_rollouts(mdp, pi, terminal_states, seed):
    pi = pi.to(dtype=torch.float64, device=mdp.r.device)
    terminal_states = {int(state) for state in terminal_states}
    rewards = []
    successes = 0

    for idx in range(NUM_TRAJECTORIES):
        current_seed = None if seed is None else int(seed) + idx
        if current_seed is not None:
            random.seed(current_seed)
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(current_seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(current_seed)

        state = int(mdp.x0)
        total_reward = 0.0
        reached_goal = False

        for _step in range(MAX_STEPS):
            action_probs = pi[state]
            prob_sum = action_probs.sum()
            if prob_sum <= 0:
                raise ValueError(f"Policy probabilities at state {state} must have positive mass.")
            action_probs = action_probs / prob_sum
            action = int(torch.multinomial(action_probs, num_samples=1).item())

            row_idx = state * mdp.A + action
            reward = mdp.r[row_idx]
            total_reward += float(reward.item() if isinstance(reward, torch.Tensor) else reward)

            transition_probs = mdp.P[row_idx].to(dtype=torch.float64, device=mdp.r.device)
            next_state = int(torch.multinomial(transition_probs, num_samples=1).item())
            reached_goal = next_state == GOAL_GRID
            if next_state in terminal_states:
                break
            state = next_state

        rewards.append(total_reward)
        successes += int(reached_goal)

    return {
        "success_rate": float(successes / NUM_TRAJECTORIES) if NUM_TRAJECTORIES else 0.0,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


def finite_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def run_candidate(candidate, mdp, planner, dataset_path, device):
    del planner
    start = time.perf_counter()
    row = base_row(candidate, device)
    problem_name = candidate["problem"]
    terminal_states = set(PROBLEMS[problem_name]["terminal_states"])

    try:
        solver = make_solver(
            problem_name=problem_name,
            dataset_path=dataset_path,
            device=device,
            candidate=candidate,
        )
        solver.run(
            alpha=candidate["alpha"],
            eta=candidate["eta"],
            rho=candidate["rho"],
            T=candidate["T"],
            D_theta=candidate["D_theta"],
            theta_mode=candidate["theta_mode"],
            theta_lambda=candidate["theta_lambda"],
            theta_inner_steps=candidate["theta_inner_steps"],
            theta_start_mode=candidate["theta_start_mode"],
            policy_optimizer=candidate["policy_optimizer"],
            policy_gradient=candidate["policy_gradient"],
            reinforce_samples=candidate["reinforce_samples"],
            fisher_damping=candidate["fisher_damping"],
            cg_iters=candidate["cg_iters"],
            cg_tol=candidate["cg_tol"],
            tqdm_print=False,
            verbose=False,
            state_weight_update=candidate["state_weight_update"],
            beta_update=candidate["beta_update"],
        )

        solver_stats = evaluate_policy_rollouts(
            mdp=mdp,
            pi=solver.pi,
            terminal_states=terminal_states,
            seed=SEED,
        )
        greedy_stats = evaluate_policy_rollouts(
            mdp=mdp,
            pi=greedy_policy(solver.pi),
            terminal_states=terminal_states,
            seed=SEED,
        )

        row.update(
            {
                "solver_success_rate": finite_float(solver_stats["success_rate"]),
                "greedy_success_rate": finite_float(greedy_stats["success_rate"]),
                "solver_avg_reward": finite_float(solver_stats["avg_reward"]),
                "greedy_avg_reward": finite_float(greedy_stats["avg_reward"]),
            }
        )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        row["elapsed_seconds"] = float(time.perf_counter() - start)

    return row


def run_candidate_worker(payload):
    candidate, device_str, torch_threads = payload
    configure_worker_threads(torch_threads)
    set_seed(SEED)
    device = torch.device(device_str)
    mdp, planner = build_mdp(candidate["problem"], device)
    return run_candidate(
        candidate=candidate,
        mdp=mdp,
        planner=planner,
        dataset_path=Path(candidate["dataset_path"]),
        device=device,
    )


def failed_worker_row(candidate, exc):
    return base_row(candidate, DEVICE, status="failed", error=repr(exc))


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
            "elapsed_seconds",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_stats_frame(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df

    metrics = [
        "solver_success_rate",
        "greedy_success_rate",
        "solver_avg_reward",
        "greedy_avg_reward",
    ]
    rows = []
    group_columns = ["problem", "ablation", "theta_mode", "theta_start_mode"]
    for group_key, group in df.groupby(group_columns, dropna=False):
        problem, ablation, theta_mode, theta_start_mode = group_key
        ok = group[group["status"] == "ok"].copy()
        row = {
            "problem": problem,
            "ablation": ablation,
            "theta_mode": theta_mode,
            "theta_start_mode": theta_start_mode,
            "count": int(len(group)),
            "ok_count": int(len(ok)),
            "failed_count": int((group["status"] == "failed").sum()),
            "elapsed_seconds_mean": float(ok["elapsed_seconds"].mean()) if not ok.empty else np.nan,
        }
        for metric in metrics:
            values = ok[metric] if metric in ok.columns else pd.Series(dtype=float)
            row[f"{metric}_best"] = float(values.max()) if not values.empty else np.nan
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        by=["problem", "ablation", "greedy_success_rate_best", "solver_success_rate_best"],
        ascending=[True, True, False, False],
        na_position="last",
    )


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = ordered_results_frame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    if not df.empty:
        ok = df[df["status"] == "ok"]
        if not ok.empty:
            ok.head(1).to_csv(BEST_CSV, index=False)

    stats = build_stats_frame(results)
    if not stats.empty:
        stats.to_csv(STATS_CSV, index=False)


def validate_datasets(problem_names):
    for problem_name in problem_names:
        dataset_path = PROBLEMS[problem_name]["dataset_path"]
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")


def run_grid_search():
    args = parse_args()
    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    configure_worker_threads(torch_threads)
    set_seed(SEED)

    problem_names = selected_problem_names(args.problem)
    validate_datasets(problem_names)

    candidates_all = all_candidates(problem_names)
    if args.max_runs is not None:
        candidates_all = candidates_all[: max(0, int(args.max_runs))]

    results, completed = load_existing_results(args.resume)
    candidates = [
        candidate
        for candidate in candidates_all
        if candidate_key(candidate) not in completed
    ]

    print(f"Using device: {DEVICE}")
    print(f"Problems: {', '.join(problem_names)}")
    print(f"Results: {OUTPUT_CSV}")
    print(f"Stats: {STATS_CSV}")
    print(f"Workers: {workers}")
    print(f"Torch threads per worker: {torch_threads}")
    print(f"Total candidate grid size: {len(candidates_all)}")
    print(f"Candidates to run: {len(candidates)}")
    if args.resume:
        print(f"Resumed rows: {len(results)}")
    if not args.resume and OUTPUT_CSV.exists():
        print("Existing output will be overwritten because --resume was not set.")

    progress = not args.no_progress
    desc = "FinalLinearSolver 5-grid tabular theta ablation"

    if workers == 1:
        mdp_by_problem = {}
        outer = tqdm(candidates, desc=desc, unit="run", disable=not progress)
        for run_idx, candidate in enumerate(outer, start=len(results) + 1):
            problem_name = candidate["problem"]
            if problem_name not in mdp_by_problem:
                mdp_by_problem[problem_name] = build_mdp(problem_name, DEVICE)
            mdp, planner = mdp_by_problem[problem_name]
            row = run_candidate(
                candidate=candidate,
                mdp=mdp,
                planner=planner,
                dataset_path=Path(candidate["dataset_path"]),
                device=DEVICE,
            )
            row["run_idx"] = int(run_idx)
            results.append(row)
            save_results(results)

            if progress:
                outer.set_postfix(
                    {
                        "problem": row["problem"],
                        "ablation": row["ablation"],
                        "theta": row["theta_mode"],
                        "start": row["theta_start_mode"],
                        "greedy_success": row["greedy_success_rate"],
                        "status": row["status"],
                    }
                )
    else:
        payloads = [(candidate, str(DEVICE), torch_threads) for candidate in candidates]
        next_run_idx = len(results) + 1
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_candidate = {
                executor.submit(run_candidate_worker, payload): payload[0]
                for payload in payloads
            }
            outer = tqdm(
                as_completed(future_to_candidate),
                total=len(future_to_candidate),
                desc=desc,
                unit="run",
                disable=not progress,
            )
            for future in outer:
                candidate = future_to_candidate[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = failed_worker_row(candidate, exc)
                row["run_idx"] = int(next_run_idx)
                next_run_idx += 1
                results.append(row)
                save_results(results)

                if progress:
                    outer.set_postfix(
                        {
                            "problem": row.get("problem"),
                            "ablation": row.get("ablation"),
                            "theta": row.get("theta_mode"),
                            "start": row.get("theta_start_mode"),
                            "greedy_success": row.get("greedy_success_rate", np.nan),
                            "status": row.get("status"),
                        }
                    )

    save_results(results)
    df = ordered_results_frame(results)
    ok_count = int((df["status"] == "ok").sum()) if not df.empty else 0
    failed_count = int((df["status"] == "failed").sum()) if not df.empty else 0

    print("\nAblation complete.")
    print(f"Rows saved: {len(df)}")
    print(f"Successful runs: {ok_count}")
    print(f"Failed runs: {failed_count}")
    print(f"Output CSV: {OUTPUT_CSV}")
    if BEST_CSV.exists():
        print(f"Best row CSV: {BEST_CSV}")
        print("\nTop result:")
        print(pd.read_csv(BEST_CSV).to_string(index=False))
    if STATS_CSV.exists():
        print(f"Stats CSV: {STATS_CSV}")


if __name__ == "__main__":
    run_grid_search()
