"""Shared AlgaeDICE parameter-search infrastructure for the 10 x 10 grids.

Scientific role
---------------
This helper keeps AlgaeDICE parameter selection aligned with the Generalized
FOGAS search: the deterministic and stochastic entry points share MDP
construction, fixed datasets, evaluator metrics, checkpointing, and worker
handling. Only the AlgaeDICE-specific objective parameters and diagnostics
differ. The deterministic selection feeds the thesis partial-coverage table;
the stochastic selection is an additional experiment.

Inputs and outputs
------------------
Problem-specific settings choose the deterministic or stochastic 10-grid CSV
under ``data/datasets/generalization``. Candidate and best-row result tables
are written to ``data/results/generalization/hyperparam_grids/10grid``.

This helper is not executed directly. Run either
``grid_search_primal_algaedice_10grid_tabular.py`` entry point from the
repository root. Both expose smoke-test, resume, worker, and device controls.
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
    LinearQFunction,
    PrimalAlgaeDICESolver,
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

CLOSED_FORM_ALPHA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
CLOSED_FORM_ACTOR_LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
CLOSED_FORM_RIDGE_GRID = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
CLOSED_FORM_T_GRID = [500, 1000, 3000, 5000, 10000]

BATCH_ALPHA_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
BATCH_ACTOR_LR_GRID = [3e-4, 1e-3, 3e-3, 1e-2]
BATCH_RIDGE_GRID = [1e-8, 1e-6, 1e-4, 1e-3]
BATCH_T_GRID = [1000, 3000, 5000]
BATCH_SIZE_GRID = [64, 128, 256, 512, 1024]
CRITIC_LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3]
CRITIC_INNER_STEPS_GRID = [10, 25, 50, 100]

BATCH_SIZE_CENTER = 256
CRITIC_LR_CENTER = 1e-3
CRITIC_INNER_STEPS_CENTER = 50

PROBLEMS = {
    "deterministic": {
        "description": "deterministic 10x10 PrimalAlgaeDICESolver tabular grid search on the new dataset",
        "dataset_path": DATASETS_DIR / "10grid_tabular_new.csv",
        "output_csv": RESULTS_DIR / "primal_algaedice_10grid_tabular_new_grid_search.csv",
        "best_csv": RESULTS_DIR / "primal_algaedice_10grid_tabular_new_grid_search_best.csv",
        "intended_prob": 1.0,
    },
    "stochastic": {
        "description": "stochastic 10x10 PrimalAlgaeDICESolver tabular grid search",
        "dataset_path": DATASETS_DIR / "10grid_stoch_new.csv",
        "output_csv": RESULTS_DIR / "primal_algaedice_10grid_tabular_stochastic_grid_search.csv",
        "best_csv": RESULTS_DIR / "primal_algaedice_10grid_tabular_stochastic_grid_search_best.csv",
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
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print candidate counts and exit without running solvers.",
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
    alpha,
    actor_lr,
    ridge,
    T,
    critic_update,
    batch_size,
    critic_lr,
    critic_inner_steps,
):
    q_features = TabularFeatures(N, A)
    policy_features = TabularFeatures(N, A)

    return PrimalAlgaeDICESolver(
        n_states=N,
        n_actions=A,
        gamma=GAMMA,
        x0=X0,
        csv_path=str(dataset_path),
        q_function=LinearQFunction(q_features),
        policy_features=policy_features,
        seed=SEED,
        device=device,
        alpha=alpha,
        ridge=ridge,
        actor_lr=actor_lr,
        T=T,
        critic_update=critic_update,
        batch_size=batch_size,
        critic_lr=CRITIC_LR_CENTER if critic_lr is None else critic_lr,
        critic_inner_steps=(
            CRITIC_INNER_STEPS_CENTER
            if critic_inner_steps is None
            else critic_inner_steps
        ),
        terminal_states=TERMINAL_STATES,
        init_states=[X0],
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


def optional_int_key(value):
    if value is None:
        return "none"
    try:
        if pd.isna(value):
            return "none"
    except TypeError:
        pass
    return str(int(value))


def candidate_key(row):
    return (
        str(row["critic_update"]),
        float(row["alpha"]),
        float(row["actor_lr"]),
        float(row["ridge"]),
        int(row["T"]),
        optional_int_key(row.get("batch_size")),
        optional_float_key(row.get("critic_lr")),
        optional_int_key(row.get("critic_inner_steps")),
    )


def load_existing_results(resume, output_csv):
    if not resume or not output_csv.exists():
        return [], set()

    try:
        df = pd.read_csv(output_csv)
    except pd.errors.EmptyDataError:
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
        "final_objective": np.nan,
        "final_actor_loss": np.nan,
        "final_critic_loss": np.nan,
        "final_critic_objective": np.nan,
        "final_theta_norm": np.nan,
        "final_psi_norm": np.nan,
        "final_policy_grad_norm": np.nan,
        "final_actor_delta_mean": np.nan,
        "final_actor_delta_std": np.nan,
        "final_critic_delta_mean": np.nan,
        "final_critic_delta_std": np.nan,
        "done_fraction": np.nan,
    }


def base_row(candidate, problem_name, device, status="ok", error=""):
    problem = PROBLEMS[problem_name]
    row = {
        "critic_update": str(candidate["critic_update"]),
        "alpha": float(candidate["alpha"]),
        "actor_lr": float(candidate["actor_lr"]),
        "ridge": float(candidate["ridge"]),
        "T": int(candidate["T"]),
        "batch_size": candidate["batch_size"],
        "critic_lr": candidate["critic_lr"],
        "critic_inner_steps": candidate["critic_inner_steps"],
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


def failed_worker_row(candidate, problem_name, exc, device=None):
    return base_row(candidate, problem_name, device or DEVICE, status="failed", error=repr(exc))


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


def run_candidate(candidate, problem_name, mdp, planner, dataset_path, device, d_star, v_star):
    start = time.perf_counter()
    row = base_row(candidate, problem_name, device)

    try:
        solver = make_solver(
            dataset_path=dataset_path,
            device=device,
            alpha=candidate["alpha"],
            actor_lr=candidate["actor_lr"],
            ridge=candidate["ridge"],
            T=candidate["T"],
            critic_update=candidate["critic_update"],
            batch_size=candidate["batch_size"],
            critic_lr=candidate["critic_lr"],
            critic_inner_steps=candidate["critic_inner_steps"],
        )
        evaluator = FOGASEvaluator(solver=solver, mdp=mdp, planner=planner)

        solver.run(
            alpha=candidate["alpha"],
            actor_lr=candidate["actor_lr"],
            ridge=candidate["ridge"],
            T=candidate["T"],
            critic_update=candidate["critic_update"],
            batch_size=candidate["batch_size"],
            critic_lr=candidate["critic_lr"],
            critic_inner_steps=candidate["critic_inner_steps"],
            tqdm_print=False,
            verbose=False,
        )

        row.update(evaluate_policy(planner, evaluator, "solver", d_star, v_star))
        row.update(evaluate_policy(planner, evaluator, "greedy", d_star, v_star))

        diagnostics = solver.get_diagnostics() or []
        if diagnostics:
            final = diagnostics[-1]
            row.update(
                {
                    "final_objective": finite_float(final.get("objective")),
                    "final_actor_loss": finite_float(final.get("actor_loss")),
                    "final_critic_loss": finite_float(final.get("critic_loss")),
                    "final_critic_objective": finite_float(final.get("critic_objective")),
                    "final_theta_norm": finite_float(final.get("theta_norm")),
                    "final_psi_norm": finite_float(final.get("psi_norm")),
                    "final_policy_grad_norm": finite_float(final.get("policy_grad_norm")),
                    "final_actor_delta_mean": finite_float(final.get("actor_delta_mean")),
                    "final_actor_delta_std": finite_float(final.get("actor_delta_std")),
                    "final_critic_delta_mean": finite_float(final.get("critic_delta_mean")),
                    "final_critic_delta_std": finite_float(final.get("critic_delta_std")),
                    "done_fraction": finite_float(final.get("done_fraction")),
                }
            )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        row["elapsed_seconds"] = float(time.perf_counter() - start)

    return row


def run_candidate_worker(payload):
    candidate, problem_name, dataset_path_str, device_str, torch_threads = payload
    configure_worker_threads(torch_threads)
    set_seed(SEED)
    device = torch.device(device_str)
    mdp, planner = build_mdp(problem_name, device)
    d_star = planner.state_mu_star.detach().cpu()
    v_star = planner.v_star.detach().cpu()
    return run_candidate(
        candidate=candidate,
        problem_name=problem_name,
        mdp=mdp,
        planner=planner,
        dataset_path=Path(dataset_path_str),
        device=device,
        d_star=d_star,
        v_star=v_star,
    )


def closed_form_candidates():
    return [
        {
            "critic_update": "closed_form",
            "alpha": alpha,
            "actor_lr": actor_lr,
            "ridge": ridge,
            "T": T,
            "batch_size": None,
            "critic_lr": None,
            "critic_inner_steps": None,
        }
        for alpha, actor_lr, ridge, T in itertools.product(
            CLOSED_FORM_ALPHA_GRID,
            CLOSED_FORM_ACTOR_LR_GRID,
            CLOSED_FORM_RIDGE_GRID,
            CLOSED_FORM_T_GRID,
        )
    ]


def batch_adam_candidates():
    candidates = []
    seen = set()

    base_grid = itertools.product(
        BATCH_ALPHA_GRID,
        BATCH_ACTOR_LR_GRID,
        BATCH_RIDGE_GRID,
        BATCH_T_GRID,
    )

    for alpha, actor_lr, ridge, T in base_grid:
        critic_settings = []
        critic_settings.extend(
            (batch_size, CRITIC_LR_CENTER, CRITIC_INNER_STEPS_CENTER)
            for batch_size in BATCH_SIZE_GRID
        )
        critic_settings.extend(
            (BATCH_SIZE_CENTER, critic_lr, CRITIC_INNER_STEPS_CENTER)
            for critic_lr in CRITIC_LR_GRID
        )
        critic_settings.extend(
            (BATCH_SIZE_CENTER, CRITIC_LR_CENTER, critic_inner_steps)
            for critic_inner_steps in CRITIC_INNER_STEPS_GRID
        )

        for batch_size, critic_lr, critic_inner_steps in critic_settings:
            candidate = {
                "critic_update": "batch_adam",
                "alpha": alpha,
                "actor_lr": actor_lr,
                "ridge": ridge,
                "T": T,
                "batch_size": batch_size,
                "critic_lr": critic_lr,
                "critic_inner_steps": critic_inner_steps,
            }
            key = candidate_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

    return candidates


def all_candidates():
    return closed_form_candidates() + batch_adam_candidates()


def candidate_counts():
    closed_count = len(closed_form_candidates())
    batch_count = len(batch_adam_candidates())
    return {
        "closed_form": closed_count,
        "batch_adam": batch_count,
        "total": closed_count + batch_count,
    }


def run_grid_search(problem_name):
    if problem_name not in PROBLEMS:
        raise ValueError(f"Unknown problem {problem_name!r}. Expected one of {sorted(PROBLEMS)}")

    multiprocessing.set_start_method("spawn", force=True)

    problem = PROBLEMS[problem_name]
    args = parse_args(problem["description"])
    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    configure_worker_threads(torch_threads)
    set_seed(SEED)

    if args.devices:
        device_list = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        device_list = [str(DEVICE)]

    counts = candidate_counts()
    if args.count_only:
        print(f"closed_form candidates: {counts['closed_form']}")
        print(f"batch_adam candidates: {counts['batch_adam']}")
        print(f"total candidates: {counts['total']}")
        return

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
    print(f"Closed-form candidates: {counts['closed_form']}")
    print(f"Batch-Adam candidates: {counts['batch_adam']}")
    print(f"Total grid size: {counts['total']}")

    candidates_all = all_candidates()
    if args.max_runs is not None:
        candidates_all = candidates_all[: max(0, int(args.max_runs))]

    results, completed = load_existing_results(args.resume, output_csv)
    candidates = [
        candidate
        for candidate in candidates_all
        if candidate_key(candidate) not in completed
    ]

    print(f"Candidates to run: {len(candidates)}")
    if args.resume:
        print(f"Resumed rows: {len(results)}")
    if not args.resume and output_csv.exists():
        print("Existing output will be overwritten because --resume was not set.")

    progress = not args.no_progress
    desc = f"PrimalAlgaeDICESolver {problem_name} 10-grid tabular search"

    if workers == 1:
        mdp, planner = build_mdp(problem_name, DEVICE)
        d_star = planner.state_mu_star.detach().cpu()
        v_star = planner.v_star.detach().cpu()

        outer = tqdm(candidates, desc=desc, unit="run", disable=not progress)
        for run_idx, candidate in enumerate(outer, start=len(results) + 1):
            row = run_candidate(
                candidate=candidate,
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
                candidate,
                problem_name,
                str(dataset_path),
                device_list[i % len(device_list)],
                torch_threads,
            )
            for i, candidate in enumerate(candidates)
        ]
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
                try:
                    row = future.result()
                except Exception as exc:
                    row = failed_worker_row(future_to_candidate[future], problem_name, exc)
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
