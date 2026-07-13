"""Produce the focused stabilization sweep used in the beta ablation.

Scientific role
---------------
This thesis-facing diagnostic keeps the selected stochastic 5 x 5
Generalized FOGAS configuration fixed and varies only ``rho``. Checkpointed
evaluation across seeds shows how stabilization changes the learning curve,
not only the final reward.

Inputs and outputs
------------------
The script reads ``data/datasets/generalization/5grid_stochastic.csv`` and
writes raw checkpoint rows plus a grouped summary to
``data/results/generalization/ablations/beta``. ``notebooks/ablations.ipynb``
loads the summary for the focused rho plot.

Run this file directly from the repository root. Five seeds and evaluations
every ``T / 20`` iterations are used by default; completed rho/seed runs are
saved immediately and can be skipped with ``--resume``.
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


def find_root(current_path):
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / "src" / "rl_methods").exists() and (parent / "data").exists():
            return parent
    return current_path


PROJECT_ROOT = find_root(Path(__file__).resolve())
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets" / "generalization"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "generalization" / "ablations" / "beta"
OUTPUT_CSV = RESULTS_DIR / "final_linear_5grid_stochastic_rho_sweep_checkpoints.csv"
SUMMARY_CSV = RESULTS_DIR / "final_linear_5grid_stochastic_rho_sweep_checkpoints_summary.csv"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
torch.set_default_dtype(torch.float64)

from rl_methods.fogas_generalization import (  # noqa: E402
    FinalLinearSolver,
    LinearFunction,
    LinearQFunction,
    TabularFeatures,
)
from rl_methods.mdp import DiscreteMDP, Planner  # noqa: E402


BASE_SEED = 42
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
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

# Stochastic settings mirrored from grid_search_final_linear_5grid_tabular_beta_ablation.py.
DATASET_PATH = DATASETS_DIR / "5grid_stochastic.csv"
GAMMA = 0.9
TERMINAL_STATES = {GOAL_GRID, PIT_GRID}
BASE_ALPHA = 1e-3
BASE_ETA = 1e-4
BASE_RHO = 1.0
BASE_T = 3000
BASE_THETA_LAMBDA = 3e-7
BASE_THETA_LR = 1e-3
BASE_THETA_INNER_STEPS = 10
BASE_POLICY_OPTIMIZER = "adam"
BASE_POLICY_GRADIENT = "exact"
BASE_REINFORCE_SAMPLES = 4
BASE_STATE_WEIGHT_UPDATE = "normal"
BASE_BETA_UPDATE = "fogas_full"

DEFAULT_RHO_VALUES = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
CHECKPOINT_COUNT = 20


def parse_float_list(values):
    parsed = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                parsed.append(float(part))
    return parsed


def parse_int_list(values):
    parsed = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                parsed.append(int(part))
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep rho on the stochastic 5x5 FinalLinearSolver baseline."
    )
    parser.add_argument(
        "--rho-values",
        nargs="+",
        default=None,
        help=(
            "Rho values to evaluate. Accepts space-separated values or comma lists. "
            f"Default: {DEFAULT_RHO_VALUES}"
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help=(
            "Seeds to run. Accepts space-separated values or comma lists. "
            f"Default: {DEFAULT_SEEDS}"
        ),
    )
    parser.add_argument(
        "--T",
        type=int,
        default=BASE_T,
        help=f"Training iterations. Default: {BASE_T}.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit rho/seed candidates after construction. Useful for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rho/seed candidates already completed in the output CSV.",
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


def stochastic_transition_probs(s, a, intended_prob=INTENDED_PROB):
    s = int(s)
    a = int(a)
    probs_by_state = {}

    for candidate_a in range(A):
        prob = (1.0 - intended_prob) / A
        if candidate_a == a:
            prob += intended_prob

        sp = move_deterministic(s, candidate_a)
        probs_by_state[sp] = probs_by_state.get(sp, 0.0) + prob

    return probs_by_state


def reward_from_next_state(sp):
    sp = int(sp)
    if sp == GOAL_GRID:
        return 1.0
    if sp == PIT_GRID:
        return -1.0
    return -0.1


def build_mdp(device):
    def transition_fn(s, a):
        probs = torch.zeros(N, dtype=torch.float64)
        for sp, prob in stochastic_transition_probs(s, a).items():
            probs[sp] = prob
        return probs

    def reward_fn(s, a):
        return sum(
            prob * reward_from_next_state(sp)
            for sp, prob in stochastic_transition_probs(s, a).items()
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


def make_solver(dataset_path, device, seed):
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
        seed=seed,
        device=device,
        theta_include_beta_cov=False,
        theta_mode="reg_fixed",
        theta_lambda=BASE_THETA_LAMBDA,
        theta_optimizer="adam",
        theta_inner_steps=BASE_THETA_INNER_STEPS,
        theta_lr=BASE_THETA_LR,
        theta_start_mode="warm",
        beta_update=BASE_BETA_UPDATE,
    )


def checkpoint_steps(T):
    T = int(T)
    if T <= 0:
        raise ValueError("T must be positive")
    interval = max(1, T // CHECKPOINT_COUNT)
    steps = list(range(interval, T + 1, interval))
    if steps[-1] != T:
        steps.append(T)
    return sorted(set(steps))


def greedy_policy(pi):
    pi = pi.to(dtype=torch.float64)
    greedy = torch.zeros_like(pi)
    best_actions = torch.argmax(pi, dim=1)
    greedy[torch.arange(pi.shape[0], device=pi.device), best_actions] = 1.0
    return greedy


def evaluate_policy_rollouts(mdp, pi, seed):
    pi = pi.to(dtype=torch.float64, device=mdp.r.device)
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
            if next_state in TERMINAL_STATES:
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


def unique_sorted(values):
    return sorted({float(value) for value in values})


def candidates_from_args(args):
    rho_values = DEFAULT_RHO_VALUES if args.rho_values is None else parse_float_list(args.rho_values)
    rho_values = unique_sorted([*rho_values, BASE_RHO])

    seeds = DEFAULT_SEEDS if args.seeds is None else parse_int_list(args.seeds)
    seeds = sorted({int(seed) for seed in seeds})

    candidates = []
    for rho in rho_values:
        for seed in seeds:
            candidates.append(
                {
                    "rho": float(rho),
                    "seed": int(seed),
                    "T": int(args.T),
                    "alpha": float(BASE_ALPHA),
                    "eta": float(BASE_ETA),
                    "theta_lambda": float(BASE_THETA_LAMBDA),
                    "theta_lr": float(BASE_THETA_LR),
                    "theta_inner_steps": int(BASE_THETA_INNER_STEPS),
                    "beta_update": BASE_BETA_UPDATE,
                    "policy_optimizer": BASE_POLICY_OPTIMIZER,
                    "policy_gradient": BASE_POLICY_GRADIENT,
                    "reinforce_samples": int(BASE_REINFORCE_SAMPLES),
                    "state_weight_update": BASE_STATE_WEIGHT_UPDATE,
                    "dataset_path": str(DATASET_PATH),
                }
            )
    return candidates


def candidate_key(candidate):
    return (
        float(candidate["rho"]),
        int(candidate["seed"]),
        int(candidate["T"]),
        float(candidate["alpha"]),
        float(candidate["eta"]),
        float(candidate["theta_lambda"]),
        float(candidate["theta_lr"]),
        int(candidate["theta_inner_steps"]),
        str(candidate["beta_update"]),
        str(candidate["policy_optimizer"]),
        str(candidate["policy_gradient"]),
    )


def row_candidate_key(row):
    return candidate_key(
        {
            "rho": row["rho"],
            "seed": row["seed"],
            "T": row["T"],
            "alpha": row["alpha"],
            "eta": row["eta"],
            "theta_lambda": row["theta_lambda"],
            "theta_lr": row["theta_lr"],
            "theta_inner_steps": row["theta_inner_steps"],
            "beta_update": row["beta_update"],
            "policy_optimizer": row["policy_optimizer"],
            "policy_gradient": row["policy_gradient"],
        }
    )


def load_existing_results(resume):
    if not resume or not OUTPUT_CSV.exists():
        return [], set()

    df = pd.read_csv(OUTPUT_CSV)
    rows = df.to_dict("records")
    completed = set()

    if df.empty or "status" not in df.columns:
        return rows, completed

    for _key, group in df.groupby(
        [
            "rho",
            "seed",
            "T",
            "alpha",
            "eta",
            "theta_lambda",
            "theta_lr",
            "theta_inner_steps",
            "beta_update",
            "policy_optimizer",
            "policy_gradient",
        ],
        dropna=False,
    ):
        ok = group[group["status"] == "ok"].copy()
        if ok.empty:
            continue
        T = int(ok["T"].iloc[0])
        expected = set(checkpoint_steps(T))
        observed = {int(step) for step in ok["step"].dropna().astype(int)}
        if expected.issubset(observed):
            completed.add(row_candidate_key(ok.iloc[0].to_dict()))

    return rows, completed


def base_row(candidate, device, status="ok", error=""):
    row = dict(candidate)
    row.update(
        {
            "problem": "stochastic",
            "gamma": float(GAMMA),
            "intended_prob": float(INTENDED_PROB),
            "num_trajectories": int(NUM_TRAJECTORIES),
            "max_steps": int(MAX_STEPS),
            "checkpoint_count": int(CHECKPOINT_COUNT),
            "device": str(device),
            "status": status,
            "error": error,
            "step": np.nan,
            "checkpoint_idx": np.nan,
            "solver_avg_reward": np.nan,
            "solver_success_rate": np.nan,
            "greedy_avg_reward": np.nan,
            "greedy_success_rate": np.nan,
            "total_loss": np.nan,
            "policy_objective": np.nan,
            "beta_objective": np.nan,
            "q_objective": np.nan,
            "theta_norm": np.nan,
            "policy_grad_norm": np.nan,
            "beta_grad_norm": np.nan,
            "theta_grad_norm": np.nan,
            "train_elapsed_seconds": np.nan,
            "elapsed_seconds": np.nan,
        }
    )
    return row


def run_candidate(candidate, mdp, dataset_path, device):
    del dataset_path
    started = time.perf_counter()
    rows = []
    seed = int(candidate["seed"])
    set_seed(seed)

    try:
        solver = make_solver(
            dataset_path=Path(candidate["dataset_path"]),
            device=device,
            seed=seed,
        )
        train_started = time.perf_counter()
        solver.run(
            alpha=candidate["alpha"],
            eta=candidate["eta"],
            rho=candidate["rho"],
            T=candidate["T"],
            policy_optimizer=candidate["policy_optimizer"],
            policy_gradient=candidate["policy_gradient"],
            reinforce_samples=candidate["reinforce_samples"],
            tqdm_print=False,
            verbose=False,
            state_weight_update=candidate["state_weight_update"],
            beta_update=candidate["beta_update"],
        )
        train_elapsed = float(time.perf_counter() - train_started)

        diagnostics_history = solver.get_diagnostics() or []
        for checkpoint_idx, step in enumerate(checkpoint_steps(candidate["T"]), start=1):
            history_idx = int(step) - 1
            pi = solver._linear_policy_matrix(solver.psi_history[history_idx])
            solver_stats = evaluate_policy_rollouts(mdp=mdp, pi=pi, seed=seed)
            greedy_stats = evaluate_policy_rollouts(mdp=mdp, pi=greedy_policy(pi), seed=seed)

            diagnostics = diagnostics_history[history_idx] if history_idx < len(diagnostics_history) else {}
            row = base_row(candidate, device)
            row.update(
                {
                    "step": int(step),
                    "checkpoint_idx": int(checkpoint_idx),
                    "solver_avg_reward": finite_float(solver_stats["avg_reward"]),
                    "solver_success_rate": finite_float(solver_stats["success_rate"]),
                    "greedy_avg_reward": finite_float(greedy_stats["avg_reward"]),
                    "greedy_success_rate": finite_float(greedy_stats["success_rate"]),
                    "total_loss": finite_float(diagnostics.get("total_loss")),
                    "policy_objective": finite_float(diagnostics.get("policy_objective")),
                    "beta_objective": finite_float(diagnostics.get("beta_objective")),
                    "q_objective": finite_float(diagnostics.get("q_objective")),
                    "theta_norm": finite_float(diagnostics.get("theta_norm")),
                    "policy_grad_norm": finite_float(diagnostics.get("policy_grad_norm")),
                    "beta_grad_norm": finite_float(diagnostics.get("beta_grad_norm")),
                    "theta_grad_norm": finite_float(diagnostics.get("theta_grad_norm")),
                    "train_elapsed_seconds": train_elapsed,
                    "elapsed_seconds": float(time.perf_counter() - started),
                }
            )
            rows.append(row)
    except Exception as exc:
        row = base_row(candidate, device, status="failed", error=repr(exc))
        row["elapsed_seconds"] = float(time.perf_counter() - started)
        rows.append(row)

    return rows


def run_candidate_worker(payload):
    candidate, device_str, torch_threads = payload
    configure_worker_threads(torch_threads)
    set_seed(int(candidate["seed"]))
    device = torch.device(device_str)
    mdp, _planner = build_mdp(device)
    return run_candidate(
        candidate=candidate,
        mdp=mdp,
        dataset_path=Path(candidate["dataset_path"]),
        device=device,
    )


def failed_worker_rows(candidate, exc):
    return [base_row(candidate, DEVICE, status="failed", error=repr(exc))]


def ordered_results_frame(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.sort_values(
        by=["rho", "seed", "step"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_summary_frame(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    grouped = ok.groupby(["rho", "step"], dropna=False)
    rows = []
    for (rho, step), group in grouped:
        row = {
            "rho": float(rho),
            "step": int(step),
            "seed_count": int(group["seed"].nunique()),
        }
        for metric in [
            "solver_avg_reward",
            "solver_success_rate",
            "greedy_avg_reward",
            "greedy_success_rate",
        ]:
            values = group[metric].astype(float)
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = std
            row[f"{metric}_sem"] = std / math.sqrt(len(values)) if len(values) > 0 else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["rho", "step"]).reset_index(drop=True)


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ordered = ordered_results_frame(results)
    ordered.to_csv(OUTPUT_CSV, index=False)

    summary = build_summary_frame(results)
    if not summary.empty:
        summary.to_csv(SUMMARY_CSV, index=False)


def run_grid_search():
    args = parse_args()
    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    configure_worker_threads(torch_threads)
    set_seed(BASE_SEED)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    candidates_all = candidates_from_args(args)
    if args.max_runs is not None:
        candidates_all = candidates_all[: max(0, int(args.max_runs))]

    if args.count_only:
        print(f"rho values: {sorted({candidate['rho'] for candidate in candidates_all})}")
        print(f"seeds: {sorted({candidate['seed'] for candidate in candidates_all})}")
        print(f"rho/seed runs: {len(candidates_all)}")
        print(f"checkpoint rows per successful run: {len(checkpoint_steps(args.T))}")
        print(f"total checkpoint rows: {len(candidates_all) * len(checkpoint_steps(args.T))}")
        return

    results, completed = load_existing_results(args.resume)
    candidates = [candidate for candidate in candidates_all if candidate_key(candidate) not in completed]

    print(f"Using device: {DEVICE}")
    print("Problem: stochastic 5x5")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Results: {OUTPUT_CSV}")
    print(f"Summary: {SUMMARY_CSV}")
    print(f"Workers: {workers}")
    print(f"Torch threads per worker: {torch_threads}")
    print(f"Rho values: {sorted({candidate['rho'] for candidate in candidates_all})}")
    print(f"Seeds: {sorted({candidate['seed'] for candidate in candidates_all})}")
    print(f"T: {int(args.T)}")
    print(f"Checkpoint steps: {checkpoint_steps(args.T)}")
    print(f"Total rho/seed runs: {len(candidates_all)}")
    print(f"Runs to execute: {len(candidates)}")
    if args.resume:
        print(f"Resumed rows: {len(results)}")
    if not args.resume and OUTPUT_CSV.exists():
        print("Existing output will be overwritten because --resume was not set.")

    progress = not args.no_progress
    desc = "FinalLinearSolver stochastic rho sweep"

    if workers == 1:
        mdp, _planner = build_mdp(DEVICE)
        outer = tqdm(candidates, desc=desc, unit="run", disable=not progress)
        for run_idx, candidate in enumerate(outer, start=1):
            rows = run_candidate(
                candidate=candidate,
                mdp=mdp,
                dataset_path=Path(candidate["dataset_path"]),
                device=DEVICE,
            )
            for row in rows:
                row["run_idx"] = int(run_idx)
            results.extend(rows)
            save_results(results)

            if progress:
                last = rows[-1]
                outer.set_postfix(
                    {
                        "rho": candidate["rho"],
                        "seed": candidate["seed"],
                        "avg": last.get("solver_avg_reward", np.nan),
                        "status": last.get("status"),
                    }
                )
    else:
        payloads = [(candidate, str(DEVICE), torch_threads) for candidate in candidates]
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
            for run_idx, future in enumerate(outer, start=1):
                candidate = future_to_candidate[future]
                try:
                    rows = future.result()
                except Exception as exc:
                    rows = failed_worker_rows(candidate, exc)
                for row in rows:
                    row["run_idx"] = int(run_idx)
                results.extend(rows)
                save_results(results)

                if progress:
                    last = rows[-1]
                    outer.set_postfix(
                        {
                            "rho": candidate["rho"],
                            "seed": candidate["seed"],
                            "avg": last.get("solver_avg_reward", np.nan),
                            "status": last.get("status"),
                        }
                    )

    save_results(results)
    df = ordered_results_frame(results)
    ok_count = int((df["status"] == "ok").sum()) if not df.empty else 0
    failed_count = int((df["status"] == "failed").sum()) if not df.empty else 0
    completed_runs = len(candidates_all) - len(candidates)

    print("\nRho sweep complete.")
    print(f"Checkpoint rows saved: {len(df)}")
    print(f"Successful checkpoint rows: {ok_count}")
    print(f"Failed rows: {failed_count}")
    print(f"Previously completed rho/seed runs skipped: {completed_runs}")
    print(f"Output CSV: {OUTPUT_CSV}")
    if SUMMARY_CSV.exists():
        print(f"Summary CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    run_grid_search()
