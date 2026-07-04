"""
FinalLinearSolver policy ablation for the deterministic 10x10 tabular grid.

The script mirrors the 5-grid policy ablation while using the deterministic
10-grid MDP and fixed Generalized FOGAS hyperparameters from the 10-grid
dataset-grid experiment.
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
RESULTS_DIR = (
    PROJECT_ROOT / "data" / "results" / "generalization" / "ablations" / "policy" / "10grid"
)

DATASET_PATH = DATASETS_DIR / "10grid_tabular_new.csv"
OUTPUT_CSV = RESULTS_DIR / "final_linear_10grid_tabular_policy_ablation.csv"
BEST_CSV = RESULTS_DIR / "final_linear_10grid_tabular_policy_ablation_best.csv"
STATS_CSV = RESULTS_DIR / "final_linear_10grid_tabular_policy_ablation_stats.csv"
CURVES_CSV = RESULTS_DIR / "final_linear_10grid_tabular_policy_ablation_curves.csv"
CURVE_STATS_CSV = RESULTS_DIR / "final_linear_10grid_tabular_policy_ablation_curve_stats.csv"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
torch.set_default_dtype(torch.float64)

from rl_methods.fogas import FOGASEvaluator  # noqa: E402
from rl_methods.fogas_generalization import (  # noqa: E402
    FinalLinearSolver,
    LinearFunction,
    LinearQFunction,
    TabularFeatures,
)
from rl_methods.mdp import DiscreteMDP, Planner  # noqa: E402


SEED = 42
SWEEP_SEEDS = [42, 43, 44, 45, 46]
EVAL_SEED = 10_000
NUM_TRAJECTORIES = 100
MAX_STEPS = 50
CHECKPOINT_COUNT = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
INTENDED_PROB = 1.0

BASE_ALPHA = 0.001
BASE_ETA = 3e-5
BASE_RHO = 1.0
BASE_T = 10_000
BASE_THETA_LR = 0.01
BASE_THETA_INNER_STEPS = 10
BASE_THETA_LAMBDA = 1e-8
BASE_REINFORCE_SAMPLES = 4
BASE_FISHER_DAMPING = 1e-3
CG_ITERS = 50
CG_TOL = 1e-10

NPG_ALPHA_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SGD_ALPHA_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
FISHER_DAMPING_GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
REINFORCE_SAMPLES_GRID = [2**power for power in range(8)]

RANK_COLUMNS = [
    "greedy_success_rate",
    "solver_success_rate",
    "greedy_avg_return",
    "solver_avg_return",
    "elapsed_seconds",
]
RANK_ASCENDING = [False, False, False, False, True]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FinalLinearSolver policy ablations on the deterministic 10x10 tabular grid."
    )
    parser.add_argument("--resume", action="store_true", help="Skip successful completed candidates.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit candidate count for smoke tests.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes.")
    parser.add_argument("--torch-threads", type=int, default=1, help="Torch CPU threads per worker.")
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated worker devices, e.g. cuda:0,cuda:1. Defaults to cuda if available else cpu.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
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


def parse_device_list(devices):
    if devices:
        parsed = [device.strip() for device in str(devices).split(",") if device.strip()]
        if parsed:
            return parsed
    return [str(DEVICE)]


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
    return s if sp in WALL_STATES else sp


def transition_probs(s, a, intended_prob=INTENDED_PROB):
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


def build_mdp(device):
    def transition_fn(s, a):
        probs = torch.zeros(N, dtype=torch.float64)
        for sp, prob in transition_probs(s, a).items():
            probs[sp] = prob
        return probs

    def reward_fn(s, a):
        if int(s) in TERMINAL_STATES:
            return 0.0
        return sum(prob * reward_from_next_state(sp) for sp, prob in transition_probs(s, a).items())

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


def make_solver(device, seed):
    u_features = TabularFeatures(N, A)
    q_features = TabularFeatures(N, A)
    policy_features = TabularFeatures(N, A)
    return FinalLinearSolver(
        n_states=N,
        n_actions=A,
        gamma=GAMMA,
        x0=X0,
        csv_path=str(DATASET_PATH),
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
        beta_update="fogas_full",
    )


def finite_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def checkpoint_steps(T):
    T = int(T)
    steps = sorted({max(1, int(round(T * idx / CHECKPOINT_COUNT))) for idx in range(1, CHECKPOINT_COUNT + 1)})
    if steps[-1] != T:
        steps[-1] = T
    return steps


def greedy_policy(pi):
    pi = pi.to(dtype=torch.float64)
    greedy = torch.zeros_like(pi)
    best_actions = torch.argmax(pi, dim=1)
    greedy[torch.arange(pi.shape[0], device=pi.device), best_actions] = 1.0
    return greedy


def evaluate_policy_tensor(mdp, planner, pi, d_star, v_star, seed):
    pi = pi.to(dtype=torch.float64, device=mdp.r.device)
    returns = []
    successes = 0

    for idx in range(NUM_TRAJECTORIES):
        current_seed = int(seed) + idx
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(current_seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(current_seed)

        state = int(mdp.x0)
        total_return = 0.0
        reached_goal = False
        for step in range(MAX_STEPS):
            action_probs = pi[state]
            prob_sum = action_probs.sum()
            if prob_sum <= 0:
                raise ValueError(f"Policy probabilities at state {state} must have positive mass.")
            action = int(torch.multinomial(action_probs / prob_sum, num_samples=1).item())
            row_idx = state * mdp.A + action
            reward = mdp.r[row_idx]
            reward_value = float(reward.item() if isinstance(reward, torch.Tensor) else reward)
            total_return += (float(mdp.gamma) ** step) * reward_value
            transition_probs_tensor = mdp.P[row_idx].to(dtype=torch.float64, device=mdp.r.device)
            next_state = int(torch.multinomial(transition_probs_tensor, num_samples=1).item())
            reached_goal = next_state == GOAL_GRID
            if next_state in TERMINAL_STATES:
                break
            state = next_state

        returns.append(total_return)
        successes += int(reached_goal)

    returns = np.asarray(returns, dtype=float)
    v_pi, _ = planner.evaluate_policy(pi)
    v_gap = float((d_star * (v_star - v_pi.detach().cpu())).sum().item())
    return {
        "success_rate": float(successes / NUM_TRAJECTORIES) if NUM_TRAJECTORIES else 0.0,
        "avg_return": float(returns.mean()) if returns.size else 0.0,
        "v_x0": float(v_pi[planner.x0].detach().cpu().item()),
        "v_gap": finite_float(v_gap),
    }


def make_candidate(
    ablation,
    policy_optimizer,
    policy_gradient,
    alpha=BASE_ALPHA,
    reinforce_samples=BASE_REINFORCE_SAMPLES,
    fisher_damping=BASE_FISHER_DAMPING,
    seed=SEED,
):
    return {
        "problem": "deterministic_10grid",
        "ablation": str(ablation),
        "beta_update": "fogas_full",
        "T": int(BASE_T),
        "alpha": float(alpha),
        "eta": float(BASE_ETA),
        "rho": float(BASE_RHO),
        "theta_mode": "reg_fixed",
        "theta_lambda": float(BASE_THETA_LAMBDA),
        "theta_lr": float(BASE_THETA_LR),
        "theta_inner_steps": int(BASE_THETA_INNER_STEPS),
        "theta_optimizer": "adam",
        "theta_start_mode": "warm",
        "theta_include_beta_cov": False,
        "policy_optimizer": str(policy_optimizer),
        "policy_gradient": str(policy_gradient),
        "reinforce_samples": int(reinforce_samples),
        "fisher_damping": float(fisher_damping),
        "cg_iters": int(CG_ITERS),
        "cg_tol": float(CG_TOL),
        "state_weight_update": "normal",
        "intended_prob": float(INTENDED_PROB),
        "dataset_path": str(DATASET_PATH),
        "seed": int(seed),
    }


def all_candidates():
    candidates = []
    for seed in SWEEP_SEEDS:
        candidates.append(
            make_candidate(
                ablation="adam_exact_baseline",
                policy_optimizer="adam",
                policy_gradient="exact",
                seed=seed,
            )
        )
        for alpha, fisher_damping in itertools.product(NPG_ALPHA_GRID, FISHER_DAMPING_GRID):
            candidates.append(
                make_candidate(
                    ablation="npg_exact_alpha_fisher",
                    policy_optimizer="npg",
                    policy_gradient="exact",
                    alpha=alpha,
                    fisher_damping=fisher_damping,
                    seed=seed,
                )
            )
        for alpha in SGD_ALPHA_GRID:
            candidates.append(
                make_candidate(
                    ablation="sgd_exact_alpha",
                    policy_optimizer="sgd",
                    policy_gradient="exact",
                    alpha=alpha,
                    seed=seed,
                )
            )
        for reinforce_samples in REINFORCE_SAMPLES_GRID:
            candidates.append(
                make_candidate(
                    ablation="adam_reinforce_samples",
                    policy_optimizer="adam",
                    policy_gradient="reinforce",
                    reinforce_samples=reinforce_samples,
                    seed=seed,
                )
            )
    return candidates


def candidate_key(row):
    return (
        str(row["ablation"]),
        str(row["policy_optimizer"]),
        str(row["policy_gradient"]),
        float(row["alpha"]),
        int(row["reinforce_samples"]),
        float(row["fisher_damping"]),
        int(row["seed"]),
    )


def load_existing_results(resume):
    if not resume or not OUTPUT_CSV.exists():
        return [], [], set()
    try:
        df = pd.read_csv(OUTPUT_CSV)
    except pd.errors.EmptyDataError:
        return [], [], set()
    rows = df.to_dict("records")
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    completed = {candidate_key(row) for row in ok.to_dict("records")}
    curve_rows = []
    if CURVES_CSV.exists():
        curve_rows = pd.read_csv(CURVES_CSV).to_dict("records")
    return rows, curve_rows, completed


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
        "elapsed_seconds": np.nan,
    }


def base_row(candidate, device, status="ok", error=""):
    row = dict(candidate)
    row.update(
        {
            "grid_size": int(GRID_SIZE),
            "n_states": int(N),
            "n_actions": int(A),
            "gamma": float(GAMMA),
            "x0": int(X0),
            "goal_state": int(GOAL_GRID),
            "pit_states": ",".join(str(x) for x in sorted(PIT_GRIDS)),
            "terminal_states": ",".join(str(x) for x in sorted(TERMINAL_STATES)),
            "num_trajectories": int(NUM_TRAJECTORIES),
            "max_steps": int(MAX_STEPS),
            "eval_seed": int(EVAL_SEED),
            "device": str(device),
            "status": status,
            "error": error,
        }
    )
    row.update(blank_metrics())
    return row


def curve_row(candidate, checkpoint_idx, checkpoint_step, solver_stats, greedy_stats, diagnostics):
    row = dict(candidate)
    row.update(
        {
            "checkpoint_idx": int(checkpoint_idx),
            "checkpoint_step": int(checkpoint_step),
            "num_trajectories": int(NUM_TRAJECTORIES),
            "max_steps": int(MAX_STEPS),
            "eval_seed": int(EVAL_SEED),
            "solver_success_rate": finite_float(solver_stats["success_rate"]),
            "solver_avg_return": finite_float(solver_stats["avg_return"]),
            "solver_v_x0": finite_float(solver_stats["v_x0"]),
            "solver_v_gap": finite_float(solver_stats["v_gap"]),
            "greedy_success_rate": finite_float(greedy_stats["success_rate"]),
            "greedy_avg_return": finite_float(greedy_stats["avg_return"]),
            "greedy_v_x0": finite_float(greedy_stats["v_x0"]),
            "greedy_v_gap": finite_float(greedy_stats["v_gap"]),
            "total_loss": finite_float(diagnostics.get("total_loss")),
            "policy_objective": finite_float(diagnostics.get("policy_objective")),
            "beta_objective": finite_float(diagnostics.get("beta_objective")),
            "q_objective": finite_float(diagnostics.get("q_objective")),
            "theta_norm": finite_float(diagnostics.get("theta_norm")),
            "policy_grad_norm": finite_float(diagnostics.get("policy_grad_norm")),
            "beta_grad_norm": finite_float(diagnostics.get("beta_grad_norm")),
            "theta_grad_norm": finite_float(diagnostics.get("theta_grad_norm")),
            "status": "ok",
            "error": "",
        }
    )
    return row


def evaluate_policy_curve(candidate, solver, mdp, planner, d_star, v_star):
    rows = []
    diagnostics_history = solver.get_diagnostics() or []
    for checkpoint_idx, step in enumerate(checkpoint_steps(candidate["T"]), start=1):
        history_idx = int(step) - 1
        pi = solver._linear_policy_matrix(solver.psi_history[history_idx])
        solver_stats = evaluate_policy_tensor(mdp, planner, pi, d_star, v_star, EVAL_SEED)
        greedy_stats = evaluate_policy_tensor(mdp, planner, greedy_policy(pi), d_star, v_star, EVAL_SEED)
        diagnostics = diagnostics_history[history_idx] if history_idx < len(diagnostics_history) else {}
        rows.append(curve_row(candidate, checkpoint_idx, step, solver_stats, greedy_stats, diagnostics))
    return rows


def final_metrics_from_curve(curve_rows):
    if not curve_rows:
        return None
    last = max(curve_rows, key=lambda row: int(row["checkpoint_step"]))
    return {
        "solver_success_rate": last["solver_success_rate"],
        "solver_avg_return": last["solver_avg_return"],
        "solver_v_x0": last["solver_v_x0"],
        "solver_v_gap": last["solver_v_gap"],
        "greedy_success_rate": last["greedy_success_rate"],
        "greedy_avg_return": last["greedy_avg_return"],
        "greedy_v_x0": last["greedy_v_x0"],
        "greedy_v_gap": last["greedy_v_gap"],
        "final_total_loss": last["total_loss"],
        "final_policy_objective": last["policy_objective"],
        "final_beta_objective": last["beta_objective"],
        "final_q_objective": last["q_objective"],
        "final_theta_norm": last["theta_norm"],
        "final_policy_grad_norm": last["policy_grad_norm"],
        "final_beta_grad_norm": last["beta_grad_norm"],
        "final_theta_grad_norm": last["theta_grad_norm"],
    }


def run_candidate(candidate, mdp, planner, device, d_star, v_star):
    start = time.perf_counter()
    row = base_row(candidate, device)
    curve_rows = []
    try:
        solver = make_solver(device=device, seed=int(candidate["seed"]))
        _evaluator = FOGASEvaluator(solver=solver, mdp=mdp, planner=planner)
        solver.run(
            alpha=candidate["alpha"],
            eta=candidate["eta"],
            rho=candidate["rho"],
            T=candidate["T"],
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
        curve_rows = evaluate_policy_curve(candidate, solver, mdp, planner, d_star, v_star)
        row.update(final_metrics_from_curve(curve_rows) or {})
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        row["elapsed_seconds"] = float(time.perf_counter() - start)
    return row, curve_rows


def run_candidate_worker(payload):
    candidate, device_str, torch_threads = payload
    configure_worker_threads(torch_threads)
    set_seed(int(candidate["seed"]))
    device = torch.device(device_str)
    mdp, planner = build_mdp(device)
    d_star = planner.state_mu_star.detach().cpu()
    v_star = planner.v_star.detach().cpu()
    return run_candidate(candidate, mdp, planner, device, d_star, v_star)


def failed_worker_row(candidate, exc):
    return base_row(candidate, DEVICE, status="failed", error=repr(exc)), []


def ordered_results_frame(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.sort_values(RANK_COLUMNS, ascending=RANK_ASCENDING, na_position="last").reset_index(drop=True)


def aggregate_frame(rows, group_columns):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metrics = ["solver_success_rate", "greedy_success_rate", "solver_avg_return", "greedy_avg_return"]
    out = []
    for group_key, group in df.groupby(group_columns, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_columns, key_values))
        ok = group[group["status"] == "ok"].copy() if "status" in group.columns else group.copy()
        row.update(
            {
                "count": int(len(group)),
                "seed_count": int(ok["seed"].nunique()) if "seed" in ok.columns and not ok.empty else 0,
                "ok_count": int(len(ok)),
                "failed_count": int((group["status"] == "failed").sum()) if "status" in group.columns else 0,
                "elapsed_seconds_mean": float(ok["elapsed_seconds"].mean())
                if "elapsed_seconds" in ok.columns and not ok.empty
                else np.nan,
            }
        )
        for metric in metrics:
            values = ok[metric] if metric in ok.columns else pd.Series(dtype=float)
            row[f"{metric}_best"] = float(values.max()) if not values.empty else np.nan
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            metric_std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            metric_sem = metric_std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{metric}_std"] = metric_std
            row[f"{metric}_sem"] = metric_sem
            row[f"{metric}_ci95"] = 1.96 * metric_sem
        out.append(row)
    return pd.DataFrame(out)


def build_stats_frame(results):
    group_columns = [
        "ablation",
        "policy_optimizer",
        "policy_gradient",
        "alpha",
        "reinforce_samples",
        "fisher_damping",
        "cg_iters",
        "cg_tol",
    ]
    df = aggregate_frame(results, group_columns)
    if df.empty:
        return df
    return df.sort_values(
        ["ablation", "greedy_success_rate_best", "solver_success_rate_best", "greedy_avg_return_best"],
        ascending=[True, False, False, False],
        na_position="last",
    )


def build_curve_stats_frame(curve_results):
    group_columns = [
        "ablation",
        "policy_optimizer",
        "policy_gradient",
        "alpha",
        "reinforce_samples",
        "fisher_damping",
        "cg_iters",
        "cg_tol",
        "checkpoint_step",
    ]
    df = aggregate_frame(curve_results, group_columns)
    if df.empty:
        return df
    return df.sort_values(
        ["ablation", "policy_optimizer", "policy_gradient", "alpha", "fisher_damping", "checkpoint_step"],
        ascending=True,
        na_position="last",
    )


def save_results(results, curve_results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = ordered_results_frame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    ok = df[df["status"] == "ok"] if not df.empty else df
    if not ok.empty:
        ok.head(1).to_csv(BEST_CSV, index=False)

    stats = build_stats_frame(results)
    if not stats.empty:
        stats.to_csv(STATS_CSV, index=False)

    curves = pd.DataFrame(curve_results)
    if not curves.empty:
        curves.to_csv(CURVES_CSV, index=False)

    curve_stats = build_curve_stats_frame(curve_results)
    if not curve_stats.empty:
        curve_stats.to_csv(CURVE_STATS_CSV, index=False)


def run_grid_search():
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    device_list = parse_device_list(args.devices)
    configure_worker_threads(torch_threads)
    set_seed(SEED)

    candidates_all = all_candidates()
    if args.max_runs is not None:
        candidates_all = candidates_all[: max(0, int(args.max_runs))]

    results, curve_results, completed = load_existing_results(args.resume)
    candidates = [candidate for candidate in candidates_all if candidate_key(candidate) not in completed]

    print(f"Devices: {device_list}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Results: {OUTPUT_CSV}")
    print(f"Curves: {CURVES_CSV}")
    print(f"Workers: {workers}")
    print(f"Torch threads per worker: {torch_threads}")
    print(f"Total candidate grid size: {len(candidates_all)}")
    print(f"Candidates to run: {len(candidates)}")
    if args.resume:
        print(f"Resumed rows: {len(results)}")
    if not args.resume and OUTPUT_CSV.exists():
        print("Existing output will be overwritten because --resume was not set.")

    progress = not args.no_progress
    desc = "FinalLinearSolver 10-grid tabular policy ablation"

    if workers == 1:
        device = torch.device(device_list[0])
        mdp, planner = build_mdp(device)
        d_star = planner.state_mu_star.detach().cpu()
        v_star = planner.v_star.detach().cpu()
        outer = tqdm(candidates, desc=desc, unit="run", disable=not progress)
        for run_idx, candidate in enumerate(outer, start=len(results) + 1):
            row, new_curve_rows = run_candidate(candidate, mdp, planner, device, d_star, v_star)
            row["run_idx"] = int(run_idx)
            for curve in new_curve_rows:
                curve["run_idx"] = int(run_idx)
            results.append(row)
            curve_results.extend(new_curve_rows)
            save_results(results, curve_results)
            if progress:
                outer.set_postfix(
                    {
                        "ablation": row["ablation"],
                        "greedy_success": row["greedy_success_rate"],
                        "status": row["status"],
                    }
                )
    else:
        payloads = [
            (candidate, device_list[i % len(device_list)], torch_threads)
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
                candidate = future_to_candidate[future]
                try:
                    row, new_curve_rows = future.result()
                except Exception as exc:
                    row, new_curve_rows = failed_worker_row(candidate, exc)
                row["run_idx"] = int(next_run_idx)
                for curve in new_curve_rows:
                    curve["run_idx"] = int(next_run_idx)
                next_run_idx += 1
                results.append(row)
                curve_results.extend(new_curve_rows)
                save_results(results, curve_results)
                if progress:
                    outer.set_postfix(
                        {
                            "ablation": row.get("ablation"),
                            "greedy_success": row.get("greedy_success_rate", np.nan),
                            "status": row.get("status"),
                        }
                    )

    save_results(results, curve_results)
    df = ordered_results_frame(results)
    ok_count = int((df["status"] == "ok").sum()) if not df.empty else 0
    failed_count = int((df["status"] == "failed").sum()) if not df.empty else 0
    print("\nPolicy ablation complete.")
    print(f"Rows saved: {len(df)}")
    print(f"Successful runs: {ok_count}")
    print(f"Failed runs: {failed_count}")
    print(f"Output CSV: {OUTPUT_CSV}")
    if BEST_CSV.exists():
        print(f"Best row CSV: {BEST_CSV}")
        print(pd.read_csv(BEST_CSV).to_string(index=False))


if __name__ == "__main__":
    run_grid_search()
