"""
FinalLinearSolver tabular beta-update ablation for the deterministic 10x10 grid.

This standalone script combines three workflows:

  ablation  - beta-update grid search
  curves    - rerun best beta-update settings and save learning curves
  rho-sweep - checkpointed rho sweep for the fogas_full baseline

The MDP and fixed baseline hyperparameters match the deterministic 10-grid
dataset-grid Generalized FOGAS setup.
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
    PROJECT_ROOT / "data" / "results" / "generalization" / "ablations" / "beta" / "10grid"
)

DATASET_PATH = DATASETS_DIR / "10grid_tabular_new.csv"
ABLATION_CSV = RESULTS_DIR / "final_linear_10grid_tabular_beta_ablation.csv"
BEST_CSV = RESULTS_DIR / "final_linear_10grid_tabular_beta_ablation_best.csv"
STATS_CSV = RESULTS_DIR / "final_linear_10grid_tabular_beta_ablation_stats.csv"
CURVES_CSV = RESULTS_DIR / "final_linear_10grid_beta_update_learning_curves.csv"
CURVES_PNG = RESULTS_DIR / "final_linear_10grid_beta_update_learning_curves.png"
RHO_SWEEP_CSV = RESULTS_DIR / "final_linear_10grid_rho_sweep_checkpoints.csv"
RHO_SWEEP_SUMMARY_CSV = RESULTS_DIR / "final_linear_10grid_rho_sweep_checkpoints_summary.csv"

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
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
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
BASE_THETA_MODE = "reg_fixed"
BASE_THETA_OPTIMIZER = "adam"
BASE_THETA_START_MODE = "warm"
BASE_THETA_INCLUDE_BETA_COV = False
BASE_POLICY_OPTIMIZER = "adam"
BASE_POLICY_GRADIENT = "exact"
BASE_REINFORCE_SAMPLES = 4
BASE_STATE_WEIGHT_UPDATE = "normal"

BETA_UPDATE_ORDER = [
    "fogas_full",
    "fogas_diag",
    "projected_gradient",
    "fenchel_br",
    "fenchel_mirror",
]
BETA_UPDATE_LABELS = {
    "fogas_full": "FOGAS full",
    "fogas_diag": "FOGAS diag",
    "projected_gradient": "Projected grad",
    "fenchel_br": "Fenchel BR",
    "fenchel_mirror": "Fenchel mirror",
}

DIAG_ETA_GRID = [3e-5, 1e-4, 3e-4, 1e-3]
DIAG_RHO_GRID = [0.01, 0.1, 0.5, 1.0, 2.0]
PROJECTED_ETA_GRID = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
PROJECTION_RADIUS_GRID = [None, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
MIRROR_ETA_GRID = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0]
SENSITIVITY_ETA_GRID = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
SENSITIVITY_RHO_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 5.0]
DEFAULT_RHO_SWEEP_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 5.0]

RANK_COLUMNS = [
    "greedy_success_rate",
    "solver_success_rate",
    "greedy_avg_return",
    "solver_avg_return",
    "elapsed_seconds",
]
RANK_ASCENDING = [False, False, False, False, True]


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


def make_solver(config, device, seed):
    u_features = TabularFeatures(N, A)
    q_features = TabularFeatures(N, A)
    policy_features = TabularFeatures(N, A)

    return FinalLinearSolver(
        n_states=N,
        n_actions=A,
        gamma=GAMMA,
        x0=X0,
        csv_path=str(config["dataset_path"]),
        u_function=LinearFunction(u_features),
        q_function=LinearQFunction(q_features),
        policy_features=policy_features,
        seed=seed,
        device=device,
        theta_include_beta_cov=bool(config["theta_include_beta_cov"]),
        theta_mode=str(config["theta_mode"]),
        theta_lambda=float(config["theta_lambda"]),
        theta_optimizer=str(config["theta_optimizer"]),
        theta_inner_steps=int(config["theta_inner_steps"]),
        theta_lr=float(config["theta_lr"]),
        theta_start_mode=str(config["theta_start_mode"]),
        beta_update=str(config["beta_update"]),
        beta_projection_radius=config.get("beta_projection_radius"),
    )


def none_if_nan(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def finite_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def optional_float_key(value):
    value = none_if_nan(value)
    if value is None:
        return "none"
    return f"{float(value):.12g}"


def candidate_key(row):
    return (
        str(row.get("sweep_family", "main_ablation")),
        str(row["beta_update"]),
        int(row["T"]),
        float(row["alpha"]),
        float(row["eta"]),
        float(row["rho"]),
        optional_float_key(row.get("beta_projection_radius")),
        float(row["theta_lr"]),
        int(row["theta_inner_steps"]),
        float(row["theta_lambda"]),
        str(row["theta_mode"]),
        str(row["policy_gradient"]),
    )


def make_candidate(
    beta_update,
    eta,
    rho,
    beta_projection_radius=None,
    sweep_family="main_ablation",
    T=BASE_T,
):
    return {
        "sweep_family": str(sweep_family),
        "problem": "deterministic_10grid",
        "beta_update": str(beta_update),
        "T": int(T),
        "alpha": float(BASE_ALPHA),
        "eta": float(eta),
        "rho": float(rho),
        "beta_projection_radius": beta_projection_radius,
        "theta_lr": float(BASE_THETA_LR),
        "theta_inner_steps": int(BASE_THETA_INNER_STEPS),
        "theta_lambda": float(BASE_THETA_LAMBDA),
        "theta_mode": BASE_THETA_MODE,
        "theta_optimizer": BASE_THETA_OPTIMIZER,
        "theta_start_mode": BASE_THETA_START_MODE,
        "theta_include_beta_cov": BASE_THETA_INCLUDE_BETA_COV,
        "policy_optimizer": BASE_POLICY_OPTIMIZER,
        "policy_gradient": BASE_POLICY_GRADIENT,
        "reinforce_samples": int(BASE_REINFORCE_SAMPLES),
        "state_weight_update": BASE_STATE_WEIGHT_UPDATE,
        "intended_prob": float(INTENDED_PROB),
        "dataset_path": str(DATASET_PATH),
    }


def all_ablation_candidates():
    candidates = [
        make_candidate(
            beta_update="fogas_full",
            eta=BASE_ETA,
            rho=BASE_RHO,
        )
    ]

    for eta, rho in itertools.product(DIAG_ETA_GRID, DIAG_RHO_GRID):
        candidates.append(make_candidate("fogas_diag", eta=eta, rho=rho))

    for eta, radius in itertools.product(PROJECTED_ETA_GRID, PROJECTION_RADIUS_GRID):
        candidates.append(
            make_candidate(
                "projected_gradient",
                eta=eta,
                rho=BASE_RHO,
                beta_projection_radius=radius,
            )
        )

    candidates.append(make_candidate("fenchel_br", eta=BASE_ETA, rho=BASE_RHO))

    for eta in MIRROR_ETA_GRID:
        candidates.append(make_candidate("fenchel_mirror", eta=eta, rho=BASE_RHO))

    for eta, rho in itertools.product(SENSITIVITY_ETA_GRID, SENSITIVITY_RHO_GRID):
        candidates.append(
            make_candidate(
                "fogas_full",
                eta=eta,
                rho=rho,
                sweep_family="eta_rho_sweep",
            )
        )

    return candidates


def blank_ablation_metrics():
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


def base_row(config, device, seed, status="ok", error=""):
    row = dict(config)
    row["beta_projection_radius"] = (
        np.nan
        if none_if_nan(config.get("beta_projection_radius")) is None
        else float(config["beta_projection_radius"])
    )
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
            "seed": int(seed),
            "device": str(device),
            "status": status,
            "error": error,
        }
    )
    row.update(blank_ablation_metrics())
    return row


def evaluate_policy(planner, evaluator, policy_mode, d_star, v_star):
    pi = evaluator.get_policy(policy_mode)
    v_pi, _ = planner.evaluate_policy(pi)
    v_gap = float((d_star * (v_star - v_pi.detach().cpu())).sum().item())
    return {
        f"{policy_mode}_success_rate": finite_float(
            evaluator.success_rate(
                goal_state=GOAL_GRID,
                policy_mode=policy_mode,
                num_trajectories=NUM_TRAJECTORIES,
                max_steps=MAX_STEPS,
                seed=SEED,
                terminal_states=TERMINAL_STATES,
            )["policy"]
        ),
        f"{policy_mode}_avg_return": finite_float(
            evaluator.average_return(
                policy_mode=policy_mode,
                num_trajectories=NUM_TRAJECTORIES,
                max_steps=MAX_STEPS,
                seed=SEED,
                terminal_states=TERMINAL_STATES,
            )["policy"]
        ),
        f"{policy_mode}_v_x0": finite_float(v_pi[planner.x0].detach().cpu().item()),
        f"{policy_mode}_v_gap": finite_float(v_gap),
    }


def add_final_diagnostics(row, solver):
    diagnostics = solver.get_diagnostics() or []
    if not diagnostics:
        return
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


def run_ablation_candidate(config, mdp, planner, device, d_star, v_star):
    start = time.perf_counter()
    row = base_row(config, device=device, seed=SEED)
    try:
        solver = make_solver(config, device=device, seed=SEED)
        evaluator = FOGASEvaluator(solver=solver, mdp=mdp, planner=planner)
        solver.run(
            alpha=config["alpha"],
            eta=config["eta"],
            rho=config["rho"],
            T=config["T"],
            theta_mode=config["theta_mode"],
            theta_lr=config["theta_lr"],
            theta_inner_steps=config["theta_inner_steps"],
            theta_lambda=config["theta_lambda"],
            policy_optimizer=config["policy_optimizer"],
            policy_gradient=config["policy_gradient"],
            reinforce_samples=config["reinforce_samples"],
            tqdm_print=False,
            verbose=False,
            state_weight_update=config["state_weight_update"],
            beta_update=config["beta_update"],
            beta_projection_radius=config["beta_projection_radius"],
        )
        row.update(evaluate_policy(planner, evaluator, "solver", d_star, v_star))
        row.update(evaluate_policy(planner, evaluator, "greedy", d_star, v_star))
        add_final_diagnostics(row, solver)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        row["elapsed_seconds"] = float(time.perf_counter() - start)
    return row


def run_ablation_candidate_worker(payload):
    config, device_str, torch_threads = payload
    configure_worker_threads(torch_threads)
    set_seed(SEED)
    device = torch.device(device_str)
    mdp, planner = build_mdp(device)
    d_star = planner.state_mu_star.detach().cpu()
    v_star = planner.v_star.detach().cpu()
    return run_ablation_candidate(config, mdp, planner, device, d_star, v_star)


def failed_ablation_row(config, exc, device=None):
    return base_row(config, device or DEVICE, seed=SEED, status="failed", error=repr(exc))


def ordered_ablation_frame(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.sort_values(
        by=RANK_COLUMNS,
        ascending=RANK_ASCENDING,
        na_position="last",
    ).reset_index(drop=True)


def build_stats_frame(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df
    metrics = ["solver_success_rate", "greedy_success_rate", "solver_avg_return", "greedy_avg_return"]
    rows = []
    for (sweep_family, beta_update), group in df.groupby(["sweep_family", "beta_update"], dropna=False):
        ok = group[group["status"] == "ok"].copy()
        row = {
            "sweep_family": sweep_family,
            "beta_update": beta_update,
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
        by=["sweep_family", "greedy_success_rate_best", "solver_success_rate_best"],
        ascending=[True, False, False],
        na_position="last",
    )


def save_ablation_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = ordered_ablation_frame(results)
    df.to_csv(ABLATION_CSV, index=False)
    if not df.empty:
        ok = df[df["status"] == "ok"]
        if not ok.empty:
            ok.head(1).to_csv(BEST_CSV, index=False)
    stats = build_stats_frame(results)
    if not stats.empty:
        stats.to_csv(STATS_CSV, index=False)


def load_existing_ablation_results(resume):
    if not resume or not ABLATION_CSV.exists():
        return [], set()
    try:
        df = pd.read_csv(ABLATION_CSV)
    except pd.errors.EmptyDataError:
        return [], set()
    rows = df.to_dict("records")
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    completed = {candidate_key(row) for row in ok.to_dict("records")}
    return rows, completed


def checkpoint_steps(T, checkpoint_count=CHECKPOINT_COUNT):
    T = int(T)
    checkpoint_count = max(1, int(checkpoint_count))
    steps = sorted({max(1, int(round(T * idx / checkpoint_count))) for idx in range(1, checkpoint_count + 1)})
    if steps[-1] != T:
        steps[-1] = T
    return steps


def evaluate_policy_tensor(mdp, planner, pi, d_star, v_star, seed, num_trajectories, max_steps):
    pi = pi.to(dtype=torch.float64, device=mdp.r.device)
    returns = []
    successes = 0
    terminal_states = {int(state) for state in TERMINAL_STATES}

    for idx in range(int(num_trajectories)):
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

        for step in range(int(max_steps)):
            action_probs = pi[state]
            prob_sum = action_probs.sum()
            if prob_sum <= 0:
                raise ValueError(f"Policy probabilities at state {state} must have positive mass.")
            action = int(torch.multinomial(action_probs / prob_sum, num_samples=1).item())
            row_idx = state * mdp.A + action
            reward = mdp.r[row_idx]
            reward_value = float(reward.item() if isinstance(reward, torch.Tensor) else reward)
            total_return += (float(mdp.gamma) ** step) * reward_value
            transition_probs = mdp.P[row_idx].to(dtype=torch.float64, device=mdp.r.device)
            next_state = int(torch.multinomial(transition_probs, num_samples=1).item())
            reached_goal = next_state == GOAL_GRID
            if next_state in terminal_states:
                break
            state = next_state

        returns.append(total_return)
        successes += int(reached_goal)

    returns = np.asarray(returns, dtype=float)
    v_pi, _ = planner.evaluate_policy(pi)
    v_gap = float((d_star * (v_star - v_pi.detach().cpu())).sum().item())
    return {
        "avg_return": float(returns.mean()) if returns.size else 0.0,
        "return_std": float(returns.std(ddof=1)) if returns.size > 1 else 0.0,
        "return_sem": float(returns.std(ddof=1) / math.sqrt(returns.size)) if returns.size > 1 else 0.0,
        "success_rate": float(successes / num_trajectories) if num_trajectories else 0.0,
        "v_x0": float(v_pi[planner.x0].detach().cpu().item()),
        "v_gap": finite_float(v_gap),
    }


def normalize_beta_csv(df):
    if "status" in df.columns:
        df = df[df["status"].eq("ok")].copy()
    if "sweep_family" not in df.columns:
        df["sweep_family"] = "main_ablation"
    return df


def select_curve_configs(beta_csv, require_complete=True):
    if not beta_csv.exists():
        raise FileNotFoundError(f"Missing beta ablation CSV: {beta_csv}")

    df = normalize_beta_csv(pd.read_csv(beta_csv))
    df = df[df["beta_update"].isin(BETA_UPDATE_ORDER)].copy()
    if df.empty:
        raise ValueError("No successful beta-ablation rows found.")

    rank_cols = [col for col in RANK_COLUMNS if col in df.columns]
    rank_ascending = RANK_ASCENDING[: len(rank_cols)]
    df["beta_update"] = pd.Categorical(df["beta_update"], categories=BETA_UPDATE_ORDER, ordered=True)
    best = (
        df.sort_values(["beta_update", *rank_cols], ascending=[True, *rank_ascending])
        .groupby(["beta_update"], as_index=False, observed=True)
        .head(1)
        .sort_values("beta_update")
        .reset_index(drop=True)
    )

    configs = []
    for row in best.to_dict("records"):
        config = {
            "sweep_family": str(row.get("sweep_family", "main_ablation")),
            "problem": "deterministic_10grid",
            "dataset_path": str(DATASET_PATH),
            "beta_update": str(row["beta_update"]),
            "beta_projection_radius": none_if_nan(row.get("beta_projection_radius")),
            "T": int(row["T"]),
            "alpha": float(row["alpha"]),
            "eta": float(row["eta"]),
            "rho": float(row["rho"]),
            "theta_lr": float(row["theta_lr"]),
            "theta_inner_steps": int(row["theta_inner_steps"]),
            "theta_lambda": float(row["theta_lambda"]),
            "theta_mode": str(row["theta_mode"]),
            "theta_optimizer": str(row["theta_optimizer"]),
            "theta_start_mode": str(row["theta_start_mode"]),
            "theta_include_beta_cov": parse_bool(row["theta_include_beta_cov"]),
            "policy_optimizer": str(row["policy_optimizer"]),
            "policy_gradient": str(row["policy_gradient"]),
            "reinforce_samples": int(row["reinforce_samples"]),
            "state_weight_update": str(row["state_weight_update"]),
            "intended_prob": float(INTENDED_PROB),
            "source_run_idx": int(row["run_idx"]) if "run_idx" in row and pd.notna(row["run_idx"]) else -1,
            "source_greedy_success_rate": finite_float(row.get("greedy_success_rate")),
            "source_solver_success_rate": finite_float(row.get("solver_success_rate")),
            "source_greedy_avg_return": finite_float(row.get("greedy_avg_return")),
            "source_solver_avg_return": finite_float(row.get("solver_avg_return")),
        }
        configs.append(config)

    found = {config["beta_update"] for config in configs}
    missing = [update for update in BETA_UPDATE_ORDER if update not in found]
    if require_complete and missing:
        raise ValueError(f"Missing best rows for beta updates: {missing}")
    return configs, missing


def curve_key(config, seed):
    return (str(config["beta_update"]), int(seed))


def load_existing_checkpoint_results(output_csv, resume, seed=None, key_cols=None):
    if not resume or not output_csv.exists():
        return [], set()
    try:
        df = pd.read_csv(output_csv)
    except pd.errors.EmptyDataError:
        return [], set()
    rows = df.to_dict("records")
    completed = set()
    ok = df[df["status"].eq("ok")].copy() if "status" in df.columns else df.copy()
    if ok.empty:
        return rows, completed
    group_cols = key_cols
    if group_cols is None:
        group_cols = ["beta_update", "seed"]
        if "rho" in ok.columns:
            group_cols = ["beta_update", "rho", "seed"]
    for key, group in ok.groupby(group_cols):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row_seed = int(key_tuple[-1])
        if seed is not None and row_seed != int(seed):
            continue
        T = int(group["T"].iloc[0])
        expected = set(checkpoint_steps(T))
        observed = {int(step) for step in group["step"].dropna().astype(int)}
        if expected.issubset(observed):
            completed.add(tuple(key_tuple))
    return rows, completed


def run_curve_config(config, seed, num_trajectories, max_steps, device):
    started = time.perf_counter()
    set_seed(seed)
    mdp, planner = build_mdp(device)
    d_star = planner.state_mu_star.detach().cpu()
    v_star = planner.v_star.detach().cpu()
    rows = []

    try:
        solver = make_solver(config, device=device, seed=seed)
        train_started = time.perf_counter()
        solver.run(
            alpha=config["alpha"],
            eta=config["eta"],
            rho=config["rho"],
            T=config["T"],
            theta_mode=config["theta_mode"],
            theta_lr=config["theta_lr"],
            theta_inner_steps=config["theta_inner_steps"],
            theta_lambda=config["theta_lambda"],
            policy_optimizer=config["policy_optimizer"],
            policy_gradient=config["policy_gradient"],
            reinforce_samples=config["reinforce_samples"],
            tqdm_print=False,
            verbose=False,
            state_weight_update=config["state_weight_update"],
            beta_update=config["beta_update"],
            beta_projection_radius=config["beta_projection_radius"],
        )
        train_elapsed = float(time.perf_counter() - train_started)
        diagnostics_history = solver.get_diagnostics() or []

        for checkpoint_idx, step in enumerate(checkpoint_steps(config["T"]), start=1):
            history_idx = int(step) - 1
            pi = solver._linear_policy_matrix(solver.psi_history[history_idx])
            stats = evaluate_policy_tensor(
                mdp=mdp,
                planner=planner,
                pi=pi,
                d_star=d_star,
                v_star=v_star,
                seed=seed,
                num_trajectories=num_trajectories,
                max_steps=max_steps,
            )
            diagnostics = diagnostics_history[history_idx] if history_idx < len(diagnostics_history) else {}
            row = dict(config)
            row.update(
                {
                    "seed": int(seed),
                    "num_trajectories": int(num_trajectories),
                    "max_steps": int(max_steps),
                    "checkpoint_count": int(CHECKPOINT_COUNT),
                    "step": int(step),
                    "checkpoint_idx": int(checkpoint_idx),
                    "avg_return": finite_float(stats["avg_return"]),
                    "return_std": finite_float(stats["return_std"]),
                    "return_sem": finite_float(stats["return_sem"]),
                    "success_rate": finite_float(stats["success_rate"]),
                    "v_x0": finite_float(stats["v_x0"]),
                    "v_gap": finite_float(stats["v_gap"]),
                    "total_loss": finite_float(diagnostics.get("total_loss")),
                    "policy_objective": finite_float(diagnostics.get("policy_objective")),
                    "beta_objective": finite_float(diagnostics.get("beta_objective")),
                    "q_objective": finite_float(diagnostics.get("q_objective")),
                    "theta_norm": finite_float(diagnostics.get("theta_norm")),
                    "policy_grad_norm": finite_float(diagnostics.get("policy_grad_norm")),
                    "beta_grad_norm": finite_float(diagnostics.get("beta_grad_norm")),
                    "theta_grad_norm": finite_float(diagnostics.get("theta_grad_norm")),
                    "train_elapsed_seconds": finite_float(train_elapsed),
                    "elapsed_seconds": finite_float(time.perf_counter() - started),
                    "device": str(device),
                    "status": "ok",
                    "error": "",
                }
            )
            rows.append(row)
    except Exception as exc:
        row = dict(config)
        row.update(
            {
                "seed": int(seed),
                "num_trajectories": int(num_trajectories),
                "max_steps": int(max_steps),
                "checkpoint_count": int(CHECKPOINT_COUNT),
                "step": np.nan,
                "checkpoint_idx": np.nan,
                "avg_return": np.nan,
                "return_std": np.nan,
                "return_sem": np.nan,
                "success_rate": np.nan,
                "v_x0": np.nan,
                "v_gap": np.nan,
                "train_elapsed_seconds": np.nan,
                "elapsed_seconds": finite_float(time.perf_counter() - started),
                "device": str(device),
                "status": "failed",
                "error": repr(exc),
            }
        )
        rows.append(row)

    return rows


def run_curve_config_worker(payload):
    config, seed, num_trajectories, max_steps, device_str, torch_threads = payload
    configure_worker_threads(torch_threads)
    device = torch.device(device_str)
    return run_curve_config(config, seed, num_trajectories, max_steps, device)


def save_checkpoint_results(rows, output_csv, sort_cols):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        present_cols = [col for col in sort_cols if col in df.columns]
        if present_cols:
            df = df.sort_values(present_cols, na_position="last").reset_index(drop=True)
    df.to_csv(output_csv, index=False)


def summarize_rho_results(rows):
    df = pd.DataFrame(rows)
    if df.empty or "rho" not in df.columns:
        return pd.DataFrame()
    ok = df[df["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    final = ok.sort_values("step").groupby(["rho", "seed"], as_index=False).tail(1)
    summary = (
        final.groupby("rho", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            success_rate_mean=("success_rate", "mean"),
            success_rate_std=("success_rate", "std"),
            avg_return_mean=("avg_return", "mean"),
            avg_return_std=("avg_return", "std"),
            v_x0_mean=("v_x0", "mean"),
            v_gap_mean=("v_gap", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
        )
        .sort_values(["success_rate_mean", "avg_return_mean", "rho"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return summary


def plot_curves(output_csv, plot_path):
    import matplotlib.pyplot as plt

    df = pd.read_csv(output_csv)
    ok = df[df["status"].eq("ok")].copy()
    if ok.empty:
        raise ValueError("No successful curve rows to plot.")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for update in BETA_UPDATE_ORDER:
        sub = ok[ok["beta_update"].eq(update)].sort_values("step")
        if sub.empty:
            continue
        grouped = sub.groupby("step", as_index=False).agg(
            avg_return=("avg_return", "mean"),
            return_sem=("return_sem", "mean"),
        )
        ax.plot(grouped["step"], grouped["avg_return"], label=BETA_UPDATE_LABELS.get(update, update))
        ax.fill_between(
            grouped["step"],
            grouped["avg_return"] - grouped["return_sem"].fillna(0.0),
            grouped["avg_return"] + grouped["return_sem"].fillna(0.0),
            alpha=0.15,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average discounted return")
    ax.set_title("10grid deterministic beta-update learning curves")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def add_shared_runtime_args(parser):
    parser.add_argument("--resume", action="store_true", help="Skip completed rows in existing output.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit candidate count for smoke tests.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes.")
    parser.add_argument("--torch-threads", type=int, default=1, help="Torch CPU threads per worker.")
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated devices for worker round-robin, e.g. cuda:0,cuda:1.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="10grid deterministic FinalLinearSolver beta-update ablation workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ablation = subparsers.add_parser("ablation", help="Run beta-update ablation grid search.")
    add_shared_runtime_args(ablation)

    curves = subparsers.add_parser("curves", help="Rerun best beta-update settings with checkpoints.")
    add_shared_runtime_args(curves)
    curves.add_argument("--beta-csv", type=Path, default=ABLATION_CSV, help="Ablation CSV to select from.")
    curves.add_argument("--output-csv", type=Path, default=CURVES_CSV, help="Checkpoint output CSV.")
    curves.add_argument("--plot-path", type=Path, default=CURVES_PNG, help="Output PNG for --plot.")
    curves.add_argument("--plot", action="store_true", help="Save a learning-curve PNG after running.")
    curves.add_argument("--count-only", action="store_true", help="Print selected configs and exit.")
    curves.add_argument("--seed", type=int, default=SEED, help="Training/evaluation seed.")
    curves.add_argument(
        "--num-trajectories",
        type=int,
        default=NUM_TRAJECTORIES,
        help="Evaluation rollouts per checkpoint.",
    )
    curves.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Evaluation horizon.")

    rho_sweep = subparsers.add_parser("rho-sweep", help="Run checkpointed fogas_full rho sweep.")
    add_shared_runtime_args(rho_sweep)
    rho_sweep.add_argument(
        "--rho-values",
        nargs="+",
        default=None,
        help=f"Rho values, as space-separated values or comma lists. Default: {DEFAULT_RHO_SWEEP_VALUES}",
    )
    rho_sweep.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help=f"Seeds, as space-separated values or comma lists. Default: {DEFAULT_SEEDS}",
    )
    rho_sweep.add_argument("--T", type=int, default=BASE_T, help=f"Training iterations. Default: {BASE_T}.")
    rho_sweep.add_argument(
        "--output-csv",
        type=Path,
        default=RHO_SWEEP_CSV,
        help="Checkpoint output CSV.",
    )
    rho_sweep.add_argument(
        "--summary-csv",
        type=Path,
        default=RHO_SWEEP_SUMMARY_CSV,
        help="Rho summary output CSV.",
    )
    rho_sweep.add_argument(
        "--num-trajectories",
        type=int,
        default=NUM_TRAJECTORIES,
        help="Evaluation rollouts per checkpoint.",
    )
    rho_sweep.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Evaluation horizon.")

    return parser


def run_ablation(args):
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    device_list = parse_device_list(args.devices)
    configure_worker_threads(torch_threads)
    set_seed(SEED)

    candidates_all = all_ablation_candidates()
    if args.max_runs is not None:
        candidates_all = candidates_all[: max(0, int(args.max_runs))]

    results, completed = load_existing_ablation_results(args.resume)
    candidates = [candidate for candidate in candidates_all if candidate_key(candidate) not in completed]

    print(f"Devices: {device_list}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Results: {ABLATION_CSV}")
    print(f"Stats: {STATS_CSV}")
    print(f"Workers: {workers}")
    print(f"Torch threads per worker: {torch_threads}")
    print(f"Total candidate grid size: {len(candidates_all)}")
    print(f"Candidates to run: {len(candidates)}")
    if args.resume:
        print(f"Resumed rows: {len(results)}")
    if not args.resume and ABLATION_CSV.exists():
        print("Existing output will be overwritten because --resume was not set.")

    progress = not args.no_progress
    desc = "FinalLinearSolver 10-grid tabular beta ablation"

    if workers == 1:
        device = torch.device(device_list[0])
        mdp, planner = build_mdp(device)
        d_star = planner.state_mu_star.detach().cpu()
        v_star = planner.v_star.detach().cpu()
        outer = tqdm(candidates, desc=desc, unit="run", disable=not progress)
        for run_idx, candidate in enumerate(outer, start=len(results) + 1):
            row = run_ablation_candidate(candidate, mdp, planner, device, d_star, v_star)
            row["run_idx"] = int(run_idx)
            results.append(row)
            save_ablation_results(results)
            if progress:
                outer.set_postfix(
                    {
                        "beta": row["beta_update"],
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
                executor.submit(run_ablation_candidate_worker, payload): payload[0]
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
                    row = failed_ablation_row(candidate, exc)
                row["run_idx"] = int(next_run_idx)
                next_run_idx += 1
                results.append(row)
                save_ablation_results(results)
                if progress:
                    outer.set_postfix(
                        {
                            "beta": row.get("beta_update"),
                            "greedy_success": row.get("greedy_success_rate", np.nan),
                            "status": row.get("status"),
                        }
                    )

    save_ablation_results(results)
    df = ordered_ablation_frame(results)
    ok_count = int((df["status"] == "ok").sum()) if not df.empty else 0
    failed_count = int((df["status"] == "failed").sum()) if not df.empty else 0
    print("\nAblation complete.")
    print(f"Rows saved: {len(df)}")
    print(f"Successful runs: {ok_count}")
    print(f"Failed runs: {failed_count}")
    print(f"Output CSV: {ABLATION_CSV}")
    if BEST_CSV.exists():
        print(f"Best row CSV: {BEST_CSV}")
        print(pd.read_csv(BEST_CSV).to_string(index=False))


def run_curves(args):
    configs, missing = select_curve_configs(args.beta_csv, require_complete=not args.count_only)
    if args.count_only:
        print(f"Selected configs: {len(configs)}")
        for config in configs:
            print(
                f"{config['beta_update']}: sweep={config['sweep_family']} "
                f"eta={config['eta']} rho={config['rho']} "
                f"radius={config['beta_projection_radius']} T={config['T']}"
            )
        if missing:
            print(f"Missing beta updates: {missing}")
        return

    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    device_list = parse_device_list(args.devices)
    configure_worker_threads(torch_threads)
    set_seed(args.seed)

    results, completed = load_existing_checkpoint_results(
        args.output_csv,
        args.resume,
        seed=args.seed,
        key_cols=["beta_update", "seed"],
    )
    configs_to_run = [config for config in configs if curve_key(config, args.seed) not in completed]
    if args.max_runs is not None:
        configs_to_run = configs_to_run[: max(0, int(args.max_runs))]

    print(f"Devices: {device_list}")
    print(f"Source beta CSV: {args.beta_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Configs to run: {len(configs_to_run)}")
    if args.resume:
        print(f"Resumed rows: {len(results)}")

    progress = not args.no_progress
    desc = "FinalLinearSolver 10-grid beta curves"

    if workers == 1:
        device = torch.device(device_list[0])
        outer = tqdm(configs_to_run, desc=desc, unit="config", disable=not progress)
        for config in outer:
            rows = run_curve_config(
                config=config,
                seed=args.seed,
                num_trajectories=args.num_trajectories,
                max_steps=args.max_steps,
                device=device,
            )
            results.extend(rows)
            save_checkpoint_results(results, args.output_csv, ["beta_update", "seed", "step"])
            if progress:
                last = rows[-1]
                outer.set_postfix({"beta": last.get("beta_update"), "status": last.get("status")})
    else:
        payloads = [
            (
                config,
                args.seed,
                args.num_trajectories,
                args.max_steps,
                device_list[i % len(device_list)],
                torch_threads,
            )
            for i, config in enumerate(configs_to_run)
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_config = {
                executor.submit(run_curve_config_worker, payload): payload[0]
                for payload in payloads
            }
            outer = tqdm(
                as_completed(future_to_config),
                total=len(future_to_config),
                desc=desc,
                unit="config",
                disable=not progress,
            )
            for future in outer:
                try:
                    rows = future.result()
                except Exception as exc:
                    config = future_to_config[future]
                    row = dict(config)
                    row.update({"seed": args.seed, "status": "failed", "error": repr(exc)})
                    rows = [row]
                results.extend(rows)
                save_checkpoint_results(results, args.output_csv, ["beta_update", "seed", "step"])
                if progress:
                    last = rows[-1]
                    outer.set_postfix({"beta": last.get("beta_update"), "status": last.get("status")})

    save_checkpoint_results(results, args.output_csv, ["beta_update", "seed", "step"])
    if args.plot:
        plot_curves(args.output_csv, args.plot_path)
        print(f"Plot: {args.plot_path}")
    ok_count = sum(1 for row in results if row.get("status") == "ok")
    failed_count = sum(1 for row in results if row.get("status") == "failed")
    print("\nCurves complete.")
    print(f"Rows saved: {len(results)}")
    print(f"Successful rows: {ok_count}")
    print(f"Failed rows: {failed_count}")
    print(f"Output CSV: {args.output_csv}")


def run_rho_sweep(args):
    rho_values = (
        DEFAULT_RHO_SWEEP_VALUES if args.rho_values is None else parse_float_list(args.rho_values)
    )
    seeds = DEFAULT_SEEDS if args.seeds is None else parse_int_list(args.seeds)
    candidates = []
    for rho, seed in itertools.product(rho_values, seeds):
        config = make_candidate(
            beta_update="fogas_full",
            eta=BASE_ETA,
            rho=float(rho),
            sweep_family="rho_sweep",
            T=int(args.T),
        )
        candidates.append((config, int(seed)))
    if args.max_runs is not None:
        candidates = candidates[: max(0, int(args.max_runs))]

    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    device_list = parse_device_list(args.devices)
    configure_worker_threads(torch_threads)
    set_seed(SEED)

    results, completed = load_existing_checkpoint_results(
        args.output_csv,
        args.resume,
        seed=None,
        key_cols=["beta_update", "rho", "seed"],
    )
    candidates_to_run = [
        (config, seed)
        for config, seed in candidates
        if ("fogas_full", float(config["rho"]), int(seed)) not in completed
    ]

    print(f"Devices: {device_list}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Rho values: {rho_values}")
    print(f"Seeds: {seeds}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Summary CSV: {args.summary_csv}")
    print(f"Candidates to run: {len(candidates_to_run)}")

    progress = not args.no_progress
    desc = "FinalLinearSolver 10-grid rho sweep"

    if workers == 1:
        device = torch.device(device_list[0])
        outer = tqdm(candidates_to_run, desc=desc, unit="run", disable=not progress)
        for config, seed in outer:
            rows = run_curve_config(
                config=config,
                seed=seed,
                num_trajectories=args.num_trajectories,
                max_steps=args.max_steps,
                device=device,
            )
            results.extend(rows)
            save_checkpoint_results(results, args.output_csv, ["rho", "seed", "step"])
            summary = summarize_rho_results(results)
            if not summary.empty:
                args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
                summary.to_csv(args.summary_csv, index=False)
            if progress:
                last = rows[-1]
                outer.set_postfix({"rho": last.get("rho"), "seed": last.get("seed"), "status": last.get("status")})
    else:
        payloads = [
            (
                config,
                seed,
                args.num_trajectories,
                args.max_steps,
                device_list[i % len(device_list)],
                torch_threads,
            )
            for i, (config, seed) in enumerate(candidates_to_run)
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_candidate = {
                executor.submit(run_curve_config_worker, payload): (payload[0], payload[1])
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
                    rows = future.result()
                except Exception as exc:
                    config, seed = future_to_candidate[future]
                    row = dict(config)
                    row.update({"seed": seed, "status": "failed", "error": repr(exc)})
                    rows = [row]
                results.extend(rows)
                save_checkpoint_results(results, args.output_csv, ["rho", "seed", "step"])
                summary = summarize_rho_results(results)
                if not summary.empty:
                    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
                    summary.to_csv(args.summary_csv, index=False)
                if progress:
                    last = rows[-1]
                    outer.set_postfix(
                        {"rho": last.get("rho"), "seed": last.get("seed"), "status": last.get("status")}
                    )

    save_checkpoint_results(results, args.output_csv, ["rho", "seed", "step"])
    summary = summarize_rho_results(results)
    if not summary.empty:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_csv, index=False)
    ok_count = sum(1 for row in results if row.get("status") == "ok")
    failed_count = sum(1 for row in results if row.get("status") == "failed")
    print("\nRho sweep complete.")
    print(f"Rows saved: {len(results)}")
    print(f"Successful rows: {ok_count}")
    print(f"Failed rows: {failed_count}")
    print(f"Output CSV: {args.output_csv}")
    if args.summary_csv.exists():
        print(f"Summary CSV: {args.summary_csv}")


def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = build_arg_parser().parse_args()
    if args.command == "ablation":
        run_ablation(args)
    elif args.command == "curves":
        run_curves(args)
    elif args.command == "rho-sweep":
        run_rho_sweep(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
