"""Build the thesis learning curves for the 5 x 5 beta-update ablation.

Scientific role
---------------
The main beta grid selects configurations by their final evaluation. This
thesis-facing follow-up reruns the best successful setting for every occupancy
update on each 5 x 5 problem and records intermediate evaluations, making the
learning speed and stability shown in ``notebooks/ablations.ipynb`` visible.

Inputs and outputs
------------------
The script reads
``data/results/generalization/ablations/beta/final_linear_5grid_tabular_beta_ablation.csv``
and the two fixed 5 x 5 datasets. It evaluates the solver policy every
``T / 20`` iterations and writes the checkpoint table and optional summary
figure back to the beta-ablation result directory.

Run this file directly from the repository root after the beta grid search.
Use ``--help`` to inspect selection, output, and plotting options; the default
fixed seed keeps the selected learning curves reproducible.
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
BETA_ABLATION_CSV = RESULTS_DIR / "final_linear_5grid_tabular_beta_ablation.csv"
OUTPUT_CSV = RESULTS_DIR / "final_linear_5grid_beta_update_learning_curves.csv"
PLOT_PATH = RESULTS_DIR / "final_linear_5grid_beta_update_learning_curves.png"

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
CHECKPOINT_COUNT = 20

PROBLEM_ORDER = ["deterministic", "stochastic"]
UPDATE_ORDER = [
    "fogas_full",
    "fogas_diag",
    "projected_gradient",
    "metric_no_stabilization",
    "euclidean_stabilized",
    "fenchel_br",
    "fenchel_mirror",
]
UPDATE_LABELS = {
    "fogas_full": "FOGAS full",
    "fogas_diag": "FOGAS diag",
    "projected_gradient": "Projected grad",
    "metric_no_stabilization": "Metric no stab.",
    "euclidean_stabilized": "Euclidean stabilized",
    "fenchel_br": "Fenchel BR",
    "fenchel_mirror": "Fenchel mirror",
}
RANK_COLUMNS = [
    "greedy_success_rate",
    "solver_success_rate",
    "greedy_avg_reward",
    "solver_avg_reward",
    "elapsed_seconds",
]
RANK_ASCENDING = [False, False, False, False, True]

PROBLEMS = {
    "deterministic": {
        "dataset_path": DATASETS_DIR / "5grid.csv",
        "gamma": 0.99,
        "terminal_states": {GOAL_GRID},
        "stochastic": False,
    },
    "stochastic": {
        "dataset_path": DATASETS_DIR / "5grid_stochastic.csv",
        "gamma": 0.9,
        "terminal_states": {GOAL_GRID, PIT_GRID},
        "stochastic": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build beta-update learning curves from best ablation settings."
    )
    parser.add_argument(
        "--beta-csv",
        type=Path,
        default=BETA_ABLATION_CSV,
        help="CSV produced by grid_search_final_linear_5grid_tabular_beta_ablation.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="Where checkpoint curve rows are written.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=PLOT_PATH,
        help="Where --plot writes the PNG.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Single training/evaluation seed used for all ten runs.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=NUM_TRAJECTORIES,
        help="Evaluation rollouts per checkpoint; std is computed over these returns.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Evaluation horizon per rollout.",
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
        "--resume",
        action="store_true",
        help="Skip problem/update pairs already completed in the output CSV.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a deterministic/stochastic two-panel PNG after running.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print selected configurations and exit.",
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


def make_solver(config, device, seed):
    u_features = TabularFeatures(N, A)
    q_features = TabularFeatures(N, A)
    policy_features = TabularFeatures(N, A)

    return FinalLinearSolver(
        n_states=N,
        n_actions=A,
        gamma=float(config["gamma"]),
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
        beta_projection_radius=config["beta_projection_radius"],
    )


def checkpoint_steps(T):
    T = int(T)
    interval = max(1, T // CHECKPOINT_COUNT)
    steps = list(range(interval, T + 1, interval))
    if steps[-1] != T:
        steps.append(T)
    return sorted(set(steps))


def finite_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def none_if_nan(value):
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def normalize_beta_csv(df):
    rename = {
        "solver_avg_return": "solver_avg_reward",
        "greedy_avg_return": "greedy_avg_reward",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "status" in df.columns:
        df = df[df["status"].eq("ok")].copy()
    missing = [
        c
        for c in [
            "problem",
            "beta_update",
            "T",
            "alpha",
            "eta",
            "rho",
            "theta_lambda",
            "theta_lr",
            "theta_inner_steps",
            "policy_optimizer",
            "policy_gradient",
            "reinforce_samples",
            "state_weight_update",
            "theta_mode",
            "theta_optimizer",
            "theta_start_mode",
            "theta_include_beta_cov",
            "greedy_success_rate",
            "solver_success_rate",
            "greedy_avg_reward",
            "solver_avg_reward",
        ]
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required beta-ablation columns: {missing}")
    if "beta_projection_radius" not in df.columns:
        df["beta_projection_radius"] = np.nan
    return df


def select_best_configs(beta_csv):
    if not beta_csv.exists():
        raise FileNotFoundError(f"Missing beta ablation CSV: {beta_csv}")

    df = normalize_beta_csv(pd.read_csv(beta_csv))
    df = df[
        df["problem"].isin(PROBLEM_ORDER)
        & df["beta_update"].isin(UPDATE_ORDER)
    ].copy()
    if df.empty:
        raise ValueError("No successful beta-ablation rows found for requested updates/problems.")

    rank_cols = [c for c in RANK_COLUMNS if c in df.columns]
    rank_ascending = RANK_ASCENDING[: len(rank_cols)]
    df["problem"] = pd.Categorical(df["problem"], categories=PROBLEM_ORDER, ordered=True)
    df["beta_update"] = pd.Categorical(df["beta_update"], categories=UPDATE_ORDER, ordered=True)

    best = (
        df.sort_values(
            ["problem", "beta_update", *rank_cols],
            ascending=[True, True, *rank_ascending],
        )
        .groupby(["problem", "beta_update"], as_index=False, observed=True)
        .head(1)
        .sort_values(["problem", "beta_update"])
        .reset_index(drop=True)
    )

    configs = []
    for row in best.to_dict("records"):
        problem_name = str(row["problem"])
        config = {
            "problem": problem_name,
            "gamma": float(PROBLEMS[problem_name]["gamma"]),
            "dataset_path": str(PROBLEMS[problem_name]["dataset_path"]),
            "beta_update": str(row["beta_update"]),
            "beta_projection_radius": none_if_nan(row.get("beta_projection_radius")),
            "T": int(row["T"]),
            "alpha": float(row["alpha"]),
            "eta": float(row["eta"]),
            "rho": float(row["rho"]),
            "theta_lambda": float(row["theta_lambda"]),
            "theta_lr": float(row["theta_lr"]),
            "theta_inner_steps": int(row["theta_inner_steps"]),
            "policy_optimizer": str(row["policy_optimizer"]),
            "policy_gradient": str(row["policy_gradient"]),
            "reinforce_samples": int(row["reinforce_samples"]),
            "state_weight_update": str(row["state_weight_update"]),
            "theta_mode": str(row["theta_mode"]),
            "theta_optimizer": str(row["theta_optimizer"]),
            "theta_start_mode": str(row["theta_start_mode"]),
            "theta_include_beta_cov": parse_bool(row["theta_include_beta_cov"]),
            "source_run_idx": int(row["run_idx"]) if "run_idx" in row and pd.notna(row["run_idx"]) else -1,
            "source_greedy_success_rate": finite_float(row.get("greedy_success_rate")),
            "source_solver_success_rate": finite_float(row.get("solver_success_rate")),
            "source_greedy_avg_reward": finite_float(row.get("greedy_avg_reward")),
            "source_solver_avg_reward": finite_float(row.get("solver_avg_reward")),
        }
        configs.append(config)

    expected = len(PROBLEM_ORDER) * len(UPDATE_ORDER)
    if len(configs) != expected:
        found = {(config["problem"], config["beta_update"]) for config in configs}
        missing = [
            (problem, update)
            for problem in PROBLEM_ORDER
            for update in UPDATE_ORDER
            if (problem, update) not in found
        ]
        raise ValueError(f"Expected {expected} configs, found {len(configs)}. Missing: {missing}")
    return configs


def config_key(config, seed):
    return (str(config["problem"]), str(config["beta_update"]), int(seed))


def load_existing_results(output_csv, resume, seed):
    if not resume or not output_csv.exists():
        return [], set()
    df = pd.read_csv(output_csv)
    rows = df.to_dict("records")
    completed = set()
    if df.empty:
        return rows, completed
    ok = df[df["status"].eq("ok")].copy() if "status" in df.columns else df.copy()
    for (problem, update, row_seed), group in ok.groupby(["problem", "beta_update", "seed"]):
        if int(row_seed) != int(seed):
            continue
        T = int(group["T"].iloc[0])
        expected = set(checkpoint_steps(T))
        observed = {int(step) for step in group["step"].dropna().astype(int)}
        if expected.issubset(observed):
            completed.add((str(problem), str(update), int(row_seed)))
    return rows, completed


def evaluate_policy_rollouts(mdp, pi, terminal_states, seed, num_trajectories, max_steps):
    pi = pi.to(dtype=torch.float64, device=mdp.r.device)
    rewards = []
    successes = 0
    terminal_states = {int(state) for state in terminal_states}

    for idx in range(num_trajectories):
        current_seed = int(seed) + idx
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

        for _step in range(max_steps):
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

    rewards = np.asarray(rewards, dtype=float)
    return {
        "avg_reward": float(rewards.mean()) if rewards.size else 0.0,
        "reward_std": float(rewards.std(ddof=1)) if rewards.size > 1 else 0.0,
        "reward_sem": float(rewards.std(ddof=1) / math.sqrt(rewards.size)) if rewards.size > 1 else 0.0,
        "success_rate": float(successes / num_trajectories) if num_trajectories else 0.0,
    }


def run_config(config, seed, num_trajectories, max_steps, device):
    started = time.perf_counter()
    set_seed(seed)
    mdp, _planner = build_mdp(config["problem"], device)
    terminal_states = PROBLEMS[config["problem"]]["terminal_states"]
    rows = []

    try:
        solver = make_solver(config=config, device=device, seed=seed)
        train_started = time.perf_counter()
        solver.run(
            alpha=config["alpha"],
            eta=config["eta"],
            rho=config["rho"],
            T=config["T"],
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
            stats = evaluate_policy_rollouts(
                mdp=mdp,
                pi=pi,
                terminal_states=terminal_states,
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
                    "avg_reward": finite_float(stats["avg_reward"]),
                    "reward_std": finite_float(stats["reward_std"]),
                    "reward_sem": finite_float(stats["reward_sem"]),
                    "success_rate": finite_float(stats["success_rate"]),
                    "total_loss": finite_float(diagnostics.get("total_loss")),
                    "policy_objective": finite_float(diagnostics.get("policy_objective")),
                    "beta_objective": finite_float(diagnostics.get("beta_objective")),
                    "q_objective": finite_float(diagnostics.get("q_objective")),
                    "theta_norm": finite_float(diagnostics.get("theta_norm")),
                    "policy_grad_norm": finite_float(diagnostics.get("policy_grad_norm")),
                    "beta_grad_norm": finite_float(diagnostics.get("beta_grad_norm")),
                    "theta_grad_norm": finite_float(diagnostics.get("theta_grad_norm")),
                    "device": str(device),
                    "status": "ok",
                    "error": "",
                    "train_elapsed_seconds": train_elapsed,
                    "elapsed_seconds": float(time.perf_counter() - started),
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
                "avg_reward": np.nan,
                "reward_std": np.nan,
                "reward_sem": np.nan,
                "success_rate": np.nan,
                "device": str(device),
                "status": "failed",
                "error": repr(exc),
                "train_elapsed_seconds": np.nan,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
        rows.append(row)

    return rows


def run_config_worker(payload):
    config, seed, num_trajectories, max_steps, device_str, torch_threads = payload
    configure_worker_threads(torch_threads)
    return run_config(
        config=config,
        seed=seed,
        num_trajectories=num_trajectories,
        max_steps=max_steps,
        device=torch.device(device_str),
    )


def failed_worker_rows(config, seed, exc):
    row = dict(config)
    row.update(
        {
            "seed": int(seed),
            "status": "failed",
            "error": repr(exc),
            "step": np.nan,
            "checkpoint_idx": np.nan,
            "avg_reward": np.nan,
            "reward_std": np.nan,
            "reward_sem": np.nan,
            "success_rate": np.nan,
            "device": str(DEVICE),
        }
    )
    return [row]


def ordered_results_frame(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df
    df["problem"] = pd.Categorical(df["problem"], categories=PROBLEM_ORDER, ordered=True)
    df["beta_update"] = pd.Categorical(df["beta_update"], categories=UPDATE_ORDER, ordered=True)
    return df.sort_values(
        ["problem", "beta_update", "seed", "step"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def save_results(results, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ordered_results_frame(results).to_csv(output_csv, index=False)


def plot_results(output_csv, plot_path):
    import matplotlib.pyplot as plt

    df = pd.read_csv(output_csv)
    ok = df[df["status"].eq("ok")].copy()
    if ok.empty:
        raise ValueError(f"No successful rows to plot in {output_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, problem in zip(axes, PROBLEM_ORDER):
        sub = ok[ok["problem"].eq(problem)].copy()
        for update in UPDATE_ORDER:
            line = sub[sub["beta_update"].eq(update)].sort_values("step")
            if line.empty:
                continue
            x = line["step"].to_numpy(dtype=float)
            y = line["avg_reward"].to_numpy(dtype=float)
            err = line["reward_std"].to_numpy(dtype=float)
            label = UPDATE_LABELS.get(update, update)
            ax.plot(x, y, linewidth=2, label=label)
            ax.fill_between(x, y - err, y + err, alpha=0.15)

        ax.set_title(problem.capitalize())
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Average Episode Reward")
        ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=True, fontsize=9)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    return plot_path


def print_selected_configs(configs):
    cols = [
        "problem",
        "beta_update",
        "T",
        "alpha",
        "eta",
        "rho",
        "beta_projection_radius",
        "theta_lambda",
        "theta_lr",
        "theta_inner_steps",
        "source_greedy_success_rate",
        "source_solver_avg_reward",
    ]
    print(pd.DataFrame(configs)[cols].to_string(index=False))


def main():
    args = parse_args()
    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    configure_worker_threads(torch_threads)
    set_seed(args.seed)

    configs_all = select_best_configs(args.beta_csv)
    for config in configs_all:
        dataset_path = Path(config["dataset_path"])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if args.count_only:
        print_selected_configs(configs_all)
        print(f"\nRuns: {len(configs_all)}")
        print(f"Checkpoint rows if all successful: {sum(len(checkpoint_steps(c['T'])) for c in configs_all)}")
        return

    results, completed = load_existing_results(args.output_csv, args.resume, args.seed)
    configs = [config for config in configs_all if config_key(config, args.seed) not in completed]

    print(f"Using device: {DEVICE}")
    print(f"Seed: {args.seed}")
    print(f"Beta CSV: {args.beta_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Workers: {workers}")
    print(f"Torch threads per worker: {torch_threads}")
    print(f"Runs selected: {len(configs_all)}")
    print(f"Runs to execute: {len(configs)}")
    if args.resume:
        print(f"Resumed rows: {len(results)}")
    if not args.resume and args.output_csv.exists():
        print("Existing output will be overwritten because --resume was not set.")

    progress = not args.no_progress
    desc = "FinalLinearSolver beta-update curves"

    if workers == 1:
        outer = tqdm(configs, desc=desc, unit="run", disable=not progress)
        for run_idx, config in enumerate(outer, start=1):
            rows = run_config(
                config=config,
                seed=args.seed,
                num_trajectories=args.num_trajectories,
                max_steps=args.max_steps,
                device=DEVICE,
            )
            for row in rows:
                row["run_idx"] = int(run_idx)
            results.extend(rows)
            save_results(results, args.output_csv)
            if progress:
                last = rows[-1]
                outer.set_postfix(
                    {
                        "problem": config["problem"],
                        "update": config["beta_update"],
                        "avg": last.get("avg_reward", np.nan),
                        "status": last.get("status"),
                    }
                )
    else:
        payloads = [
            (config, args.seed, args.num_trajectories, args.max_steps, str(DEVICE), torch_threads)
            for config in configs
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_config = {
                executor.submit(run_config_worker, payload): payload[0]
                for payload in payloads
            }
            outer = tqdm(
                as_completed(future_to_config),
                total=len(future_to_config),
                desc=desc,
                unit="run",
                disable=not progress,
            )
            for run_idx, future in enumerate(outer, start=1):
                config = future_to_config[future]
                try:
                    rows = future.result()
                except Exception as exc:
                    rows = failed_worker_rows(config, args.seed, exc)
                for row in rows:
                    row["run_idx"] = int(run_idx)
                results.extend(rows)
                save_results(results, args.output_csv)
                if progress:
                    last = rows[-1]
                    outer.set_postfix(
                        {
                            "problem": config["problem"],
                            "update": config["beta_update"],
                            "avg": last.get("avg_reward", np.nan),
                            "status": last.get("status"),
                        }
                    )

    save_results(results, args.output_csv)
    df = ordered_results_frame(results)
    ok_count = int((df["status"] == "ok").sum()) if not df.empty else 0
    failed_count = int((df["status"] == "failed").sum()) if not df.empty else 0

    print("\nLearning-curve reruns complete.")
    print(f"Rows saved: {len(df)}")
    print(f"Successful checkpoint rows: {ok_count}")
    print(f"Failed rows: {failed_count}")
    print(f"Output CSV: {args.output_csv}")

    if args.plot:
        plot_path = plot_results(args.output_csv, args.plot_path)
        print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
