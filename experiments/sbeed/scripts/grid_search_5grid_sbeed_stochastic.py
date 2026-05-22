"""Grid search for tabular MultiLinearSBEED on the stochastic 5x5 grid.

This script evaluates the multi-step linear SBEED implementation after the
building-version stages. The environment is the 5x5 grid with walls, a goal,
a pit, and stochastic action execution. Results are written as CSV rows so
partially completed searches can be resumed or inspected later.

Only experiment orchestration lives here: environment definition, config
sampling, training calls, evaluation, and CSV serialization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from rl_methods.sbeed.building_versions import MultiLinearSBEED  # noqa: E402
from rl_methods.sbeed.features import (  # noqa: E402
    TabularStateActionFeatures,
    TabularStateFeatures,
)
from rl_methods.sbeed.sbeed_spec import DiscreteMDPSpec  # noqa: E402


# ============================================================
# DEFAULT SEARCH SETTINGS
# ============================================================

# Output and run-count defaults. Command-line arguments can override these when
# launching a shorter local run or a longer cluster run.
DEFAULT_SEARCH_DIR = REPO_ROOT / "data/results/sbeed"
RESULTS_CSV_NAME = "sbeed_tabular_stochastic_grid_search.csv"

N_RUNS = 46

EPISODES = 500
COLLECT_PER_EPISODE = 20
UPDATES_PER_EPISODE = 10
INITIAL_COLLECT_STEPS = 1024

MAX_BUFFER_SIZE = 12000

EVAL_EVERY_EPISODES = 100
N_EVAL_EPISODES_DURING = 80
N_EVAL_EPISODES_FINAL = 300
MAX_STEPS_PER_EVAL_EPISODE = 100

EARLY_STOP_AFTER_EPISODES = 250
EARLY_STOP_MARGIN = 0.20

BASE_SEED = 42
CONFIG_SEED = 123


# ============================================================
# STOCHASTIC 5x5 GRID PROBLEM
# ============================================================

# Grid convention: states are row-major indices, actions are
# 0=up, 1=down, 2=left, 3=right. Goal and pit terminate episodes.
states = torch.arange(25, dtype=torch.long)
actions = torch.arange(4, dtype=torch.long)

N = len(states)
A = len(actions)
gamma = 0.9

grid_size = 5

x_0 = 0

goal_grid = 24
pit_grid = 18

wall_states = {6, 7, 12}
terminal_states = {goal_grid, pit_grid}


def state_to_pos(s: int) -> Tuple[int, int]:
    return divmod(int(s), grid_size)


def pos_to_state(row: int, col: int) -> int:
    return int(row) * grid_size + int(col)


def move_deterministic(s: int, a: int) -> int:
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
        raise ValueError("action must be in {0,1,2,3}")

    if not (0 <= new_row < grid_size and 0 <= new_col < grid_size):
        return s

    sp = pos_to_state(new_row, new_col)
    if sp in wall_states:
        return s

    return sp


def transition_probs(s: int, a: int) -> List[Tuple[int, float]]:
    """
    Stochastic transition:
        80% intended action
        20% random action uniformly over all actions.
    """
    s = int(s)
    a = int(a)

    probs_by_state: Dict[int, float] = {}
    for candidate_a in range(A):
        prob = 0.8 + 0.2 / A if candidate_a == a else 0.2 / A
        sp = move_deterministic(s, candidate_a)
        probs_by_state[sp] = probs_by_state.get(sp, 0.0) + prob

    return list(probs_by_state.items())


def next_state(s: int, a: int) -> int:
    probs = transition_probs(s, a)
    next_states = [sp for sp, _ in probs]
    probabilities = torch.tensor([p for _, p in probs], dtype=torch.float64)
    idx = torch.multinomial(probabilities, num_samples=1).item()
    return int(next_states[idx])


def reward_fn(s: int, a: int, sp: int) -> float:
    sp = int(sp)
    if sp == goal_grid:
        return 1.0
    if sp == pit_grid:
        return -1.0
    return -0.1


value_features = TabularStateFeatures(n_states=N)
rho_features = TabularStateActionFeatures(n_states=N, n_actions=A)

mdp_spec = DiscreteMDPSpec(
    n_states=N,
    n_actions=A,
    gamma=gamma,
    value_features=value_features,
    rho_features=rho_features,
    x0=x_0,
)


# ============================================================
# POLICY / EVALUATION HELPERS
# ============================================================

@torch.no_grad()
def greedy_action_from_policy(pi: torch.Tensor, s: int) -> int:
    return int(torch.argmax(pi[int(s)]).item())


@torch.no_grad()
def sample_policy_action(pi: torch.Tensor, s: int, rng: np.random.Generator) -> int:
    probs = pi[int(s)].detach().cpu().numpy()
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def evaluate_policy(
    solver: Any,
    transition_fn: Any,
    reward_fn: Any,
    *,
    start_state: int,
    terminal_states: set[int],
    reset_state_fn: Any = None,
    n_eval_episodes: int = 200,
    max_steps_per_episode: int = 100,
    gamma: Optional[float] = None,
    stochastic_policy: bool = False,
    seed: int = 0,
    goal_state: Optional[int] = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    gamma = solver.gamma if gamma is None else gamma
    pi = solver.get_policy_matrix()

    returns = []
    lengths = []
    dones = []
    successes = []

    torch_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        for _ in range(n_eval_episodes):
            s = int(reset_state_fn()) if reset_state_fn is not None else int(start_state)

            g = 0.0
            discount = 1.0
            done = False
            success = False

            for t in range(max_steps_per_episode):
                if s in terminal_states:
                    done = True
                    success = goal_state is not None and s == goal_state
                    break

                if stochastic_policy:
                    a = sample_policy_action(pi, s, rng)
                else:
                    a = greedy_action_from_policy(pi, s)

                sp = int(transition_fn(s, a))
                r = float(reward_fn(s, a, sp))

                g += discount * r
                discount *= gamma
                s = sp

                if s in terminal_states:
                    done = True
                    success = goal_state is not None and s == goal_state
                    break

            returns.append(g)
            lengths.append(t + 1)
            dones.append(float(done))
            successes.append(float(success))
    finally:
        torch.random.set_rng_state(torch_state)

    returns_arr = np.asarray(returns, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.float64)
    dones_arr = np.asarray(dones, dtype=np.float64)
    successes_arr = np.asarray(successes, dtype=np.float64)

    out = {
        "eval_return_mean": float(returns_arr.mean()),
        "eval_return_std": float(returns_arr.std(ddof=1)) if len(returns_arr) > 1 else 0.0,
        "eval_return_median": float(np.median(returns_arr)),
        "eval_return_min": float(returns_arr.min()),
        "eval_return_max": float(returns_arr.max()),
        "eval_length_mean": float(lengths_arr.mean()),
        "eval_done_rate": float(dones_arr.mean()),
    }

    if goal_state is not None:
        out["eval_success_rate"] = float(successes_arr.mean())

    return out


@torch.no_grad()
def extract_policy_summary(solver: Any) -> Dict[str, Any]:
    pi = solver.get_policy_matrix().detach().cpu()

    best_actions = torch.argmax(pi, dim=1).numpy().astype(int).tolist()
    best_probs = torch.max(pi, dim=1).values.numpy().astype(float).tolist()
    entropy = -(pi.clamp_min(1e-12) * pi.clamp_min(1e-12).log()).sum(dim=1)

    return {
        "pi": pi,
        "best_actions": best_actions,
        "best_probs": best_probs,
        "mean_best_prob": float(np.mean(best_probs)),
        "min_best_prob": float(np.min(best_probs)),
        "mean_entropy": float(entropy.mean().item()),
    }


def print_policy(pi: torch.Tensor, *, decimals: int = 3) -> None:
    pi = pi.detach().cpu()
    n_states, n_actions = pi.shape

    print("\n========== SBEED POLICY ==========\n")
    for s in range(n_states):
        parts = [f"pi({a}|{s})={float(pi[s, a]):.{decimals}f}" for a in range(n_actions)]
        best_a = int(torch.argmax(pi[s]).item())
        print(f"State {s}: " + "  ".join(parts) + f"  --> best action: {best_a}")
    print("\n==================================\n")


def tail_values(history: List[Dict[str, Any]], key: str, tail: int = 100) -> List[float]:
    vals = []
    for h in history[-tail:]:
        if key in h:
            v = h[key]
            if isinstance(v, (float, int)) and math.isfinite(v):
                vals.append(float(v))
    return vals


def tail_mean(history: List[Dict[str, Any]], key: str, tail: int = 100) -> float:
    vals = tail_values(history, key, tail)
    return float(np.mean(vals)) if vals else float("nan")


def tail_std(history: List[Dict[str, Any]], key: str, tail: int = 100) -> float:
    vals = tail_values(history, key, tail)
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


# ============================================================
# CONFIG GENERATION
# ============================================================

def make_budgeted_sbeed_configs(*, n_runs: int = 40, seed: int = 123) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    configs = []

    def sample_one(k: int) -> Dict[str, Any]:
        if k == 1:
            return {
                "rollout_length": 1,
                "lambda_entropy": rng.choice([0.01, 0.02, 0.05, 0.08]),
                "eta": rng.choice([0.0, 0.05, 0.1, 0.2]),
                "lr_value": rng.choice([3e-3, 1e-2, 3e-2]),
                "lr_rho": rng.choice([3e-3, 1e-2, 3e-2]),
                "lr_policy": rng.choice([3e-3, 1e-2, 3e-2]),
                "fisher_damping": rng.choice([1e-4, 1e-3, 1e-2]),
                "batch_size": rng.choice([128, 256, 512]),
                "epsilon": rng.choice([0.15, 0.25, 0.30, 0.40]),
            }

        if k == 2:
            return {
                "rollout_length": 2,
                "lambda_entropy": rng.choice([0.02, 0.05, 0.08, 0.10]),
                "eta": rng.choice([0.0, 0.05, 0.1, 0.2]),
                "lr_value": rng.choice([3e-3, 1e-2, 3e-2]),
                "lr_rho": rng.choice([3e-3, 1e-2]),
                "lr_policy": rng.choice([1e-3, 3e-3, 1e-2]),
                "fisher_damping": rng.choice([1e-3, 1e-2]),
                "batch_size": rng.choice([256, 512, 1024]),
                "epsilon": rng.choice([0.20, 0.30, 0.40]),
            }

        if k == 3:
            return {
                "rollout_length": 3,
                "lambda_entropy": rng.choice([0.02, 0.05, 0.08]),
                "eta": rng.choice([0.05, 0.1, 0.2]),
                "lr_value": rng.choice([3e-3, 1e-2]),
                "lr_rho": rng.choice([3e-3, 1e-2]),
                "lr_policy": rng.choice([1e-3, 3e-3, 1e-2]),
                "fisher_damping": rng.choice([1e-3, 1e-2]),
                "batch_size": rng.choice([512, 1024, None]),
                "epsilon": rng.choice([0.20, 0.30, 0.40]),
            }

        raise ValueError(f"Unsupported k={k}")

    handpicked = [
        {
            "rollout_length": 1,
            "lambda_entropy": 0.05,
            "eta": 0.1,
            "lr_value": 1e-2,
            "lr_rho": 1e-2,
            "lr_policy": 1e-2,
            "fisher_damping": 1e-3,
            "batch_size": 256,
            "epsilon": 0.30,
        },
        {
            "rollout_length": 2,
            "lambda_entropy": 0.05,
            "eta": 0.1,
            "lr_value": 1e-2,
            "lr_rho": 1e-2,
            "lr_policy": 3e-3,
            "fisher_damping": 1e-3,
            "batch_size": 512,
            "epsilon": 0.30,
        },
        {
            "rollout_length": 2,
            "lambda_entropy": 0.02,
            "eta": 0.1,
            "lr_value": 1e-2,
            "lr_rho": 3e-3,
            "lr_policy": 3e-3,
            "fisher_damping": 1e-2,
            "batch_size": 512,
            "epsilon": 0.30,
        },
        {
            "rollout_length": 3,
            "lambda_entropy": 0.05,
            "eta": 0.1,
            "lr_value": 1e-2,
            "lr_rho": 1e-2,
            "lr_policy": 3e-3,
            "fisher_damping": 1e-3,
            "batch_size": 512,
            "epsilon": 0.30,
        },
        {
            "rollout_length": 3,
            "lambda_entropy": 0.05,
            "eta": 0.05,
            "lr_value": 1e-2,
            "lr_rho": 3e-3,
            "lr_policy": 3e-3,
            "fisher_damping": 1e-2,
            "batch_size": 1024,
            "epsilon": 0.30,
        },
        {
            "rollout_length": 3,
            "lambda_entropy": 0.02,
            "eta": 0.1,
            "lr_value": 3e-3,
            "lr_rho": 3e-3,
            "lr_policy": 1e-3,
            "fisher_damping": 1e-2,
            "batch_size": 1024,
            "epsilon": 0.40,
        },
    ]

    configs.extend(handpicked)

    target_k_counts = {1: 12, 2: 16, 3: 12}
    current_k_counts = {1: 0, 2: 0, 3: 0}
    for cfg in configs:
        current_k_counts[cfg["rollout_length"]] += 1

    for k, target in target_k_counts.items():
        while current_k_counts[k] < target:
            cfg = sample_one(k)
            if cfg["rollout_length"] >= 2 and cfg["lr_policy"] == 3e-2:
                continue
            if cfg["rollout_length"] == 3 and cfg["batch_size"] in [128, 256]:
                continue
            if cfg["rollout_length"] == 3 and cfg["fisher_damping"] == 1e-4:
                continue
            if cfg["eta"] >= 0.2 and cfg["lr_rho"] == 3e-2:
                continue

            configs.append(cfg)
            current_k_counts[k] += 1

    longer_budget_configs = [
        {
            "rollout_length": 2,
            "lambda_entropy": 0.05,
            "eta": 0.2,
            "lr_value": 3e-3,
            "lr_rho": 3e-3,
            "lr_policy": 3e-3,
            "fisher_damping": 1e-3,
            "batch_size": 1024,
            "epsilon": 0.20,
        },
        {
            "rollout_length": 2,
            "lambda_entropy": 0.05,
            "eta": 0.1,
            "lr_value": 1e-2,
            "lr_rho": 1e-2,
            "lr_policy": 3e-3,
            "fisher_damping": 1e-3,
            "batch_size": 512,
            "epsilon": 0.30,
        },
        {
            "rollout_length": 1,
            "lambda_entropy": 0.01,
            "eta": 0.1,
            "lr_value": 3e-3,
            "lr_rho": 1e-2,
            "lr_policy": 1e-2,
            "fisher_damping": 1e-4,
            "batch_size": 128,
            "epsilon": 0.40,
        },
        {
            "rollout_length": 3,
            "lambda_entropy": 0.05,
            "eta": 0.05,
            "lr_value": 1e-2,
            "lr_rho": 3e-3,
            "lr_policy": 3e-3,
            "fisher_damping": 1e-2,
            "batch_size": 1024,
            "epsilon": 0.30,
        },
        {
            "rollout_length": 2,
            "lambda_entropy": 0.02,
            "eta": 0.2,
            "lr_value": 3e-3,
            "lr_rho": 1e-2,
            "lr_policy": 1e-3,
            "fisher_damping": 1e-2,
            "batch_size": 1024,
            "epsilon": 0.30,
        },
        {
            "rollout_length": 3,
            "lambda_entropy": 0.02,
            "eta": 0.1,
            "lr_value": 3e-3,
            "lr_rho": 3e-3,
            "lr_policy": 1e-3,
            "fisher_damping": 1e-2,
            "batch_size": 1024,
            "epsilon": 0.40,
        },
    ]
    for cfg in longer_budget_configs:
        cfg.update(
            episodes=800,
            collect_per_episode=25,
            updates_per_episode=15,
            initial_collect_steps=2000,
        )
    configs.extend(longer_budget_configs)

    rng.shuffle(configs)
    return configs[:n_runs]


# ============================================================
# SCORING AND SAVING
# ============================================================

CSV_FIELDS = [
    "ok",
    "run_id",
    "seed",
    "seconds",
    "episodes_completed",
    "stopped_early",
    "stop_reason",
    "episodes",
    "collect_per_episode",
    "updates_per_episode",
    "initial_collect_steps",
    "max_buffer_size",
    "rollout_length",
    "lambda_entropy",
    "eta",
    "lr_value",
    "lr_rho",
    "lr_policy",
    "fisher_damping",
    "batch_size",
    "epsilon",
    "eval_return_mean",
    "eval_return_std",
    "eval_return_median",
    "eval_return_min",
    "eval_return_max",
    "eval_length_mean",
    "eval_done_rate",
    "eval_success_rate",
    "score",
    "mean_best_prob",
    "min_best_prob",
    "mean_entropy",
    "best_actions",
    "best_probs",
    "last_objective",
    "last_primal_mse",
    "last_dual_mse",
    "last_theta_grad_norm",
    "last_beta_grad_norm",
    "last_policy_grad_norm",
    "tail_objective_mean",
    "tail_objective_std",
    "tail_primal_mse_mean",
    "tail_primal_mse_std",
    "tail_dual_mse_mean",
    "tail_dual_mse_std",
    "tail_policy_grad_mean",
    "tail_policy_grad_std",
    "tail_theta_grad_mean",
    "tail_theta_grad_std",
    "eval_history",
    "error",
]


def score_result(r: Dict[str, Any]) -> float:
    if not r.get("ok", False):
        return -float("inf")

    def finite_or_zero(value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    return (
        float(r["eval_return_mean"])
        - 0.25 * finite_or_zero(r.get("eval_return_std", 0.0))
        - 0.01 * finite_or_zero(r.get("tail_policy_grad_std", 0.0))
        - 0.01 * finite_or_zero(r.get("tail_theta_grad_std", 0.0))
    )


def result_for_csv(r: Dict[str, Any]) -> Dict[str, Any]:
    skip = {"solver", "pi"}
    out = {}

    for k, v in r.items():
        if k in skip or isinstance(v, torch.Tensor):
            continue
        if isinstance(v, (list, tuple, dict)):
            out[k] = json.dumps(v)
        else:
            out[k] = v

    out["score"] = score_result(r)
    return out


def append_result_csv(path: Path, r: Dict[str, Any]) -> None:
    row = result_for_csv(r)
    file_exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_best_artifacts(best: Dict[str, Any], search_dir: Path) -> None:
    return


# ============================================================
# ONE TRAINING RUN, CHUNKED
# ============================================================

def train_one_config_chunked(
    cfg: Dict[str, Any],
    *,
    run_id: int,
    seed: int,
    current_global_best_score: float,
    device: torch.device,
    episodes: int,
    collect_per_episode: int,
    updates_per_episode: int,
    initial_collect_steps: int,
    max_buffer_size: int,
    eval_every_episodes: int,
    n_eval_episodes_during: int,
    n_eval_episodes_final: int,
    max_steps_per_eval_episode: int,
    early_stop_after_episodes: int,
    early_stop_margin: Optional[float],
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    start_time = time.time()
    run_episodes = int(cfg.get("episodes", episodes))
    run_collect_per_episode = int(cfg.get("collect_per_episode", collect_per_episode))
    run_updates_per_episode = int(cfg.get("updates_per_episode", updates_per_episode))
    run_initial_collect_steps = int(cfg.get("initial_collect_steps", initial_collect_steps))
    run_max_buffer_size = int(cfg.get("max_buffer_size", max_buffer_size))

    solver = MultiLinearSBEED(
        spec=mdp_spec,
        lambda_entropy=cfg["lambda_entropy"],
        eta=cfg["eta"],
        ridge=1e-6,
        lr_value=cfg["lr_value"],
        lr_policy=cfg["lr_policy"],
        lr_rho=cfg["lr_rho"],
        tau=100000.0,
        buffer_mode="fifo",
        max_buffer_size=run_max_buffer_size,
        batch_size=cfg["batch_size"],
        rollout_length=cfg["rollout_length"],
        value_optimizer="adam",
        rho_optimizer="adam",
        policy_optimizer="npg_cg",
        fisher_damping=cfg["fisher_damping"],
        cg_iters=10,
        cg_tol=1e-12,
        seed=seed,
        device=device,
    )

    eval_history = []
    stopped_early = False
    stop_reason = ""
    episode = 0

    try:
        if run_initial_collect_steps > 0:
            solver.collect_steps(
                transition_fn=next_state,
                n_steps=run_initial_collect_steps,
                start_state=x_0,
                reward_fn=reward_fn,
                behavior="uniform",
                epsilon=0.0,
                terminal_states=terminal_states,
            )

        for episode in range(1, run_episodes + 1):
            solver.collect_steps(
                transition_fn=next_state,
                n_steps=run_collect_per_episode,
                start_state=x_0,
                reward_fn=reward_fn,
                behavior="policy",
                epsilon=cfg["epsilon"],
                terminal_states=terminal_states,
            )

            for _ in range(run_updates_per_episode):
                stats = solver.step()
                solver.loss_history.append(stats)

            if episode % eval_every_episodes == 0 or episode == run_episodes:
                eval_stats = evaluate_policy(
                    solver,
                    transition_fn=next_state,
                    reward_fn=reward_fn,
                    start_state=x_0,
                    terminal_states=terminal_states,
                    n_eval_episodes=n_eval_episodes_during,
                    max_steps_per_episode=max_steps_per_eval_episode,
                    stochastic_policy=False,
                    seed=seed + 10_000 + episode,
                    goal_state=goal_grid,
                )

                eval_stats["episode"] = episode
                eval_history.append(eval_stats)

                temp_result = {
                    "ok": True,
                    **eval_stats,
                    "tail_policy_grad_std": tail_std(solver.loss_history, "policy_grad_norm"),
                    "tail_theta_grad_std": tail_std(solver.loss_history, "theta_grad_norm"),
                }
                temp_score = score_result(temp_result)

                print(
                    f"    eval ep={episode:04d} | "
                    f"return={eval_stats['eval_return_mean']:.5f} "
                    f"std={eval_stats['eval_return_std']:.5f} "
                    f"success={eval_stats.get('eval_success_rate', float('nan')):.3f} "
                    f"score={temp_score:.5f}",
                    flush=True,
                )

                if (
                    early_stop_margin is not None
                    and episode >= early_stop_after_episodes
                    and math.isfinite(current_global_best_score)
                    and temp_score < current_global_best_score - early_stop_margin
                ):
                    stopped_early = True
                    stop_reason = (
                        f"score {temp_score:.5f} below global best "
                        f"{current_global_best_score:.5f} by margin {early_stop_margin}"
                    )
                    print(f"    early stop: {stop_reason}", flush=True)
                    break

        final_eval_stats = evaluate_policy(
            solver,
            transition_fn=next_state,
            reward_fn=reward_fn,
            start_state=x_0,
            terminal_states=terminal_states,
            n_eval_episodes=n_eval_episodes_final,
            max_steps_per_episode=max_steps_per_eval_episode,
            stochastic_policy=False,
            seed=seed + 999_999,
            goal_state=goal_grid,
        )

        policy_summary = extract_policy_summary(solver)
        last_loss = solver.loss_history[-1] if len(solver.loss_history) > 0 else {}

        result = {
            "ok": True,
            "run_id": run_id,
            "seed": seed,
            "seconds": time.time() - start_time,
            "episodes_completed": episode,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            **cfg,
            "episodes": run_episodes,
            "collect_per_episode": run_collect_per_episode,
            "updates_per_episode": run_updates_per_episode,
            "initial_collect_steps": run_initial_collect_steps,
            "max_buffer_size": run_max_buffer_size,
            **final_eval_stats,
            "score": None,
            "mean_best_prob": policy_summary["mean_best_prob"],
            "min_best_prob": policy_summary["min_best_prob"],
            "mean_entropy": policy_summary["mean_entropy"],
            "best_actions": policy_summary["best_actions"],
            "best_probs": policy_summary["best_probs"],
            "pi": policy_summary["pi"],
            "last_objective": float(last_loss.get("objective", float("nan"))),
            "last_primal_mse": float(last_loss.get("primal_mse", float("nan"))),
            "last_dual_mse": float(last_loss.get("dual_mse", float("nan"))),
            "last_theta_grad_norm": float(last_loss.get("theta_grad_norm", float("nan"))),
            "last_beta_grad_norm": float(last_loss.get("beta_grad_norm", float("nan"))),
            "last_policy_grad_norm": float(last_loss.get("policy_grad_norm", float("nan"))),
            "tail_objective_mean": tail_mean(solver.loss_history, "objective"),
            "tail_objective_std": tail_std(solver.loss_history, "objective"),
            "tail_primal_mse_mean": tail_mean(solver.loss_history, "primal_mse"),
            "tail_primal_mse_std": tail_std(solver.loss_history, "primal_mse"),
            "tail_dual_mse_mean": tail_mean(solver.loss_history, "dual_mse"),
            "tail_dual_mse_std": tail_std(solver.loss_history, "dual_mse"),
            "tail_policy_grad_mean": tail_mean(solver.loss_history, "policy_grad_norm"),
            "tail_policy_grad_std": tail_std(solver.loss_history, "policy_grad_norm"),
            "tail_theta_grad_mean": tail_mean(solver.loss_history, "theta_grad_norm"),
            "tail_theta_grad_std": tail_std(solver.loss_history, "theta_grad_norm"),
            "eval_history": eval_history,
            "solver": solver,
        }

        result["score"] = score_result(result)

    except Exception as e:
        result = {
            "ok": False,
            "run_id": run_id,
            "seed": seed,
            "seconds": time.time() - start_time,
            "episodes_completed": episode,
            "stopped_early": True,
            "stop_reason": "exception",
            **cfg,
            "episodes": run_episodes,
            "collect_per_episode": run_collect_per_episode,
            "updates_per_episode": run_updates_per_episode,
            "initial_collect_steps": run_initial_collect_steps,
            "max_buffer_size": run_max_buffer_size,
            "error": repr(e),
            "eval_return_mean": -float("inf"),
            "eval_return_std": float("inf"),
            "score": -float("inf"),
            "solver": None,
            "pi": None,
        }

    return result


# ============================================================
# MAIN GRID SEARCH
# ============================================================

def run_wrapper_stochastic(
    i: int,
    cfg: Dict[str, Any],
    base_seed: int,
    completed_seeds: set[int],
    device: torch.device,
    episodes: int,
    collect_per_episode: int,
    updates_per_episode: int,
    initial_collect_steps: int,
    max_buffer_size: int,
    eval_every_episodes: int,
    n_eval_episodes_during: int,
    n_eval_episodes_final: int,
    max_steps_per_eval_episode: int,
    early_stop_after_episodes: int,
    early_stop_margin: Optional[float],
):
    seed = base_seed + i
    if seed in completed_seeds:
        return None

    return train_one_config_chunked(
        cfg,
        run_id=i,
        seed=seed,
        current_global_best_score=-float("inf"),
        device=device,
        episodes=episodes,
        collect_per_episode=collect_per_episode,
        updates_per_episode=updates_per_episode,
        initial_collect_steps=initial_collect_steps,
        max_buffer_size=max_buffer_size,
        eval_every_episodes=eval_every_episodes,
        n_eval_episodes_during=n_eval_episodes_during,
        n_eval_episodes_final=n_eval_episodes_final,
        max_steps_per_eval_episode=max_steps_per_eval_episode,
        early_stop_after_episodes=early_stop_after_episodes,
        early_stop_margin=early_stop_margin,
    )


def run_budgeted_sbeed_grid_search(
    *,
    device: torch.device,
    n_runs: int,
    base_seed: int,
    config_seed: int,
    search_dir: Path,
    episodes: int,
    collect_per_episode: int,
    updates_per_episode: int,
    initial_collect_steps: int,
    max_buffer_size: int,
    eval_every_episodes: int,
    n_eval_episodes_during: int,
    n_eval_episodes_final: int,
    max_steps_per_eval_episode: int,
    early_stop_after_episodes: int,
    early_stop_margin: Optional[float],
    workers: int = 1,
    resume: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    search_dir.mkdir(parents=True, exist_ok=True)
    csv_path = search_dir / RESULTS_CSV_NAME

    configs = make_budgeted_sbeed_configs(n_runs=n_runs, seed=config_seed)

    # Resume logic
    completed_seeds = set()
    if resume and csv_path.exists():
        try:
            with csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("ok") == "True":
                        completed_seeds.add(int(row["seed"]))
        except Exception as e:
            print(f"Warning: Could not read {csv_path} for resume: {e}")

    results = []
    best = None
    best_score = -float("inf")

    print("\n========== SBEED BUDGETED GRID SEARCH ==========")
    print(f"Problem: stochastic 5x5 grid, start={x_0}, goal={goal_grid}, pit={pit_grid}")
    print(f"Runs: {len(configs)}")
    print(f"Workers: {workers}")
    print(f"Episodes per full run: {episodes}")
    print(f"Device: {device}")
    print(f"Results dir: {search_dir}")
    if resume:
        print(f"Resume enabled. Found {len(completed_seeds)} completed runs.")
    print("================================================\n", flush=True)

    global_start = time.time()

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(
                    run_wrapper_stochastic,
                    i,
                    cfg,
                    base_seed,
                    completed_seeds,
                    device,
                    episodes,
                    collect_per_episode,
                    updates_per_episode,
                    initial_collect_steps,
                    max_buffer_size,
                    eval_every_episodes,
                    n_eval_episodes_during,
                    n_eval_episodes_final,
                    max_steps_per_eval_episode,
                    early_stop_after_episodes,
                    early_stop_margin,
                ): i
                for i, cfg in enumerate(configs)
            }
            
            for future in as_completed(future_to_idx):
                result = future.result()
                if result is None:
                    continue
                
                results.append(result)
                append_result_csv(csv_path, result)
                
                if result["ok"] and result["score"] > best_score:
                    best = result
                    best_score = result["score"]
                    save_best_artifacts(best, search_dir)
                    print(f"\n  >>> NEW BEST (Run {result['run_id']}) | score={best_score:.5f} <<<", flush=True)

                elapsed_hours = (time.time() - global_start) / 3600.0
                avg_minutes = (time.time() - global_start) / 60.0 / len(results)
                remaining = len(configs) - len(completed_seeds) - len(results)
                eta_hours = (remaining * avg_minutes / 60.0) / workers if remaining > 0 else 0
                
                print(
                    f"Progress: {len(results)+len(completed_seeds)}/{len(configs)} | "
                    f"Elapsed: {elapsed_hours:.2f}h | "
                    f"ETA: {eta_hours:.2f}h | "
                    f"Last return: {result['eval_return_mean']:.3f}",
                    flush=True,
                )
    else:
        for i, cfg in enumerate(configs):
            result = run_wrapper_stochastic(
                i,
                cfg,
                base_seed,
                completed_seeds,
                device,
                episodes,
                collect_per_episode,
                updates_per_episode,
                initial_collect_steps,
                max_buffer_size,
                eval_every_episodes,
                n_eval_episodes_during,
                n_eval_episodes_final,
                max_steps_per_eval_episode,
                early_stop_after_episodes,
                early_stop_margin,
            )
            if result is None:
                print(f"Skipping seed {base_seed + i}")
                continue

            results.append(result)
            append_result_csv(csv_path, result)

            if result["ok"] and result["score"] > best_score:
                best = result
                best_score = result["score"]
                save_best_artifacts(best, search_dir)
                print(f"\n  >>> NEW BEST (Run {result['run_id']}) | score={best_score:.5f} <<<", flush=True)

            elapsed_hours = (time.time() - global_start) / 3600.0
            avg_minutes = (time.time() - global_start) / 60.0 / len(results)
            remaining = len(configs) - len(completed_seeds) - len(results)
            eta_hours = remaining * avg_minutes / 60.0
            print(
                f"Progress: {len(results)+len(completed_seeds)}/{len(configs)} | "
                f"Elapsed: {elapsed_hours:.2f}h | "
                f"ETA: {eta_hours:.2f}h | "
                f"Last return: {result['eval_return_mean']:.3f}",
                flush=True,
            )

    print("\n========== SEARCH DONE ==========")
    print(f"Total time: {(time.time() - global_start) / 3600.0:.2f} hours")
    print(f"CSV saved to: {csv_path}", flush=True)

    return results, best


# ============================================================
# REPORTING
# ============================================================

def summarize_top_results(results: List[Dict[str, Any]], top_k: int = 10) -> None:
    valid = [r for r in results if r.get("ok", False)]
    valid = sorted(valid, key=score_result, reverse=True)

    print(f"\n========== TOP {min(top_k, len(valid))} CONFIGS ==========\n")

    for rank, r in enumerate(valid[:top_k], start=1):
        print(
            f"#{rank:02d} | "
            f"run={r['run_id']:03d} | "
            f"score={r['score']:.5f} | "
            f"return={r['eval_return_mean']:.5f} +/- {r['eval_return_std']:.5f} | "
            f"success={r.get('eval_success_rate', float('nan')):.3f} | "
            f"k={r['rollout_length']} | "
            f"bs={r['batch_size']} | "
            f"lambda={r['lambda_entropy']} | "
            f"eta={r['eta']} | "
            f"lrV={r['lr_value']} | "
            f"lrRho={r['lr_rho']} | "
            f"lrPi={r['lr_policy']} | "
            f"damp={r['fisher_damping']} | "
            f"eps={r['epsilon']} | "
            f"obj={r['last_objective']:.6f} | "
            f"primal={r['last_primal_mse']:.6f} | "
            f"dual={r['last_dual_mse']:.6f} | "
            f"stopped={r['stopped_early']}"
        )


def print_best_result(best: Optional[Dict[str, Any]]) -> None:
    if best is None:
        print("\nNo successful run found.")
        return

    print("\n========== BEST RESULT ==========\n")

    keys = [
        "run_id",
        "seed",
        "score",
        "eval_return_mean",
        "eval_return_std",
        "eval_return_median",
        "eval_done_rate",
        "eval_success_rate",
        "eval_length_mean",
        "episodes_completed",
        "stopped_early",
        "stop_reason",
        "lambda_entropy",
        "eta",
        "rollout_length",
        "batch_size",
        "lr_value",
        "lr_rho",
        "lr_policy",
        "fisher_damping",
        "epsilon",
        "last_objective",
        "last_primal_mse",
        "last_dual_mse",
        "last_theta_grad_norm",
        "last_beta_grad_norm",
        "last_policy_grad_norm",
        "tail_policy_grad_mean",
        "tail_policy_grad_std",
        "tail_theta_grad_mean",
        "tail_theta_grad_std",
        "mean_best_prob",
        "min_best_prob",
        "mean_entropy",
        "seconds",
    ]

    for k in keys:
        if k in best:
            print(f"{k}: {best[k]}")

    print("\nBest actions:")
    print(best["best_actions"])

    print("\nBest action probabilities:")
    print([round(x, 4) for x in best["best_probs"]])

    print_policy(best["pi"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a budgeted MultiLinearSBEED grid search on the stochastic 5x5 gridworld."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--collect-per-episode", type=int, default=COLLECT_PER_EPISODE)
    parser.add_argument("--updates-per-episode", type=int, default=UPDATES_PER_EPISODE)
    parser.add_argument("--initial-collect-steps", type=int, default=INITIAL_COLLECT_STEPS)
    parser.add_argument("--max-buffer-size", type=int, default=MAX_BUFFER_SIZE)
    parser.add_argument("--eval-every-episodes", type=int, default=EVAL_EVERY_EPISODES)
    parser.add_argument("--n-eval-episodes-during", type=int, default=N_EVAL_EPISODES_DURING)
    parser.add_argument("--n-eval-episodes-final", type=int, default=N_EVAL_EPISODES_FINAL)
    parser.add_argument("--max-steps-per-eval-episode", type=int, default=MAX_STEPS_PER_EVAL_EPISODE)
    parser.add_argument("--early-stop-after-episodes", type=int, default=EARLY_STOP_AFTER_EPISODES)
    parser.add_argument("--early-stop-margin", type=float, default=EARLY_STOP_MARGIN)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--config-seed", type=int, default=CONFIG_SEED)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--torch-threads", type=int, default=1, help="Threads per worker.")
    parser.add_argument("--resume", action="store_true", help="Skip already completed seeds.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous result artifacts in output-dir before starting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if args.overwrite and output_dir.exists():
        for name in [RESULTS_CSV_NAME]:
            path = output_dir / name
            if path.exists():
                path.unlink()

    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    early_stop_margin = None if args.disable_early_stop else args.early_stop_margin

    results, best = run_budgeted_sbeed_grid_search(
        device=device,
        n_runs=args.n_runs,
        base_seed=args.base_seed,
        config_seed=args.config_seed,
        search_dir=output_dir,
        episodes=args.episodes,
        collect_per_episode=args.collect_per_episode,
        updates_per_episode=args.updates_per_episode,
        initial_collect_steps=args.initial_collect_steps,
        max_buffer_size=args.max_buffer_size,
        eval_every_episodes=args.eval_every_episodes,
        n_eval_episodes_during=args.n_eval_episodes_during,
        n_eval_episodes_final=args.n_eval_episodes_final,
        max_steps_per_eval_episode=args.max_steps_per_eval_episode,
        early_stop_after_episodes=args.early_stop_after_episodes,
        early_stop_margin=early_stop_margin,
        workers=args.workers,
        resume=args.resume,
    )

    summarize_top_results(results, top_k=10)
    print_best_result(best)


if __name__ == "__main__":
    main()
