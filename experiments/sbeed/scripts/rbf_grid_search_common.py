"""Shared utilities for fixed RBF SBEED grid searches.

The deterministic and stochastic RBF scripts import this module to avoid
duplicating the 5x5 gridworld, RBF feature construction, evaluation metrics,
CSV schema, and multiprocessing loop.

The helpers intentionally keep the experiment logic explicit: each top-level
script provides the list of configurations to try, while this file handles the
common mechanics of running, scoring, resuming, and printing results.
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from rl_methods.sbeed.building_versions import MultiLinearSBEED  # noqa: E402
from rl_methods.sbeed.features import (  # noqa: E402
    RBFStateActionFeatures,
    RBFStateFeatures,
)
from rl_methods.sbeed.sbeed_spec import DiscreteMDPSpec  # noqa: E402


STATES = torch.arange(25, dtype=torch.long)
ACTIONS = torch.arange(4, dtype=torch.long)
N = len(STATES)
A = len(ACTIONS)
GAMMA = 0.9

GRID_SIZE = 5
X0 = 0
GOAL_GRID = 24
PIT_GRID = 18
WALL_STATES = {6, 7, 12}
TERMINAL_STATES = {GOAL_GRID, PIT_GRID}

# CSV columns are kept stable so result files from different RBF searches can
# be ranked and compared with the same analysis code.
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
    "tau",
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


def state_to_pos(s: int) -> Tuple[int, int]:
    return divmod(int(s), GRID_SIZE)


def pos_to_state(row: int, col: int) -> int:
    return int(row) * GRID_SIZE + int(col)


def move_deterministic(s: int, a: int) -> int:
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
        raise ValueError("action must be in {0,1,2,3}")

    if not (0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE):
        return s

    sp = pos_to_state(new_row, new_col)
    if sp in WALL_STATES:
        return s
    return sp


def deterministic_transition_probs(s: int, a: int) -> List[Tuple[int, float]]:
    return [(move_deterministic(s, a), 1.0)]


def stochastic_transition_probs(s: int, a: int) -> List[Tuple[int, float]]:
    s = int(s)
    a = int(a)
    probs_by_state: Dict[int, float] = {}

    for candidate_a in range(A):
        prob = 0.8 + 0.2 / A if candidate_a == a else 0.2 / A
        sp = move_deterministic(s, candidate_a)
        probs_by_state[sp] = probs_by_state.get(sp, 0.0) + prob

    return list(probs_by_state.items())


def next_state_deterministic(s: int, a: int) -> int:
    probs = deterministic_transition_probs(s, a)
    return int(probs[0][0])


def next_state_stochastic(s: int, a: int) -> int:
    probs = stochastic_transition_probs(s, a)
    next_states = [sp for sp, _ in probs]
    probabilities = torch.tensor([p for _, p in probs], dtype=torch.float64)
    idx = torch.multinomial(probabilities, num_samples=1).item()
    return int(next_states[idx])


def reward_fn(s: int, a: int, sp: int) -> float:
    sp = int(sp)
    if sp == GOAL_GRID:
        return 1.0
    if sp == PIT_GRID:
        return -1.0
    return -0.1


def build_rbf_mdp_spec(
    *,
    value_bandwidth_scale: float = 0.25,
    policy_bandwidth_scale: float = 0.55,
) -> DiscreteMDPSpec:
    center_lin = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    centers = torch.tensor([[r, c] for r in center_lin for c in center_lin], dtype=torch.float64)
    state_coords = torch.tensor(
        [[r / 4.0, c / 4.0] for r in range(5) for c in range(5)],
        dtype=torch.float64,
    )

    value_features = RBFStateFeatures(
        n_states=N,
        centers=centers,
        state_coords=state_coords,
        bandwidth="nearest",
        bandwidth_scale=value_bandwidth_scale,
        include_bias=True,
    )
    rho_features = RBFStateActionFeatures(
        state_features=value_features,
        n_actions=A,
    )
    policy_features = RBFStateFeatures(
        n_states=N,
        centers=centers,
        state_coords=state_coords,
        bandwidth="nearest",
        bandwidth_scale=policy_bandwidth_scale,
        include_bias=True,
    )

    return DiscreteMDPSpec(
        n_states=N,
        n_actions=A,
        gamma=GAMMA,
        value_features=value_features,
        rho_features=rho_features,
        policy_features=policy_features,
        x0=X0,
    )


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
    transition_fn: Callable[[int, int], int],
    *,
    n_eval_episodes: int,
    max_steps_per_episode: int,
    stochastic_policy: bool = False,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    pi = solver.get_policy_matrix()

    returns = []
    lengths = []
    dones = []
    successes = []

    torch_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        for _ in range(n_eval_episodes):
            s = X0
            g = 0.0
            discount = 1.0
            done = False
            success = False

            for t in range(max_steps_per_episode):
                if s in TERMINAL_STATES:
                    done = True
                    success = s == GOAL_GRID
                    break

                a = (
                    sample_policy_action(pi, s, rng)
                    if stochastic_policy
                    else greedy_action_from_policy(pi, s)
                )
                sp = int(transition_fn(s, a))
                r = float(reward_fn(s, a, sp))

                g += discount * r
                discount *= solver.gamma
                s = sp

                if s in TERMINAL_STATES:
                    done = True
                    success = s == GOAL_GRID
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

    return {
        "eval_return_mean": float(returns_arr.mean()),
        "eval_return_std": float(returns_arr.std(ddof=1)) if len(returns_arr) > 1 else 0.0,
        "eval_return_median": float(np.median(returns_arr)),
        "eval_return_min": float(returns_arr.min()),
        "eval_return_max": float(returns_arr.max()),
        "eval_length_mean": float(lengths_arr.mean()),
        "eval_done_rate": float(dones_arr.mean()),
        "eval_success_rate": float(successes_arr.mean()),
    }


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
        v = h.get(key)
        if isinstance(v, (float, int)) and math.isfinite(v):
            vals.append(float(v))
    return vals


def tail_mean(history: List[Dict[str, Any]], key: str, tail: int = 100) -> float:
    vals = tail_values(history, key, tail)
    return float(np.mean(vals)) if vals else float("nan")


def tail_std(history: List[Dict[str, Any]], key: str, tail: int = 100) -> float:
    vals = tail_values(history, key, tail)
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def finite_or_zero(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def score_result(r: Dict[str, Any]) -> float:
    if not r.get("ok", False):
        return -float("inf")
    return (
        float(r["eval_return_mean"])
        - 0.25 * finite_or_zero(r.get("eval_return_std", 0.0))
        - 0.01 * finite_or_zero(r.get("tail_policy_grad_std", 0.0))
        - 0.01 * finite_or_zero(r.get("tail_theta_grad_std", 0.0))
    )


def result_for_csv(r: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in r.items():
        if k in {"solver", "pi"} or isinstance(v, torch.Tensor):
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


def save_best_artifacts(best: Dict[str, Any], output_dir: Path) -> None:
    return


def train_one_config(
    cfg: Dict[str, Any],
    *,
    run_id: int,
    seed: int,
    mdp_spec: DiscreteMDPSpec,
    transition_fn: Callable[[int, int], int],
    device: torch.device,
    training_kwargs: Dict[str, Any],
    eval_every_episodes: int,
    n_eval_episodes_during: int,
    n_eval_episodes_final: int,
    max_steps_per_eval_episode: int,
    current_global_best_score: float,
    early_stop_after_episodes: int,
    early_stop_margin: Optional[float],
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    episode = 0
    run_training_kwargs = {
        **training_kwargs,
        **{k: cfg[k] for k in training_kwargs.keys() if k in cfg},
    }

    solver = MultiLinearSBEED(
        spec=mdp_spec,
        lambda_entropy=cfg["lambda_entropy"],
        eta=cfg["eta"],
        ridge=1e-6,
        lr_value=cfg["lr_value"],
        lr_rho=cfg["lr_rho"],
        lr_policy=cfg["lr_policy"],
        tau=run_training_kwargs["tau"],
        buffer_mode="fifo",
        max_buffer_size=run_training_kwargs["max_buffer_size"],
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

    try:
        initial_collect_steps = int(run_training_kwargs["initial_collect_steps"])
        if initial_collect_steps > 0:
            # RBF runs collect a uniform warm-up buffer before switching to the
            # current policy, reducing early instability from sparse coverage.
            solver.collect_steps(
                transition_fn=transition_fn,
                n_steps=initial_collect_steps,
                start_state=X0,
                reward_fn=reward_fn,
                behavior="uniform",
                epsilon=0.0,
                terminal_states=TERMINAL_STATES,
            )

        episodes = int(run_training_kwargs["episodes"])
        for episode in range(1, episodes + 1):
            # Each loop alternates fresh on-policy collection with several
            # SBEED updates on the FIFO replay buffer.
            solver.collect_steps(
                transition_fn=transition_fn,
                n_steps=run_training_kwargs["collect_per_episode"],
                start_state=X0,
                reward_fn=reward_fn,
                behavior="policy",
                epsilon=cfg["epsilon"],
                terminal_states=TERMINAL_STATES,
            )

            for _ in range(run_training_kwargs["updates_per_episode"]):
                stats = solver.step()
                solver.loss_history.append(stats)

            if episode % eval_every_episodes == 0 or episode == episodes:
                # Periodic evaluation uses a separate rollout pass so training
                # diagnostics and control performance remain distinct.
                eval_stats = evaluate_policy(
                    solver,
                    transition_fn,
                    n_eval_episodes=n_eval_episodes_during,
                    max_steps_per_episode=max_steps_per_eval_episode,
                    seed=seed + 10_000 + episode,
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
                    f"success={eval_stats['eval_success_rate']:.3f} "
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
            transition_fn,
            n_eval_episodes=n_eval_episodes_final,
            max_steps_per_episode=max_steps_per_eval_episode,
            seed=seed + 999_999,
        )
        policy_summary = extract_policy_summary(solver)
        last_loss = solver.loss_history[-1] if solver.loss_history else {}

        result = {
            "ok": True,
            "run_id": run_id,
            "seed": seed,
            "seconds": time.time() - start_time,
            "episodes_completed": episode,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            **cfg,
            **run_training_kwargs,
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
        return result

    except Exception as exc:
        return {
            "ok": False,
            "run_id": run_id,
            "seed": seed,
            "seconds": time.time() - start_time,
            "episodes_completed": episode,
            "stopped_early": True,
            "stop_reason": "exception",
            **cfg,
            **run_training_kwargs,
            "error": repr(exc),
            "eval_return_mean": -float("inf"),
            "eval_return_std": float("inf"),
            "score": -float("inf"),
            "solver": None,
            "pi": None,
        }


def run_wrapper_rbf(
    i: int,
    cfg: Dict[str, Any],
    base_seed: int,
    completed_seeds: set[list],
    mdp_spec: Any,
    transition_fn: Any,
    device: torch.device,
    training_kwargs: Dict[str, Any],
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

    return train_one_config(
        cfg,
        run_id=i,
        seed=seed,
        mdp_spec=mdp_spec,
        transition_fn=transition_fn,
        device=device,
        training_kwargs=training_kwargs,
        eval_every_episodes=eval_every_episodes,
        n_eval_episodes_during=n_eval_episodes_during,
        n_eval_episodes_final=n_eval_episodes_final,
        max_steps_per_eval_episode=max_steps_per_eval_episode,
        current_global_best_score=-float("inf"),
        early_stop_after_episodes=early_stop_after_episodes,
        early_stop_margin=early_stop_margin,
    )


def run_fixed_rbf_grid_search(
    *,
    name: str,
    configs: List[Dict[str, Any]],
    stochastic: bool,
    training_kwargs: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    base_seed: int,
    n_runs: Optional[int],
    eval_every_episodes: int,
    n_eval_episodes_during: int,
    n_eval_episodes_final: int,
    max_steps_per_eval_episode: int,
    early_stop_after_episodes: int,
    early_stop_margin: Optional[float],
    workers: int = 1,
    resume: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_names = {
        "deterministic_rbf": "sbeed_rbf_deterministic_grid_search.csv",
        "stochastic_rbf": "sbeed_rbf_stochastic_grid_search.csv",
    }
    csv_path = output_dir / csv_names.get(name, "results.csv")
    mdp_spec = build_rbf_mdp_spec()
    transition_fn = next_state_stochastic if stochastic else next_state_deterministic
    selected_configs = configs if n_runs is None else configs[:n_runs]

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
    global_start = time.time()

    print("\n========== SBEED RBF FIXED GRID SEARCH ==========")
    print(f"Name: {name}")
    print(f"Problem: {'stochastic' if stochastic else 'deterministic'} 5x5 RBF grid")
    print(f"Runs: {len(selected_configs)}")
    print(f"Workers: {workers}")
    print(f"Episodes per full run: {training_kwargs['episodes']}")
    print(f"Device: {device}")
    print(f"Results dir: {output_dir}")
    if resume:
        print(f"Resume enabled. Found {len(completed_seeds)} completed runs.")
    print("=================================================\n", flush=True)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(
                    run_wrapper_rbf,
                    i,
                    cfg,
                    base_seed,
                    completed_seeds,
                    mdp_spec,
                    transition_fn,
                    device,
                    training_kwargs,
                    eval_every_episodes,
                    n_eval_episodes_during,
                    n_eval_episodes_final,
                    max_steps_per_eval_episode,
                    early_stop_after_episodes,
                    early_stop_margin,
                ): i
                for i, cfg in enumerate(selected_configs)
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
                    save_best_artifacts(best, output_dir)
                    print(f"\n  >>> NEW BEST (Run {result['run_id']}) | score={best_score:.5f} <<<", flush=True)

                elapsed_hours = (time.time() - global_start) / 3600.0
                avg_minutes = (time.time() - global_start) / 60.0 / len(results)
                remaining = len(selected_configs) - len(completed_seeds) - len(results)
                eta_hours = (remaining * avg_minutes / 60.0) / workers if remaining > 0 else 0
                print(
                    f"Progress: {len(results)+len(completed_seeds)}/{len(selected_configs)} | "
                    f"Elapsed: {elapsed_hours:.2f}h | "
                    f"ETA: {eta_hours:.2f}h | "
                    f"Last return: {result['eval_return_mean']:.3f}",
                    flush=True,
                )
    else:
        for i, cfg in enumerate(selected_configs):
            result = run_wrapper_rbf(
                i,
                cfg,
                base_seed,
                completed_seeds,
                mdp_spec,
                transition_fn,
                device,
                training_kwargs,
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
                save_best_artifacts(best, output_dir)
                print(f"\n  >>> NEW BEST (Run {result['run_id']}) | score={best_score:.5f} <<<", flush=True)

            elapsed_hours = (time.time() - global_start) / 3600.0
            avg_minutes = (time.time() - global_start) / 60.0 / len(results)
            remaining = len(selected_configs) - len(completed_seeds) - len(results)
            eta_hours = remaining * avg_minutes / 60.0
            print(
                f"Progress: {len(results)+len(completed_seeds)}/{len(selected_configs)} | "
                f"Elapsed: {elapsed_hours:.2f}h | "
                f"ETA: {eta_hours:.2f}h | "
                f"Last return: {result['eval_return_mean']:.3f}",
                flush=True,
            )

    print("\n========== SEARCH DONE ==========")
    print(f"Total time: {(time.time() - global_start) / 3600.0:.2f} hours")
    print(f"CSV saved to: {csv_path}", flush=True)
    return results, best


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
            f"success={r['eval_success_rate']:.3f} | "
            f"k={r['rollout_length']} | "
            f"bs={r['batch_size']} | "
            f"lambda={r['lambda_entropy']} | "
            f"eta={r['eta']} | "
            f"lrV={r['lr_value']} | "
            f"lrRho={r['lr_rho']} | "
            f"lrPi={r['lr_policy']} | "
            f"damp={r['fisher_damping']} | "
            f"eps={r['epsilon']} | "
            f"stopped={r['stopped_early']}"
        )


def print_best_result(best: Optional[Dict[str, Any]]) -> None:
    if best is None:
        print("\nNo successful run found.")
        return

    print("\n========== BEST RESULT ==========\n")
    for key in [
        "run_id",
        "seed",
        "score",
        "eval_return_mean",
        "eval_return_std",
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
        "mean_best_prob",
        "min_best_prob",
        "mean_entropy",
        "seconds",
    ]:
        if key in best:
            print(f"{key}: {best[key]}")

    print("\nBest actions:")
    print(best["best_actions"])
    print("\nBest action probabilities:")
    print([round(x, 4) for x in best["best_probs"]])
    print_policy(best["pi"])


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: Path,
    default_training_kwargs: Dict[str, Any],
) -> None:
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=default_training_kwargs["episodes"])
    parser.add_argument(
        "--collect-per-episode",
        type=int,
        default=default_training_kwargs["collect_per_episode"],
    )
    parser.add_argument(
        "--updates-per-episode",
        type=int,
        default=default_training_kwargs["updates_per_episode"],
    )
    parser.add_argument(
        "--initial-collect-steps",
        type=int,
        default=default_training_kwargs["initial_collect_steps"],
    )
    parser.add_argument("--max-buffer-size", type=int, default=default_training_kwargs["max_buffer_size"])
    parser.add_argument("--tau", type=float, default=default_training_kwargs["tau"])
    parser.add_argument("--eval-every-episodes", type=int, default=100)
    parser.add_argument("--n-eval-episodes-during", type=int, default=80)
    parser.add_argument("--n-eval-episodes-final", type=int, default=300)
    parser.add_argument("--max-steps-per-eval-episode", type=int, default=100)
    parser.add_argument("--early-stop-after-episodes", type=int, default=250)
    parser.add_argument("--early-stop-margin", type=float, default=0.20)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--base-seed", type=int, default=42)
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


def training_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "episodes": args.episodes,
        "collect_per_episode": args.collect_per_episode,
        "updates_per_episode": args.updates_per_episode,
        "initial_collect_steps": args.initial_collect_steps,
        "max_buffer_size": args.max_buffer_size,
        "tau": args.tau,
    }


def clear_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for name in [
        "results.csv",
        "sbeed_rbf_deterministic_grid_search.csv",
        "sbeed_rbf_stochastic_grid_search.csv",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()
