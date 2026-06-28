"""
Refined continuous-observation FOGAS NN grid search for MountainCar.

This script reuses the original NN grid-search implementation and narrows the
candidate grid around the best previous NN regions.  Results are ranked by the
stochastic solver policy mean steps, with greedy performance as a tie-breaker.
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

import grid_search_continuous_mountaincar_fogas_nn as base


OUTPUT_CSV = base.RESULTS_DIR / "continuous_fogas_nn_refined_grid_search.csv"
BEST_CSV = base.RESULTS_DIR / "continuous_fogas_nn_refined_grid_search_best.csv"
CHECKPOINT_CSV = base.RESULTS_DIR / "continuous_fogas_nn_refined_eval_checkpoints.csv"

ALPHA_GRID = [1e-4, 2e-4, 3e-4]
ETA_GRID = [3e-7, 1e-6]
RHO_GRID = [0.005, 0.02, 0.05, 0.1, 0.5]
THETA_LR_GRID = [3e-4, 1e-3]
THETA_LAMBDA_GRID = [1e-9, 1e-8, 3e-8, 1e-7, 1e-6]

EVAL_TRAJECTORIES = 10
EVAL_MAX_STEPS = 200


def finite_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def blank_metrics():
    return {
        "greedy_mean_steps": np.nan,
        "greedy_std_steps": np.nan,
        "greedy_success_rate": np.nan,
        "greedy_avg_env_return": np.nan,
        "greedy_std_env_return": np.nan,
        "solver_mean_steps": np.nan,
        "solver_std_steps": np.nan,
        "solver_success_rate": np.nan,
        "solver_avg_env_return": np.nan,
        "solver_std_env_return": np.nan,
        "final_total_loss": np.nan,
        "final_policy_objective": np.nan,
        "final_beta_objective": np.nan,
        "final_q_objective": np.nan,
        "final_theta_norm": np.nan,
        "final_policy_grad_norm": np.nan,
        "final_beta_grad_norm": np.nan,
        "final_theta_grad_norm": np.nan,
    }


def evaluate_policy_rollouts(
    solver,
    num_trajectories,
    max_steps,
    seed,
    deterministic,
):
    env = gym.make(
        base.ENV_ID,
        max_episode_steps=max_steps,
        goal_velocity=base.GOAL_VELOCITY,
    )
    steps_list = []
    returns = []
    successes = 0
    num_trajectories = int(num_trajectories)

    try:
        for idx in range(num_trajectories):
            obs, _ = env.reset(seed=int(seed) + idx)
            done = False
            steps = 0
            total_return = 0.0

            while not done and steps < int(max_steps):
                action = solver.sample_action(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_return += float(reward)
                steps += 1
                done = bool(terminated) or bool(truncated)

            if steps < int(max_steps):
                successes += 1
            steps_list.append(steps)
            returns.append(total_return)
    finally:
        env.close()

    ddof = 1 if num_trajectories > 1 else 0
    return {
        "mean_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list, ddof=ddof)),
        "success_rate": float(successes / max(1, num_trajectories)),
        "avg_env_return": float(np.mean(returns)),
        "std_env_return": float(np.std(returns, ddof=ddof)),
    }


def prefixed_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def evaluate_solver_policy(
    solver,
    num_trajectories,
    max_steps,
    seed,
):
    greedy_metrics = evaluate_policy_rollouts(
        solver=solver,
        num_trajectories=num_trajectories,
        max_steps=max_steps,
        seed=seed,
        deterministic=True,
    )
    solver_metrics = evaluate_policy_rollouts(
        solver=solver,
        num_trajectories=num_trajectories,
        max_steps=max_steps,
        seed=seed,
        deterministic=False,
    )
    return {
        **prefixed_metrics("greedy", greedy_metrics),
        **prefixed_metrics("solver", solver_metrics),
    }


def ordered_results_frame(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df

    for column in (
        "solver_mean_steps",
        "greedy_mean_steps",
        "solver_success_rate",
        "greedy_success_rate",
        "elapsed_seconds",
    ):
        if column not in df.columns:
            df[column] = np.nan

    return df.sort_values(
        by=[
            "solver_mean_steps",
            "greedy_mean_steps",
            "solver_success_rate",
            "greedy_success_rate",
            "elapsed_seconds",
        ],
        ascending=[True, True, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def save_results(results, output_csv, best_csv):
    base.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = ordered_results_frame(results)
    df.to_csv(output_csv, index=False)

    successful = df[df["status"] == "ok"] if not df.empty and "status" in df.columns else df
    if not successful.empty:
        successful.head(1).to_csv(best_csv, index=False)


def checkpoint_row(base, iteration, eval_metrics, diagnostics):
    row = dict(base)
    row["checkpoint_iter"] = int(iteration)
    row["greedy_mean_steps"] = eval_metrics["greedy_mean_steps"]
    row["greedy_std_steps"] = eval_metrics["greedy_std_steps"]
    row["greedy_success_rate"] = eval_metrics["greedy_success_rate"]
    row["greedy_avg_env_return"] = eval_metrics["greedy_avg_env_return"]
    row["greedy_std_env_return"] = eval_metrics["greedy_std_env_return"]
    row["solver_mean_steps"] = eval_metrics["solver_mean_steps"]
    row["solver_std_steps"] = eval_metrics["solver_std_steps"]
    row["solver_success_rate"] = eval_metrics["solver_success_rate"]
    row["solver_avg_env_return"] = eval_metrics["solver_avg_env_return"]
    row["solver_std_env_return"] = eval_metrics["solver_std_env_return"]
    row.update(
        {
            "checkpoint_total_loss": finite_float(diagnostics.get("total_loss")),
            "checkpoint_policy_objective": finite_float(diagnostics.get("policy_objective")),
            "checkpoint_beta_objective": finite_float(diagnostics.get("beta_objective")),
            "checkpoint_q_objective": finite_float(diagnostics.get("q_objective")),
            "checkpoint_theta_norm": finite_float(diagnostics.get("theta_norm")),
            "checkpoint_policy_grad_norm": finite_float(diagnostics.get("policy_grad_norm")),
            "checkpoint_beta_grad_norm": finite_float(diagnostics.get("beta_grad_norm")),
            "checkpoint_theta_grad_norm": finite_float(diagnostics.get("theta_grad_norm")),
        }
    )
    return row


def parse_args():
    parser = base.argparse.ArgumentParser(
        description="Run refined ContinuousFinalParametrizedSolver NN grid search on MountainCar."
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", type=str, default=str(base.DEFAULT_DEVICE))
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated devices, e.g. cuda:0,cuda:1, or 'auto'.",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--time-budget-hours", type=float, default=0.0)
    parser.add_argument("--dataset-path", type=Path, default=base.DATASET_PATH)
    parser.add_argument("--eval-trajectories", type=int, default=EVAL_TRAJECTORIES)
    parser.add_argument("--eval-max-steps", type=int, default=EVAL_MAX_STEPS)
    return parser.parse_args()


def apply_refined_configuration():
    base.OUTPUT_CSV = OUTPUT_CSV
    base.BEST_CSV = BEST_CSV
    base.CHECKPOINT_CSV = CHECKPOINT_CSV

    base.ALPHA_GRID = ALPHA_GRID
    base.ETA_GRID = ETA_GRID
    base.RHO_GRID = RHO_GRID
    base.THETA_LR_GRID = THETA_LR_GRID
    base.THETA_LAMBDA_GRID = THETA_LAMBDA_GRID

    base.EVAL_TRAJECTORIES = EVAL_TRAJECTORIES
    base.EVAL_MAX_STEPS = EVAL_MAX_STEPS

    base.blank_metrics = blank_metrics
    base.evaluate_policy_rollouts = evaluate_policy_rollouts
    base.evaluate_solver_policy = evaluate_solver_policy
    base.ordered_results_frame = ordered_results_frame
    base.save_results = save_results
    base.checkpoint_row = checkpoint_row
    base.parse_args = parse_args


def main():
    apply_refined_configuration()
    if base.total_grid_size() != 300:
        raise RuntimeError(f"Expected exactly 300 candidates, got {base.total_grid_size()}")
    base.run_grid_search()


if __name__ == "__main__":
    main()
