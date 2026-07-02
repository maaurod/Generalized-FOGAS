"""
Continuous-observation FOGAS NN grid search for MountainCar.

This script uses the generic ContinuousFinalParametrizedSolver on the
continuous MountainCar dataset and evaluates candidates by rollout mean steps.
Lower greedy_mean_steps is better, with solver_mean_steps used as the tie-break.
"""

from __future__ import annotations

import argparse
import itertools
import math
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
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
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "generalization" / "mountain_car"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from rl_methods.fogas_generalization import (  # noqa: E402
    ContinuousDiscretePolicyParam,
    ContinuousFinalParametrizedSolver,
    ContinuousNeuralQParam,
    ContinuousNeuralUParam,
    ContinuousStateActionMLPModule,
    ContinuousStateMLPPolicyModule,
)


SEED = 44
NN_SEED = 44
EVAL_SEED = 42
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

ENV_ID = "MountainCar-v0"
GAMMA = 0.9
GOAL_VELOCITY = 0.0
X0_OBS = np.array([-0.5, 0.0], dtype=np.float64)
OBS_DIM = 2
N_ACTIONS = 3

DATASET_PATH = RESULTS_DIR / "mountaincar_data_obs_columns.csv"
OUTPUT_CSV = RESULTS_DIR / "continuous_fogas_nn_grid_search.csv"
BEST_CSV = RESULTS_DIR / "continuous_fogas_nn_grid_search_best.csv"
CHECKPOINT_CSV = RESULTS_DIR / "continuous_fogas_nn_eval_checkpoints.csv"

T_FIXED = 20_000
THETA_INNER_STEPS = 10
HIDDEN_SIZES = (16, 16)
BETA_REG = None
POLICY_GRADIENT = "exact"
POLICY_OPTIMIZER = "adam"

ALPHA_GRID = [3e-5, 1e-4, 3e-4]
ETA_GRID = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6]
RHO_GRID = [0.005, 0.02, 0.1, 0.5]
THETA_LR_GRID = [1e-4, 3e-4, 1e-3]
THETA_LAMBDA_GRID = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

EVAL_TRAJECTORIES = 50
EVAL_MAX_STEPS = 200
EVAL_CHECKPOINTS = [1000, 2500, 5000, 10000, 15000, 20000]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ContinuousFinalParametrizedSolver NN grid search on MountainCar."
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", type=str, default=str(DEFAULT_DEVICE))
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated devices, e.g. cuda:0,cuda:1, or 'auto'.",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--time-budget-hours", type=float, default=0.0)
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--eval-trajectories", type=int, default=EVAL_TRAJECTORIES)
    parser.add_argument("--eval-max-steps", type=int, default=EVAL_MAX_STEPS)
    return parser.parse_args()


def configure_torch_threads(torch_threads):
    torch_threads = max(1, int(torch_threads))
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(torch_threads)
    except RuntimeError:
        pass


def resolve_devices(args):
    if args.devices is None:
        devices = [str(args.device)]
    elif str(args.devices).strip().lower() == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            devices = [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
        else:
            devices = [str(args.device)]
    else:
        devices = [device.strip() for device in str(args.devices).split(",") if device.strip()]

    if not devices:
        raise ValueError("No devices were provided. Use --device or --devices.")

    if args.workers is None:
        workers = len(devices) if args.devices is not None else 1
    else:
        workers = max(1, int(args.workers))

    return devices, workers


def configure_device(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    return device


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


def prepare_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    header = pd.read_csv(dataset_path, nrows=0).columns
    required_generic = {"obs_0", "obs_1", "next_obs_0", "next_obs_1", "action", "reward"}
    if required_generic.issubset(set(header)):
        return dataset_path

    raise ValueError(
        "MountainCar dataset must contain obs_0, obs_1, next_obs_0, next_obs_1, "
        f"action, and reward columns. Found columns: {list(header)}"
    )


def dataset_row_count(dataset_path):
    with Path(dataset_path).open() as handle:
        return max(0, sum(1 for _ in handle) - 1)


def hidden_sizes_label(hidden_sizes):
    return "x".join(str(int(size)) for size in hidden_sizes)


def candidate_key(row):
    return (
        float(row["alpha"]),
        float(row["eta"]),
        float(row["rho"]),
        float(row["theta_lr"]),
        float(row["theta_lambda"]),
    )


def all_candidates():
    return [
        tuple(candidate)
        for candidate in itertools.product(
            ALPHA_GRID,
            ETA_GRID,
            RHO_GRID,
            THETA_LR_GRID,
            THETA_LAMBDA_GRID,
        )
    ]


def total_grid_size():
    return len(all_candidates())


def load_existing_results(resume, output_csv):
    if not resume or not output_csv.exists():
        return [], set()
    df = pd.read_csv(output_csv)
    if "greedy_mean_steps" not in df.columns:
        return [], set()
    df = df[df["greedy_mean_steps"].notna()].copy()
    rows = df.to_dict("records")
    completed = {candidate_key(row) for row in rows}
    return rows, completed


def ordered_results_frame(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df
    for column in ("greedy_mean_steps", "solver_mean_steps", "elapsed_seconds"):
        if column not in df.columns:
            df[column] = np.nan
    return df.sort_values(
        by=["greedy_mean_steps", "solver_mean_steps", "elapsed_seconds"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def save_results(results, output_csv, best_csv):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = ordered_results_frame(results)
    df.to_csv(output_csv, index=False)
    successful = df[df["status"] == "ok"] if not df.empty else df
    if not successful.empty:
        successful.head(1).to_csv(best_csv, index=False)


def save_checkpoints(checkpoint_rows, checkpoint_csv):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(checkpoint_rows)
    df.to_csv(checkpoint_csv, index=False)


def finite_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def blank_metrics():
    return {
        "greedy_mean_steps": np.nan,
        "greedy_success_rate": np.nan,
        "greedy_avg_env_return": np.nan,
        "solver_mean_steps": np.nan,
        "solver_success_rate": np.nan,
        "solver_avg_env_return": np.nan,
        "final_total_loss": np.nan,
        "final_policy_objective": np.nan,
        "final_beta_objective": np.nan,
        "final_q_objective": np.nan,
        "final_theta_norm": np.nan,
        "final_policy_grad_norm": np.nan,
        "final_beta_grad_norm": np.nan,
        "final_theta_grad_norm": np.nan,
    }


def base_row(params, device, status="ok", error="", batch_size=np.nan):
    alpha, eta, rho, theta_lr, theta_lambda = params
    row = {
        "alpha": float(alpha),
        "eta": float(eta),
        "rho": float(rho),
        "T": int(T_FIXED),
        "theta_lr": float(theta_lr),
        "theta_inner_steps": int(THETA_INNER_STEPS),
        "theta_lambda": float(theta_lambda),
        "batch_size": "" if batch_size is None else batch_size,
        "hidden_sizes": hidden_sizes_label(HIDDEN_SIZES),
        "beta_reg": BETA_REG,
        "theta_mode": "reg_fixed",
        "theta_optimizer": "adam",
        "theta_start_mode": "warm",
        "beta_update": "fogas_diag",
        "policy_optimizer": POLICY_OPTIMIZER,
        "policy_gradient": POLICY_GRADIENT,
        "nn_seed": int(NN_SEED),
        "seed": int(SEED),
        "device": str(device),
        "status": status,
        "error": error,
        "elapsed_seconds": np.nan,
    }
    row.update(blank_metrics())
    return row


def make_solver(
    dataset_path,
    device,
    theta_lr,
    theta_lambda,
    batch_size,
    u_jacobian_batch_size,
    value_batch_size,
):
    set_seed(NN_SEED)
    u_param = ContinuousNeuralUParam(
        ContinuousStateActionMLPModule(
            obs_dim=OBS_DIM,
            action_dim=1,
            hidden_sizes=HIDDEN_SIZES,
            dtype=torch.float64,
        )
    )
    q_param = ContinuousNeuralQParam(
        ContinuousStateActionMLPModule(
            obs_dim=OBS_DIM,
            action_dim=1,
            hidden_sizes=HIDDEN_SIZES,
            dtype=torch.float64,
        )
    )
    policy_param = ContinuousDiscretePolicyParam(
        ContinuousStateMLPPolicyModule(
            obs_dim=OBS_DIM,
            n_actions=N_ACTIONS,
            hidden_sizes=HIDDEN_SIZES,
            dtype=torch.float64,
        )
    )

    return ContinuousFinalParametrizedSolver(
        obs_dim=OBS_DIM,
        action_type="discrete",
        n_actions=N_ACTIONS,
        gamma=GAMMA,
        x0_obs=X0_OBS,
        csv_path=str(dataset_path),
        u_param=u_param,
        q_param=q_param,
        policy_param=policy_param,
        seed=NN_SEED,
        device=device,
        theta_mode="reg_fixed",
        theta_lambda=theta_lambda,
        theta_optimizer="adam",
        theta_inner_steps=THETA_INNER_STEPS,
        theta_lr=theta_lr,
        theta_start_mode="warm",
        beta_update="fogas_diag",
        beta_reg=BETA_REG,
        batch_size=batch_size,
        u_jacobian_batch_size=u_jacobian_batch_size,
        value_batch_size=value_batch_size,
        dataset_verbose=False,
    )


def evaluate_policy_rollouts(
    solver,
    num_trajectories,
    max_steps,
    seed,
    deterministic,
):
    env = gym.make(ENV_ID, max_episode_steps=max_steps, goal_velocity=GOAL_VELOCITY)
    steps_list = []
    returns = []
    successes = 0

    try:
        for idx in range(int(num_trajectories)):
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

    return {
        "mean_steps": float(np.mean(steps_list)),
        "success_rate": float(successes / max(1, int(num_trajectories))),
        "avg_env_return": float(np.mean(returns)),
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


def diagnostics_metrics(diagnostics):
    if not diagnostics:
        return {}
    final = diagnostics[-1]
    return {
        "final_total_loss": finite_float(final.get("total_loss")),
        "final_policy_objective": finite_float(final.get("policy_objective")),
        "final_beta_objective": finite_float(final.get("beta_objective")),
        "final_q_objective": finite_float(final.get("q_objective")),
        "final_theta_norm": finite_float(final.get("theta_norm")),
        "final_policy_grad_norm": finite_float(final.get("policy_grad_norm")),
        "final_beta_grad_norm": finite_float(final.get("beta_grad_norm")),
        "final_theta_grad_norm": finite_float(final.get("theta_grad_norm")),
    }


def checkpoint_row(base, iteration, eval_metrics, diagnostics):
    row = dict(base)
    row["checkpoint_iter"] = int(iteration)
    row["greedy_mean_steps"] = eval_metrics["greedy_mean_steps"]
    row["greedy_success_rate"] = eval_metrics["greedy_success_rate"]
    row["greedy_avg_env_return"] = eval_metrics["greedy_avg_env_return"]
    row["solver_mean_steps"] = eval_metrics["solver_mean_steps"]
    row["solver_success_rate"] = eval_metrics["solver_success_rate"]
    row["solver_avg_env_return"] = eval_metrics["solver_avg_env_return"]
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


def run_candidate(
    params,
    dataset_path,
    device,
    eval_trajectories,
    eval_max_steps,
    batch_size=None,
):
    alpha, eta, rho, theta_lr, theta_lambda = params
    start = time.perf_counter()
    row = base_row(params, device, batch_size=batch_size)
    checkpoint_rows = []

    try:
        solver = make_solver(
            dataset_path=dataset_path,
            device=device,
            theta_lr=theta_lr,
            theta_lambda=theta_lambda,
            batch_size=batch_size,
            u_jacobian_batch_size=None,
            value_batch_size=None,
        )
        checkpoint_set = set(EVAL_CHECKPOINTS)

        def on_checkpoint(current_solver, iteration, diagnostics):
            if int(iteration) not in checkpoint_set:
                return
            eval_metrics = evaluate_solver_policy(
                solver=current_solver,
                num_trajectories=eval_trajectories,
                max_steps=eval_max_steps,
                seed=EVAL_SEED + int(iteration),
            )
            checkpoint_rows.append(
                checkpoint_row(
                    base=row,
                    iteration=iteration,
                    eval_metrics=eval_metrics,
                    diagnostics=diagnostics,
                )
            )

        solver.run(
            alpha=alpha,
            eta=eta,
            rho=rho,
            T=T_FIXED,
            theta_lr=theta_lr,
            theta_inner_steps=THETA_INNER_STEPS,
            theta_lambda=theta_lambda,
            policy_optimizer=POLICY_OPTIMIZER,
            policy_gradient=POLICY_GRADIENT,
            tqdm_print=False,
            verbose=False,
            checkpoint_callback=on_checkpoint,
        )

        final_eval = evaluate_solver_policy(
            solver=solver,
            num_trajectories=eval_trajectories,
            max_steps=eval_max_steps,
            seed=EVAL_SEED,
        )
        row.update(final_eval)
        row.update(diagnostics_metrics(solver.get_diagnostics() or []))
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        row["elapsed_seconds"] = float(time.perf_counter() - start)
        for cp_row in checkpoint_rows:
            cp_row["elapsed_seconds"] = row["elapsed_seconds"]

    return row, checkpoint_rows


def run_candidate_worker(
    params,
    dataset_path,
    device_name,
    torch_threads,
    eval_trajectories,
    eval_max_steps,
    batch_size=None,
):
    start = time.perf_counter()
    device = torch.device(device_name)
    try:
        configure_torch_threads(torch_threads)
        device = configure_device(device_name)
        set_seed(SEED)
        return run_candidate(
            params=params,
            dataset_path=Path(dataset_path),
            device=device,
            eval_trajectories=eval_trajectories,
            eval_max_steps=eval_max_steps,
            batch_size=batch_size,
        )
    except Exception as exc:
        row = base_row(params, device, status="failed", error=repr(exc), batch_size=batch_size)
        row["elapsed_seconds"] = float(time.perf_counter() - start)
        return row, []


def run_grid_search():
    args = parse_args()
    configure_torch_threads(args.torch_threads)
    set_seed(SEED)
    devices, workers = resolve_devices(args)
    primary_device = torch.device(devices[0])

    dataset_path = prepare_dataset(args.dataset_path)
    print(f"Using devices: {', '.join(devices)}")
    print(f"Dataset: {dataset_path}")
    print(f"Results: {OUTPUT_CSV}")
    print(f"Checkpoints: {CHECKPOINT_CSV}")
    print(f"Workers: {workers}")
    print(f"Torch threads: {max(1, int(args.torch_threads))}")
    print(f"Fixed T: {T_FIXED}")
    print(f"Fixed hidden sizes: {HIDDEN_SIZES}")
    print(f"Fixed NN seed: {NN_SEED}")
    if dataset_row_count(dataset_path) <= 0:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    print(f"Total grid size: {total_grid_size()}")
    batch_size = None
    print("Mini-batch size: full dataset")

    candidates_all = all_candidates()
    if args.max_runs is not None:
        candidates_all = candidates_all[: max(0, int(args.max_runs))]

    results, completed = load_existing_results(args.resume, OUTPUT_CSV)
    checkpoint_rows = []
    if args.resume and CHECKPOINT_CSV.exists() and results:
        checkpoint_df = pd.read_csv(CHECKPOINT_CSV)
        if "greedy_mean_steps" in checkpoint_df.columns:
            checkpoint_df = checkpoint_df[checkpoint_df["greedy_mean_steps"].notna()].copy()
            checkpoint_rows = checkpoint_df.to_dict("records")

    candidates = [
        candidate
        for candidate in candidates_all
        if candidate_key(base_row(candidate, primary_device, batch_size=batch_size)) not in completed
    ]

    print(f"Candidates to run: {len(candidates)}")
    if args.resume:
        print(f"Resumed result rows: {len(results)}")
        print(f"Resumed checkpoint rows: {len(checkpoint_rows)}")
    if not args.resume and OUTPUT_CSV.exists():
        print("Existing output will be overwritten because --resume was not set.")

    time_budget_seconds = None
    if args.time_budget_hours is not None and float(args.time_budget_hours) > 0:
        time_budget_seconds = float(args.time_budget_hours) * 3600.0

    started_at = time.perf_counter()
    progress = not args.no_progress
    stopped_for_budget = False

    if workers == 1:
        device = configure_device(devices[0])
        outer = tqdm(
            candidates,
            desc="Continuous FOGAS MountainCar NN grid",
            unit="run",
            disable=not progress,
        )
        for run_idx, params in enumerate(outer, start=len(results) + 1):
            if time_budget_seconds is not None and time.perf_counter() - started_at >= time_budget_seconds:
                stopped_for_budget = True
                break

            row, cp_rows = run_candidate(
                params=params,
                dataset_path=dataset_path,
                device=device,
                eval_trajectories=max(1, int(args.eval_trajectories)),
                eval_max_steps=max(1, int(args.eval_max_steps)),
                batch_size=batch_size,
            )
            row["run_idx"] = int(run_idx)
            for cp_row in cp_rows:
                cp_row["run_idx"] = int(run_idx)
            results.append(row)
            checkpoint_rows.extend(cp_rows)
            save_results(results, OUTPUT_CSV, BEST_CSV)
            save_checkpoints(checkpoint_rows, CHECKPOINT_CSV)

            if progress:
                outer.set_postfix(
                    {
                        "greedy_steps": row["greedy_mean_steps"],
                        "solver_steps": row["solver_mean_steps"],
                        "status": row["status"],
                    }
                )
    else:
        next_run_idx = len(results) + 1
        next_submit_idx = 0
        future_to_candidate = {}
        context = mp.get_context("spawn")

        def can_submit_more():
            if next_submit_idx >= len(candidates):
                return False
            if time_budget_seconds is None:
                return True
            return time.perf_counter() - started_at < time_budget_seconds

        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            def submit_next():
                nonlocal next_submit_idx
                if not can_submit_more():
                    return False
                submit_idx = next_submit_idx
                params = candidates[submit_idx]
                device_name = devices[submit_idx % len(devices)]
                future = executor.submit(
                    run_candidate_worker,
                    params,
                    str(dataset_path),
                    device_name,
                    max(1, int(args.torch_threads)),
                    max(1, int(args.eval_trajectories)),
                    max(1, int(args.eval_max_steps)),
                    batch_size,
                )
                future_to_candidate[future] = (submit_idx, params, device_name)
                next_submit_idx += 1
                return True

            for _ in range(min(workers, len(candidates))):
                submit_next()

            progress_bar = tqdm(
                total=len(candidates),
                initial=0,
                desc="Continuous FOGAS MountainCar NN grid",
                unit="run",
                disable=not progress,
            )
            try:
                while future_to_candidate:
                    for future in as_completed(list(future_to_candidate)):
                        _submit_idx, _params, _device_name = future_to_candidate.pop(future)
                        row, cp_rows = future.result()
                        row["run_idx"] = int(next_run_idx)
                        for cp_row in cp_rows:
                            cp_row["run_idx"] = int(next_run_idx)
                        next_run_idx += 1
                        results.append(row)
                        checkpoint_rows.extend(cp_rows)
                        save_results(results, OUTPUT_CSV, BEST_CSV)
                        save_checkpoints(checkpoint_rows, CHECKPOINT_CSV)
                        progress_bar.update(1)
                        if progress:
                            progress_bar.set_postfix(
                                {
                                    "greedy_steps": row["greedy_mean_steps"],
                                    "solver_steps": row["solver_mean_steps"],
                                    "status": row["status"],
                                }
                            )
                        if not submit_next():
                            if (
                                time_budget_seconds is not None
                                and time.perf_counter() - started_at >= time_budget_seconds
                            ):
                                stopped_for_budget = True
                        break
            finally:
                progress_bar.close()

    save_results(results, OUTPUT_CSV, BEST_CSV)
    save_checkpoints(checkpoint_rows, CHECKPOINT_CSV)
    print(f"Saved grid-search results to {OUTPUT_CSV}")
    print(f"Saved best row to {BEST_CSV}")
    print(f"Saved evaluation checkpoints to {CHECKPOINT_CSV}")
    if stopped_for_budget:
        print("Stopped because the time budget was reached.")


if __name__ == "__main__":
    run_grid_search()
