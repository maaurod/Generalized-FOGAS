"""Grid search for ContinuousSBEED on Pendulum-v1.

This script evaluates the final continuous solver with neural value/rho models
and an RFF Gaussian policy. It records both training diagnostics and periodic
deterministic evaluation returns so runs can be ranked after long jobs finish.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def find_root(current_path: Path) -> Path:
    current_path = current_path.resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / "src" / "rl_methods").exists() and (parent / "data").exists():
            return parent
    return current_path


PROJECT_ROOT = find_root(Path(__file__))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SEED = 42
ENV_ID = "Pendulum-v1"

# Values that stay fixed across the continuous search. The grid below only
# changes the learning rates, eta/lambda/rollout/batch choices, and scheduler.
FIXED_VALUES = {
    "gamma": 0.995,
    "initial_random_steps": 20000,
    "collect_per_episode": 1000,
    "updates_per_episode": 25,
    "max_buffer_size": 400000,
    "hidden_size": 64,
    "rff_features": 100,
    "nu": None,
    "init_log_std": -1.5,
    "fisher_damping": 1e-2,
    "cg_iters": 10,
    "seed": SEED,
    "log_every": 5,
}

# Scheduler variants tested for all optimizer groups.
SCHEDULER_GRID = [
    {
        "name": "inverse_time_tau_5000",
        "lr_schedulers": {
            "value": "inverse_time",
            "rho": "inverse_time",
            "policy": "inverse_time",
        },
        "tau": 5000.0,
    },
    {
        "name": "inverse_time_tau_20000",
        "lr_schedulers": {
            "value": "inverse_time",
            "rho": "inverse_time",
            "policy": "inverse_time",
        },
        "tau": 20000.0,
    },
]

# Stable CSV schema for post-processing and ranking runs.
FIELDNAMES = [
    "experiment_id",
    "status",
    "error",
    "seconds",
    "env_id",
    "seed",
    "device",
    "episodes",
    "env_steps",
    "optimization_updates",
    "initial_random_steps",
    "collect_per_episode",
    "updates_per_episode",
    "max_buffer_size",
    "hidden_size",
    "rff_features",
    "nu",
    "init_log_std",
    "gamma",
    "learning_rate",
    "lr_value",
    "lr_rho",
    "lr_policy",
    "eta",
    "lambda_entropy",
    "rollout_length",
    "batch_size",
    "scheduler_name",
    "value_lr_scheduler",
    "rho_lr_scheduler",
    "policy_lr_scheduler",
    "tau",
    "fisher_damping",
    "cg_iters",
    "buffer_size",
    "train_return_count",
    "train_return_mean",
    "train_return_recent_10_mean",
    "avg_reward",
    "best_eval_episode",
    "best_eval_update_index",
    "best_eval_avg_reward",
    "best_eval_return_mean",
    "best_eval_return_std",
    "best_eval_return_min",
    "best_eval_return_max",
    "best_eval_return_median",
    "best_eval_length_mean",
    "final_eval_avg_reward",
    "final_eval_return_mean",
    "final_eval_return_std",
    "final_eval_return_min",
    "final_eval_return_max",
    "final_eval_return_median",
    "final_eval_length_mean",
    "eval_return_mean",
    "eval_return_std",
    "eval_return_min",
    "eval_return_max",
    "eval_return_median",
    "eval_length_mean",
    "last_objective",
    "last_primal_mse",
    "last_dual_mse",
    "last_theta_grad_norm",
    "last_beta_grad_norm",
    "last_policy_grad_norm",
    "last_policy_direction_norm",
    "tail_objective_mean",
    "tail_objective_std",
    "tail_primal_mse_mean",
    "tail_primal_mse_std",
    "tail_dual_mse_mean",
    "tail_dual_mse_std",
    "tail_theta_grad_norm_mean",
    "tail_theta_grad_norm_std",
    "tail_beta_grad_norm_mean",
    "tail_beta_grad_norm_std",
    "tail_policy_grad_norm_mean",
    "tail_policy_grad_norm_std",
    "eval_returns_json",
    "periodic_eval_history_json",
]


def finite_mean(values: Iterable[float]) -> float:
    import numpy as np

    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def finite_std(values: Iterable[float]) -> float:
    import numpy as np

    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    return float(arr.std(ddof=1)) if arr.size > 1 else 0.0 if arr.size == 1 else float("nan")


def tail_values(history: list[Dict[str, float]], key: str, tail: int = 100) -> list[float]:
    return [float(row[key]) for row in history[-tail:] if key in row and math.isfinite(float(row[key]))]


def build_solver(env: Any, cfg: Dict[str, Any], device: str) -> Any:
    import numpy as np
    import torch

    from rl_methods.sbeed.features import (
        ContinuousNeuralRhoParam,
        ContinuousNeuralValueParam,
        ContinuousStateActionMLPModule,
        ContinuousStateMLPValueModule,
        RFFGaussianPolicyParam,
    )
    from rl_methods.sbeed.solvers import ContinuousSBEED

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    hidden_sizes = (int(cfg["hidden_size"]), int(cfg["hidden_size"]))

    value_param = ContinuousNeuralValueParam(
        ContinuousStateMLPValueModule(
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            dtype=torch.float32,
        )
    )
    rho_param = ContinuousNeuralRhoParam(
        ContinuousStateActionMLPModule(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            dtype=torch.float32,
        )
    )
    policy_param = RFFGaussianPolicyParam(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_features=int(cfg["rff_features"]),
        nu=cfg["nu"],
        init_log_std=float(cfg["init_log_std"]),
        dtype=torch.float32,
        seed=int(cfg["seed"]),
    )

    return ContinuousSBEED(
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=float(cfg["gamma"]),
        value_param=value_param,
        rho_param=rho_param,
        policy_param=policy_param,
        lambda_entropy=float(cfg["lambda_entropy"]),
        eta=float(cfg["eta"]),
        lr_value=float(cfg["lr_value"]),
        lr_rho=float(cfg["lr_rho"]),
        lr_policy=float(cfg["lr_policy"]),
        lr_schedulers=cfg["lr_schedulers"],
        batch_size=int(cfg["batch_size"]),
        rollout_length=int(cfg["rollout_length"]),
        max_buffer_size=int(cfg["max_buffer_size"]),
        fisher_damping=float(cfg["fisher_damping"]),
        cg_iters=int(cfg["cg_iters"]),
        tau=float(cfg["tau"]),
        seed=int(cfg["seed"]),
        device=device,
    )


def evaluate_policy(
    solver: Any,
    *,
    env_id: str,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> Dict[str, Any]:
    import gymnasium as gym
    import numpy as np

    env = gym.make(env_id)
    returns = []
    lengths = []
    try:
        for ep in range(int(episodes)):
            reset_result = env.reset(seed=seed + ep)
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            done = False
            ep_return = 0.0
            steps = 0
            while not done:
                action = solver.sample_action(obs, deterministic=deterministic, clip=True)
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, _ = step_result
                    done = bool(terminated) or bool(truncated)
                else:
                    obs, reward, done, _ = step_result
                    done = bool(done)
                ep_return += float(reward)
                steps += 1
            returns.append(float(ep_return))
            lengths.append(int(steps))
    finally:
        env.close()

    returns_arr = np.asarray(returns, dtype=float)
    lengths_arr = np.asarray(lengths, dtype=float)
    return {
        "avg_reward": float(returns_arr.mean()) if returns_arr.size else float("nan"),
        "eval_return_mean": float(returns_arr.mean()) if returns_arr.size else float("nan"),
        "eval_return_std": float(returns_arr.std(ddof=1)) if returns_arr.size > 1 else 0.0,
        "eval_return_min": float(returns_arr.min()) if returns_arr.size else float("nan"),
        "eval_return_max": float(returns_arr.max()) if returns_arr.size else float("nan"),
        "eval_return_median": float(np.median(returns_arr)) if returns_arr.size else float("nan"),
        "eval_length_mean": float(lengths_arr.mean()) if lengths_arr.size else float("nan"),
        "eval_returns_json": json.dumps(returns),
    }


def prefixed_stats(stats: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {
        f"{prefix}_avg_reward": stats.get("avg_reward", float("nan")),
        f"{prefix}_return_mean": stats.get("eval_return_mean", float("nan")),
        f"{prefix}_return_std": stats.get("eval_return_std", float("nan")),
        f"{prefix}_return_min": stats.get("eval_return_min", float("nan")),
        f"{prefix}_return_max": stats.get("eval_return_max", float("nan")),
        f"{prefix}_return_median": stats.get("eval_return_median", float("nan")),
        f"{prefix}_length_mean": stats.get("eval_length_mean", float("nan")),
    }


def best_eval_from_history(eval_history: list[Dict[str, Any]]) -> Dict[str, Any]:
    finite_rows = [
        row for row in eval_history
        if math.isfinite(float(row.get("eval_return_mean", float("nan"))))
    ]
    if not finite_rows:
        return {
            "best_eval_episode": "",
            "best_eval_update_index": "",
            "best_eval_avg_reward": float("nan"),
            "best_eval_return_mean": float("nan"),
            "best_eval_return_std": float("nan"),
            "best_eval_return_min": float("nan"),
            "best_eval_return_max": float("nan"),
            "best_eval_return_median": float("nan"),
            "best_eval_length_mean": float("nan"),
        }
    best = max(finite_rows, key=lambda row: float(row["eval_return_mean"]))
    return {
        "best_eval_episode": int(best.get("episode", -1)),
        "best_eval_update_index": int(best.get("update_index", -1)),
        **prefixed_stats(best, "best_eval"),
    }


def make_grid() -> list[Dict[str, Any]]:
    configs = []
    product = itertools.product(
        [220, 330],
        [0.001, 0.0001],
        [0.004, 0.01, 0.04, 0.1],
        [0.001, 0.01, 0.1],
        [1, 10, 20],
        [10000, 20000],
        SCHEDULER_GRID,
    )
    for experiment_id, (
        episodes,
        learning_rate,
        eta,
        lambda_entropy,
        rollout_length,
        batch_size,
        scheduler,
    ) in enumerate(product):
        cfg = {
            **FIXED_VALUES,
            "experiment_id": experiment_id,
            "env_id": ENV_ID,
            "episodes": int(episodes),
            "learning_rate": float(learning_rate),
            "lr_value": float(learning_rate),
            "lr_rho": float(learning_rate),
            "lr_policy": float(learning_rate),
            "eta": float(eta),
            "lambda_entropy": float(lambda_entropy),
            "rollout_length": int(rollout_length),
            "batch_size": int(batch_size),
            "scheduler_name": scheduler["name"],
            "lr_schedulers": dict(scheduler["lr_schedulers"]),
            "tau": float(scheduler["tau"]),
        }
        cfg["env_steps"] = cfg["initial_random_steps"] + cfg["episodes"] * cfg["collect_per_episode"]
        cfg["optimization_updates"] = cfg["episodes"] * cfg["updates_per_episode"]
        configs.append(cfg)
    return configs


def run_one(
    cfg: Dict[str, Any],
    device: str,
    eval_episodes: int,
    eval_every_episodes: int,
) -> Dict[str, Any]:
    start_time = time.time()
    env = None
    solver: Optional[Any] = None
    try:
        import gymnasium as gym
        import numpy as np
        import torch

        seed = int(cfg["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        env = gym.make(cfg["env_id"])
        solver = build_solver(env, cfg, device=device)

        def eval_callback(eval_solver: Any, episode: int, _: Optional[Dict[str, float]]) -> Dict[str, Any]:
            stats = evaluate_policy(
                eval_solver,
                env_id=str(cfg["env_id"]),
                episodes=int(eval_episodes),
                seed=seed + 10_000 + int(episode),
                deterministic=True,
            )
            print(
                f"    eval episode={episode:04d} "
                f"avg_reward={stats['avg_reward']:.3f} "
                f"std={stats['eval_return_std']:.3f}",
                flush=True,
            )
            return stats

        result = solver.run_env(
            env,
            episodes=int(cfg["episodes"]),
            initial_random_steps=int(cfg["initial_random_steps"]),
            collect_per_episode=int(cfg["collect_per_episode"]),
            updates_per_episode=int(cfg["updates_per_episode"]),
            log_every=int(cfg["log_every"]),
            eval_every_episodes=int(eval_every_episodes),
            eval_callback=eval_callback,
        )
        eval_stats = evaluate_policy(
            solver,
            env_id=str(cfg["env_id"]),
            episodes=int(eval_episodes),
            seed=seed + 999_999,
            deterministic=True,
        )
        returns = [float(x) for x in result["episode_returns"]]
        last_loss = solver.loss_history[-1] if solver.loss_history else result.get("last_stats") or {}
        periodic_eval_history = result.get("eval_history", [])
        best_eval_stats = best_eval_from_history(periodic_eval_history)

        row = {
            "experiment_id": int(cfg["experiment_id"]),
            "status": "ok",
            "error": "",
            "seconds": float(time.time() - start_time),
            "device": device,
            "buffer_size": int(result["buffer_size"]),
            "train_return_count": int(len(returns)),
            "train_return_mean": finite_mean(returns),
            "train_return_recent_10_mean": finite_mean(returns[-10:]),
            **{k: v for k, v in cfg.items() if k != "lr_schedulers"},
            "value_lr_scheduler": cfg["lr_schedulers"]["value"],
            "rho_lr_scheduler": cfg["lr_schedulers"]["rho"],
            "policy_lr_scheduler": cfg["lr_schedulers"]["policy"],
            **best_eval_stats,
            **prefixed_stats(eval_stats, "final_eval"),
            **eval_stats,
            "periodic_eval_history_json": json.dumps(periodic_eval_history),
            "last_objective": float(last_loss.get("objective", float("nan"))),
            "last_primal_mse": float(last_loss.get("primal_mse", float("nan"))),
            "last_dual_mse": float(last_loss.get("dual_mse", float("nan"))),
            "last_theta_grad_norm": float(last_loss.get("theta_grad_norm", float("nan"))),
            "last_beta_grad_norm": float(last_loss.get("beta_grad_norm", float("nan"))),
            "last_policy_grad_norm": float(last_loss.get("policy_grad_norm", float("nan"))),
            "last_policy_direction_norm": float(last_loss.get("policy_direction_norm", float("nan"))),
        }
        for key in (
            "objective",
            "primal_mse",
            "dual_mse",
            "theta_grad_norm",
            "beta_grad_norm",
            "policy_grad_norm",
        ):
            values = tail_values(solver.loss_history, key)
            row[f"tail_{key}_mean"] = finite_mean(values)
            row[f"tail_{key}_std"] = finite_std(values)
        return normalize_row(row)
    except Exception as exc:
        row = {
            "experiment_id": int(cfg["experiment_id"]),
            "status": "error",
            "error": repr(exc),
            "seconds": float(time.time() - start_time),
            "device": device,
            **{k: v for k, v in cfg.items() if k != "lr_schedulers"},
            "value_lr_scheduler": cfg["lr_schedulers"]["value"],
            "rho_lr_scheduler": cfg["lr_schedulers"]["rho"],
            "policy_lr_scheduler": cfg["lr_schedulers"]["policy"],
            "avg_reward": -float("inf"),
            "best_eval_avg_reward": -float("inf"),
            "best_eval_return_mean": -float("inf"),
            "final_eval_avg_reward": -float("inf"),
            "final_eval_return_mean": -float("inf"),
            "eval_return_mean": -float("inf"),
            "eval_returns_json": "[]",
            "periodic_eval_history_json": "[]",
        }
        return normalize_row(row)
    finally:
        if env is not None:
            env.close()
        del solver
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for name in FIELDNAMES:
        value = row.get(name, "")
        if value is None:
            value = ""
        normalized[name] = value
    return normalized


def read_finished_ids(csv_path: Path, retry_failed: bool) -> set[int]:
    if not csv_path.exists():
        return set()
    finished = set()
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("experiment_id"):
                continue
            if retry_failed and row.get("status") != "ok":
                continue
            finished.add(int(row["experiment_id"]))
    return finished


def append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def parse_devices(raw: str, workers: int) -> list[str]:
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                return [f"cuda:{i}" for i in range(count)]
    except ImportError:
        pass
    return ["cpu"] * max(1, workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 576-experiment Pendulum ContinuousSBEED grid.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "results" / "sbeed" / "continuous_pendulum_grid.csv",
    )
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers. Use 0 to match visible device count.")
    parser.add_argument("--devices", type=str, default="")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-every-episodes", type=int, default=10)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Run on CPU when no CUDA devices are visible.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = make_grid()
    if len(grid) != 576:
        raise RuntimeError(f"Expected 576 configs, got {len(grid)}")

    finished_ids = read_finished_ids(args.csv, retry_failed=args.retry_failed)
    pending = [cfg for cfg in grid if int(cfg["experiment_id"]) not in finished_ids]
    if args.limit is not None:
        pending = pending[: int(args.limit)]

    print(f"csv={args.csv}")
    print(f"total=576 finished={len(finished_ids)} pending={len(pending)}")
    if args.dry_run:
        for cfg in pending[:5]:
            print(cfg)
        return
    if not pending:
        return

    devices = parse_devices(args.devices, max(1, int(args.workers)))
    if all(device == "cpu" for device in devices) and not args.allow_cpu:
        raise RuntimeError(
            "No CUDA devices are visible. Re-run on a GPU node/session, pass --devices explicitly, "
            "or add --allow-cpu if you really want the full grid on CPU."
        )
    workers = len(devices) if int(args.workers) <= 0 else max(1, int(args.workers))
    print(f"workers={workers} devices={devices}")

    if workers == 1:
        for done_count, cfg in enumerate(pending, start=1):
            device = devices[int(cfg["experiment_id"]) % len(devices)]
            print(
                f"[{done_count}/{len(pending)}] "
                f"experiment_id={cfg['experiment_id']} device={device}",
                flush=True,
            )
            row = run_one(
                cfg,
                device=device,
                eval_episodes=int(args.eval_episodes),
                eval_every_episodes=int(args.eval_every_episodes),
            )
            append_row(args.csv, row)
            print(
                f"    status={row['status']} avg_reward={row['avg_reward']} "
                f"seconds={float(row['seconds']):.1f}",
                flush=True,
            )
        return

    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        async_results = []
        for cfg in pending:
            device = devices[int(cfg["experiment_id"]) % len(devices)]
            async_results.append(
                pool.apply_async(
                    run_one,
                    kwds={
                        "cfg": cfg,
                        "device": device,
                        "eval_episodes": int(args.eval_episodes),
                        "eval_every_episodes": int(args.eval_every_episodes),
                    },
                )
            )

        for done_count, result in enumerate(async_results, start=1):
            row = result.get()
            append_row(args.csv, row)
            print(
                f"[{done_count}/{len(async_results)}] "
                f"experiment_id={row['experiment_id']} status={row['status']} "
                f"avg_reward={row['avg_reward']} seconds={float(row['seconds']):.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
