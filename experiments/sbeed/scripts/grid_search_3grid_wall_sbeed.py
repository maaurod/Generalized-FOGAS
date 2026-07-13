"""Parallel offline SBEED grid search for the deterministic 3x3 wall grid.

The ranking metric is greedy-policy average return with a hard 15-step
evaluation horizon. Each hyperparameter configuration is trained once up to
5k updates and evaluated at 1k, 2k, and 5k updates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from rl_methods.sbeed import (  # noqa: E402
    DiscreteSBEED,
    DiscreteSBEEDDataset,
    LinearRhoParam,
    LinearValueParam,
    SoftmaxLinearPolicyParam,
    TabularStateActionFeatures,
    TabularStateFeatures,
)


N = 9
A = 4
GAMMA = 0.9
X0 = 0

GOAL = 8
PIT = 5
WALL = 4
TERMINAL_STATES = {GOAL, PIT}

STEP_COST = -0.1
GOAL_REWARD = 1.0
PIT_REWARD = -1.0

ACTION_NAMES = ["up", "down", "left", "right"]
STEP_CHECKPOINTS = [1000, 2000, 5000]
MAX_EVAL_STEPS = 15


def to_rc(s: int) -> tuple[int, int]:
    return divmod(int(s), 3)


def to_s(r: int, c: int) -> int:
    return int(r) * 3 + int(c)


def next_state(s: int, a: int) -> int:
    s = int(s)
    a = int(a)

    if s in TERMINAL_STATES:
        return s

    r, c = to_rc(s)

    if a == 0:
        r2, c2 = max(0, r - 1), c
    elif a == 1:
        r2, c2 = min(2, r + 1), c
    elif a == 2:
        r2, c2 = r, max(0, c - 1)
    elif a == 3:
        r2, c2 = r, min(2, c + 1)
    else:
        raise ValueError(f"Unknown action: {a}")

    sp = to_s(r2, c2)
    if sp == WALL:
        return s
    return sp


def reward_for_next_state(sp: int) -> float:
    if int(sp) == GOAL:
        return GOAL_REWARD
    if int(sp) == PIT:
        return PIT_REWARD
    return STEP_COST


def load_dataset(dataset_path: Path, device: torch.device) -> DiscreteSBEEDDataset:
    df = pd.read_csv(dataset_path)
    done = df["next_state"].astype(int).isin([GOAL, PIT]).to_numpy()
    dataset = DiscreteSBEEDDataset(
        X=df["state"].to_numpy(),
        A=df["action"].to_numpy(),
        R=df["reward"].to_numpy(),
        X_next=df["next_state"].to_numpy(),
        D=done,
    )
    dataset.validate(N, A)
    return dataset.to(device)


def evaluate_greedy_policy(pi: np.ndarray, max_eval_steps: int = MAX_EVAL_STEPS) -> Dict[str, float]:
    s = X0
    total_reward = 0.0

    for step in range(max_eval_steps):
        a = int(np.argmax(pi[s]))
        sp = next_state(s, a)
        total_reward += reward_for_next_state(sp)
        s = sp

        if s == GOAL:
            return {
                "avg_return": float(total_reward),
                "success_rate": 1.0,
                "avg_solve_steps": float(step + 1),
                "avg_steps": float(step + 1),
            }
        if s == PIT:
            return {
                "avg_return": float(total_reward),
                "success_rate": 0.0,
                "avg_solve_steps": np.nan,
                "avg_steps": float(step + 1),
            }

    return {
        "avg_return": float(total_reward),
        "success_rate": 0.0,
        "avg_solve_steps": np.nan,
        "avg_steps": float(max_eval_steps),
    }


def policy_columns(pi: np.ndarray) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    best_actions = []
    for s in range(N):
        best_a = int(np.argmax(pi[s]))
        best_actions.append(ACTION_NAMES[best_a])
        row[f"best_s{s}"] = ACTION_NAMES[best_a]
        for a, name in enumerate(ACTION_NAMES):
            row[f"pi_s{s}_{name}"] = float(pi[s, a])
    row["best_actions_json"] = json.dumps(best_actions)
    return row


def make_solver(
    *,
    lr_value: float,
    lr_rho: float,
    lr_policy: float,
    lambda_entropy: float,
    eta: float,
    seed: int,
    device: torch.device,
) -> DiscreteSBEED:
    value_features = TabularStateFeatures(N)
    rho_features = TabularStateActionFeatures(N, A)
    return DiscreteSBEED(
        n_states=N,
        n_actions=A,
        gamma=GAMMA,
        value_param=LinearValueParam(value_features, N),
        rho_param=LinearRhoParam(rho_features, N, A),
        policy_param=SoftmaxLinearPolicyParam(value_features, N, A),
        lambda_entropy=lambda_entropy,
        eta=eta,
        lr_value=lr_value,
        lr_rho=lr_rho,
        lr_policy=lr_policy,
        tau=500.0,
        max_buffer_size=1,
        batch_size=None,
        rollout_length=1,
        seed=seed,
        device=device,
    )


def run_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    torch.set_num_threads(int(config["torch_threads"]))
    device = torch.device(config["device"])

    # Each worker loads the fixed offline buffer, attaches it to a fresh solver,
    # and evaluates the same configuration at the predefined step checkpoints.
    dataset = load_dataset(Path(config["dataset_path"]), device)
    solver = make_solver(
        lr_value=float(config["lr_value"]),
        lr_rho=float(config["lr_rho"]),
        lr_policy=float(config["lr_policy"]),
        lambda_entropy=float(config["lambda_entropy"]),
        eta=float(config["eta"]),
        seed=int(config["seed"]),
        device=device,
    )
    solver.dataset = dataset
    solver.n = solver.dataset.n
    solver.max_buffer_size = solver.n

    rows = []
    last_stats = None
    for t in range(1, max(STEP_CHECKPOINTS) + 1):
        last_stats = solver.step()

        if t in STEP_CHECKPOINTS:
            # Checkpoints make it possible to compare learning speed, not just
            # the final policy after all 5k updates.
            pi = solver.get_policy_matrix().detach().cpu().numpy()
            eval_stats = evaluate_greedy_policy(pi, max_eval_steps=MAX_EVAL_STEPS)
            row = {
                "run": int(config["run"]),
                "seed": int(config["seed"]),
                "steps": t,
                "lr_value": float(config["lr_value"]),
                "lr_rho": float(config["lr_rho"]),
                "lr_policy": float(config["lr_policy"]),
                "lambda_entropy": float(config["lambda_entropy"]),
                "eta": float(config["eta"]),
                "max_eval_steps": MAX_EVAL_STEPS,
                "avg_return": eval_stats["avg_return"],
                "success_rate": eval_stats["success_rate"],
                "avg_solve_steps": eval_stats["avg_solve_steps"],
                "avg_steps": eval_stats["avg_steps"],
                "objective": float(last_stats["objective"]),
                "primal_mse": float(last_stats["primal_mse"]),
                "dual_mse": float(last_stats["dual_mse"]),
            }
            row.update(policy_columns(pi))
            rows.append(row)

    return rows


def build_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    configs = []
    run = 0
    for lr_value, lr_rho, lr_policy, lambda_entropy, eta in product(
        args.lr_values,
        args.lr_values,
        args.lr_values,
        args.lambda_values,
        args.eta_values,
    ):
        run += 1
        configs.append(
            {
                "run": run,
                "dataset_path": str(Path(args.dataset_path).resolve()),
                "lr_value": lr_value,
                "lr_rho": lr_rho,
                "lr_policy": lr_policy,
                "lambda_entropy": lambda_entropy,
                "eta": eta,
                "seed": args.seed,
                "device": args.device,
                "torch_threads": args.torch_threads,
            }
        )
    return configs


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel offline DiscreteSBEED grid search on 3grid_wall.csv."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=REPO_ROOT / "data/datasets/generalization/3grid_wall.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "data/results/sbeed/sbeed_3grid_wall_grid_search.csv",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lr-values",
        type=parse_float_list,
        default=parse_float_list("3e-4,1e-3,3e-3,1e-2,3e-2"),
    )
    parser.add_argument(
        "--lambda-values",
        type=parse_float_list,
        default=parse_float_list("0.0,1e-3,1e-2,5e-2,1e-1"),
    )
    parser.add_argument(
        "--eta-values",
        type=parse_float_list,
        default=parse_float_list("0.25,0.5,0.75,1.0"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = build_configs(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("SBEED 3grid_wall parallel grid search")
    print(f"Dataset       : {Path(args.dataset_path).resolve()}")
    print(f"Output CSV    : {Path(args.output_csv).resolve()}")
    print(f"Configs       : {len(configs)}")
    print(f"Workers       : {args.workers}")
    print(f"Checkpoints   : {STEP_CHECKPOINTS}")
    print(f"Max eval steps: {MAX_EVAL_STEPS}")
    print("Ranking       : avg_return desc, success_rate desc, avg_solve_steps asc")

    all_rows: List[Dict[str, Any]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_config, config) for config in configs]
        for future in as_completed(futures):
            rows = future.result()
            all_rows.extend(rows)
            completed += 1
            if completed % 10 == 0 or completed == len(configs):
                print(f"Completed {completed}/{len(configs)} configs")

    results_df = pd.DataFrame(all_rows)
    results_df = results_df.sort_values(
        ["avg_return", "success_rate", "avg_solve_steps", "objective"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    results_df.to_csv(args.output_csv, index=False)

    print("\nTop 20 by avg_return")
    print("-" * 120)
    display_cols = [
        "run",
        "steps",
        "avg_return",
        "success_rate",
        "avg_solve_steps",
        "lr_value",
        "lr_rho",
        "lr_policy",
        "lambda_entropy",
        "eta",
        "objective",
        "primal_mse",
        "dual_mse",
        "best_actions_json",
    ]
    print(results_df[display_cols].head(20).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved ranked results to: {args.output_csv}")


if __name__ == "__main__":
    main()
