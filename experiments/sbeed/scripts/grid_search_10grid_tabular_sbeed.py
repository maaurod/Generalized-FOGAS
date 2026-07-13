"""Parallel offline DiscreteSBEED grid search for the stochastic 10x10 grid.

The search uses the tabular value, rho, and softmax-linear policy
parametrizations on fixed offline CSV datasets. One row is written per
dataset/hyperparameter/step-count configuration after SBEED updates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


N = 100
A = 4
GAMMA = 0.9
X0 = 0
GRID_SIZE = 10

GOAL_GRID = 99
PIT_GRIDS = {18, 32, 57, 61, 75}
WALL_STATES = {
    4, 11, 14, 17, 21, 22, 27, 34, 37,
    40, 42, 43, 44, 45, 46, 47, 49,
    54, 62, 64, 66, 72, 76, 82, 84, 86, 87, 94,
}
TERMINAL_STATES = {GOAL_GRID, *PIT_GRIDS}
ACTION_NAMES = ["up", "down", "left", "right"]

ETA = 1.0
TAU = 500.0
ROLLOUT_LENGTH = 1
BATCH_SIZE = None
STEP_GRID = [3_000, 5_000, 10_000]
MAX_EVAL_STEPS = 100
MAX_EVAL_STEPS_LONG = 200
N_EVAL_EPISODES = 100

LAMBDA_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
LR_VALUE_GRID = [1e-3, 3e-3, 1e-2]
LR_RHO_GRID = [1e-3, 3e-3, 1e-2, 3e-2]
LR_POLICY_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
DEFAULT_DATASET_PATH = (
    REPO_ROOT / "data/datasets/10grid_tabular.csv"
)
DEFAULT_DATASET_PATHS = (
    REPO_ROOT / "data/datasets/10grid_tabular.csv",
    REPO_ROOT / "data/datasets/10grid_tabular_prueba.csv",
)

CSV_FIELDS = [
    "ok",
    "run_id",
    "seed",
    "device",
    "torch_threads",
    "seconds",
    "dataset_id",
    "dataset_name",
    "dataset_path",
    "dataset_n",
    "steps",
    "lambda_entropy",
    "eta",
    "lr_value",
    "lr_rho",
    "lr_policy",
    "tau",
    "rollout_length",
    "batch_size",
    "max_eval_steps",
    "max_eval_steps_long",
    "n_eval_episodes",
    "eval_return_mean",
    "eval_return_std",
    "eval_return_median",
    "eval_return_min",
    "eval_return_max",
    "eval_length_mean",
    "eval_done_rate",
    "eval_success_rate",
    "eval_success_rate_200",
    "eval_pit_rate",
    "greedy_path",
    "greedy_actions",
    "greedy_return",
    "greedy_success",
    "greedy_hit_pit",
    "greedy_final_state",
    "greedy_length",
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
        raise ValueError("action must be in {0, 1, 2, 3}")

    if not (0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE):
        return s

    sp = pos_to_state(new_row, new_col)
    if sp in WALL_STATES:
        return s
    return sp


def transition_probs(s: int, a: int, intended_prob: float = 0.8) -> Dict[int, float]:
    probs_by_state: Dict[int, float] = {}
    for candidate_a in range(A):
        prob = (1.0 - intended_prob) / A
        if candidate_a == int(a):
            prob += intended_prob
        sp = move_deterministic(s, candidate_a)
        probs_by_state[sp] = probs_by_state.get(sp, 0.0) + prob
    return probs_by_state


def sample_next_state(s: int, a: int, rng: np.random.Generator) -> int:
    probs = transition_probs(s, a)
    next_states = np.fromiter(probs.keys(), dtype=np.int64)
    probabilities = np.fromiter(probs.values(), dtype=np.float64)
    return int(rng.choice(next_states, p=probabilities))


def reward_from_next_state(sp: int) -> float:
    sp = int(sp)
    if sp == GOAL_GRID:
        return 1.0
    if sp in PIT_GRIDS:
        return -5.0
    return -0.1


def load_dataset(dataset_path: Path, device: torch.device) -> DiscreteSBEEDDataset:
    df = pd.read_csv(dataset_path)
    done = df["next_state"].astype(int).isin(TERMINAL_STATES).to_numpy()
    dataset = DiscreteSBEEDDataset(
        X=df["state"].to_numpy(),
        A=df["action"].to_numpy(),
        R=df["reward"].to_numpy(),
        X_next=df["next_state"].to_numpy(),
        D=done,
    )
    dataset.validate(N, A)
    return dataset.to(device)


def build_solver(config: Dict[str, Any], *, dataset_n: int, device: torch.device) -> DiscreteSBEED:
    value_features = TabularStateFeatures(N)
    rho_features = TabularStateActionFeatures(N, A)
    return DiscreteSBEED(
        n_states=N,
        n_actions=A,
        gamma=GAMMA,
        value_param=LinearValueParam(value_features, N),
        rho_param=LinearRhoParam(rho_features, N, A),
        policy_param=SoftmaxLinearPolicyParam(value_features, N, A),
        lambda_entropy=float(config["lambda_entropy"]),
        eta=ETA,
        lr_value=float(config["lr_value"]),
        lr_rho=float(config["lr_rho"]),
        lr_policy=float(config["lr_policy"]),
        tau=TAU,
        max_buffer_size=dataset_n,
        batch_size=BATCH_SIZE,
        rollout_length=ROLLOUT_LENGTH,
        seed=int(config["seed"]),
        device=device,
    )


def policy_summary(pi: np.ndarray) -> Dict[str, Any]:
    best_probs = pi.max(axis=1)
    best_actions = pi.argmax(axis=1)
    entropy = -(np.clip(pi, 1e-12, 1.0) * np.log(np.clip(pi, 1e-12, 1.0))).sum(axis=1)
    return {
        "mean_best_prob": float(best_probs.mean()),
        "min_best_prob": float(best_probs.min()),
        "mean_entropy": float(entropy.mean()),
        "best_actions": json.dumps([ACTION_NAMES[int(a)] for a in best_actions]),
        "best_probs": json.dumps([float(x) for x in best_probs]),
    }


def greedy_rollout(pi: np.ndarray, *, seed: int, max_steps: int = MAX_EVAL_STEPS) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    s = X0
    path = [s]
    actions = []
    total_return = 0.0
    discount = 1.0
    hit_pit = False

    for _ in range(max_steps):
        if s in TERMINAL_STATES:
            break
        a = int(np.argmax(pi[s]))
        sp = sample_next_state(s, a, rng)
        r = reward_from_next_state(sp)
        actions.append(a)
        total_return += discount * r
        discount *= GAMMA
        path.append(sp)
        s = sp
        if s in PIT_GRIDS:
            hit_pit = True
        if s in TERMINAL_STATES:
            break

    return {
        "greedy_path": json.dumps(path),
        "greedy_actions": json.dumps(actions),
        "greedy_return": float(total_return),
        "greedy_success": bool(path[-1] == GOAL_GRID),
        "greedy_hit_pit": bool(hit_pit),
        "greedy_final_state": int(path[-1]),
        "greedy_length": max(0, len(path) - 1),
    }


def evaluate_greedy_policy(
    pi: np.ndarray,
    *,
    seed: int,
    n_episodes: int = N_EVAL_EPISODES,
    max_steps: int = MAX_EVAL_STEPS,
) -> Dict[str, float]:
    returns = []
    lengths = []
    done_count = 0
    success_count = 0
    pit_count = 0

    for episode in range(n_episodes):
        rollout = greedy_rollout(pi, seed=seed + episode, max_steps=max_steps)
        path = json.loads(rollout["greedy_path"])
        returns.append(float(rollout["greedy_return"]))
        lengths.append(max(0, len(path) - 1))
        final_state = int(path[-1])
        if final_state in TERMINAL_STATES:
            done_count += 1
        if final_state == GOAL_GRID:
            success_count += 1
        if final_state in PIT_GRIDS:
            pit_count += 1

    returns_np = np.asarray(returns, dtype=np.float64)
    lengths_np = np.asarray(lengths, dtype=np.float64)
    return {
        "eval_return_mean": float(returns_np.mean()),
        "eval_return_std": float(returns_np.std()),
        "eval_return_median": float(np.median(returns_np)),
        "eval_return_min": float(returns_np.min()),
        "eval_return_max": float(returns_np.max()),
        "eval_length_mean": float(lengths_np.mean()),
        "eval_done_rate": float(done_count / n_episodes),
        "eval_success_rate": float(success_count / n_episodes),
        "eval_pit_rate": float(pit_count / n_episodes),
    }


def evaluate_greedy_success_rate(
    pi: np.ndarray,
    *,
    seed: int,
    n_episodes: int = N_EVAL_EPISODES,
    max_steps: int,
) -> float:
    success_count = 0
    for episode in range(n_episodes):
        rollout = greedy_rollout(pi, seed=seed + episode, max_steps=max_steps)
        path = json.loads(rollout["greedy_path"])
        if int(path[-1]) == GOAL_GRID:
            success_count += 1
    return float(success_count / n_episodes)


def tail_mean(values: List[float]) -> float:
    return float(np.mean(values[-100:])) if values else float("nan")


def tail_std(values: List[float]) -> float:
    return float(np.std(values[-100:])) if values else float("nan")


def run_config(config: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    try:
        torch.set_num_threads(int(config["torch_threads"]))
        device = torch.device(config["device"])
        # Offline search: every run starts from a fixed CSV replay buffer, then
        # applies only SBEED optimization steps before policy evaluation.
        dataset = load_dataset(Path(config["dataset_path"]), device)
        solver = build_solver(config, dataset_n=dataset.n, device=device)
        solver.dataset = dataset
        solver.n = dataset.n

        objective_tail: List[float] = []
        primal_tail: List[float] = []
        dual_tail: List[float] = []
        last_stats: Dict[str, Any] = {}

        for _ in range(int(config["steps"])):
            last_stats = solver.step()
            objective_tail.append(float(last_stats["objective"]))
            primal_tail.append(float(last_stats["primal_mse"]))
            dual_tail.append(float(last_stats["dual_mse"]))
            if len(objective_tail) > 100:
                # Keep a rolling tail so CSV rows show both final metrics and
                # recent stability without storing the full loss history.
                objective_tail.pop(0)
                primal_tail.pop(0)
                dual_tail.pop(0)

        pi = solver.get_policy_matrix().detach().cpu().numpy()
        row: Dict[str, Any] = {
            "ok": True,
            "run_id": int(config["run_id"]),
            "seed": int(config["seed"]),
            "device": str(device),
            "torch_threads": int(config["torch_threads"]),
            "seconds": float(time.time() - start),
            "dataset_id": int(config["dataset_id"]),
            "dataset_name": str(config["dataset_name"]),
            "dataset_path": str(config["dataset_path"]),
            "dataset_n": int(dataset.n),
            "steps": int(config["steps"]),
            "lambda_entropy": float(config["lambda_entropy"]),
            "eta": ETA,
            "lr_value": float(config["lr_value"]),
            "lr_rho": float(config["lr_rho"]),
            "lr_policy": float(config["lr_policy"]),
            "tau": TAU,
            "rollout_length": ROLLOUT_LENGTH,
            "batch_size": "" if BATCH_SIZE is None else BATCH_SIZE,
            "max_eval_steps": MAX_EVAL_STEPS,
            "max_eval_steps_long": MAX_EVAL_STEPS_LONG,
            "n_eval_episodes": N_EVAL_EPISODES,
            "last_objective": float(last_stats.get("objective", np.nan)),
            "last_primal_mse": float(last_stats.get("primal_mse", np.nan)),
            "last_dual_mse": float(last_stats.get("dual_mse", np.nan)),
            "last_theta_grad_norm": float(last_stats.get("theta_grad_norm", np.nan)),
            "last_beta_grad_norm": float(last_stats.get("beta_grad_norm", np.nan)),
            "last_policy_grad_norm": float(last_stats.get("policy_grad_norm", np.nan)),
            "tail_objective_mean": tail_mean(objective_tail),
            "tail_objective_std": tail_std(objective_tail),
            "tail_primal_mse_mean": tail_mean(primal_tail),
            "tail_primal_mse_std": tail_std(primal_tail),
            "tail_dual_mse_mean": tail_mean(dual_tail),
            "tail_dual_mse_std": tail_std(dual_tail),
            "error": "",
        }
        row.update(policy_summary(pi))
        row.update(evaluate_greedy_policy(pi, seed=int(config["seed"])))
        row["eval_success_rate_200"] = evaluate_greedy_success_rate(
            pi,
            seed=int(config["seed"]),
            max_steps=MAX_EVAL_STEPS_LONG,
        )
        row.update(greedy_rollout(pi, seed=int(config["seed"])))
        return row
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "run_id": int(config["run_id"]),
            "seed": int(config["seed"]),
            "device": str(config["device"]),
            "torch_threads": int(config["torch_threads"]),
            "seconds": float(time.time() - start),
            "dataset_id": int(config["dataset_id"]),
            "dataset_name": str(config["dataset_name"]),
            "dataset_path": str(config["dataset_path"]),
            "dataset_n": "",
            "steps": int(config["steps"]),
            "lambda_entropy": float(config["lambda_entropy"]),
            "eta": ETA,
            "lr_value": float(config["lr_value"]),
            "lr_rho": float(config["lr_rho"]),
            "lr_policy": float(config["lr_policy"]),
            "tau": TAU,
            "rollout_length": ROLLOUT_LENGTH,
            "batch_size": "" if BATCH_SIZE is None else BATCH_SIZE,
            "max_eval_steps": MAX_EVAL_STEPS,
            "max_eval_steps_long": MAX_EVAL_STEPS_LONG,
            "n_eval_episodes": N_EVAL_EPISODES,
            "error": repr(exc),
        }


def build_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    # The Cartesian product below is the actual hyperparameter grid. Dataset
    # paths are part of the grid so several offline buffers can be compared.
    configs: List[Dict[str, Any]] = []
    run_id = 0
    devices = list(args.devices)
    for dataset_id, dataset_path in enumerate(args.dataset_paths):
        dataset_path = Path(dataset_path)
        dataset_name = dataset_path.stem
        for steps, lambda_entropy, lr_value, lr_rho, lr_policy in product(
            args.step_grid,
            args.lambda_grid,
            args.lr_value_grid,
            args.lr_rho_grid,
            args.lr_policy_grid,
        ):
            run_id += 1
            configs.append(
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "dataset_path": str(dataset_path.resolve()),
                    "lambda_entropy": lambda_entropy,
                    "lr_value": lr_value,
                    "lr_rho": lr_rho,
                    "lr_policy": lr_policy,
                    "seed": args.seed,
                    "device": devices[(run_id - 1) % len(devices)],
                    "torch_threads": args.torch_threads,
                    "steps": steps,
                }
            )
    return configs


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_path_list(value: str) -> List[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def parse_devices(value: str) -> List[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("At least one device is required.")
    return devices


def completed_run_ids(output_csv: Path) -> set[int]:
    if not output_csv.exists():
        return set()
    try:
        df = pd.read_csv(output_csv, usecols=["run_id", "ok"])
    except Exception:
        return set()
    df = df[df["ok"].astype(str).str.lower().isin(["true", "1"])]
    return {int(run_id) for run_id in df["run_id"].dropna().tolist()}


def append_row(output_csv: Path, row: Dict[str, Any]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = output_csv.exists()
    with output_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def print_top_results(output_csv: Path, n: int = 20) -> None:
    if not output_csv.exists():
        return
    df = pd.read_csv(output_csv)
    if df.empty:
        return
    ok = df[df["ok"].astype(str).str.lower().isin(["true", "1"])].copy()
    if ok.empty:
        print("No successful runs yet.")
        return
    ok["_greedy_success_sort"] = ok["greedy_success"].astype(str).str.lower().isin(["true", "1"])
    goal_count = int(ok["_greedy_success_sort"].sum())
    ok = ok.sort_values(
        ["_greedy_success_sort", "eval_success_rate", "eval_return_mean", "eval_length_mean", "last_objective"],
        ascending=[False, False, False, True, True],
        na_position="last",
    )
    cols = [
        "run_id",
        "dataset_name",
        "greedy_success",
        "greedy_final_state",
        "greedy_length",
        "greedy_return",
        "eval_success_rate",
        "eval_pit_rate",
        "eval_return_mean",
        "eval_length_mean",
        "lambda_entropy",
        "lr_value",
        "lr_rho",
        "lr_policy",
        "last_objective",
        "last_primal_mse",
        "last_dual_mse",
    ]
    print("\nTop results")
    print("-" * 120)
    print(f"Greedy goal-reaching configs found: {goal_count}")
    print(ok[cols].head(n).to_string(index=False, float_format=lambda x: f"{x:.6g}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel offline DiscreteSBEED grid search on both 10grid tabular datasets."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Backward-compatible single-dataset option. Ignored when --dataset-paths is passed.",
    )
    parser.add_argument(
        "--dataset-paths",
        type=parse_path_list,
        default=list(DEFAULT_DATASET_PATHS),
        help="Comma-separated dataset CSV paths. Defaults to 10grid_tabular.csv and 10grid_tabular_prueba.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "data/results/sbeed/sbeed_10grid_tabular_two_datasets_grid_search.csv",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default=None,
        help=(
            "Comma-separated devices assigned round-robin, e.g. cpu or cuda:0,cuda:1. "
            "When omitted, --device is used for every worker."
        ),
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--step-grid",
        type=parse_int_list,
        default=STEP_GRID,
        help="Comma-separated SBEED update counts. Defaults to 3000,5000,10000.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lambda-grid", type=parse_float_list, default=LAMBDA_GRID)
    parser.add_argument("--lr-value-grid", type=parse_float_list, default=LR_VALUE_GRID)
    parser.add_argument("--lr-rho-grid", type=parse_float_list, default=LR_RHO_GRID)
    parser.add_argument("--lr-policy-grid", type=parse_float_list, default=LR_POLICY_GRID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "--dataset-paths" not in sys.argv and "--dataset-path" in sys.argv:
        args.dataset_paths = [args.dataset_path]
    if args.devices is None:
        args.devices = parse_devices(args.device)

    configs = build_configs(args)
    output_csv = Path(args.output_csv)
    missing_dataset_paths = [Path(path) for path in args.dataset_paths if not Path(path).exists()]
    if missing_dataset_paths:
        missing = "\n".join(f"  - {path}" for path in missing_dataset_paths)
        raise FileNotFoundError(
            "The grid search is configured to run both datasets, but these paths do not exist:\n"
            f"{missing}\n"
            "Create the missing CSV(s), or pass --dataset-paths with the available paths."
        )

    if args.resume:
        done = completed_run_ids(output_csv)
        configs = [cfg for cfg in configs if int(cfg["run_id"]) not in done]
        print(f"Resume enabled: skipping {len(done)} successful completed run(s).")

    print("SBEED 10grid tabular parallel grid search")
    print("Datasets   :")
    for dataset_path in args.dataset_paths:
        print(f"  - {Path(dataset_path).resolve()}")
    print(f"Output CSV : {output_csv.resolve()}")
    print(f"Configs    : {len(configs)}")
    print(f"Workers    : {args.workers}")
    print(f"Devices    : {', '.join(args.devices)}")
    print(f"Step grid  : {args.step_grid}")
    print(f"Eval       : {N_EVAL_EPISODES} greedy stochastic rollouts, horizon {MAX_EVAL_STEPS}")
    print("Ranking    : greedy_success desc, eval_success_rate desc, eval_return_mean desc")

    if not configs:
        print_top_results(output_csv)
        return

    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_run = {executor.submit(run_config, cfg): int(cfg["run_id"]) for cfg in configs}
        for future in as_completed(future_to_run):
            row = future.result()
            append_row(output_csv, row)
            completed += 1
            status = "ok" if row.get("ok") else "failed"
            goal_marker = " GOAL" if row.get("greedy_success") else ""
            print(
                f"[{completed}/{len(configs)}] run_id={row.get('run_id')} {status} "
                f"dataset={row.get('dataset_name')} "
                f"greedy_success={row.get('greedy_success', '')}{goal_marker} "
                f"eval_success={row.get('eval_success_rate', '')} "
                f"seconds={float(row.get('seconds', 0.0)):.1f}"
            )

    print_top_results(output_csv)
    print(f"\nSaved results to: {output_csv}")


if __name__ == "__main__":
    main()
