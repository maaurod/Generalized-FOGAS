"""Produce the thesis AlgaeDICE partial-coverage table on the 10 x 10 grid.

Scientific role
---------------
This thesis-facing entry point evaluates the Fenchel/Primal AlgaeDICE baseline
over the same deterministic dataset-generation grid used for FOGAS, FQI,
SBEED, and Generalized FOGAS. Its solver hyperparameters are fixed to the best
row selected by the preceding AlgaeDICE parameter search, so only the offline
data distribution changes.

Inputs and outputs
------------------
Each candidate generates a temporary matched dataset and evaluates the fixed
tabular solver. The final table
``primal_algaedice_dataset_grid_10grid_tabular_new_best_hparams.csv`` is written
under ``data/results/generalization/hyperparam_grids/10grid`` and loaded by
``notebooks/10grid_comparison.ipynb``.

Run this file directly from the repository root after the deterministic
AlgaeDICE hyperparameter search. Use ``--help`` for worker and device controls,
``--max-datasets`` for a smoke test, and ``--resume`` for the full dataset sweep.
Only the parent process writes the checkpointed result table.
"""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - only for minimal environments
    tqdm = None

import grid_search_10grid_dataset_sbeed_generalized_fogas as dataset_grid

from rl_methods.fogas import FOGASEvaluator, FOGASDataset
from rl_methods.fogas_generalization import (
    LinearQFunction,
    PrimalAlgaeDICESolver,
    TabularFeatures,
)
from rl_methods.mdp import Planner


RESULTS_DIR = dataset_grid.RESULTS_DIR
OUTPUT_FILE = "primal_algaedice_dataset_grid_10grid_tabular_new_best_hparams.csv"

ALGAEDICE_CRITIC_UPDATE = "closed_form"
ALGAEDICE_ALPHA = 0.03
ALGAEDICE_ACTOR_LR = 0.03
ALGAEDICE_RIDGE = 1e-10
ALGAEDICE_T = 3_000
ALGAEDICE_BATCH_SIZE = None
ALGAEDICE_CRITIC_LR = 1e-3
ALGAEDICE_CRITIC_INNER_STEPS = 50

FIELDNAMES = dataset_grid.BASE_COLUMNS + [
    "critic_update",
    "alpha",
    "actor_lr",
    "ridge",
    "T",
    "batch_size",
    "critic_lr",
    "critic_inner_steps",
    "device",
    "torch_threads",
    "final_objective",
    "final_actor_loss",
    "final_critic_loss",
    "final_critic_objective",
    "final_theta_norm",
    "final_psi_norm",
    "final_policy_grad_norm",
    "final_actor_delta_mean",
    "final_actor_delta_std",
    "final_critic_delta_mean",
    "final_critic_delta_std",
    "done_fraction",
]


# AlgaeDICE-specific solver construction and shared comparison metrics.
def blank_primal_metrics():
    metrics = dataset_grid.blank_metrics()
    metrics.update(
        {
            "final_objective": np.nan,
            "final_actor_loss": np.nan,
            "final_critic_loss": np.nan,
            "final_critic_objective": np.nan,
            "final_theta_norm": np.nan,
            "final_psi_norm": np.nan,
            "final_policy_grad_norm": np.nan,
            "final_actor_delta_mean": np.nan,
            "final_actor_delta_std": np.nan,
            "final_critic_delta_mean": np.nan,
            "final_critic_delta_std": np.nan,
            "done_fraction": np.nan,
        }
    )
    return metrics


def base_row(candidate, coverage, device, torch_threads, status="ok", error=""):
    row = {
        "run_idx": int(candidate["run_idx"]),
        "algorithm": "Primal AlgaeDICE",
        "dataset_size": int(candidate["dataset_size"]),
        "epsilon": float(candidate["epsilon"]),
        "proportions": candidate["proportion_label"],
        "proportion_key": candidate["proportion_key"],
        "reset_mode": candidate["reset_name"],
        "extra_terminal_steps": dataset_grid.EXTRA_TERMINAL_STEPS,
        "seed": dataset_grid.SEED,
        "status": status,
        "error": error,
        "elapsed_seconds": np.nan,
        "critic_update": ALGAEDICE_CRITIC_UPDATE,
        "alpha": ALGAEDICE_ALPHA,
        "actor_lr": ALGAEDICE_ACTOR_LR,
        "ridge": ALGAEDICE_RIDGE,
        "T": ALGAEDICE_T,
        "batch_size": "" if ALGAEDICE_BATCH_SIZE is None else ALGAEDICE_BATCH_SIZE,
        "critic_lr": ALGAEDICE_CRITIC_LR,
        "critic_inner_steps": ALGAEDICE_CRITIC_INNER_STEPS,
        "device": str(device),
        "torch_threads": int(torch_threads),
    }
    row.update(blank_primal_metrics())
    row["feature_coverage"] = coverage
    return row


def build_primal_solver(dataset_path, device):
    q_features = TabularFeatures(dataset_grid.N, dataset_grid.A)
    policy_features = TabularFeatures(dataset_grid.N, dataset_grid.A)

    return PrimalAlgaeDICESolver(
        n_states=dataset_grid.N,
        n_actions=dataset_grid.A,
        gamma=dataset_grid.GAMMA,
        x0=dataset_grid.X0,
        csv_path=str(dataset_path),
        q_function=LinearQFunction(q_features),
        policy_features=policy_features,
        seed=dataset_grid.SEED,
        device=device,
        alpha=ALGAEDICE_ALPHA,
        ridge=ALGAEDICE_RIDGE,
        actor_lr=ALGAEDICE_ACTOR_LR,
        T=ALGAEDICE_T,
        critic_update=ALGAEDICE_CRITIC_UPDATE,
        batch_size=ALGAEDICE_BATCH_SIZE,
        critic_lr=ALGAEDICE_CRITIC_LR,
        critic_inner_steps=ALGAEDICE_CRITIC_INNER_STEPS,
        terminal_states=dataset_grid.TERMINAL_STATES,
        init_states=[dataset_grid.X0],
    )


def evaluate_policy_family(evaluator, dataset, policy_mode, d_star, v_star):
    pi = evaluator.get_policy(policy_mode)
    v_pi, _ = evaluator.planner.evaluate_policy(pi)
    v_gap = float((d_star * (v_star - v_pi.detach().cpu())).sum().item())
    return {
        f"{policy_mode}_on_data_quality": evaluator.on_data_quality(
            dataset=dataset,
            policy_mode=policy_mode,
            compare_with_optimal=True,
        )["policy"],
        f"{policy_mode}_optimal_states_quality": evaluator.optimal_states_quality(
            policy_mode=policy_mode,
            num_trajectories=dataset_grid.NUM_TRAJECTORIES,
            max_steps=dataset_grid.MAX_STEPS,
            seed=dataset_grid.SEED,
        )["policy"],
        f"{policy_mode}_avg_reward": evaluator.average_return(
            policy_mode=policy_mode,
            num_trajectories=dataset_grid.NUM_TRAJECTORIES,
            max_steps=dataset_grid.MAX_STEPS,
            seed=dataset_grid.SEED,
            terminal_states=dataset_grid.TERMINAL_STATES,
        )["policy"],
        f"{policy_mode}_success_rate": evaluator.success_rate(
            goal_state=dataset_grid.GOAL_GRID,
            policy_mode=policy_mode,
            num_trajectories=dataset_grid.NUM_TRAJECTORIES,
            max_steps=dataset_grid.MAX_STEPS,
            seed=dataset_grid.SEED,
            terminal_states=dataset_grid.TERMINAL_STATES,
        )["policy"],
        f"{policy_mode}_v_x0": float(v_pi[evaluator.mdp.x0].item()),
        f"{policy_mode}_v_gap": v_gap,
    }


def finite_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def run_primal_algaedice(
    candidate,
    dataset_path,
    dataset,
    mdp,
    planner,
    d_star,
    v_star,
    coverage,
    device,
    torch_threads,
    base_elapsed,
):
    start = time.perf_counter()
    row = base_row(candidate, coverage, device, torch_threads)

    try:
        solver = build_primal_solver(dataset_path, device)
        evaluator = FOGASEvaluator(solver=solver, mdp=mdp, planner=planner)

        solver.run(
            alpha=ALGAEDICE_ALPHA,
            actor_lr=ALGAEDICE_ACTOR_LR,
            ridge=ALGAEDICE_RIDGE,
            T=ALGAEDICE_T,
            critic_update=ALGAEDICE_CRITIC_UPDATE,
            batch_size=ALGAEDICE_BATCH_SIZE,
            critic_lr=ALGAEDICE_CRITIC_LR,
            critic_inner_steps=ALGAEDICE_CRITIC_INNER_STEPS,
            tqdm_print=False,
            verbose=False,
        )

        for mode in ("greedy", "solver"):
            row.update(evaluate_policy_family(evaluator, dataset, mode, d_star, v_star))

        diagnostics = solver.get_diagnostics() or []
        if diagnostics:
            final = diagnostics[-1]
            row.update(
                {
                    "final_objective": finite_float(final.get("objective")),
                    "final_actor_loss": finite_float(final.get("actor_loss")),
                    "final_critic_loss": finite_float(final.get("critic_loss")),
                    "final_critic_objective": finite_float(final.get("critic_objective")),
                    "final_theta_norm": finite_float(final.get("theta_norm")),
                    "final_psi_norm": finite_float(final.get("psi_norm")),
                    "final_policy_grad_norm": finite_float(final.get("policy_grad_norm")),
                    "final_actor_delta_mean": finite_float(final.get("actor_delta_mean")),
                    "final_actor_delta_std": finite_float(final.get("actor_delta_std")),
                    "final_critic_delta_mean": finite_float(final.get("critic_delta_mean")),
                    "final_critic_delta_std": finite_float(final.get("critic_delta_std")),
                    "done_fraction": finite_float(final.get("done_fraction")),
                }
            )
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        row["elapsed_seconds"] = float(base_elapsed + time.perf_counter() - start)

    return row


def failed_row(candidate, exc, elapsed):
    row = base_row(candidate, np.nan, "unknown", 1, status="failed", error=repr(exc))
    row["elapsed_seconds"] = float(elapsed)
    return row


def run_dataset_worker(payload):
    candidate, device_str, torch_threads = payload
    dataset_grid.configure_worker_threads(torch_threads)
    dataset_grid.set_seed(dataset_grid.SEED)
    device = torch.device(device_str)

    worker_start = time.perf_counter()
    try:
        mdp, phi, states, actions = dataset_grid.build_mdp()
        planner = Planner(mdp)
        phi_full = dataset_grid.full_feature_matrix(states, actions, phi)
        d_star = (
            planner.mu_star.cpu() / (planner.mu_star.cpu().sum() + 1e-300)
        ).reshape(dataset_grid.N, dataset_grid.A).sum(dim=1)
        v_star = planner.v_star.detach().cpu()

        with tempfile.TemporaryDirectory(prefix="dataset_grid_10grid_", dir="/tmp") as tmp:
            dataset_path = Path(tmp) / f"dataset_{candidate['run_idx']}.csv"
            dataset_rows = dataset_grid.collect_dataset_csv(
                mdp,
                planner.pi_star,
                candidate,
                dataset_path,
            )
            coverage = dataset_grid.feature_coverage(
                dataset_rows,
                phi_full,
                planner.mu_star,
                dataset_grid.COVERAGE_BETA,
            )
            dataset = FOGASDataset(dataset_path)
            base_elapsed = time.perf_counter() - worker_start

            mdp.to(device)
            planner.to(device)

            return run_primal_algaedice(
                candidate=candidate,
                dataset_path=dataset_path,
                dataset=dataset,
                mdp=mdp,
                planner=planner,
                d_star=d_star,
                v_star=v_star,
                coverage=coverage,
                device=device,
                torch_threads=torch_threads,
                base_elapsed=base_elapsed,
            )
    except Exception as exc:  # noqa: BLE001
        return failed_row(candidate, exc, time.perf_counter() - worker_start)


# Resumable dataset tasks and parent-owned result aggregation.
def parse_devices(value):
    devices = [item.strip() for item in str(value).split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("At least one device is required.")
    return devices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the deterministic 10-grid dataset search for Primal AlgaeDICE."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default=parse_devices("cpu"),
        help="Comma-separated devices assigned round-robin, e.g. cpu or cuda:0,cuda:1.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch CPU threads per worker.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip successful dataset rows already present in the output CSV.",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Limit the number of dataset variants. Useful for smoke tests.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where the result CSV is written.",
    )
    return parser.parse_args()


def candidate_key(candidate_or_row):
    return (
        int(candidate_or_row["dataset_size"]),
        float(candidate_or_row["epsilon"]),
        str(candidate_or_row["proportion_key"]),
        str(
            candidate_or_row["reset_mode"]
            if "reset_mode" in candidate_or_row
            else candidate_or_row["reset_name"]
        ),
    )


def successful_completed_keys(output_csv):
    if not output_csv.exists():
        return set()
    completed = set()
    with output_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("status", "")).lower() != "ok":
                continue
            try:
                completed.add(candidate_key(row))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def append_row(output_csv, row):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_tasks(candidates, output_csv, resume, devices, torch_threads):
    completed = successful_completed_keys(output_csv) if resume else set()
    tasks = []
    for candidate in candidates:
        if candidate_key(candidate) in completed:
            continue
        device = devices[len(tasks) % len(devices)]
        tasks.append((candidate, device, torch_threads))
    return tasks, completed


def progress_iter(iterable, total, desc, disable):
    if tqdm is None or disable:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="dataset")


def main():
    args = parse_args()
    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    output_dir = Path(args.output_dir)
    output_csv = output_dir / OUTPUT_FILE

    candidates = dataset_grid.dataset_candidates()
    if args.max_datasets is not None:
        candidates = candidates[: max(0, int(args.max_datasets))]

    if not args.resume and output_csv.exists():
        output_csv.unlink()

    tasks, completed = build_tasks(
        candidates=candidates,
        output_csv=output_csv,
        resume=args.resume,
        devices=args.devices,
        torch_threads=torch_threads,
    )

    print("10-grid deterministic Primal AlgaeDICE dataset search")
    print(f"Output CSV       : {output_csv.resolve()}")
    print(f"Dataset variants : {len(candidates)}")
    print(f"Tasks to run     : {len(tasks)}")
    print(f"Workers          : {workers}")
    print(f"Devices          : {', '.join(args.devices)}")
    print(f"Torch threads    : {torch_threads}")
    print(
        "Hyperparameters  : "
        f"critic_update={ALGAEDICE_CRITIC_UPDATE}, "
        f"alpha={ALGAEDICE_ALPHA}, "
        f"actor_lr={ALGAEDICE_ACTOR_LR}, "
        f"ridge={ALGAEDICE_RIDGE}, "
        f"T={ALGAEDICE_T}"
    )
    if args.resume:
        print(f"Resume completed : {len(completed)} completed row(s)")

    if not tasks:
        print("No tasks to run.")
        return

    completed_count = 0
    if workers == 1:
        iterator = progress_iter(tasks, len(tasks), "Dataset grid", args.no_progress)
        for task in iterator:
            row = run_dataset_worker(task)
            append_row(output_csv, row)
            completed_count += 1
            if args.no_progress:
                print(
                    f"[{completed_count}/{len(tasks)}] "
                    f"run_idx={task[0]['run_idx']} status={row.get('status')}"
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(run_dataset_worker, task): task for task in tasks}
            iterator = progress_iter(
                as_completed(future_to_task),
                len(future_to_task),
                "Dataset grid",
                args.no_progress,
            )
            for future in iterator:
                task = future_to_task[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = failed_row(task[0], exc, 0.0)
                append_row(output_csv, row)
                completed_count += 1
                if args.no_progress:
                    print(
                        f"[{completed_count}/{len(tasks)}] "
                        f"run_idx={task[0]['run_idx']} status={row.get('status')}"
                    )

    print("\nDataset grid complete.")
    print(f"Primal AlgaeDICE: {output_csv}")


if __name__ == "__main__":
    main()
