from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from rbf_grid_search_common import (
    A,
    GAMMA,
    GOAL_GRID,
    N,
    REPO_ROOT,
    TERMINAL_STATES,
    X0,
    evaluate_policy,
    extract_policy_summary,
    next_state_deterministic,
    print_best_result,
    result_for_csv,
    reward_fn,
    score_result,
    summarize_top_results,
    tail_mean,
    tail_std,
)

from rl_methods.sbeed.features import (  # noqa: E402
    IdentityHead,
    NeuralPolicyParam,
    NeuralRhoParam,
    NeuralValueParam,
    StateActionMLPModule,
    StateMLPPolicyModule,
    StateMLPValueModule,
)
from rl_methods.sbeed.solvers import DiscreteSBEED  # noqa: E402


def make_grid_150() -> List[Dict[str, Any]]:
    grid_150 = []
    for episodes in [400, 800]:
        for batch_size in [512, 1024]:
            for tau in [1000.0, 10000.0, 100000.0]:
                for lambda_entropy in [0.005, 0.01, 0.05, 0.1]:
                    for lr_policy, fisher_damping in [
                        (1e-3, 1e-1),
                        (1e-3, 3e-1),
                        (3e-3, 1e-1),
                        (3e-3, 3e-1),
                        (1e-2, 3e-1),
                    ]:
                        grid_150.append(
                            dict(
                                episodes=episodes,
                                batch_size=batch_size,
                                tau=tau,
                                lr_value=1e-3,
                                lr_rho=3e-3,
                                lr_policy=lr_policy,
                                lambda_entropy=lambda_entropy,
                                fisher_damping=fisher_damping,
                            )
                        )
    return grid_150


CONFIGS_TO_TRY = make_grid_150()
SEED = 42
REQUESTED_DEVICE = "auto"

NEURAL_DEFAULTS = dict(
    eta=0.1,
    rollout_length=3,
    batch_size=256,
    epsilon=0.3,
)

NEURAL_TRAINING_KWARGS = dict(
    episodes=400,
    collect_per_episode=20,
    updates_per_episode=10,
    initial_collect_steps=512,
    max_buffer_size=12000,
    tau=1000.0,
)

NEURAL_CSV_FIELDS = [
    "ok",
    "run_id",
    "config_id",
    "seed",
    "device",
    "requested_device",
    "torch_threads",
    "transition_name",
    "network_init_seed",
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
    "greedy_path",
    "greedy_actions",
    "greedy_return",
    "greedy_success",
    "last_objective",
    "main_objective_loss",
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
    "objective_loss_history",
    "eval_history",
    "error",
]


def seed_everything(seed: int, device: Optional[torch.device] = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)


def build_neural_solver(cfg: Dict[str, Any], *, seed: int, device: torch.device) -> DiscreteSBEED:
    seed_everything(seed, device)
    value_param = NeuralValueParam(
        module=StateMLPValueModule(
            n_states=N,
            hidden_sizes=(64, 64),
            activation=nn.Tanh,
        ),
    )
    rho_param = NeuralRhoParam(
        embed_module=StateActionMLPModule(
            n_states=N,
            n_actions=A,
            hidden_sizes=(64, 64),
            output_dim=1,
            activation=nn.Tanh,
        ),
        head_module=IdentityHead(),
    )
    policy_param = NeuralPolicyParam(
        module=StateMLPPolicyModule(
            n_states=N,
            n_actions=A,
            hidden_sizes=(64, 64),
            activation=nn.Tanh,
        ),
    )

    solver = DiscreteSBEED(
        n_states=N,
        n_actions=A,
        gamma=GAMMA,
        lambda_entropy=cfg["lambda_entropy"],
        eta=cfg["eta"],
        value_param=value_param,
        rho_param=rho_param,
        policy_param=policy_param,
        lr_value=cfg["lr_value"],
        lr_policy=cfg["lr_policy"],
        lr_rho=cfg["lr_rho"],
        tau=cfg["tau"],
        max_buffer_size=cfg["max_buffer_size"],
        batch_size=cfg["batch_size"],
        rollout_length=cfg["rollout_length"],
        fisher_damping=cfg["fisher_damping"],
        cg_iters=10,
        cg_tol=1e-8,
        seed=seed,
        device=device,
    )
    solver.loss_history = []
    return solver


def resolve_run_device(requested_device: str, run_id: int) -> torch.device:
    requested_device = str(requested_device).lower()
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device.startswith("cuda:"):
        return torch.device(requested_device)
    if requested_device not in {"auto", "cuda"}:
        return torch.device(requested_device)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n_gpus = torch.cuda.device_count()
    return torch.device(f"cuda:{run_id % max(n_gpus, 1)}")


def greedy_rollout(
    solver: DiscreteSBEED,
    transition_fn: Any,
    *,
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    pi = solver.get_policy_matrix()
    s = int(X0)
    path = [s]
    actions = []
    total_return = 0.0
    discount = 1.0

    torch_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        for _ in range(max_steps):
            if s in TERMINAL_STATES:
                break
            a = int(torch.argmax(pi[s]).item())
            sp = int(transition_fn(s, a))
            r = float(reward_fn(s, a, sp))
            actions.append(a)
            total_return += discount * r
            discount *= float(GAMMA)
            path.append(sp)
            s = sp
            if s in TERMINAL_STATES:
                break
    finally:
        torch.random.set_rng_state(torch_state)

    return {
        "greedy_path": path,
        "greedy_actions": actions,
        "greedy_return": float(total_return),
        "greedy_success": bool(path[-1] == GOAL_GRID),
    }


def notebook_replay_code(result: Dict[str, Any], *, transition_fn_name: str) -> str:
    device = result["device"]
    seed = int(result["seed"])
    return f'''# Replays the best config produced by grid_search_5grid_sbeed_neural_deterministic.py.
# Run this in a fresh notebook kernel after defining the 5x5 grid, `next_state`,
# `reward_fn`, and importing the SBEED neural classes.
import random
import numpy as np
import torch
from torch import nn

SEED = {seed}
DEVICE = torch.device("{device}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.set_device(DEVICE)
    torch.cuda.manual_seed_all(SEED)

value_param = NeuralValueParam(
    module=StateMLPValueModule(
        n_states=N,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
    ),
)

rho_param = NeuralRhoParam(
    embed_module=StateActionMLPModule(
        n_states=N,
        n_actions=A,
        hidden_sizes=(64, 64),
        output_dim=1,
        activation=nn.Tanh,
    ),
    head_module=IdentityHead(),
)

policy_param = NeuralPolicyParam(
    module=StateMLPPolicyModule(
        n_states=N,
        n_actions=A,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
    ),
)

solver_sbeed = DiscreteSBEED(
    n_states=N,
    n_actions=A,
    gamma=gamma,
    eta={float(result["eta"])!r},
    value_param=value_param,
    rho_param=rho_param,
    policy_param=policy_param,
    lr_value={float(result["lr_value"])!r},
    lr_rho={float(result["lr_rho"])!r},
    lr_policy={float(result["lr_policy"])!r},
    lambda_entropy={float(result["lambda_entropy"])!r},
    fisher_damping={float(result["fisher_damping"])!r},
    tau={float(result["tau"])!r},
    max_buffer_size={int(result["max_buffer_size"])},
    batch_size={int(result["batch_size"])},
    rollout_length={int(result["rollout_length"])},
    cg_iters=10,
    cg_tol=1e-8,
    seed=SEED,
    device=DEVICE,
)

pi_sbeed = solver_sbeed.run(
    transition_fn={transition_fn_name},
    reward_fn=reward_fn,
    episodes={int(result["episodes"])},
    collect_per_episode={int(result["collect_per_episode"])},
    updates_per_episode={int(result["updates_per_episode"])},
    initial_collect_steps={int(result["initial_collect_steps"])},
    start_state=x_0,
    epsilon={float(result["epsilon"])!r},
    terminal_states={{goal_grid, pit_grid}},
    log_every=50,
)
'''


def save_best_notebook_replay(
    best: Optional[Dict[str, Any]],
    output_dir: Path,
    *,
    experiment_name: str,
    transition_fn_name: str,
) -> Optional[Path]:
    if best is None or not best.get("ok"):
        return None
    path = output_dir / f"{experiment_name}_best_notebook.py"
    path.write_text(notebook_replay_code(best, transition_fn_name=transition_fn_name))

    metadata_path = output_dir / f"{experiment_name}_best_metadata.json"
    metadata = result_for_csv(best)
    metadata["notebook_replay_file"] = str(path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return path


def append_neural_result_csv(path: Path, r: Dict[str, Any]) -> None:
    row = result_for_csv(r)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NEURAL_CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_neural_config(
    cfg: Dict[str, Any],
    *,
    run_id: int,
    config_id: int,
    seed: int,
    requested_device: str,
    training_kwargs: Dict[str, Any],
    eval_every_episodes: int,
    n_eval_episodes_during: int,
    n_eval_episodes_final: int,
    max_steps_per_eval_episode: int,
    current_global_best_score: float,
    early_stop_after_episodes: int,
    early_stop_margin: Optional[float],
    torch_threads: int,
) -> Dict[str, Any]:
    torch.set_num_threads(torch_threads)
    device = resolve_run_device(requested_device, run_id)
    seed_everything(seed, device)

    start_time = time.time()
    episode = 0
    full_cfg = {
        **NEURAL_DEFAULTS,
        **cfg,
        **training_kwargs,
        **{k: cfg[k] for k in training_kwargs.keys() if k in cfg},
    }

    try:
        solver = build_neural_solver(full_cfg, seed=seed, device=device)

        initial_collect_steps = int(full_cfg["initial_collect_steps"])
        if initial_collect_steps > 0:
            solver.collect_steps(
                transition_fn=next_state_deterministic,
                n_steps=initial_collect_steps,
                start_state=X0,
                reward_fn=reward_fn,
                epsilon=full_cfg["epsilon"],
                terminal_states=TERMINAL_STATES,
            )

        eval_history = []
        stopped_early = False
        stop_reason = ""
        episodes = int(full_cfg["episodes"])

        for episode in range(1, episodes + 1):
            solver.collect_steps(
                transition_fn=next_state_deterministic,
                n_steps=full_cfg["collect_per_episode"],
                start_state=X0,
                reward_fn=reward_fn,
                epsilon=full_cfg["epsilon"],
                terminal_states=TERMINAL_STATES,
            )

            for _ in range(full_cfg["updates_per_episode"]):
                stats = solver.step()
                solver.loss_history.append(stats)

            if episode % eval_every_episodes == 0 or episode == episodes:
                eval_stats = evaluate_policy(
                    solver,
                    next_state_deterministic,
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
                    f"    run={run_id:03d} cfg={config_id:02d} "
                    f"device={device} eval ep={episode:04d} | "
                    f"return={eval_stats['eval_return_mean']:.5f} "
                    f"std={eval_stats['eval_return_std']:.5f} "
                    f"success={eval_stats['eval_success_rate']:.3f} "
                    f"score={temp_score:.5f}",
                    flush=True,
                )

        final_eval_stats = evaluate_policy(
            solver,
            next_state_deterministic,
            n_eval_episodes=n_eval_episodes_final,
            max_steps_per_episode=max_steps_per_eval_episode,
            seed=seed + 999_999,
        )
        policy_summary = extract_policy_summary(solver)
        greedy_stats = greedy_rollout(
            solver,
            next_state_deterministic,
            max_steps=max_steps_per_eval_episode,
            seed=seed + 888_888,
        )
        last_loss = solver.loss_history[-1] if solver.loss_history else {}
        objective_loss_history = [
            float(h["objective"])
            for h in solver.loss_history
            if isinstance(h.get("objective"), (float, int))
        ]
        main_objective_loss = float(last_loss.get("objective", float("nan")))

        result = {
            "ok": True,
            "run_id": run_id,
            "config_id": config_id,
            "seed": seed,
            "device": str(device),
            "requested_device": requested_device,
            "torch_threads": torch_threads,
            "transition_name": "deterministic",
            "network_init_seed": seed,
            "seconds": time.time() - start_time,
            "episodes_completed": episode,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            **full_cfg,
            **final_eval_stats,
            "score": None,
            "mean_best_prob": policy_summary["mean_best_prob"],
            "min_best_prob": policy_summary["min_best_prob"],
            "mean_entropy": policy_summary["mean_entropy"],
            "best_actions": policy_summary["best_actions"],
            "best_probs": policy_summary["best_probs"],
            **greedy_stats,
            "pi": policy_summary["pi"],
            "last_objective": main_objective_loss,
            "main_objective_loss": main_objective_loss,
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
            "objective_loss_history": objective_loss_history,
            "eval_history": eval_history,
            "solver": solver,
        }
        result["score"] = score_result(result)
        return result

    except Exception as exc:
        return {
            "ok": False,
            "run_id": run_id,
            "config_id": config_id,
            "seed": seed,
            "device": str(device),
            "requested_device": requested_device,
            "torch_threads": torch_threads,
            "transition_name": "deterministic",
            "network_init_seed": seed,
            "seconds": time.time() - start_time,
            "episodes_completed": episode,
            "stopped_early": True,
            "stop_reason": "exception",
            **full_cfg,
            "error": repr(exc),
            "eval_return_mean": -float("inf"),
            "eval_return_std": float("inf"),
            "score": -float("inf"),
            "solver": None,
            "pi": None,
        }


def make_jobs(configs: List[Dict[str, Any]]) -> List[Tuple[int, int, int, Dict[str, Any]]]:
    return [(config_id, config_id, SEED, cfg) for config_id, cfg in enumerate(configs)]


def completed_config_ids_from_csv(csv_path: Path) -> set[int]:
    completed: set[int] = set()
    if not csv_path.exists():
        return completed
    try:
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("ok") == "True":
                    completed.add(int(row["config_id"]))
    except Exception as exc:
        print(f"Warning: could not read {csv_path} for resume: {exc}", flush=True)
    return completed


def auto_workers() -> int:
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def auto_torch_threads(workers: int) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count // max(workers, 1))


def run_neural_grid_search(
    *,
    configs: List[Dict[str, Any]],
    training_kwargs: Dict[str, Any],
    output_dir: Path,
    eval_every_episodes: int,
    n_eval_episodes_during: int,
    n_eval_episodes_final: int,
    max_steps_per_eval_episode: int,
    early_stop_after_episodes: int,
    early_stop_margin: Optional[float],
    resume: bool,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sbeed_neural_deterministic_grid_search.csv"
    jobs = make_jobs(configs)
    completed_config_ids = completed_config_ids_from_csv(csv_path) if resume else set()
    jobs = [job for job in jobs if job[1] not in completed_config_ids]
    requested_device = REQUESTED_DEVICE
    workers = auto_workers()
    torch_threads = auto_torch_threads(workers)

    results = []
    best = None
    best_score = -float("inf")
    global_start = time.time()

    print("\n========== SBEED NEURAL FIXED GRID SEARCH ==========")
    print("Name: deterministic_neural")
    print("Problem: deterministic 5x5 grid")
    print(f"Configs: {len(configs)}")
    print(f"Seed: {SEED}")
    print(f"Submitted jobs: {len(jobs)}")
    print(f"Workers: {workers} (auto)")
    print(f"Torch threads per worker: {torch_threads} (auto)")
    print(f"Episodes per full run: {training_kwargs['episodes']}")
    print(f"Requested device: {requested_device} (auto)")
    if torch.cuda.is_available():
        print(f"Visible GPUs: {torch.cuda.device_count()}")
    print(f"Results dir: {output_dir}")
    if resume:
        print(f"Resume enabled. Found {len(completed_config_ids)} completed configs.")
    print("====================================================\n", flush=True)

    if workers > 1:
        executor_kwargs = {"max_workers": workers}
        if str(requested_device).lower() in {"auto", "cuda"} or str(requested_device).lower().startswith("cuda:"):
            executor_kwargs["mp_context"] = mp.get_context("spawn")

        with ProcessPoolExecutor(**executor_kwargs) as executor:
            future_to_run = {
                executor.submit(
                    train_one_neural_config,
                    cfg,
                    run_id=run_id,
                    config_id=config_id,
                    seed=seed,
                    requested_device=requested_device,
                    training_kwargs=training_kwargs,
                    eval_every_episodes=eval_every_episodes,
                    n_eval_episodes_during=n_eval_episodes_during,
                    n_eval_episodes_final=n_eval_episodes_final,
                    max_steps_per_eval_episode=max_steps_per_eval_episode,
                    current_global_best_score=best_score,
                    early_stop_after_episodes=early_stop_after_episodes,
                    early_stop_margin=early_stop_margin,
                    torch_threads=torch_threads,
                ): run_id
                for run_id, config_id, seed, cfg in jobs
            }
            for future in as_completed(future_to_run):
                result = future.result()
                results.append(result)
                append_neural_result_csv(csv_path, result)
                if result["ok"] and result["score"] > best_score:
                    best = result
                    best_score = result["score"]
                    replay_path = save_best_notebook_replay(
                        best,
                        output_dir,
                        experiment_name="sbeed_neural_deterministic",
                        transition_fn_name="next_state",
                    )
                    print(f"\n  >>> NEW BEST (Run {result['run_id']}) | score={best_score:.5f} <<<", flush=True)
                    print(f"  Notebook replay saved to: {replay_path}", flush=True)

                elapsed_hours = (time.time() - global_start) / 3600.0
                avg_minutes = (time.time() - global_start) / 60.0 / len(results)
                remaining = len(jobs) - len(results)
                eta_hours = (remaining * avg_minutes / 60.0) / workers if remaining > 0 else 0.0
                print(
                    f"Progress: {len(results)}/{len(jobs)} | "
                    f"Elapsed: {elapsed_hours:.2f}h | "
                    f"ETA: {eta_hours:.2f}h | "
                    f"Last return: {result['eval_return_mean']:.3f}",
                    flush=True,
                )
    else:
        for run_id, config_id, seed, cfg in jobs:
            result = train_one_neural_config(
                cfg,
                run_id=run_id,
                config_id=config_id,
                seed=seed,
                requested_device=requested_device,
                training_kwargs=training_kwargs,
                eval_every_episodes=eval_every_episodes,
                n_eval_episodes_during=n_eval_episodes_during,
                n_eval_episodes_final=n_eval_episodes_final,
                max_steps_per_eval_episode=max_steps_per_eval_episode,
                current_global_best_score=best_score,
                early_stop_after_episodes=early_stop_after_episodes,
                early_stop_margin=early_stop_margin,
                torch_threads=torch_threads,
            )
            results.append(result)
            append_neural_result_csv(csv_path, result)
            if result["ok"] and result["score"] > best_score:
                best = result
                best_score = result["score"]
                replay_path = save_best_notebook_replay(
                    best,
                    output_dir,
                    experiment_name="sbeed_neural_deterministic",
                    transition_fn_name="next_state",
                )
                print(f"\n  >>> NEW BEST (Run {result['run_id']}) | score={best_score:.5f} <<<", flush=True)
                print(f"  Notebook replay saved to: {replay_path}", flush=True)

            elapsed_hours = (time.time() - global_start) / 3600.0
            avg_minutes = (time.time() - global_start) / 60.0 / len(results)
            remaining = len(jobs) - len(results)
            eta_hours = remaining * avg_minutes / 60.0
            print(
                f"Progress: {len(results)}/{len(jobs)} | "
                f"Elapsed: {elapsed_hours:.2f}h | "
                f"ETA: {eta_hours:.2f}h | "
                f"Last return: {result['eval_return_mean']:.3f}",
                flush=True,
            )

    print("\n========== SEARCH DONE ==========")
    print(f"Total time: {(time.time() - global_start) / 3600.0:.2f} hours")
    print(f"CSV saved to: {csv_path}", flush=True)
    replay_path = save_best_notebook_replay(
        best,
        output_dir,
        experiment_name="sbeed_neural_deterministic",
        transition_fn_name="next_state",
    )
    if replay_path is not None:
        print(f"Best notebook replay saved to: {replay_path}", flush=True)
    return results, best


def clear_neural_outputs(output_dir: Path) -> None:
    path = output_dir / "sbeed_neural_deterministic_grid_search.csv"
    if path.exists():
        path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed SBEED neural configs on the deterministic 5x5 gridworld."
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/results/sbeed")
    parser.add_argument("--episodes", type=int, default=NEURAL_TRAINING_KWARGS["episodes"])
    parser.add_argument(
        "--collect-per-episode",
        type=int,
        default=NEURAL_TRAINING_KWARGS["collect_per_episode"],
    )
    parser.add_argument(
        "--updates-per-episode",
        type=int,
        default=NEURAL_TRAINING_KWARGS["updates_per_episode"],
    )
    parser.add_argument(
        "--initial-collect-steps",
        type=int,
        default=NEURAL_TRAINING_KWARGS["initial_collect_steps"],
    )
    parser.add_argument("--max-buffer-size", type=int, default=NEURAL_TRAINING_KWARGS["max_buffer_size"])
    parser.add_argument("--tau", type=float, default=NEURAL_TRAINING_KWARGS["tau"])
    parser.add_argument("--eval-every-episodes", type=int, default=50)
    parser.add_argument("--n-eval-episodes-during", type=int, default=80)
    parser.add_argument("--n-eval-episodes-final", type=int, default=300)
    parser.add_argument("--max-steps-per-eval-episode", type=int, default=100)
    parser.add_argument(
        "--early-stop-after-episodes",
        type=int,
        default=0,
        help="Ignored for this search; every config is run to completion.",
    )
    parser.add_argument(
        "--early-stop-margin",
        type=float,
        default=None,
        help="Ignored for this search; every config is run to completion.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already completed config ids.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous neural result CSV in output-dir before starting.",
    )
    return parser.parse_args()


def training_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "episodes": args.episodes,
        "collect_per_episode": args.collect_per_episode,
        "updates_per_episode": args.updates_per_episode,
        "initial_collect_steps": args.initial_collect_steps,
        "max_buffer_size": args.max_buffer_size,
        "tau": args.tau,
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    if args.overwrite:
        clear_neural_outputs(output_dir)

    results, best = run_neural_grid_search(
        configs=CONFIGS_TO_TRY,
        training_kwargs=training_kwargs_from_args(args),
        output_dir=output_dir,
        eval_every_episodes=args.eval_every_episodes,
        n_eval_episodes_during=args.n_eval_episodes_during,
        n_eval_episodes_final=args.n_eval_episodes_final,
        max_steps_per_eval_episode=args.max_steps_per_eval_episode,
        early_stop_after_episodes=args.early_stop_after_episodes,
        early_stop_margin=None,
        resume=args.resume,
    )
    summarize_top_results(results, top_k=10)
    print_best_result(best)


if __name__ == "__main__":
    main()
