"""Run the additional FOGAS coverage sweep on the stochastic 10 x 10 grid.

Scientific role
---------------
This supplementary experiment varies offline dataset collection for original
FOGAS under stochastic grid transitions. It extends the deterministic
partial-coverage protocol but is not one of the main thesis comparison tables.
FOGAS hyperparameters and the evaluation MDP remain fixed while data size,
exploration, policy mixture, and reset conditions change.

Inputs and outputs
------------------
Candidate datasets are created as temporary files under ``/tmp`` and removed
after evaluation. Only the checkpointed search and best-row CSVs are retained
under ``data/results/generalization/hyperparam_grids/10grid``.

Run this file directly from the repository root. Use ``--max-datasets`` for a
smoke test, ``--resume`` for the additional full sweep, and the worker/device
options described by ``--help``. The parent process serializes CSV writes.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import multiprocessing
import os
import random
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - only for minimal environments
    tqdm = None


def find_root(current_path):
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / "src" / "rl_methods").exists() and (parent / "data").exists():
            return parent
    return current_path


PROJECT_ROOT = find_root(Path(__file__).resolve())
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "generalization" / "hyperparam_grids" / "10grid"
OUTPUT_CSV = RESULTS_DIR / "fogas_10grid_stochastic_dataset_generation_search.csv"
BEST_CSV = RESULTS_DIR / "fogas_10grid_stochastic_dataset_generation_search_best.csv"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
torch.set_default_dtype(torch.float64)

from rl_methods.fogas import FOGASEvaluator, FOGASDataset, FOGASSolver  # noqa: E402
from rl_methods.mdp import DiscreteMDP, Planner  # noqa: E402


# ---------------------------------------------------------------------
# Fixed MDP and run settings.
# ---------------------------------------------------------------------
SEED = 42
NUM_TRAJECTORIES = 100
EVAL_MAX_STEPS = (50, 100, 200)
MAX_STEPS = 50
EXTRA_TERMINAL_STEPS = 3
COVERAGE_BETA = 1e-7

N = 100
A = 4
GAMMA = 0.9
X0 = 0
GRID_SIZE = 10
GOAL_GRID = 99
PIT_GRIDS = {18, 32, 57, 61, 75}
WALL_STATES = {
    4,
    11,
    14,
    17,
    21,
    22,
    27,
    34,
    37,
    40,
    42,
    43,
    44,
    45,
    46,
    47,
    49,
    54,
    62,
    64,
    66,
    72,
    76,
    82,
    84,
    86,
    87,
    94,
}
TERMINAL_STATES = {GOAL_GRID, *PIT_GRIDS}
INTENDED_PROB = 0.8


# ---------------------------------------------------------------------
# Fixed classic FOGAS hyperparameters.
# ---------------------------------------------------------------------
FOGAS_ALPHA = 0.005
FOGAS_ETA = 0.002
FOGAS_RHO = 0.0005
FOGAS_D_THETA = 60.0
FOGAS_T = 20_000
FOGAS_BETA = 1e-7


# ---------------------------------------------------------------------
# Dataset-generation grid.
# ---------------------------------------------------------------------
DATASET_SIZES = [1000, 2000, 4000, 8000, 12000]
EPSILON_VALUES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
PROPORTION_CONFIGS = {
    "100_0": ([1.0, 0.0], "100% optimal-eps / 0% random"),
    "95_5": ([0.95, 0.05], "95% optimal-eps / 5% random"),
    "90_10": ([0.9, 0.1], "90% optimal-eps / 10% random"),
    "80_20": ([0.8, 0.2], "80% optimal-eps / 20% random"),
    "70_30": ([0.7, 0.3], "70% optimal-eps / 30% random"),
    "60_40": ([0.6, 0.4], "60% optimal-eps / 40% random"),
    "50_50": ([0.5, 0.5], "50% optimal-eps / 50% random"),
    "20_80": ([0.2, 0.8], "20% optimal-eps / 80% random"),
    "0_100": ([0.0, 1.0], "0% optimal-eps / 100% random"),
}
RESET_CONFIGS = {
    "x0": {"reset_probs": {"x0": 1.0}},
    "uniform_valid": {"reset_probs": {"random": 1.0}},
    "x0_80_uniform_20": {"reset_probs": {"x0": 0.8, "random": 0.2}},
    "x0_50_uniform_50": {"reset_probs": {"x0": 0.5, "random": 0.5}},
    "x0_20_uniform_80": {"reset_probs": {"x0": 0.2, "random": 0.8}},
    "occupancy": {"reset_probs": {"occupancy": 1.0}},
    "occupancy_uniform": {"reset_probs": {"occupancy_uniform": 1.0}},
    "x0_50_occupancy_50": {"reset_probs": {"x0": 0.5, "occupancy": 0.5}},
    "x0_50_occupancy_uniform_50": {"reset_probs": {"x0": 0.5, "occupancy_uniform": 0.5}},
    "uniform_50_occupancy_50": {"reset_probs": {"random": 0.5, "occupancy": 0.5}},
    "uniform_50_occupancy_uniform_50": {"reset_probs": {"random": 0.5, "occupancy_uniform": 0.5}},
}


FIELDNAMES = [
    "run_idx",
    "status",
    "error",
    "elapsed_seconds",
    "device",
    "torch_threads",
    "seed",
    "dataset_size",
    "epsilon",
    "proportions",
    "proportion_key",
    "reset_mode",
    "reset_support_size",
    "extra_terminal_steps",
    "collection_max_steps",
    "intended_prob",
    "fogas_T",
    "fogas_alpha",
    "fogas_eta",
    "fogas_rho",
    "fogas_D_theta",
    "fogas_beta",
    "coverage_beta",
    "feature_coverage",
    "unique_states",
    "unique_state_actions",
    "state_coverage_ratio",
    "state_action_coverage_ratio",
    "goal_transitions",
    "pit_transitions",
    "terminal_transitions",
    "missing_nonwall_nonterminal_states",
    "observed_goal_reachable",
    "observed_goal_shortest_steps",
    "reward_mean",
    "reward_min",
    "reward_max",
    "planner_policy_return",
    "planner_v_x0",
    "planner_success_rate_50",
    "planner_success_rate_100",
    "planner_success_rate_200",
    "planner_avg_return_50",
    "planner_avg_return_100",
    "planner_avg_return_200",
    "greedy_success_rate_50",
    "greedy_success_rate_100",
    "greedy_success_rate_200",
    "greedy_avg_return_50",
    "greedy_policy_return",
    "greedy_v_x0",
    "greedy_v_gap",
    "greedy_on_data_quality",
    "solver_success_rate_50",
    "solver_success_rate_100",
    "solver_success_rate_200",
    "solver_avg_return_50",
    "solver_policy_return",
    "solver_v_x0",
    "solver_v_gap",
    "solver_on_data_quality",
    "final_lambda_norm",
    "final_theta_bar_norm",
]


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


def move_deterministic(s, a):
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


def transition_probs(s, a, intended_prob=INTENDED_PROB):
    probs_by_state = {}

    for candidate_a in range(A):
        prob = (1.0 - float(intended_prob)) / A
        if candidate_a == int(a):
            prob += float(intended_prob)

        sp = move_deterministic(s, candidate_a)
        probs_by_state[sp] = probs_by_state.get(sp, 0.0) + prob

    return probs_by_state


def reward_from_next_state(sp):
    sp = int(sp)
    if sp == GOAL_GRID:
        return 1.0
    if sp in PIT_GRIDS:
        return -5.0
    return -0.1


def reward_fn(s, a):
    if int(s) in TERMINAL_STATES:
        return 0.0
    return sum(
        prob * reward_from_next_state(sp)
        for sp, prob in transition_probs(s, a, INTENDED_PROB).items()
    )


def build_mdp():
    states = torch.arange(N, dtype=torch.long)
    actions = torch.arange(A, dtype=torch.long)

    def phi(s, a):
        feat = torch.zeros(N * A, dtype=torch.float64)
        feat[int(s) * A + int(a)] = 1.0
        return feat

    def transition_fn(s, a):
        probs = torch.zeros(N, dtype=torch.float64)
        for sp, prob in transition_probs(s, a, INTENDED_PROB).items():
            probs[sp] = prob
        return probs

    omega = torch.empty(N * A, dtype=torch.float64)
    for state in range(N):
        for action in range(A):
            omega[state * A + action] = reward_fn(state, action)

    mdp = DiscreteMDP(
        states=states,
        actions=actions,
        gamma=GAMMA,
        x0=X0,
        omega=omega,
        phi=phi,
        transition_fn=transition_fn,
        terminal_states=list(TERMINAL_STATES),
    )
    return mdp, phi


def dataset_candidates():
    candidates = []
    run_idx = 0
    for (reset_name, reset_cfg), (prop_key, (proportions, prop_label)), eps, n_steps in itertools.product(
        RESET_CONFIGS.items(),
        PROPORTION_CONFIGS.items(),
        EPSILON_VALUES,
        DATASET_SIZES,
    ):
        # If the behavior is 100% random, epsilon no longer changes the policy.
        if prop_key == "0_100" and float(eps) != 0.0:
            continue
        run_idx += 1
        candidates.append(
            {
                "run_idx": run_idx,
                "reset_name": reset_name,
                "reset_cfg": reset_cfg,
                "proportion_key": prop_key,
                "proportions": proportions,
                "proportion_label": prop_label,
                "epsilon": float(eps),
                "dataset_size": int(n_steps),
            }
        )
    return candidates


def valid_reset_states():
    forbidden = TERMINAL_STATES | WALL_STATES
    return np.asarray([state for state in range(N) if state not in forbidden], dtype=np.int64)


def reset_distribution(reset_cfg, state_occupancy):
    valid_states = valid_reset_states()
    combined = np.zeros(N, dtype=np.float64)
    reset_probs = reset_cfg["reset_probs"]
    total_weight = float(sum(reset_probs.values()))
    if total_weight <= 0.0:
        raise ValueError("reset probabilities must have positive mass")

    for mode, raw_weight in reset_probs.items():
        weight = float(raw_weight) / total_weight
        probs = np.zeros(N, dtype=np.float64)

        if mode == "x0":
            probs[X0] = 1.0
        elif mode == "random":
            probs[valid_states] = 1.0 / len(valid_states)
        elif mode == "occupancy":
            probs = np.asarray(state_occupancy, dtype=np.float64).copy()
            probs[list(TERMINAL_STATES | WALL_STATES)] = 0.0
            mass = probs.sum()
            if mass <= 1e-12:
                raise ValueError("occupancy reset has no valid positive mass")
            probs /= mass
        elif mode == "occupancy_uniform":
            occ = np.asarray(state_occupancy, dtype=np.float64).copy()
            occ[list(TERMINAL_STATES | WALL_STATES)] = 0.0
            support = np.where(occ > 1e-12)[0]
            if support.size == 0:
                raise ValueError("occupancy_uniform reset has no valid occupied states")
            probs[support] = 1.0 / support.size
        else:
            raise ValueError(f"Unsupported reset mode: {mode}")

        combined += weight * probs

    combined_sum = combined.sum()
    if combined_sum <= 1e-12:
        raise ValueError("combined reset distribution has no mass")
    return combined / combined_sum


def sample_matrix_policy_action(rng, policy_matrix, state):
    probs = np.asarray(policy_matrix[int(state)], dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    mass = probs.sum()
    if mass <= 1e-12:
        probs = np.ones(A, dtype=np.float64) / A
    else:
        probs /= mass
    return int(rng.choice(A, p=probs))


def select_policy_index(rng, proportions):
    probs = np.asarray(proportions, dtype=np.float64)
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def collect_dataset_csv(mdp, pi_star, state_occupancy, candidate, save_path):
    rng = np.random.default_rng(SEED)
    p = mdp.P.detach().cpu().numpy()
    r = mdp.r.detach().cpu().numpy().reshape(-1)
    pi_star = pi_star.detach().cpu().numpy()
    reset_probs = reset_distribution(candidate["reset_cfg"], state_occupancy)

    rows = []
    state = int(rng.choice(N, p=reset_probs))
    episode_policy_idx = select_policy_index(rng, candidate["proportions"])
    step = 0
    terminal_extra_remaining = None

    for _ in range(int(candidate["dataset_size"])):
        if episode_policy_idx == 0:
            if rng.random() < float(candidate["epsilon"]):
                action = int(rng.integers(A))
            else:
                action = sample_matrix_policy_action(rng, pi_star, state)
        else:
            action = int(rng.integers(A))

        row_idx = int(state) * A + int(action)
        if int(state) in TERMINAL_STATES:
            next_state = int(state)
            reward = float(r[row_idx])
            terminated = True
            truncated = False
        else:
            next_state = int(rng.choice(N, p=p[row_idx]))
            reward = float(r[row_idx])
            terminated = next_state in TERMINAL_STATES
            truncated = step + 1 >= MAX_STEPS

        rows.append(
            {
                "state": int(state),
                "action": int(action),
                "reward": reward,
                "next_state": int(next_state),
            }
        )

        step += 1
        if truncated:
            state = int(rng.choice(N, p=reset_probs))
            step = 0
            terminal_extra_remaining = None
            episode_policy_idx = select_policy_index(rng, candidate["proportions"])
        elif terminated:
            if terminal_extra_remaining is None:
                terminal_extra_remaining = EXTRA_TERMINAL_STEPS
            else:
                terminal_extra_remaining -= 1

            if terminal_extra_remaining > 0:
                state = next_state
            else:
                state = int(rng.choice(N, p=reset_probs))
                step = 0
                terminal_extra_remaining = None
                episode_policy_idx = select_policy_index(rng, candidate["proportions"])
        else:
            state = next_state

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["state", "action", "reward", "next_state"])
        writer.writeheader()
        writer.writerows(rows)

    return rows, int(np.count_nonzero(reset_probs > 1e-12))


def feature_coverage(rows, optimal_occupancy, beta):
    counts = np.zeros(N * A, dtype=np.float64)
    for row in rows:
        counts[int(row["state"]) * A + int(row["action"])] += 1.0

    occ = optimal_occupancy.detach().cpu().numpy().reshape(N * A).astype(np.float64)
    denom = float(beta) + counts / max(1, len(rows))
    return float(np.sum((occ * occ) / denom))


def observed_goal_shortest_steps(rows):
    graph = {state: set() for state in range(N)}
    for row in rows:
        graph[int(row["state"])].add(int(row["next_state"]))

    queue = deque([(X0, 0)])
    seen = {X0}
    while queue:
        state, dist = queue.popleft()
        if state == GOAL_GRID:
            return dist
        if state in TERMINAL_STATES and state != GOAL_GRID:
            continue
        for next_state in graph.get(state, ()):
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, dist + 1))
    return math.nan


def dataset_diagnostics(rows):
    states = [int(row["state"]) for row in rows]
    actions = [int(row["action"]) for row in rows]
    next_states = [int(row["next_state"]) for row in rows]
    rewards = [float(row["reward"]) for row in rows]
    pairs = {(state, action) for state, action in zip(states, actions)}
    nonwall_nonterminal = set(range(N)) - WALL_STATES - TERMINAL_STATES
    shortest = observed_goal_shortest_steps(rows)

    return {
        "unique_states": len(set(states)),
        "unique_state_actions": len(pairs),
        "state_coverage_ratio": len(set(states)) / N,
        "state_action_coverage_ratio": len(pairs) / (N * A),
        "goal_transitions": sum(1 for state in next_states if state == GOAL_GRID),
        "pit_transitions": sum(1 for state in next_states if state in PIT_GRIDS),
        "terminal_transitions": sum(1 for state in next_states if state in TERMINAL_STATES),
        "missing_nonwall_nonterminal_states": len(nonwall_nonterminal - set(states)),
        "observed_goal_reachable": bool(math.isfinite(shortest)),
        "observed_goal_shortest_steps": shortest,
        "reward_mean": float(np.mean(rewards)) if rewards else math.nan,
        "reward_min": float(np.min(rewards)) if rewards else math.nan,
        "reward_max": float(np.max(rewards)) if rewards else math.nan,
    }


def simulate_policy_metrics(policy, p, r, max_steps, seed):
    successes = 0
    returns = []
    policy = np.asarray(policy, dtype=np.float64)
    for idx in range(NUM_TRAJECTORIES):
        rng = np.random.default_rng(int(seed) + idx)
        state = X0
        discounted_return = 0.0
        for step in range(int(max_steps)):
            action = sample_matrix_policy_action(rng, policy, state)
            row_idx = int(state) * A + int(action)
            reward = float(r[row_idx])
            next_state = int(rng.choice(N, p=p[row_idx]))
            discounted_return += (GAMMA ** step) * reward
            if next_state in TERMINAL_STATES:
                successes += int(next_state == GOAL_GRID)
                break
            state = next_state
        returns.append(discounted_return)
    return {
        "success_rate": float(successes / NUM_TRAJECTORIES),
        "avg_return": float(np.mean(returns)) if returns else 0.0,
    }


def planner_baseline_metrics(mdp, planner):
    p = mdp.P.detach().cpu().numpy()
    r = mdp.r.detach().cpu().numpy().reshape(-1)
    pi_star = planner.pi_star.detach().cpu().numpy()
    out = {
        "planner_policy_return": float(planner.optimal_policy_return()),
        "planner_v_x0": float(planner.v_star[X0].detach().cpu().item()),
    }
    for max_steps in EVAL_MAX_STEPS:
        metrics = simulate_policy_metrics(pi_star, p, r, max_steps, SEED)
        out[f"planner_success_rate_{max_steps}"] = metrics["success_rate"]
        out[f"planner_avg_return_{max_steps}"] = metrics["avg_return"]
    return out


def evaluate_policy_family(row, evaluator, dataset, policy_mode, d_star, v_star):
    pi = evaluator.get_policy(policy_mode)
    v_pi, _ = evaluator.planner.evaluate_policy(pi)
    v_gap = float((d_star * (v_star - v_pi.detach().cpu())).sum().item())

    row[f"{policy_mode}_avg_return_50"] = float(
        evaluator.average_return(
            policy_mode=policy_mode,
            num_trajectories=NUM_TRAJECTORIES,
            max_steps=50,
            seed=SEED,
            terminal_states=TERMINAL_STATES,
        )["policy"]
    )
    for max_steps in EVAL_MAX_STEPS:
        row[f"{policy_mode}_success_rate_{max_steps}"] = float(
            evaluator.success_rate(
                goal_state=GOAL_GRID,
                policy_mode=policy_mode,
                num_trajectories=NUM_TRAJECTORIES,
                max_steps=max_steps,
                seed=SEED,
                terminal_states=TERMINAL_STATES,
            )["policy"]
        )

    row[f"{policy_mode}_policy_return"] = float(evaluator.planner.policy_return(pi))
    row[f"{policy_mode}_v_x0"] = float(v_pi[evaluator.mdp.x0].detach().cpu().item())
    row[f"{policy_mode}_v_gap"] = v_gap
    row[f"{policy_mode}_on_data_quality"] = float(
        evaluator.on_data_quality(
            dataset=dataset,
            policy_mode=policy_mode,
            compare_with_optimal=True,
        )["policy"]
    )


def finite_float(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def blank_metrics():
    return {field: math.nan for field in FIELDNAMES}


def base_row(candidate, device, torch_threads, fogas_T, status="ok", error=""):
    row = blank_metrics()
    row.update(
        {
            "run_idx": int(candidate["run_idx"]),
            "status": status,
            "error": error,
            "device": str(device),
            "torch_threads": int(torch_threads),
            "seed": SEED,
            "dataset_size": int(candidate["dataset_size"]),
            "epsilon": float(candidate["epsilon"]),
            "proportions": candidate["proportion_label"],
            "proportion_key": candidate["proportion_key"],
            "reset_mode": candidate["reset_name"],
            "extra_terminal_steps": EXTRA_TERMINAL_STEPS,
            "collection_max_steps": MAX_STEPS,
            "intended_prob": INTENDED_PROB,
            "fogas_T": int(fogas_T),
            "fogas_alpha": FOGAS_ALPHA,
            "fogas_eta": FOGAS_ETA,
            "fogas_rho": FOGAS_RHO,
            "fogas_D_theta": FOGAS_D_THETA,
            "fogas_beta": FOGAS_BETA,
            "coverage_beta": COVERAGE_BETA,
        }
    )
    return row


def run_dataset_worker(payload):
    candidate, device_str, torch_threads, fogas_T = payload
    configure_worker_threads(torch_threads)
    set_seed(SEED)
    device = torch.device(device_str)

    start = time.perf_counter()
    row = base_row(candidate, device, torch_threads, fogas_T)
    try:
        mdp, phi = build_mdp()
        planner = Planner(mdp)
        d_star = planner.state_mu_star.detach().cpu()
        v_star = planner.v_star.detach().cpu()
        state_occupancy = d_star.numpy()
        optimal_occupancy = planner.mu_star.detach().cpu()
        row.update(planner_baseline_metrics(mdp, planner))

        with tempfile.TemporaryDirectory(prefix="fogas_stoch_10grid_", dir="/tmp") as tmp:
            dataset_path = Path(tmp) / f"dataset_{candidate['run_idx']}.csv"
            rows, reset_support_size = collect_dataset_csv(
                mdp=mdp,
                pi_star=planner.pi_star,
                state_occupancy=state_occupancy,
                candidate=candidate,
                save_path=dataset_path,
            )
            row["reset_support_size"] = reset_support_size
            row["feature_coverage"] = feature_coverage(rows, optimal_occupancy, COVERAGE_BETA)
            row.update(dataset_diagnostics(rows))
            dataset = FOGASDataset(dataset_path)

            mdp.to(device)
            planner.to(device)
            solver = FOGASSolver(
                mdp=mdp,
                phi=phi,
                csv_path=str(dataset_path),
                device=device,
                seed=SEED,
                beta=FOGAS_BETA,
                print_params=False,
                dataset_verbose=False,
            )
            pi = solver.run(
                T=int(fogas_T),
                alpha=FOGAS_ALPHA,
                eta=FOGAS_ETA,
                rho=FOGAS_RHO,
                D_theta=FOGAS_D_THETA,
                tqdm_print=False,
                verbose=False,
            )
            solver.pi = pi.to(dtype=torch.float64, device=device)
            evaluator = FOGASEvaluator(solver=solver, mdp=mdp, planner=planner)

            evaluate_policy_family(row, evaluator, dataset, "greedy", d_star, v_star)
            evaluate_policy_family(row, evaluator, dataset, "solver", d_star, v_star)

            if solver.lambda_T is not None:
                row["final_lambda_norm"] = finite_float(torch.linalg.norm(solver.lambda_T).detach().cpu().item())
            if solver.theta_bar_history:
                row["final_theta_bar_norm"] = finite_float(
                    torch.linalg.norm(solver.theta_bar_history[-1]).detach().cpu().item()
                )
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        row["elapsed_seconds"] = float(time.perf_counter() - start)
    return row


def candidate_key(row):
    reset_value = row["reset_mode"] if "reset_mode" in row else row["reset_name"]
    return (
        int(row["dataset_size"]),
        float(row["epsilon"]),
        str(row["proportion_key"]),
        str(reset_value),
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


def as_float(row, key, default=math.nan):
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def sort_key(row):
    ok = str(row.get("status", "")).lower() == "ok"
    return (
        int(ok),
        as_float(row, "greedy_success_rate_50", -math.inf),
        as_float(row, "solver_success_rate_50", -math.inf),
        as_float(row, "greedy_success_rate_100", -math.inf),
        as_float(row, "solver_success_rate_100", -math.inf),
        as_float(row, "greedy_success_rate_200", -math.inf),
        as_float(row, "solver_success_rate_200", -math.inf),
        as_float(row, "greedy_avg_return_50", -math.inf),
        as_float(row, "solver_avg_return_50", -math.inf),
        -as_float(row, "greedy_v_gap", math.inf),
        -as_float(row, "solver_v_gap", math.inf),
        -as_float(row, "feature_coverage", math.inf),
        -as_float(row, "elapsed_seconds", math.inf),
    )


def refresh_best_csv(output_csv, best_csv):
    if not output_csv.exists():
        return
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=sort_key, reverse=True)
    best_csv.parent.mkdir(parents=True, exist_ok=True)
    with best_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_devices(value):
    devices = [item.strip() for item in str(value).split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("At least one device is required.")
    return devices


def parse_args():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Run the stochastic 10-grid FOGAS dataset-generation search."
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
        default=parse_devices(default_device),
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
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="Result CSV path.",
    )
    parser.add_argument(
        "--best-csv",
        type=Path,
        default=None,
        help="Best-first sorted result CSV path. Defaults to '<output>_best.csv'.",
    )
    parser.add_argument(
        "--fogas-T",
        dest="fogas_T",
        type=int,
        default=FOGAS_T,
        help="FOGAS iterations per dataset.",
    )
    return parser.parse_args()


def progress_iter(iterable, total, desc, disable):
    if tqdm is None or disable:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="dataset")


def best_csv_for_output(output_csv, explicit_best_csv):
    if explicit_best_csv is not None:
        return explicit_best_csv
    if output_csv == OUTPUT_CSV:
        return BEST_CSV
    return output_csv.with_name(f"{output_csv.stem}_best{output_csv.suffix}")


def build_tasks(candidates, output_csv, resume, devices, torch_threads, fogas_T):
    completed = successful_completed_keys(output_csv) if resume else set()
    tasks = []
    for candidate in candidates:
        if candidate_key(candidate) in completed:
            continue
        device = devices[len(tasks) % len(devices)]
        tasks.append((candidate, device, torch_threads, fogas_T))
    return tasks, completed


def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()
    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    output_csv = Path(args.output_csv)
    best_csv = best_csv_for_output(output_csv, args.best_csv)

    candidates = dataset_candidates()
    if args.max_datasets is not None:
        candidates = candidates[: max(0, int(args.max_datasets))]

    if not args.resume:
        for path in (output_csv, best_csv):
            if path.exists():
                path.unlink()

    tasks, completed = build_tasks(
        candidates=candidates,
        output_csv=output_csv,
        resume=args.resume,
        devices=args.devices,
        torch_threads=torch_threads,
        fogas_T=args.fogas_T,
    )

    print("Stochastic 10-grid FOGAS dataset-generation search")
    print(f"Output CSV       : {output_csv.resolve()}")
    print(f"Best CSV         : {best_csv.resolve()}")
    print(f"Dataset variants : {len(candidates)}")
    print(f"Tasks to run     : {len(tasks)}")
    print(f"Resume completed : {len(completed)}")
    print(f"Workers          : {workers}")
    print(f"Devices          : {', '.join(args.devices)}")
    print(f"Torch threads    : {torch_threads}")
    print(f"FOGAS T          : {args.fogas_T}")

    if not tasks:
        refresh_best_csv(output_csv, best_csv)
        print("No tasks to run.")
        return

    completed_count = 0
    if workers == 1:
        iterator = progress_iter(tasks, len(tasks), "FOGAS dataset grid", args.no_progress)
        for task in iterator:
            row = run_dataset_worker(task)
            append_row(output_csv, row)
            refresh_best_csv(output_csv, best_csv)
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
                "FOGAS dataset grid",
                args.no_progress,
            )
            for future in iterator:
                task = future_to_task[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = base_row(task[0], task[1], task[2], task[3], status="failed", error=repr(exc))
                    row["elapsed_seconds"] = math.nan
                append_row(output_csv, row)
                refresh_best_csv(output_csv, best_csv)
                completed_count += 1
                if args.no_progress:
                    print(
                        f"[{completed_count}/{len(tasks)}] "
                        f"run_idx={task[0]['run_idx']} status={row.get('status')}"
                    )

    print("\nDataset-generation search complete.")
    print(f"Output CSV: {output_csv}")
    print(f"Best CSV: {best_csv}")


if __name__ == "__main__":
    main()
