"""
Dataset grid search for SBEED and generalized FOGAS on the clean 10x10 tabular grid.

The dataset grid mirrors:
    experiments/fogas_clean/scripts/grid_search_10grid_fqi.py
    experiments/fogas_clean/scripts/grid_search_10grid_fogas.py

One deterministic 10-grid dataset variant is collected per worker task. The
worker computes feature coverage once, then runs the requested fixed-parameter
algorithms. The parent process owns CSV writes so parallel workers never write
to the same file concurrently.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import random
import sys
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


def find_root(current_path, marker="setup.py"):
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    return current_path


PROJECT_ROOT = find_root(Path(__file__).resolve())
RESULTS_DIR = PROJECT_ROOT / "data" / "results_clean" / "10grid_tabular"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
torch.set_default_dtype(torch.float64)

from rl_methods.fogas_clean import FOGASEvaluator, FOGASDataset  # noqa: E402
from rl_methods.fogas_generalization_clean import (  # noqa: E402
    FinalLinearSolver,
    LinearFunction,
    LinearQFunction,
    TabularFeatures,
)
from rl_methods.mdp_clean import DiscreteMDP, Planner  # noqa: E402
from rl_methods.sbeed import (  # noqa: E402
    DiscreteSBEED,
    DiscreteSBEEDDataset,
    LinearRhoParam,
    LinearValueParam,
    SoftmaxLinearPolicyParam,
    TabularStateActionFeatures,
    TabularStateFeatures,
)


# ---------------------------------------------------------------------
# Fixed run settings copied from the existing 10-grid dataset searches.
# ---------------------------------------------------------------------
SEED = 42
NUM_TRAJECTORIES = 100
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


# ---------------------------------------------------------------------
# Fixed SBEED hyperparameters from 10grid_comparation.ipynb.
# ---------------------------------------------------------------------
SBEED_STEPS = 3_000
SBEED_LAMBDA_ENTROPY = 1e-4
SBEED_ETA = 1.0
SBEED_LR_VALUE = 3e-3
SBEED_LR_RHO = 1e-3
SBEED_LR_POLICY = 3e-3
SBEED_TAU = 500.0
SBEED_ROLLOUT_LENGTH = 1
SBEED_BATCH_SIZE = None


# ---------------------------------------------------------------------
# Fixed generalized FOGAS hyperparameters from the 10grid_tabular_new
# hyperparameter search best row.
# ---------------------------------------------------------------------
GEN_ALPHA = 0.001
GEN_ETA = 3e-05
GEN_RHO = 1.0
GEN_T = 10_000
GEN_THETA_LR = 0.01
GEN_THETA_INNER_STEPS = 10
GEN_THETA_LAMBDA = 1e-8
GEN_THETA_MODE = "reg_fixed"
GEN_THETA_OPTIMIZER = "adam"
GEN_THETA_START_MODE = "warm"
GEN_THETA_INCLUDE_BETA_COV = False
GEN_BETA_UPDATE = "fogas_full"
GEN_POLICY_OPTIMIZER = "adam"
GEN_POLICY_GRADIENT = "exact"
GEN_REINFORCE_SAMPLES = 4
GEN_STATE_WEIGHT_UPDATE = "normal"


# ---------------------------------------------------------------------
# Dataset grid copied from FQI/FOGAS dataset searches.
# ---------------------------------------------------------------------
DATASET_SIZES = [4000, 8000, 12000, 16000, 20000]
EPSILON_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7]
PROPORTION_CONFIGS = {
    "100_0": ([1.0, 0.0], "100% optimal-eps / 0% random"),
    "80_20": ([0.8, 0.2], "80% optimal-eps / 20% random"),
    "60_40": ([0.6, 0.4], "60% optimal-eps / 40% random"),
    "40_60": ([0.4, 0.6], "40% optimal-eps / 60% random"),
    "20_80": ([0.2, 0.8], "20% optimal-eps / 80% random"),
    "0_100": ([0.0, 1.0], "0% optimal-eps / 100% random"),
}
RESET_CONFIGS = {
    "custom_x0": {"reset_probs": {"custom": 1.0}, "initial_states": [0]},
    "x0": {"reset_probs": {"x0": 1.0}, "initial_states": None},
    "x0_80_random_20": {"reset_probs": {"x0": 0.8, "random": 0.2}, "initial_states": None},
    "x0_50_custom_50": {"reset_probs": {"x0": 0.5, "custom": 0.5}, "initial_states": [0]},
    "x0_20_random_80": {"reset_probs": {"x0": 0.2, "random": 0.8}, "initial_states": None},
    "random": {"reset_probs": {"random": 1.0}, "initial_states": None},
}

ALGORITHM_ALIASES = {
    "sbeed": "sbeed",
    "generalized_fogas": "generalized_fogas",
    "generalized-fogas": "generalized_fogas",
    "gen_fogas": "generalized_fogas",
    "gen-fogas": "generalized_fogas",
}

OUTPUT_FILES = {
    "sbeed": "sbeed_dataset_grid.csv",
    "generalized_fogas": "generalized_fogas_dataset_grid_10grid_tabular_new_best_hparams.csv",
}

BASE_COLUMNS = [
    "run_idx",
    "algorithm",
    "dataset_size",
    "epsilon",
    "proportions",
    "proportion_key",
    "reset_mode",
    "extra_terminal_steps",
    "seed",
    "status",
    "error",
    "feature_coverage",
    "greedy_on_data_quality",
    "greedy_optimal_states_quality",
    "greedy_avg_reward",
    "greedy_success_rate",
    "greedy_v_x0",
    "greedy_v_gap",
    "solver_on_data_quality",
    "solver_optimal_states_quality",
    "solver_avg_reward",
    "solver_success_rate",
    "solver_v_x0",
    "solver_v_gap",
    "elapsed_seconds",
]

SBEED_EXTRA_COLUMNS = [
    "steps",
    "lambda_entropy",
    "eta",
    "lr_value",
    "lr_rho",
    "lr_policy",
    "tau",
    "rollout_length",
    "batch_size",
    "device",
    "torch_threads",
    "last_objective",
    "last_primal_mse",
    "last_dual_mse",
    "last_theta_grad_norm",
    "last_beta_grad_norm",
    "last_policy_grad_norm",
]

GEN_EXTRA_COLUMNS = [
    "alpha",
    "eta",
    "rho",
    "T",
    "theta_lr",
    "theta_inner_steps",
    "theta_lambda",
    "theta_mode",
    "theta_optimizer",
    "theta_start_mode",
    "theta_include_beta_cov",
    "beta_update",
    "policy_optimizer",
    "policy_gradient",
    "reinforce_samples",
    "state_weight_update",
    "device",
    "torch_threads",
    "final_total_loss",
    "final_policy_objective",
    "final_beta_objective",
    "final_q_objective",
    "final_theta_norm",
    "final_policy_grad_norm",
    "final_beta_grad_norm",
    "final_theta_grad_norm",
]

FIELDNAMES = {
    "sbeed": BASE_COLUMNS + SBEED_EXTRA_COLUMNS,
    "generalized_fogas": BASE_COLUMNS + GEN_EXTRA_COLUMNS,
}


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


def next_state_deterministic(s, a):
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
    return s if sp in WALL_STATES else sp


def build_mdp():
    states = torch.arange(N, dtype=torch.int64)
    actions = torch.arange(A, dtype=torch.int64)

    def phi(x, a):
        vec = torch.zeros(N * A, dtype=torch.float64)
        vec[int(x) * A + int(a)] = 1.0
        return vec

    omega = torch.empty(N * A, dtype=torch.float64)
    for state in range(N):
        for action in range(A):
            sp = next_state_deterministic(state, action)
            idx = state * A + action
            if state in TERMINAL_STATES:
                omega[idx] = 0.0
            elif sp == GOAL_GRID:
                omega[idx] = 1.0
            elif sp in PIT_GRIDS:
                omega[idx] = -5.0
            else:
                omega[idx] = -0.1

    def psi(xp):
        v = torch.zeros(N * A, dtype=torch.float64)
        for x in states:
            for a in actions:
                if next_state_deterministic(int(x), int(a)) == int(xp):
                    v[int(x) * A + int(a)] = 1.0
        return v

    mdp = DiscreteMDP(
        states=states,
        actions=actions,
        gamma=GAMMA,
        x0=X0,
        omega=omega,
        phi=phi,
        psi=psi,
        terminal_states=TERMINAL_STATES,
    )
    return mdp, phi, states, actions


def full_feature_matrix(states, actions, phi):
    return torch.vstack(
        [
            phi(int(s), int(a)).to(dtype=torch.float64)
            for s in states
            for a in actions
        ]
    )


def dataset_candidates():
    candidates = []
    run_idx = 0
    for (reset_name, reset_cfg), (prop_key, (proportions, prop_label)), eps, n_steps in itertools.product(
        RESET_CONFIGS.items(),
        PROPORTION_CONFIGS.items(),
        EPSILON_VALUES,
        DATASET_SIZES,
    ):
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


def normalize_probs(prob_dict):
    modes = list(prob_dict.keys())
    probs = np.asarray([prob_dict[mode] for mode in modes], dtype=np.float64)
    total = probs.sum()
    if total <= 0.0:
        raise ValueError("reset probabilities must have positive mass")
    return modes, probs / total


def sample_reset(rng, reset_cfg):
    modes, probs = normalize_probs(reset_cfg["reset_probs"])
    mode = str(rng.choice(modes, p=probs))
    forbidden = TERMINAL_STATES | WALL_STATES
    valid_start_states = [s for s in range(N) if s not in forbidden]

    if mode == "x0":
        return X0
    if mode == "random":
        return int(rng.choice(valid_start_states))
    if mode == "custom":
        initial_states = reset_cfg["initial_states"]
        if not initial_states:
            raise ValueError("custom reset requires initial_states")
        return int(rng.choice(initial_states))
    raise ValueError(f"Unsupported reset mode for this grid: {mode}")


def sample_matrix_policy_action(rng, policy_matrix, state):
    probs = np.asarray(policy_matrix[int(state)], dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total <= 0.0:
        probs = np.ones(A, dtype=np.float64) / A
    else:
        probs = probs / total
    return int(rng.choice(A, p=probs))


def collect_dataset_csv(mdp, pi_star, candidate, save_path):
    rng = np.random.default_rng(SEED)
    p = mdp.P.detach().cpu().numpy()
    r = mdp.r.detach().cpu().numpy().reshape(-1)
    pi_star = pi_star.detach().cpu().numpy()
    proportions = np.asarray(candidate["proportions"], dtype=np.float64)
    proportions = proportions / proportions.sum()

    rows = []
    state = sample_reset(rng, candidate["reset_cfg"])
    episode_policy_idx = int(rng.choice(2, p=proportions))
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

        row_idx = int(state) * A + action
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
                "reward": float(reward),
                "next_state": int(next_state),
            }
        )

        step += 1
        if truncated:
            state = sample_reset(rng, candidate["reset_cfg"])
            step = 0
            terminal_extra_remaining = None
            episode_policy_idx = int(rng.choice(2, p=proportions))
        elif terminated:
            if terminal_extra_remaining is None:
                terminal_extra_remaining = EXTRA_TERMINAL_STEPS
            else:
                terminal_extra_remaining -= 1

            if terminal_extra_remaining > 0:
                state = next_state
            else:
                state = sample_reset(rng, candidate["reset_cfg"])
                step = 0
                terminal_extra_remaining = None
                episode_policy_idx = int(rng.choice(2, p=proportions))
        else:
            state = next_state

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["state", "action", "reward", "next_state"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def feature_coverage(rows, phi_full, optimal_occupancy, beta):
    states = np.asarray([row["state"] for row in rows], dtype=np.int64)
    actions = np.asarray([row["action"] for row in rows], dtype=np.int64)
    phi_flat = phi_full.detach().cpu().numpy()
    d = int(phi_flat.shape[1])
    phi_by_pair = phi_flat.reshape(N, A, d)
    phi_data = phi_by_pair[states, actions]
    occ = optimal_occupancy.detach().cpu().numpy().reshape(N * A)
    covariance = float(beta) * np.eye(d) + (phi_data.T @ phi_data) / len(rows)
    lambda_target = phi_flat.T @ occ
    return float(lambda_target.T @ np.linalg.solve(covariance, lambda_target))


def load_sbeed_dataset(dataset_path, device):
    states = []
    actions = []
    rewards = []
    next_states = []
    with Path(dataset_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            states.append(int(row["state"]))
            actions.append(int(row["action"]))
            rewards.append(float(row["reward"]))
            next_states.append(int(row["next_state"]))

    done = [sp in TERMINAL_STATES for sp in next_states]
    dataset = DiscreteSBEEDDataset(
        X=states,
        A=actions,
        R=rewards,
        X_next=next_states,
        D=done,
    )
    dataset.validate(N, A)
    return dataset.to(device)


def build_sbeed_solver(dataset_n, device):
    value_features = TabularStateFeatures(N)
    rho_features = TabularStateActionFeatures(N, A)
    return DiscreteSBEED(
        n_states=N,
        n_actions=A,
        gamma=GAMMA,
        value_param=LinearValueParam(value_features, N),
        rho_param=LinearRhoParam(rho_features, N, A),
        policy_param=SoftmaxLinearPolicyParam(value_features, N, A),
        lambda_entropy=SBEED_LAMBDA_ENTROPY,
        eta=SBEED_ETA,
        lr_value=SBEED_LR_VALUE,
        lr_rho=SBEED_LR_RHO,
        lr_policy=SBEED_LR_POLICY,
        tau=SBEED_TAU,
        max_buffer_size=dataset_n,
        batch_size=SBEED_BATCH_SIZE,
        rollout_length=SBEED_ROLLOUT_LENGTH,
        seed=SEED,
        device=device,
    )


def build_gen_fogas_solver(dataset_path, device):
    u_features = TabularFeatures(N, A)
    q_features = TabularFeatures(N, A)
    policy_features = TabularFeatures(N, A)

    return FinalLinearSolver(
        n_states=N,
        n_actions=A,
        gamma=GAMMA,
        x0=X0,
        csv_path=str(dataset_path),
        u_function=LinearFunction(u_features),
        q_function=LinearQFunction(q_features),
        policy_features=policy_features,
        seed=SEED,
        device=device,
        theta_mode=GEN_THETA_MODE,
        theta_optimizer=GEN_THETA_OPTIMIZER,
        theta_start_mode=GEN_THETA_START_MODE,
        theta_include_beta_cov=GEN_THETA_INCLUDE_BETA_COV,
        theta_lr=GEN_THETA_LR,
        theta_inner_steps=GEN_THETA_INNER_STEPS,
        theta_lambda=GEN_THETA_LAMBDA,
        beta_update=GEN_BETA_UPDATE,
    )


def blank_metrics():
    return {
        "feature_coverage": np.nan,
        "greedy_on_data_quality": np.nan,
        "greedy_optimal_states_quality": np.nan,
        "greedy_avg_reward": np.nan,
        "greedy_success_rate": np.nan,
        "greedy_v_x0": np.nan,
        "greedy_v_gap": np.nan,
        "solver_on_data_quality": np.nan,
        "solver_optimal_states_quality": np.nan,
        "solver_avg_reward": np.nan,
        "solver_success_rate": np.nan,
        "solver_v_x0": np.nan,
        "solver_v_gap": np.nan,
    }


def base_row(candidate, algorithm, coverage, status="ok", error=""):
    row = {
        "run_idx": int(candidate["run_idx"]),
        "algorithm": algorithm,
        "dataset_size": int(candidate["dataset_size"]),
        "epsilon": float(candidate["epsilon"]),
        "proportions": candidate["proportion_label"],
        "proportion_key": candidate["proportion_key"],
        "reset_mode": candidate["reset_name"],
        "extra_terminal_steps": EXTRA_TERMINAL_STEPS,
        "seed": SEED,
        "status": status,
        "error": error,
        "elapsed_seconds": np.nan,
    }
    row.update(blank_metrics())
    row["feature_coverage"] = coverage
    return row


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
            num_trajectories=NUM_TRAJECTORIES,
            max_steps=MAX_STEPS,
            seed=SEED,
        )["policy"],
        f"{policy_mode}_avg_reward": evaluator.average_return(
            policy_mode=policy_mode,
            num_trajectories=NUM_TRAJECTORIES,
            max_steps=MAX_STEPS,
            seed=SEED,
            terminal_states=TERMINAL_STATES,
        )["policy"],
        f"{policy_mode}_success_rate": evaluator.success_rate(
            goal_state=GOAL_GRID,
            policy_mode=policy_mode,
            num_trajectories=NUM_TRAJECTORIES,
            max_steps=MAX_STEPS,
            seed=SEED,
            terminal_states=TERMINAL_STATES,
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


def add_common_metrics(row, solver, mdp, planner, dataset, d_star, v_star):
    solver.mdp = mdp
    evaluator = FOGASEvaluator(solver=solver, mdp=mdp, planner=planner)
    for mode in ("greedy", "solver"):
        row.update(evaluate_policy_family(evaluator, dataset, mode, d_star, v_star))


def run_sbeed(candidate, dataset_path, dataset, mdp, planner, d_star, v_star, coverage, device, torch_threads, base_elapsed):
    start = time.perf_counter()
    row = base_row(candidate, "SBEED", coverage)
    row.update(
        {
            "steps": SBEED_STEPS,
            "lambda_entropy": SBEED_LAMBDA_ENTROPY,
            "eta": SBEED_ETA,
            "lr_value": SBEED_LR_VALUE,
            "lr_rho": SBEED_LR_RHO,
            "lr_policy": SBEED_LR_POLICY,
            "tau": SBEED_TAU,
            "rollout_length": SBEED_ROLLOUT_LENGTH,
            "batch_size": "" if SBEED_BATCH_SIZE is None else SBEED_BATCH_SIZE,
            "device": str(device),
            "torch_threads": int(torch_threads),
            "last_objective": np.nan,
            "last_primal_mse": np.nan,
            "last_dual_mse": np.nan,
            "last_theta_grad_norm": np.nan,
            "last_beta_grad_norm": np.nan,
            "last_policy_grad_norm": np.nan,
        }
    )

    try:
        sbeed_dataset = load_sbeed_dataset(dataset_path, device)
        solver = build_sbeed_solver(sbeed_dataset.n, device)
        solver.dataset = sbeed_dataset
        solver.n = sbeed_dataset.n

        last_stats = {}
        for _ in range(SBEED_STEPS):
            last_stats = solver.step()

        solver.pi = solver.get_policy_matrix().to(dtype=torch.float64, device=device)
        add_common_metrics(row, solver, mdp, planner, dataset, d_star, v_star)
        row.update(
            {
                "last_objective": finite_float(last_stats.get("objective")),
                "last_primal_mse": finite_float(last_stats.get("primal_mse")),
                "last_dual_mse": finite_float(last_stats.get("dual_mse")),
                "last_theta_grad_norm": finite_float(last_stats.get("theta_grad_norm")),
                "last_beta_grad_norm": finite_float(last_stats.get("beta_grad_norm")),
                "last_policy_grad_norm": finite_float(last_stats.get("policy_grad_norm")),
            }
        )
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        row["elapsed_seconds"] = float(base_elapsed + time.perf_counter() - start)
    return row


def run_generalized_fogas(candidate, dataset_path, dataset, mdp, planner, d_star, v_star, coverage, device, torch_threads, base_elapsed):
    start = time.perf_counter()
    row = base_row(candidate, "Generalized FOGAS", coverage)
    row.update(
        {
            "alpha": GEN_ALPHA,
            "eta": GEN_ETA,
            "rho": GEN_RHO,
            "T": GEN_T,
            "theta_lr": GEN_THETA_LR,
            "theta_inner_steps": GEN_THETA_INNER_STEPS,
            "theta_lambda": GEN_THETA_LAMBDA,
            "theta_mode": GEN_THETA_MODE,
            "theta_optimizer": GEN_THETA_OPTIMIZER,
            "theta_start_mode": GEN_THETA_START_MODE,
            "theta_include_beta_cov": GEN_THETA_INCLUDE_BETA_COV,
            "beta_update": GEN_BETA_UPDATE,
            "policy_optimizer": GEN_POLICY_OPTIMIZER,
            "policy_gradient": GEN_POLICY_GRADIENT,
            "reinforce_samples": GEN_REINFORCE_SAMPLES,
            "state_weight_update": GEN_STATE_WEIGHT_UPDATE,
            "device": str(device),
            "torch_threads": int(torch_threads),
            "final_total_loss": np.nan,
            "final_policy_objective": np.nan,
            "final_beta_objective": np.nan,
            "final_q_objective": np.nan,
            "final_theta_norm": np.nan,
            "final_policy_grad_norm": np.nan,
            "final_beta_grad_norm": np.nan,
            "final_theta_grad_norm": np.nan,
        }
    )

    try:
        solver = build_gen_fogas_solver(dataset_path, device)
        pi = solver.run(
            alpha=GEN_ALPHA,
            eta=GEN_ETA,
            rho=GEN_RHO,
            T=GEN_T,
            theta_lr=GEN_THETA_LR,
            theta_inner_steps=GEN_THETA_INNER_STEPS,
            theta_lambda=GEN_THETA_LAMBDA,
            policy_optimizer=GEN_POLICY_OPTIMIZER,
            policy_gradient=GEN_POLICY_GRADIENT,
            reinforce_samples=GEN_REINFORCE_SAMPLES,
            state_weight_update=GEN_STATE_WEIGHT_UPDATE,
            tqdm_print=False,
            verbose=False,
        )
        solver.pi = pi.to(dtype=torch.float64, device=device)
        add_common_metrics(row, solver, mdp, planner, dataset, d_star, v_star)

        diagnostics = solver.get_diagnostics() or []
        if diagnostics:
            final = diagnostics[-1]
            row.update(
                {
                    "final_total_loss": finite_float(final.get("total_loss")),
                    "final_policy_objective": finite_float(final.get("policy_objective")),
                    "final_beta_objective": finite_float(final.get("beta_objective")),
                    "final_q_objective": finite_float(final.get("q_objective")),
                    "final_theta_norm": finite_float(final.get("theta_norm")),
                    "final_policy_grad_norm": finite_float(final.get("policy_grad_norm")),
                    "final_beta_grad_norm": finite_float(final.get("beta_grad_norm")),
                    "final_theta_grad_norm": finite_float(final.get("theta_grad_norm")),
                }
            )
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        row["elapsed_seconds"] = float(base_elapsed + time.perf_counter() - start)
    return row


def failed_row(candidate, algorithm, exc, elapsed):
    display_name = "SBEED" if algorithm == "sbeed" else "Generalized FOGAS"
    row = base_row(candidate, display_name, np.nan, status="failed", error=repr(exc))
    row["elapsed_seconds"] = float(elapsed)
    return row


def run_dataset_worker(payload):
    candidate, algorithms, device_str, torch_threads = payload
    configure_worker_threads(torch_threads)
    set_seed(SEED)
    device = torch.device(device_str)

    worker_start = time.perf_counter()
    rows = {}

    try:
        mdp, phi, states, actions = build_mdp()
        planner = Planner(mdp)
        phi_full = full_feature_matrix(states, actions, phi)
        d_star = (
            planner.mu_star.cpu() / (planner.mu_star.cpu().sum() + 1e-300)
        ).reshape(N, A).sum(dim=1)
        v_star = planner.v_star.detach().cpu()

        with tempfile.TemporaryDirectory(prefix="dataset_grid_10grid_", dir="/tmp") as tmp:
            dataset_path = Path(tmp) / f"dataset_{candidate['run_idx']}.csv"
            dataset_rows = collect_dataset_csv(mdp, planner.pi_star, candidate, dataset_path)
            coverage = feature_coverage(dataset_rows, phi_full, planner.mu_star, COVERAGE_BETA)
            dataset = FOGASDataset(dataset_path)
            base_elapsed = time.perf_counter() - worker_start

            mdp.to(device)
            planner.to(device)

            if "sbeed" in algorithms:
                rows["sbeed"] = run_sbeed(
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
                )

            if "generalized_fogas" in algorithms:
                rows["generalized_fogas"] = run_generalized_fogas(
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
                )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - worker_start
        for algorithm in algorithms:
            rows[algorithm] = failed_row(candidate, algorithm, exc, elapsed)

    return rows


def parse_algorithms(value):
    algorithms = []
    for item in str(value).split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in ALGORITHM_ALIASES:
            raise argparse.ArgumentTypeError(
                f"Unknown algorithm {item!r}. Expected a comma-separated subset of "
                "sbeed,generalized_fogas."
            )
        canonical = ALGORITHM_ALIASES[key]
        if canonical not in algorithms:
            algorithms.append(canonical)
    if not algorithms:
        raise argparse.ArgumentTypeError("At least one algorithm is required.")
    return algorithms


def parse_devices(value):
    devices = [item.strip() for item in str(value).split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("At least one device is required.")
    return devices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the deterministic 10-grid dataset search for SBEED and generalized FOGAS."
    )
    parser.add_argument(
        "--algorithms",
        type=parse_algorithms,
        default=parse_algorithms("generalized_fogas"),
        help="Comma-separated algorithms: sbeed,generalized_fogas.",
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
        help="Skip successful dataset rows already present in the output CSVs.",
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
        help="Directory where the two result CSVs are written.",
    )
    return parser.parse_args()


def candidate_key(candidate_or_row):
    return (
        int(candidate_or_row["dataset_size"]),
        float(candidate_or_row["epsilon"]),
        str(candidate_or_row["proportion_key"]),
        str(candidate_or_row["reset_mode"] if "reset_mode" in candidate_or_row else candidate_or_row["reset_name"]),
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


def append_row(output_csv, algorithm, row):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES[algorithm], extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_tasks(candidates, algorithms, output_dir, resume, devices, torch_threads):
    completed_by_algorithm = {}
    for algorithm in algorithms:
        output_csv = output_dir / OUTPUT_FILES[algorithm]
        completed_by_algorithm[algorithm] = successful_completed_keys(output_csv) if resume else set()

    tasks = []
    for candidate in candidates:
        needed = [
            algorithm
            for algorithm in algorithms
            if candidate_key(candidate) not in completed_by_algorithm[algorithm]
        ]
        if not needed:
            continue
        device = devices[len(tasks) % len(devices)]
        tasks.append((candidate, needed, device, torch_threads))
    return tasks, completed_by_algorithm


def progress_iter(iterable, total, desc, disable):
    if tqdm is None or disable:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="dataset")


def main():
    args = parse_args()
    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    output_dir = Path(args.output_dir)

    candidates = dataset_candidates()
    if args.max_datasets is not None:
        candidates = candidates[: max(0, int(args.max_datasets))]

    if not args.resume:
        for algorithm in args.algorithms:
            output_csv = output_dir / OUTPUT_FILES[algorithm]
            if output_csv.exists():
                output_csv.unlink()

    tasks, completed_by_algorithm = build_tasks(
        candidates=candidates,
        algorithms=args.algorithms,
        output_dir=output_dir,
        resume=args.resume,
        devices=args.devices,
        torch_threads=torch_threads,
    )

    print("10-grid deterministic dataset search")
    print(f"Algorithms       : {', '.join(args.algorithms)}")
    print(f"Output directory : {output_dir.resolve()}")
    print(f"Dataset variants : {len(candidates)}")
    print(f"Tasks to run     : {len(tasks)}")
    print(f"Workers          : {workers}")
    print(f"Devices          : {', '.join(args.devices)}")
    print(f"Torch threads    : {torch_threads}")
    if args.resume:
        for algorithm in args.algorithms:
            print(f"Resume {algorithm:18s}: {len(completed_by_algorithm[algorithm])} completed row(s)")

    if not tasks:
        print("No tasks to run.")
        return

    completed = 0
    if workers == 1:
        iterator = progress_iter(tasks, len(tasks), "Dataset grid", args.no_progress)
        for task in iterator:
            rows = run_dataset_worker(task)
            for algorithm, row in rows.items():
                append_row(output_dir / OUTPUT_FILES[algorithm], algorithm, row)
            completed += 1
            if args.no_progress:
                print(f"[{completed}/{len(tasks)}] run_idx={task[0]['run_idx']} done")
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
                    rows = future.result()
                except Exception as exc:  # noqa: BLE001
                    rows = {
                        algorithm: failed_row(task[0], algorithm, exc, 0.0)
                        for algorithm in task[1]
                    }
                for algorithm, row in rows.items():
                    append_row(output_dir / OUTPUT_FILES[algorithm], algorithm, row)
                completed += 1
                if args.no_progress:
                    statuses = ", ".join(f"{alg}={row.get('status')}" for alg, row in rows.items())
                    print(f"[{completed}/{len(tasks)}] run_idx={task[0]['run_idx']} {statuses}")

    print("\nDataset grid complete.")
    for algorithm in args.algorithms:
        print(f"{algorithm}: {output_dir / OUTPUT_FILES[algorithm]}")


if __name__ == "__main__":
    main()
