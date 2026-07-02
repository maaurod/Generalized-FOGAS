"""
Shared FinalParametrizedSolver neural grid-search utilities for 5x5 grids.

This mirrors the FinalLinearSolver RBF search but uses small tanh MLP
parametrizations for u_beta, Q_theta, and the policy. It writes results after
every candidate and can shard candidates across multiple GPU worker processes.
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
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "generalization" / "hyperparam_grids" / "5grid"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from rl_methods.fogas import FOGASEvaluator  # noqa: E402
from rl_methods.fogas_generalization import (  # noqa: E402
    FinalParametrizedSolver,
    NeuralPolicyParam,
    NeuralQParam,
    NeuralUParam,
    StateActionMLPModule,
    StateMLPPolicyModule,
)
from rl_methods.mdp import DiscreteMDP, Planner  # noqa: E402


SEED = 42
NUM_TRAJECTORIES = 100
MAX_STEPS = 20
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

STATES = torch.arange(25, dtype=torch.long)
ACTIONS = torch.arange(4, dtype=torch.long)
N = len(STATES)
A = len(ACTIONS)
GRID_SIZE = 5
X0 = 0
GOAL_GRID = 24
PIT_GRID = 18
WALL_STATES = {6, 7, 12}
INTENDED_PROB = 0.8

# Small-network search. At roughly a few seconds per run on one GPU, this gives
# enough candidates for an overnight run while keeping each candidate cheap.
ALPHA_GRID = [1e-3, 3e-3]
ETA_GRID = [1e-5, 3e-5, 1e-4, 3e-4]
RHO_GRID = [0.01, 0.05, 0.1]
T_GRID = [300, 600, 1000]
THETA_LR_GRID = [3e-3, 1e-2, 3e-2]
THETA_INNER_STEPS_GRID = [2, 3]
THETA_LAMBDA_GRID = [1e-4, 1e-3, 1e-2]
HIDDEN_SIZES_GRID = [(8,), (16,)]
BETA_REG_GRID = [1e-4, 1e-3]
POLICY_GRADIENT_GRID = ["exact", "reinforce"]
NN_SEED_GRID = [42, 123]

BETA_UPDATE = "fogas_diag"
POLICY_OPTIMIZER = "adam"
REINFORCE_SAMPLES = 1
STATE_WEIGHT_UPDATE = "normal"

# The stochastic grid is harder than the deterministic one: transition noise
# and the pit make raw solver policies much less reliable than greedy rollouts.
# These candidates put the next rerun around the best observed stochastic
# regions first, then append the old broad baseline grid as a fallback.
STOCHASTIC_BASE_POINTS = [
    # Best greedy policy from the previous stochastic run.
    (1e-3, 1e-5, 0.1, 1000, 3e-2, 3, 1e-4, (8,), 1e-3, "exact", 123),
    # Best reinforce greedy policy.
    (1e-3, 1e-5, 0.01, 600, 3e-2, 2, 1e-4, (16,), 1e-3, "reinforce", 123),
    # Best raw solver policy.
    (1e-3, 3e-5, 0.1, 1000, 3e-2, 3, 1e-3, (8,), 1e-4, "exact", 123),
    # Good solver/greedy compromise with stronger theta regularization.
    (1e-3, 3e-5, 0.1, 600, 1e-2, 3, 1e-2, (16,), 1e-3, "exact", 42),
    # Good reinforce row with longer horizon and theta_lambda=1e-2.
    (1e-3, 1e-5, 0.01, 1000, 1e-2, 3, 1e-2, (16,), 1e-4, "reinforce", 42),
]

STOCHASTIC_ONE_FACTOR_SWEEPS = {
    0: [3e-4, 5e-4, 1e-3, 2e-3, 3e-3],
    1: [3e-6, 1e-5, 3e-5, 1e-4],
    2: [0.003, 0.01, 0.03, 0.05, 0.1, 0.2],
    3: [300, 600, 1000, 1500, 2000],
    4: [3e-3, 1e-2, 3e-2, 6e-2],
    5: [2, 3, 5],
    6: [1e-5, 1e-4, 1e-3, 1e-2, 3e-2],
    7: [(8,), (16,), (32,), (16, 16)],
    8: [1e-5, 1e-4, 1e-3, 1e-2],
    9: ["exact", "reinforce"],
    10: [42, 123, 777],
}

PROBLEMS = {
    "deterministic": {
        "description": "deterministic 5x5 FinalParametrizedSolver neural grid search",
        "dataset_path": DATASETS_DIR / "5grid.csv",
        "output_csv": RESULTS_DIR / "final_parametrized_5grid_neural_grid_search.csv",
        "best_csv": RESULTS_DIR / "final_parametrized_5grid_neural_grid_search_best.csv",
        "gamma": 0.99,
        "terminal_states": {GOAL_GRID},
        "stochastic": False,
    },
    "stochastic": {
        "description": "stochastic 5x5 FinalParametrizedSolver neural grid search",
        "dataset_path": DATASETS_DIR / "5grid_stochastic.csv",
        "output_csv": RESULTS_DIR / "final_parametrized_5grid_stochastic_neural_grid_search.csv",
        "best_csv": RESULTS_DIR / "final_parametrized_5grid_stochastic_neural_grid_search_best.csv",
        "gamma": 0.9,
        "terminal_states": {GOAL_GRID, PIT_GRID},
        "stochastic": True,
    },
}


def parse_args(description):
    parser = argparse.ArgumentParser(description=f"Run the {description}.")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit the number of candidates. Useful for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip parameter combinations already present in the output CSV.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(DEFAULT_DEVICE),
        help="Torch device for a single-worker run, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help=(
            "Comma-separated devices for parallel runs, e.g. cuda:0,cuda:1. "
            "Use 'auto' to use all visible CUDA devices. Defaults to --device."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Parallel candidate workers. Defaults to the number of --devices, "
            "or 1 when --devices is omitted."
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch CPU threads per worker. Default 1 for GPU grid searches.",
    )
    parser.add_argument(
        "--time-budget-hours",
        type=float,
        default=0.0,
        help="Stop after this many hours. Default 0 disables the time limit.",
    )
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


def state_to_pos(s):
    return divmod(int(s), GRID_SIZE)


def pos_to_state(row, col):
    return int(row) * GRID_SIZE + int(col)


def move_deterministic(s, a, terminal_states):
    s = int(s)
    a = int(a)

    if s in terminal_states:
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


def deterministic_next_state(s, a, terminal_states):
    return move_deterministic(s, a, terminal_states)


def stochastic_transition_probs(s, a, terminal_states, intended_prob=INTENDED_PROB):
    s = int(s)
    a = int(a)
    probs_by_state = {}

    for candidate_a in range(A):
        prob = (1.0 - intended_prob) / A
        if candidate_a == a:
            prob += intended_prob

        sp = move_deterministic(s, candidate_a, terminal_states)
        probs_by_state[sp] = probs_by_state.get(sp, 0.0) + prob

    return probs_by_state


def reward_from_next_state(sp, stochastic):
    sp = int(sp)
    if sp == GOAL_GRID:
        return 1.0
    if stochastic and sp == PIT_GRID:
        return -1.0
    return -0.1 if stochastic else -0.01


def build_mdp(problem_name, device):
    problem = PROBLEMS[problem_name]
    terminal_states = set(problem["terminal_states"])
    stochastic = bool(problem["stochastic"])
    gamma = float(problem["gamma"])

    def transition_fn(s, a):
        probs = torch.zeros(N, dtype=torch.float64)
        if stochastic:
            for sp, prob in stochastic_transition_probs(s, a, terminal_states).items():
                probs[sp] = prob
        else:
            sp = deterministic_next_state(s, a, terminal_states)
            probs[sp] = 1.0
        return probs

    def reward_fn(s, a):
        if stochastic:
            return sum(
                prob * reward_from_next_state(sp, stochastic=True)
                for sp, prob in stochastic_transition_probs(s, a, terminal_states).items()
            )

        sp = deterministic_next_state(s, a, terminal_states)
        return reward_from_next_state(sp, stochastic=False)

    mdp = DiscreteMDP(
        states=STATES,
        actions=ACTIONS,
        gamma=gamma,
        x0=X0,
        reward_fn=reward_fn,
        transition_fn=transition_fn,
        terminal_states=list(terminal_states),
    ).to(device)
    planner = Planner(mdp).to(device)
    return mdp, planner


def state_inputs():
    return torch.tensor(
        [[r / 4.0, c / 4.0] for r in range(GRID_SIZE) for c in range(GRID_SIZE)],
        dtype=torch.float64,
    )


def make_solver(
    gamma,
    dataset_path,
    device,
    theta_lr,
    theta_inner_steps,
    theta_lambda,
    hidden_sizes,
    beta_reg,
    nn_seed,
):
    set_seed(nn_seed)
    inputs = state_inputs()
    u_param = NeuralUParam(
        StateActionMLPModule(
            n_states=N,
            n_actions=A,
            state_inputs=inputs,
            hidden_sizes=hidden_sizes,
            dtype=torch.float64,
        )
    )
    q_param = NeuralQParam(
        StateActionMLPModule(
            n_states=N,
            n_actions=A,
            state_inputs=inputs,
            hidden_sizes=hidden_sizes,
            dtype=torch.float64,
        )
    )
    policy_param = NeuralPolicyParam(
        StateMLPPolicyModule(
            n_states=N,
            n_actions=A,
            state_inputs=inputs,
            hidden_sizes=hidden_sizes,
            dtype=torch.float64,
        )
    )

    return FinalParametrizedSolver(
        n_states=N,
        n_actions=A,
        gamma=gamma,
        x0=X0,
        csv_path=str(dataset_path),
        u_param=u_param,
        q_param=q_param,
        policy_param=policy_param,
        seed=nn_seed,
        device=device,
        theta_include_beta_cov=False,
        theta_mode="reg_fixed",
        theta_lambda=theta_lambda,
        theta_optimizer="adam",
        theta_inner_steps=theta_inner_steps,
        theta_lr=theta_lr,
        theta_start_mode="warm",
        beta_update=BETA_UPDATE,
        beta_reg=beta_reg,
    )


def hidden_sizes_label(hidden_sizes):
    return "x".join(str(int(size)) for size in hidden_sizes)


def parse_hidden_sizes(label):
    return tuple(int(part) for part in str(label).split("x") if part)


def candidate_key(row):
    return (
        float(row["alpha"]),
        float(row["eta"]),
        float(row["rho"]),
        int(row["T"]),
        float(row["theta_lr"]),
        int(row["theta_inner_steps"]),
        float(row["theta_lambda"]),
        str(row["hidden_sizes"]),
        float(row["beta_reg"]),
        str(row["policy_gradient"]),
        int(row["nn_seed"]),
    )


def load_existing_results(resume, output_csv):
    if not resume or not output_csv.exists():
        return [], set()

    df = pd.read_csv(output_csv)
    rows = df.to_dict("records")
    completed = {candidate_key(row) for row in rows}
    return rows, completed


def ordered_results_frame(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df

    return df.sort_values(
        by=[
            "greedy_success_rate",
            "solver_success_rate",
            "greedy_avg_return",
            "solver_avg_return",
            "elapsed_seconds",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def save_results(results, output_csv, best_csv):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = ordered_results_frame(results)
    df.to_csv(output_csv, index=False)

    successful = df[df["status"] == "ok"] if not df.empty else df
    if not successful.empty:
        successful.head(1).to_csv(best_csv, index=False)


def blank_metrics():
    return {
        "solver_success_rate": np.nan,
        "solver_avg_return": np.nan,
        "solver_v_x0": np.nan,
        "solver_v_gap": np.nan,
        "greedy_success_rate": np.nan,
        "greedy_avg_return": np.nan,
        "greedy_v_x0": np.nan,
        "greedy_v_gap": np.nan,
        "final_total_loss": np.nan,
        "final_policy_objective": np.nan,
        "final_beta_objective": np.nan,
        "final_q_objective": np.nan,
        "final_theta_norm": np.nan,
        "final_policy_grad_norm": np.nan,
        "final_beta_grad_norm": np.nan,
        "final_theta_grad_norm": np.nan,
    }


def base_row(params, device, status="ok", error=""):
    (
        alpha,
        eta,
        rho,
        T,
        theta_lr,
        theta_inner_steps,
        theta_lambda,
        hidden_sizes,
        beta_reg,
        policy_gradient,
        nn_seed,
    ) = params
    row = {
        "alpha": float(alpha),
        "eta": float(eta),
        "rho": float(rho),
        "T": int(T),
        "theta_lr": float(theta_lr),
        "theta_inner_steps": int(theta_inner_steps),
        "theta_lambda": float(theta_lambda),
        "hidden_sizes": hidden_sizes_label(hidden_sizes),
        "beta_reg": float(beta_reg),
        "theta_mode": "reg_fixed",
        "theta_optimizer": "adam",
        "theta_start_mode": "warm",
        "theta_include_beta_cov": False,
        "beta_update": BETA_UPDATE,
        "policy_optimizer": POLICY_OPTIMIZER,
        "policy_gradient": str(policy_gradient),
        "reinforce_samples": int(REINFORCE_SAMPLES),
        "state_weight_update": STATE_WEIGHT_UPDATE,
        "num_trajectories": int(NUM_TRAJECTORIES),
        "max_steps": int(MAX_STEPS),
        "seed": int(SEED),
        "nn_seed": int(nn_seed),
        "device": str(device),
        "status": status,
        "error": error,
        "elapsed_seconds": np.nan,
    }
    row.update(blank_metrics())
    return row


def evaluate_policy(planner, evaluator, policy_mode, terminal_states, d_star, v_star):
    pi = evaluator.get_policy(policy_mode)
    v_pi, _ = planner.evaluate_policy(pi)
    v_gap = float((d_star * (v_star - v_pi.detach().cpu())).sum().item())
    return {
        f"{policy_mode}_success_rate": float(
            evaluator.success_rate(
                goal_state=GOAL_GRID,
                policy_mode=policy_mode,
                num_trajectories=NUM_TRAJECTORIES,
                max_steps=MAX_STEPS,
                seed=SEED,
                terminal_states=terminal_states,
            )["policy"]
        ),
        f"{policy_mode}_avg_return": float(
            evaluator.average_return(
                policy_mode=policy_mode,
                num_trajectories=NUM_TRAJECTORIES,
                max_steps=MAX_STEPS,
                seed=SEED,
                terminal_states=terminal_states,
            )["policy"]
        ),
        f"{policy_mode}_v_x0": float(v_pi[planner.x0].detach().cpu().item()),
        f"{policy_mode}_v_gap": v_gap,
    }


def finite_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def run_candidate(params, problem_name, mdp, planner, dataset_path, device, d_star, v_star):
    (
        alpha,
        eta,
        rho,
        T,
        theta_lr,
        theta_inner_steps,
        theta_lambda,
        hidden_sizes,
        beta_reg,
        policy_gradient,
        nn_seed,
    ) = params
    start = time.perf_counter()
    terminal_states = set(PROBLEMS[problem_name]["terminal_states"])
    row = base_row(params, device)

    try:
        solver = make_solver(
            gamma=mdp.gamma,
            dataset_path=dataset_path,
            device=device,
            theta_lr=theta_lr,
            theta_inner_steps=theta_inner_steps,
            theta_lambda=theta_lambda,
            hidden_sizes=hidden_sizes,
            beta_reg=beta_reg,
            nn_seed=nn_seed,
        )
        evaluator = FOGASEvaluator(solver=solver, mdp=mdp, planner=planner)

        solver.run(
            alpha=alpha,
            eta=eta,
            rho=rho,
            T=T,
            theta_lr=theta_lr,
            theta_inner_steps=theta_inner_steps,
            theta_lambda=theta_lambda,
            policy_optimizer=POLICY_OPTIMIZER,
            policy_gradient=policy_gradient,
            reinforce_samples=REINFORCE_SAMPLES,
            tqdm_print=False,
            verbose=False,
            state_weight_update=STATE_WEIGHT_UPDATE,
        )

        row.update(evaluate_policy(planner, evaluator, "solver", terminal_states, d_star, v_star))
        row.update(evaluate_policy(planner, evaluator, "greedy", terminal_states, d_star, v_star))

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
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        row["elapsed_seconds"] = float(time.perf_counter() - start)

    return row


def run_candidate_worker(params, problem_name, dataset_path, device_name, torch_threads):
    start = time.perf_counter()
    device = torch.device(device_name)
    try:
        configure_torch_threads(torch_threads)
        device = configure_device(device_name)
        set_seed(SEED)

        mdp, planner = build_mdp(problem_name, device)
        d_star = planner.state_mu_star.detach().cpu()
        v_star = planner.v_star.detach().cpu()
        return run_candidate(
            params=params,
            problem_name=problem_name,
            mdp=mdp,
            planner=planner,
            dataset_path=Path(dataset_path),
            device=device,
            d_star=d_star,
            v_star=v_star,
        )
    except Exception as exc:
        row = base_row(params, device, status="failed", error=repr(exc))
        row["elapsed_seconds"] = float(time.perf_counter() - start)
        return row


def baseline_candidates():
    return [
        tuple(candidate)
        for candidate in itertools.product(
            ALPHA_GRID,
            ETA_GRID,
            RHO_GRID,
            T_GRID,
            THETA_LR_GRID,
            THETA_INNER_STEPS_GRID,
            THETA_LAMBDA_GRID,
            HIDDEN_SIZES_GRID,
            BETA_REG_GRID,
            POLICY_GRADIENT_GRID,
            NN_SEED_GRID,
        )
    ]


def dedupe_candidates(candidates):
    unique = []
    seen = set()
    for candidate in candidates:
        candidate = tuple(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def stochastic_one_factor_candidates():
    candidates = list(STOCHASTIC_BASE_POINTS)
    for base in STOCHASTIC_BASE_POINTS:
        base = list(base)
        for idx, values in STOCHASTIC_ONE_FACTOR_SWEEPS.items():
            for value in values:
                candidate = list(base)
                candidate[idx] = value
                candidates.append(tuple(candidate))
    return candidates


def stochastic_exact_focus_candidates():
    return list(
        itertools.product(
            [5e-4, 1e-3, 2e-3],
            [1e-5, 3e-5],
            [0.05, 0.1, 0.2],
            [1000, 1500, 2000],
            [3e-2, 6e-2],
            [3, 5],
            [1e-4, 1e-3, 1e-2],
            [(8,), (16,)],
            [1e-4, 1e-3],
            ["exact"],
            [42, 123],
        )
    )


def stochastic_reinforce_focus_candidates():
    return list(
        itertools.product(
            [5e-4, 1e-3],
            [1e-5, 3e-5],
            [0.01, 0.05, 0.1],
            [600, 1000, 1500],
            [1e-2, 3e-2],
            [2, 3],
            [1e-4, 1e-3, 1e-2],
            [(8,), (16,)],
            [1e-4, 1e-3],
            ["reinforce"],
            [42, 123],
        )
    )


def stochastic_candidates():
    priority_candidates = (
        stochastic_one_factor_candidates()
        + stochastic_exact_focus_candidates()
        + stochastic_reinforce_focus_candidates()
    )
    return dedupe_candidates(priority_candidates + baseline_candidates())


def all_candidates(problem_name=None):
    if problem_name == "stochastic":
        return stochastic_candidates()
    return baseline_candidates()


def total_grid_size(problem_name=None):
    return len(all_candidates(problem_name))


def run_grid_search(problem_name):
    if problem_name not in PROBLEMS:
        raise ValueError(f"Unknown problem {problem_name!r}. Expected one of {sorted(PROBLEMS)}")

    problem = PROBLEMS[problem_name]
    args = parse_args(problem["description"])
    configure_torch_threads(args.torch_threads)
    set_seed(SEED)
    devices, workers = resolve_devices(args)
    primary_device = torch.device(devices[0])

    dataset_path = problem["dataset_path"]
    output_csv = problem["output_csv"]
    best_csv = problem["best_csv"]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Using devices: {', '.join(devices)}")
    print(f"Problem: {problem_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Results: {output_csv}")
    print(f"Workers: {workers}")
    print(f"Torch threads: {max(1, int(args.torch_threads))}")
    if args.time_budget_hours is None or float(args.time_budget_hours) <= 0:
        print("Time budget hours: disabled")
    else:
        print(f"Time budget hours: {args.time_budget_hours}")

    candidates_all = all_candidates(problem_name)
    if args.max_runs is not None:
        candidates_all = candidates_all[: max(0, int(args.max_runs))]

    results, completed = load_existing_results(args.resume, output_csv)
    candidates = [
        candidate
        for candidate in candidates_all
        if candidate_key(base_row(candidate, primary_device)) not in completed
    ]

    print(f"Total grid size: {total_grid_size(problem_name)}")
    print(f"Candidates to run: {len(candidates)}")
    if args.resume:
        print(f"Resumed rows: {len(results)}")
    if not args.resume and output_csv.exists():
        print("Existing output will be overwritten because --resume was not set.")

    time_budget_seconds = None
    if args.time_budget_hours is not None and float(args.time_budget_hours) > 0:
        time_budget_seconds = float(args.time_budget_hours) * 3600.0

    started_at = time.perf_counter()
    progress = not args.no_progress
    desc = f"FinalParametrizedSolver {problem_name} 5-grid neural search"
    outer = tqdm(candidates, desc=desc, unit="run", disable=not progress)

    stopped_for_budget = False
    if workers == 1:
        device = configure_device(devices[0])
        mdp, planner = build_mdp(problem_name, device)
        d_star = planner.state_mu_star.detach().cpu()
        v_star = planner.v_star.detach().cpu()

        for run_idx, params in enumerate(outer, start=len(results) + 1):
            if time_budget_seconds is not None and time.perf_counter() - started_at >= time_budget_seconds:
                stopped_for_budget = True
                break

            row = run_candidate(
                params=params,
                problem_name=problem_name,
                mdp=mdp,
                planner=planner,
                dataset_path=dataset_path,
                device=device,
                d_star=d_star,
                v_star=v_star,
            )
            row["run_idx"] = int(run_idx)
            results.append(row)
            save_results(results, output_csv, best_csv)

            if progress:
                outer.set_postfix(
                    {
                        "greedy_success": row["greedy_success_rate"],
                        "greedy_return": row["greedy_avg_return"],
                        "status": row["status"],
                    }
                )
    else:
        outer.close()
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
                    problem_name,
                    str(dataset_path),
                    device_name,
                    max(1, int(args.torch_threads)),
                )
                future_to_candidate[future] = (submit_idx, params, device_name)
                next_submit_idx += 1
                return True

            for _ in range(min(workers, len(candidates))):
                submit_next()

            outer = tqdm(
                total=len(candidates),
                desc=desc,
                unit="run",
                disable=not progress,
            )
            try:
                while future_to_candidate:
                    for future in as_completed(list(future_to_candidate)):
                        _, params, device_name = future_to_candidate.pop(future)
                        try:
                            row = future.result()
                        except Exception as exc:
                            row = base_row(
                                params,
                                torch.device(device_name),
                                status="failed",
                                error=repr(exc),
                            )
                        row["run_idx"] = int(next_run_idx)
                        next_run_idx += 1
                        results.append(row)
                        save_results(results, output_csv, best_csv)

                        if progress:
                            outer.update(1)
                            outer.set_postfix(
                                {
                                    "greedy_success": row["greedy_success_rate"],
                                    "greedy_return": row["greedy_avg_return"],
                                    "status": row["status"],
                                }
                            )

                        if can_submit_more():
                            submit_next()
                        else:
                            stopped_for_budget = next_submit_idx < len(candidates)
                        break
            finally:
                outer.close()

    save_results(results, output_csv, best_csv)
    df = ordered_results_frame(results)
    ok_count = int((df["status"] == "ok").sum()) if not df.empty else 0
    failed_count = int((df["status"] == "failed").sum()) if not df.empty else 0

    print("\nGrid search complete.")
    if stopped_for_budget:
        print("Stopped because the time budget was reached.")
    print(f"Rows saved: {len(df)}")
    print(f"Successful runs: {ok_count}")
    print(f"Failed runs: {failed_count}")
    print(f"Output CSV: {output_csv}")
    if best_csv.exists():
        print(f"Best row CSV: {best_csv}")
        print("\nTop result:")
        print(pd.read_csv(best_csv).to_string(index=False))
