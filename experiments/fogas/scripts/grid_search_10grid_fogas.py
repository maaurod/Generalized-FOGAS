"""Measure FOGAS sensitivity to offline-data coverage on the 10 x 10 grid.

For each candidate, this script rebuilds the tabular MDP, generates a dataset,
runs FOGAS with fixed hyperparameters, and records feature coverage, return,
success, on-data quality, and value gaps. Dataset size, epsilon exploration,
optimal/random policy proportion, and reset distribution are varied. The grid
is intentionally identical to ``grid_search_10grid_fqi.py`` so the comparison
in ``notebooks/10grid_tabular.ipynb`` uses matched data-generation conditions.

Run from the repository root with
``python3 experiments/fogas/scripts/grid_search_10grid_fogas.py``. Results are
checkpointed after every candidate in
``data/results/10grid_tabular/fogas_dataset_grid.csv``.
"""

import itertools
import random
import sys
import tempfile
import time
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
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "10grid_tabular"
OUTPUT_CSV = RESULTS_DIR / "fogas_dataset_grid.csv"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_methods.data_collection import DatasetAnalyzer, DiscreteDataBuffer
from rl_methods.fogas import FOGASEvaluator, FOGASDataset, FOGASSolver
from rl_methods.mdp import DiscreteMDP, Planner


# ---------------------------------------------------------------------
# Fixed run settings
# ---------------------------------------------------------------------
SEED = 42
NUM_TRAJECTORIES = 100
MAX_STEPS = 50
EXTRA_TERMINAL_STEPS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Fixed FOGAS hyperparameters
# ---------------------------------------------------------------------
FOGAS_ALPHA = 0.005
FOGAS_ETA = 0.002
FOGAS_RHO = 0.0005
FOGAS_D_THETA = 60
FOGAS_T = 20_000
FOGAS_BETA = 1e-7


# ---------------------------------------------------------------------
# Dataset grid
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def build_mdp():
    states = torch.arange(100, dtype=torch.int64)
    actions = torch.arange(4, dtype=torch.int64)
    n_states = len(states)
    n_actions = len(actions)
    gamma = 0.9
    x0 = 0
    goal = 99
    pits = {18, 32, 57, 61, 75}
    walls = {
        4, 11, 14, 17, 21, 22, 27, 34, 37,
        40, 42, 43, 44, 45, 46, 47, 49,
        54, 62, 64, 66, 72, 76, 82, 84, 86, 87, 94,
    }

    def phi(x, a):
        vec = torch.zeros(n_states * n_actions, dtype=torch.float64)
        vec[int(x) * n_actions + int(a)] = 1.0
        return vec

    def to_rc(s):
        return divmod(int(s), 10)

    def to_s(r, c):
        return r * 10 + c

    def next_state(s, a):
        s = int(s)
        a = int(a)
        if s == goal or s in pits:
            return s
        r, c = to_rc(s)
        if a == 0:
            r2, c2 = max(0, r - 1), c
        elif a == 1:
            r2, c2 = min(9, r + 1), c
        elif a == 2:
            r2, c2 = r, max(0, c - 1)
        elif a == 3:
            r2, c2 = r, min(9, c + 1)
        else:
            raise ValueError("Invalid action")
        sp = to_s(r2, c2)
        return s if sp in walls else sp

    terminal_states = {goal, *pits}
    step_cost = -0.1
    goal_reward = 1.0
    pit_reward = -5.0
    terminal_self_loop_reward = 0.0

    omega = torch.empty(n_states * n_actions, dtype=torch.float64)
    for state in range(n_states):
        for action in range(n_actions):
            sp = next_state(state, action)
            idx = state * n_actions + action
            if state in terminal_states:
                omega[idx] = terminal_self_loop_reward
            elif sp == goal:
                omega[idx] = goal_reward
            elif sp in pits:
                omega[idx] = pit_reward
            else:
                omega[idx] = step_cost

    def psi(xp):
        v = torch.zeros(n_states * n_actions, dtype=torch.float64)
        for x in states:
            for a in actions:
                if next_state(int(x), int(a)) == int(xp):
                    v[int(x) * n_actions + int(a)] = 1.0
        return v

    mdp = DiscreteMDP(
        states=states,
        actions=actions,
        phi=phi,
        omega=omega,
        gamma=gamma,
        x0=x0,
        psi=psi,
        terminal_states=terminal_states,
    )
    return mdp, phi, states, actions, goal, pits, walls, terminal_states


def full_feature_matrix(states, actions, phi):
    return torch.vstack([
        phi(int(s), int(a)).to(dtype=torch.float64)
        for s in states
        for a in actions
    ])


def collect_dataset(mdp, planner, goal, pits, walls, reset_cfg, eps, proportions, n_steps, save_path):
    collector = DiscreteDataBuffer(
        mdp=mdp,
        reset_probs=reset_cfg["reset_probs"],
        initial_states=reset_cfg["initial_states"],
        restricted_states=walls,
        max_steps=MAX_STEPS,
        terminal_states={goal, *pits},
        seed=SEED,
    )
    epsilon_policy = (planner.pi_star, eps)
    return collector.collect(
        policies=[epsilon_policy, "random"],
        proportions=proportions,
        n_steps=n_steps,
        extra_terminal_steps=EXTRA_TERMINAL_STEPS,
        episode_based=True,
        save_path=str(save_path),
        verbose=False,
    )


def evaluate_policy_family(evaluator, dataset, policy_mode, goal, terminal_states, d_star, v_star):
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
            terminal_states=terminal_states,
        )["policy"],
        f"{policy_mode}_success_rate": evaluator.success_rate(
            goal_state=goal,
            policy_mode=policy_mode,
            num_trajectories=NUM_TRAJECTORIES,
            max_steps=MAX_STEPS,
            seed=SEED,
            terminal_states=terminal_states,
        )["policy"],
        f"{policy_mode}_v_x0": float(v_pi[evaluator.mdp.x0].item()),
        f"{policy_mode}_v_gap": v_gap,
    }


def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = list(itertools.product(
        RESET_CONFIGS.items(),
        PROPORTION_CONFIGS.items(),
        EPSILON_VALUES,
        DATASET_SIZES,
    ))
    rows = []
    with tempfile.TemporaryDirectory(prefix="fogas_10grid_", dir="/tmp") as tmp:
        tmp_dir = Path(tmp)
        for run_idx, ((reset_name, reset_cfg), (prop_key, (proportions, prop_label)), eps, n_steps) in enumerate(
            tqdm(candidates, desc="FOGAS 10-grid dataset search", unit="run"),
            start=1,
        ):
            start = time.perf_counter()
            dataset_path = tmp_dir / f"dataset_{run_idx}.csv"
            row = {
                "run_idx": run_idx,
                "algorithm": "FOGAS",
                "dataset_size": int(n_steps),
                "epsilon": float(eps),
                "proportions": prop_label,
                "proportion_key": prop_key,
                "reset_mode": reset_name,
                "extra_terminal_steps": EXTRA_TERMINAL_STEPS,
                "seed": SEED,
                "status": "ok",
                "error": "",
            }
            try:
                mdp, phi, states, actions, goal, pits, walls, terminal_states = build_mdp()
                planner = Planner(mdp)
                phi_full = full_feature_matrix(states, actions, phi)
                d_star = (planner.mu_star.cpu() / (planner.mu_star.cpu().sum() + 1e-300)).reshape(mdp.N, mdp.A).sum(dim=1)
                v_star = planner.v_star.detach().cpu()

                collect_dataset(mdp, planner, goal, pits, walls, reset_cfg, eps, proportions, n_steps, dataset_path)
                analyzer = DatasetAnalyzer(dataset_path)
                dataset = FOGASDataset(dataset_path)
                row["feature_coverage"] = analyzer.feature_coverage(
                    phi_full,
                    optimal_occupancy=planner.mu_star,
                    n_states=mdp.N,
                    n_actions=mdp.A,
                    beta=FOGAS_BETA,
                )

                solver = FOGASSolver(
                    mdp=mdp,
                    phi=phi,
                    csv_path=str(dataset_path),
                    device=DEVICE,
                    seed=SEED,
                    beta=FOGAS_BETA,
                )
                planner.to(DEVICE)
                solver.run(
                    alpha=FOGAS_ALPHA,
                    eta=FOGAS_ETA,
                    rho=FOGAS_RHO,
                    D_theta=FOGAS_D_THETA,
                    T=FOGAS_T,
                    tqdm_print=False,
                )
                evaluator = FOGASEvaluator(solver=solver, mdp=mdp, planner=planner)
                for mode in ("greedy", "solver"):
                    row.update(evaluate_policy_family(
                        evaluator, dataset, mode, goal, terminal_states, d_star, v_star
                    ))
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)
            finally:
                row["elapsed_seconds"] = time.perf_counter() - start
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
