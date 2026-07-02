import os
import numpy as np
import random
import torch
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
def find_root(current_path):
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / "src" / "rl_methods").exists() and (parent / "data").exists():
            return parent
    return current_path

PROJECT_ROOT = find_root(Path.cwd())
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
ASSETS_DIR = PROJECT_ROOT / "experiments" / "shared" / "assets"
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_methods import PolicySolver, EnvDataCollector
from rl_methods.fqi.fqi_solver import FQISolver
from rl_methods.fqi.fqi_evaluator import FQIEvaluator
from rl_methods.dataset_collection import DatasetAnalyzer

# --- Seeds ---
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# --- MDP Initialization (10x10 Gridworld) ---
states  = torch.arange(100, dtype=torch.int64)
actions = torch.arange(4, dtype=torch.int64)
N = len(states)
A = len(actions)
gamma = 0.9
x_0 = 0
goal = 99
pits = {18, 32, 57, 61, 75}
walls = {
    4, 11, 14, 17, 21, 22, 27, 34, 37,
    40, 42, 43, 44, 45, 46, 47, 49,
    54, 62, 64, 66, 72, 76, 82, 84, 86, 87, 94
}

# --- Tabular feature map ---
def phi(x, a):
    vec = torch.zeros(N * A, dtype=torch.float64)
    vec[int(x) * A + int(a)] = 1.0
    return vec

# --- Reward weights (tabular, exact) ---
step_cost   = -0.1
goal_reward =  1.0
pit_reward  = -5.0

omega = torch.full((N * A,), step_cost, dtype=torch.float64)
omega[goal * A : goal * A + A] = goal_reward
for p in pits:
    omega[p * A : p * A + A] = pit_reward

# --- Helpers ---
def to_rc(s):  return divmod(s, 10)
def to_s(r, c): return r * 10 + c

def next_state(s, a):
    if s == goal or s in pits: return s
    r, c = to_rc(s)
    if   a == 0: r2, c2 = max(0, r - 1), c
    elif a == 1: r2, c2 = min(9, r + 1), c
    elif a == 2: r2, c2 = r, max(0, c - 1)
    elif a == 3: r2, c2 = r, min(9, c + 1)
    else: raise ValueError("Invalid action")
    sp = to_s(r2, c2)
    if sp in walls: return s
    return sp

def psi(xp):
    v = torch.zeros(N * A, dtype=torch.float64)
    for x in states:
        for a in actions:
            if next_state(int(x), int(a)) == xp:
                v[int(x) * A + int(a)] = 1.0
    return v

# --- Build MDP ---
mdp = PolicySolver(
    states=states, actions=actions, phi=phi, omega=omega,
    gamma=gamma, x0=x_0, psi=psi
)

# --- Precompute exact suboptimality metrics ---
_mu_star      = mdp.mu_star.cpu()
_mu_star_norm = _mu_star / (_mu_star.sum() + 1e-300)
_d_star       = _mu_star_norm.reshape(N, A).sum(dim=1)
_v_star       = mdp.v_star.cpu()
_q_star_flat  = mdp.q_star.cpu().reshape(-1)

# ============================================================
# Grid Search Configuration
# ============================================================

# Fixed FQI hyperparameters
FQI_PARAMS = {
    'K':     5000,
    'tau':   0.1,
    'ridge': 1e-2,
}

dataset_sizes  = [4000, 8000, 12000, 16000, 20000]   # ← aligned with FOGAS
epsilon_values = [0.1, 0.3, 0.5, 0.7, 0.9]
extra_steps_values = [0, 1]
proportion_configs = {
    "100/0": ([1.0, 0.0], "100% Eps-Greedy"),
    "80/20": ([0.8, 0.2], "80% Eps-Greedy / 20% Random"),
    "60/40": ([0.6, 0.4], "60% Eps-Greedy / 40% Random"),
    "40/60": ([0.4, 0.6], "40% Eps-Greedy / 60% Random"),
    "20/80": ([0.2, 0.8], "20% Eps-Greedy / 80% Random"),
    "0/100": ([0.0, 1.0], "0% Eps-Greedy / 100% Random"),
}

reset_configs = {
    "100/0": {'x0': 1.0},
    "80/20": {'random': 0.2, 'x0': 0.8},
    "50/50": {'custom': 0.5, 'x0': 0.5},
    "20/80": {'random': 0.8, 'x0': 0.2},
    "0/100": {'random': 1.0, 'x0': 0.0},
}

beta_val = 1e-4          # ← aligned with FOGAS (was 1e-2)
temp_dir = "temp_grid_search_fqi"
os.makedirs(temp_dir, exist_ok=True)

results = []
total_iters = (
    len(reset_configs)
    * len(extra_steps_values)
    * len(proportion_configs)
    * len(epsilon_values)
    * len(dataset_sizes)
)

print(f"🚀 Starting FQI Grid Search ({total_iters} scenarios)...")
print(f"   Sweeping: reset, extra_steps, proportions, epsilon, dataset size")
print(f"   Collection: collect_mixed_dataset_terminal_aware (episode-based)")
print(f"   Fixed FQI: K={FQI_PARAMS['K']}, tau={FQI_PARAMS['tau']}, ridge={FQI_PARAMS['ridge']}")
print(f"   Computing 5 metrics per scenario:")
print(f"   1. Coverage Ratio")
print(f"   2. Convergence (1 = reached goal, 0 = did not)")
print(f"   3. Final Reward")
print(f"   4. V Optimal Gap (E_{{s~d_π*}}[V*(s) - V^π(s)])")
print(f"   5. Q Optimal Gap (E_{{(x,a)~μ_π*}}[Q*(x,a) - Q^π(x,a)])\n")

with tqdm(total=total_iters, desc="FQI Grid Searching") as pbar:
    for reset_label, reset_probs in reset_configs.items():

        # One collector per reset config (avoids re-building LinearMDPEnv repeatedly)
        collector = EnvDataCollector(
            mdp=mdp,
            env_name="10grid_wall",
            restricted_states=list(walls),
            terminal_states=(list(pits) + [goal]),
            reset_probs=reset_probs,
            max_steps=50,
            seed=seed,
        )

        for extra_steps in extra_steps_values:
            for prop_label, (props, prop_name) in proportion_configs.items():
                for eps in epsilon_values:
                    epsilon_policy = (mdp.pi_star, eps)

                    for n_steps in dataset_sizes:
                        fname = (
                            f"fqi_{reset_label}_extra{extra_steps}"
                            f"_{prop_label}_eps{eps}_n{n_steps}.csv"
                        )
                        save_path = os.path.join(temp_dir, fname)

                        # ── A. Collect Dataset (terminal-aware mixed) ──────────
                        try:
                            collector.collect_mixed_dataset_terminal_aware(
                                policies=[epsilon_policy, "random"],
                                proportions=props,
                                n_steps=n_steps,
                                episode_based=True,
                                save_path=save_path,
                                verbose=False,
                                extra_steps=extra_steps,
                            )
                        except Exception as e:
                            print(f"\n⚠️  Dataset collection failed: {e}")
                            pbar.update(1)
                            continue

                        # ── B. Feature Coverage ────────────────────────────────
                        try:
                            analyzer = DatasetAnalyzer(save_path)
                            coverage_ratio = analyzer.feature_coverage_ratio(
                                mdp=mdp, beta=beta_val,
                                use_optimal_policy=True, verbose=False
                            )
                        except Exception:
                            coverage_ratio = np.nan

                        # ── C. Train FQI + Compute Metrics ────────────────────
                        try:
                            solver_fqi = FQISolver(
                                mdp=mdp,
                                csv_path=save_path,
                                device=device,
                                seed=seed,
                                ridge=FQI_PARAMS['ridge'],
                            )
                            solver_fqi.run(
                                K=FQI_PARAMS['K'],
                                tau=FQI_PARAMS['tau'],
                                verbose=False,
                            )
                            evaluator_fqi = FQIEvaluator(solver_fqi)

                            # Metric 1: Convergence — greedy path reaches state 99?
                            try:
                                traj = evaluator_fqi.simulate_trajectory(
                                    max_steps=200, seed=42
                                )
                                convergence = int(traj[-1][3] == 99)  # traj entries: (s,a,r,s')
                            except Exception:
                                convergence = np.nan

                            # Metric 2: Final Reward
                            final_reward = evaluator_fqi.final_reward()

                            # Metrics 3 & 4: Exact Suboptimality Gaps
                            v_pi, q_pi = mdp.evaluate_policy(solver_fqi.pi)
                            v_pi_cpu      = v_pi.cpu()
                            q_pi_flat_cpu = q_pi.reshape(-1).cpu()

                            v_gap = (_d_star * (_v_star - v_pi_cpu)).sum().item()
                            q_gap = (_mu_star_norm * (_q_star_flat - q_pi_flat_cpu)).sum().item()

                        except Exception as e:
                            print(f"\n⚠️  FQI training/evaluation failed: {e}")
                            convergence  = np.nan
                            final_reward = np.nan
                            v_gap        = np.nan
                            q_gap        = np.nan

                        results.append({
                            "Dataset Size":   n_steps,
                            "Epsilon":        eps,
                            "Proportions":    prop_name,
                            "Init Mode":      reset_label,
                            "Extra Steps":    extra_steps,
                            "Coverage Ratio": coverage_ratio,
                            "Log Coverage":   np.log10(coverage_ratio) if (not np.isnan(coverage_ratio) and coverage_ratio > 0) else np.nan,
                            "Convergence":    convergence,
                            "Final Reward":   final_reward,
                            "V Optimal Gap":  v_gap,
                            "Q Optimal Gap":  q_gap,
                        })

                        if os.path.exists(save_path):
                            os.remove(save_path)
                        pbar.update(1)

# Cleanup
if os.path.exists(temp_dir):
    try:
        os.rmdir(temp_dir)
    except Exception:
        pass

df_results = pd.DataFrame(results)
output_filename = "grid_search_results_sbatch_fqi.csv"
output_path = RESULTS_DIR / "grids" / output_filename
df_results.to_csv(output_path, index=False)

print("\n✅ FQI Grid Search Complete!")
print(f"   Total scenarios:  {len(df_results)}")
print(f"   Successful runs:  {df_results['Convergence'].notna().sum()}")
print(f"   Converged runs:   {int(df_results['Convergence'].sum(skipna=True))}")
print(f"   Failed runs:      {df_results['Convergence'].isna().sum()}\n")
print(f"✅ Results saved to '{output_path}'")
