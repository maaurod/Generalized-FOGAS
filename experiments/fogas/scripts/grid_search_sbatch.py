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

from rl_methods import EnvDataCollector
from rl_methods.mdp import DiscreteMDP, Planner
from rl_methods.fogas import (
    FOGASSolver,
    FOGASEvaluator,
)
from rl_methods.data_collection import DatasetAnalyzer

# --- Setup Parameters ---
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

def phi(x, a):
    vec = torch.zeros(N * A, dtype=torch.float64)
    vec[int(x) * A + int(a)] = 1.0
    return vec

step_cost = -0.1
goal_reward = 1.0
pit_reward  = -5.0
omega = torch.full((N * A,), step_cost, dtype=torch.float64)
omega[goal * A : goal * A + A] = goal_reward
for p in pits:
    omega[p * A : p * A + A] = pit_reward

def to_rc(s):  return divmod(s, 10)
def to_s(r, c): return r * 10 + c

def next_state(s, a):
    if s == goal or s in pits: return s
    r, c = to_rc(s)
    if a == 0: r2, c2 = max(0, r - 1), c
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

mdp = DiscreteMDP(
    states=states, actions=actions, omega=omega,
    gamma=gamma, x0=x_0, psi=psi, phi=phi
)
planner = Planner(mdp)

Phi = torch.vstack([
    phi(int(s), int(a)).to(dtype=torch.float64)
    for s in states
    for a in actions
])

# --- Grid Search Config ---
dataset_sizes = [4000, 8000, 12000, 16000, 20000]
epsilon_values = [0.1, 0.3, 0.5, 0.7, 0.9]

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
    "20/80": {'random': 0.8, 'x0': 0.2},
    "50/50": {'custom': 0.5, 'x0': 0.5},
    "80/20": {'random': 0.2, 'x0': 0.8},
    "0/100": {'random': 1.0, 'x0': 0.0}
}


FOGAS_PARAMS = {
    'alpha': 0.001 / 2,
    'eta': 0.0002,
    'rho': 0.05,
    'T': 12000
}

# Precompute constants for exact suboptimality metrics
_mu_star = planner.mu_star.cpu()                       # shape (N*A,)
_mu_star_norm = _mu_star / (_mu_star.sum() + 1e-300)  # Total discounted occupancy
_d_star = _mu_star_norm.reshape(N, A).sum(dim=1)       # State occupancy d_π*(s)

_v_star = planner.v_star.cpu()                         # shape (N,)
_q_star_flat = planner.q_star.cpu().reshape(-1)        # shape (N*A,)

beta_val = 1e-4
temp_dir = "temp_grid_search"
os.makedirs(temp_dir, exist_ok=True)

results = []
total_iters = len(dataset_sizes) * len(epsilon_values) * len(proportion_configs) * len(reset_configs)

print(f"🚀 Starting Extended Grid Search ({total_iters} scenarios)...")
print(f"   Computing 5 metrics per scenario:")
print(f"   1. Coverage Ratio")
print(f"   2. Convergence (1 = reached goal, 0 = did not)")
print(f"   3. Final Reward")
print(f"   4. V Optimal Gap (E_{{s~d_π*}}[V*(s) - V^π(s)])")
print(f"   5. Q Optimal Gap (E_{{(x,a)~μ_π*}}[Q*(x,a) - Q^π(x,a)])\n")

with tqdm(total=total_iters, desc="Grid Searching") as pbar:
    for reset_label, reset_probs in reset_configs.items():
        collector = EnvDataCollector(
            mdp=planner,
            env_name="10grid_wall",
            restricted_states=list(walls),
            terminal_states=(list(pits) + [goal]),
            reset_probs=reset_probs,
            max_steps=50,
            seed=seed
        )
        
        for prop_label, (props, prop_name) in proportion_configs.items():
            for eps in epsilon_values:
                epsilon_policy = (planner.pi_star, eps)
                
                for n_steps in dataset_sizes:
                    fname = f"gs_{reset_label}_{prop_label}_eps{eps}_n{n_steps}.csv"
                    save_path = os.path.join(temp_dir, fname)
                    
                    # A. Collect Dataset
                    try:
                        collector.collect_mixed_dataset(
                            policies=[epsilon_policy, "random"],
                            proportions=props,
                            n_steps=n_steps,
                            episode_based=True,
                            save_path=save_path,
                            verbose=False
                        )
                    except Exception as e:
                        print(f"\n⚠️  Dataset collection failed: {e}")
                        pbar.update(1)
                        continue
                    
                    # B. Analyze Feature Coverage
                    try:
                        analyzer = DatasetAnalyzer(save_path)
                        coverage_ratio = analyzer.feature_coverage(
                            phi=Phi,
                            optimal_occupancy=planner.mu_star,
                            beta=beta_val,
                            n_states=mdp.N,
                            n_actions=mdp.A,
                        )
                    except:
                        coverage_ratio = np.nan
                        
                    # C. Train FOGAS and Compute Metrics
                    try:
                        temp_solver = FOGASSolver(
                            mdp=mdp, phi=phi, csv_path=save_path, device=device, 
                            beta=beta_val, seed=seed
                        )
                        temp_solver.run(
                            alpha=FOGAS_PARAMS['alpha'], 
                            eta=FOGAS_PARAMS['eta'], 
                            rho=FOGAS_PARAMS['rho'], 
                            T=FOGAS_PARAMS['T'], 
                            tqdm_print=False
                        )
                        
                        temp_eval = FOGASEvaluator(temp_solver, planner=planner)
                        
                        # Metric 1: Convergence — does the greedy path reach state 99?
                        try:
                            traj = temp_eval.simulate_trajectory(
                                policy_mode="greedy", max_steps=200, seed=42, goal_state=99
                            )
                            convergence = int(traj[-1]['next_state'] == 99)
                        except Exception:
                            convergence = np.nan
                        
                        # Metric 2: Final Reward
                        final_reward = temp_eval.average_return(
                            policy_mode="solver",
                            num_trajectories=10,
                            max_steps=200,
                            seed=42,
                            goal_state=99,
                        )["policy"]
                        
                        # Metric 3 & 4: Exact Suboptimality Gaps
                        v_pi, q_pi = planner.evaluate_policy(temp_solver.pi)
                        v_pi_cpu = v_pi.cpu()
                        q_pi_flat_cpu = q_pi.reshape(-1).cpu()
                        
                        v_gap = (_d_star * (_v_star - v_pi_cpu)).sum().item()
                        q_gap = (_mu_star_norm * (_q_star_flat - q_pi_flat_cpu)).sum().item()
                        
                    except Exception as e:
                        print(f"\n⚠️  FOGAS training/evaluation failed: {e}")
                        convergence = np.nan
                        final_reward = np.nan
                        v_gap = np.nan
                        q_gap = np.nan
                    
                    results.append({
                        "Dataset Size": n_steps,
                        "Epsilon": eps,
                        "Proportions": prop_name,
                        "Init Mode": reset_label,
                        "Coverage Ratio": coverage_ratio,
                        "Log Coverage": np.log10(coverage_ratio) if (not np.isnan(coverage_ratio) and coverage_ratio > 0) else np.nan,
                        "Convergence": convergence,
                        "Final Reward": final_reward,
                        "V Optimal Gap": v_gap,
                        "Q Optimal Gap": q_gap,
                    })
                    
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    pbar.update(1)

# Cleanup
if os.path.exists(temp_dir):
    try:
        os.rmdir(temp_dir)
    except:
        pass

df_results = pd.DataFrame(results)  
output_filename = 'grid_search_results_sbatch.csv'
output_path = RESULTS_DIR / "grids" / output_filename
df_results.to_csv(output_path, index=False)

print("\n✅ Grid Search Complete!")
print(f"   Total scenarios: {len(df_results)}")
print(f"   Successful runs: {df_results['Convergence'].notna().sum()}")
print(f"   Converged runs:  {int(df_results['Convergence'].sum(skipna=True))}")
print(f"   Failed runs:     {df_results['Convergence'].isna().sum()}\n")
print(f"✅ Results saved to '{output_path}'")
