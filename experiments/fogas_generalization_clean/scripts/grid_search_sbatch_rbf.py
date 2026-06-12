import os
import numpy as np
import random
import torch
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans

# Add project root to sys.path
def find_root(current_path, marker="setup.py"):
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    return current_path

PROJECT_ROOT = find_root(Path.cwd())
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
ASSETS_DIR = PROJECT_ROOT / "experiments" / "shared" / "assets"
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_methods import PolicySolver, EnvDataCollector
from rl_methods.fogas import (
    FOGASSolverVectorized,
    FOGASEvaluator,
)
from rl_methods.dataset_collection import DatasetAnalyzer

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

# --- Helpers: index <-> (row, col) ---
def to_rc(s):  return divmod(s, 10)
def to_s(r, c): return r * 10 + c

def get_norm_coords(s):
    r, c = divmod(int(s), 10)
    return torch.tensor([r / 9.0, c / 9.0], dtype=torch.float64)

# --- Transition dynamics ---
def next_state(s, a):
    if s == goal or s in pits: return s
    r, c = to_rc(s)
    if a == 0:   r2, c2 = max(0, r - 1), c
    elif a == 1: r2, c2 = min(9, r + 1), c
    elif a == 2: r2, c2 = r, max(0, c - 1)
    elif a == 3: r2, c2 = r, min(9, c + 1)
    else: raise ValueError("Invalid action")
    sp = to_s(r2, c2)
    if sp in walls: return s
    return sp

# --- Transition matrix P (N*A x N) ---
P = torch.zeros((N * A, N), dtype=torch.float64)
for x in range(N):
    for a in range(A):
        if x in pits or x == goal:
            P[x * A + a, x] = 1.0
            continue
        r, c = divmod(x, 10)
        if   a == 0: rn, cn = max(0, r-1), c
        elif a == 1: rn, cn = min(9, r+1), c
        elif a == 2: rn, cn = r, max(0, c-1)
        elif a == 3: rn, cn = r, min(9, c+1)
        ns = rn * 10 + cn
        if ns in walls:
            ns = x
        P[x * A + a, ns] = 1.0

# --- Reward function (used instead of omega) ---
def reward_fn(x, a):
    x_int = int(x)
    if x_int == goal:   return  1.0
    if x_int in pits:   return -5.0
    return -0.1

# ============================================================
# RBF Feature Engineering
# ============================================================

def calculate_local_sigma(centers, k=2):
    """Sigma based on k-nearest neighbors (avg distance to 1st NN)."""
    dist_matrix = torch.cdist(centers, centers, p=2)
    # k+1 because the closest point is the center itself (dist=0)
    topk_dists, _ = torch.topk(dist_matrix, k + 1, largest=False, dim=1)
    nearest_neighbor_dists = topk_dists[:, 1]
    return torch.mean(nearest_neighbor_dists)

def get_anchored_centers(num_free_centers=35, anchor_states=None):
    """
    Free centers  → K-Means on non-wall, non-anchor states.
    Anchor centers → placed exactly at goal/pit states.
    Returns (all_centers, n_free, n_anchors).
    """
    anchor_states = set(anchor_states or [])
    valid_coords = []
    for s in range(100):
        if s not in anchor_states:
            r, c = divmod(s, 10)
            valid_coords.append([r / 9.0, c / 9.0])
    kmeans = KMeans(n_clusters=num_free_centers, n_init=10, random_state=42).fit(valid_coords)
    free_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float64)

    anchor_coords = [[divmod(s, 10)[0] / 9.0, divmod(s, 10)[1] / 9.0]
                     for s in sorted(anchor_states)]
    anchor_centers = torch.tensor(anchor_coords, dtype=torch.float64)

    all_centers = torch.cat([free_centers, anchor_centers], dim=0)
    return all_centers, len(free_centers), len(anchor_centers)

# --- RBF Hyperparameters ---
num_free_centers = 35
anchor_states    = {goal} | pits
sigma_scale      = 0.25          # anchor sigma = global_sigma * scale

centers, n_free, n_anchors = get_anchored_centers(
    num_free_centers=num_free_centers,
    anchor_states=anchor_states,
)

sigma_global = calculate_local_sigma(centers, k=2)
sigma_anchor = sigma_global * sigma_scale

# Per-center sigma vector: shape (K,)
sigmas = torch.full((len(centers),), sigma_global, dtype=torch.float64)
sigmas[n_free:] = sigma_anchor   # sharper sigma for anchor centers

print(f"RBF setup: {n_free} free + {n_anchors} anchors = {len(centers)} total centers")
print(f"  sigma_global = {sigma_global:.4f},  sigma_anchor = {sigma_anchor:.4f}  (scale={sigma_scale})")

# --- Feature maps ---
def phi_state(x):
    """
    Pure RBF features with per-center sigma.
      Free centers   → sigma_global  (smooth spatial generalization)
      Anchor centers → sigma_anchor  (sharp, localized reward isolation)
    Includes a bias term.
    """
    coords  = get_norm_coords(x)
    dist_sq = torch.sum((coords - centers) ** 2, dim=1)      # (K,)
    rbf     = torch.exp(-dist_sq / (2 * sigmas ** 2))         # (K,)
    return torch.cat([rbf, torch.ones(1, dtype=torch.float64)])  # (K+1,) with bias

def phi(x, a):
    """Coupled Feature Map: phi(x, a) = e_a ⊗ phi_state(x)."""
    s_feat      = phi_state(x)
    e_a         = torch.zeros(A, dtype=torch.float64)
    e_a[int(a)] = 1.0
    return torch.kron(e_a, s_feat)

d = int(phi(states[0], actions[0]).shape[0])
print(f"  Feature dimension d = A * (K + 1) = {A} * ({len(centers)} + 1) = {d}")

# --- Build MDP ---
# NOTE: We pass reward_fn (not omega) so FOGASSolverVectorized will
# automatically estimate omega via ridge regression from the dataset.
mdp = PolicySolver(
    states=states, actions=actions, phi=phi,
    reward_fn=reward_fn, gamma=gamma, x0=x_0, P=P,
)

# ============================================================
# Grid Search Configuration
# ============================================================
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
    'T': 13000          # ← bumped from 12000 to match notebook tuned value
}

# --- Precompute exact suboptimality metrics ---
_mu_star      = mdp.mu_star.cpu()
_mu_star_norm = _mu_star / (_mu_star.sum() + 1e-300)
_d_star       = _mu_star_norm.reshape(N, A).sum(dim=1)

_v_star      = mdp.v_star.cpu()
_q_star_flat = mdp.q_star.cpu().reshape(-1)

beta_val = 1e-4
temp_dir = "temp_grid_search_rbf"
os.makedirs(temp_dir, exist_ok=True)

extra_steps_values = [0, 1, 3]   # 3 = known-working value from notebook

results = []
total_iters = (
    len(dataset_sizes)
    * len(epsilon_values)
    * len(proportion_configs)
    * len(reset_configs)
    * len(extra_steps_values)
)

print(f"\n🚀 Starting RBF Grid Search ({total_iters} scenarios)...")
print(f"   Computing 5 metrics per scenario:")
print(f"   1. Coverage Ratio")
print(f"   2. Convergence (1 = reached goal, 0 = did not)")
print(f"   3. Final Reward")
print(f"   4. V Optimal Gap (E_{{s~d_π*}}[V*(s) - V^π(s)])")
print(f"   5. Q Optimal Gap (E_{{(x,a)~μ_π*}}[Q*(x,a) - Q^π(x,a)])\n")
print(f"   NOTE: omega is estimated from data (reward_fn-based MDP)\n")

with tqdm(total=total_iters, desc="Grid Searching") as pbar:
    for reset_label, reset_probs in reset_configs.items():
        collector = EnvDataCollector(
            mdp=mdp,
            env_name="10grid_wall",
            restricted_states=list(walls),
            terminal_states=(list(pits) + [goal]),
            reset_probs=reset_probs,
            max_steps=50,
            seed=seed
        )

        for extra_steps in extra_steps_values:
            for prop_label, (props, prop_name) in proportion_configs.items():
                for eps in epsilon_values:
                    epsilon_policy = (mdp.pi_star, eps)

                    for n_steps in dataset_sizes:
                        fname = f"gs_rbf_{reset_label}_extra{extra_steps}_{prop_label}_eps{eps}_n{n_steps}.csv"
                        save_path = os.path.join(temp_dir, fname)

                        # A. Collect Dataset (terminal-aware mixed)
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

                        # B. Analyze Feature Coverage
                        try:
                            analyzer = DatasetAnalyzer(save_path)
                            coverage_ratio = analyzer.feature_coverage_ratio(
                                mdp=mdp, beta=beta_val, use_optimal_policy=True, verbose=False
                            )
                        except Exception:
                            coverage_ratio = np.nan

                        # C. Train FOGAS and Compute Metrics
                        try:
                            temp_solver = FOGASSolverVectorized(
                                mdp=mdp, csv_path=save_path, device=device,
                                beta=beta_val, seed=seed
                            )
                            temp_solver.run(
                                alpha=FOGAS_PARAMS['alpha'],
                                eta=FOGAS_PARAMS['eta'],
                                rho=FOGAS_PARAMS['rho'],
                                T=FOGAS_PARAMS['T'],
                                tqdm_print=False
                            )

                            temp_eval = FOGASEvaluator(temp_solver)

                            # Metric 1: Convergence
                            try:
                                traj = temp_eval.simulate_trajectory(
                                    pi=None, max_steps=200, seed=42, goal_state=99
                                )
                                convergence = int(traj[-1]['next_state'] == 99)
                            except Exception:
                                convergence = np.nan

                            # Metric 2: Final Reward
                            final_reward = temp_eval.final_reward()

                            # Metrics 3 & 4: Exact Suboptimality Gaps
                            v_pi, q_pi = mdp.evaluate_policy(temp_solver.pi)
                            v_pi_cpu       = v_pi.cpu()
                            q_pi_flat_cpu  = q_pi.reshape(-1).cpu()

                            v_gap = (_d_star * (_v_star - v_pi_cpu)).sum().item()
                            q_gap = (_mu_star_norm * (_q_star_flat - q_pi_flat_cpu)).sum().item()

                        except Exception as e:
                            print(f"\n⚠️  FOGAS training/evaluation failed: {e}")
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
output_filename = 'grid_search_results_sbatch_rbf.csv'
output_path = RESULTS_DIR / "grids" / output_filename
df_results.to_csv(output_path, index=False)

print("\n✅ RBF Grid Search Complete!")
print(f"   Total scenarios: {len(df_results)}")
print(f"   Successful runs: {df_results['Convergence'].notna().sum()}")
print(f"   Converged runs:  {int(df_results['Convergence'].sum(skipna=True))}")
print(f"   Failed runs:     {df_results['Convergence'].isna().sum()}\n")
print(f"✅ Results saved to '{output_path}'")
