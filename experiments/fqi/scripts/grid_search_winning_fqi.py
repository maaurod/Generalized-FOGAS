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
N, A = 100, 4
gamma = 0.9
x_0 = 0
goal = 99
pits = {18, 32, 57, 61, 75}
walls = {
    4, 11, 14, 17, 21, 22, 27, 34, 37,
    40, 42, 43, 44, 45, 46, 47, 49,
    54, 62, 64, 66, 72, 76, 82, 84, 86, 87, 94
}

# --- RBF Feature Helpers ---
def get_norm_coords(s):
    r, c = divmod(int(s), 10)
    return torch.tensor([r / 9.0, c / 9.0], dtype=torch.float64)

def calculate_local_sigma(centers, k=2):
    dist_matrix = torch.cdist(centers, centers, p=2)
    topk_dists, _ = torch.topk(dist_matrix, k + 1, largest=False, dim=1)
    nearest_neighbor_dists = topk_dists[:, 1]
    return torch.mean(nearest_neighbor_dists)

def reward_fn(x, a):
    x_int = int(x)
    r, c = divmod(x_int, 10)
    if   a == 0: r_next, c_next = max(0, r-1), c
    elif a == 1: r_next, c_next = min(9, r+1), c
    elif a == 2: r_next, c_next = r, max(0, c-1)
    elif a == 3: r_next, c_next = r, min(9, c+1)
    next_s = r_next * 10 + c_next
    if next_s in walls: next_s = x_int
    if next_s == goal: return 1.0
    if next_s in pits: return -5.0
    return -0.1

P = torch.zeros((N * A, N), dtype=torch.float64)
for x in range(N):
    for a in range(A):
        if x in pits or x == goal:
            P[x * A + a, x] = 1.0
            continue
        r, c = divmod(x, 10)
        if   a == 0: r_next, c_next = max(0, r-1), c
        elif a == 1: r_next, c_next = min(9, r+1), c
        elif a == 2: r_next, c_next = r, max(0, c-1)
        elif a == 3: r_next, c_next = r, min(9, c+1)
        next_state_val = r_next * 10 + c_next
        if next_state_val in walls: next_state_val = x
        P[x * A + a, next_state_val] = 1.0

# --- Reconstruct WINNING features ---
winning_n = 71
winning_sigma_mult = 0.1

valid_coords = [[r / 9.0, c / 9.0] for r in range(10) for c in range(10)]
kmeans = KMeans(n_clusters=winning_n, n_init=10, random_state=42).fit(valid_coords)
centers_win = torch.tensor(kmeans.cluster_centers_, dtype=torch.float64)
sigma_win = calculate_local_sigma(centers_win, k=2) * winning_sigma_mult

def phi_win(x, a):
    coords = get_norm_coords(x)
    dist_sq = torch.sum((coords - centers_win.to(coords.device))**2, dim=1)
    rbf = torch.exp(-dist_sq / (2 * sigma_win**2))
    total_act = torch.sum(rbf)
    if total_act > 1e-12: rbf = rbf / total_act
    e_a = torch.zeros(A, dtype=torch.float64, device=rbf.device)
    e_a[int(a)] = 1.0
    return torch.kron(e_a, rbf)

# --- Oracle solving ---
print("Precomputing Oracle RBF MDP...")
mdp = PolicySolver(
    states=states, actions=actions, phi=phi_win,
    reward_fn=reward_fn, gamma=gamma, x0=x_0, P=P,
)

# For metrics calculation
_mu_star = mdp.mu_star.cpu()
_mu_star_norm = _mu_star / (_mu_star.sum() + 1e-300)
_d_star = _mu_star_norm.reshape(N, A).sum(dim=1)
_v_star = mdp.v_star.cpu()
_q_star_flat = mdp.q_star.cpu().reshape(-1)

# ============================================================
# Grid Search Configuration (Matching the sbatch script)
# ============================================================
FQI_PARAMS = {'K': 4000, 'tau': 0.2, 'ridge': 1e-6}
beta_val = 1e-4

dataset_sizes  = [4000, 8000, 12000, 16000, 20000]
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

temp_dir = "temp_grid_winning"
os.makedirs(temp_dir, exist_ok=True)
results = []
total_iters = len(reset_configs) * len(extra_steps_values) * len(proportion_configs) * len(epsilon_values) * len(dataset_sizes)

print(f"🚀 Starting WINNING Grid Search ({total_iters} scenarios)...")

with tqdm(total=total_iters, desc="Grid Searching") as pbar:
    for reset_label, reset_probs in reset_configs.items():
        collector = EnvDataCollector(
            mdp=mdp, env_name="10grid_wall", restricted_states=list(walls),
            terminal_states=(list(pits) + [goal]), reset_probs=reset_probs,
            max_steps=50, seed=seed,
        )
        for extra_steps in extra_steps_values:
            for prop_label, (props, prop_name) in proportion_configs.items():
                for eps in epsilon_values:
                    epsilon_policy = (mdp.pi_star, eps)
                    for n_steps in dataset_sizes:
                        fname = f"win_{reset_label}_x{extra_steps}_{prop_label}_e{eps}_n{n_steps}.csv"
                        save_path = os.path.join(temp_dir, fname)

                        # A. Collect
                        collector.collect_mixed_dataset_terminal_aware(
                            policies=[epsilon_policy, "random"], proportions=props,
                            n_steps=n_steps, episode_based=True, save_path=save_path,
                            verbose=False, extra_steps=extra_steps
                        )
                        
                        # B. Coverage
                        try:
                            analyzer = DatasetAnalyzer(save_path)
                            coverage_ratio = analyzer.feature_coverage_ratio(mdp=mdp, beta=beta_val, use_optimal_policy=True)
                        except: coverage_ratio = np.nan

                        # C. Train FQI
                        try:
                            solver_fqi = FQISolver(mdp=mdp, csv_path=save_path, device=device, seed=seed, ridge=FQI_PARAMS['ridge'])
                            solver_fqi.run(K=FQI_PARAMS['K'], tau=FQI_PARAMS['tau'], verbose=False)
                            evaluator_fqi = FQIEvaluator(solver_fqi)

                            traj = evaluator_fqi.simulate_trajectory(max_steps=100, seed=42)
                            convergence = int(traj[-1][3] == 99)
                            final_reward = evaluator_fqi.final_reward()
                            
                            v_pi, q_pi = mdp.evaluate_policy(solver_fqi.pi)
                            v_gap = (_d_star * (_v_star - v_pi.cpu())).sum().item()
                            q_gap = (_mu_star_norm * (_q_star_flat - q_pi.reshape(-1).cpu())).sum().item()
                        except:
                            convergence = final_reward = v_gap = q_gap = np.nan

                        results.append({
                            "Dataset Size": n_steps, "Epsilon": eps, "Proportions": prop_name,
                            "Init Mode": reset_label, "Extra Steps": extra_steps,
                            "Coverage Ratio": coverage_ratio, "Convergence": convergence,
                            "Final Reward": final_reward, "V Optimal Gap": v_gap, "Q Optimal Gap": q_gap,
                        })
                        if os.path.exists(save_path): os.remove(save_path)
                        pbar.update(1)

# ============================================================
# Uniform Dataset Experiments
# ============================================================
uniform_samples_values = [1, 3, 5, 10, 20]
print(f"\n🚀 Starting Uniform Dataset Search ({len(uniform_samples_values)} scenarios)...")

collector_uniform = EnvDataCollector(
    mdp=mdp,
    env_name="10grid_rbf",
    restricted_states=list(walls),
    terminal_states=(list(pits) + [goal]),
    max_steps=50,
    seed=seed,
)

with tqdm(total=len(uniform_samples_values), desc="Uniform Searching") as pbar:
    for samples_per_pair in uniform_samples_values:
        fname = f"win_uniform_s{samples_per_pair}.csv"
        save_path = os.path.join(temp_dir, fname)

        # A. Collect
        collector_uniform.collect_uniform_dataset(
            samples_per_pair=samples_per_pair, 
            save_path=save_path, 
            verbose=False
        )
        
        try:
            actual_n_steps = len(pd.read_csv(save_path))
        except:
            actual_n_steps = 400 * samples_per_pair

        # B. Coverage
        try:
            analyzer = DatasetAnalyzer(save_path)
            coverage_ratio = analyzer.feature_coverage_ratio(mdp=mdp, beta=beta_val, use_optimal_policy=True)
        except: coverage_ratio = np.nan

        # C. Train FQI
        try:
            solver_fqi = FQISolver(mdp=mdp, csv_path=save_path, device=device, seed=seed, ridge=FQI_PARAMS['ridge'])
            solver_fqi.run(K=FQI_PARAMS['K'], tau=FQI_PARAMS['tau'], verbose=False)
            evaluator_fqi = FQIEvaluator(solver_fqi)

            traj = evaluator_fqi.simulate_trajectory(max_steps=100, seed=42)
            convergence = int(traj[-1][3] == 99)
            final_reward = evaluator_fqi.final_reward()
            
            v_pi, q_pi = mdp.evaluate_policy(solver_fqi.pi)
            v_gap = (_d_star * (_v_star - v_pi.cpu())).sum().item()
            q_gap = (_mu_star_norm * (_q_star_flat - q_pi.reshape(-1).cpu())).sum().item()
        except:
            convergence = final_reward = v_gap = q_gap = np.nan

        results.append({
            "Dataset Size": actual_n_steps, "Epsilon": np.nan, "Proportions": "Uniform",
            "Init Mode": f"uniform_{samples_per_pair}", "Extra Steps": np.nan,
            "Coverage Ratio": coverage_ratio, "Convergence": convergence,
            "Final Reward": final_reward, "V Optimal Gap": v_gap, "Q Optimal Gap": q_gap,
        })
        if os.path.exists(save_path): os.remove(save_path)
        pbar.update(1)
df_results = pd.DataFrame(results)
output_filename = "grid_search_winning_fqi_results.csv"
output_path = RESULTS_DIR / "grids" / output_filename
df_results.to_csv(output_path, index=False)
if not os.listdir(temp_dir): os.rmdir(temp_dir)

print(f"\n✅ Complete! Results saved to '{output_path}'")
