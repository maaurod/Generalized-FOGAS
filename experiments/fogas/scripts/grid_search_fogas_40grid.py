"""
FOGAS Hyperparameter Grid Search — 40×40 Grid
==============================================
Sweeps alpha and eta (multiply / divide the base values by
100, 50, 20, 10, 5, 2, 1, 1/2, 1/5, 1/10, 1/20, 1/50, 1/100)
keeping only pairs where alpha > eta  (algorithmic requirement).

Outputs
-------
  grid_search_results_fogas_40grid.csv   — metrics for every valid pair
  grid_search_paths_fogas_40grid.pt      — full path tensors (torch dict)
"""

import os
import sys
import json
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")          # no display needed on a cluster node
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────────────────────
# Project root discovery
# ─────────────────────────────────────────────────────────────
def find_root(current_path):
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / "src" / "rl_methods").exists() and (parent / "data").exists():
            return parent
    return current_path

PROJECT_ROOT = find_root(Path(__file__).resolve())
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_methods.mdp import DiscreteMDP, Planner
from rl_methods.fogas import FOGASSolver, FOGASEvaluator

# ─────────────────────────────────────────────────────────────
# Seeds & device
# ─────────────────────────────────────────────────────────────
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ─────────────────────────────────────────────────────────────
# Grid layout (20×20 base)
# ─────────────────────────────────────────────────────────────
grid_layout = [
    "S...#....#...P...#..",
    ".#.P.......#...#....",
    ".#................P.",
    ".#####..P.###.......",
    ".....#........####..",
    ".###.#..#.P.........",
    ".P.#.#...#....##..P.",
    "...#.#######...#....",
    "...#.......#...P.#..",
    "...#######.#........",
    "P........#.#........",
    "..##..P..#.######.P.",
    "....#....#......#...",
    ".P.......######.#...",
    "...##.P.......#.#...",
    "#.......##....#.###P",
    "...P..........#....#",
    "..####....#...####.#",
    "......P..........#.#",
    "P........#....P..#.G"
]

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def scale_grid(small_grid, factor=2):
    large_grid = np.repeat(np.repeat(small_grid, factor, axis=0), factor, axis=1)
    large_grid[large_grid == "S"] = "."
    large_grid[large_grid == "G"] = "."
    large_grid[0, 0] = "S"
    large_grid[-1, -1] = "G"
    return large_grid


def calculate_local_sigma(centers, k=2):
    dist_matrix = torch.cdist(centers, centers, p=2)
    topk_dists, _ = torch.topk(dist_matrix, k + 1, largest=False, dim=1)
    return torch.mean(topk_dists[:, 1])


def get_phi_state_func(centers, sigma, size, pits, goal):
    def phi_state(x):
        r, c = divmod(int(x), size)
        coords = torch.tensor(
            [r / (size - 1), c / (size - 1)], dtype=torch.float64
        )
        dist_sq = torch.sum((coords - centers) ** 2, dim=1)
        rbf = torch.exp(-dist_sq / (2 * sigma ** 2))
        is_pit  = 1.0 if int(x) in pits else 0.0
        is_goal = 1.0 if int(x) == goal  else 0.0
        indicators = torch.tensor([is_pit, is_goal], dtype=torch.float64)
        return torch.cat([rbf, torch.ones(1, dtype=torch.float64), indicators])
    return phi_state


def generate_mdp_regular_grid(grid_data, gamma=0.99):
    size = grid_data.shape[0]
    N    = size * size
    A    = 4

    walls      = set(np.where(grid_data.flatten() == "#")[0])
    pits       = set(np.where(grid_data.flatten() == "P")[0])
    goal       = int(np.where(grid_data.flatten() == "G")[0][0])
    start_node = int(np.where(grid_data.flatten() == "S")[0][0])

    target_num_centers = int(N * 0.5)
    centers_per_side   = int(np.sqrt(target_num_centers))
    ticks  = np.linspace(0.0, 1.0, centers_per_side)
    c_x, c_y = np.meshgrid(ticks, ticks)
    centers = torch.tensor(
        np.column_stack([c_x.ravel(), c_y.ravel()]), dtype=torch.float64
    )

    def reward_fn(x, a):
        x_int = int(x)
        if x_int == goal:      return 10.0
        if x_int in pits:      return -10.0
        return -0.01

    P = torch.zeros((N * A, N), dtype=torch.float64)
    for x in range(N):
        for a in range(A):
            if x in pits or x == goal:
                P[x * A + a, x] = 1.0
                continue
            r, c = divmod(x, size)
            if   a == 0: r_n, c_n = max(0, r - 1), c
            elif a == 1: r_n, c_n = min(size - 1, r + 1), c
            elif a == 2: r_n, c_n = r, max(0, c - 1)
            elif a == 3: r_n, c_n = r, min(size - 1, c + 1)
            next_s = r_n * size + c_n
            if next_s in walls:
                next_s = x
            P[x * A + a, next_s] = 1.0

    return {
        "centers": centers, "walls": walls, "pits": pits,
        "goal": goal, "start": start_node, "reward_fn": reward_fn,
        "P": P, "N": N, "A": A, "gamma": gamma,
    }


# ─────────────────────────────────────────────────────────────
# Build grids & MDP
# ─────────────────────────────────────────────────────────────
print("⚙️  Building grids and MDP …")
grid_20    = np.array([list(row) for row in grid_layout])
grid_40    = scale_grid(grid_20, factor=2)
size_20    = grid_20.shape[0]
size_40    = grid_40.shape[0]

mdp_data_20 = generate_mdp_regular_grid(grid_20)
mdp_data_40 = generate_mdp_regular_grid(grid_40)

fixed_centers = mdp_data_20["centers"]
fixed_sigma   = calculate_local_sigma(fixed_centers, k=2)

# 20×20 feature map
phi_s_20 = get_phi_state_func(
    fixed_centers, fixed_sigma, size_20,
    mdp_data_20["pits"], mdp_data_20["goal"]
)
def phi_20_fixed(x, a):
    s_feat = phi_s_20(x)
    e_a    = torch.zeros(4, dtype=torch.float64)
    e_a[int(a)] = 1.0
    return torch.kron(e_a, s_feat)

# 40×40 feature map (same centers, same sigma → scale-invariant)
phi_s_40 = get_phi_state_func(
    fixed_centers, fixed_sigma, size_40,
    mdp_data_40["pits"], mdp_data_40["goal"]
)
def phi_40_fixed(x, a):
    s_feat = phi_s_40(x)
    e_a    = torch.zeros(4, dtype=torch.float64)
    e_a[int(a)] = 1.0
    return torch.kron(e_a, s_feat)

states_40  = torch.arange(size_40 * size_40, dtype=torch.int64)
actions_40 = torch.arange(4,               dtype=torch.int64)

mdp_40 = DiscreteMDP(
    states    = states_40,
    actions   = actions_40,
    reward_fn = mdp_data_40["reward_fn"],
    gamma     = mdp_data_40["gamma"],
    x0        = mdp_data_40["start"],
    P         = mdp_data_40["P"],
)
planner_40 = Planner(mdp_40)
print("✅ MDP built.")

# ─────────────────────────────────────────────────────────────
# Dataset path
# ─────────────────────────────────────────────────────────────
DATASET_PATH = DATASETS_DIR / "40grid.csv"
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
print(f"📂 Dataset: {DATASET_PATH}")

# ─────────────────────────────────────────────────────────────
# Initialise solver & evaluator (reused across all runs)
# ─────────────────────────────────────────────────────────────
solver = FOGASSolver(
    mdp      = mdp_40,
    phi      = phi_40_fixed,
    csv_path = str(DATASET_PATH),
    device   = device,
    beta     = 1e-7,
    seed     = seed,
)
evaluator = FOGASEvaluator(solver, planner=planner_40)

# ─────────────────────────────────────────────────────────────
# Grid search configuration
# ─────────────────────────────────────────────────────────────
BASE_ALPHA = 0.00001
BASE_ETA   = 0.000002
FIXED_RHO  = 0.5
FIXED_T    = 70_000
GOAL_STATE = mdp_data_40["goal"]
MAX_SIM_STEPS = 500

FACTORS = [100, 50, 20, 10, 5, 2, 1, 1/2, 1/5, 1/10, 1/20, 1/50, 1/100]

alpha_candidates = sorted({BASE_ALPHA * f for f in FACTORS})
eta_candidates   = sorted({BASE_ETA   * f for f in FACTORS})

# Keep only pairs with alpha > eta
valid_pairs = [
    (a, e)
    for a, e in itertools.product(alpha_candidates, eta_candidates)
    if a > e
]

print(f"\n📐 alpha candidates ({len(alpha_candidates)}): "
      f"{[f'{v:.2e}' for v in alpha_candidates]}")
print(f"📐 eta   candidates ({len(eta_candidates)}): "
      f"{[f'{v:.2e}' for v in eta_candidates]}")
print(f"🔎 Valid pairs after α > η filter: {len(valid_pairs)}\n")

# ─────────────────────────────────────────────────────────────
# Run grid search
# ─────────────────────────────────────────────────────────────
results      = []
saved_paths  = {}   # (alpha, eta) → list[int]

for run_idx, (alpha_val, eta_val) in enumerate(valid_pairs):
    print(f"\n{'─'*60}")
    print(f"  [{run_idx + 1}/{len(valid_pairs)}]  "
          f"alpha={alpha_val:.2e}  |  eta={eta_val:.2e}  "
          f"(α/η = {alpha_val/eta_val:.1f})")
    print(f"{'─'*60}")

    try:
        solver.run(
            alpha       = alpha_val,
            eta         = eta_val,
            rho         = FIXED_RHO,
            T           = FIXED_T,
            tqdm_print  = True,
        )

        # ── Convergence metric (greedy success rate) ───────────
        metric_fn  = evaluator.get_metric(
            "success_rate",
            goal_state=GOAL_STATE,
            num_trajectories=10,
            max_steps=MAX_SIM_STEPS,
            seed=seed,
        )
        conv_dist  = -metric_fn()

        # ── Final reward ───────────────────────────────────────
        final_rew  = evaluator.average_return(
            policy_mode="solver",
            num_trajectories=10,
            max_steps=MAX_SIM_STEPS,
            seed=seed,
            goal_state=GOAL_STATE,
        )["policy"]

        # ── Simulate trajectory and save path ───────────────────
        trajectory  = evaluator.simulate_trajectory(
            policy_mode = "solver",
            max_steps  = MAX_SIM_STEPS,
            seed       = seed,
            goal_state = GOAL_STATE,
        )
        path_states = (
            [step["state"] for step in trajectory]
            + [trajectory[-1]["next_state"]]
        )
        reached     = bool(trajectory[-1].get("reached_goal", False))
        final_state = int(path_states[-1])

        print(f"  conv_dist={conv_dist:.4f}  |  final_reward={final_rew:.4f}  "
              f"|  reached_goal={reached}  |  final_state={final_state}  "
              f"|  path_len={len(trajectory)}")

    except Exception as exc:
        print(f"  ⚠️  Run failed: {exc}")
        conv_dist   = np.nan
        final_rew   = np.nan
        reached     = False
        path_states = []
        final_state = -1

    key = (float(alpha_val), float(eta_val))
    saved_paths[key] = path_states

    results.append({
        "alpha":         float(alpha_val),
        "eta":           float(eta_val),
        "alpha_eta_ratio": float(alpha_val / eta_val),
        "conv_dist":     conv_dist,
        "final_reward":  final_rew,
        "reached_goal":  reached,
        "final_state":   final_state,
        "path_length":   len(path_states) - 1,   # steps (excluding final)
    })

# ─────────────────────────────────────────────────────────────
# Save CSV results
# ─────────────────────────────────────────────────────────────
df = pd.DataFrame(results).sort_values("conv_dist").reset_index(drop=True)

out_csv = RESULTS_DIR / "grids" / "grid_search_results_fogas_40grid.csv"
df.to_csv(out_csv, index=False)
print(f"\n✅ Results saved → {out_csv}")

# ─────────────────────────────────────────────────────────────
# Save paths (as JSON — lightweight, no extra deps)
# ─────────────────────────────────────────────────────────────
serialisable_paths = {
    f"alpha={k[0]:.6e}_eta={k[1]:.6e}": v
    for k, v in saved_paths.items()
}
out_json = RESULTS_DIR / "grids" / "grid_search_paths_fogas_40grid.json"
with open(out_json, "w") as fh:
    json.dump(serialisable_paths, fh)
print(f"✅ Paths  saved → {out_json}")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  GRID SEARCH SUMMARY")
print(f"{'='*60}")
print(f"  Total runs       : {len(results)}")
print(f"  Successful runs  : {df['conv_dist'].notna().sum()}")
print(f"  Reached goal     : {int(df['reached_goal'].sum())}")
print(f"  Failed runs      : {df['conv_dist'].isna().sum()}")

best = df.iloc[0]
print(f"\n  ★  Best config:")
print(f"       alpha        = {best['alpha']:.2e}")
print(f"       eta          = {best['eta']:.2e}")
print(f"       α/η ratio    = {best['alpha_eta_ratio']:.1f}")
print(f"       conv_dist    = {best['conv_dist']:.4f}")
print(f"       final_reward = {best['final_reward']:.4f}")
print(f"       reached_goal = {best['reached_goal']}")
print(f"       final_state  = {int(best['final_state'])}")
print(f"\n  Top-5 results (sorted by convergence distance):")
print(df.head(5).to_string(index=False))

goal_df = df[df["reached_goal"]].head(5)
if not goal_df.empty:
    print(f"\n  🎯  Configs that reached goal (top-5):")
    print(goal_df.to_string(index=False))
else:
    print("\n  ⚠️  No configuration reached the goal in this search.")

print(f"\n{'='*60}")
print("  Job complete.")
print(f"{'='*60}")
