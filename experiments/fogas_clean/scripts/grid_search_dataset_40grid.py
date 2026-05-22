"""
FOGAS Dataset-Variation Grid Search — 40×40 Grid
=================================================
Three independent experiment families that test how robust convergence is
when the dataset composition changes, while keeping the algorithm hyper-
parameters fixed at values that are known to converge.

Fixed FOGAS hyper-parameters (from the working notebook):
    alpha = 5e-5,  eta = 2e-5,  rho = 1.0,  T = 80_000

──────────────────────────────────────────────────────────────
EXPERIMENT FAMILY A  –  Manual Augmentation Baseline
──────────────────────────────────────────────────────────────
Starting from the *working* base dataset (proportions=[0.8, 0.2, 0.0],
70 k steps), we inject synthetic (x, a, r, x') transitions for every
(state, action) pair that is NOT visited by either policy.

Grid axes:
  • n_uniform ∈ {1, 10, 25, 50}   – samples added per (state, action) pair
  • coverage  ∈ {0.20, 0.40, 0.60, 0.80, 1.00}
                                   – fraction of unvisited pairs to include
These two axes are crossed → 4 × 5 = 20 configurations.

──────────────────────────────────────────────────────────────
EXPERIMENT FAMILY B  –  Epsilon Variation
──────────────────────────────────────────────────────────────
Both policies get the SAME epsilon (epsilon-greedy exploration).
Dataset: n_steps = 100 000.  Three proportion configs are tested:
  [0.8, 0.2, 0.0]  – no random-start data
  [0.7, 0.2, 0.1]  – 10 % random-start data
  [0.6, 0.2, 0.2]  – 20 % random-start data

Grid axes:
  • proportions ∈ the three configs above
  • epsilon     ∈ {0.0, 0.05, 0.1, 0.2, 0.3, 0.5}

──────────────────────────────────────────────────────────────
EXPERIMENT FAMILY C  –  Random-Start Random Policy Coverage
──────────────────────────────────────────────────────────────
Fixes epsilon = 0.0 for both guided policies.  Varies the proportion
allocated to a "random policy that always starts from a uniform random
non-wall state" (i.e. reset_probs without occupancy bias).

Dataset: n_steps = 100 000.
Proportions are [p_opt, p_alt, p_rand] where p_rand sweeps:

  • p_rand ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
    p_opt   = 0.8 - p_rand/2  (dominant policy keeps ≥ 0.55)
    p_alt   = 0.2 - p_rand/2  (secondary keeps ≥ 0.05 for p_rand ≤ 0.4)
    → guarantees proportions sum to 1.0

Outputs
-------
  grid_search_dataset_40grid_A.csv   – Family A metrics
  grid_search_dataset_40grid_B.csv   – Family B metrics
  grid_search_dataset_40grid_C.csv   – Family C metrics
"""

import os
import sys
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Project root discovery
# ─────────────────────────────────────────────────────────────
def find_root(current_path, marker="setup.py"):
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    return current_path


PROJECT_ROOT = find_root(Path(__file__).resolve())
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_methods import EnvDataCollector
from rl_methods.mdp_clean import DiscreteMDP, Planner
from rl_methods.fogas_clean import FOGASSolver, FOGASEvaluator

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
# Grid layout (20×20 base → scaled to 40×40)
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
        if x_int == goal:  return  10.0
        if x_int in pits:  return -10.0
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
# Waypoint policy helpers (same as notebook)
# ─────────────────────────────────────────────────────────────
def build_waypoint_policy(N, A, waypoints, grid_size=40):
    pi = np.ones((N, A)) / A
    for i in range(len(waypoints) - 1):
        r1, c1 = waypoints[i]
        r2, c2 = waypoints[i + 1]
        dr, dc = np.sign(r2 - r1), np.sign(c2 - c1)
        curr_r, curr_c = r1, c1
        while (curr_r, curr_c) != (r2, c2):
            s = int(curr_r * grid_size + curr_c)
            action = -1
            if dr != 0 and curr_r != r2:
                action  = 1 if dr > 0 else 0
                curr_r += dr
            elif dc != 0 and curr_c != c2:
                action  = 3 if dc > 0 else 2
                curr_c += dc
            if action != -1:
                pi[s]         = 0.0
                pi[s, action] = 1.0
    return torch.tensor(pi, dtype=torch.float64)


def compute_optimal_path(planner, start, goal, size, max_steps=500):
    """Simulate the greedy policy from 'start' until 'goal' or max_steps."""
    s = int(start)
    visited = [s]
    pi = planner.pi_star.numpy()  # (N, A)
    for _ in range(max_steps):
        if s == goal:
            break
        a = int(np.argmax(pi[s]))
        row, col = divmod(s, size)
        if   a == 0: row, col = max(0, row - 1), col
        elif a == 1: row, col = min(size - 1, row + 1), col
        elif a == 2: row, col = row, max(0, col - 1)
        elif a == 3: row, col = row, min(size - 1, col + 1)
        s = row * size + col
        visited.append(s)
    return visited


# ─────────────────────────────────────────────────────────────
# Build grids & MDP
# ─────────────────────────────────────────────────────────────
print("⚙️  Building grids and MDP …")
grid_20 = np.array([list(row) for row in grid_layout])
grid_40 = scale_grid(grid_20, factor=2)
size_20 = grid_20.shape[0]
size_40 = grid_40.shape[0]

mdp_data_20 = generate_mdp_regular_grid(grid_20)
mdp_data_40 = generate_mdp_regular_grid(grid_40)

fixed_centers = mdp_data_20["centers"]
fixed_sigma   = calculate_local_sigma(fixed_centers, k=2)

phi_s_20 = get_phi_state_func(
    fixed_centers, fixed_sigma, size_20,
    mdp_data_20["pits"], mdp_data_20["goal"]
)
def phi_20_fixed(x, a):
    s_feat      = phi_s_20(x)
    e_a         = torch.zeros(4, dtype=torch.float64)
    e_a[int(a)] = 1.0
    return torch.kron(e_a, s_feat)

phi_s_40 = get_phi_state_func(
    fixed_centers, fixed_sigma, size_40,
    mdp_data_40["pits"], mdp_data_40["goal"]
)
def phi_40_fixed(x, a):
    s_feat      = phi_s_40(x)
    e_a         = torch.zeros(4, dtype=torch.float64)
    e_a[int(a)] = 1.0
    return torch.kron(e_a, s_feat)

states_40  = torch.arange(size_40 * size_40, dtype=torch.int64)
actions_40 = torch.arange(4,                dtype=torch.int64)

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
# Build the two policies (same as notebook)
# ─────────────────────────────────────────────────────────────
N_40  = size_40 * size_40
A_40  = 4
GOAL  = mdp_data_40["goal"]
START = mdp_data_40["start"]

new_path_waypoints = [
    (0,  0),
    (9,  0),
    (9,  8),
    (17, 8),  (17, 20),
    (25, 20), (25, 30),
    (32, 30),
    (32, 37), (38, 37),
    (38, 39),
    (39, 39),
]
parallel_pi_star = build_waypoint_policy(N_40, A_40, new_path_waypoints, grid_size=40)

pi_fogas_40 = planner_40.pi_star.clone()

# ─────────────────────────────────────────────────────────────
# Pre-compute which (state, action) pairs are visited by the policies
# ─────────────────────────────────────────────────────────────
def get_visited_state_action_pairs(pi: torch.Tensor, mdp, size, goal, walls, pits,
                                   max_steps: int = 500):
    """
    Simulate the deterministic greedy policy from the start state.
    Returns a set of (state, action) pairs visited along the trajectory.
    """
    visited = set()
    s = mdp.x0
    pi_np = pi.numpy()
    for _ in range(max_steps):
        if s == goal or s in pits:
            break
        a = int(np.argmax(pi_np[s]))
        visited.add((int(s), a))
        row, col = divmod(int(s), size)
        if   a == 0: row, col = max(0, row - 1), col
        elif a == 1: row, col = min(size - 1, row + 1), col
        elif a == 2: row, col = row, max(0, col - 1)
        elif a == 3: row, col = row, min(size - 1, col + 1)
        s = row * size + col
    return visited


print("⚙️  Computing policy-visited (state, action) pairs …")
visited_fogas   = get_visited_state_action_pairs(
    pi_fogas_40,   mdp_40, size_40, GOAL,
    mdp_data_40["walls"], mdp_data_40["pits"]
)
visited_waypoint = get_visited_state_action_pairs(
    parallel_pi_star, mdp_40, size_40, GOAL,
    mdp_data_40["walls"], mdp_data_40["pits"]
)
visited_by_any = visited_fogas | visited_waypoint

# All non-wall, non-terminal (state, action) pairs NOT touched by either policy
all_sa_pairs = [
    (s, a)
    for s in range(N_40)
    for a in range(A_40)
    if s not in mdp_data_40["walls"]
    and s not in mdp_data_40["pits"]
    and s != GOAL
]
unvisited_sa_pairs = sorted(set(all_sa_pairs) - {(s, a) for s, a in visited_by_any})

print(f"   Total non-wall/non-terminal (s,a) pairs : {len(all_sa_pairs)}")
print(f"   Visited by either policy                : {len(visited_by_any)}")
print(f"   Unvisited (augmentation candidates)     : {len(unvisited_sa_pairs)}")


# ─────────────────────────────────────────────────────────────
# Dataset collection helpers
# ─────────────────────────────────────────────────────────────

RESET_OPTS_40 = {
    "x0":        0.2,
    "occupancy": 0.8,
}

RESET_OPTS_RANDOM = {
    "x0":     0.0,
    "random": 1.0,
}

FOGAS_PARAMS = {
    "alpha": 5e-5,
    "eta":   2e-5,
    "rho":   1.0,
    "T":     80_000,
}
BETA_VAL      = 1e-7
MAX_SIM_STEPS = 500


def transition(s, a, size, walls, pits, goal):
    """Deterministic single-step transition for the 40-grid MDP."""
    if s in pits or s == goal:
        return s, mdp_data_40["reward_fn"](s, a)
    row, col = divmod(int(s), size)
    if   a == 0: row, col = max(0, row - 1), col
    elif a == 1: row, col = min(size - 1, row + 1), col
    elif a == 2: row, col = row, max(0, col - 1)
    elif a == 3: row, col = row, min(size - 1, col + 1)
    ns = row * size + col
    if ns in walls:
        ns = s
    return ns, mdp_data_40["reward_fn"](s, a)


def build_augmentation_rows(sa_subset):
    """
    For each (state, action) in sa_subset return ONE synthetic transition row.
    Columns match EnvDataCollector CSV output: state, action, reward, next_state.
    """
    rows = []
    for (s, a) in sa_subset:
        ns, r = transition(s, a, size_40,
                           mdp_data_40["walls"],
                           mdp_data_40["pits"], GOAL)
        rows.append({"state": s, "action": a, "reward": r, "next_state": ns})
    return rows


def run_fogas(csv_path: str):
    """Train FOGAS on csv_path and return metrics dict."""
    solver = FOGASSolver(
        mdp     = mdp_40,
        phi     = phi_40_fixed,
        csv_path = csv_path,
        device   = device,
        beta     = BETA_VAL,
        seed     = seed,
    )
    solver.run(
        alpha      = FOGAS_PARAMS["alpha"],
        eta        = FOGAS_PARAMS["eta"],
        rho        = FOGAS_PARAMS["rho"],
        T          = FOGAS_PARAMS["T"],
        tqdm_print = False,
    )
    evaluator = FOGASEvaluator(solver, planner=planner_40)

    # Convergence: did the greedy policy reach the goal?
    try:
        traj       = evaluator.simulate_trajectory(
            policy_mode="greedy", max_steps=MAX_SIM_STEPS, seed=seed, goal_state=GOAL
        )
        convergence = int(traj[-1]["next_state"] == GOAL)
        path_len    = len(traj)
    except Exception:
        convergence = 0
        path_len    = -1

    # Final reward (expected reward under learned policy)
    try:
        final_reward = evaluator.average_return(
            policy_mode="solver",
            num_trajectories=10,
            max_steps=MAX_SIM_STEPS,
            seed=seed,
            goal_state=GOAL,
        )["policy"]
    except Exception:
        final_reward = float("nan")

    # Sub-optimality gaps (v_gap, q_gap)
    try:
        mu_star      = planner_40.mu_star.cpu()
        mu_star_norm = mu_star / (mu_star.sum() + 1e-300)
        d_star       = mu_star_norm.reshape(N_40, A_40).sum(dim=1)

        v_star      = planner_40.v_star.cpu()
        q_star_flat = planner_40.q_star.cpu().reshape(-1)

        v_pi, q_pi  = planner_40.evaluate_policy(solver.pi)
        v_pi_cpu    = v_pi.cpu()
        q_pi_flat   = q_pi.reshape(-1).cpu()

        v_gap = (d_star * (v_star - v_pi_cpu)).sum().item()
        q_gap = (mu_star_norm * (q_star_flat - q_pi_flat)).sum().item()
    except Exception:
        v_gap = float("nan")
        q_gap = float("nan")

    # Convergence distance metric
    try:
        metric_fn  = evaluator.get_metric(
            "success_rate",
            goal_state=GOAL,
            num_trajectories=10,
            max_steps=MAX_SIM_STEPS,
            seed=seed,
        )
        conv_dist  = -metric_fn()
    except Exception:
        conv_dist = float("nan")

    return {
        "convergence":  convergence,
        "conv_dist":    conv_dist,
        "final_reward": final_reward,
        "v_gap":        v_gap,
        "q_gap":        q_gap,
        "path_len":     path_len,
    }


# ─────────────────────────────────────────────────────────────
# EXPERIMENT FAMILY A  –  Manual Augmentation Baseline
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EXPERIMENT FAMILY A  –  Manual Augmentation")
print("=" * 60)

n_uniform_values = [1, 2, 4, 8]
coverage_values  = [0.20, 0.40, 0.60, 0.80, 1.00]

# Shuffle once so that different coverages pick a consistent prefix
rng_aug = np.random.default_rng(seed)
shuffled_unvisited = rng_aug.permutation(unvisited_sa_pairs).tolist()

results_A = []
total_A = len(n_uniform_values) * len(coverage_values)

collector_base = EnvDataCollector(
    mdp              = planner_40,
    env_name         = "40grid",
    restricted_states= mdp_data_40["walls"],
    reset_probs      = RESET_OPTS_40,
    max_steps        = 200,
)

with tqdm(total=total_A, desc="[Family A] Augmentation Grid") as pbar:
    for coverage in coverage_values:
        n_pairs = max(1, int(len(shuffled_unvisited) * coverage))
        chosen_sa = shuffled_unvisited[:n_pairs]

        for n_uniform in n_uniform_values:
            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, dir="/tmp"
            ) as f:
                base_csv = f.name

            # 1. Collect base dataset (same as working notebook)
            try:
                epsilon_policy_fogas    = (planner_40.pi_star,    0.0)
                epsilon_policy_waypoint = (parallel_pi_star,  0.0)

                collector_base.collect_mixed_dataset_terminal_aware(
                    policies    = [epsilon_policy_fogas, epsilon_policy_waypoint, "random"],
                    proportions = [0.8, 0.2, 0.0],
                    n_steps     = 70_000,
                    episode_based = True,
                    save_path   = base_csv,
                    verbose     = False,
                    extra_steps = 40,
                )
            except Exception as e:
                print(f"\n⚠️  Base dataset collection failed: {e}")
                pbar.update(1)
                continue

            # 2. Build augmentation rows and append n_uniform copies
            aug_rows = build_augmentation_rows(chosen_sa) * n_uniform

            # 3. Merge with the base dataset
            try:
                base_df = pd.read_csv(base_csv)
                aug_df  = pd.DataFrame(aug_rows)
                merged_df = pd.concat([base_df, aug_df], ignore_index=True)
                merged_df.to_csv(base_csv, index=False)
            except Exception as e:
                print(f"\n⚠️  Augmentation merge failed: {e}")
                os.unlink(base_csv)
                pbar.update(1)
                continue

            # 4. Train FOGAS and record metrics
            try:
                metrics = run_fogas(base_csv)
            except Exception as e:
                print(f"\n⚠️  FOGAS run failed: {e}")
                metrics = {
                    "convergence": 0, "conv_dist": float("nan"),
                    "final_reward": float("nan"), "v_gap": float("nan"),
                    "q_gap": float("nan"), "path_len": -1,
                }

            os.unlink(base_csv)

            results_A.append({
                "experiment":       "A_augmentation",
                "n_uniform":        n_uniform,
                "coverage":         coverage,
                "n_unvisited_pairs": n_pairs,
                "total_aug_rows":   len(aug_rows),
                "total_dataset_size": int(70_000 + len(aug_rows)),
                **metrics,
            })
            pbar.update(1)

df_A = pd.DataFrame(results_A)
out_A = RESULTS_DIR / "grids" / "grid_search_dataset_40grid_A.csv"
df_A.to_csv(out_A, index=False)
print(f"\n✅ Family A results saved → {out_A}")
print(df_A[["n_uniform", "coverage", "convergence", "conv_dist", "final_reward"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
# EXPERIMENT FAMILY B  –  Epsilon Variation
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EXPERIMENT FAMILY B  –  Epsilon Variation")
print("=" * 60)

epsilon_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

# Proportion configs: (p_opt, p_alt, p_rand, label)
# p_rand is the fraction collected by the random-start policy
B_PROP_CONFIGS = [
    (0.8, 0.2, 0.0, "80/20/0"),
    (0.7, 0.2, 0.1, "70/20/10"),
    (0.6, 0.2, 0.2, "60/20/20"),
]

N_STEPS_B = 100_000

# Collector for the two guided (epsilon-greedy) policies
collector_eps = EnvDataCollector(
    mdp              = planner_40,
    env_name         = "40grid",
    restricted_states= mdp_data_40["walls"],
    reset_probs      = RESET_OPTS_40,     # occupancy-based for the guided policies
    max_steps        = 200,
)

# Collector for the random policy with uniform random starting states
collector_rand_start = EnvDataCollector(
    mdp              = mdp_40,
    env_name         = "40grid_rand",
    restricted_states= mdp_data_40["walls"],
    reset_probs      = RESET_OPTS_RANDOM,  # purely uniform random starts
    max_steps        = 200,
)

results_B = []
total_B = len(B_PROP_CONFIGS) * len(epsilon_values)

with tqdm(total=total_B, desc="[Family B] Epsilon x Proportions") as pbar:
    for p_opt, p_alt, p_rand, prop_label in B_PROP_CONFIGS:
        n_guided = int(N_STEPS_B * (p_opt + p_alt))
        n_random = int(N_STEPS_B * p_rand)
        # Within the guided portion, keep the opt:alt ratio
        guided_proportions = [p_opt / (p_opt + p_alt), p_alt / (p_opt + p_alt)]

        for eps in epsilon_values:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, dir="/tmp") as f:
                csv_guided = f.name
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, dir="/tmp") as f:
                csv_rand = f.name

            try:
                epsilon_policy_fogas    = (planner_40.pi_star,   eps)
                epsilon_policy_waypoint = (parallel_pi_star, eps)

                # Collect guided portion
                collector_eps.collect_mixed_dataset_terminal_aware(
                    policies    = [epsilon_policy_fogas, epsilon_policy_waypoint],
                    proportions = guided_proportions,
                    n_steps     = n_guided,
                    episode_based = True,
                    save_path   = csv_guided,
                    verbose     = False,
                    extra_steps = 40,
                )
                df_guided = pd.read_csv(csv_guided)

                # Collect random-start portion (only when p_rand > 0)
                if n_random > 0:
                    collector_rand_start.collect_dataset_terminal_aware(
                        policy      = "random",
                        n_steps     = n_random,
                        save_path   = csv_rand,
                        verbose     = False,
                        extra_steps = 20,
                    )
                    df_rand   = pd.read_csv(csv_rand)
                else:
                    df_rand = pd.DataFrame(columns=["state", "action", "reward", "next_state"])

                # Merge and save to the guided CSV (reuse as final path)
                merged = pd.concat([df_guided, df_rand], ignore_index=True)
                merged.to_csv(csv_guided, index=False)

            except Exception as e:
                print(f"\n⚠️  Dataset collection failed (eps={eps}, props={prop_label}): {e}")
                for p in [csv_guided, csv_rand]:
                    if os.path.exists(p): os.unlink(p)
                results_B.append({
                    "experiment":   "B_epsilon",
                    "proportions":  prop_label,
                    "p_opt": p_opt, "p_alt": p_alt, "p_rand": p_rand,
                    "epsilon":      eps,
                    "convergence": 0, "conv_dist": float("nan"),
                    "final_reward": float("nan"),
                    "v_gap": float("nan"), "q_gap": float("nan"), "path_len": -1,
                })
                pbar.update(1)
                continue

            try:
                metrics = run_fogas(csv_guided)
            except Exception as e:
                print(f"\n⚠️  FOGAS run failed (eps={eps}, props={prop_label}): {e}")
                metrics = {
                    "convergence": 0, "conv_dist": float("nan"),
                    "final_reward": float("nan"),
                    "v_gap": float("nan"), "q_gap": float("nan"), "path_len": -1,
                }

            for p in [csv_guided, csv_rand]:
                if os.path.exists(p): os.unlink(p)

            results_B.append({
                "experiment":   "B_epsilon",
                "proportions":  prop_label,
                "p_opt": p_opt, "p_alt": p_alt, "p_rand": p_rand,
                "epsilon":      eps,
                **metrics,
            })
            pbar.update(1)

df_B = pd.DataFrame(results_B)
out_B = RESULTS_DIR / "grids" / "grid_search_dataset_40grid_B.csv"
df_B.to_csv(out_B, index=False)
print(f"\n✅ Family B results saved → {out_B}")
print(df_B[["proportions", "epsilon", "convergence", "conv_dist", "final_reward"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
# EXPERIMENT FAMILY C  –  Random-Start Policy Coverage
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EXPERIMENT FAMILY C  –  Random-Start Random Policy Coverage")
print("=" * 60)

# p_rand varies; scale p_opt and p_alt proportionally so that
# p_opt + p_alt + p_rand = 1.0
# Base (p_rand=0): p_opt=0.8, p_alt=0.2
p_rand_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

def compute_props_C(p_rand):
    """Return (p_opt, p_alt, p_rand) that sum to 1.0."""
    remaining = 1.0 - p_rand
    # Keep the 4:1 ratio between opt and alt from the base case
    p_opt = round(remaining * 0.8, 10)
    p_alt = round(remaining * 0.2, 10)
    return p_opt, p_alt, p_rand

collector_C_guided = EnvDataCollector(
    mdp              = planner_40,
    env_name         = "40grid",
    restricted_states= mdp_data_40["walls"],
    reset_probs      = RESET_OPTS_40,
    max_steps        = 200,
)

collector_C_random = EnvDataCollector(
    mdp              = mdp_40,
    env_name         = "40grid_rand",
    restricted_states= mdp_data_40["walls"],
    reset_probs      = RESET_OPTS_RANDOM,  # uniform random starts (NOT occupancy-based)
    max_steps        = 200,
)

results_C = []
N_STEPS_C = 100_000

with tqdm(total=len(p_rand_values), desc="[Family C] Random-Start Coverage") as pbar:
    for p_rand in p_rand_values:
        p_opt, p_alt, p_rnd = compute_props_C(p_rand)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, dir="/tmp") as f:
            csv_guided = f.name
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, dir="/tmp") as f:
            csv_random = f.name

        try:
            epsilon_policy_fogas    = (planner_40.pi_star,   0.0)
            epsilon_policy_waypoint = (parallel_pi_star, 0.0)

            n_guided = int(N_STEPS_C * (p_opt + p_alt))
            n_random = int(N_STEPS_C * p_rnd)

            if n_guided > 0:
                # Normalize the guided proportions within the guided sub-collection
                guided_proportions = [
                    p_opt / (p_opt + p_alt),
                    p_alt / (p_opt + p_alt),
                ]
                collector_C_guided.collect_mixed_dataset_terminal_aware(
                    policies    = [epsilon_policy_fogas, epsilon_policy_waypoint],
                    proportions = guided_proportions,
                    n_steps     = n_guided,
                    episode_based = True,
                    save_path   = csv_guided,
                    verbose     = False,
                    extra_steps = 40,
                )
                df_guided = pd.read_csv(csv_guided)
            else:
                df_guided = pd.DataFrame(columns=["state", "action", "reward", "next_state"])

            if n_random > 0:
                # The random policy uses uniform random starting states (not occupancy-based)
                collector_C_random.collect_dataset_terminal_aware(
                    policy      = "random",
                    n_steps     = n_random,
                    save_path   = csv_random,
                    verbose     = False,
                    extra_steps = 20,
                )
                df_random = pd.read_csv(csv_random)
            else:
                df_random = pd.DataFrame(columns=["state", "action", "reward", "next_state"])

            merged = pd.concat([df_guided, df_random], ignore_index=True)
            merged.to_csv(csv_guided, index=False)

        except Exception as e:
            print(f"\n⚠️  Dataset collection failed (p_rand={p_rand}): {e}")
            for p in [csv_guided, csv_random]:
                if os.path.exists(p): os.unlink(p)
            results_C.append({
                "experiment": "C_random_start",
                "p_rand": p_rand, "p_opt": p_opt, "p_alt": p_alt,
                "convergence": 0, "conv_dist": float("nan"),
                "final_reward": float("nan"),
                "v_gap": float("nan"), "q_gap": float("nan"), "path_len": -1,
            })
            pbar.update(1)
            continue

        try:
            metrics = run_fogas(csv_guided)
        except Exception as e:
            print(f"\n⚠️  FOGAS run failed (p_rand={p_rand}): {e}")
            metrics = {
                "convergence": 0, "conv_dist": float("nan"),
                "final_reward": float("nan"),
                "v_gap": float("nan"), "q_gap": float("nan"), "path_len": -1,
            }

        for p in [csv_guided, csv_random]:
            if os.path.exists(p): os.unlink(p)

        results_C.append({
            "experiment":   "C_random_start",
            "p_rand":       p_rand,
            "p_opt":        p_opt,
            "p_alt":        p_alt,
            "n_guided":     n_guided,
            "n_random":     n_random,
            **metrics,
        })
        pbar.update(1)

df_C = pd.DataFrame(results_C)
out_C = RESULTS_DIR / "grids" / "grid_search_dataset_40grid_C.csv"
df_C.to_csv(out_C, index=False)
print(f"\n✅ Family C results saved → {out_C}")
print(df_C[["p_rand", "convergence", "conv_dist", "final_reward"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  GRID SEARCH COMPLETE")
print("=" * 60)
for label, df, out in [
    ("A – Augmentation", df_A, out_A),
    ("B – Epsilon",      df_B, out_B),
    ("C – Random Start", df_C, out_C),
]:
    conv_col = "convergence"
    print(f"\n  Family {label}:")
    print(f"    Runs         : {len(df)}")
    if conv_col in df.columns:
        print(f"    Converged    : {int(df[conv_col].sum(skipna=True))}")
        print(f"    Not converged: {(df[conv_col] == 0).sum()}")
    print(f"    Saved to     : {out}")
print("\n" + "=" * 60)
