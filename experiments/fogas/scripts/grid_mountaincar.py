"""Compare FOGAS and FQI on matched discretized Mountain Car datasets.

The experiment first learns a near-optimal tabular Q-learning policy, uses one
of its trajectories to define a custom reset distribution, and collects fixed
Gymnasium datasets from mixtures of that policy and random behavior. It then
trains FOGAS and FQI with the same action-block RBF features while varying the
behavior mixture, epsilon, and nominal-start/custom-reset proportions. This is
the batch producer for the final analysis in
``experiments/fogas/notebooks/mountainCar.ipynb``.

Run from the repository root with
``python3 experiments/fogas/scripts/grid_mountaincar.py``. Each completed
configuration is checkpointed to
``data/results/mountainCar/grids/grid_mountaincar.csv``; temporary transition
datasets are removed when the run finishes.
"""

import json
import os
import random
import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import itertools
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
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "mountainCar"
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from rl_methods.data_collection import GymDataBuffer  # noqa: E402
from rl_methods.fogas import FOGASSolver  # noqa: E402
from rl_methods.fqi import FQISolver  # noqa: E402
from rl_methods.mdp import (  # noqa: E402
    ActionDiscretizer,
    FeaturesMDP,
    StateDiscretizer,
)
from rl_methods.q_learning import run_q_learning  # noqa: E402


SEED = 44
ENV_ID = "MountainCar-v0"
GAMMA = 0.9
GOAL_POSITION = 0.5
GOAL_VELOCITY = 0.0
MIN_POSITION = -1.2
MAX_POSITION = 0.6
MAX_SPEED = 0.07
TIME_LIMIT = 200

STATE_BINS = np.array([20, 20], dtype=np.int64)
OBS_LOW = np.array([MIN_POSITION, -MAX_SPEED], dtype=np.float64)
OBS_HIGH = np.array([MAX_POSITION, MAX_SPEED], dtype=np.float64)

ACTION_IDS = np.array([0, 1, 2], dtype=np.int64)
ACTION_LABELS = {0: "left", 1: "coast", 2: "right"}
INITIAL_OBS_REFERENCE = np.array([-0.5, 0.0], dtype=np.float64)

RBF_BINS = np.array([15, 15], dtype=np.int64)
VARIANCE_SCALE = 0.05

Q_LEARNING_CONFIG = {
    "episodes": 6000,
    "alpha": 0.9,
    "gamma": 0.9,
    "epsilon_start": 1.0,
}

DATASET_GRID = {
    "n_transitions": [25_000],
    "epsilon": [0.0, 0.1, 0.2, 0.4, 0.6],
    "proportions": [
        (1.0, 0.0),
        (0.8, 0.2),
        (0.6, 0.4),
        (0.4, 0.6),
        (0.2, 0.8),
        (0.0, 1.0),
    ],
}

RESET_CONFIGS = [
    {"name": "x0_0_custom_100", "reset_probs": {"x0": 0.0, "custom": 1.0}},
    {"name": "x0_10_custom_90", "reset_probs": {"x0": 0.1, "custom": 0.9}},
    {"name": "x0_20_custom_80", "reset_probs": {"x0": 0.2, "custom": 0.8}},
    {"name": "x0_40_custom_60", "reset_probs": {"x0": 0.4, "custom": 0.6}},
    {"name": "x0_60_custom_40", "reset_probs": {"x0": 0.6, "custom": 0.4}},
    {"name": "x0_80_custom_20", "reset_probs": {"x0": 0.8, "custom": 0.2}},
    {"name": "x0_100_custom_0", "reset_probs": {"x0": 1.0, "custom": 0.0}},
]

FEATURE_TYPES = ("rbf",)

FOGAS_CONFIG = {
    "beta": 1e-6,
    "alpha": 1e-4,
    "eta": 2e-5,
    "rho": 0.5,
    "T": 20_000,
}

FQI_CONFIG = {
    "K": 5_000,
    "tau": 0.1,
    "ridge": 1e-2,
    "augment_terminal_transitions": True,
}

EVAL_CONFIG = {
    "n_trials": 10,
    "max_steps": TIME_LIMIT,
}

OUTPUT_CSV = RESULTS_DIR / "grids" / "grid_mountaincar.csv"


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_abstraction():
    state_disc = StateDiscretizer(
        low=OBS_LOW,
        high=OBS_HIGH,
        bins=STATE_BINS,
        terminal_obs_predicate=lambda obs: (
            obs[0] >= GOAL_POSITION and obs[1] >= GOAL_VELOCITY
        ),
    )
    action_disc = ActionDiscretizer(
        action_values=ACTION_IDS,
        action_labels=ACTION_LABELS,
    )
    return state_disc, action_disc


def build_rbf_mdp(state_disc, action_disc):
    states = torch.arange(state_disc.n_states, dtype=torch.int64)
    actions = torch.arange(action_disc.n_actions, dtype=torch.int64)

    rbf_edges = [
        np.linspace(lo, hi, n_bins + 1, dtype=np.float64)
        for lo, hi, n_bins in zip(OBS_LOW, OBS_HIGH, RBF_BINS)
    ]
    rbf_grid_centers = [0.5 * (edges[:-1] + edges[1:]) for edges in rbf_edges]
    rbf_centers = np.array(list(itertools.product(*rbf_grid_centers)), dtype=np.float64)

    width_position = (MAX_POSITION - MIN_POSITION) / RBF_BINS[0]
    width_velocity = (MAX_SPEED - (-MAX_SPEED)) / RBF_BINS[1]
    sigma_squared = np.array(
        [width_position**2, width_velocity**2],
        dtype=np.float64,
    ) * VARIANCE_SCALE

    k_centers = len(rbf_centers)
    d_rbf = k_centers * action_disc.n_actions

    def phi_rbf(state_id, action_id):
        obs = state_disc.state_id_to_center_obs(state_id)
        if obs is None:
            return torch.zeros(d_rbf, dtype=torch.float64)

        diff_sq = (rbf_centers - obs) ** 2
        f_x = np.exp(-0.5 * np.sum(diff_sq / sigma_squared, axis=1))

        feat = np.zeros(d_rbf, dtype=np.float64)
        start_idx = int(action_id) * k_centers
        feat[start_idx:start_idx + k_centers] = f_x
        return torch.from_numpy(feat)

    return FeaturesMDP(
        states=states,
        actions=actions,
        phi=phi_rbf,
        gamma=GAMMA ** 5,
        x0=state_disc.obs_to_state_id(INITIAL_OBS_REFERENCE),
        omega=None,
    )


def build_mdp(feature_type, state_disc, action_disc):
    if feature_type == "rbf":
        return build_rbf_mdp(state_disc, action_disc)
    raise ValueError(f"Unsupported feature_type: {feature_type}")


def build_q_learning_policy(state_disc, action_disc, seed):
    q_star_ql, _, _ = run_q_learning(
        episodes=Q_LEARNING_CONFIG["episodes"],
        alpha=Q_LEARNING_CONFIG["alpha"],
        gamma=Q_LEARNING_CONFIG["gamma"],
        epsilon_start=Q_LEARNING_CONFIG["epsilon_start"],
        render=False,
        seed=seed,
        env_id=ENV_ID,
        env_kwargs={"goal_velocity": GOAL_VELOCITY},
        obs_to_state_id=state_disc.obs_to_state_id,
        n_states=state_disc.n_states,
        n_actions=action_disc.n_actions,
        terminal_state_id=state_disc.absorbing_state_id,
        initial_state_id=state_disc.obs_to_state_id(INITIAL_OBS_REFERENCE),
        action_id_to_label=action_disc.action_id_to_label,
        plot=False,
        print_summary=False,
    )

    greedy_actions_ql = torch.argmax(q_star_ql, dim=1)
    pi_matrix_ql = torch.zeros(
        (state_disc.n_states, action_disc.n_actions),
        dtype=torch.float64,
    )
    pi_matrix_ql[torch.arange(state_disc.n_states), greedy_actions_ql] = 1.0
    pi_matrix_ql[state_disc.absorbing_state_id] = 1.0 / action_disc.n_actions
    return pi_matrix_ql


def evaluate_policy_mean_steps(
    policy,
    state_disc,
    action_disc,
    env_id=ENV_ID,
    goal_velocity=GOAL_VELOCITY,
    n_trials=10,
    max_steps=200,
    seed=SEED,
    action_selection="greedy",
):
    if hasattr(policy, "detach"):
        policy = policy.detach().cpu()
    policy = torch.as_tensor(policy, dtype=torch.float64)
    if policy.ndim == 1:
        policy = policy.reshape(state_disc.n_states, action_disc.n_actions)

    if action_selection not in {"greedy", "sample"}:
        raise ValueError(
            "action_selection must be one of {'greedy', 'sample'}, "
            f"got {action_selection}"
        )

    rng = np.random.default_rng(seed)
    env = gym.make(env_id, max_episode_steps=max_steps, goal_velocity=goal_velocity)
    total_steps_list = []
    success_count = 0

    for trial in range(n_trials):
        obs, _ = env.reset(seed=seed + trial)
        done = False
        steps = 0

        while not done:
            state_id = state_disc.obs_to_state_id(obs)
            action_probs = policy[state_id]
            if action_selection == "greedy":
                action_id = int(torch.argmax(action_probs).item())
            else:
                probs = action_probs.numpy().astype(np.float64)
                probs = np.clip(probs, 0.0, None)
                prob_sum = probs.sum()
                if not np.isfinite(prob_sum) or np.isclose(prob_sum, 0.0):
                    probs = np.full(action_disc.n_actions, 1.0 / action_disc.n_actions)
                else:
                    probs = probs / prob_sum
                action_id = int(rng.choice(action_disc.n_actions, p=probs))

            env_action = action_disc.action_id_to_env_action(action_id)
            obs, _, terminated, truncated, _ = env.step(env_action)
            steps += 1
            done = terminated or truncated

        if steps < max_steps:
            success_count += 1
        total_steps_list.append(steps)

    env.close()
    mean_steps = float(np.mean(total_steps_list))
    return mean_steps, success_count


def run_fogas(mdp, dataset_path, state_disc, action_disc, device, seed):
    solver = FOGASSolver(
        mdp=mdp,
        phi=mdp.phi,
        csv_path=dataset_path,
        device=device,
        beta=FOGAS_CONFIG["beta"],
        seed=seed,
    )
    solver.run(
        alpha=FOGAS_CONFIG["alpha"],
        eta=FOGAS_CONFIG["eta"],
        rho=FOGAS_CONFIG["rho"],
        T=FOGAS_CONFIG["T"],
        tqdm_print=False,
    )

    greedy_mean_steps, greedy_success_count = evaluate_policy_mean_steps(
        policy=solver.pi,
        state_disc=state_disc,
        action_disc=action_disc,
        n_trials=EVAL_CONFIG["n_trials"],
        max_steps=EVAL_CONFIG["max_steps"],
        seed=seed,
        action_selection="greedy",
    )
    solver_mean_steps, solver_success_count = evaluate_policy_mean_steps(
        policy=solver.pi,
        state_disc=state_disc,
        action_disc=action_disc,
        n_trials=EVAL_CONFIG["n_trials"],
        max_steps=EVAL_CONFIG["max_steps"],
        seed=seed + 100_000,
        action_selection="sample",
    )

    return {
        "mean_steps": greedy_mean_steps,
        "successes": greedy_success_count,
        "solver_alpha": FOGAS_CONFIG["alpha"],
        "solver_eta": FOGAS_CONFIG["eta"],
        "solver_rho": FOGAS_CONFIG["rho"],
        "solver_T": FOGAS_CONFIG["T"],
        "solver_beta": FOGAS_CONFIG["beta"],
        "fogas_mean_steps": greedy_mean_steps,
        "fogas_successes": greedy_success_count,
        "fogas_greedy_mean_steps": greedy_mean_steps,
        "fogas_greedy_successes": greedy_success_count,
        "fogas_solver_mean_steps": solver_mean_steps,
        "fogas_solver_successes": solver_success_count,
        "fogas_alpha": FOGAS_CONFIG["alpha"],
        "fogas_eta": FOGAS_CONFIG["eta"],
        "fogas_rho": FOGAS_CONFIG["rho"],
        "fogas_T": FOGAS_CONFIG["T"],
        "fogas_beta": FOGAS_CONFIG["beta"],
        "fogas_status": "ok",
        "fogas_error": "",
    }


def run_fqi(mdp, dataset_path, state_disc, action_disc, device, seed):
    solver = FQISolver(
        mdp=mdp,
        phi=mdp.phi,
        csv_path=dataset_path,
        device=device,
        seed=seed,
        ridge=FQI_CONFIG["ridge"],
        augment_terminal_transitions=FQI_CONFIG["augment_terminal_transitions"],
    )
    solver.run(
        K=FQI_CONFIG["K"],
        tau=FQI_CONFIG["tau"],
        verbose=False,
    )

    mean_steps, success_count = evaluate_policy_mean_steps(
        policy=solver.pi,
        state_disc=state_disc,
        action_disc=action_disc,
        n_trials=EVAL_CONFIG["n_trials"],
        max_steps=EVAL_CONFIG["max_steps"],
        seed=seed,
    )

    theta_delta = np.nan
    if len(solver.theta_history) >= 2:
        theta_delta = float(
            torch.linalg.norm(
                solver.theta_history[-1] - solver.theta_history[-2]
            ).detach().cpu().item()
        )

    return {
        "fqi_mean_steps": mean_steps,
        "fqi_successes": success_count,
        "fqi_greedy_mean_steps": mean_steps,
        "fqi_greedy_successes": success_count,
        "fqi_solver_mean_steps": mean_steps,
        "fqi_solver_successes": success_count,
        "fqi_K": FQI_CONFIG["K"],
        "fqi_tau": FQI_CONFIG["tau"],
        "fqi_ridge": FQI_CONFIG["ridge"],
        "fqi_augment_terminal_transitions": FQI_CONFIG[
            "augment_terminal_transitions"
        ],
        "fqi_added_terminal_samples": int(
            getattr(solver, "added_terminal_samples", 0)
        ),
        "fqi_final_theta_delta": theta_delta,
        "fqi_status": "ok",
        "fqi_error": "",
    }


def run_single_experiment(
    feature_type,
    reset_config,
    proportions,
    epsilon,
    n_transitions,
    state_disc,
    action_disc,
    pi_matrix_ql,
    trajectory_reset_distribution,
    device,
    dataset_path,
    seed,
):
    dataset_df = GymDataBuffer.collect(
        policy_matrix=pi_matrix_ql,
        state_disc=state_disc,
        action_disc=action_disc,
        env_id=ENV_ID,
        n_transitions=n_transitions,
        epsilon=epsilon,
        proportions=proportions,
        episode_based=True,
        max_steps_per_episode=TIME_LIMIT,
        reset_probs=reset_config["reset_probs"],
        custom_reset_distribution=trajectory_reset_distribution,
        reset_obs_mode="uniform_in_bin",
        seed=seed,
        save_path=dataset_path,
        verbose=False,
        drop_self_transitions=False,
        start_obs=INITIAL_OBS_REFERENCE,
        goal_velocity=GOAL_VELOCITY,
        wait_for_state_change=False,
    )

    mdp = build_mdp(feature_type, state_disc, action_disc)

    row = {
        "feature_type": feature_type,
        "reset_name": reset_config["name"],
        "reset_probs": json.dumps(reset_config["reset_probs"], sort_keys=True),
        "policy_fraction": proportions[0],
        "random_fraction": proportions[1],
        "epsilon": epsilon,
        "n_transitions": n_transitions,
        "dataset_rows": len(dataset_df),
        "status": "ok",
        "error": "",
    }

    try:
        row.update(
            run_fogas(
                mdp=mdp,
                dataset_path=dataset_path,
                state_disc=state_disc,
                action_disc=action_disc,
                device=device,
                seed=seed,
            )
        )
    except Exception as exc:
        row.update({
            "mean_steps": np.nan,
            "successes": np.nan,
            "solver_alpha": FOGAS_CONFIG["alpha"],
            "solver_eta": FOGAS_CONFIG["eta"],
            "solver_rho": FOGAS_CONFIG["rho"],
            "solver_T": FOGAS_CONFIG["T"],
            "solver_beta": FOGAS_CONFIG["beta"],
            "fogas_mean_steps": np.nan,
            "fogas_successes": np.nan,
            "fogas_greedy_mean_steps": np.nan,
            "fogas_greedy_successes": np.nan,
            "fogas_solver_mean_steps": np.nan,
            "fogas_solver_successes": np.nan,
            "fogas_alpha": FOGAS_CONFIG["alpha"],
            "fogas_eta": FOGAS_CONFIG["eta"],
            "fogas_rho": FOGAS_CONFIG["rho"],
            "fogas_T": FOGAS_CONFIG["T"],
            "fogas_beta": FOGAS_CONFIG["beta"],
            "fogas_status": "error",
            "fogas_error": str(exc),
        })

    try:
        row.update(
            run_fqi(
                mdp=mdp,
                dataset_path=dataset_path,
                state_disc=state_disc,
                action_disc=action_disc,
                device=device,
                seed=seed,
            )
        )
    except Exception as exc:
        row.update({
            "fqi_mean_steps": np.nan,
            "fqi_successes": np.nan,
            "fqi_greedy_mean_steps": np.nan,
            "fqi_greedy_successes": np.nan,
            "fqi_solver_mean_steps": np.nan,
            "fqi_solver_successes": np.nan,
            "fqi_K": FQI_CONFIG["K"],
            "fqi_tau": FQI_CONFIG["tau"],
            "fqi_ridge": FQI_CONFIG["ridge"],
            "fqi_augment_terminal_transitions": FQI_CONFIG[
                "augment_terminal_transitions"
            ],
            "fqi_added_terminal_samples": np.nan,
            "fqi_final_theta_delta": np.nan,
            "fqi_status": "error",
            "fqi_error": str(exc),
        })

    if row["fogas_status"] != "ok" or row["fqi_status"] != "ok":
        row["status"] = "partial_error"
        row["error"] = "one_or_more_algorithms_failed"

    return row


def main():
    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    state_disc, action_disc = build_abstraction()
    pi_matrix_ql = build_q_learning_policy(state_disc, action_disc, seed=SEED)
    _, trajectory_reset_distribution, trajectory_steps = (
        GymDataBuffer.create_distribution(
            policy_matrix=pi_matrix_ql,
            state_disc=state_disc,
            action_disc=action_disc,
            env_id=ENV_ID,
            start_obs=INITIAL_OBS_REFERENCE,
            max_steps=TIME_LIMIT,
            seed=SEED,
            goal_velocity=GOAL_VELOCITY,
        )
    )
    print(
        "Built Q-learning policy and custom reset distribution "
        f"from a trajectory of {trajectory_steps} steps."
    )

    total_runs = (
        len(FEATURE_TYPES)
        * len(RESET_CONFIGS)
        * len(DATASET_GRID["proportions"])
        * len(DATASET_GRID["epsilon"])
        * len(DATASET_GRID["n_transitions"])
    )
    results = []

    with tempfile.TemporaryDirectory(prefix="mountaincar_grid_") as temp_dir:
        run_idx = 0
        with tqdm(total=total_runs, desc="MountainCar grid") as pbar:
            for feature_type in FEATURE_TYPES:
                for reset_config in RESET_CONFIGS:
                    for proportions in DATASET_GRID["proportions"]:
                        for epsilon in DATASET_GRID["epsilon"]:
                            for n_transitions in DATASET_GRID["n_transitions"]:
                                run_seed = SEED + run_idx
                                dataset_path = str(
                                    Path(temp_dir)
                                    / (
                                        f"{feature_type}_"
                                        f"{reset_config['name']}_"
                                        f"p{int(100 * proportions[0])}_"
                                        f"eps{str(epsilon).replace('.', '_')}_"
                                        f"n{n_transitions}.csv"
                                    )
                                )
                                try:
                                    row = run_single_experiment(
                                        feature_type=feature_type,
                                        reset_config=reset_config,
                                        proportions=proportions,
                                        epsilon=epsilon,
                                        n_transitions=n_transitions,
                                        state_disc=state_disc,
                                        action_disc=action_disc,
                                        pi_matrix_ql=pi_matrix_ql,
                                        trajectory_reset_distribution=trajectory_reset_distribution,
                                        device=device,
                                        dataset_path=dataset_path,
                                        seed=run_seed,
                                    )
                                except Exception as exc:
                                    row = {
                                        "feature_type": feature_type,
                                        "reset_name": reset_config["name"],
                                        "reset_probs": json.dumps(
                                            reset_config["reset_probs"],
                                            sort_keys=True,
                                        ),
                                        "policy_fraction": proportions[0],
                                        "random_fraction": proportions[1],
                                        "epsilon": epsilon,
                                        "n_transitions": n_transitions,
                                        "dataset_rows": np.nan,
                                        "mean_steps": np.nan,
                                        "successes": np.nan,
                                        "solver_alpha": FOGAS_CONFIG["alpha"],
                                        "solver_eta": FOGAS_CONFIG["eta"],
                                        "solver_rho": FOGAS_CONFIG["rho"],
                                        "solver_T": FOGAS_CONFIG["T"],
                                        "solver_beta": FOGAS_CONFIG["beta"],
                                        "fogas_mean_steps": np.nan,
                                        "fogas_successes": np.nan,
                                        "fogas_greedy_mean_steps": np.nan,
                                        "fogas_greedy_successes": np.nan,
                                        "fogas_solver_mean_steps": np.nan,
                                        "fogas_solver_successes": np.nan,
                                        "fogas_alpha": FOGAS_CONFIG["alpha"],
                                        "fogas_eta": FOGAS_CONFIG["eta"],
                                        "fogas_rho": FOGAS_CONFIG["rho"],
                                        "fogas_T": FOGAS_CONFIG["T"],
                                        "fogas_beta": FOGAS_CONFIG["beta"],
                                        "fogas_status": "not_run",
                                        "fogas_error": "",
                                        "fqi_mean_steps": np.nan,
                                        "fqi_successes": np.nan,
                                        "fqi_greedy_mean_steps": np.nan,
                                        "fqi_greedy_successes": np.nan,
                                        "fqi_solver_mean_steps": np.nan,
                                        "fqi_solver_successes": np.nan,
                                        "fqi_K": FQI_CONFIG["K"],
                                        "fqi_tau": FQI_CONFIG["tau"],
                                        "fqi_ridge": FQI_CONFIG["ridge"],
                                        "fqi_augment_terminal_transitions": FQI_CONFIG[
                                            "augment_terminal_transitions"
                                        ],
                                        "fqi_added_terminal_samples": np.nan,
                                        "fqi_final_theta_delta": np.nan,
                                        "fqi_status": "not_run",
                                        "fqi_error": "",
                                        "status": "error",
                                        "error": str(exc),
                                    }

                                results.append(row)
                                run_idx += 1
                                pbar.update(1)

                                output_df = pd.DataFrame(results)
                                OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
                                output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved grid-search results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
