from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


FOGAS_COLUMNS = ["state", "action", "reward", "next_state"]


def _to_numpy_policy_matrix(policy_matrix, n_states: int, n_actions: int) -> np.ndarray:
    if hasattr(policy_matrix, "detach"):
        policy_matrix = policy_matrix.detach().cpu().numpy()
    else:
        policy_matrix = np.asarray(policy_matrix, dtype=np.float64)

    expected_shape = (n_states, n_actions)
    if policy_matrix.shape != expected_shape:
        raise ValueError(
            f"policy_matrix must have shape {expected_shape}, got {policy_matrix.shape}"
        )

    row_sums = policy_matrix.sum(axis=1, keepdims=True)
    bad_rows = np.isclose(row_sums, 0.0).squeeze()
    if np.any(bad_rows):
        policy_matrix = policy_matrix.copy()
        policy_matrix[bad_rows] = 1.0 / n_actions
        row_sums = policy_matrix.sum(axis=1, keepdims=True)

    return policy_matrix / row_sums


def _make_env(env_id: str, env_kwargs: Optional[dict], goal_velocity):
    kwargs = {} if env_kwargs is None else dict(env_kwargs)
    if goal_velocity is not None and "goal_velocity" not in kwargs:
        kwargs["goal_velocity"] = goal_velocity
    return gym.make(env_id, **kwargs)


class GymDataBuffer:
    """Static-style Gym dataset collector for clean feature-only MDP workflows."""

    @staticmethod
    def create_distribution(
        policy_matrix,
        state_disc,
        action_disc,
        env_id,
        start_obs,
        max_steps=200,
        seed=42,
        goal_velocity=0.0,
        env_kwargs: Optional[dict] = None,
    ):
        policy_matrix = _to_numpy_policy_matrix(
            policy_matrix=policy_matrix,
            n_states=state_disc.n_states,
            n_actions=action_disc.n_actions,
        )

        env = _make_env(
            env_id=env_id,
            env_kwargs=env_kwargs,
            goal_velocity=goal_velocity,
        )
        try:
            env.reset(seed=seed)

            obs = np.asarray(start_obs, dtype=np.float64).copy()
            env.unwrapped.state = obs.copy()

            trajectory_state_ids = []
            steps = 0

            for _ in range(max_steps):
                if state_disc.is_terminal_obs(obs):
                    break

                state_id = state_disc.obs_to_state_id(obs)
                trajectory_state_ids.append(state_id)

                action_id = int(np.argmax(policy_matrix[state_id]))
                env_action = action_disc.action_id_to_env_action(action_id)

                obs, _, terminated, truncated, _ = env.step(env_action)
                obs = np.asarray(obs, dtype=np.float64)

                steps += 1
                if terminated or truncated:
                    break
        finally:
            env.close()

        unique_state_ids = list(dict.fromkeys(trajectory_state_ids))
        if len(unique_state_ids) == 0:
            raise ValueError(
                "The trajectory from the initial state did not visit any non-terminal states."
            )

        uniform_prob = 1.0 / len(unique_state_ids)
        reset_distribution = {int(s): uniform_prob for s in unique_state_ids}
        return unique_state_ids, reset_distribution, steps

    @staticmethod
    def collect(
        policy_matrix,
        state_disc,
        action_disc,
        env_id,
        n_transitions=50_000,
        gamma=0.9,
        epsilon=0.2,
        proportions=(0.7, 0.3),
        episode_based=True,
        max_steps_per_episode=500,
        reset_probs=None,
        custom_reset_distribution=None,
        reset_obs_mode="uniform_in_bin",
        seed=42,
        save_path=None,
        verbose=True,
        max_repeat_same_state=10_000,
        drop_self_transitions=False,
        add_goal_self_loops=True,
        goal_self_loops_per_hit=1,
        goal_self_loop_reward=0.0,
        goal_self_loop_action="sample",
        start_obs=None,
        goal_velocity=0.0,
        env_kwargs: Optional[dict] = None,
        wait_for_state_change=True,
    ):
        rng = np.random.default_rng(seed)
        env = _make_env(
            env_id=env_id,
            env_kwargs=env_kwargs,
            goal_velocity=goal_velocity,
        )

        n_states = state_disc.n_states
        n_actions = action_disc.n_actions
        core_state_count = state_disc.core_state_count
        goal_state_id = state_disc.absorbing_state_id
        bin_edges = state_disc.bin_edges

        if start_obs is None:
            env.close()
            raise ValueError("start_obs must be provided")

        if reset_probs is None:
            reset_probs = {"x0": 0.5, "random": 0.5}

        policy_matrix = _to_numpy_policy_matrix(
            policy_matrix=policy_matrix,
            n_states=n_states,
            n_actions=n_actions,
        )

        p_policy, p_random = proportions
        if not np.isclose(p_policy + p_random, 1.0):
            env.close()
            raise ValueError("proportions must sum to 1.0")

        reset_mode_names = list(reset_probs.keys())
        reset_mode_probs = np.array(list(reset_probs.values()), dtype=np.float64)
        if np.any(reset_mode_probs < 0):
            env.close()
            raise ValueError("reset_probs must be nonnegative")
        if np.isclose(reset_mode_probs.sum(), 0.0):
            env.close()
            raise ValueError("reset_probs must have positive total mass")
        reset_mode_probs = reset_mode_probs / reset_mode_probs.sum()

        if "custom" in reset_mode_names:
            if custom_reset_distribution is None:
                env.close()
                raise ValueError(
                    "custom_reset_distribution must be provided when reset_probs includes 'custom'."
                )
            if not isinstance(custom_reset_distribution, dict):
                env.close()
                raise TypeError(
                    "custom_reset_distribution must be a dict mapping state_id -> probability."
                )

            custom_state_ids = np.array(list(custom_reset_distribution.keys()), dtype=np.int64)
            custom_state_probs = np.array(list(custom_reset_distribution.values()), dtype=np.float64)

            if len(custom_state_ids) == 0:
                env.close()
                raise ValueError("custom_reset_distribution cannot be empty.")
            if np.any(custom_state_ids < 0) or np.any(custom_state_ids >= core_state_count):
                env.close()
                raise ValueError(
                    f"custom reset state ids must be in [0, {core_state_count - 1}]."
                )
            if np.any(custom_state_probs < 0):
                env.close()
                raise ValueError("custom reset probabilities must be nonnegative.")
            if np.isclose(custom_state_probs.sum(), 0.0):
                env.close()
                raise ValueError("custom_reset_distribution must have positive total mass.")

            custom_state_probs = custom_state_probs / custom_state_probs.sum()
        else:
            custom_state_ids = None
            custom_state_probs = None

        def sample_obs_from_state_id(state_id, mode="uniform_in_bin"):
            state_id = int(state_id)
            if goal_state_id is not None and state_id == goal_state_id:
                raise ValueError("Cannot reset directly into GOAL_STATE_ID")

            if mode == "center":
                return state_disc.state_id_to_center_obs(state_id).copy()

            if mode == "uniform_in_bin":
                multi_bin = state_disc.state_id_to_multi_bin(state_id)
                obs = np.empty(len(multi_bin), dtype=np.float64)

                for d, idx in enumerate(multi_bin):
                    lo = bin_edges[d][idx]
                    hi = bin_edges[d][idx + 1]
                    obs[d] = rng.uniform(lo, hi)

                obs = state_disc.clip(obs)

                for _ in range(10):
                    if (not state_disc.is_terminal_obs(obs)) and (
                        state_disc.obs_to_state_id(obs) == state_id
                    ):
                        return obs

                    for d, idx in enumerate(multi_bin):
                        lo = bin_edges[d][idx]
                        hi = bin_edges[d][idx + 1]
                        obs[d] = rng.uniform(lo, hi)
                    obs = state_disc.clip(obs)

                return state_disc.state_id_to_center_obs(state_id).copy()

            raise ValueError(f"Unknown reset_obs_mode: {mode}")

        def reset_env():
            mode = rng.choice(reset_mode_names, p=reset_mode_probs)
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))

            if mode == "x0":
                obs = np.asarray(start_obs, dtype=np.float64).copy()
                env.unwrapped.state = obs.copy()
                return obs, mode

            if mode == "random":
                while state_disc.is_terminal_obs(obs):
                    obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
                return np.asarray(obs, dtype=np.float64), mode

            if mode == "custom":
                sampled_idx = int(rng.choice(len(custom_state_ids), p=custom_state_probs))
                sampled_state = int(custom_state_ids[sampled_idx])
                obs = sample_obs_from_state_id(sampled_state, mode=reset_obs_mode)
                env.unwrapped.state = obs.copy()
                return obs, mode

            raise ValueError(f"Unknown reset mode: {mode}")

        def choose_policy_id():
            return int(rng.choice([0, 1], p=[p_policy, p_random]))

        def choose_action_id(obs, policy_id):
            state_id = state_disc.obs_to_state_id(obs)

            if policy_id == 1:
                return int(rng.integers(n_actions))
            if rng.random() < epsilon:
                return int(rng.integers(n_actions))
            return int(rng.choice(n_actions, p=policy_matrix[state_id]))

        def choose_goal_loop_action_id():
            if goal_self_loop_action == "sample":
                return int(rng.integers(n_actions))
            return int(goal_self_loop_action)

        data = {
            "episode": [],
            "step": [],
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "policy_id": [],
            "reset_mode": [],
        }

        episode = 0
        step_in_episode = 0
        obs, reset_mode = reset_env()
        current_policy_id = choose_policy_id() if episode_based else None

        def append_row(ep, st, state_id, action_id, reward, next_state_id, policy_id, mode):
            data["episode"].append(ep)
            data["step"].append(st)
            data["state"].append(state_id)
            data["action"].append(action_id)
            data["reward"].append(reward)
            data["next_state"].append(next_state_id)
            data["policy_id"].append(policy_id)
            data["reset_mode"].append(mode)

        try:
            with tqdm(total=n_transitions, desc="Collecting dataset") as pbar:
                while len(data["state"]) < n_transitions:
                    state_id = state_disc.obs_to_state_id(obs)
                    policy_id = current_policy_id if episode_based else choose_policy_id()
                    action_id = choose_action_id(obs, policy_id)
                    env_action = action_disc.action_id_to_env_action(action_id)

                    accumulated_reward = 0.0
                    terminated = False
                    truncated = False
                    next_obs = obs
                    primitive_steps = 0

                    while True:
                        next_obs, reward, term, trunc, _ = env.step(env_action)
                        next_obs = np.asarray(next_obs, dtype=np.float64)

                        accumulated_reward += (gamma ** primitive_steps) * float(reward)
                        primitive_steps += 1
                        step_in_episode += 1

                        terminated = terminated or term
                        truncated = truncated or trunc

                        if terminated:
                            next_state_id = goal_state_id
                            break

                        next_state_id = state_disc.obs_to_state_id(next_obs)
                        if not wait_for_state_change:
                            break
                        if next_state_id != state_id:
                            break
                        if truncated or step_in_episode >= max_steps_per_episode:
                            break
                        if primitive_steps >= max_repeat_same_state:
                            break

                    keep_row = (
                        next_state_id != state_id or terminated or (not drop_self_transitions)
                    )

                    if keep_row and len(data["state"]) < n_transitions:
                        append_row(
                            episode,
                            step_in_episode,
                            state_id,
                            action_id,
                            -1.0 * accumulated_reward,
                            next_state_id,
                            policy_id,
                            reset_mode,
                        )
                        pbar.update(1)

                    if terminated and add_goal_self_loops and goal_state_id is not None:
                        for _ in range(goal_self_loops_per_hit):
                            if len(data["state"]) >= n_transitions:
                                break
                            append_row(
                                episode,
                                step_in_episode,
                                goal_state_id,
                                choose_goal_loop_action_id(),
                                goal_self_loop_reward,
                                goal_state_id,
                                policy_id,
                                reset_mode,
                            )
                            pbar.update(1)

                    obs = next_obs

                    if terminated or truncated or step_in_episode >= max_steps_per_episode:
                        episode += 1
                        step_in_episode = 0
                        obs, reset_mode = reset_env()
                        if episode_based:
                            current_policy_id = choose_policy_id()
        finally:
            env.close()

        df = pd.DataFrame(data)

        if verbose:
            counts = df["policy_id"].value_counts(normalize=True).sort_index()
            reset_counts = df["reset_mode"].value_counts(normalize=True)
            print(
                f"Collected {len(df)} transitions over "
                f"{df['episode'].nunique()} episodes"
            )
            print(f"Policy 0 (policy + epsilon exploration): {counts.get(0, 0.0):.3f}")
            print(f"Policy 1 (random):                       {counts.get(1, 0.0):.3f}")
            print("Reset mode frequencies:")
            print(reset_counts.sort_index())

        df_fogas = df[FOGAS_COLUMNS]

        if save_path is not None:
            df_fogas.to_csv(save_path, index=False)
            if verbose:
                print(f"Saved dataset to {save_path}")

        return df_fogas
