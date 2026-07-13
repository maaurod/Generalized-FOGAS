"""
DiscreteDataBuffer
------------------

Offline dataset builder for finite DiscreteMDP-style objects.

The buffer simulates directly from an MDP transition matrix and reward vector.
It intentionally does not accept Gym environments, environment names, or
planners. Planner-owned quantities such as policies and occupancy measures
must be passed explicitly. FOGAS and generalized FOGAS gridworld experiments
use this module to generate reproducible CSV datasets under random, optimal,
epsilon-greedy, mixed-policy, occupancy-based, and coarse/fine collection
schemes.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import pandas as pd


FOGAS_COLUMNS = ["state", "action", "reward", "next_state"]
RESET_MODES = {"x0", "random", "custom", "restricted", "occupancy", "occupancy_uniform"}


def _to_numpy(value, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    out = np.asarray(value, dtype=dtype)
    return out


class DiscreteDataBuffer:
    """
    Collect offline transition datasets from a finite discrete MDP.

    Parameters
    ----------
    mdp:
        DiscreteMDP-style object exposing N, A, x0, P, and r.
    max_steps:
        Maximum simulated steps before truncating an episode.
    terminal_states:
        Optional explicit terminal state indices. If omitted, mdp.terminal_states
        is used when non-empty; otherwise absorbing states are detected from P.
    restricted_states:
        States excluded from normal random resets. They are sampled only by the
        "restricted" reset mode.
    initial_states:
        Custom reset states for the "custom" reset mode.
    reset_probs:
        Mapping from reset mode to probability. Defaults to {"x0": 1.0}.
    occupancy:
        Explicit state or state-action occupancy used by occupancy reset modes.
    seed:
        Random seed for reproducible collection.
    detect_absorbing:
        Whether to detect absorbing terminal states when no terminal states are
        supplied by the caller or MDP.
    """

    def __init__(
        self,
        mdp,
        *,
        max_steps=1000,
        terminal_states=None,
        restricted_states=None,
        initial_states=None,
        reset_probs=None,
        occupancy=None,
        seed=42,
        detect_absorbing=True,
    ):
        self._validate_mdp_input(mdp)

        self.mdp = mdp
        self.N = int(mdp.N)
        self.A = int(mdp.A)
        self.x0 = int(mdp.x0)
        self.max_steps = int(max_steps)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self._validate_state_id(self.x0, "mdp.x0")

        self.P = _to_numpy(mdp.P, dtype=np.float64).copy()
        self.r = _to_numpy(mdp.r, dtype=np.float64).reshape(-1).copy()
        self._validate_model_arrays()

        self.terminal_states = self._resolve_terminal_states(
            terminal_states=terminal_states,
            detect_absorbing=detect_absorbing,
        )

        self.restricted_states = self._coerce_state_set(
            restricted_states,
            name="restricted_states",
            allow_terminal=False,
        )
        self.initial_states = self._coerce_state_list(
            initial_states,
            name="initial_states",
            allow_terminal=False,
        )
        custom_restricted_overlap = set(self.initial_states) & self.restricted_states
        if custom_restricted_overlap:
            raise ValueError(
                "initial_states cannot include restricted states; use reset mode "
                f"'restricted' for those starts: {sorted(custom_restricted_overlap)}"
            )

        self.forbidden_resets = set(self.terminal_states) | set(self.restricted_states)
        self.valid_start_states = [
            s for s in range(self.N) if s not in self.forbidden_resets
        ]
        self.restricted_start_states = [
            s for s in self.restricted_states if s not in self.terminal_states
        ]

        self.reset_probs = self._normalize_reset_probs(reset_probs)
        self.occupancy = self._coerce_occupancy(occupancy)
        self._validate_reset_requirements()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Validation and setup
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_mdp_input(mdp):
        if hasattr(mdp, "mdp") and hasattr(mdp, "pi_star"):
            raise ValueError(
                "DiscreteDataBuffer expects a DiscreteMDP, not a Planner. "
                "Pass planner.mdp and pass planner.pi_star/occupancy explicitly."
            )

        required = ("N", "A", "x0", "P", "r")
        missing = [name for name in required if not hasattr(mdp, name)]
        if missing:
            raise TypeError(
                "DiscreteDataBuffer expects a DiscreteMDP-style object exposing "
                f"{required}. Missing: {missing}"
            )

    def _validate_model_arrays(self):
        expected_p_shape = (self.N * self.A, self.N)
        if self.P.shape != expected_p_shape:
            raise ValueError(f"mdp.P must have shape {expected_p_shape}, got {self.P.shape}.")
        if self.r.shape != (self.N * self.A,):
            raise ValueError(f"mdp.r must have shape ({self.N * self.A},), got {self.r.shape}.")
        if np.any(self.P < -1e-10):
            raise ValueError("mdp.P contains negative probabilities.")
        row_sums = self.P.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            bad = np.where(np.abs(row_sums - 1.0) > 1e-6)[0][:10].tolist()
            raise ValueError(f"Rows of mdp.P must sum to 1. Bad row indices: {bad}")
        self.P = np.maximum(self.P, 0.0)
        self.P = self.P / self.P.sum(axis=1, keepdims=True)

    def _validate_state_id(self, state, name):
        state = int(state)
        if state < 0 or state >= self.N:
            raise ValueError(f"{name} must be in [0, {self.N - 1}], got {state}.")

    def _validate_action_id(self, action, name="action"):
        action = int(action)
        if action < 0 or action >= self.A:
            raise ValueError(f"{name} must be in [0, {self.A - 1}], got {action}.")
        return action

    def _coerce_state_set(self, states, *, name, allow_terminal):
        if states is None:
            return set()
        out = {int(s) for s in states}
        for state in out:
            self._validate_state_id(state, name)
        if not allow_terminal:
            terminal_overlap = out & getattr(self, "terminal_states", set())
            if terminal_overlap:
                raise ValueError(f"{name} cannot include terminal states: {sorted(terminal_overlap)}")
        return out

    def _coerce_state_list(self, states, *, name, allow_terminal):
        if states is None:
            return []
        out = [int(s) for s in states]
        for state in out:
            self._validate_state_id(state, name)
        if not allow_terminal:
            terminal_overlap = set(out) & getattr(self, "terminal_states", set())
            if terminal_overlap:
                raise ValueError(f"{name} cannot include terminal states: {sorted(terminal_overlap)}")
        return out

    def _resolve_terminal_states(self, *, terminal_states, detect_absorbing):
        if terminal_states is not None:
            return self._coerce_terminal_states(terminal_states)

        mdp_terminal_states = getattr(self.mdp, "terminal_states", None)
        if mdp_terminal_states:
            return self._coerce_terminal_states(mdp_terminal_states)

        if detect_absorbing:
            return self._detect_absorbing_states()

        return set()

    def _coerce_terminal_states(self, states):
        out = {int(s) for s in states}
        for state in out:
            self._validate_state_id(state, "terminal_states")
        return out

    def _detect_absorbing_states(self, tol=1e-8):
        absorbing = set()
        for state in range(self.N):
            is_absorbing = True
            for action in range(self.A):
                probs = self.P[self._row_index(state, action)]
                if not np.isclose(probs[state], 1.0, atol=tol):
                    is_absorbing = False
                    break
                if np.any(np.delete(probs, state) > tol):
                    is_absorbing = False
                    break
            if is_absorbing:
                absorbing.add(state)
        return absorbing

    def _normalize_reset_probs(self, reset_probs):
        if reset_probs is None:
            reset_probs = {"x0": 1.0}
        if not isinstance(reset_probs, dict):
            raise TypeError("reset_probs must be a dict mapping reset mode to probability.")
        if not reset_probs:
            raise ValueError("reset_probs cannot be empty.")

        unknown = set(reset_probs) - RESET_MODES
        if unknown:
            raise ValueError(f"Unknown reset mode(s): {sorted(unknown)}")

        modes = list(reset_probs.keys())
        probs = np.array([reset_probs[mode] for mode in modes], dtype=np.float64)
        if np.any(probs < 0):
            raise ValueError("reset_probs must be nonnegative.")
        total = probs.sum()
        if np.isclose(total, 0.0):
            raise ValueError("reset_probs must have positive total mass.")
        return dict(zip(modes, (probs / total).tolist()))

    def _coerce_occupancy(self, occupancy):
        if occupancy is None:
            return None

        occ = _to_numpy(occupancy, dtype=np.float64)
        if occ.shape == (self.N,):
            state_occ = occ.copy()
        elif occ.shape == (self.N, self.A):
            state_occ = occ.sum(axis=1)
        elif occ.ndim == 1 and occ.shape == (self.N * self.A,):
            state_occ = occ.reshape(self.N, self.A).sum(axis=1)
        else:
            raise ValueError(
                "occupancy must have shape (N,), (N, A), or (N*A,), "
                f"got {occ.shape}."
            )

        if np.any(state_occ < -1e-12):
            raise ValueError("occupancy cannot contain negative probabilities.")
        return np.maximum(state_occ, 0.0)

    def _validate_reset_requirements(self):
        modes = {mode for mode, prob in self.reset_probs.items() if prob > 0.0}
        if "random" in modes and not self.valid_start_states:
            raise ValueError("reset mode 'random' requires at least one non-terminal, non-restricted state.")
        if "custom" in modes and not self.initial_states:
            raise ValueError("reset mode 'custom' requires non-empty initial_states.")
        if "restricted" in modes and not self.restricted_start_states:
            raise ValueError("reset mode 'restricted' requires non-terminal restricted_states.")
        if {"occupancy", "occupancy_uniform"} & modes and self.occupancy is None:
            raise ValueError("occupancy reset modes require an explicit occupancy argument.")

    # ------------------------------------------------------------------
    # Policies
    # ------------------------------------------------------------------
    class RandomPolicy:
        def __init__(self, n_actions, rng):
            self.n_actions = int(n_actions)
            self.rng = rng

        def sample(self, state):
            return int(self.rng.integers(self.n_actions))

    class MatrixPolicy:
        def __init__(self, policy_matrix, n_states, n_actions, rng):
            self.rng = rng
            matrix = _to_numpy(policy_matrix, dtype=np.float64)
            expected = (int(n_states), int(n_actions))
            if matrix.shape != expected:
                raise ValueError(f"policy matrix must have shape {expected}, got {matrix.shape}.")
            if np.any(matrix < -1e-12):
                raise ValueError("policy matrix cannot contain negative probabilities.")
            matrix = np.maximum(matrix, 0.0)
            row_sums = matrix.sum(axis=1, keepdims=True)
            bad_rows = np.isclose(row_sums, 0.0).squeeze()
            if np.any(bad_rows):
                matrix = matrix.copy()
                matrix[bad_rows] = 1.0 / expected[1]
                row_sums = matrix.sum(axis=1, keepdims=True)
            self.policy_matrix = matrix / row_sums

        def sample(self, state):
            state = int(state)
            return int(self.rng.choice(self.policy_matrix.shape[1], p=self.policy_matrix[state]))

    class EpsilonGreedyPolicy:
        def __init__(self, base_policy, epsilon, n_actions, rng):
            self.base_policy = base_policy
            self.epsilon = float(epsilon)
            self.n_actions = int(n_actions)
            self.rng = rng
            if self.epsilon < 0.0 or self.epsilon > 1.0:
                raise ValueError("epsilon must be in [0, 1].")

        def sample(self, state):
            if self.rng.random() < self.epsilon:
                return int(self.rng.integers(self.n_actions))
            return self.base_policy.sample(state)

    def _make_policy(self, policy):
        if isinstance(policy, tuple) and len(policy) == 2:
            base_policy, epsilon = policy
            return self.EpsilonGreedyPolicy(
                self._make_policy(base_policy),
                epsilon,
                self.A,
                self.rng,
            )

        if isinstance(policy, str):
            name = policy.lower()
            if name in {"random", "uniform"}:
                return self.RandomPolicy(self.A, self.rng)
            raise ValueError("Unknown policy name. Supported: 'random', 'uniform'.")

        if hasattr(policy, "sample"):
            return policy

        return self.MatrixPolicy(policy, self.N, self.A, self.rng)

    # ------------------------------------------------------------------
    # Direct simulation
    # ------------------------------------------------------------------
    def _row_index(self, state, action):
        return int(state) * self.A + int(action)

    def _model_reward(self, state, action):
        return float(self.r[self._row_index(state, action)])

    def _step_from(self, state, action, step_count):
        state = int(state)
        action = self._validate_action_id(action)

        if state in self.terminal_states:
            return state, self._model_reward(state, action), True, False

        probs = self.P[self._row_index(state, action)]
        next_state = int(self.rng.choice(self.N, p=probs))
        reward = self._model_reward(state, action)
        terminated = next_state in self.terminal_states
        truncated = int(step_count) + 1 >= self.max_steps
        return next_state, reward, terminated, truncated

    def _sample_reset(self):
        modes = list(self.reset_probs.keys())
        probs = np.array([self.reset_probs[mode] for mode in modes], dtype=np.float64)
        mode = str(self.rng.choice(modes, p=probs))

        if mode == "x0":
            return self.x0, mode, False
        if mode == "random":
            return int(self.rng.choice(self.valid_start_states)), mode, False
        if mode == "custom":
            return int(self.rng.choice(self.initial_states)), mode, False
        if mode == "restricted":
            return int(self.rng.choice(self.restricted_start_states)), mode, True
        if mode == "occupancy":
            return self._sample_occupancy_state(uniform=False), mode, False
        if mode == "occupancy_uniform":
            return self._sample_occupancy_state(uniform=True), mode, False

        raise ValueError(f"Unknown reset mode: {mode}")

    def _sample_occupancy_state(self, *, uniform):
        probs = self.occupancy.astype(np.float64).copy()
        for state in self.forbidden_resets:
            probs[state] = 0.0

        if uniform:
            touched = np.where(probs > 1e-12)[0]
            if len(touched) == 0:
                raise ValueError("occupancy_uniform reset has no valid occupied states after masking.")
            return int(self.rng.choice(touched))

        total = probs.sum()
        if total <= 1e-12:
            raise ValueError("occupancy reset has no positive mass after masking.")
        probs = probs / total
        return int(self.rng.choice(self.N, p=probs))

    def _select_policy_index(self, proportions):
        return int(self.rng.choice(len(proportions), p=proportions))

    @staticmethod
    def _fine_to_coarse_state(x_fine, *, fine_size, coarse_size, factor):
        r_f, c_f = divmod(int(x_fine), int(fine_size))
        r_c, c_c = r_f // int(factor), c_f // int(factor)
        return int(r_c * int(coarse_size) + c_c)

    @staticmethod
    def _save_fogas_csv(df, save_path):
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        df[FOGAS_COLUMNS].to_csv(save_path, index=False)

    # ------------------------------------------------------------------
    # Collection API
    # ------------------------------------------------------------------
    def collect(
        self,
        *,
        policy="random",
        policies=None,
        proportions=None,
        n_steps=1000,
        episode_based=True,
        extra_terminal_steps=0,
        restricted_max_steps=1,
        save_path=None,
        verbose=True,
    ):
        """
        Collect one-step transitions under a single or mixed behavior policy.

        Saved CSVs contain only FOGAS columns. The returned DataFrame also
        includes metadata useful for diagnostics.
        """
        n_steps = int(n_steps)
        extra_terminal_steps = int(extra_terminal_steps)
        restricted_max_steps = int(restricted_max_steps)

        if n_steps < 0:
            raise ValueError("n_steps must be nonnegative.")
        if extra_terminal_steps < 0:
            raise ValueError("extra_terminal_steps must be nonnegative.")
        if restricted_max_steps <= 0:
            raise ValueError("restricted_max_steps must be positive.")

        if policies is None:
            raw_policies = [policy]
        else:
            raw_policies = list(policies)
            if len(raw_policies) == 0:
                raise ValueError("policies cannot be empty.")

        policy_objects = [self._make_policy(p) for p in raw_policies]
        proportions = self._normalize_proportions(proportions, len(policy_objects))

        data = {
            "episode": [],
            "step": [],
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "policy_id": [],
            "reset_mode": [],
            "restricted_start": [],
        }

        episode = 0
        step = 0
        state, reset_mode, restricted_start = self._sample_reset()
        restricted_steps = 0
        terminal_extra_remaining: Optional[int] = None

        current_policy_idx = 0
        if episode_based:
            current_policy_idx = self._select_policy_index(proportions)

        def reset_episode():
            nonlocal episode, step, state, reset_mode, restricted_start
            nonlocal restricted_steps, terminal_extra_remaining, current_policy_idx
            episode += 1
            step = 0
            state, reset_mode, restricted_start = self._sample_reset()
            restricted_steps = 0
            terminal_extra_remaining = None
            if episode_based:
                current_policy_idx = self._select_policy_index(proportions)

        for _ in range(n_steps):
            if restricted_start:
                policy_id = -1
                action = int(self.rng.integers(self.A))
                restricted_steps += 1
            else:
                if episode_based:
                    policy_id = current_policy_idx
                else:
                    policy_id = self._select_policy_index(proportions)
                action = self._validate_action_id(policy_objects[policy_id].sample(state))

            next_state, reward, terminated, truncated = self._step_from(state, action, step)
            if restricted_start and restricted_steps >= restricted_max_steps:
                truncated = True

            data["episode"].append(episode)
            data["step"].append(step)
            data["state"].append(int(state))
            data["action"].append(int(action))
            data["reward"].append(float(reward))
            data["next_state"].append(int(next_state))
            data["policy_id"].append(int(policy_id))
            data["reset_mode"].append(reset_mode)
            data["restricted_start"].append(bool(restricted_start))

            step += 1

            if truncated:
                reset_episode()
            elif terminated:
                if terminal_extra_remaining is None:
                    terminal_extra_remaining = extra_terminal_steps
                else:
                    terminal_extra_remaining -= 1

                if terminal_extra_remaining > 0:
                    state = next_state
                else:
                    reset_episode()
            else:
                state = next_state

        df = pd.DataFrame(data)

        if save_path is not None:
            self._save_fogas_csv(df, save_path)

        if verbose:
            self._print_collect_summary(df, save_path=save_path)

        return df

    def collect_macro_dataset_n_repeated_actions(
        self,
        *,
        policy="random",
        n_macro_steps=1000,
        gamma=0.99,
        fine_size=20,
        coarse_size=10,
        factor=2,
        n_repeats=None,
        save_path=None,
        verbose=True,
    ):
        """
        Collect macro transitions by repeating the same fine action.

        Each row saved to CSV is:
            coarse(s_t), a_t, sum_k gamma^k r_{t+k}, coarse(s_{t+n_repeats})

        The returned DataFrame includes diagnostic fine-state and macro metadata.
        """
        n_macro_steps = int(n_macro_steps)
        fine_size = int(fine_size)
        coarse_size = int(coarse_size)
        factor = int(factor)
        n_repeats = factor if n_repeats is None else int(n_repeats)

        if n_macro_steps < 0:
            raise ValueError("n_macro_steps must be nonnegative.")
        if n_repeats <= 0:
            raise ValueError("n_repeats must be positive.")
        if fine_size <= 0 or coarse_size <= 0 or factor <= 0:
            raise ValueError("fine_size, coarse_size, and factor must be positive.")
        if coarse_size * factor != fine_size:
            raise ValueError(
                "Expected fine_size == coarse_size * factor, got "
                f"{fine_size} != {coarse_size} * {factor}."
            )
        if fine_size * fine_size != self.N:
            raise ValueError(
                "fine_size must match the MDP state count, got "
                f"{fine_size}^2 != {self.N}."
            )

        policy_object = self._make_policy(policy)

        data = {
            "episode": [],
            "macro_step": [],
            "fine_state": [],
            "state": [],
            "action": [],
            "reward": [],
            "next_fine_state": [],
            "next_state": [],
            "macro_complete": [],
            "reset_mode": [],
            "restricted_start": [],
        }

        episode = 0
        macro_step = 0
        fine_step = 0
        state, reset_mode, restricted_start = self._sample_reset()

        for _ in range(n_macro_steps):
            fine_state = int(state)
            coarse_state = self._fine_to_coarse_state(
                fine_state,
                fine_size=fine_size,
                coarse_size=coarse_size,
                factor=factor,
            )

            if restricted_start:
                action = int(self.rng.integers(self.A))
            else:
                action = self._validate_action_id(policy_object.sample(state))

            macro_reward = 0.0
            any_terminated = False
            any_truncated = False

            for k in range(n_repeats):
                next_state, reward, terminated, truncated = self._step_from(
                    state,
                    action,
                    fine_step,
                )
                macro_reward += (float(gamma) ** k) * float(reward)

                state = next_state
                fine_step += 1

                if terminated:
                    any_terminated = True
                if truncated:
                    any_truncated = True
                    break

            next_fine_state = int(state)
            next_coarse_state = self._fine_to_coarse_state(
                next_fine_state,
                fine_size=fine_size,
                coarse_size=coarse_size,
                factor=factor,
            )

            data["episode"].append(episode)
            data["macro_step"].append(macro_step)
            data["fine_state"].append(fine_state)
            data["state"].append(coarse_state)
            data["action"].append(int(action))
            data["reward"].append(float(macro_reward))
            data["next_fine_state"].append(next_fine_state)
            data["next_state"].append(next_coarse_state)
            data["macro_complete"].append(not any_truncated)
            data["reset_mode"].append(reset_mode)
            data["restricted_start"].append(bool(restricted_start))

            macro_step += 1

            if any_terminated or any_truncated:
                episode += 1
                macro_step = 0
                fine_step = 0
                state, reset_mode, restricted_start = self._sample_reset()

        df = pd.DataFrame(data)

        if save_path is not None:
            self._save_fogas_csv(df, save_path)

        if verbose:
            print(f"Collected {len(df)} macro transitions ({n_repeats} fine steps each).")
            if save_path is not None:
                print(f"Saved FOGAS macro dataset to {save_path}")

        return df

    @staticmethod
    def _normalize_proportions(proportions, n_policies):
        if proportions is None:
            return np.ones(n_policies, dtype=np.float64) / n_policies

        out = np.asarray(proportions, dtype=np.float64)
        if out.shape != (n_policies,):
            raise ValueError(f"proportions must have shape ({n_policies},), got {out.shape}.")
        if np.any(out < 0):
            raise ValueError("proportions must be nonnegative.")
        total = out.sum()
        if not np.isclose(total, 1.0):
            raise ValueError(f"proportions must sum to 1.0, got {total}.")
        return out

    @staticmethod
    def _print_collect_summary(df, *, save_path):
        print(f"Collected {len(df)} transitions over {df['episode'].nunique()} episodes.")
        if len(df) > 0:
            print("Policy distribution:")
            print(df["policy_id"].value_counts(normalize=True).sort_index())
            print("Reset mode distribution:")
            print(df["reset_mode"].value_counts(normalize=True).sort_index())
        if save_path is not None:
            print(f"Saved FOGAS dataset to {save_path}")

    def collect_uniform(self, samples_per_pair=1, save_path=None, verbose=True):
        """
        Collect exactly samples_per_pair transitions for every state-action pair.
        """
        samples_per_pair = int(samples_per_pair)
        if samples_per_pair <= 0:
            raise ValueError("samples_per_pair must be positive.")

        rows = []
        episode = 0
        for state in range(self.N):
            for action in range(self.A):
                for _ in range(samples_per_pair):
                    next_state, reward, _, _ = self._step_from(state, action, step_count=0)
                    rows.append(
                        {
                            "episode": episode,
                            "step": 0,
                            "state": state,
                            "action": action,
                            "reward": reward,
                            "next_state": next_state,
                        }
                    )
                    episode += 1

        df = pd.DataFrame(rows)
        if save_path is not None:
            self._save_fogas_csv(df, save_path)
        if verbose:
            print(f"Collected {len(df)} uniform transitions.")
            if save_path is not None:
                print(f"Saved FOGAS dataset to {save_path}")
        return df

    def collect_manual(
        self,
        states,
        samples_per_pair=10,
        save_path=None,
        append=False,
        verbose=True,
    ):
        """
        Collect samples_per_pair transitions for every action in the provided states.
        """
        samples_per_pair = int(samples_per_pair)
        if samples_per_pair <= 0:
            raise ValueError("samples_per_pair must be positive.")

        states = self._coerce_state_list(states, name="states", allow_terminal=True)
        rows = []
        episode = -1
        for state in states:
            for action in range(self.A):
                for _ in range(samples_per_pair):
                    next_state, reward, _, _ = self._step_from(state, action, step_count=0)
                    rows.append(
                        {
                            "episode": episode,
                            "step": 0,
                            "state": state,
                            "action": action,
                            "reward": reward,
                            "next_state": next_state,
                        }
                    )
                    episode -= 1

        df = pd.DataFrame(rows)
        if save_path is not None:
            if append and os.path.exists(save_path):
                existing = pd.read_csv(save_path)
                combined = pd.concat([existing[FOGAS_COLUMNS], df[FOGAS_COLUMNS]], ignore_index=True)
                self._save_fogas_csv(combined, save_path)
            else:
                self._save_fogas_csv(df, save_path)

        if verbose:
            print(f"Collected {len(df)} manual transitions.")
            if save_path is not None:
                action = "Appended to" if append else "Saved FOGAS dataset to"
                print(f"{action} {save_path}")
        return df
