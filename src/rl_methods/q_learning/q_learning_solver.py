"""Tabular Q-learning baseline.

This module is used when an experiment can map observations to a finite state
id. The code keeps the baseline explicit: epsilon-greedy collection, one Q-table
update per environment step, and a greedy policy read out at the end.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch


@dataclass
class QLearningResult:
    """Training result returned by `QLearningSolver.run`."""

    q_values: torch.Tensor
    greedy_actions: torch.Tensor
    rewards_per_episode: np.ndarray
    mean_rewards: np.ndarray


class QLearningSolver:
    """
    Tabular Q-learning for Gymnasium environments with a user-provided
    abstraction obs -> state_id.
    """

    def __init__(
        self,
        env_factory: Callable[..., gym.Env],
        n_states: int,
        n_actions: int,
        obs_to_state_id: Optional[Callable] = None,
        terminal_state_id: Optional[int] = None,
        initial_state_id: Optional[int] = None,
        action_id_to_label: Optional[Callable[[int], str]] = None,
        default_gamma: Optional[float] = None,
        seed: Optional[int] = 42,
    ):
        self.env_factory = env_factory
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.obs_to_state_id = obs_to_state_id
        self.terminal_state_id = None if terminal_state_id is None else int(terminal_state_id)
        self.initial_state_id = None if initial_state_id is None else int(initial_state_id)
        self.action_id_to_label = action_id_to_label
        self.default_gamma = default_gamma
        self.seed = seed

        if self.n_states <= 0:
            raise ValueError("n_states must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")

    @classmethod
    def from_env_id(
        cls,
        env_id: str,
        n_states: int,
        n_actions: int,
        obs_to_state_id: Optional[Callable] = None,
        terminal_state_id: Optional[int] = None,
        initial_state_id: Optional[int] = None,
        action_id_to_label: Optional[Callable[[int], str]] = None,
        env_kwargs: Optional[dict] = None,
        seed: Optional[int] = 42,
    ) -> "QLearningSolver":
        env_kwargs = {} if env_kwargs is None else dict(env_kwargs)

        def env_factory(render: bool = False):
            kwargs = dict(env_kwargs)
            kwargs["render_mode"] = "human" if render else None
            return gym.make(env_id, **kwargs)

        return cls(
            env_factory=env_factory,
            n_states=n_states,
            n_actions=n_actions,
            obs_to_state_id=obs_to_state_id,
            terminal_state_id=terminal_state_id,
            initial_state_id=initial_state_id,
            action_id_to_label=action_id_to_label,
            default_gamma=None,
            seed=seed,
        )

    def run(
        self,
        episodes: int = 5000,
        alpha: float = 0.9,
        gamma: Optional[float] = None,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.0,
        epsilon_decay_rate: Optional[float] = None,
        reward_floor: float = -1000.0,
        render: bool = False,
        seed: Optional[int] = None,
        plot: bool = False,
        plot_window: int = 100,
        plot_title: str = "Q-learning",
        print_summary: bool = True,
    ) -> QLearningResult:
        if episodes <= 0:
            raise ValueError("episodes must be positive")

        gamma = self.default_gamma if gamma is None else gamma
        if gamma is None:
            raise ValueError("gamma must be provided when no default_gamma is available")

        run_seed = self.seed if seed is None else seed
        if run_seed is not None:
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)

        env = self._make_env(render=render)
        rng = np.random.default_rng(run_seed)
        q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        rewards_per_episode = np.zeros(episodes, dtype=np.float64)

        # Linear epsilon decay is the default used by the older notebook code.
        # A custom decay can be passed when an experiment needs a fixed schedule.
        epsilon = float(epsilon_start)
        if epsilon_decay_rate is None:
            epsilon_decay_rate = 2.0 / episodes if episodes > 0 else 0.0

        for ep in range(episodes):
            obs, _ = env.reset(seed=None if run_seed is None else run_seed + ep)
            s = self._map_obs_to_state_id(obs)

            terminated = False
            ep_reward = 0.0

            while (not terminated) and (ep_reward > reward_floor):
                # Behavior policy: epsilon-greedy with respect to the current
                # table. This is the only exploration mechanism in the solver.
                if rng.random() < epsilon:
                    a = self._sample_action(env, rng)
                else:
                    a = int(np.argmax(q[s]))

                next_obs, reward, terminated, truncated, _ = env.step(a)

                if terminated and self.terminal_state_id is not None:
                    next_s = self.terminal_state_id
                    target = reward + gamma * np.max(q[next_s])
                elif terminated:
                    next_s = None
                    target = reward
                else:
                    next_s = self._map_obs_to_state_id(next_obs)
                    target = reward + gamma * np.max(q[next_s])

                # One-step tabular Q-learning update.
                q[s, a] = q[s, a] + alpha * (target - q[s, a])

                if next_s is not None:
                    s = next_s
                ep_reward += reward

                if truncated:
                    break

            epsilon = max(epsilon - epsilon_decay_rate, epsilon_end)
            rewards_per_episode[ep] = ep_reward

        env.close()

        q_torch = torch.from_numpy(q).to(dtype=torch.float64)
        greedy_actions = torch.argmax(q_torch, dim=1)
        mean_rewards = self._rolling_mean(rewards_per_episode, plot_window)

        if plot:
            self._plot_rewards(mean_rewards, plot_title=plot_title, plot_window=plot_window)

        if print_summary:
            self._print_summary(q_torch, greedy_actions)

        return QLearningResult(
            q_values=q_torch,
            greedy_actions=greedy_actions,
            rewards_per_episode=rewards_per_episode,
            mean_rewards=mean_rewards,
        )

    def _make_env(self, render: bool):
        try:
            return self.env_factory(render=render)
        except TypeError:
            return self.env_factory()

    def _map_obs_to_state_id(self, obs) -> int:
        if self.obs_to_state_id is None:
            if np.isscalar(obs):
                return int(obs)
            raise ValueError(
                "obs_to_state_id is required when environment observations are not scalar state ids"
            )
        return int(self.obs_to_state_id(obs))

    def _sample_action(self, env, rng: np.random.Generator) -> int:
        return int(rng.integers(self.n_actions))

    def _print_summary(self, q_values: torch.Tensor, greedy_actions: torch.Tensor) -> None:
        if self.initial_state_id is not None:
            if self.action_id_to_label is not None:
                label = self.action_id_to_label(int(greedy_actions[self.initial_state_id].item()))
            else:
                label = int(greedy_actions[self.initial_state_id].item())
            print("Initial state greedy action:", label)
            print("Q(initial state):", q_values[self.initial_state_id])

    @staticmethod
    def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return values.copy()
        return np.array(
            [values[max(0, t - window + 1): t + 1].mean() for t in range(len(values))],
            dtype=np.float64,
        )

    @staticmethod
    def _plot_rewards(mean_rewards: np.ndarray, plot_title: str, plot_window: int) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(mean_rewards)
        plt.xlabel("Episode")
        plt.ylabel(f"Mean reward (last {plot_window})")
        plt.title(plot_title)
        plt.show()


def run_q_learning(
    episodes: int = 5000,
    alpha: float = 0.9,
    gamma: Optional[float] = None,
    epsilon_start: float = 1.0,
    render: bool = False,
    seed: Optional[int] = 44,
    *,
    env_id: Optional[str] = None,
    env_kwargs: Optional[dict] = None,
    env_factory: Optional[Callable[..., gym.Env]] = None,
    obs_to_state_id: Optional[Callable] = None,
    n_states: Optional[int] = None,
    n_actions: Optional[int] = None,
    terminal_state_id: Optional[int] = None,
    initial_state_id: Optional[int] = None,
    action_id_to_label: Optional[Callable[[int], str]] = None,
    epsilon_end: float = 0.0,
    epsilon_decay_rate: Optional[float] = None,
    reward_floor: float = -1000.0,
    plot: bool = True,
    plot_window: int = 100,
    plot_title: str = "Q-learning",
    print_summary: bool = True,
):
    """
    Convenience wrapper returning the same tuple as the original notebook code:
        q_values, greedy_actions, rewards_per_episode

    Usage patterns:
    - run_q_learning(env_id=\"MountainCar-v0\", obs_to_state_id=..., n_states=..., ...)
    - run_q_learning(env_factory=my_env_factory, obs_to_state_id=..., n_states=..., ...)
    """
    if env_factory is not None:
        if n_states is None or n_actions is None:
            raise ValueError("n_states and n_actions are required when env_factory is used")
        solver = QLearningSolver(
            env_factory=env_factory,
            n_states=n_states,
            n_actions=n_actions,
            obs_to_state_id=obs_to_state_id,
            terminal_state_id=terminal_state_id,
            initial_state_id=initial_state_id,
            action_id_to_label=action_id_to_label,
            seed=seed,
        )
    elif env_id is not None:
        if n_states is None or n_actions is None:
            raise ValueError("n_states and n_actions are required when env_id is used")
        solver = QLearningSolver.from_env_id(
            env_id=env_id,
            n_states=n_states,
            n_actions=n_actions,
            obs_to_state_id=obs_to_state_id,
            terminal_state_id=terminal_state_id,
            initial_state_id=initial_state_id,
            action_id_to_label=action_id_to_label,
            env_kwargs=env_kwargs,
            seed=seed,
        )
    else:
        raise ValueError("Provide one of: env_id or env_factory")

    result = solver.run(
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_rate=epsilon_decay_rate,
        reward_floor=reward_floor,
        render=render,
        seed=seed,
        plot=plot,
        plot_window=plot_window,
        plot_title=plot_title,
        print_summary=print_summary,
    )
    return result.q_values, result.greedy_actions, result.rewards_per_episode
