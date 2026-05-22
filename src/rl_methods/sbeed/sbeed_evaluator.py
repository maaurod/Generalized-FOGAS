"""Evaluation helpers for finite discrete SBEED experiments."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

from .sbeed_base import SBEEDSolverProtocol

ArrayLike = Union[np.ndarray, torch.Tensor]


class SBEEDEvaluator:
    """
    Minimal evaluator for discrete SBEED value accuracy.

    The only reported metric is the squared L2 distance to the ordinary
    optimal value function:

        ||V_solver - V*||_2^2 = sum_s (V_solver(s) - V*(s))^2

    The evaluator needs a finite tabular model. Provide it as:
        - P and R arrays,
        - mdp=... with .P and .r,
        - or deterministic transition/reward functions.
    """

    def __init__(
        self,
        solver: SBEEDSolverProtocol,
        P: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
        mdp: Optional[Any] = None,
        next_state_fn: Optional[Callable[[int, int], Any]] = None,
        transition_fn: Optional[Callable[[int, int], Any]] = None,
        reward_fn: Optional[Callable[..., float]] = None,
        terminal_states: Optional[set] = None,
        **_: Any,
    ):
        self.solver = solver
        self.n_states = int(solver.n_states)
        self.n_actions = int(solver.n_actions)
        self.gamma = float(solver.gamma)
        self.x0 = getattr(getattr(solver, "spec", None), "x0", None)
        self.terminal_states = set() if terminal_states is None else {int(s) for s in terminal_states}

        if mdp is not None:
            P = getattr(mdp, "P", P)
            R = getattr(mdp, "r", R)
            if hasattr(mdp, "x0") and self.x0 is None:
                self.x0 = int(mdp.x0)

        if P is not None or R is not None:
            if P is None or R is None:
                raise ValueError("P and R must be provided together")
            self.P = self._as_transition_tensor(P)
            self.R = self._as_reward_matrix(R)
        else:
            model_fn = next_state_fn if next_state_fn is not None else transition_fn
            if model_fn is None:
                raise ValueError(
                    "SBEEDEvaluator needs a model. Pass P and R, mdp=..., "
                    "or next_state_fn/transition_fn with reward_fn."
                )
            self.P, self.R = self._model_from_deterministic_functions(model_fn, reward_fn)

        if self.terminal_states:
            self._apply_terminal_states()

    @staticmethod
    def _to_numpy(x: ArrayLike) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _as_policy_matrix(self, pi: ArrayLike) -> np.ndarray:
        pi_np = self._to_numpy(pi).astype(np.float64, copy=True)
        if pi_np.shape != (self.n_states, self.n_actions):
            raise ValueError(f"pi must have shape ({self.n_states}, {self.n_actions})")
        if not np.all(np.isfinite(pi_np)):
            raise ValueError("pi contains non-finite probabilities")
        if np.any(pi_np < -1e-12):
            raise ValueError("pi contains negative action probabilities")

        pi_np = np.clip(pi_np, 0.0, None)
        row_sums = pi_np.sum(axis=1, keepdims=True)
        bad_rows = np.where(row_sums[:, 0] <= 0.0)[0]
        if bad_rows.size:
            raise ValueError(f"pi rows must have positive probability mass; bad rows: {bad_rows.tolist()}")

        return pi_np / row_sums

    def _as_transition_tensor(self, P: ArrayLike) -> np.ndarray:
        P_np = self._to_numpy(P).astype(np.float64, copy=False)
        if P_np.shape == (self.n_states * self.n_actions, self.n_states):
            P_np = P_np.reshape(self.n_states, self.n_actions, self.n_states)
        if P_np.shape != (self.n_states, self.n_actions, self.n_states):
            raise ValueError(
                "P must have shape "
                f"({self.n_states}, {self.n_actions}, {self.n_states}) "
                f"or ({self.n_states * self.n_actions}, {self.n_states}); got {P_np.shape}"
            )
        if np.any(P_np < -1e-12):
            raise ValueError("P contains negative transition probabilities")
        if not np.allclose(P_np.sum(axis=2), 1.0, atol=1e-6):
            raise ValueError("Each P[s, a, :] row must sum to 1")
        return P_np

    def _as_reward_matrix(self, R: ArrayLike) -> np.ndarray:
        R_np = self._to_numpy(R).astype(np.float64, copy=False)
        if R_np.shape == (self.n_states * self.n_actions,):
            R_np = R_np.reshape(self.n_states, self.n_actions)
        if R_np.shape != (self.n_states, self.n_actions):
            raise ValueError(
                f"R must have shape ({self.n_states}, {self.n_actions}) "
                f"or ({self.n_states * self.n_actions},); got {R_np.shape}"
            )
        return R_np

    def _apply_terminal_states(self) -> None:
        for state in self.terminal_states:
            if state < 0 or state >= self.n_states:
                raise ValueError(f"terminal state {state} is outside [0, n_states)")
            self.P[state, :, :] = 0.0
            self.P[state, :, state] = 1.0
            self.R[state, :] = 0.0

    def _model_from_deterministic_functions(
        self,
        transition_fn: Callable[[int, int], Any],
        reward_fn: Optional[Callable[..., float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        P = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.float64)
        R = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_state, reward = self._parse_transition_result(transition_fn(s, a))
                if next_state < 0 or next_state >= self.n_states:
                    raise ValueError(f"transition_fn({s}, {a}) returned invalid state {next_state}")
                if reward is None:
                    if reward_fn is None:
                        raise ValueError("reward_fn is required when transition_fn does not return reward")
                    reward = self._call_reward_fn(reward_fn, s, a, next_state)
                P[s, a, next_state] = 1.0
                R[s, a] = float(reward)

        return P, R

    @staticmethod
    def _parse_transition_result(result: Any) -> Tuple[int, Optional[float]]:
        if not isinstance(result, tuple):
            return int(result), None
        if len(result) == 2:
            next_state, reward = result
            return int(next_state), float(reward)
        if len(result) in {3, 4, 5}:
            next_state, reward = result[:2]
            return int(next_state), float(reward)
        raise ValueError("Unsupported transition tuple length")

    @staticmethod
    def _call_reward_fn(
        reward_fn: Callable[..., float],
        state: int,
        action: int,
        next_state: int,
    ) -> float:
        try:
            return float(reward_fn(state, action, next_state))
        except TypeError:
            return float(reward_fn(state, action))

    def learned_value(self) -> np.ndarray:
        """Return the solver's current value vector V_solver."""
        with torch.no_grad():
            if hasattr(self.solver, "PHI_S") and hasattr(self.solver, "theta"):
                value = self.solver.PHI_S @ self.solver.theta
            else:
                value = torch.stack(
                    [torch.as_tensor(self.solver.value(s)).reshape(()) for s in range(self.n_states)]
                )
        return value.detach().cpu().numpy().astype(np.float64, copy=False)

    def learned_policy(self) -> np.ndarray:
        """Return the current learned policy matrix pi(a | s)."""
        if getattr(self.solver, "pi", None) is None:
            policy = self.solver.get_policy_matrix()
        else:
            policy = self.solver.pi.detach().clone()
        return self._as_policy_matrix(policy)

    def print_policy(self, pi: Optional[ArrayLike] = None) -> None:
        """Pretty-print a discrete policy table."""
        pi_np = self.learned_policy() if pi is None else self._as_policy_matrix(pi)
        self._print_policy_table(pi_np, title="SBEED POLICY")

    def optimal_value(
        self,
        tol: float = 1e-10,
        max_iter: int = 100_000,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute V* with standard hard-max value iteration."""
        V = np.zeros(self.n_states, dtype=np.float64)
        diff = np.inf
        it = -1

        for it in range(int(max_iter)):
            Q = self.R + self.gamma * np.einsum("sat,t->sa", self.P, V)
            V_new = np.max(Q, axis=1)
            diff = float(np.max(np.abs(V_new - V)))
            V = V_new
            if diff < tol:
                break

        info = {"iterations": it + 1, "final_diff": diff, "converged": diff < tol}
        return V, info

    def optimal_policy(
        self,
        tol: float = 1e-10,
        max_iter: int = 100_000,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return a deterministic greedy policy for V* as a [S, A] matrix."""
        V_star, info = self.optimal_value(tol=tol, max_iter=max_iter)
        Q_star = self.R + self.gamma * np.einsum("sat,t->sa", self.P, V_star)
        best_actions = np.argmax(Q_star, axis=1)
        pi_star = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        pi_star[np.arange(self.n_states), best_actions] = 1.0
        return pi_star, info

    def uniform_policy(self) -> np.ndarray:
        """Return the uniform random policy as a [S, A] matrix."""
        return np.full(
            (self.n_states, self.n_actions),
            1.0 / self.n_actions,
            dtype=np.float64,
        )

    def print_optimal_policy(
        self,
        tol: float = 1e-10,
        max_iter: int = 100_000,
    ) -> None:
        """Pretty-print the deterministic greedy policy induced by V*."""
        pi_star, _ = self.optimal_policy(tol=tol, max_iter=max_iter)
        self._print_policy_table(pi_star, title="OPTIMAL POLICY")

    def simulate_trajectory(
        self,
        pi: Optional[ArrayLike] = None,
        max_steps: int = 100,
        start_state: Optional[int] = None,
        seed: Optional[int] = None,
        greedy: bool = False,
        discounted: bool = False,
    ) -> Dict[str, Any]:
        """
        Simulate one trajectory and return its cumulative reward.

        If discounted=True, rewards are accumulated as sum_t gamma^t r_t.
        Otherwise this returns the plain sum of rewards along the rollout.
        """
        rng = np.random.default_rng(seed)
        pi_np = self.learned_policy() if pi is None else self._as_policy_matrix(pi)

        state = int(start_state if start_state is not None else (self.x0 if self.x0 is not None else 0))
        total_reward = 0.0
        trajectory = []

        for step in range(int(max_steps)):
            if greedy:
                action = int(np.argmax(pi_np[state]))
            else:
                action = int(rng.choice(self.n_actions, p=pi_np[state]))
            next_state = int(rng.choice(self.n_states, p=self.P[state, action]))
            reward = float(self.R[state, action])
            total_reward += (self.gamma ** step) * reward if discounted else reward
            trajectory.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "step": step,
                }
            )
            if next_state in self.terminal_states:
                break
            state = next_state

        return {"total_reward": float(total_reward), "trajectory": trajectory}

    def average_reward(
        self,
        pi: Optional[ArrayLike] = None,
        n_trajectories: int = 10,
        max_steps: int = 100,
        start_state: Optional[int] = None,
        seed: Optional[int] = None,
        greedy: bool = False,
        discounted: bool = False,
    ) -> Dict[str, Any]:
        """Simulate trajectories and return average cumulative reward."""
        if n_trajectories <= 0:
            raise ValueError("n_trajectories must be positive")
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        rewards = []
        base_rng = np.random.default_rng(seed)
        for _ in range(int(n_trajectories)):
            rollout_seed = int(base_rng.integers(0, np.iinfo(np.int32).max))
            result = self.simulate_trajectory(
                pi=pi,
                max_steps=max_steps,
                start_state=start_state,
                seed=rollout_seed,
                greedy=greedy,
                discounted=discounted,
            )
            rewards.append(result["total_reward"])

        rewards_np = np.asarray(rewards, dtype=np.float64)
        return {
            "avg_reward": float(np.mean(rewards_np)),
            "std_reward": float(np.std(rewards_np)),
            "rewards": rewards,
            "n_trajectories": int(n_trajectories),
            "max_steps": int(max_steps),
            "discounted": bool(discounted),
        }

    def average_optimal_reward(
        self,
        n_trajectories: int = 10,
        max_steps: int = 100,
        start_state: Optional[int] = None,
        seed: Optional[int] = None,
        discounted: bool = False,
        tol: float = 1e-10,
        max_iter: int = 100_000,
    ) -> Dict[str, Any]:
        """Simulate the greedy V* policy and return average cumulative reward."""
        pi_star, info = self.optimal_policy(tol=tol, max_iter=max_iter)
        stats = self.average_reward(
            pi=pi_star,
            n_trajectories=n_trajectories,
            max_steps=max_steps,
            start_state=start_state,
            seed=seed,
            greedy=True,
            discounted=discounted,
        )
        stats["optimal_value_info"] = info
        return stats

    def average_uniform_reward(
        self,
        n_trajectories: int = 10,
        max_steps: int = 100,
        start_state: Optional[int] = None,
        seed: Optional[int] = None,
        discounted: bool = False,
    ) -> Dict[str, Any]:
        """Simulate the uniform random policy and return average cumulative reward."""
        return self.average_reward(
            pi=self.uniform_policy(),
            n_trajectories=n_trajectories,
            max_steps=max_steps,
            start_state=start_state,
            seed=seed,
            discounted=discounted,
        )

    def reward_comparison(
        self,
        n_trajectories: int = 10,
        max_steps: int = 100,
        start_state: Optional[int] = None,
        seed: Optional[int] = None,
        discounted: bool = False,
        tol: float = 1e-10,
        max_iter: int = 100_000,
    ) -> Dict[str, Any]:
        """Return simulated average reward for learned policy and optimal policy."""
        learned = self.average_reward(
            n_trajectories=n_trajectories,
            max_steps=max_steps,
            start_state=start_state,
            seed=seed,
            discounted=discounted,
        )
        optimal = self.average_optimal_reward(
            n_trajectories=n_trajectories,
            max_steps=max_steps,
            start_state=start_state,
            seed=seed,
            discounted=discounted,
            tol=tol,
            max_iter=max_iter,
        )
        uniform = self.average_uniform_reward(
            n_trajectories=n_trajectories,
            max_steps=max_steps,
            start_state=start_state,
            seed=seed,
            discounted=discounted,
        )
        return {
            "avg_reward": learned["avg_reward"],
            "uniform_avg_reward": uniform["avg_reward"],
            "optimal_avg_reward": optimal["avg_reward"],
            "reward_gap": optimal["avg_reward"] - learned["avg_reward"],
            "learned": learned,
            "uniform": uniform,
            "optimal": optimal,
        }

    def value_error_squared(
        self,
        tol: float = 1e-10,
        max_iter: int = 100_000,
    ) -> float:
        """Return ||V_solver - V*||_2^2."""
        V_solver = self.learned_value()
        V_star, _ = self.optimal_value(tol=tol, max_iter=max_iter)
        return float(np.sum((V_solver - V_star) ** 2))

    def evaluate(
        self,
        tol: float = 1e-10,
        max_iter: int = 100_000,
    ) -> Dict[str, Any]:
        """Return the metric plus the two value vectors for inspection."""
        V_solver = self.learned_value()
        V_star, info = self.optimal_value(tol=tol, max_iter=max_iter)
        return {
            "value_error_squared": float(np.sum((V_solver - V_star) ** 2)),
            "V_solver": V_solver,
            "V_star": V_star,
            "optimal_value_info": info,
        }

    def _print_policy_table(self, pi: np.ndarray, title: str) -> None:
        print(f"\n========== {title} ==========\n")
        for s in range(self.n_states):
            probs = "  ".join(f"pi({a}|{s})={pi[s, a]:.3f}" for a in range(self.n_actions))
            best = int(np.argmax(pi[s]))
            print(f"State {s}: {probs}  --> best action: {best}")
        print("\n==================================\n")

    def get_metric(self, name: str = "value_error_squared", **kwargs: Any):
        """
        Return a zero-argument metric callable.

        Supported metric names:
            value_error_squared, value_l2_squared, hard_value_error_l2_squared
            avg_reward, average_reward, simulated_reward
            uniform_avg_reward, average_uniform_reward, uniform_reward
            optimal_avg_reward, average_optimal_reward, optimal_reward
        """
        if name not in {"value_error_squared", "value_l2_squared", "hard_value_error_l2_squared"}:
            if name in {"avg_reward", "average_reward", "simulated_reward"}:
                n_trajectories = kwargs.get("n_trajectories", 10)
                max_steps = kwargs.get("max_steps", 100)
                start_state = kwargs.get("start_state", None)
                seed = kwargs.get("seed", None)
                greedy = kwargs.get("greedy", False)
                discounted = kwargs.get("discounted", False)
                return lambda: self.average_reward(
                    n_trajectories=n_trajectories,
                    max_steps=max_steps,
                    start_state=start_state,
                    seed=seed,
                    greedy=greedy,
                    discounted=discounted,
                )["avg_reward"]
            if name in {"uniform_avg_reward", "average_uniform_reward", "uniform_reward"}:
                n_trajectories = kwargs.get("n_trajectories", 10)
                max_steps = kwargs.get("max_steps", 100)
                start_state = kwargs.get("start_state", None)
                seed = kwargs.get("seed", None)
                discounted = kwargs.get("discounted", False)
                return lambda: self.average_uniform_reward(
                    n_trajectories=n_trajectories,
                    max_steps=max_steps,
                    start_state=start_state,
                    seed=seed,
                    discounted=discounted,
                )["avg_reward"]
            if name in {"optimal_avg_reward", "average_optimal_reward", "optimal_reward"}:
                n_trajectories = kwargs.get("n_trajectories", 10)
                max_steps = kwargs.get("max_steps", 100)
                start_state = kwargs.get("start_state", None)
                seed = kwargs.get("seed", None)
                discounted = kwargs.get("discounted", False)
                tol = kwargs.get("tol", 1e-10)
                max_iter = kwargs.get("max_iter", 100_000)
                return lambda: self.average_optimal_reward(
                    n_trajectories=n_trajectories,
                    max_steps=max_steps,
                    start_state=start_state,
                    seed=seed,
                    discounted=discounted,
                    tol=tol,
                    max_iter=max_iter,
                )["avg_reward"]
            raise ValueError(
                "Unknown metric "
                f"{name!r}. Supported metrics are value_error_squared, "
                "avg_reward, uniform_avg_reward, and optimal_avg_reward."
            )
        tol = kwargs.get("tol", 1e-10)
        max_iter = kwargs.get("max_iter", 100_000)
        return lambda: self.value_error_squared(tol=tol, max_iter=max_iter)
