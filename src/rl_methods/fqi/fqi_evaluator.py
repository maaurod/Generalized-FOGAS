"""Evaluation helpers for Fitted Q Iteration experiments."""

from __future__ import annotations

import random
from typing import Any, Iterable, Optional

import numpy as np
import torch


class FQIEvaluator:
    """
    Small evaluator for `FQISolver`.

    The methods mirror the calls used by the FQI notebooks and scripts: policy
    printing, greedy trajectory simulation, final discounted return, and value
    comparisons against an MDP that exposes optimal values.
    """

    def __init__(self, solver: Any):
        self.solver = solver
        self.mdp = solver.mdp

    @staticmethod
    def _set_seed(seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _policy_matrix(self, pi: Optional[Any] = None) -> torch.Tensor:
        if pi is None:
            if self.solver.pi is None:
                raise ValueError("Run solver.run() before evaluating the FQI policy.")
            pi = self.solver.pi
        if isinstance(pi, torch.Tensor):
            out = pi.to(dtype=torch.float64)
        else:
            out = torch.as_tensor(pi, dtype=torch.float64)
        expected = (int(self.mdp.N), int(self.mdp.A))
        if tuple(out.shape) != expected:
            raise ValueError(f"Policy must have shape {expected}, got {tuple(out.shape)}")
        return out

    def _transition_row(self, state: int, action: int) -> torch.Tensor:
        if not hasattr(self.mdp, "P"):
            raise AttributeError("FQI trajectory evaluation requires mdp.P.")
        return self.mdp.P[int(state) * int(self.mdp.A) + int(action)].to(dtype=torch.float64)

    def _reward(self, state: int, action: int) -> float:
        if hasattr(self.mdp, "r"):
            value = self.mdp.r[int(state) * int(self.mdp.A) + int(action)]
            return float(value.item() if isinstance(value, torch.Tensor) else value)
        if hasattr(self.mdp, "get_reward"):
            rewards = self.mdp.get_reward()
            value = rewards[int(state) * int(self.mdp.A) + int(action)]
            return float(value.item() if isinstance(value, torch.Tensor) else value)
        raise AttributeError("FQI trajectory evaluation requires mdp.r or mdp.get_reward().")

    def _terminal_states(self, terminal_states: Optional[Iterable[int]] = None) -> set[int]:
        states: set[int] = set()
        if terminal_states is not None:
            states.update(int(state) for state in terminal_states)
        elif hasattr(self.mdp, "terminal_states"):
            states.update(int(state) for state in self.mdp.terminal_states)
        return states

    def print_policy(self, pi: Optional[Any] = None) -> None:
        """Print the selected FQI policy using the MDP's policy formatter."""
        policy = self._policy_matrix(pi)
        if hasattr(self.mdp, "print_policy"):
            self.mdp.print_policy(policy.detach().cpu())
            return
        print("\n========== LEARNED POLICY (FQI) ==========")
        for state in range(int(self.mdp.N)):
            best_action = int(torch.argmax(policy[state]).item())
            probs = " ".join(
                f"pi(a={action}|s={state})={policy[state, action].item():.2f}"
                for action in range(int(self.mdp.A))
            )
            print(f"state {state}: {probs} --> best action: {best_action}")

    def simulate_trajectory(
        self,
        pi: Optional[Any] = None,
        max_steps: int = 100,
        seed: Optional[int] = None,
        terminal_states: Optional[Iterable[int]] = None,
    ) -> list[tuple[int, int, float, int]]:
        """
        Simulate one greedy-policy trajectory.

        Returns tuples `(state, action, reward, next_state)` for compatibility
        with the existing FQI scripts.
        """
        self._set_seed(seed)
        policy = self._policy_matrix(pi)
        terminals = self._terminal_states(terminal_states)
        state = int(self.mdp.x0)
        trajectory: list[tuple[int, int, float, int]] = []

        for _ in range(int(max_steps)):
            action = int(torch.argmax(policy[state]).item())
            reward = self._reward(state, action)
            transition_probs = self._transition_row(state, action)
            next_state = int(torch.multinomial(transition_probs, num_samples=1).item())
            trajectory.append((state, action, reward, next_state))
            if next_state in terminals:
                break
            state = next_state

        return trajectory

    def final_reward(
        self,
        pi: Optional[Any] = None,
        max_steps: int = 100,
        seed: Optional[int] = 42,
    ) -> float:
        """Return the discounted reward of one greedy FQI trajectory."""
        trajectory = self.simulate_trajectory(pi=pi, max_steps=max_steps, seed=seed)
        gamma = float(self.mdp.gamma)
        total = 0.0
        for step, (_, _, reward, _) in enumerate(trajectory):
            total += (gamma ** step) * float(reward)
        return float(total)

    def compare_final_rewards(self, max_steps: int = 100, seed: Optional[int] = 42) -> dict[str, Optional[float]]:
        """Print FQI return and, when available, the optimal initial value."""
        fqi_return = self.final_reward(max_steps=max_steps, seed=seed)
        optimal = None
        if hasattr(self.mdp, "v_star"):
            optimal_value = self.mdp.v_star[int(self.mdp.x0)]
            optimal = float(optimal_value.item() if isinstance(optimal_value, torch.Tensor) else optimal_value)
        print("\n========== FINAL REWARD COMPARISON ==========")
        print(f"J(pi_FQI) = {fqi_return:.6f}")
        if optimal is not None:
            print(f"J(pi*)    = {optimal:.6f}")
            print(f"gap       = {optimal - fqi_return:.6f}")
        return {"fqi": fqi_return, "optimal": optimal, "gap": None if optimal is None else optimal - fqi_return}

    def compare_value_functions(self, print_each: bool = True) -> dict[str, float]:
        """Compare the FQI greedy policy values with the MDP optimal values."""
        if not hasattr(self.mdp, "evaluate_policy"):
            raise AttributeError("compare_value_functions requires mdp.evaluate_policy(policy).")
        if not hasattr(self.mdp, "v_star") or not hasattr(self.mdp, "q_star"):
            raise AttributeError("compare_value_functions requires mdp.v_star and mdp.q_star.")
        policy = self._policy_matrix()
        v_pi, q_pi = self.mdp.evaluate_policy(policy)
        v_star = self.mdp.v_star.to(dtype=torch.float64, device=v_pi.device)
        q_star = self.mdp.q_star.to(dtype=torch.float64, device=q_pi.device)

        print("\n========== VALUE FUNCTION COMPARISON ==========")
        if print_each:
            print("State-wise V comparison:")
            for state in range(int(self.mdp.N)):
                print(
                    f"State {state}: "
                    f"V*(s)={v_star[state].item(): .6f} | "
                    f"V_FQI(s)={v_pi[state].item(): .6f} | "
                    f"diff={(v_pi[state] - v_star[state]).item(): .6e}"
                )

            print("\nAction-value Q comparison:")
            q_pi_matrix = q_pi.reshape(int(self.mdp.N), int(self.mdp.A))
            q_star_matrix = q_star.reshape(int(self.mdp.N), int(self.mdp.A))
            for state in range(int(self.mdp.N)):
                for action in range(int(self.mdp.A)):
                    print(
                        f"(s={state}, a={action}): "
                        f"Q*(s,a)={q_star_matrix[state, action].item(): .6f} | "
                        f"Q_FQI(s,a)={q_pi_matrix[state, action].item(): .6f} | "
                        f"diff={(q_pi_matrix[state, action] - q_star_matrix[state, action]).item(): .6e}"
                    )

        v_error = float(torch.linalg.norm(v_pi - v_star).item())
        q_error = float(torch.linalg.norm(q_pi.reshape(-1) - q_star.reshape(-1)).item())
        print("\nNorm diagnostics:")
        print(f"||V_FQI - V*||_2 = {v_error:.6e}")
        print(f"||Q_FQI - Q*||_2 = {q_error:.6e}")
        return {"v_l2_error": v_error, "q_l2_error": q_error}

    def print_optimal_path(self, max_steps: int = 50, seed: Optional[int] = 42) -> list[tuple[int, int, float, int]]:
        """Print one greedy FQI trajectory from the MDP initial state."""
        trajectory = self.simulate_trajectory(max_steps=max_steps, seed=seed)
        print("\n========== OPTIMAL PATH (FQI Policy) ==========")
        for step, (state, action, reward, next_state) in enumerate(trajectory):
            print(f"{step:03d}: s={state} a={action} r={reward:.4f} -> s'={next_state}")
        print(f"discounted_return={self.final_reward(max_steps=max_steps, seed=seed):.6f}")
        return trajectory

