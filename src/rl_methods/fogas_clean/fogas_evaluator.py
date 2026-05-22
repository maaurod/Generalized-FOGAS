import random

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


class FOGASEvaluator:
    """
    Evaluation utilities for FOGAS policies.

    The evaluator can be constructed with a solver, an MDP, and optionally a
    planner. Planner-dependent methods fail explicitly when no planner is
    available.
    """

    VALID_POLICY_MODES = {"solver", "greedy"}

    def __init__(self, solver=None, mdp=None, planner=None):
        if solver is None and mdp is None:
            raise ValueError("FOGASEvaluator requires either a solver or an mdp.")

        self.solver = solver
        self.mdp = mdp if mdp is not None else solver.mdp
        self.planner = planner

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _require_solver(self):
        if self.solver is None:
            raise ValueError("This method requires a solver.")
        return self.solver

    def _require_trained_solver(self):
        solver = self._require_solver()
        if solver.pi is None:
            raise ValueError("Run solver.run() first.")
        return solver

    def _require_planner(self):
        if self.planner is None:
            raise ValueError("This method requires a Planner. Construct FOGASEvaluator(..., planner=planner).")
        return self.planner

    @staticmethod
    def _set_seed(seed):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    def _as_policy_tensor(self, pi):
        if isinstance(pi, torch.Tensor):
            out = pi.to(dtype=torch.float64, device=self.mdp.r.device)
        else:
            out = torch.tensor(pi, dtype=torch.float64, device=self.mdp.r.device)
        if out.shape != (self.mdp.N, self.mdp.A):
            raise ValueError(f"Policy must have shape ({self.mdp.N}, {self.mdp.A}), got {tuple(out.shape)}")
        return out

    @staticmethod
    def _comparison_result(policy_value, optimal_value, extra=None):
        result = {
            "policy": float(policy_value),
            "optimal": None if optimal_value is None else float(optimal_value),
            "difference": None if optimal_value is None else float(optimal_value - policy_value),
        }
        if extra:
            result.update(extra)
        return result

    @staticmethod
    def _discounted_return(trajectory, gamma):
        total = 0.0
        for step in trajectory:
            total += (gamma ** int(step["step"])) * float(step["reward"])
        return float(total)

    # ------------------------------------------------------------------
    # Policy selection
    # ------------------------------------------------------------------
    def greedy_policy(self, pi=None):
        """
        Convert a stochastic policy into a deterministic argmax policy.
        """
        if pi is None:
            pi = self._require_trained_solver().pi
        pi = self._as_policy_tensor(pi)
        greedy = torch.zeros_like(pi)
        best_actions = torch.argmax(pi, dim=1)
        greedy[torch.arange(self.mdp.N, device=pi.device), best_actions] = 1.0
        return greedy

    def get_policy(self, policy_mode):
        """
        Return the requested solver policy.

        policy_mode must be "solver" or "greedy".
        """
        if policy_mode not in self.VALID_POLICY_MODES:
            raise ValueError(f"policy_mode must be one of {sorted(self.VALID_POLICY_MODES)}, got {policy_mode!r}")

        solver = self._require_trained_solver()
        if policy_mode == "solver":
            return self._as_policy_tensor(solver.pi)
        return self.greedy_policy(solver.pi)

    # ------------------------------------------------------------------
    # Simulation metrics
    # ------------------------------------------------------------------
    def average_return(
        self,
        policy_mode,
        num_trajectories,
        max_steps,
        seed=None,
        goal_state=None,
        terminal_states=None,
        compare_with_optimal=False,
    ):
        """
        Average discounted simulated return over multiple trajectories.
        """
        pi = self.get_policy(policy_mode)
        policy_value = self._average_simulated_return(
            pi=pi,
            num_trajectories=num_trajectories,
            max_steps=max_steps,
            seed=seed,
            terminal_states=self._terminal_states(terminal_states, goal_state),
        )

        optimal_value = None
        if compare_with_optimal:
            planner = self._require_planner()
            optimal_value = self._average_simulated_return(
                pi=planner.pi_star,
                num_trajectories=num_trajectories,
                max_steps=max_steps,
                seed=seed,
                terminal_states=self._terminal_states(terminal_states, goal_state),
            )

        return self._comparison_result(
            policy_value,
            optimal_value,
            {
                "num_trajectories": int(num_trajectories),
                "max_steps": int(max_steps),
            },
        )

    def success_rate(
        self,
        goal_state,
        num_trajectories,
        max_steps,
        seed=None,
        compare_with_optimal=False,
    ):
        """
        Fraction of greedy-policy trajectories that reach goal_state.
        """
        pi = self.get_policy("greedy")
        policy_value = self._success_rate(
            pi=pi,
            goal_state=goal_state,
            num_trajectories=num_trajectories,
            max_steps=max_steps,
            seed=seed,
        )

        optimal_value = None
        if compare_with_optimal:
            planner = self._require_planner()
            optimal_value = self._success_rate(
                pi=planner.pi_star,
                goal_state=goal_state,
                num_trajectories=num_trajectories,
                max_steps=max_steps,
                seed=seed,
            )

        return self._comparison_result(
            policy_value,
            optimal_value,
            {
                "goal_state": int(goal_state),
                "num_trajectories": int(num_trajectories),
                "max_steps": int(max_steps),
            },
        )

    def _average_simulated_return(self, pi, num_trajectories, max_steps, seed=None, terminal_states=None):
        returns = []
        for idx in range(int(num_trajectories)):
            current_seed = None if seed is None else int(seed) + idx
            trajectory = self.simulate_trajectory(
                pi=pi,
                max_steps=max_steps,
                seed=current_seed,
                terminal_states=terminal_states,
            )
            returns.append(self._discounted_return(trajectory, self.mdp.gamma))
        return float(np.mean(returns)) if returns else 0.0

    def _success_rate(self, pi, goal_state, num_trajectories, max_steps, seed=None):
        successes = 0
        for idx in range(int(num_trajectories)):
            current_seed = None if seed is None else int(seed) + idx
            trajectory = self.simulate_trajectory(
                pi=pi,
                max_steps=max_steps,
                seed=current_seed,
                terminal_states=[goal_state],
            )
            successes += int(bool(trajectory) and trajectory[-1]["next_state"] == int(goal_state))
        return float(successes / num_trajectories) if num_trajectories else 0.0

    # ------------------------------------------------------------------
    # Value-quality metrics
    # ------------------------------------------------------------------
    def on_data_quality(self, dataset, policy_mode, compare_with_optimal=False):
        """
        Evaluate the selected policy on the empirical state distribution.

        Without optimal comparison, the policy score is sqrt(E_data[V_pi(s)^2]).
        With optimal comparison, the policy score is the weighted L2 value gap
        sqrt(E_data[(V_star(s) - V_pi(s))^2]).
        """
        planner = self._require_planner()
        states = dataset.X.to(dtype=torch.int64, device=self.mdp.r.device)
        pi = self.get_policy(policy_mode)
        v_pi, _ = planner.evaluate_policy(pi)

        if compare_with_optimal:
            policy_value = self._weighted_l2(planner.v_star[states] - v_pi[states])
            optimal_value = 0.0
        else:
            policy_value = self._weighted_l2(v_pi[states])
            optimal_value = None

        return self._comparison_result(
            policy_value,
            optimal_value,
            {
                "num_states": int(states.numel()),
                "metric": "weighted_l2",
            },
        )

    def optimal_states_quality(self, policy_mode, num_trajectories=1000, max_steps=100, seed=None):
        """
        Weighted L2 value gap on states visited by the optimal policy.
        """
        planner = self._require_planner()
        pi = self.get_policy(policy_mode)
        v_pi, _ = planner.evaluate_policy(pi)

        states = self._sample_policy_states(
            pi=planner.pi_star,
            num_trajectories=num_trajectories,
            max_steps=max_steps,
            seed=seed,
        )
        gaps = planner.v_star[states] - v_pi[states]
        policy_value = self._weighted_l2(gaps)

        return self._comparison_result(
            policy_value,
            0.0,
            {
                "num_states": int(states.numel()),
                "num_trajectories": int(num_trajectories),
                "max_steps": int(max_steps),
                "metric": "weighted_l2_gap",
            },
        )

    @staticmethod
    def _weighted_l2(values):
        values = values.to(dtype=torch.float64)
        return float(torch.sqrt(torch.mean(values ** 2)).item())

    def _sample_policy_states(self, pi, num_trajectories, max_steps, seed=None):
        visited_states = []
        for idx in range(int(num_trajectories)):
            current_seed = None if seed is None else int(seed) + idx
            trajectory = self.simulate_trajectory(
                pi=pi,
                max_steps=max_steps,
                seed=current_seed,
            )
            visited_states.extend(step["state"] for step in trajectory)

        if not visited_states:
            raise ValueError("No states were sampled. Increase num_trajectories or max_steps.")

        return torch.tensor(visited_states, dtype=torch.int64, device=self.mdp.r.device)

    # ------------------------------------------------------------------
    # Metric factory
    # ------------------------------------------------------------------
    def get_metric(self, name, **kwargs):
        """
        Return a zero-argument scalar callable for hyperparameter optimization.
        """
        if name == "average_return":
            policy_mode = kwargs["policy_mode"]
            num_trajectories = kwargs.get("num_trajectories", 10)
            max_steps = kwargs.get("max_steps", 100)
            seed = kwargs.get("seed")
            goal_state = kwargs.get("goal_state")
            terminal_states = kwargs.get("terminal_states")
            maximize = kwargs.get("maximize", True)
            sign = -1.0 if maximize else 1.0
            return lambda: sign * self.average_return(
                policy_mode=policy_mode,
                num_trajectories=num_trajectories,
                max_steps=max_steps,
                seed=seed,
                goal_state=goal_state,
                terminal_states=terminal_states,
            )["policy"]

        if name == "greedy_average_return":
            num_trajectories = kwargs.get("num_trajectories", 10)
            max_steps = kwargs.get("max_steps", 100)
            seed = kwargs.get("seed")
            goal_state = kwargs.get("goal_state")
            terminal_states = kwargs.get("terminal_states")
            maximize = kwargs.get("maximize", True)
            sign = -1.0 if maximize else 1.0
            return lambda: sign * self.average_return(
                policy_mode="greedy",
                num_trajectories=num_trajectories,
                max_steps=max_steps,
                seed=seed,
                goal_state=goal_state,
                terminal_states=terminal_states,
            )["policy"]

        if name == "success_rate":
            goal_state = kwargs["goal_state"]
            num_trajectories = kwargs.get("num_trajectories", 10)
            max_steps = kwargs.get("max_steps", 100)
            seed = kwargs.get("seed")
            maximize = kwargs.get("maximize", True)
            sign = -1.0 if maximize else 1.0
            return lambda: sign * self.success_rate(
                goal_state=goal_state,
                num_trajectories=num_trajectories,
                max_steps=max_steps,
                seed=seed,
            )["policy"]

        if name == "on_data_quality":
            dataset = kwargs["dataset"]
            policy_mode = kwargs["policy_mode"]
            compare_with_optimal = kwargs.get("compare_with_optimal", True)
            return lambda: self.on_data_quality(
                dataset=dataset,
                policy_mode=policy_mode,
                compare_with_optimal=compare_with_optimal,
            )["policy"]

        if name == "optimal_states_quality":
            policy_mode = kwargs["policy_mode"]
            num_trajectories = kwargs.get("num_trajectories", 1000)
            max_steps = kwargs.get("max_steps", 100)
            seed = kwargs.get("seed")
            return lambda: self.optimal_states_quality(
                policy_mode=policy_mode,
                num_trajectories=num_trajectories,
                max_steps=max_steps,
                seed=seed,
            )["policy"]

        raise ValueError(
            "Unknown metric "
            f"{name!r}. Valid metrics: average_return, greedy_average_return, "
            "success_rate, on_data_quality, optimal_states_quality."
        )

    # ------------------------------------------------------------------
    # Diagnostics and printing
    # ------------------------------------------------------------------
    def compare_value_functions(self, policy_mode="solver", print_each=True):
        """
        Print and compare optimal vs selected-policy V and Q functions.
        """
        planner = self._require_planner()
        pi = self.get_policy(policy_mode)
        device = self.mdp.r.device
        v_star = planner.v_star.to(dtype=torch.float64, device=device)
        q_star = planner.q_star.to(dtype=torch.float64, device=device)
        v_pi, q_pi = planner.evaluate_policy(pi)

        print("\n========== VALUE FUNCTION COMPARISON ==========\n")
        print(f"Policy mode: {policy_mode}\n")

        if print_each:
            print("State-wise V comparison:")
            for x in range(self.mdp.N):
                diff = v_pi[x] - v_star[x]
                print(
                    f"State {x}: "
                    f"V*(x) = {v_star[x].item(): .6f} | "
                    f"V^pi(x) = {v_pi[x].item(): .6f} | "
                    f"delta = {diff.item(): .6e}"
                )

            print("\nAction-value Q comparison:")
            for x in range(self.mdp.N):
                for a in range(self.mdp.A):
                    diff = q_pi[x, a] - q_star[x, a]
                    print(
                        f"(x={x}, a={a}): "
                        f"Q*(x,a) = {q_star[x,a].item(): .6f} | "
                        f"Q^pi(x,a) = {q_pi[x,a].item(): .6f} | "
                        f"delta = {diff.item(): .6e}"
                    )

        v_err = torch.linalg.norm(v_pi - v_star)
        q_err = torch.linalg.norm(q_pi - q_star)
        print("\nNorm diagnostics:")
        print(f"||V^pi - V*||_2 = {v_err.item():.6e}")
        print(f"||Q^pi - Q*||_2 = {q_err.item():.6e}")
        print("\n===============================================\n")

    def print_solver_policy(self, policy_mode="solver"):
        pi = self.get_policy(policy_mode)
        self.mdp.print_policy(pi)

    def print_optimal_policy(self):
        planner = self._require_planner()
        self.mdp.print_policy(planner.pi_star)

    @staticmethod
    def _terminal_states(terminal_states=None, goal_state=None):
        states = set()
        if terminal_states is not None:
            if isinstance(terminal_states, (int, np.integer)):
                states.add(int(terminal_states))
            else:
                states.update(int(state) for state in terminal_states)
        if goal_state is not None:
            states.add(int(goal_state))
        return states

    def simulate_trajectory(
        self,
        pi=None,
        policy_mode=None,
        max_steps=100,
        seed=None,
        goal_state=None,
        terminal_states=None,
    ):
        """
        Simulate a single trajectory.

        Pass either an explicit pi matrix or policy_mode="solver"/"greedy".
        """
        if pi is not None and policy_mode is not None:
            raise ValueError("Pass either pi or policy_mode, not both.")
        if pi is None and policy_mode is None:
            raise ValueError("Pass either pi or policy_mode.")

        self._set_seed(seed)

        if pi is None:
            pi = self.get_policy(policy_mode)
        else:
            pi = self._as_policy_tensor(pi)

        terminal_states = self._terminal_states(terminal_states, goal_state)
        trajectory = []
        state = int(self.mdp.x0)

        for step in range(int(max_steps)):
            action_probs = pi[state]
            prob_sum = action_probs.sum()
            if prob_sum <= 0:
                raise ValueError(f"Policy probabilities at state {state} must have positive mass.")
            action_probs = action_probs / prob_sum
            action = int(torch.multinomial(action_probs, num_samples=1).item())

            reward = self.mdp.r[state * self.mdp.A + action]
            reward = float(reward.item() if isinstance(reward, torch.Tensor) else reward)

            transition_probs = self.mdp.P[state * self.mdp.A + action]
            transition_probs = transition_probs.to(dtype=torch.float64, device=self.mdp.r.device)
            next_state = int(torch.multinomial(transition_probs, num_samples=1).item())

            trajectory.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "step": step,
                    "was_self_loop": next_state == state,
                    "reached_goal": bool(goal_state is not None and next_state == int(goal_state)),
                    "terminal": next_state in terminal_states,
                }
            )

            if next_state in terminal_states:
                break
            state = next_state

        return trajectory

    def print_optimal_path(
        self,
        policy_mode="solver",
        use_optimal=False,
        num_trajectories=1,
        max_steps=50,
        seed=42,
        show_probabilities=False,
        show_probabilities_first_n=5,
        goal_state=None,
        terminal_states=None,
        show_value_info=True,
    ):
        """
        Display trajectories for the selected solver policy or optimal policy.
        """
        if use_optimal:
            planner = self._require_planner()
            pi = planner.pi_star
            policy_name = "Optimal Policy"
        else:
            pi = self.get_policy(policy_mode)
            policy_name = f"Solver Policy ({policy_mode})"

        print(f"\n{'=' * 70}")
        print(f"  PATH VISUALIZATION - {policy_name}")
        print(f"{'=' * 70}\n")
        print(f"Initial State: {self.mdp.states[self.mdp.x0]}")
        if goal_state is not None:
            print(f"Goal State: {self.mdp.states[goal_state]}")
        terminal_states = self._terminal_states(terminal_states, goal_state)
        if terminal_states:
            print(f"Terminal States: {[int(self.mdp.states[state]) for state in sorted(terminal_states)]}")
        print(f"Discount Factor (gamma): {self.mdp.gamma}")
        print(f"\n{'-' * 70}\n")

        for traj_idx in range(int(num_trajectories)):
            current_seed = None if seed is None else int(seed) + traj_idx
            trajectory = self.simulate_trajectory(
                pi=pi,
                max_steps=max_steps,
                seed=current_seed,
                goal_state=goal_state,
                terminal_states=terminal_states,
            )

            if num_trajectories > 1:
                print(f"\n  --- Trajectory {traj_idx + 1} ---\n")

            discounted_return = self._discounted_return(trajectory, self.mdp.gamma)

            for i, step in enumerate(trajectory):
                s = step["state"]
                a = step["action"]
                r = step["reward"]
                sp = step["next_state"]
                indicators = ""
                if step["was_self_loop"]:
                    indicators += " SELF-LOOP"
                if step["reached_goal"]:
                    indicators += " GOAL REACHED"

                print(
                    f"  Step {i:3d} | "
                    f"State: {str(self.mdp.states[s]):15s} | "
                    f"Action: {str(self.mdp.actions[a]):15s} | "
                    f"Reward: {r:7.3f} | "
                    f"-> {self.mdp.states[sp]}{indicators}"
                )

                if show_probabilities and i < show_probabilities_first_n:
                    print(f"           | Policy at state {s}:")
                    for act_idx in range(self.mdp.A):
                        prob = float(pi[s, act_idx].item())
                        bar_len = int(prob * 20)
                        bar = "#" * bar_len + "." * (20 - bar_len)
                        print(f"           |   pi(a={act_idx}|s={s}) = {prob:.3f} {bar}")
                    print("           |")

            final_state = trajectory[-1]["next_state"] if trajectory else self.mdp.x0
            print(f"\n  {'-' * 66}")
            print(f"  Trajectory Length: {len(trajectory)} steps")
            print(f"  Discounted Return: {discounted_return:.6f}")
            print(f"  Final State: {self.mdp.states[final_state]}")

        if show_value_info:
            planner = self._require_planner()
            v_pi, _ = planner.evaluate_policy(pi)
            print(f"\n{'-' * 70}")
            print(f"  Expected Return (from V): {float(v_pi[self.mdp.x0].item()):.6f}")
            print(f"  Optimal Return: {float(planner.v_star[self.mdp.x0].item()):.6f}")

        print(f"\n{'=' * 70}\n")

    def analyze_reward_approximation(self, walls=None, pits=None, goal=None, show_plot=True, print_each=False):
        """
        Analyze how well solver.Phi @ solver.omega represents mdp.r.
        """
        solver = self._require_solver()
        if getattr(solver, "omega", None) is None:
            raise ValueError("Solver must have an 'omega' attribute (estimated or provided).")
        if not hasattr(solver, "Phi"):
            raise ValueError("Solver must have a 'Phi' feature matrix.")

        omega = solver.omega.detach().cpu() if isinstance(solver.omega, torch.Tensor) else torch.tensor(solver.omega)
        phi_cpu = solver.Phi.detach().cpu()
        r_true = self.mdp.r.detach().cpu()
        r_hat = phi_cpu @ omega

        error = r_hat - r_true
        abs_error = torch.abs(error)

        print("\n" + "=" * 50)
        print("     REWARD APPROXIMATION ANALYSIS")
        print("=" * 50)
        print(f"{'Metric':<30} {'Value':>12}")
        print("-" * 44)
        print(f"{'Max |error|':<30} {abs_error.max().item():>12.6f}")
        print(f"{'Mean |error|':<30} {abs_error.mean().item():>12.6f}")
        print(f"{'RMSE':<30} {torch.sqrt(torch.mean(error ** 2)).item():>12.6f}")

        r_true_var = r_true.var()
        if r_true_var > 1e-12:
            r2 = 1.0 - error.var() / r_true_var
            print(f"{'R2 (explained variance)':<30} {r2.item():>12.6f}")
        else:
            print(f"{'R2 (explained variance)':<30} {'N/A (var=0)':>12}")

        if print_each:
            print("\n" + "-" * 50)
            print(f"{'State':<12} {'Action':<10} {'r_true':>10} {'r_hat':>10} {'error':>10}")
            print("-" * 56)
            action_names = ["Up", "Down", "Left", "Right"]
            for x in range(self.mdp.N):
                state_desc = str(x)
                if walls and x in walls:
                    state_desc += " [Wall]"
                elif pits and x in pits:
                    state_desc += " [Pit]"
                elif goal is not None and x == goal:
                    state_desc += " [Goal]"

                for a in range(self.mdp.A):
                    idx = x * self.mdp.A + a
                    action_desc = action_names[a] if a < len(action_names) else str(a)
                    print(
                        f"{state_desc:<12} {action_desc:<10} "
                        f"{r_true[idx].item():>10.4f} "
                        f"{r_hat[idx].item():>10.4f} "
                        f"{error[idx].item():>10.4f}"
                    )

        if not show_plot:
            return

        grid_size = int(np.sqrt(self.mdp.N))
        if grid_size * grid_size != self.mdp.N:
            raise ValueError("Reward heatmap requires a square-grid number of states.")

        abs_error_grid = abs_error.reshape(self.mdp.N, self.mdp.A).mean(dim=1).reshape(grid_size, grid_size).numpy()
        r_true_grid = r_true.reshape(self.mdp.N, self.mdp.A).mean(dim=1).reshape(grid_size, grid_size).numpy()

        _, axes = plt.subplots(1, 2, figsize=(14, 5))
        im0 = axes[0].imshow(r_true_grid, cmap="RdYlGn", origin="upper", vmin=r_true_grid.min(), vmax=r_true_grid.max())
        axes[0].set_title("True Reward")
        plt.colorbar(im0, ax=axes[0], label="Reward Value")

        vmin_err = max(1e-4, float(abs_error_grid.min()))
        vmax_err = max(vmin_err, float(abs_error_grid.max()))
        norm1 = mcolors.LogNorm(vmin=vmin_err, vmax=vmax_err)
        im1 = axes[1].imshow(abs_error_grid, cmap="hot_r", origin="upper", norm=norm1)
        axes[1].set_title("Mean Absolute Error")
        plt.colorbar(im1, ax=axes[1], label="|Estimated - True|")

        for ax in axes:
            if walls:
                for s in walls:
                    row, col = divmod(s, grid_size)
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color="black", alpha=0.8))
            if pits:
                for s in pits:
                    row, col = divmod(s, grid_size)
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color="magenta", alpha=0.6))
            if goal is not None:
                row, col = divmod(goal, grid_size)
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color="gold", alpha=0.8))
            ax.set_xticks(range(grid_size))
            ax.set_yticks(range(grid_size))

        plt.tight_layout()
        plt.show()
