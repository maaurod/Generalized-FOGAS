"""Stage 1 SBEED solver: one-step linear prototype.

This file is intentionally close to the mathematical derivation in the report.
It keeps all trainable objects as explicit tensors:

    theta -> value weights V(s)
    beta  -> dual weights rho(s, a)
    W     -> softmax policy weights pi(a | s)

The important implementation choice in this first version is the dual update:
`beta` is fitted by closed-form ridge regression to the one-step smoothed
Bellman target. Later versions replace that exact solve with SGD/Adam so the
same update order can work with nonlinear parametrizations.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import trange

from ..datasets.discrete_sbeed_dataset import DiscreteSBEEDDataset as SBEEDDataset
from ..sbeed_spec import DiscreteMDPSpec


class SBEEDSolver:
    """
    Linear SBEED prototype for finite discrete online RL problems.

    The solver learns:
        V_theta(s) = theta^T value_features(s)
        rho_beta(s, a) = beta^T rho_features(s, a)
        pi_W(a | s) = softmax(W policy_features(s))[a]

    Stage purpose:
        Validate the one-step SBEED objective and the analytic linear
        gradients in the simplest setting.

    Difference from the next version:
        This solver stores four-field transitions `(s, a, r, s_next)` and
        always bootstraps from `s_next`. `SBEEDSolverSGDRho` adds terminal flags
        and changes the rho update from closed-form ridge regression to SGD.

    The solver starts with an empty replay buffer D. Each `run` call resets D
    and grows a fresh buffer by interacting with a user-provided discrete
    transition function.
    """

    def __init__(
        self,
        n_states: Optional[int] = None,
        n_actions: Optional[int] = None,
        gamma: Optional[float] = None,
        value_features: Optional[Callable[[int], torch.Tensor]] = None,
        rho_features: Optional[Callable[[int, int], torch.Tensor]] = None,
        policy_features: Optional[Callable[[int], torch.Tensor]] = None,
        spec: Optional[DiscreteMDPSpec] = None,
        lambda_entropy: float = 0.01,
        eta: float = 1.0,
        ridge: float = 1e-6,
        lr_value: float = 1e-2,
        lr_policy: float = 1e-2,
        tau: float = 1.0,
        buffer_mode: str = "growing",
        max_buffer_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        seed: Optional[int] = 42,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if spec is None:
            if n_states is None or n_actions is None or gamma is None:
                raise ValueError("n_states, n_actions, and gamma are required when spec is not provided")
            if value_features is None or rho_features is None:
                raise ValueError("value_features and rho_features are required when spec is not provided")
            spec = DiscreteMDPSpec(
                n_states=n_states,
                n_actions=n_actions,
                gamma=gamma,
                value_features=value_features,
                rho_features=rho_features,
                policy_features=policy_features,
            )

        self.spec = spec
        self.n_states = spec.n_states
        self.n_actions = spec.n_actions
        self.gamma = spec.gamma
        self.value_features = spec.value_features
        self.rho_features = spec.rho_features
        self.policy_features = spec.policy_features
        self.value_dim = spec.value_dim
        self.rho_dim = spec.rho_dim
        self.policy_dim = spec.policy_dim

        self.lambda_entropy = float(lambda_entropy)
        self.eta = float(eta)
        self.ridge = float(ridge)
        self.lr_value = float(lr_value)
        self.lr_policy = float(lr_policy)
        self.tau = float(tau)
        self.buffer_mode = str(buffer_mode)
        self.max_buffer_size = None if max_buffer_size is None else int(max_buffer_size)
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if self.lambda_entropy < 0.0:
            raise ValueError("lambda_entropy must be non-negative")
        if not (0.0 <= self.eta <= 1.0):
            raise ValueError("eta must be in [0, 1]")
        if self.ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        if self.lr_value <= 0.0 or self.lr_policy <= 0.0:
            raise ValueError("lr_value and lr_policy must be positive")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.buffer_mode not in {"growing", "fifo"}:
            raise ValueError("buffer_mode must be 'growing' or 'fifo'")
        if self.max_buffer_size is not None and self.max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive")
        if self.buffer_mode == "fifo" and self.max_buffer_size is None:
            raise ValueError("max_buffer_size is required when buffer_mode='fifo'")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._seed_all(seed)

        # D = {(s_i, a_i, r_i, s'_i)}. SBEED optimizes empirical expectations
        # over this replay buffer; full-batch means every expectation is an
        # average over all rows, mini-batch means a stochastic estimate.
        self.dataset = SBEEDDataset.empty(device=self.device)
        self.dataset.validate(self.n_states, self.n_actions)
        self.n = self.dataset.n

        self._precompute_features()

        self.theta = torch.zeros(self.value_dim, dtype=torch.float64, device=self.device)
        self.beta = torch.zeros(self.rho_dim, dtype=torch.float64, device=self.device)
        self.W = torch.zeros((self.n_actions, self.policy_dim), dtype=torch.float64, device=self.device)

        self.theta_history = []
        self.beta_history = []
        self.W_history = []
        self.loss_history = []
        self.pi: Optional[torch.Tensor] = None
        self.update_index = 0

    @staticmethod
    def _seed_all(seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    def _feature_to_device(self, value: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float64, device=self.device).reshape(-1)

    def _precompute_features(self) -> None:
        """
        Build feature design tensors once.

        PHI_S[s]        = phi(s), the value feature.
        POLICY_PHI_S[s] = chi(s), the policy feature.
        RHO_SA[s, a]    = zeta(s, a), the dual residual feature.
        Phi[i]          = phi(s_i)
        Phi_next[i]     = phi(s'_i)
        PolicyPhi[i]    = chi(s_i)
        Rho[i]          = zeta(s_i, a_i)

        I use `rho_features` / `zeta` language here instead of the paper's
        generic rho function name to avoid confusion with LinearMDP psi.
        """
        value_all = [
            self._feature_to_device(self.value_features(s))
            for s in range(self.n_states)
        ]
        policy_all = [
            self._feature_to_device(self.policy_features(s))
            for s in range(self.n_states)
        ]
        rho_all = [
            [
                self._feature_to_device(self.rho_features(s, a))
                for a in range(self.n_actions)
            ]
            for s in range(self.n_states)
        ]

        value_dims = {feat.numel() for feat in value_all}
        policy_dims = {feat.numel() for feat in policy_all}
        rho_dims = {feat.numel() for row in rho_all for feat in row}
        if value_dims != {self.value_dim}:
            raise ValueError("value_features must return a fixed feature dimension")
        if policy_dims != {self.policy_dim}:
            raise ValueError("policy_features must return a fixed feature dimension")
        if rho_dims != {self.rho_dim}:
            raise ValueError("rho_features must return a fixed feature dimension")

        self.PHI_S = torch.stack(value_all, dim=0)
        self.POLICY_PHI_S = torch.stack(policy_all, dim=0)
        self.RHO_SA = torch.stack([torch.stack(row, dim=0) for row in rho_all], dim=0)

        self._refresh_dataset_features()

    def _refresh_dataset_features(self) -> None:
        """
        Refresh row-wise feature tensors after D grows.

        Online SBEED appends new transitions to D. The state feature table
        PHI_S, POLICY_PHI_S, and RHO_SA are fixed, but Phi, Phi_next,
        PolicyPhi, and Rho depend on the current contents of D.
        """
        self.n = self.dataset.n
        if self.n == 0:
            self.Phi = torch.empty((0, self.value_dim), dtype=torch.float64, device=self.device)
            self.Phi_next = torch.empty((0, self.value_dim), dtype=torch.float64, device=self.device)
            self.PolicyPhi = torch.empty((0, self.policy_dim), dtype=torch.float64, device=self.device)
            self.Rho = torch.empty((0, self.rho_dim), dtype=torch.float64, device=self.device)
            return

        X = self.dataset.X.long()
        A = self.dataset.A.long()
        X_next = self.dataset.X_next.long()
        self.Phi = self.PHI_S[X]
        self.Phi_next = self.PHI_S[X_next]
        self.PolicyPhi = self.POLICY_PHI_S[X]
        self.Rho = self.RHO_SA[X, A]

    @staticmethod
    def _row_softmax(logits: torch.Tensor) -> torch.Tensor:
        z = logits - logits.max(dim=1, keepdim=True).values
        exp_z = torch.exp(z)
        return exp_z / exp_z.sum(dim=1, keepdim=True)

    def _policy_matrix(self, W: Optional[torch.Tensor] = None) -> torch.Tensor:
        # pi_W(. | s) = softmax(W chi(s)).
        # POLICY_PHI_S has shape (N, d_pi), W has shape (A, d_pi).
        W = self.W if W is None else W
        logits = self.POLICY_PHI_S @ W.T
        return self._row_softmax(logits)

    def _batch_indices(self) -> torch.Tensor:
        if self.n == 0:
            raise ValueError("Replay buffer D is empty. Collect data before calling step().")
        if self.batch_size is None or self.batch_size >= self.n:
            return torch.arange(self.n, device=self.device)
        return torch.randint(self.n, (int(self.batch_size),), device=self.device)

    def _compute_delta(
        self,
        theta: torch.Tensor,
        W: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the one-step smoothed Bellman target sample:

            delta_i = r_i + gamma V_theta(s'_i)
                      - lambda_entropy log pi_W(a_i | s_i).

        At the SBEED optimum, temporal consistency says V(s) should match this
        quantity for every observed (s, a, s').
        """
        X = self.dataset.X[indices].long()
        A = self.dataset.A[indices].long()
        R = self.dataset.R[indices]
        Phi_next = self.Phi_next[indices]

        pi_mat = self._policy_matrix(W)
        pi_sa = pi_mat[X, A].clamp_min(1e-12)
        log_pi = torch.log(pi_sa)
        delta = R + self.gamma * (Phi_next @ theta) - self.lambda_entropy * log_pi
        return delta, pi_mat, log_pi

    def _fit_dual(self, delta: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Inner dual update.

        For fixed theta and W, SBEED fits the dual/residual model rho_beta to
        delta by least squares:

            beta = argmin_b sum_i (delta_i - b^T zeta(s_i, a_i))^2

        The ridge term makes the solve stable when the batch does not cover all
        state-action features.
        """
        Rho = self.Rho[indices]
        gram = Rho.T @ Rho + self.ridge * torch.eye(
            self.rho_dim, dtype=torch.float64, device=self.device
        )
        rhs = Rho.T @ delta.detach()
        return torch.linalg.solve(gram, rhs)

    def objective(self) -> Dict[str, float]:
        if self.n == 0:
            raise ValueError("Replay buffer D is empty. Cannot compute objective.")
        with torch.no_grad():
            indices = torch.arange(self.n, device=self.device)
            delta, _, _ = self._compute_delta(self.theta, self.W, indices)
            V = self.Phi @ self.theta
            rho = self.Rho @ self.beta

            # Empirical SBEED objective:
            #   E[(delta - V(s))^2] - eta E[(delta - rho(s,a))^2].
            # The second term is the variance-cancellation dual term from the
            # paper. Since beta is fitted by least squares, the solver reports
            # the primal and dual MSEs separately for debugging.
            primal = torch.mean((delta - V) ** 2)
            dual = torch.mean((delta - rho) ** 2)
            return {
                "objective": float((primal - self.eta * dual).item()),
                "primal_mse": float(primal.item()),
                "dual_mse": float(dual.item()),
            }

    def step(self) -> Dict[str, float]:
        if self.n == 0:
            raise ValueError("Replay buffer D is empty. Collect data before calling step().")
        indices = self._batch_indices()
        Phi = self.Phi[indices]
        Phi_next = self.Phi_next[indices]
        PolicyPhi = self.PolicyPhi[indices]
        Rho = self.Rho[indices]
        A = self.dataset.A[indices].long()

        # 1. Build delta with the current value and policy.
        delta, _, _ = self._compute_delta(self.theta, self.W, indices)

        # 2. Solve the inner dual problem exactly for this batch.
        self.beta = self._fit_dual(delta, indices)

        V = Phi @ self.theta
        rho = Rho @ self.beta
        residual_v = delta - V
        residual_rho = delta - rho

        # 3. Value gradient from Theorem 4, specialized to
        #    V_theta(s) = theta^T phi(s):
        #
        # grad_theta =
        #   2 E[(delta - V(s)) (gamma phi(s') - phi(s))]
        #   - 2 eta gamma E[(delta - rho(s,a)) phi(s')].
        grad_theta = (
            2.0 * (residual_v[:, None] * (self.gamma * Phi_next - Phi)).mean(dim=0)
            - 2.0 * self.eta * self.gamma * (residual_rho[:, None] * Phi_next).mean(dim=0)
        )

        # 4. Softmax-linear policy gradient.
        #
        # For logits_l = W_l^T chi(s):
        #   grad_W log pi(a|s) = (one_hot(a) - pi(.|s)) outer chi(s).
        action_probs = self._policy_matrix(self.W)[self.dataset.X[indices].long()]
        grad_log_pi = -action_probs[:, :, None] * PolicyPhi[:, None, :]
        grad_log_pi[torch.arange(indices.numel(), device=self.device), A] += PolicyPhi

        # Theorem 4 policy signal:
        #   A_sbeed = (1 - eta) delta + eta rho(s,a) - V(s)
        #   grad_W = -2 E[A_sbeed grad_W log pi(a|s)].
        advantage = (1.0 - self.eta) * delta + self.eta * rho - V
        grad_W = -2.0 * (
            advantage[:, None, None] * grad_log_pi
        ).mean(dim=0)

        # 5. Euclidean mirror descent = ordinary gradient descent with a
        # shared tau-controlled step-size decay across value and policy
        # updates: zeta_j = zeta_0 / (1 + j / tau).
        self.update_index += 1
        decay = 1.0 / (1.0 + float(self.update_index) / self.tau)
        value_step_size = self.lr_value * decay
        policy_step_size = self.lr_policy * decay
        self.theta = self.theta - value_step_size * grad_theta
        self.W = self.W - policy_step_size * grad_W

        stats = self.objective()
        stats.update(
            {
                "theta_grad_norm": float(torch.linalg.norm(grad_theta).item()),
                "policy_grad_norm": float(torch.linalg.norm(grad_W).item()),
                "value_step_size": float(value_step_size),
                "policy_step_size": float(policy_step_size),
                "update_index": int(self.update_index),
            }
        )
        return stats

    def sample_action(
        self,
        state: int,
        behavior: str = "policy",
        epsilon: float = 0.0,
    ) -> int:
        """
        Sample a discrete action for data collection.

        `behavior="uniform"` gives the paper's simple random behavior policy.
        `behavior="policy"` follows the current softmax policy, which is uniform
        at initialization because W starts at zero. With epsilon > 0, this
        becomes epsilon-greedy exploration around the current policy.
        """
        state = int(state)
        if state < 0 or state >= self.n_states:
            raise ValueError("state must be in [0, n_states)")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")

        if behavior not in {"policy", "uniform"}:
            raise ValueError("behavior must be 'policy' or 'uniform'")

        if behavior == "uniform" or self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))

        probs = self.get_policy_matrix()[state].detach().cpu().numpy()
        return int(self.rng.choice(self.n_actions, p=probs))

    @staticmethod
    def _parse_transition_result(result: Any) -> Tuple[int, Optional[float], bool]:
        """
        Accept common discrete transition signatures:
            next_state
            (next_state, reward)
            (next_state, reward, done)
            (next_state, reward, terminated, truncated)
            (next_state, reward, terminated, truncated, info)
        """
        if isinstance(result, tuple):
            if len(result) == 2:
                next_state, reward = result
                return int(next_state), float(reward), False
            if len(result) == 3:
                next_state, reward, done = result
                return int(next_state), float(reward), bool(done)
            if len(result) == 4:
                next_state, reward, terminated, truncated = result
                return int(next_state), float(reward), bool(terminated) or bool(truncated)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                return int(next_state), float(reward), bool(terminated) or bool(truncated)
            raise ValueError("Unsupported transition_fn return tuple length")
        return int(result), None, False

    def collect_steps(
        self,
        transition_fn: Callable[[int, int], Any],
        n_steps: int,
        start_state: Optional[int] = None,
        reward_fn: Optional[Callable[[int, int, int], float]] = None,
        behavior: str = "policy",
        epsilon: float = 0.0,
        terminal_states: Optional[set] = None,
        reset_state_fn: Optional[Callable[[], int]] = None,
    ) -> int:
        """
        Collect transitions into D with the current behavior policy.

        This is the online part of Algorithm 1:
            execute behavior policy pi_b, append (s, a, r, s') to D.

        For simple deterministic grids, `transition_fn(s, a)` may return only
        next_state and `reward_fn(s, a, next_state)` supplies the reward.
        If a terminal state is reached, collection resets to x0 or
        reset_state_fn().
        """
        n_steps = int(n_steps)
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")

        terminal_states = set() if terminal_states is None else {int(s) for s in terminal_states}
        if start_state is None:
            state = self.spec.x0 if self.spec.x0 is not None else 0
        else:
            state = int(start_state)

        for _ in range(n_steps):
            if state in terminal_states:
                state = int(reset_state_fn()) if reset_state_fn is not None else (
                    self.spec.x0 if self.spec.x0 is not None else 0
                )

            action = self.sample_action(state, behavior=behavior, epsilon=epsilon)
            next_state, reward, done = self._parse_transition_result(transition_fn(state, action))
            if reward is None:
                if reward_fn is None:
                    raise ValueError("reward_fn is required when transition_fn does not return reward")
                reward = float(reward_fn(state, action, next_state))
            done = bool(done or next_state in terminal_states)

            if self.buffer_mode == "fifo":
                self.dataset.append_fifo(
                    state,
                    action,
                    reward,
                    next_state,
                    capacity=self.max_buffer_size,
                )
            else:
                self.dataset.append(state, action, reward, next_state)

            if done:
                state = int(reset_state_fn()) if reset_state_fn is not None else (
                    self.spec.x0 if self.spec.x0 is not None else 0
                )
            else:
                state = int(next_state)

        self.dataset.validate(self.n_states, self.n_actions)
        self._refresh_dataset_features()
        return self.dataset.n

    def run(
        self,
        transition_fn: Callable[[int, int], Any],
        reward_fn: Optional[Callable[[int, int, int], float]] = None,
        episodes: int = 100,
        collect_per_episode: int = 10,
        updates_per_episode: int = 10,
        initial_collect_steps: int = 0,
        start_state: Optional[int] = None,
        behavior: str = "policy",
        epsilon: float = 0.1,
        terminal_states: Optional[set] = None,
        reset_state_fn: Optional[Callable[[], int]] = None,
        verbose: bool = False,
        log_every: int = 10,
        tqdm_print: bool = True,
        store_history: bool = True,
    ) -> torch.Tensor:
        """
        Online SBEED loop matching Algorithm 1 at a notebook-friendly level.

        Each invocation starts from a fresh replay buffer D.

        For each episode:
            1. collect K transitions with behavior policy pi_b into D,
            2. run N primal/dual updates by sampling from the growing D,
            3. continue collecting with the updated policy.

        Reproducibility is controlled by the solver seed. The initial policy is
        uniform because W is initialized to zeros. The update counter j resets
        to 0 at the start of each run, so the effective learning rates at
        update j >= 1 are lr_value / (1 + j / tau) and
        lr_policy / (1 + j / tau).
        """
        episodes = int(episodes)
        collect_per_episode = int(collect_per_episode)
        updates_per_episode = int(updates_per_episode)
        initial_collect_steps = int(initial_collect_steps)
        log_every = int(log_every)
        if episodes < 0 or collect_per_episode < 0 or updates_per_episode < 0:
            raise ValueError("episodes, collect_per_episode, and updates_per_episode must be non-negative")
        if initial_collect_steps < 0:
            raise ValueError("initial_collect_steps must be non-negative")
        if log_every <= 0:
            raise ValueError("log_every must be positive")

        if store_history:
            self.theta_history = []
            self.beta_history = []
            self.W_history = []
            self.loss_history = []

        self.dataset = SBEEDDataset.empty(device=self.device)
        self._refresh_dataset_features()
        self.update_index = 0

        if initial_collect_steps > 0:
            self.collect_steps(
                transition_fn=transition_fn,
                n_steps=initial_collect_steps,
                start_state=start_state,
                reward_fn=reward_fn,
                behavior="uniform",
                epsilon=0.0,
                terminal_states=terminal_states,
                reset_state_fn=reset_state_fn,
            )

        iterator = trange(episodes, disable=not tqdm_print)
        last_stats = None
        for episode in iterator:
            # Periodic logging happens before collecting the next block of data.
            # The reported metrics are from the previous completed update block.
            if verbose and last_stats is not None and episode % log_every == 0:
                print(
                    f"episode={episode}/{episodes} "
                    f"buffer={self.n} "
                    f"objective={last_stats['objective']:.6f} "
                    f"primal_mse={last_stats['primal_mse']:.6f} "
                    f"dual_mse={last_stats['dual_mse']:.6f} "
                    f"theta_grad={last_stats['theta_grad_norm']:.3e} "
                    f"policy_grad={last_stats['policy_grad_norm']:.3e}"
                )

            self.collect_steps(
                transition_fn=transition_fn,
                n_steps=collect_per_episode,
                start_state=start_state,
                reward_fn=reward_fn,
                behavior=behavior,
                epsilon=epsilon,
                terminal_states=terminal_states,
                reset_state_fn=reset_state_fn,
            )

            for _ in range(updates_per_episode):
                last_stats = self.step()
                if store_history:
                    self.theta_history.append(self.theta.detach().clone())
                    self.beta_history.append(self.beta.detach().clone())
                    self.W_history.append(self.W.detach().clone())
                    self.loss_history.append(last_stats)

            if tqdm_print:
                postfix = {"buffer": self.n}
                if last_stats is not None:
                    postfix.update(
                        {
                            "objective": f"{last_stats['objective']:.4g}",
                            "primal_mse": f"{last_stats['primal_mse']:.4g}",
                            "dual_mse": f"{last_stats['dual_mse']:.4g}",
                            "theta_grad": f"{last_stats['theta_grad_norm']:.2e}",
                            "policy_grad": f"{last_stats['policy_grad_norm']:.2e}",
                        }
                    )
                iterator.set_postfix(postfix)

        if verbose and last_stats is not None:
            print(
                f"episode={episodes}/{episodes} "
                f"buffer={self.n} "
                f"objective={last_stats['objective']:.6f} "
                f"primal_mse={last_stats['primal_mse']:.6f} "
                f"dual_mse={last_stats['dual_mse']:.6f} "
                f"theta_grad={last_stats['theta_grad_norm']:.3e} "
                f"policy_grad={last_stats['policy_grad_norm']:.3e}"
            )

        self.pi = self.get_policy_matrix()
        return self.pi

    def get_policy_matrix(self, W: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._policy_matrix(W).detach().clone()

    def policy_fn(self, state: int) -> torch.Tensor:
        state = int(state)
        if state < 0 or state >= self.n_states:
            raise ValueError("state must be in [0, n_states)")
        return self.get_policy_matrix()[state]

    def value(self, state: int) -> torch.Tensor:
        feat = self.PHI_S[int(state)]
        return feat @ self.theta

    def rho(self, state: int, action: int) -> torch.Tensor:
        feat = self.RHO_SA[int(state), int(action)]
        return feat @ self.beta
