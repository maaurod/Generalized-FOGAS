"""Fitted Q Iteration baseline.

The solver fits a linear Q-function over user-provided state-action features.
It can run from offline transition data, from a known model, or from an
oracle-style optimal target used to study representation error.
"""

import torch
import random
import numpy as np
from tqdm import trange

from ..fogas.fogas_dataset import FOGASDataset


class FQISolver:
    """
    Fitted Q Iteration implementation.

    Two modes:
    - Dataset-based (default): targets from dataset transitions only.
      target_i = r_i + gamma * max_a' Q(x'_i, a'; theta_k), fit on (x_i, a_i).
      With few RBF centers, many (s,a) share similar features so the fit can fail.
    - Model-based (use_model_based_backup=True): use full MDP (P, r) to compute
      targets for every state-action: target(s,a) = r(s,a) + gamma * E_s'[max_a' Q(s',a')].
      Then fit theta on all (s,a). Works with RBF/linear features when the model is known.
    - Optimal Model-based (use_optimal_target_backup=True): same as Model-based, but
      targets use the true V* from the MDP: target(s,a) = r(s,a) + gamma * E_s'[V*(s')].
      This regresses the feature weights towards the best possible feature representation of Q*.

    Update rule:
      1. target_i = r_i + gamma * max_a' Q(x'_i, a'; theta_k)  [or full model / optimal V*]
      2. theta_{k+1}^+ = argmin_theta sum (target - Q(.; theta))^2 = (Phi^T Phi + ridge I)^{-1} Phi^T Y
      3. theta_{k+1} = tau * theta_k + (1 - tau) * theta_{k+1}^+

    Where Q(s, a; theta) = phi(s, a)^T theta.
    """

    def __init__(
        self,
        mdp,
        phi=None,
        csv_path=None,
        gamma=None,
        ridge=1.0,
        dataset_verbose=False,
        seed=42,
        device=None,
        planner=None,
        augment_terminal_transitions=True,
        use_model_based_backup=False,
        use_optimal_target_backup=False,
    ):
        self.mdp = mdp
        self.planner = planner
        self.gamma = gamma if gamma is not None else mdp.gamma
        self.ridge = ridge
        self.seed = seed
        self.augment_terminal_transitions = augment_terminal_transitions
        self.use_model_based_backup = use_model_based_backup
        self.use_optimal_target_backup = use_optimal_target_backup

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        if hasattr(self.mdp, 'to'):
            self.mdp.to(self.device)

        self.A = mdp.A
        self.phi = self._resolve_phi(phi)
        self.d = self._resolve_feature_dim()
        self.N = getattr(mdp, 'N', None)

        if use_model_based_backup or use_optimal_target_backup:
            self._init_model_based()
        else:
            if csv_path is None:
                raise ValueError("csv_path is required when use_model_based_backup or use_optimal_target_backup is False.")
            self._init_dataset_based(csv_path, dataset_verbose)

        self.theta_history = []
        self.final_theta = None
        self.pi = None

    def _init_model_based(self):
        """Use full MDP P and r for Bellman backup over all state-actions (no dataset)."""
        if self.N is None:
            self.N = len(self.mdp.states)
        # Phi: (N*A, d) in row order (s0,a0), (s0,a1), ..., (sN-1, aA-1)
        Phi_full = self._full_feature_matrix()
        r = self.mdp.get_reward().to(dtype=torch.float64, device=self.device)
        P = self.mdp.get_transition_matrix().to(dtype=torch.float64, device=self.device)
        if Phi_full.shape[0] != self.N * self.A or P.shape != (self.N * self.A, self.N):
            raise ValueError(
                "Model-based FQI requires mdp.Phi (N*A, d) and transition matrix P (N*A, N)."
            )
        Gram = Phi_full.T @ Phi_full + self.ridge * torch.eye(
            self.d, dtype=torch.float64, device=self.device
        )
        self.M = torch.linalg.solve(Gram, Phi_full.T)
        self.Phi_full = Phi_full
        self.r_full = r
        self.P_full = P
        self.n = self.N * self.A
        self._state_to_index = self._build_state_to_index()

    def _resolve_phi(self, phi):
        if phi is not None:
            return phi
        if hasattr(self.mdp, "phi"):
            return self.mdp.phi
        raise ValueError(
            "FQISolver requires phi when using mdp.DiscreteMDP, "
            "because this MDP implementation does not store feature functions."
        )

    def _resolve_feature_dim(self):
        if hasattr(self.mdp, "d"):
            return int(self.mdp.d)
        state = self.mdp.states[0]
        action = self.mdp.actions[0]
        state = int(state.item()) if isinstance(state, torch.Tensor) else int(state)
        action = int(action.item()) if isinstance(action, torch.Tensor) else int(action)
        feat = self.phi(state, action)
        return int(torch.as_tensor(feat).reshape(-1).numel())

    def _full_feature_matrix(self):
        if hasattr(self.mdp, "Phi"):
            return self.mdp.Phi.to(dtype=torch.float64, device=self.device)
        rows = []
        for state in self.mdp.states:
            x = int(state.item()) if isinstance(state, torch.Tensor) else int(state)
            for action in self.mdp.actions:
                a = int(action.item()) if isinstance(action, torch.Tensor) else int(action)
                rows.append(self.phi(x, a).to(dtype=torch.float64, device=self.device))
        return torch.vstack(rows)

    def _init_dataset_based(self, csv_path, dataset_verbose):
        """Standard FQI: load dataset and build regression from dataset samples."""
        self.dataset = FOGASDataset(csv_path=csv_path, verbose=dataset_verbose)
        self.n = self.dataset.n
        self.Xs = self.dataset.X.to(self.device)
        self.As = self.dataset.A.to(self.device)
        self.Rs = self.dataset.R.to(self.device)
        self.X_nexts = self.dataset.X_next.to(self.device)
        self._state_to_index = self._build_state_to_index()
        self.added_terminal_samples = 0
        if self.augment_terminal_transitions:
            self._augment_missing_terminal_transitions(dataset_verbose=dataset_verbose)
        self._precompute_dataset_features()
        self._build_regression_solver()
        if self.N is None:
            self.N = len(self.mdp.states)

    def _build_state_to_index(self):
        """Build map from raw state value to row index in mdp.states."""
        if not hasattr(self.mdp, "states"):
            return None

        state_to_index = {}
        for idx, s in enumerate(self.mdp.states):
            s_val = int(s.item()) if isinstance(s, torch.Tensor) else int(s)
            state_to_index[s_val] = idx
        return state_to_index

    def _get_terminal_states(self):
        """Extract terminal states from common MDP attributes."""
        for attr in ("terminal_states", "T", "terminal_set"):
            if hasattr(self.mdp, attr):
                raw = getattr(self.mdp, attr)
                return {
                    int(s.item()) if isinstance(s, torch.Tensor) else int(s)
                    for s in raw
                }
        return set()

    def _state_action_row_index(self, state, action):
        """Map (state value, action index) to row index in mdp.r / mdp.P."""
        state = int(state)
        action = int(action)
        if self._state_to_index is None:
            return state * self.A + action
        if state not in self._state_to_index:
            raise ValueError(f"State {state} not found in mdp.states.")
        return self._state_to_index[state] * self.A + action

    def _reward_from_mdp(self, state, action):
        """
        Resolve reward for a synthetic transition from model data.
        Prefer mdp.r if available, otherwise fallback to <phi, omega>.
        """
        if hasattr(self.mdp, "r"):
            row_idx = self._state_action_row_index(state, action)
            r_val = self.mdp.r[row_idx]
            if isinstance(r_val, torch.Tensor):
                return float(r_val.item())
            return float(r_val)

        if not hasattr(self.mdp, "omega"):
            raise AttributeError(
                "Cannot infer synthetic reward: mdp must expose either `r` or `omega`."
            )

        feat = self.phi(state, action).to(dtype=torch.float64, device=self.device)
        omega = self.mdp.omega
        if isinstance(omega, torch.Tensor):
            omega = omega.to(dtype=torch.float64, device=self.device)
        else:
            omega = torch.as_tensor(omega, dtype=torch.float64, device=self.device)
        return float(torch.dot(feat, omega).item())

    def _augment_missing_terminal_transitions(self, dataset_verbose=False):
        """
        Add one synthetic self-loop transition for every missing terminal (s, a).

        This prevents degenerate bootstrapping when offline data never contains
        terminal-state current samples (common when episodes reset immediately).
        """
        terminal_states = self._get_terminal_states()
        if not terminal_states:
            return

        existing_pairs = {
            (int(s.item()), int(a.item()))
            for s, a in zip(self.Xs, self.As)
        }
        to_add = []

        for s in sorted(terminal_states):
            if self._state_to_index is not None and s not in self._state_to_index:
                continue
            for a in range(self.A):
                if (s, a) in existing_pairs:
                    continue
                reward = self._reward_from_mdp(s, a)
                to_add.append((s, a, reward, s))

        if not to_add:
            return

        x_aug = torch.tensor([row[0] for row in to_add], dtype=self.Xs.dtype, device=self.device)
        a_aug = torch.tensor([row[1] for row in to_add], dtype=self.As.dtype, device=self.device)
        r_aug = torch.tensor([row[2] for row in to_add], dtype=self.Rs.dtype, device=self.device)
        x_next_aug = torch.tensor([row[3] for row in to_add], dtype=self.X_nexts.dtype, device=self.device)

        self.Xs = torch.cat([self.Xs, x_aug], dim=0)
        self.As = torch.cat([self.As, a_aug], dim=0)
        self.Rs = torch.cat([self.Rs, r_aug], dim=0)
        self.X_nexts = torch.cat([self.X_nexts, x_next_aug], dim=0)
        self.n = int(self.Xs.shape[0])
        self.added_terminal_samples = len(to_add)

        if dataset_verbose:
            print(
                f"Added {self.added_terminal_samples} synthetic terminal transitions "
                "to improve terminal value bootstrapping."
            )

    def _precompute_dataset_features(self):
        """
        1. Phi: Features for (x_i, a_i). Shape (n, d).
        2. Phi_next_all: Features for (x'_i, a') for all a'. Shape (n, A, d).
           Used for efficiently computing max_a' Q(x'_i, a').
        """
        n, d, A = self.n, self.d, self.A
        phi = self.phi
        device = self.device

        # Regression inputs: each offline row contributes one feature vector
        # for the observed current state-action pair.
        Phi_list = [
            phi(int(x.item()), int(a.item())).to(dtype=torch.float64, device=device)
            for x, a in zip(self.Xs, self.As)
        ]
        self.Phi = torch.vstack(Phi_list)  # (n, d)

        # Bellman targets need max_a' Q(x'_i, a'), so cache all action features
        # for every next state once instead of rebuilding them each iteration.
        Phi_next_list = []
        for x_next in self.X_nexts:
            x_next_val = int(x_next.item())
            feats = [
                phi(x_next_val, a).to(dtype=torch.float64, device=device)
                for a in range(A)
            ]
            Phi_next_list.append(torch.stack(feats)) # (A, d)
        
        self.Phi_next_all = torch.stack(Phi_next_list) # (n, A, d)

    def _build_regression_solver(self):
        """
        Precompute the matrix M = (Phi^T Phi + ridge*I)^(-1) Phi^T
        Then theta_new = M @ targets
        """
        d = self.d
        n = self.n
        ridge = self.ridge
        
        Phi = self.Phi # (n, d)
        
        # Regularized Gram matrix: Phi^T Phi + lambda*I
        Gram = Phi.T @ Phi + ridge * torch.eye(d, dtype=torch.float64, device=self.device)
        
        # Inverse
        Gram_inv = torch.linalg.inv(Gram)
        
        # Projection matrix M: (d, d) @ (d, n) -> (d, n)
        self.M = Gram_inv @ Phi.T

    def run(self, K=100, tau=0.1, theta_init=None, verbose=False):
        """
        Run FQI for K iterations.

        Args:
            K (int): Number of iterations.
            tau (float): Soft-update weight (theta_{k+1} = tau*theta_k + (1-tau)*theta_plus).
            theta_init (torch.Tensor): Initial theta (d,). If None, starts at zeros.
            verbose (bool): Whether to show progress.
        """
        d = self.d
        device = self.device

        if theta_init is None:
            theta = torch.zeros(d, dtype=torch.float64, device=device)
        else:
            theta = theta_init.clone().to(dtype=torch.float64, device=device)

        params_history = []
        iterator = trange(K, desc="FQI", disable=not verbose)

        # The three modes share the same least-squares projection step but
        # differ in how they construct Bellman targets.
        if self.use_optimal_target_backup:
            self._run_optimal_target_based(theta, params_history, iterator, tau, verbose)
        elif self.use_model_based_backup:
            self._run_model_based(theta, params_history, iterator, tau, verbose)
        else:
            self._run_dataset_based(theta, params_history, iterator, tau, verbose)

        self.final_theta = theta
        self.theta_history = params_history
        self.pi = self.get_policy_matrix(theta)
        return self.get_greedy_policy(theta)

    def _run_model_based(self, theta, params_history, iterator, tau, verbose=False):
        """Bellman backup using full P and r over all state-actions."""
        N, A = self.N, self.A
        Phi_full = self.Phi_full
        r = self.r_full
        P = self.P_full
        for k in iterator:
            # Q(s,a) for all (s,a): (N*A,) from (N*A, d) @ (d,)
            Q_vec = Phi_full @ theta
            # V(s) = max_a Q(s,a): (N,)
            Q_mat = Q_vec.reshape(N, A)
            V, _ = torch.max(Q_mat, dim=1)
            # target(s,a) = r(s,a) + gamma * sum_s' P(s'|s,a) V(s')
            targets = r + self.gamma * (P @ V)
            theta_plus = self.M @ targets
            theta.mul_(tau).add_(theta_plus, alpha=1.0 - tau)
            params_history.append(theta.clone())
            if verbose and (k % 10 == 0):
                iterator.set_postfix(theta_norm=f"{torch.linalg.norm(theta).item():.4f}")

    def _run_optimal_target_based(self, theta, params_history, iterator, tau, verbose=False):
        """Bellman backup using full P and r, but fixed V* from a planner/old PolicySolver."""
        if self.planner is not None:
            V_star = self.planner.v_star.to(dtype=torch.float64, device=self.device)
        elif hasattr(self.mdp, 'v_star'):
            V_star = self.mdp.v_star.to(dtype=torch.float64, device=self.device)
        else:
            raise AttributeError(
                "Optimal target backup requires planner=Planner(mdp) with v_star, "
                "or an old MDP exposing v_star."
            )
        
        N, A = self.N, self.A
        r = self.r_full
        P = self.P_full
        
        # target(s,a) = r(s,a) + gamma * sum_s' P(s'|s,a) V*(s')
        # This target is fixed!
        targets = r + self.gamma * (P @ V_star)
        
        # Precompute theta_plus once since targets are fixed
        theta_plus = self.M @ targets
        
        for k in iterator:
            theta.mul_(tau).add_(theta_plus, alpha=1.0 - tau)
            params_history.append(theta.clone())
            if verbose and (k % 10 == 0):
                iterator.set_postfix(theta_norm=f"{torch.linalg.norm(theta).item():.4f}")

    def _run_dataset_based(self, theta, params_history, iterator, tau, verbose=False):
        """Bellman backup using only dataset transitions (standard FQI)."""
        for k in iterator:
            # Standard FQI target for each sampled transition:
            # r_i + gamma max_a' Q_theta(x'_i, a').
            Q_next_all = torch.einsum('nad,d->na', self.Phi_next_all, theta)
            max_Q_next, _ = torch.max(Q_next_all, dim=1)
            targets = self.Rs + self.gamma * max_Q_next
            theta_plus = self.M @ targets
            theta.mul_(tau).add_(theta_plus, alpha=1.0 - tau)
            params_history.append(theta.clone())
            if verbose and (k % 10 == 0):
                iterator.set_postfix(theta_norm=f"{torch.linalg.norm(theta).item():.4f}")

    def get_greedy_policy(self, theta=None):
        """
        Returns a function pi(x) -> probabilities (one-hot greedy)
        or can return a full (N, A) matrix if needed.
        """
        if theta is None:
            theta = self.final_theta
            
        def policy_fn(state_idx):
            state_idx = int(state_idx)
            # Compute Q(s, a) for all a
            q_values = []
            for a in range(self.A):
                feat = self.phi(state_idx, a).to(dtype=torch.float64, device=self.device)
                q = torch.dot(theta, feat)
                q_values.append(q)
            q_values = torch.stack(q_values)
            
            # Greedy
            best_a = torch.argmax(q_values)
            probs = torch.zeros(self.A, dtype=torch.float64, device=self.device)
            probs[best_a] = 1.0
            return probs
            
        return policy_fn
    
    def get_policy_matrix(self, theta=None):
        """
        Returns the (N, A) policy matrix for the greedy policy w.r.t theta.
        """
        if theta is None:
            theta = self.final_theta
            
        N = self.mdp.N
        A = self.A
        pi_mat = torch.zeros((N, A), dtype=torch.float64, device=self.device)
        
        for x_idx, x_val in enumerate(self.mdp.states):
            x = int(x_val.item()) if isinstance(x_val, torch.Tensor) else int(x_val)
            # Compute Q(x, :)
            q_vals = torch.zeros(A, dtype=torch.float64, device=self.device)
            for a in range(A):
                feat = self.phi(x, a).to(dtype=torch.float64, device=self.device)
                q_vals[a] = torch.dot(theta, feat)
            
            best_a = torch.argmax(q_vals)
            pi_mat[x_idx, best_a] = 1.0
            
        return pi_mat

    @property
    def theta_bar_history(self):
        """Alias for compatibility with Evaluators."""
        return self.theta_history
    
    @property
    def mod_alpha(self):
        """FQI doesn't use alpha, but Evaluator expects it for some metrics."""
        return 1.0
