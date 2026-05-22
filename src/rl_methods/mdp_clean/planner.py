import torch


class Planner:
    """
    Dynamic-programming utilities for a DiscreteMDP.

    The planner computes exact value functions, action-value functions,
    policies, and discounted occupancy measures by composition over an MDP.
    """

    def __init__(self, mdp, mode="deterministic", temperature=1.0):
        self.mdp = mdp
        self.N = mdp.N
        self.A = mdp.A
        self.gamma = mdp.gamma
        self.states = mdp.states
        self.actions = mdp.actions

        self.pi_star, self.v_star, self.q_star = self.policy_iteration(
            mode=mode,
            temperature=temperature,
            verbose=False,
        )
        self.mu_star = self.occupancy_measure(self.pi_star)

    def to(self, device):
        self.mdp.to(device)
        self.states = self.mdp.states
        self.actions = self.mdp.actions
        self.pi_star = self.pi_star.to(device)
        self.v_star = self.v_star.to(device)
        self.q_star = self.q_star.to(device)
        self.mu_star = self.mu_star.to(device)
        return self

    @property
    def r(self):
        return self.mdp.r

    @property
    def P(self):
        return self.mdp.P

    @property
    def x0(self):
        return self.mdp.x0

    @property
    def nu0(self):
        return self.mdp.nu0

    def evaluate_policy(self, pi, verbose=False):
        device = self.mdp.r.device
        pi = pi.to(dtype=torch.float64, device=device)
        r = self.mdp.r
        P = self.mdp.P

        r_pi = (pi * r.reshape(self.N, self.A)).sum(dim=1)

        P_pi = torch.zeros((self.N, self.N), dtype=torch.float64, device=device)
        for a in range(self.A):
            P_pi += torch.diag(pi[:, a]) @ P[a::self.A, :]

        eye = torch.eye(self.N, dtype=torch.float64, device=device)
        v = torch.linalg.solve(eye - self.gamma * P_pi, r_pi)
        q = (r + self.gamma * P @ v).reshape(self.N, self.A)

        if verbose:
            self.print_results(pi=pi, v=v, q=q)

        return v, q

    def policy_iteration(self, mode="deterministic", temperature=1.0, max_iter=1000, eps=1e-8, verbose=False):
        device = self.mdp.r.device
        pi = torch.ones((self.N, self.A), dtype=torch.float64, device=device) / self.A

        for it in range(max_iter):
            _, q = self.evaluate_policy(pi)

            if mode == "deterministic":
                new_pi = torch.zeros_like(pi)
                best_a = torch.argmax(q, dim=1)
                new_pi[torch.arange(self.N, device=device), best_a] = 1.0
            elif mode == "softmax":
                logits = q / float(temperature)
                logits = logits - logits.max(dim=1, keepdim=True).values
                new_pi = torch.exp(logits)
                new_pi = new_pi / new_pi.sum(dim=1, keepdim=True)
            else:
                raise ValueError("mode must be 'deterministic' or 'softmax'")

            if torch.allclose(new_pi, pi, atol=eps):
                pi = new_pi
                if verbose:
                    print(f"Converged at iteration {it + 1}")
                break
            pi = new_pi

        v, q = self.evaluate_policy(pi)
        return pi, v, q

    def occupancy_measure(self, pi):
        device = self.mdp.r.device
        pi = pi.to(dtype=torch.float64, device=device)

        comp_pi = torch.zeros((self.N * self.A, self.N), dtype=torch.float64, device=device)
        for x in range(self.N):
            for a in range(self.A):
                comp_pi[x * self.A + a, x] = pi[x, a]

        eye = torch.eye(self.N * self.A, dtype=torch.float64, device=device)
        rhs = (1.0 - self.gamma) * (comp_pi @ self.mdp.nu0)
        return torch.linalg.solve(eye - self.gamma * comp_pi @ self.mdp.P.T, rhs)

    @property
    def state_mu_star(self):
        return self.mu_star.reshape(self.N, self.A).sum(dim=1)

    def policy_return(self, pi):
        v, _ = self.evaluate_policy(pi)
        return float((1.0 - self.gamma) * self.mdp.nu0 @ v)

    def optimal_policy_return(self):
        return float((1.0 - self.gamma) * self.mdp.nu0 @ self.v_star)

    def optimal_q_feature_weights(self, Phi, ridge=1e-10):
        Phi = Phi.to(dtype=torch.float64, device=self.mdp.r.device)
        y = self.q_star.reshape(-1)
        d = int(Phi.shape[1])
        gram = Phi.T @ Phi + ridge * torch.eye(d, dtype=torch.float64, device=Phi.device)
        return torch.linalg.solve(gram, Phi.T @ y)

    def optimal_feature_occupancy(self, Phi):
        Phi = Phi.to(dtype=torch.float64, device=self.mdp.r.device)
        return Phi.T @ self.mu_star

    def policy_from_optimal_q(self, mode="set", temperature=0.05, tie_eps=1e-10):
        q = self.q_star
        dtype = q.dtype

        if mode == "set":
            qmax = q.max(dim=1, keepdim=True).values
            mask = q >= (qmax - tie_eps)
            pi = mask.to(dtype)
            return pi / pi.sum(dim=1, keepdim=True)

        if mode == "softmax":
            logits = (q - q.max(dim=1, keepdim=True).values) / temperature
            return torch.softmax(logits, dim=1)

        if mode == "masked_softmax":
            qmax = q.max(dim=1, keepdim=True).values
            mask = q >= (qmax - tie_eps)
            logits = (q - qmax) / temperature
            logits = torch.where(mask, logits, torch.full_like(logits, -torch.inf))
            return torch.softmax(logits, dim=1)

        raise ValueError("mode must be 'set', 'softmax', or 'masked_softmax'")

    def print_results(self, pi=None, v=None, q=None):
        print("\n========== POLICY - VALUE RESULTS ==========\n")
        if v is not None:
            for idx, s in enumerate(self.mdp.states):
                print(f"State {s}: V = {v[idx].item():.4f}")
            print()

        if q is not None:
            print("Action-Value Function (Q):")
            for i, s in enumerate(self.mdp.states):
                print(f"  State {s}:")
                for j, a in enumerate(self.mdp.actions):
                    print(f"    Action {a}: Q(s={s}, a={a}) = {q[i, j].item():.4f}")
                print()

        if pi is not None:
            self.mdp.print_policy(pi)

        print("=============================================\n")

    def print_optimals(self, occupancy=False):
        self.print_results(self.pi_star, self.v_star, self.q_star)
        if occupancy:
            self.print_occupancy(self.mu_star)

    def print_occupancy(self, mu):
        mu_matrix = mu.reshape(self.N, self.A)
        print("\nDiscounted Occupancy Measure")
        header = "State | " + " | ".join(f"{a:>8}" for a in self.mdp.actions) + " |   Sum"
        print(header)
        for i, s in enumerate(self.mdp.states):
            row_vals = " | ".join(f"{mu_matrix[i, j].item():8.5f}" for j in range(self.A))
            row_sum = torch.sum(mu_matrix[i]).item()
            print(f"{str(s):>5} | {row_vals} | {row_sum:6.5f}")
        print(f"Total sum = {torch.sum(mu_matrix).item():.6f} (should be approx 1.0)\n")
