import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix


class CTMDPTumorBase:
    def load_common_params(self, params, *, default_n_actions=101):
        self.p = dict(params)
        required = ["N", "r", "delta", "alpha", "penalty"]
        missing = [key for key in required if key not in self.p]
        if missing:
            raise KeyError(f"Missing required parameter(s): {missing}")
        self.N = int(self.p["N"])
        self.r = float(self.p["r"])
        self.delta = float(self.p["delta"])
        self.delta0 = float(self.p.get("delta0", 0.0))
        self.alpha = float(self.p["alpha"])
        self.penalty = float(self.p["penalty"])
        # Keep delta0 explicit for downstream code.
        self.p["delta0"] = self.delta0
        # Accept both naming conventions.
        if "n_actions" in self.p:
            self.n_actions = int(self.p["n_actions"])
        elif "n_a" in self.p:
            self.n_actions = int(self.p["n_a"])
        else:
            self.n_actions = int(default_n_actions)
        self.actions = np.linspace(0.0, 1.0, self.n_actions)
        self.update_zero_drift_action()

    def update_zero_drift_action(self):
        denom = self.r + self.delta
        if denom <= 0:
            raise ValueError("r + delta must be positive.")
        self.a_star = (self.delta + self.delta0) / denom
        # Feasible clipped action for initialization/defaults.
        self.a_init = float(np.clip(self.a_star, 0.0, 1.0))

    def _zero_like_action(self, a):
        if np.isscalar(a):
            return 0.0
        return np.zeros_like(a, dtype=float)

    def lam(self, i, a):
        if i >= self.N:
            return self._zero_like_action(a)
        return self.r * i * a

    def mu(self, i, a):
        if i <= 0:
            return self._zero_like_action(a)
        return self.delta * i * (1.0 - a) + self.delta0 * i

    def drift(self, i, a):
        if i <= 0:
            return self._zero_like_action(a)
        return self.r * a - self.delta * (1.0 - a) - self.delta0

    def zero_drift_feasible(self):
        return 0.0 <= self.a_star <= 1.0

    def max_total_rate_per_cell(self):
        return self.delta0 + max(self.r, self.delta)

    def global_uniformization_rate(self):
        Lambda = self.N * self.max_total_rate_per_cell()
        if Lambda <= 0:
            raise ValueError("Uniformization rate must be positive.")
        return Lambda

    def state_uniformization_rates(self):
        I = np.arange(self.N + 1, dtype=float)
        return I * self.max_total_rate_per_cell()

    def build_state_action_grid(self):
        I = np.arange(self.N + 1)[:, np.newaxis]
        A = self.actions[np.newaxis, :]
        return I, A

    def lam_grid(self, I, A):
        lam = self.r * I * A
        lam[self.N, :] = 0.0
        return lam

    def mu_grid(self, I, A):
        mu = self.delta * I * (1.0 - A) + self.delta0 * I
        mu[0, :] = 0.0
        return mu

    def policy_drift(self, policy=None):
        if policy is None:
            if not hasattr(self, "policy"):
                raise AttributeError("No stored policy found.")
            policy = self.policy
        policy = np.asarray(policy, dtype=float)
        return np.array([self.drift(i, policy[i]) for i in range(self.N + 1)])

    def set_common_params(self, **kwargs):
        allowed = {
            "N",
            "r",
            "delta",
            "delta0",
            "alpha",
            "penalty",
            "n_actions",
            "n_a",
        }
        for key, value in kwargs.items():
            if key not in allowed:
                raise AttributeError(f"Unknown common parameter: {key}")
            self.p[key] = value
        self.load_common_params(self.p, default_n_actions=self.n_actions)


class DormancyCTMDP_DiscountedPI(CTMDPTumorBase):
    def __init__(self, params):
        self.load_common_params(params, default_n_actions=101)
        self.load_cost_params()
        # Initialize policy with feasible clipped zero-drift action.
        self.policy = np.full(self.N + 1, self.a_init, dtype=float)
        self.policy[0] = 1.0
        self.V = np.zeros(self.N + 1, dtype=float)

    def load_cost_params(self):
        required = ["L", "kappa1", "kappa2"]
        missing = [key for key in required if key not in self.p]
        if missing:
            raise KeyError(f"Missing required cost parameter(s): {missing}")
        self.L = float(self.p["L"])
        self.kappa1 = float(self.p["kappa1"])
        self.kappa2 = float(self.p["kappa2"])
        if self.L <= 0:
            raise ValueError("L must be positive.")

    def cost(self, i, a):
        if i <= 0:
            if np.isscalar(a):
                return self.penalty
            return self.penalty * np.ones_like(a, dtype=float)
        return self.kappa1 * i * (a - self.a_star) ** 2 + self.kappa2 * (
            ((i / self.L) - 1.0) * a + 1.0
        )

    def evaluate_policy(self, policy):
        N = self.N
        alpha = self.alpha
        lower = np.zeros(N + 1, dtype=float)
        diag = np.zeros(N + 1, dtype=float)
        upper = np.zeros(N + 1, dtype=float)
        rhs = np.zeros(N + 1, dtype=float)
        for i in range(N + 1):
            a = float(policy[i])
            lam_i = self.lam(i, a)
            mu_i = self.mu(i, a)
            diag[i] = alpha + lam_i + mu_i
            rhs[i] = self.cost(i, a)
            if i > 0:
                lower[i] = -mu_i
            if i < N:
                upper[i] = -lam_i
        # Thomas algorithm for tridiagonal system.
        c_prime = np.zeros(N + 1, dtype=float)
        d_prime = np.zeros(N + 1, dtype=float)
        c_prime[0] = upper[0] / diag[0]
        d_prime[0] = rhs[0] / diag[0]
        for i in range(1, N + 1):
            denom = diag[i] - lower[i] * c_prime[i - 1]
            if abs(denom) < 1e-15:
                raise RuntimeError(
                    f"Near-singular tridiagonal solve at state {i}: denom={denom:.3e}"
                )
            c_prime[i] = upper[i] / denom if i < N else 0.0
            d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom
        V = np.zeros(N + 1, dtype=float)
        V[N] = d_prime[N]
        for i in range(N - 1, -1, -1):
            V[i] = d_prime[i] - c_prime[i] * V[i + 1]
        return V

    def improve_policy(self, V, policy_old):
        N = self.N
        alpha = self.alpha
        policy_new = np.zeros(N + 1, dtype=float)
        # Boundary condition: max proliferation at state 0.
        policy_new[0] = 1.0
        for i in range(1, N + 1):
            Vm = V[i - 1]
            Vp = V[i + 1] if i < N else 0.0
            # Vectorized over actions.
            lam_i = self.lam(i, self.actions)
            mu_i = self.mu(i, self.actions)
            vals = (self.cost(i, self.actions) + lam_i * Vp + mu_i * Vm) / (
                alpha + lam_i + mu_i
            )
            policy_new[i] = self.actions[np.argmin(vals)]
        change = float(np.max(np.abs(policy_new - policy_old)))
        return policy_new, change

    def policy_iteration(
        self,
        *,
        max_iter=200,
        policy_tol=1e-6,
        verbose=False,
    ):
        policy = np.asarray(self.policy, dtype=float).copy()
        policy[0] = 1.0
        for it in range(max_iter):
            V = self.evaluate_policy(policy)
            policy_new, change = self.improve_policy(V, policy)
            if verbose:
                print(f"PI iter {it:3d} | policy change sup = {change:.3e}")
            policy = policy_new
            if change < policy_tol:
                break
        self.policy = policy.copy()
        self.V = self.evaluate_policy(policy)
        return self.V.copy(), self.policy.copy()

    def solve(self, max_iter=200, tol=1e-6, verbose=False):
        return self.policy_iteration(
            max_iter=max_iter,
            policy_tol=tol,
            verbose=verbose,
        )

    def set_params(self, **kwargs):
        old_N = getattr(self, "N", None)
        self.p.update(kwargs)
        self.load_common_params(self.p, default_n_actions=101)
        self.load_cost_params()
        # If N changed, rebuild policy and value arrays.
        if old_N is None or self.N != old_N:
            self.policy = np.full(self.N + 1, self.a_init, dtype=float)
            self.policy[0] = 1.0
            self.V = np.zeros(self.N + 1, dtype=float)
        else:
            # Keep existing policy but enforce boundary condition.
            self.policy = np.asarray(self.policy, dtype=float)
            self.policy[0] = 1.0

    def set_initial_policy(self, policy):
        policy = np.asarray(policy, dtype=float)
        if len(policy) != self.N + 1:
            raise ValueError(f"Policy must have length {self.N + 1}.")
        self.policy = policy.copy()
        self.policy[0] = 1.0


class DormancyCTMDP_DiscountedPI_V2(CTMDPTumorBase):
    def __init__(self, params):
        self.load_common_params(params, default_n_actions=101)
        self.load_cost_params()
        # Initialize policy at feasible clipped zero-drift action.
        self.policy = np.full(self.N + 1, self.a_init, dtype=float)
        self.policy[0] = 1.0
        self.V = np.zeros(self.N + 1, dtype=float)

    def load_cost_params(self):
        required = ["c1", "c2", "kappa", "c3"]
        missing = [key for key in required if key not in self.p]
        if missing:
            raise KeyError(f"Missing required cost parameter(s): {missing}")
        self.c1 = float(self.p["c1"])
        self.c2 = float(self.p["c2"])
        self.kappa = float(self.p["kappa"])
        self.c3 = float(self.p["c3"])

    def cost(self, i, a):
        if i == 0:
            if np.isscalar(a):
                return self.penalty
            return self.penalty * np.ones_like(a, dtype=float)
        return (
            self.c1 * i**2
            + self.c2 * self.r * a * i**2
            + self.kappa * (a - self.a_star) ** 2 * i
            - self.c3 * i
        )

    def evaluate_policy(self, policy):
        N = self.N
        alpha = self.alpha
        lower = np.zeros(N + 1, dtype=float)
        diag = np.zeros(N + 1, dtype=float)
        upper = np.zeros(N + 1, dtype=float)
        rhs = np.zeros(N + 1, dtype=float)
        for i in range(N + 1):
            a = float(policy[i])
            lam_i = self.lam(i, a)
            mu_i = self.mu(i, a)
            diag[i] = alpha + lam_i + mu_i
            rhs[i] = self.cost(i, a)
            if i > 0:
                lower[i] = -mu_i
            if i < N:
                upper[i] = -lam_i
        # Thomas algorithm for tridiagonal system.
        cp = np.zeros(N + 1, dtype=float)
        dp = np.zeros(N + 1, dtype=float)
        cp[0] = upper[0] / diag[0]
        dp[0] = rhs[0] / diag[0]
        for i in range(1, N + 1):
            den = diag[i] - lower[i] * cp[i - 1]
            if abs(den) < 1e-15:
                den = 1e-15
            cp[i] = upper[i] / den if i < N else 0.0
            dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / den
        V = np.zeros(N + 1, dtype=float)
        V[N] = dp[N]
        for i in range(N - 1, -1, -1):
            V[i] = dp[i] - cp[i] * V[i + 1]
        return V

    def improve_policy(self, V):
        policy_new = np.zeros(self.N + 1, dtype=float)
        # Boundary condition: force a(0)=1.
        policy_new[0] = 1.0
        for i in range(1, self.N + 1):
            Vm = V[i - 1]
            Vp = V[i + 1] if i < self.N else 0.0
            # Vectorized over actions.
            lam_i = self.lam(i, self.actions)
            mu_i = self.mu(i, self.actions)
            costs = self.cost(i, self.actions)
            vals = (costs + lam_i * Vp + mu_i * Vm) / (self.alpha + lam_i + mu_i)
            policy_new[i] = self.actions[np.argmin(vals)]
        return policy_new

    def solve(self, max_iter=200, tol=1e-6, verbose=False):
        p = self.policy.copy()
        p[0] = 1.0
        for it in range(max_iter):
            V = self.evaluate_policy(p)
            p_new = self.improve_policy(V)
            change = np.max(np.abs(p_new - p))
            if verbose:
                print(f"PI iter {it:3d} | policy change sup = {change:.3e}")
            if change < tol:
                p = p_new
                break
            p = p_new
        self.policy = p.copy()
        self.policy[0] = 1.0
        self.V = self.evaluate_policy(self.policy)
        return self.policy.copy()

    def policy_iteration(self, max_iter=200, policy_tol=1e-6, verbose=False):
        policy = self.solve(
            max_iter=max_iter,
            tol=policy_tol,
            verbose=verbose,
        )
        return self.V.copy(), policy.copy()

    def set_params(self, **kwargs):
        old_N = getattr(self, "N", None)
        old_n_actions = getattr(self, "n_actions", None)
        self.p.update(kwargs)
        self.load_common_params(self.p, default_n_actions=101)
        self.load_cost_params()
        N_changed = old_N is None or self.N != old_N
        action_grid_changed = old_n_actions is None or self.n_actions != old_n_actions
        if N_changed:
            self.policy = np.full(self.N + 1, self.a_init, dtype=float)
            self.policy[0] = 1.0
            self.V = np.zeros(self.N + 1, dtype=float)
        else:
            self.policy = np.asarray(self.policy, dtype=float)
            self.policy[0] = 1.0
        if action_grid_changed:
            self.actions = np.linspace(0.0, 1.0, self.n_actions)

    def set_initial_policy(self, policy):
        policy = np.asarray(policy, dtype=float)
        if len(policy) != self.N + 1:
            raise ValueError(f"Policy must have length {self.N + 1}.")
        self.policy = policy.copy()
        self.policy[0] = 1.0


class DormancyCTMDP_Unbounded_Fast(CTMDPTumorBase):
    def __init__(self, params):
        self.load_common_params(params, default_n_actions=201)
        self.load_reward_params()
        self.u = np.zeros(self.N + 1, dtype=float)
        self.policy = np.full(self.N + 1, self.a_init, dtype=float)
        self.policy[0] = 1.0
        self.build_arrays()

    def load_reward_params(self):
        required = ["kappa", "c3"]
        missing = [key for key in required if key not in self.p]
        if missing:
            raise KeyError(f"Missing required reward parameter(s): {missing}")
        self.kappa = float(self.p["kappa"])
        self.c3 = float(self.p["c3"])

    def build_arrays(self):
        # State-action grid
        self.I, self.A = self.build_state_action_grid()
        # State-dependent uniformization rate.
        #
        # From CTMDPTumorBase:
        #   m_i = i * (delta0 + max(r, delta))
        #
        # This is tighter than i*(r + delta + delta0) and still guarantees
        # p_stay >= 0 for every action.
        self.m_i = self.state_uniformization_rates()[:, np.newaxis]
        self.safe_m_i = np.where(self.m_i == 0, 1.0, self.m_i)
        self.denom = self.alpha + self.m_i
        # Transition rates from shared biological model
        self.lam_mat = self.lam_grid(self.I, self.A)
        self.mu_mat = self.mu_grid(self.I, self.A)
        # Uniformized transition probabilities
        self.p_up = self.lam_mat / self.safe_m_i
        self.p_down = self.mu_mat / self.safe_m_i
        self.p_stay = (self.m_i - self.lam_mat - self.mu_mat) / self.safe_m_i
        # Numerical safety check
        if np.min(self.p_stay) < -1e-12:
            raise ValueError(
                "State-dependent uniformization rate is too small; "
                "p_stay became negative."
            )
        # Reward matrix
        #
        # Reward favors population size and penalizes deviation from the
        # formal zero-drift action.
        self.reward = (
            self.c3 * self.I - self.kappa * self.I * (self.A - self.a_star) ** 2
        )
        # Penalize extinction state
        self.reward[0, :] = -self.penalty

    def apply_operator_T(self):
        # Neighbor values
        v_up = np.roll(self.u, -1)
        v_up[self.N] = self.u[self.N]
        v_down = np.roll(self.u, 1)
        v_down[0] = self.u[0]
        # Expected future value over all state-action pairs
        expected_u = (
            self.p_up * v_up[:, np.newaxis]
            + self.p_down * v_down[:, np.newaxis]
            + self.p_stay * self.u[:, np.newaxis]
        )
        # Discounted uniformized Bellman operator
        vals = self.reward / self.denom + (self.m_i / self.denom) * expected_u
        best_idx = np.argmax(vals, axis=1)
        new_u = np.max(vals, axis=1)
        new_policy = self.A[0, best_idx]
        # Boundary condition
        new_policy[0] = 1.0
        return new_u, new_policy

    def solve(self, tol=1e-5, max_iter=5000, verbose=False):
        policy = self.policy.copy()
        policy[0] = 1.0
        for n in range(max_iter):
            old_u = self.u.copy()
            self.u, policy = self.apply_operator_T()
            err = np.max(np.abs(self.u - old_u))
            if verbose and (n % 100 == 0 or err < tol):
                print(f"VI iter {n:5d} | value change sup = {err:.3e}")
            if err < tol:
                break
        self.policy = policy.copy()
        self.policy[0] = 1.0
        return self.u.copy(), self.policy.copy()

    def set_params(self, **kwargs):
        old_N = getattr(self, "N", None)
        self.p.update(kwargs)
        self.load_common_params(self.p, default_n_actions=201)
        self.load_reward_params()
        # Resize value function and policy if N changed.
        if old_N is None or self.N != old_N:
            self.u = np.zeros(self.N + 1, dtype=float)
            self.policy = np.full(self.N + 1, self.a_init, dtype=float)
            self.policy[0] = 1.0
        else:
            self.policy = np.asarray(self.policy, dtype=float)
            self.policy[0] = 1.0
        self.build_arrays()


class TumorActorLP(CTMDPTumorBase):
    def __init__(self, params, beta):
        self.load_common_params(params, default_n_actions=50)
        self.load_lp_params(beta)
        self.update_uniformization_rate()

    def load_lp_params(self, beta):
        required = ["kappa", "c3"]
        missing = [key for key in required if key not in self.p]
        if missing:
            raise KeyError(f"Missing required LP parameter(s): {missing}")
        self.beta = float(beta)
        self.kappa = float(self.p["kappa"])
        self.c3 = float(self.p["c3"])
        if self.beta < 0:
            raise ValueError("beta must be nonnegative.")
        if self.kappa < 0:
            raise ValueError("kappa must be nonnegative.")
        if self.c3 < 0:
            raise ValueError("c3 must be nonnegative.")
        # Default initial state remains i=1, matching older code.
        self.initial_state = int(self.p.get("initial_state", 1))

    def update_uniformization_rate(self):
        self.Lambda = self.global_uniformization_rate()
        self.gamma = self.Lambda / (self.alpha + self.Lambda)
        if self.Lambda <= 0:
            raise ValueError("Uniformization rate Lambda must be positive.")
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError("Uniformized discount factor gamma must lie in [0,1).")

    def reward(self, i, a):
        if i <= 0:
            return -self.penalty
        return self.c3 * i

    def regulatory_cost(self, i, a):
        if i <= 0:
            return 0.0
        return self.kappa * i * (a - self.a_star) ** 2

    def initial_distribution(self):
        if "initial_dist" in self.p:
            rho = np.asarray(self.p["initial_dist"], dtype=float)
            if len(rho) != self.N + 1:
                raise ValueError(f"initial_dist must have length {self.N + 1}.")
            if np.any(~np.isfinite(rho)):
                raise ValueError("initial_dist contains non-finite values.")
            if np.any(rho < 0):
                raise ValueError("initial_dist must be nonnegative.")
            total = np.sum(rho)
            if total <= 0:
                raise ValueError("initial_dist must have positive total mass.")
            return rho / total
        if not (0 <= self.initial_state <= self.N):
            raise ValueError("initial_state must lie between 0 and N.")
        rho = np.zeros(self.N + 1, dtype=float)
        rho[self.initial_state] = 1.0
        return rho

    def solve(self):
        N_s = self.N + 1
        nA = self.n_actions
        n_vars = N_s * nA
        denom = self.alpha + self.Lambda
        obj = np.zeros(n_vars, dtype=float)
        reg = np.zeros(n_vars, dtype=float)
        # Objective and regulatory cost coefficients.
        for i in range(N_s):
            for a_idx, a in enumerate(self.actions):
                v_idx = i * nA + a_idx
                # Minimize negative discounted reward.
                obj[v_idx] = -self.reward(i, a) / denom
                # Discounted regulatory cost.
                reg[v_idx] = self.regulatory_cost(i, a) / denom
        # Sparse flow-balance constraints:
        #
        # sum_a x(j,a) - gamma * sum_{i,a} P(j | i,a) x(i,a) = rho(j)
        #
        # Since the birth-death process only moves to i-1, i, or i+1,
        # each state-action variable contributes to at most three rows.
        A_eq = lil_matrix((N_s, n_vars), dtype=float)
        b_eq = self.initial_distribution()
        for i in range(N_s):
            for a_idx, a in enumerate(self.actions):
                v_idx = i * nA + a_idx
                # Raw biological rates.
                q_up_raw = self.lam(i, a)
                q_dn_raw = self.mu(i, a)
                # Finite-state truncation / reflecting cap.
                # At i=N, attempted births are not allowed to leave the
                # computational state space. Therefore they should not be
                # counted in q_out unless they are routed somewhere.
                # Likewise at i=0, attempted deaths are excluded.
                q_up = q_up_raw if i < self.N else 0.0
                q_dn = q_dn_raw if i > 0 else 0.0
                q_out = q_up + q_dn
                p_stay = 1.0 - q_out / self.Lambda
                if p_stay < -1e-10:
                    raise ValueError(
                        f"Negative stay probability at state {i}, action {a:.4f}: "
                        f"p_stay={p_stay:.3e}. Check uniformization rate."
                    )
                # Numerical cleanup: tiny negative values can occur from roundoff.
                if p_stay < 0.0:
                    p_stay = 0.0
                # Own occupancy term: sum_a x(i,a)
                A_eq[i, v_idx] += 1.0
                # Stay transition i -> i
                A_eq[i, v_idx] -= self.gamma * p_stay
                # Upward transition i -> i+1
                if i < self.N and q_up > 0:
                    A_eq[i + 1, v_idx] -= self.gamma * (q_up / self.Lambda)
                # Downward transition i -> i-1
                if i > 0 and q_dn > 0:
                    A_eq[i - 1, v_idx] -= self.gamma * (q_dn / self.Lambda)
        A_eq = A_eq.tocsr()
        A_ub = csr_matrix(reg.reshape(1, -1))
        bounds = [(0.0, None)] * n_vars
        solver_options = {
            "presolve": True,
            "dual_feasibility_tolerance": 1e-8,
            "primal_feasibility_tolerance": 1e-8,
            "ipm_optimality_tolerance": 1e-8,
        }
        # Try the constrained LP first.
        # Different HiGHS backends sometimes behave differently near
        # degenerate/nonbinding budget constraints, so we try a small sequence.
        constrained_methods = ["highs", "highs-ipm", "highs-ds"]
        first_failure = None
        for method in constrained_methods:
            res = linprog(
                obj,
                A_ub=A_ub,
                b_ub=np.array([self.beta], dtype=float),
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method=method,
                options=solver_options,
            )
            if res.success:
                res.solve_method = method
                res.used_unconstrained_fallback = False
                return res
            if first_failure is None:
                first_failure = res
        # If constrained solves fail at high beta, the budget may already be
        # nonbinding. In that case the unconstrained LP is the correct solution
        # whenever its achieved regulatory cost is <= beta.
        for method in constrained_methods:
            res_free = linprog(
                obj,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method=method,
                options=solver_options,
            )
            if not res_free.success:
                continue
            achieved_cost = float(reg @ res_free.x)
            tol = 1e-7 * max(1.0, abs(self.beta), abs(achieved_cost))
            if achieved_cost <= self.beta + tol:
                res_free.solve_method = method
                res_free.used_unconstrained_fallback = True
                res_free.unconstrained_regulatory_cost = achieved_cost
                return res_free
        return first_failure

    def extract_policy(self, x):
        N_s = self.N + 1
        nA = self.n_actions
        x = np.asarray(x, dtype=float).reshape((N_s, nA))
        # Remove tiny numerical negatives from LP tolerance.
        x = np.maximum(x, 0.0)
        policy = {}
        for i in range(N_s):
            if i == 0:
                policy[i] = [(1.0, 1.0)]
                continue
            total = np.sum(x[i, :])
            if total > 1e-12:
                probs = x[i, :] / total
                active = np.where(probs > 1e-4)[0]
                policy[i] = [
                    (float(self.actions[idx]), float(probs[idx])) for idx in active
                ]
            else:
                # If state is unvisited, default to feasible clipped zero-drift action.
                policy[i] = [(float(self.a_init), 1.0)]
        return policy

    def expected_action_policy(self, x):
        N_s = self.N + 1
        nA = self.n_actions
        x = np.asarray(x, dtype=float).reshape((N_s, nA))
        # Remove tiny numerical negatives from LP tolerance.
        x = np.maximum(x, 0.0)
        a_bar = np.zeros(N_s, dtype=float)
        a_bar[0] = 1.0
        for i in range(1, N_s):
            total = np.sum(x[i, :])
            if total > 1e-20:
                probs = x[i, :] / total
                a_bar[i] = float(np.sum(probs * self.actions))
            else:
                a_bar[i] = float(self.a_init)
        return a_bar

    def occupancy_drift(self, x):
        a_bar = self.expected_action_policy(x)
        return np.array([self.drift(i, a_bar[i]) for i in range(self.N + 1)])

    def discounted_reward_value(self, x):
        N_s = self.N + 1
        nA = self.n_actions
        denom = self.alpha + self.Lambda
        x = np.asarray(x, dtype=float).reshape((N_s, nA))
        x = np.maximum(x, 0.0)
        total = 0.0
        for i in range(N_s):
            for a_idx, a in enumerate(self.actions):
                total += x[i, a_idx] * self.reward(i, a) / denom
        return float(total)

    def discounted_regulatory_cost(self, x):
        N_s = self.N + 1
        nA = self.n_actions
        denom = self.alpha + self.Lambda
        x = np.asarray(x, dtype=float).reshape((N_s, nA))
        x = np.maximum(x, 0.0)
        total = 0.0
        for i in range(N_s):
            for a_idx, a in enumerate(self.actions):
                total += x[i, a_idx] * self.regulatory_cost(i, a) / denom
        return float(total)

    def set_params(self, beta=None, **kwargs):
        if "n_a" in kwargs:
            kwargs["n_actions"] = kwargs.pop("n_a")
        self.p.update(kwargs)
        self.p.pop("n_a", None)
        self.load_common_params(self.p, default_n_actions=self.n_actions)
        if beta is None:
            beta = self.beta
        self.load_lp_params(beta)
        self.update_uniformization_rate()


class TumorActorLP_v2(CTMDPTumorBase):
    def __init__(self, params, beta):
        self.load_common_params(params, default_n_actions=50)
        self.load_lp_params(beta)
        self.update_uniformization_rate()

    def load_lp_params(self, beta):
        required = ["kappa", "c1", "c2", "c3"]
        missing = [key for key in required if key not in self.p]
        if missing:
            raise KeyError(f"Missing required LP parameter(s): {missing}")
        self.beta = float(beta)
        self.kappa = float(self.p["kappa"])
        self.c1 = float(self.p["c1"])
        self.c2 = float(self.p["c2"])
        self.c3 = float(self.p["c3"])
        if self.beta < 0:
            raise ValueError("beta must be nonnegative.")
        if self.kappa < 0:
            raise ValueError("kappa must be nonnegative.")
        if self.c1 < 0:
            raise ValueError("c1 must be nonnegative.")
        if self.c2 < 0:
            raise ValueError("c2 must be nonnegative.")
        if self.c3 < 0:
            raise ValueError("c3 must be nonnegative.")
        # Default initial state remains i=1, matching the older code.
        self.initial_state = int(self.p.get("initial_state", 1))

    def update_uniformization_rate(self):
        self.Lambda = self.global_uniformization_rate()
        self.gamma = self.Lambda / (self.alpha + self.Lambda)
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError("Uniformized discount factor gamma must lie in [0,1).")

    def preferred_size(self, a=None):
        if a is None:
            a = self.a_star
        denom = self.c1 + self.c2 * self.r * float(a)
        if denom <= 0:
            return np.inf
        return self.c3 / (2.0 * denom)

    def reward(self, i, a):
        if i <= 0:
            # Tumor-agent interpretation: extinction is terminal failure.
            return -self.penalty
        return self.c3 * i - self.c1 * i**2 - self.c2 * self.r * a * i**2

    def regulatory_cost(self, i, a):
        if i <= 0:
            return 0.0
        return self.kappa * i * (a - self.a_star) ** 2

    def initial_distribution(self):
        if "initial_dist" in self.p:
            rho = np.asarray(self.p["initial_dist"], dtype=float)
            if len(rho) != self.N + 1:
                raise ValueError(f"initial_dist must have length {self.N + 1}.")
            if np.any(~np.isfinite(rho)):
                raise ValueError("initial_dist contains non-finite values.")
            if np.any(rho < 0):
                raise ValueError("initial_dist must be nonnegative.")
            total = np.sum(rho)
            if total <= 0:
                raise ValueError("initial_dist must have positive total mass.")
            return rho / total
        if not (0 <= self.initial_state <= self.N):
            raise ValueError("initial_state must lie between 0 and N.")
        rho = np.zeros(self.N + 1, dtype=float)
        rho[self.initial_state] = 1.0
        return rho

    def solve(self):
        N_s = self.N + 1
        nA = self.n_actions
        n_vars = N_s * nA
        denom = self.alpha + self.Lambda
        obj = np.zeros(n_vars, dtype=float)
        reg = np.zeros(n_vars, dtype=float)
        for i in range(N_s):
            for a_idx, a in enumerate(self.actions):
                v_idx = i * nA + a_idx
                # Minimize negative discounted reward.
                obj[v_idx] = -self.reward(i, a) / denom
                # Discounted regulatory cost.
                reg[v_idx] = self.regulatory_cost(i, a) / denom
        # Sparse flow-balance constraints:
        #
        # sum_a x(j,a) - gamma * sum_{i,a} P(j | i,a) x(i,a) = rho(j)
        A_eq = lil_matrix((N_s, n_vars), dtype=float)
        b_eq = self.initial_distribution()
        for i in range(N_s):
            for a_idx, a in enumerate(self.actions):
                v_idx = i * nA + a_idx
                q_up = self.lam(i, a)
                q_dn = self.mu(i, a)
                q_out = q_up + q_dn
                p_stay = 1.0 - q_out / self.Lambda
                if p_stay < -1e-10:
                    raise ValueError(
                        f"Negative stay probability at state {i}, action {a:.4f}: "
                        f"p_stay={p_stay:.3e}. Check uniformization rate."
                    )
                # Own occupancy term.
                A_eq[i, v_idx] += 1.0
                # Stay transition i -> i.
                A_eq[i, v_idx] -= self.gamma * p_stay
                # Upward transition i -> i+1.
                if i < self.N and q_up > 0:
                    A_eq[i + 1, v_idx] -= self.gamma * (q_up / self.Lambda)
                # Downward transition i -> i-1.
                if i > 0 and q_dn > 0:
                    A_eq[i - 1, v_idx] -= self.gamma * (q_dn / self.Lambda)
        A_eq = A_eq.tocsr()
        A_ub = csr_matrix(reg.reshape(1, -1))
        bounds = [(0.0, None)] * n_vars
        solver_options = {
            "presolve": True,
            "dual_feasibility_tolerance": 1e-8,
            "primal_feasibility_tolerance": 1e-8,
            "ipm_optimality_tolerance": 1e-8,
        }
        constrained_methods = ["highs", "highs-ipm", "highs-ds"]
        first_failure = None
        # First try the constrained LP.
        for method in constrained_methods:
            res = linprog(
                obj,
                A_ub=A_ub,
                b_ub=np.array([self.beta], dtype=float),
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method=method,
                options=solver_options,
            )
            if res.success:
                res.solve_method = method
                res.used_unconstrained_fallback = False
                return res
            if first_failure is None:
                first_failure = res
        # If the constrained problem fails at high beta, the beta constraint
        # may be nonbinding. Try the unconstrained LP and accept it only if
        # its regulatory cost satisfies the beta budget.
        for method in constrained_methods:
            res_free = linprog(
                obj,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method=method,
                options=solver_options,
            )
            if not res_free.success:
                continue
            achieved_cost = float(reg @ res_free.x)
            tol = 1e-7 * max(1.0, abs(self.beta), abs(achieved_cost))
            if achieved_cost <= self.beta + tol:
                res_free.solve_method = method
                res_free.used_unconstrained_fallback = True
                res_free.unconstrained_regulatory_cost = achieved_cost
                return res_free
        return first_failure

    def extract_policy(self, x):
        N_s = self.N + 1
        nA = self.n_actions
        x = np.asarray(x, dtype=float).reshape((N_s, nA))
        # Remove tiny numerical negatives from LP tolerance.
        x = np.maximum(x, 0.0)
        policy = {}
        for i in range(N_s):
            if i == 0:
                policy[i] = [(1.0, 1.0)]
                continue
            total = np.sum(x[i, :])
            if total > 1e-9:
                probs = x[i, :] / total
                active = np.where(probs > 1e-4)[0]
                policy[i] = [
                    (float(self.actions[idx]), float(probs[idx])) for idx in active
                ]
            else:
                # If state is unvisited, default to feasible clipped zero-drift action.
                policy[i] = [(float(self.a_init), 1.0)]
        return policy

    def expected_action_policy(self, x):
        N_s = self.N + 1
        nA = self.n_actions
        x = np.asarray(x, dtype=float).reshape((N_s, nA))
        # Remove tiny numerical negatives from LP tolerance.
        x = np.maximum(x, 0.0)
        a_bar = np.zeros(N_s, dtype=float)
        a_bar[0] = 1.0
        for i in range(1, N_s):
            total = np.sum(x[i, :])
            if total > 1e-10:
                probs = x[i, :] / total
                a_bar[i] = float(np.sum(probs * self.actions))
            else:
                a_bar[i] = float(self.a_init)
        return a_bar

    def occupancy_drift(self, x):
        a_bar = self.expected_action_policy(x)
        # Prefer biological_drift if you added it to the base class.
        if hasattr(self, "biological_drift"):
            return np.array(
                [self.biological_drift(i, a_bar[i]) for i in range(self.N + 1)]
            )
        # Fallback for the original base class, where drift() is biological drift.
        return np.array([self.drift(i, a_bar[i]) for i in range(self.N + 1)])

    def discounted_reward_value(self, x):
        N_s = self.N + 1
        nA = self.n_actions
        denom = self.alpha + self.Lambda
        x = np.asarray(x, dtype=float).reshape((N_s, nA))
        x = np.maximum(x, 0.0)
        total = 0.0
        for i in range(N_s):
            for a_idx, a in enumerate(self.actions):
                total += x[i, a_idx] * self.reward(i, a) / denom
        return float(total)

    def discounted_regulatory_cost(self, x):
        N_s = self.N + 1
        nA = self.n_actions
        denom = self.alpha + self.Lambda
        x = np.asarray(x, dtype=float).reshape((N_s, nA))
        x = np.maximum(x, 0.0)
        total = 0.0
        for i in range(N_s):
            for a_idx, a in enumerate(self.actions):
                total += x[i, a_idx] * self.regulatory_cost(i, a) / denom
        return float(total)

    def set_params(self, beta=None, **kwargs):
        if "n_a" in kwargs:
            kwargs["n_actions"] = kwargs.pop("n_a")
        self.p.update(kwargs)
        self.p.pop("n_a", None)
        self.load_common_params(self.p, default_n_actions=self.n_actions)
        if beta is None:
            beta = self.beta
        self.load_lp_params(beta)
        self.update_uniformization_rate()
