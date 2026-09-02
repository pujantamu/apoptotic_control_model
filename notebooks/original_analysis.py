# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Apoptotic control figure code
#
# Run the model definitions first, followed by the section for the figure of
# interest. The constrained LP and convergence sections take longer to run.

# %%
# %matplotlib inline
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

param_name_dict = {
    "c1": r"c_1",
    "c2": r"c_2",
    "c3": r"c_3",
    "kappa1": r"\kappa_1",
    "kappa2": r"\kappa_2",
    "alpha": r"\alpha",
    "r": r"r",
    "delta": r"\delta",
    "delta0": r"\delta_0",
    "L": r"L",
    "N": r"N",
    "kappa": r"\kappa",
}


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


# Small helpers
def get_params(section: str) -> dict:
    if section not in PARAMS:
        raise KeyError(f"Unknown section '{section}'.")
    return {**PARAMS["shared"], **PARAMS[section]}


def policy_to_array(policy_like, N: int, a_star: float | None = None) -> np.ndarray:
    if isinstance(policy_like, np.ndarray):
        out = policy_like.astype(float).copy()
        if out.shape[0] != N + 1:
            raise ValueError(f"Policy length {out.shape[0]} != N+1 = {N+1}")
        return np.clip(out, 0.0, 1.0)

    default_a = 0.5 if a_star is None else float(np.clip(a_star, 0.0, 1.0))
    out = np.full(N + 1, default_a, dtype=float)

    if isinstance(policy_like, dict):
        for i in range(N + 1):
            entries = policy_like.get(i, None)
            if entries is None:
                continue
            out[i] = sum(float(a) * float(p) for a, p in entries)
        return np.clip(out, 0.0, 1.0)

    raise TypeError("Unsupported policy format.")


def zero_drift_action(r: float, delta: float, delta0: float = 0.0) -> float:
    return (delta + delta0) / (r + delta)


def birth_rate(i: int, a: float, r: float, N: int) -> float:
    return 0.0 if i >= N else r * i * a


def death_rate(i: int, a: float, delta: float, delta0: float = 0.0) -> float:
    return 0.0 if i <= 0 else delta * i * (1.0 - a) + delta0 * i


def get_n_actions(model):
    return int(getattr(model, "n_actions", getattr(model, "n_a", len(model.actions))))


def occupancy_support_mask(
    model,
    res,
    rel_tol=1e-8,
    abs_tol=1e-12,
):

    nA = get_n_actions(model)

    x = np.asarray(res.x, dtype=float).reshape(model.N + 1, nA)

    # Numerical LP cleanup
    x = np.maximum(x, 0.0)

    occupancy = np.sum(x, axis=1)

    max_occ = np.max(occupancy)

    tol = max(abs_tol, rel_tol * max_occ)

    support = occupancy > tol

    # Extinction is not used in per-capita policy/drift plots
    support[0] = False

    return occupancy, support, tol


def mask_constrained_solution(
    model,
    res,
    a_exp,
    drift,
    rel_tol=1e-8,
    abs_tol=1e-12,
):

    occupancy, support, tol = occupancy_support_mask(
        model,
        res,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )

    a_masked = np.asarray(a_exp, dtype=float).copy()

    drift_masked = np.asarray(drift, dtype=float).copy()

    a_masked[~support] = np.nan
    drift_masked[~support] = np.nan

    return a_masked, drift_masked, occupancy, support, tol


def normalized_drift(
    policy: np.ndarray,
    r: float,
    delta: float,
    delta0: float,
    N: int,
) -> np.ndarray:
    g = np.full(N + 1, np.nan, dtype=float)

    for i in range(1, N + 1):
        lam = birth_rate(i, policy[i], r, N)
        mu = death_rate(i, policy[i], delta, delta0)
        g[i] = (lam - mu) / i

    return g


def near_zero_drift_width(g: np.ndarray, eps: float = 0.02) -> int:
    mask = np.isfinite(g) & (np.abs(g) <= eps)
    return int(np.sum(mask))


def near_zero_drift_width_range(g, *, eps=0.02, i_min=1, i_max=100):
    g = np.asarray(g, dtype=float)

    lo = max(int(i_min), 1)
    hi = min(int(i_max), len(g) - 1)

    if hi < lo:
        return 0

    idx = np.arange(lo, hi + 1)

    mask = np.isfinite(g[idx]) & (np.abs(g[idx]) <= eps)

    return int(np.sum(mask))


def simulate_bd(
    policy: np.ndarray,
    *,
    r: float,
    delta: float,
    delta0: float = 0.0,
    N: int,
    i0: int = 10,
    T: float = 80.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(123)

    t = 0.0
    i = int(i0)

    ts = [0.0]
    xs = [i]

    while t < T:
        a = float(policy[i])

        lam = birth_rate(i, a, r, N)
        mu = death_rate(i, a, delta, delta0)

        rate = lam + mu

        if rate <= 1e-14:
            break

        dt = rng.exponential(1.0 / rate)
        t += dt

        if t > T:
            break

        if rng.random() < lam / rate:
            i = min(i + 1, N)
        else:
            i = max(i - 1, 0)

        ts.append(t)
        xs.append(i)

        if i == 0:
            break

    return np.array(ts), np.array(xs)


def sample_path_on_grid(
    ts: np.ndarray, xs: np.ndarray, t_grid: np.ndarray
) -> np.ndarray:
    idx = np.searchsorted(ts, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(xs) - 1)
    return xs[idx]


def simulate_many_on_grid(
    policy: np.ndarray,
    *,
    r: float,
    delta: float,
    delta0: float = 0.0,
    N: int,
    i0: int = 10,
    T: float = 80.0,
    n_sims: int = 200,
    n_time: int = 401,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t_grid = np.linspace(0.0, T, n_time)
    X = np.zeros((n_sims, n_time), dtype=float)

    for k in range(n_sims):
        ts, xs = simulate_bd(
            policy,
            r=r,
            delta=delta,
            delta0=delta0,
            N=N,
            i0=i0,
            T=T,
            rng=rng,
        )

        X[k, :] = sample_path_on_grid(ts, xs, t_grid)

    return t_grid, X


def summarize_paths(X: np.ndarray) -> dict:
    return {
        "q25": np.quantile(X, 0.25, axis=0),
        "median": np.quantile(X, 0.50, axis=0),
        "q75": np.quantile(X, 0.75, axis=0),
        "mean": np.mean(X, axis=0),
    }


def package_result(name: str, model, policy, value, **extra) -> dict:
    policy = policy_to_array(policy, model.N, model.a_star)

    return {
        "name": name,
        "model": model,
        "policy": policy,
        "value": value,
        "a_star": model.a_star,
        "a_init": model.a_init,
        "zero_drift_feasible": model.zero_drift_feasible(),
        "r": model.r,
        "delta": model.delta,
        "delta0": model.delta0,
        "N": model.N,
        **extra,
    }


def solve_dormancy() -> dict:
    p = get_params("dormancy")

    # New params-dictionary interface
    model = DormancyCTMDP_DiscountedPI(p)

    V, policy = model.policy_iteration()

    return package_result(
        "Threshold",
        model,
        policy,
        V,
    )


def solve_full() -> dict:
    p = get_params("full")

    # New params-dictionary interface
    model = DormancyCTMDP_DiscountedPI_V2(p)

    policy = model.solve()
    V = model.V.copy()

    return package_result(
        "Quadratic",
        model,
        policy,
        V,
    )


def solve_unbounded_unified(params=None) -> dict:
    if params is None:
        p = get_params("unbounded")
    else:
        p = dict(params)

    model = DormancyCTMDP_Unbounded_Fast(p)

    V, policy = model.solve()

    return package_result(
        "Unbounded",
        model,
        policy,
        V,
    )


def solve_constrained_unified() -> dict:
    p = get_params("constrained")

    beta = float(p["beta"])

    model = TumorActorLP(p, beta=beta)

    res = model.solve()

    if not res.success:
        raise RuntimeError(f"LP solver failed: {res.message}")

    # Raw finite expected policy:
    # keep this for stochastic simulation.
    policy_raw = model.expected_action_policy(res.x)

    # Raw biological drift
    i = np.arange(model.N + 1)

    drift_raw = np.full(model.N + 1, np.nan, dtype=float)

    for k in range(1, model.N + 1):
        drift_raw[k] = model.drift(k, policy_raw[k])

    # Occupancy-supported interpretation
    (
        policy_masked,
        drift_masked,
        occupancy,
        support,
        occ_tol,
    ) = mask_constrained_solution(
        model,
        res,
        policy_raw,
        drift_raw,
        rel_tol=1e-8,
        abs_tol=1e-12,
    )

    return package_result(
        "Constrained",
        model,
        # This is what Panels A/B/D should interpret.
        policy_masked,
        occupancy,
        # Raw finite policy retained for Panel C.
        policy_raw=policy_raw,
        drift_masked=drift_masked,
        occupancy=occupancy,
        support=support,
        occupancy_tol=occ_tol,
        raw_lp_result=res,
        randomized_policy=model.extract_policy(res.x),
    )


def solve_constrained_quad_unified() -> dict:

    p = get_params("constrained_quad")

    beta = float(p["beta"])

    model = TumorActorLP_v2(p, beta=beta)

    res = model.solve()

    if not res.success:
        raise RuntimeError("Quadratic constrained LP solver failed: " f"{res.message}")

    # Raw finite expected policy:
    # keep this for stochastic simulation.
    policy_raw = model.expected_action_policy(res.x)

    # Raw biological drift
    drift_raw = np.full(model.N + 1, np.nan, dtype=float)

    for k in range(1, model.N + 1):
        drift_raw[k] = model.drift(k, policy_raw[k])

    # Occupancy-supported interpretation
    (
        policy_masked,
        drift_masked,
        occupancy,
        support,
        occ_tol,
    ) = mask_constrained_solution(
        model,
        res,
        policy_raw,
        drift_raw,
        rel_tol=1e-8,
        abs_tol=1e-12,
    )

    return package_result(
        "Constrained quadratic",
        model,
        # Panels A/B/D use this.
        policy_masked,
        occupancy,
        # Panel C uses this.
        policy_raw=policy_raw,
        drift_masked=drift_masked,
        occupancy=occupancy,
        support=support,
        occupancy_tol=occ_tol,
        raw_lp_result=res,
        randomized_policy=model.extract_policy(res.x),
        discounted_reward=model.discounted_reward_value(res.x),
        discounted_regulatory_cost=model.discounted_regulatory_cost(res.x),
        preferred_size_at_astar=model.preferred_size(model.a_star),
    )


def solve_astar() -> dict:
    p = PARAMS["shared"]

    N = int(p["N"])
    r = float(p["r"])
    delta = float(p["delta"])
    delta0 = float(p.get("delta0", 0.0))

    a_star = zero_drift_action(r, delta, delta0)
    a_init = float(np.clip(a_star, 0.0, 1.0))

    policy = np.full(N + 1, a_init, dtype=float)
    policy[0] = 1.0

    V = np.zeros(N + 1, dtype=float)

    return {
        "name": r"Baseline $a^*$",
        "model": None,
        "policy": policy,
        "value": V,
        "a_star": a_star,
        "a_init": a_init,
        "zero_drift_feasible": 0.0 <= a_star <= 1.0,
        "r": r,
        "delta": delta,
        "delta0": delta0,
        "N": N,
    }


def solve_all_models() -> dict:
    return {
        "dormancy": solve_dormancy(),
        "full": solve_full(),
        "unbounded": solve_unbounded_unified(),
        "constrained": solve_constrained_unified(),
        "constrained_quad": solve_constrained_quad_unified(),
        "astar": solve_astar(),
    }


def plot_sample_paths(
    ax,
    results: dict,
    *,
    order=(
        "dormancy",
        "full",
        "unbounded",
        "constrained",
        "constrained_quad",
        "astar",
    ),
    n_paths_each=4,
    i0=1,
    T=80.0,
    seed=123,
):
    rng = np.random.default_rng(seed)

    for key in order:
        R = results[key]
        policy_sim = R.get("policy_raw", R["policy"])
        for rep in range(n_paths_each):
            ts, xs = simulate_bd(
                policy_sim,
                r=R["r"],
                delta=R["delta"],
                delta0=R["delta0"],
                N=R["N"],
                i0=i0,
                T=T,
                rng=rng,
            )

            ax.step(
                ts,
                xs,
                where="post",
                alpha=0.5,
                label=R["name"] if rep == 0 else None,
            )

    ax.set_title("Sample trajectories")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population size")
    ax.legend()


# Unified comparison figure
def add_panel_label(ax, label, x=-0.1, y=1.03, fontsize=24):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def make_unifying_figure(
    results: dict,
    *,
    eps: float = 0.02,
    i0: int = 10,
    T: float = 80.0,
    seed: int = 123,
    figsize: tuple[float, float] = (13, 9),
    panel_c_mode: str = "summary",
    panel_c_n_sims: int = 200,
    panel_c_n_time: int = 401,
    panel_c_n_paths_each: int = 4,
):

    # Font sizes
    title_fs = 22
    label_fs = 15
    tick_fs = 12
    legend_fs = 12
    panel_fs = 22

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    order = [
        "astar",
        "dormancy",
        "full",
        "unbounded",
        "constrained",
        "constrained_quad",
    ]

    colors = {
        "dormancy": None,
        "full": None,
        "unbounded": None,
        "constrained": None,
        "constrained_quad": None,
        "astar": "gray",
    }

    N_plot = results["dormancy"]["N"]
    xlim_state = (1, N_plot - 1)
    labels = []
    widths = []

    # Panel A: optimal policy
    ax = axs[0, 0]

    for key in order:
        R = results[key]
        i = np.arange(R["N"] + 1)

        if key == "astar":
            ax.plot(
                i,
                R["policy"],
                linestyle=":",
                linewidth=2.5,
                color=colors[key],
                label=R["name"],
            )
        else:
            ax.plot(
                i,
                R["policy"],
                color=colors[key],
                label=R["name"] + " cost",
            )

    r = results["dormancy"]["r"]
    delta = results["dormancy"]["delta"]
    delta0 = results["dormancy"]["delta0"]
    a_star = zero_drift_action(r, delta, delta0)

    ax.set_title("Optimal policy", fontsize=title_fs, fontweight="bold", pad=12)
    add_panel_label(ax, "A.", fontsize=panel_fs)
    ax.set_xlabel("State $i$", fontsize=label_fs)
    ax.set_ylabel(r"Optimal Action $\pi^*(i)$", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xlim_state)
    ax.legend(fontsize=legend_fs)

    # Panel B: normalized drift
    ax = axs[0, 1]

    for key in order:
        R = results[key]

        g = normalized_drift(
            R["policy"],
            R["r"],
            R["delta"],
            R["delta0"],
            R["N"],
        )

        i = np.arange(R["N"] + 1)

        label = R["name"] if key == "astar" else R["name"] + " cost"

        ax.plot(
            i[1:],
            g[1:],
            color=colors[key],
            label=label,
        )

        labels.append(R["name"])
        widths.append(
            near_zero_drift_width_range(
                g,
                eps=eps,
                i_min=1,
                i_max=N_plot - 1,
            )
        )

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlim(xlim_state)
    ax.grid(True, alpha=0.3)
    ax.set_title("Normalized drift", fontsize=title_fs, fontweight="bold", pad=12)
    add_panel_label(ax, "B.", fontsize=panel_fs)
    ax.set_xlabel("State $i$", fontsize=label_fs)
    ax.set_ylabel(r"$(\lambda-\mu)/i$", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.legend(fontsize=legend_fs)

    # Panel C: trajectory evidence
    ax = axs[1, 0]

    if panel_c_mode == "sample_paths":
        rng = np.random.default_rng(seed)

        for key in order:
            R = results[key]

            for rep in range(panel_c_n_paths_each):
                policy_sim = R.get("policy_raw", R["policy"])

                ts, xs = simulate_bd(
                    policy_sim,
                    r=R["r"],
                    delta=R["delta"],
                    delta0=R["delta0"],
                    N=R["N"],
                    i0=i0,
                    T=T,
                    rng=rng,
                )

                ax.step(
                    ts,
                    xs,
                    where="post",
                    alpha=0.6,
                    label=R["name"] + " cost" if rep == 0 else None,
                )

        ax.set_title(
            "Representative stochastic trajectories",
            fontsize=title_fs,
            fontweight="bold",
            pad=12,
        )
        add_panel_label(ax, "C.", fontsize=panel_fs)

    elif panel_c_mode == "summary":
        for j, key in enumerate(order):
            R = results[key]

            policy_sim = R.get("policy_raw", R["policy"])

            t_grid, X = simulate_many_on_grid(
                policy_sim,
                r=R["r"],
                delta=R["delta"],
                delta0=R["delta0"],
                N=R["N"],
                i0=i0,
                T=T,
                n_sims=panel_c_n_sims,
                n_time=panel_c_n_time,
                seed=seed + 1000 * j,
            )

            S = summarize_paths(X)

            label = R["name"] if key == "astar" else R["name"] + " cost"

            (line,) = ax.plot(
                t_grid,
                S["median"],
                color=colors[key],
                label=label,
            )

            color = line.get_color()

            ax.fill_between(
                t_grid,
                S["q25"],
                S["q75"],
                alpha=0.2,
                color=color,
            )

        ax.set_title(
            "Stochastic trajectories",
            fontsize=title_fs,
            fontweight="bold",
            pad=12,
        )
        add_panel_label(ax, "C.", fontsize=panel_fs)

    else:
        raise ValueError("panel_c_mode must be 'summary' or 'sample_paths'")

    ax.set_ylim(0, 100)
    ax.set_xlabel("Time", fontsize=label_fs)
    ax.set_ylabel("Population size", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=legend_fs)

    # Panel D: width of near-zero-drift region
    ax = axs[1, 1]

    ax.bar(
        labels,
        widths,
        edgecolor="black",
        linewidth=1.5,
    )

    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(
        r"Near-zero-drift region for $i < 100$",
        fontsize=title_fs,
        fontweight="bold",
        pad=12,
    )
    add_panel_label(ax, "D.", fontsize=panel_fs)
    ax.set_ylabel(
        r"$\#\{1\leq i < 100:\ |(\lambda-\mu)/i|< \varepsilon\}$",
        fontsize=label_fs,
    )
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()

    return fig, axs


def print_summary(results: dict, eps: float = 0.02) -> None:
    print("=== Unified regime summary ===")

    for key, R in results.items():
        g = normalized_drift(
            R["policy"],
            R["r"],
            R["delta"],
            R["delta0"],
            R["N"],
        )

        width = near_zero_drift_width(g, eps=eps)

        zero_cross = None
        finite_idx = np.where(np.isfinite(g))[0]

        for i in finite_idx[1:]:
            if np.sign(g[i - 1]) != np.sign(g[i]):
                zero_cross = i
                break

        print(
            f"{R['name']:22s} | "
            f"a*(0+)~{R['policy'][1]:.3f} | "
            f"a*(N)~{R['policy'][-1]:.3f} | "
            f"a_star={R['a_star']:.3f} | "
            f"feasible={str(R['zero_drift_feasible']):5s} | "
            f"near-zero width={width:3d} | "
            f"first zero-cross={zero_cross}"
        )


PARAMS = {
    "shared": {
        "N": 100,
        "r": 0.2,
        "delta": 0.25,
        "delta0": 0.05,  # extrinsic death pressure; set >0 for new model
        "alpha": 0.51,
        "penalty": 10e4,
    },
    # Section 1: Dormancy via Thresholding
    "dormancy": {
        "L": 35,
        "kappa1": 1,
        "kappa2": 0.5,
        "n_actions": 101,
    },
    # Section 2: Full discounted model
    "full": {
        "c1": 0.005,
        "c2": 0.02,
        "kappa": 1.2,
        "c3": 1,
        "n_actions": 101,
    },
    # Section 3: Unbounded Linear Costs
    "unbounded": {
        "N": 800,
        "kappa": 1,
        "c3": 0.8,
        "n_actions": 201,
    },
    # Section 4: Constrained Linear Costs
    "constrained": {
        "N": 200,
        "kappa": 1.2,
        "c3": 1.0,
        "beta": 0.35,
        "n_a": 101,
        "initial_state": 10,
    },
    # Section 5: Constrained quadratic reward model
    "constrained_quad": {
        "N": 200,
        "kappa": 1.3,
        "c1": 0.01,
        "c2": 0,
        "c3": 0.75,
        "beta": 0.37,
        "n_a": 101,
        "initial_state": 10,
    },
}


def run_all_and_plot(
    *,
    panel_c_mode="summary",
    panel_c_n_sims=200,
    panel_c_n_time=401,
    panel_c_n_paths_each=4,
):
    results = solve_all_models()
    print_summary(results)

    fig, axs = make_unifying_figure(
        results,
        panel_c_mode=panel_c_mode,
        panel_c_n_sims=panel_c_n_sims,
        panel_c_n_time=panel_c_n_time,
        panel_c_n_paths_each=panel_c_n_paths_each,
    )

    plt.show()

    return results, fig, axs


results, fig, axs = run_all_and_plot(
    panel_c_mode="summary",
    panel_c_n_sims=1000,
    panel_c_n_time=501,
)
# %% [markdown]
# ## Threshold model

# %%
# Helper functions


def solve_model(params, verbose=False):
    model = DormancyCTMDP_DiscountedPI(params)
    V, policy = model.policy_iteration(verbose=verbose)
    return model, V, policy


def per_capita_drift(model, policy):
    i = np.arange(model.N + 1)
    drift = np.full(model.N + 1, np.nan)

    for k in range(1, model.N + 1):
        a = float(policy[k])
        drift[k] = (model.lam(k, a) - model.mu(k, a)) / k

    return i, drift


def dormant_width(model, policy, eps=0.05):
    i, drift = per_capita_drift(model, policy)
    mask = (i > 0) & np.isfinite(drift)
    return int(np.sum(np.abs(drift[mask]) <= eps))


def state_axis(model):
    return np.arange(1, model.N)


def add_grid(ax):
    ax.grid(True, alpha=0.3)


def symmetric_drift_limits(drift_mat):
    vmax = np.nanmax(np.abs(drift_mat))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def plot_policy_curve(params):
    model, V, policy = solve_model(params)

    i = state_axis(model)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(i, policy[1 : model.N], lw=2)

    # Formal zero-drift action may be outside [0,1]; plot clipped reference.
    #     model.a_init,
    #     ls="--",
    #     color="k",
    #     label=r"clipped $a_0^*$",
    # )

    ax.axvline(
        model.L,
        ls=":",
        color="k",
        label=r"$L$",
    )

    ax.set_xlim(1, model.N - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Optimal action $a^*(i)$")
    ax.set_title("Threshold objective: optimal policy", fontweight="bold")
    add_grid(ax)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_drift_curve(params):
    model, V, policy = solve_model(params)
    i, drift = per_capita_drift(model, policy)

    x = np.arange(1, model.N)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(x, drift[1 : model.N], lw=2)

    ax.axhline(0.0, ls="--", color="k")
    ax.axvline(model.L, ls=":", color="k", label=r"$L$")

    ax.set_xlim(1, model.N - 1)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Per-capita drift $(\lambda-\mu)/i$")
    ax.set_title("Threshold objective: induced net drift", fontweight="bold")
    add_grid(ax)
    ax.legend()

    plt.tight_layout()
    plt.show()


def sweep_parameter(params, param_name, param_values):
    param_values = np.asarray(param_values, dtype=float)

    policies = []
    drifts = []
    widths = []

    for val in param_values:
        p = params.copy()
        p[param_name] = float(val)

        model, V, policy = solve_model(p)
        i, drift = per_capita_drift(model, policy)

        policies.append(policy[1 : model.N])
        drifts.append(drift[1 : model.N])
        widths.append(dormant_width(model, policy))

    return {
        "param_name": param_name,
        "param_values": param_values,
        "policies": np.asarray(policies),
        "drifts": np.asarray(drifts),
        "widths": np.asarray(widths),
    }


def plot_kappa2_heatmap(params, kappa2_values):
    sweep = sweep_parameter(params, "kappa2", kappa2_values)

    model, _, _ = solve_model(params)
    i = state_axis(model)

    # Font sizes
    title_fs = 20
    label_fs = 17
    tick_fs = 14
    cbar_label_fs = 16
    cbar_tick_fs = 13

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        sweep["policies"],
        aspect="auto",
        origin="lower",
        extent=[
            0,
            100,
            sweep["param_values"].min(),
            sweep["param_values"].max(),
        ],
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"Optimal action $\pi^*(i)$", fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    ax.set_xlim(1, 100)
    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_ylabel(r"Threshold Penalty $\kappa_2$", fontsize=label_fs)
    ax.set_title(
        r"Threshold Model: Optimal Policy",
        fontsize=title_fs,
        fontweight="bold",
        pad=10,
    )

    ax.tick_params(axis="both", labelsize=tick_fs)

    plt.tight_layout()
    plt.show()


def plot_kappa2_drift_heatmap(params, kappa2_values):
    sweep = sweep_parameter(params, "kappa2", kappa2_values)

    model, _, _ = solve_model(params)
    i = state_axis(model)

    vmin, vmax = symmetric_drift_limits(sweep["drifts"])

    # Font sizes
    title_fs = 20
    label_fs = 17
    tick_fs = 14
    cbar_label_fs = 16
    cbar_tick_fs = 13

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        sweep["drifts"],
        aspect="auto",
        origin="lower",
        extent=[
            1,
            100,
            sweep["param_values"].min(),
            sweep["param_values"].max(),
        ],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        r"Per-capita drift $(\lambda-\mu)/i$",
        fontsize=cbar_label_fs,
    )
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    # Zero-drift contour
    #     i,
    #     sweep["param_values"],
    #     sweep["drifts"],
    #     levels=[0.0],
    #     colors="k",
    #     linewidths=1.2,
    # )

    ax.set_xlim(1, 100)
    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_ylabel(r"Threshold Penalty $\kappa_2$", fontsize=label_fs)
    ax.set_title(
        "Threshold Model: Normalized Drift",
        fontsize=title_fs,
        fontweight="bold",
        pad=10,
    )

    ax.tick_params(axis="both", labelsize=tick_fs)

    plt.tight_layout()
    plt.show()


def plot_L_heatmap(params, L_values):
    sweep = sweep_parameter(params, "L", L_values)

    model, _, _ = solve_model(params)
    i = state_axis(model)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        sweep["policies"],
        aspect="auto",
        origin="lower",
        extent=[
            i.min(),
            i.max(),
            sweep["param_values"].min(),
            sweep["param_values"].max(),
        ],
    )

    plt.colorbar(im, ax=ax, label=r"Optimal action $a^*(i)$")

    # Since L varies, recognition threshold is x = L.
    ax.plot(
        sweep["param_values"] / 2,
        sweep["param_values"],
        ls=":",
        color="k",
        lw=2,
        label=r"$L$",
    )

    ax.set_xlim(1, model.N - 1)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"$L$")
    ax.set_title(r"Sweep over recognition scale $L$", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_L_drift_heatmap(params, L_values):
    sweep = sweep_parameter(params, "L", L_values)

    model, _, _ = solve_model(params)
    i = state_axis(model)

    vmin, vmax = symmetric_drift_limits(sweep["drifts"])

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        sweep["drifts"],
        aspect="auto",
        origin="lower",
        extent=[
            i.min(),
            i.max(),
            sweep["param_values"].min(),
            sweep["param_values"].max(),
        ],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, label=r"Per-capita drift $(\lambda-\mu)/i$")

    # Zero-drift contour
    ax.contour(
        i,
        sweep["param_values"],
        sweep["drifts"],
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )

    # Since L varies, recognition threshold is x = L.
    ax.plot(
        sweep["param_values"] / 2,
        sweep["param_values"],
        ls=":",
        color="k",
        lw=2,
        label=r"$L$",
    )

    ax.set_xlim(1, model.N - 1)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"$L$")
    ax.set_title(r"Drift sweep over recognition scale $L$", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_dormant_width_vs_kappa2(params, kappa2_values, eps=0.05):
    widths = []

    for k2 in kappa2_values:
        p = params.copy()
        p["kappa2"] = float(k2)

        model, V, policy = solve_model(p)
        widths.append(dormant_width(model, policy, eps=eps))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(kappa2_values, widths, marker="o", lw=2)

    ax.set_xlabel(r"$\kappa_2$")
    ax.set_ylabel(r"Number of states with $|(\lambda-\mu)/i|\leq \varepsilon$")
    ax.set_title(rf"Dormant interval width, $\varepsilon={eps}$", fontweight="bold")
    add_grid(ax)

    plt.tight_layout()
    plt.show()


def simulate_birth_death(model, policy, i0=20, T=200, seed=None):
    rng = np.random.default_rng(seed)

    t = 0.0
    i = int(i0)

    times = [t]
    states = [i]

    while t < T and 0 < i < model.N:
        a = float(policy[i])

        lam = model.lam(i, a)
        mu = model.mu(i, a)
        rate = lam + mu

        if rate <= 0:
            break

        t += rng.exponential(1.0 / rate)

        if t > T:
            break

        if rng.random() < lam / rate:
            i += 1
        else:
            i -= 1

        times.append(t)
        states.append(i)

    return np.array(times), np.array(states)


def plot_trajectories(params, i0=20, T=200, n_paths=10):
    model, V, opt_policy = solve_model(params)

    # Feasible clipped zero-drift reference.
    baseline_policy = np.full(model.N + 1, model.a_init)
    baseline_policy[0] = 1.0

    aggressive_policy = np.ones(model.N + 1)
    aggressive_policy[0] = 1.0

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for k in range(n_paths):
        t, x = simulate_birth_death(
            model,
            opt_policy,
            i0=i0,
            T=T,
            seed=100 + k,
        )

        ax.step(
            t,
            x,
            where="post",
            alpha=0.8,
            lw=1.5,
            label="optimal policy" if k == 0 else None,
        )

    t, x = simulate_birth_death(
        model,
        baseline_policy,
        i0=i0,
        T=T,
        seed=999,
    )

    ax.step(
        t,
        x,
        where="post",
        color="k",
        ls="--",
        lw=2,
        label=r"clipped $a_0^*$",
    )

    t, x = simulate_birth_death(
        model,
        aggressive_policy,
        i0=i0,
        T=T,
        seed=1234,
    )

    ax.step(
        t,
        x,
        where="post",
        color="k",
        ls=":",
        lw=2,
        label=r"aggressive $a=1$",
    )

    ax.axhline(model.L, ls=":", color="gray", label=r"$L$")

    ax.set_xlabel("Time")
    ax.set_ylabel("Population state $i(t)$")
    ax.set_title("Stochastic trajectories under threshold objective", fontweight="bold")
    add_grid(ax)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_delta0_policy_heatmap(params, delta0_values, xlim=(1, 100)):
    sweep = sweep_parameter(params, "delta0", delta0_values)

    model, _, _ = solve_model(params)

    x_min, x_max = xlim
    x_plot = np.arange(x_min, x_max + 1)
    policy_view = sweep["policies"][:, x_min - 1 : x_max]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        policy_view,
        aspect="auto",
        origin="lower",
        extent=[
            x_min,
            x_max,
            sweep["param_values"].min(),
            sweep["param_values"].max(),
        ],
    )

    plt.colorbar(im, ax=ax, label=r"Optimal action $a^*(i)$")

    # Feasibility threshold: zero drift possible only when delta0 <= r
    if sweep["param_values"].min() <= params["r"] <= sweep["param_values"].max():
        ax.axhline(
            params["r"],
            ls="--",
            color="k",
            lw=1.5,
            label=r"$\delta_0=r$",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Extrinsic death pressure $\delta_0$")
    ax.set_title(
        r"Policy sweep over extrinsic death pressure $\delta_0$", fontweight="bold"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_delta0_drift_heatmap(params, delta0_values, xlim=(1, 100)):
    sweep = sweep_parameter(params, "delta0", delta0_values)

    x_min, x_max = xlim
    x_plot = np.arange(x_min, x_max + 1)
    drift_view = sweep["drifts"][:, x_min - 1 : x_max]

    vmin, vmax = symmetric_drift_limits(drift_view)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        drift_view,
        aspect="auto",
        origin="lower",
        extent=[
            x_min,
            x_max,
            sweep["param_values"].min(),
            sweep["param_values"].max(),
        ],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, label=r"Per-capita drift $(\lambda-\mu)/i$")

    # Zero-drift contour
    ax.contour(
        x_plot,
        sweep["param_values"],
        drift_view,
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )

    # Feasibility threshold
    if sweep["param_values"].min() <= params["r"] <= sweep["param_values"].max():
        ax.axhline(
            params["r"],
            ls="--",
            color="k",
            lw=1.5,
            label=r"$\delta_0=r$",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Extrinsic death pressure $\delta_0$")
    ax.set_title(
        r"Drift sweep over extrinsic death pressure $\delta_0$", fontweight="bold"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_dormant_width_vs_delta0(params, delta0_values, eps=0.05):
    widths = []

    for d0 in delta0_values:
        p = params.copy()
        p["delta0"] = float(d0)

        model, V, policy = solve_model(p)
        widths.append(dormant_width(model, policy, eps=eps))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(delta0_values, widths, marker="o", lw=2)

    if delta0_values.min() <= params["r"] <= delta0_values.max():
        ax.axvline(
            params["r"],
            ls="--",
            color="k",
            lw=1.5,
            label=r"$\delta_0=r$",
        )

    ax.set_xlabel(r"Extrinsic death pressure $\delta_0$")
    ax.set_ylabel(r"Number of states with $|(\lambda-\mu)/i|\leq \varepsilon$")
    ax.set_title(
        rf"Dormant interval width vs. $\delta_0$, $\varepsilon={eps}$",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve_threshold_policy(params):
    model, V, policy = solve_model(params)
    i, drift = per_capita_drift(model, policy)
    return model, V, policy, i, drift


def plot_S1A_policy_slices_across_kappa2(
    kappa2_values=(0.1, 1.0, 10.0, 50.0),
    savepath=None,
):
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)

    for kappa2 in kappa2_values:
        p = dict(FIG_S1_BASE)
        p["kappa2"] = float(kappa2)

        model, V, policy, i, drift = solve_threshold_policy(p)
        x = state_axis(model)

        ax.plot(
            x,
            policy[x],
            linewidth=2,
            label=rf"$\kappa_2 = {kappa2}$",
        )

    a_star = (FIG_S1_BASE["delta"] + FIG_S1_BASE["delta0"]) / (
        FIG_S1_BASE["r"] + FIG_S1_BASE["delta"]
    )

    ax.axhline(
        a_star,
        color="k",
        linestyle="--",
        linewidth=1.5,
        label=r"Baseline $a^*$",
    )

    ax.axvline(
        FIG_S1_BASE["L"],
        color="0.6",
        linestyle=":",
        linewidth=1.5,
        label=r"$L$ threshold",
    )

    ax.set_title(
        "Policy slices across $\\kappa_2$",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel(r"State $i$", fontsize=14)
    ax.set_ylabel(r"Optimal action $\pi^*(i)$", fontsize=14)

    ax.tick_params(axis="both", labelsize=11)

    ax.set_xlim(1, FIG_S1_BASE["N"])
    ax.set_ylim(-0.02, 1.02)

    add_grid(ax)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300)

    plt.show()


def plot_S1B_drift_slices_across_kappa2(
    kappa2_values=(0.1, 1.0, 10.0, 50.0),
    savepath=None,
):
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)

    for kappa2 in kappa2_values:
        p = dict(FIG_S1_BASE)
        p["kappa2"] = float(kappa2)

        model, V, policy, i, drift = solve_threshold_policy(p)
        x = state_axis(model)

        ax.plot(
            x,
            drift[x],
            linewidth=2,
            label=rf"$\kappa_2 = {kappa2}$",
        )

    ax.axhline(
        0.0,
        color="k",
        linestyle="--",
        linewidth=1.2,
    )

    ax.axvline(
        FIG_S1_BASE["L"],
        color="0.6",
        linestyle=":",
        linewidth=1.5,
        label=r"$L$ threshold",
    )

    ax.set_title(
        "Drift slices across $\\kappa_2$",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel(r"State $i$", fontsize=14)
    ax.set_ylabel(
        r"Normalized net drift $\frac{\lambda-\mu}{i}$",
        fontsize=14,
    )

    ax.tick_params(axis="both", labelsize=11)

    ax.set_xlim(1, FIG_S1_BASE["N"])

    add_grid(ax)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300)

    plt.show()


def plot_S1C_cost_ratio_regime(
    kappa1_values=(100.0, 1.0, 0.01),
    kappa2_fixed=1.0,
    savepath=None,
):
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)

    for kappa1 in kappa1_values:
        p = dict(FIG_S1_BASE)
        p["kappa1"] = float(kappa1)
        p["kappa2"] = float(kappa2_fixed)

        model, V, policy, i, drift = solve_threshold_policy(p)
        x = state_axis(model)

        ax.plot(
            x,
            policy[x],
            linewidth=2,
            label=rf"$\kappa_1 = {kappa1}, \ \kappa_2 = {kappa2_fixed}$",
        )

    a_star = (FIG_S1_BASE["delta"] + FIG_S1_BASE["delta0"]) / (
        FIG_S1_BASE["r"] + FIG_S1_BASE["delta"]
    )

    ax.axhline(
        a_star,
        color="k",
        linestyle="--",
        linewidth=1.5,
    )

    ax.axvline(
        FIG_S1_BASE["L"],
        color="0.6",
        linestyle=":",
        linewidth=1.5,
        label=r"$L$ threshold",
    )

    ax.set_title(
        r"Cost Ratio Regime: $\kappa_1$ vs $\kappa_2$",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel(r"State $i$", fontsize=14)
    ax.set_ylabel(r"Optimal action $\pi^*(i)$", fontsize=14)

    ax.tick_params(axis="both", labelsize=11)

    ax.set_xlim(1, FIG_S1_BASE["N"])
    ax.set_ylim(-0.02, 1.02)

    add_grid(ax)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300)

    plt.show()


def plot_S1D_temporal_horizon_regime(
    alpha_values=(10.0, 1.0, 0.1, 0.01),
    savepath=None,
):
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)

    for alpha in alpha_values:
        p = dict(FIG_S1_BASE)
        p["alpha"] = float(alpha)

        model, V, policy, i, drift = solve_threshold_policy(p)
        x = state_axis(model)

        ax.plot(
            x,
            policy[x],
            linewidth=2,
            label=rf"$\alpha = {alpha}$",
        )

    ax.axvline(
        FIG_S1_BASE["L"],
        color="0.6",
        linestyle=":",
        linewidth=1.5,
        label=r"$L$ threshold",
    )

    ax.set_title(
        "Temporal Horizon Regime: Myopic vs Long-Horizon",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel(r"State $i$", fontsize=14)
    ax.set_ylabel(r"Optimal action $\pi^*(i)$", fontsize=14)

    ax.tick_params(axis="both", labelsize=11)

    ax.set_xlim(1, FIG_S1_BASE["N"])
    ax.set_ylim(-0.02, 1.02)

    add_grid(ax)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300)

    plt.show()


def plot_S1E_drift_regime(
    rd_pairs=((0.1, 1.1), (0.3, 0.9), (0.9, 0.3), (1.1, 0.1)),
    savepath=None,
):
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)

    for r_val, delta_val in rd_pairs:
        p = dict(FIG_S1_BASE)
        p["r"] = float(r_val)
        p["delta"] = float(delta_val)

        model, V, policy, i, drift = solve_threshold_policy(p)
        x = state_axis(model)

        ax.plot(
            x,
            policy[x],
            linewidth=2,
            label=rf"$r={r_val}, \delta={delta_val}$",
        )

    ax.axvline(
        FIG_S1_BASE["L"],
        color="0.6",
        linestyle=":",
        linewidth=1.5,
        label=r"$L$ threshold",
    )

    ax.set_title(
        "Drift Regime: Growth vs Decay Biased",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel(r"State $i$", fontsize=14)
    ax.set_ylabel(r"Optimal action $\pi^*(i)$", fontsize=14)

    ax.tick_params(axis="both", labelsize=11)

    ax.set_xlim(1, FIG_S1_BASE["N"])
    ax.set_ylim(-0.02, 1.02)

    add_grid(ax)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300)

    plt.show()


def plot_S1F_capacity_regime(
    L_values=(25.0, 50.0, 125.0),
    savepath=None,
):
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)

    for L_val in L_values:
        p = dict(FIG_S1_BASE)
        p["L"] = float(L_val)

        model, V, policy, i, drift = solve_threshold_policy(p)
        x = state_axis(model)

        (line,) = ax.plot(
            x,
            policy[x],
            linewidth=2,
            label=rf"$L={L_val:g}$ ($N={p['N']}$)",
        )

        ax.axvline(
            L_val,
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            color=line.get_color(),
        )

    ax.set_title(
        "Capacity Regime: Threshold Position",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel(r"State $i$", fontsize=14)
    ax.set_ylabel(r"Optimal action $\pi^*(i)$", fontsize=14)

    ax.tick_params(axis="both", labelsize=11)

    ax.set_xlim(1, FIG_S1_BASE["N"])
    ax.set_ylim(0.25, 1.02)

    add_grid(ax)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300)

    plt.show()


# Run everything

BASE = dict(
    N=200,
    r=0.30,
    delta=0.15,
    delta0=0.05,  # extrinsic death pressure
    alpha=0.5,
    L=50,
    kappa1=1.0,
    kappa2=5.0,
    penalty=10 ^ 4,
    n_actions=201,
)

plot_policy_curve(BASE)
plot_drift_curve(BASE)

kappa2_values = np.linspace(0.1, 12.0, 40)
plot_kappa2_heatmap(BASE, kappa2_values)
plot_kappa2_drift_heatmap(BASE, kappa2_values)
plot_dormant_width_vs_kappa2(BASE, kappa2_values, eps=0.05)

L_values = np.linspace(50, 180, 40)
plot_L_heatmap(BASE, L_values)
plot_L_drift_heatmap(BASE, L_values)

plot_trajectories(BASE, i0=20, T=200, n_paths=8)

delta0_values = np.linspace(0.0, 0.45, 40)

plot_delta0_policy_heatmap(BASE, delta0_values, xlim=(1, 100))
plot_delta0_drift_heatmap(BASE, delta0_values, xlim=(1, 100))
plot_dormant_width_vs_delta0(BASE, delta0_values, eps=0.05)

# Figure S1 base parameters

FIG_S1_BASE = dict(
    N=200,
    r=0.30,
    delta=0.15,
    delta0=0.05,
    alpha=0.5,
    penalty=1e4,
    L=50,
    kappa1=1.0,
    kappa2=1.0,
    n_actions=101,
)
plot_S1A_policy_slices_across_kappa2()
plot_S1B_drift_slices_across_kappa2()
plot_S1C_cost_ratio_regime()
plot_S1D_temporal_horizon_regime()
plot_S1E_drift_regime()
plot_S1F_capacity_regime()
# %% [markdown]
# ## Quadratic cost model

# %%


def solve_model_v2(params, verbose=False):
    model = DormancyCTMDP_DiscountedPI_V2(params)
    policy = (
        model.solve(verbose=verbose)
        if "verbose" in model.solve.__code__.co_varnames
        else model.solve()
    )
    V = model.V.copy() if hasattr(model, "V") else model.evaluate_policy(policy)
    return model, V, policy


def per_capita_drift_v2(model, policy):
    i_grid = np.arange(model.N + 1)
    drift = np.full(model.N + 1, np.nan)

    for i in range(1, model.N + 1):
        a = float(policy[i])
        drift[i] = (model.lam(i, a) - model.mu(i, a)) / i

    return i_grid, drift


def cost_components_v2(model, i_grid, policy):
    c1_term = model.c1 * i_grid**2
    c2_term = model.c2 * model.r * policy * i_grid**2
    kappa_term = model.kappa * (policy - model.a_star) ** 2 * i_grid
    reward_term = -model.c3 * i_grid
    total = c1_term + c2_term + kappa_term + reward_term

    return c1_term, c2_term, kappa_term, reward_term, total


def find_zero_crossing(i_grid, drift):
    mask = i_grid > 0
    i = i_grid[mask]
    d = drift[mask]

    sign_change = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]

    if len(sign_change) == 0:
        return None

    j = sign_change[0]
    return i[j], i[j + 1]


def add_grid(ax):
    ax.grid(True, alpha=0.3)


def plot_state_max():
    return 100


def symmetric_drift_limits(mat):
    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def plot_policy_curve_v2(params):
    model, V, policy = solve_model_v2(params)
    i = np.arange(1, model.N)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(i, policy[1 : model.N], lw=2)

    ax.set_xlim(1, model.N - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Optimal action $a^*(i)$")
    ax.set_title("Quadratic-cost objective: optimal policy", fontweight="bold")
    add_grid(ax)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_drift_curve_v2(params):
    model, V, policy = solve_model_v2(params)
    i, drift = per_capita_drift_v2(model, policy)

    mask = (i > 0) & (i < model.N)
    crossing = find_zero_crossing(i, drift)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(i[mask], drift[mask], lw=2)
    ax.axhline(0.0, ls="--", color="k")

    if crossing is not None:
        xmid = 0.5 * (crossing[0] + crossing[1])
        ax.axvline(xmid, ls=":", color="k", label="zero crossing")

    ax.set_xlim(1, model.N - 1)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Per-capita drift $(\lambda-\mu)/i$")
    ax.set_title("Quadratic-cost objective: induced net drift", fontweight="bold")
    add_grid(ax)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_cost_decomposition_v2(params):
    model, V, policy = solve_model_v2(params)

    i = np.arange(1, model.N)
    policy_i = policy[1 : model.N]

    c1_term, c2_term, kappa_term, reward_term, total = cost_components_v2(
        model, i, policy_i
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(i, c1_term, lw=2, label=r"$c_1 i^2$")
    ax.plot(i, c2_term, lw=2, label=r"$c_2 r a i^2$")
    ax.plot(i, kappa_term, lw=2, label=r"$\kappa(a-a_0^*)^2 i$")
    ax.plot(i, reward_term, lw=2, label=r"$-c_3 i$")
    ax.plot(i, total, lw=3, ls="--", label="total")
    ax.axhline(0.0, ls=":", color="k")

    ax.set_xlim(1, model.N - 1)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel("Instantaneous cost component")
    ax.set_title("Cost decomposition under optimal policy", fontweight="bold")
    add_grid(ax)
    ax.legend()

    plt.tight_layout()
    plt.show()


def sweep_param_policy_v2(params, param_name, values):
    policies = []

    for val in values:
        p = params.copy()
        p[param_name] = float(val)

        model, V, policy = solve_model_v2(p)
        policies.append(policy[1 : model.N])

    return np.array(policies)


def sweep_param_drift_v2(params, param_name, values):
    drifts = []

    for val in values:
        p = params.copy()
        p[param_name] = float(val)

        model, V, policy = solve_model_v2(p)
        i, drift = per_capita_drift_v2(model, policy)
        drifts.append(drift[1 : model.N])

    return np.array(drifts)


def plot_policy_heatmap_v2(params, param_name, values, title=None, x_max=100):
    mat = sweep_param_policy_v2(params, param_name, values)

    x_max = min(x_max, mat.shape[1])
    mat_view = mat[:, :x_max]

    # Font sizes
    title_fs = 20
    label_fs = 17
    tick_fs = 14
    cbar_label_fs = 16
    cbar_tick_fs = 13

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        mat_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, values.min(), values.max()],
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Optimal action $\pi^*(i)$", fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)
    pretty_param = param_name_dict.get(param_name, param_name)
    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_ylabel(rf"${pretty_param}$", fontsize=label_fs)

    if title is None:
        title = rf"Sweep over ${pretty_param}$"

    ax.set_title(
        title,
        fontsize=title_fs,
        fontweight="bold",
        pad=10,
    )

    ax.tick_params(axis="both", labelsize=tick_fs)

    # Compact y-axis tick labels so small values like 0.001 do not shrink the plot
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.yaxis.get_offset_text().set_fontsize(tick_fs)

    # Fixed margins so heatmap panels have consistent visible size across figures
    fig.subplots_adjust(
        left=0.16,
        right=0.86,
        bottom=0.18,
        top=0.88,
    )

    plt.show()


def plot_drift_heatmap_v2(params, param_name, values, title=None, x_max=100):
    mat = sweep_param_drift_v2(params, param_name, values)

    x_max = min(x_max, mat.shape[1])
    mat_view = mat[:, :x_max]
    x_plot = np.arange(1, x_max + 1)

    vmin, vmax = symmetric_drift_limits(mat_view)

    # Font sizes
    title_fs = 20
    label_fs = 17
    tick_fs = 14
    cbar_label_fs = 16
    cbar_tick_fs = 13

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        mat_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, values.min(), values.max()],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        r"Per-capita drift $(\lambda-\mu)/i$",
        fontsize=cbar_label_fs,
    )
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    #     x_plot,
    #     values,
    #     mat_view,
    #     levels=[0.0],
    #     colors="k",
    #     linewidths=1.2,
    # )
    pretty_param = param_name_dict.get(param_name, param_name)
    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_ylabel(rf"${pretty_param}$", fontsize=label_fs)

    if title is None:
        title = rf"Drift sweep over ${pretty_param}$"

    ax.set_title(
        title,
        fontsize=title_fs,
        fontweight="bold",
        pad=10,
    )

    ax.tick_params(axis="both", labelsize=tick_fs)

    # Compact y-axis tick labels so small values like 0.001 do not shrink the plot
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.yaxis.get_offset_text().set_fontsize(tick_fs)

    # Fixed margins so heatmap panels have consistent visible size across figures
    fig.subplots_adjust(
        left=0.16,
        right=0.86,
        bottom=0.18,
        top=0.88,
    )

    plt.show()


def growth_horizon_vs_param_v2(params, param_name, values):
    horizons = []

    for val in values:
        p = params.copy()
        p[param_name] = float(val)

        model, V, policy = solve_model_v2(p)
        i, drift = per_capita_drift_v2(model, policy)
        crossing = find_zero_crossing(i, drift)

        if crossing is None:
            horizons.append(np.nan)
        else:
            horizons.append(0.5 * (crossing[0] + crossing[1]))

    return np.array(horizons)


def plot_growth_horizon_v2(params, param_name, values):
    horizons = growth_horizon_vs_param_v2(params, param_name, values)

    pretty_param = param_name_dict.get(param_name, param_name)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(values, horizons, marker="o", lw=2)

    ax.set_xlabel(rf"${pretty_param}$")
    ax.set_ylabel("Growth horizon / drift zero-crossing")
    ax.set_title(rf"Interior regulating state vs ${pretty_param}$", fontweight="bold")
    add_grid(ax)

    plt.tight_layout()
    plt.show()


def max_population_under_policy(model, policy, i0=10, T=300, seed=0):
    rng = np.random.default_rng(seed)

    t = 0.0
    i = int(i0)
    max_i = i

    while t < T and 0 < i < model.N:
        a = float(policy[i])

        lam = model.lam(i, a)
        mu = model.mu(i, a)
        rate = lam + mu

        if rate <= 0:
            break

        t += rng.exponential(1.0 / rate)

        if t > T:
            break

        if rng.random() < lam / rate:
            i += 1
        else:
            i -= 1

        max_i = max(max_i, i)

    return max_i


def plot_c1_c3_phase_map_v2(params, c1_values, c3_values, i0=10, T=300):
    phase = np.zeros((len(c1_values), len(c3_values)))

    for a_idx, c1 in enumerate(c1_values):
        for b_idx, c3 in enumerate(c3_values):
            p = params.copy()
            p["c1"] = float(c1)
            p["c3"] = float(c3)

            model, V, policy = solve_model_v2(p)
            phase[a_idx, b_idx] = max_population_under_policy(
                model, policy, i0=i0, T=T, seed=123
            )

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    im = ax.imshow(
        phase,
        aspect="auto",
        origin="lower",
        extent=[c3_values.min(), c3_values.max(), c1_values.min(), c1_values.max()],
    )

    plt.colorbar(im, ax=ax, label="Maximum population reached")

    ax.set_xlabel(r"$c_3$ growth reward")
    ax.set_ylabel(r"$c_1$ quadratic penalty")
    ax.set_title(r"Growth/suppression phase map: $c_1$ vs $c_3$", fontweight="bold")

    plt.tight_layout()
    plt.show()


def simulate_birth_death_v2(model, policy, i0=10, T=300, seed=None):
    rng = np.random.default_rng(seed)

    t = 0.0
    i = int(i0)

    times = [t]
    states = [i]

    while t < T and 0 < i < model.N:
        a = float(policy[i])

        lam = model.lam(i, a)
        mu = model.mu(i, a)
        rate = lam + mu

        if rate <= 0:
            break

        t += rng.exponential(1.0 / rate)

        if t > T:
            break

        if rng.random() < lam / rate:
            i += 1
        else:
            i -= 1

        times.append(t)
        states.append(i)

    return np.array(times), np.array(states)


def plot_trajectories_v2(params, i0=10, T=300, n_paths=8):
    model, V, opt_policy = solve_model_v2(params)

    baseline_policy = np.full(model.N + 1, model.a_init)
    baseline_policy[0] = 1.0

    aggressive_policy = np.ones(model.N + 1)
    aggressive_policy[0] = 1.0

    i, drift = per_capita_drift_v2(model, opt_policy)
    crossing = find_zero_crossing(i, drift)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for k in range(n_paths):
        t, x = simulate_birth_death_v2(model, opt_policy, i0=i0, T=T, seed=100 + k)
        ax.step(
            t,
            x,
            where="post",
            alpha=0.8,
            lw=1.5,
            label="optimal policy" if k == 0 else None,
        )

    t, x = simulate_birth_death_v2(model, baseline_policy, i0=i0, T=T, seed=999)
    ax.step(
        t,
        x,
        where="post",
        color="k",
        ls="--",
        lw=2,
        label=r"clipped $a_0^*$",
    )

    t, x = simulate_birth_death_v2(model, aggressive_policy, i0=i0, T=T, seed=1234)
    ax.step(
        t,
        x,
        where="post",
        color="k",
        ls=":",
        lw=2,
        label=r"aggressive $a=1$",
    )

    if crossing is not None:
        xmid = 0.5 * (crossing[0] + crossing[1])
        ax.axhline(xmid, color="gray", ls=":", label="growth horizon")

    ax.set_xlabel("Time")
    ax.set_ylabel("Population state $i(t)$")
    ax.set_title(
        "Stochastic trajectories under quadratic-cost objective", fontweight="bold"
    )
    add_grid(ax)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_delta0_policy_heatmap_v2(params, delta0_values, x_max=100):
    plot_policy_heatmap_v2(
        params,
        "delta0",
        delta0_values,
        title=r"Policy sweep over extrinsic death pressure $\delta_0$",
        x_max=x_max,
    )


def plot_delta0_drift_heatmap_v2(params, delta0_values, x_max=100):
    mat = sweep_param_drift_v2(params, "delta0", delta0_values)

    x_max = min(x_max, mat.shape[1])
    mat_view = mat[:, :x_max]
    x_plot = np.arange(1, x_max + 1)

    vmin, vmax = symmetric_drift_limits(mat_view)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        mat_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, delta0_values.min(), delta0_values.max()],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, label=r"Per-capita drift $(\lambda-\mu)/i$")

    ax.contour(
        x_plot,
        delta0_values,
        mat_view,
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )

    # Feasibility boundary for zero drift: delta0 = r
    ax.axhline(
        params["r"],
        ls=":",
        color="k",
        lw=2,
        label=r"$\delta_0=r$",
    )

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"$\delta_0$")
    ax.set_title(
        r"Drift sweep over extrinsic death pressure $\delta_0$",
        fontweight="bold",
    )
    ax.legend()

    plt.tight_layout()
    plt.show()


# Run baseline plots

BASE_V2 = dict(
    N=250,
    r=0.30,
    delta=0.15,
    delta0=0.05,
    alpha=0.05,
    c1=0.005,
    c2=0.02,
    kappa=2.0,
    c3=1.0,
    penalty=1e4,
    n_actions=201,
)

plot_policy_curve_v2(BASE_V2)
plot_drift_curve_v2(BASE_V2)
plot_cost_decomposition_v2(BASE_V2)

c1_values = np.linspace(0.001, 0.03, 40)
c2_values = np.linspace(0.0, 0.10, 40)
kappa_values = np.linspace(0.2, 10.0, 40)
c3_values = np.linspace(0.2, 5.0, 40)
alpha_values = np.linspace(0.01, 0.30, 40)

plot_policy_heatmap_v2(
    BASE_V2, "c1", c1_values, title=r"Quadratic Cost Model: Optimal Policy", x_max=100
)
plot_policy_heatmap_v2(
    BASE_V2,
    "c2",
    c2_values,
    title=r"Policy sweep over proliferation-size penalty $c_2$",
    x_max=100,
)
plot_policy_heatmap_v2(
    BASE_V2,
    "kappa",
    kappa_values,
    title=r"Policy sweep over deviation penalty $\kappa$",
    x_max=100,
)
plot_policy_heatmap_v2(
    BASE_V2, "c3", c3_values, title=r"Policy sweep over growth reward $c_3$", x_max=100
)
plot_policy_heatmap_v2(
    BASE_V2,
    "alpha",
    alpha_values,
    title=r"Policy sweep over discount factor $\alpha$",
    x_max=100,
)

plot_drift_heatmap_v2(
    BASE_V2, "c1", c1_values, title=r"Quadratic Cost Model: Normalized Drift", x_max=100
)
plot_drift_heatmap_v2(
    BASE_V2, "c3", c3_values, title=r"Drift sweep over growth reward $c_3$", x_max=100
)

plot_growth_horizon_v2(BASE_V2, "c1", c1_values)
plot_growth_horizon_v2(BASE_V2, "c3", c3_values)

plot_c1_c3_phase_map_v2(BASE_V2, c1_values, c3_values, i0=10, T=300)

plot_trajectories_v2(BASE_V2, i0=10, T=300, n_paths=8)

delta0_values = np.linspace(0.0, 0.40, 40)

plot_policy_heatmap_v2(
    BASE_V2,
    "delta0",
    delta0_values,
    title=r"Policy sweep over extrinsic death pressure $\delta_0$",
    x_max=100,
)

plot_delta0_drift_heatmap_v2(
    BASE_V2,
    delta0_values,
    x_max=100,
)

plot_growth_horizon_v2(BASE_V2, "delta0", delta0_values)

# %% [markdown]
# ## Unbounded reward model

# %%


def solve_unbounded(params, verbose=False):
    model = DormancyCTMDP_Unbounded_Fast(params)
    u, policy = (
        model.solve(verbose=verbose)
        if "verbose" in model.solve.__code__.co_varnames
        else model.solve()
    )
    return model, u, policy


def per_capita_drift_unbounded(model, policy):
    i = np.arange(model.N + 1)
    drift = np.full(model.N + 1, np.nan)

    for k in range(1, model.N + 1):
        a = float(policy[k])
        drift[k] = (model.lam(k, a) - model.mu(k, a)) / k

    return i, drift


def find_zero_crossing(i, drift):
    mask = i > 0
    ii = i[mask]
    dd = drift[mask]

    idx = np.where(np.sign(dd[:-1]) != np.sign(dd[1:]))[0]

    if len(idx) == 0:
        return None

    j = idx[0]
    return 0.5 * (ii[j] + ii[j + 1])


def add_grid(ax):
    ax.grid(True, alpha=0.3)


def symmetric_drift_limits(mat):
    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def crop_state_matrix(mat, x_max=100):
    x_max = min(int(x_max), mat.shape[1])
    mat_view = mat[:, :x_max]
    x_plot = np.arange(1, x_max + 1)
    return mat_view, x_plot, x_max


def plot_policy_unbounded(params):
    model, u, policy = solve_unbounded(params)
    i = np.arange(1, model.N)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(i, policy[1 : model.N], lw=2)
    ax.axhline(model.a_init, ls="--", color="k", label=r"$a^*$")

    ax.set_xlim(1, model.N - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Optimal action $\pi^*(i)$")
    ax.set_title("Unbounded linear-reward model: optimal policy", fontweight="bold")
    add_grid(ax)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_drift_unbounded(params):
    model, u, policy = solve_unbounded(params)
    i, drift = per_capita_drift_unbounded(model, policy)

    mask = (i > 0) & (i < model.N)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(i[mask], drift[mask], lw=2)
    ax.axhline(0.0, ls="--", color="k", label=r"Baseline $a^*$")

    ax.set_xlim(1, model.N - 1)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Per-capita drift $(\lambda-\mu)/i$")
    ax.set_title("Unbounded linear-reward model: induced drift", fontweight="bold")
    add_grid(ax)

    plt.tight_layout()
    plt.show()


def plot_reward_decomposition_unbounded(params):
    model, u, policy = solve_unbounded(params)

    i = np.arange(1, model.N)
    p = policy[1 : model.N]

    growth_reward = model.c3 * i
    deviation_penalty = -model.kappa * i * (p - model.a_star) ** 2
    total = growth_reward + deviation_penalty

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(i, growth_reward, lw=2, label=r"$c_3 i$")
    ax.plot(i, deviation_penalty, lw=2, label=r"$-\kappa i(a-a_0^*)^2$")
    ax.plot(i, total, lw=3, ls="--", label="total reward")
    ax.axhline(0.0, ls=":", color="k")

    ax.set_xlim(1, model.N - 1)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel("Instantaneous reward component")
    ax.set_title("Reward decomposition under optimal policy", fontweight="bold")
    add_grid(ax)
    ax.legend()

    plt.tight_layout()
    plt.show()


def sweep_policy_unbounded(params, param_name, values):
    policies = []

    for val in values:
        p = params.copy()
        p[param_name] = float(val)

        model, u, policy = solve_unbounded(p)
        policies.append(policy[1 : model.N])

    return np.array(policies)


def plot_policy_heatmap_unbounded(params, param_name, values, title=None, x_max=100):
    mat = sweep_policy_unbounded(params, param_name, values)
    mat_view, x_plot, x_max = crop_state_matrix(mat, x_max=x_max)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    pretty_param = param_name_dict.get(param_name, param_name)
    im = ax.imshow(
        mat_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, values.min(), values.max()],
    )

    plt.colorbar(im, ax=ax, label=r"Optimal action $a^*(i)$")

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(rf"${pretty_param}$")

    if title is None:
        title = rf"Policy sweep over ${pretty_param}$"

    ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    plt.show()


def sweep_drift_unbounded(params, param_name, values):
    drifts = []

    for val in values:
        p = params.copy()
        p[param_name] = float(val)

        model, u, policy = solve_unbounded(p)
        i, drift = per_capita_drift_unbounded(model, policy)
        drifts.append(drift[1 : model.N])

    return np.array(drifts)


def plot_drift_heatmap_unbounded(params, param_name, values, title=None, x_max=100):
    mat = sweep_drift_unbounded(params, param_name, values)
    mat_view, x_plot, x_max = crop_state_matrix(mat, x_max=x_max)
    pretty_param = param_name_dict.get(param_name, param_name)
    vmin, vmax = symmetric_drift_limits(mat_view)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        mat_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, values.min(), values.max()],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, label=r"Per-capita drift $(\lambda-\mu)/i$")

    # Zero-drift contour
    ax.contour(
        x_plot,
        values,
        mat_view,
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(rf"${pretty_param}$")

    if title is None:
        title = rf"Drift sweep over ${pretty_param}$"

    ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_delta0_drift_heatmap_unbounded(params, delta0_values, x_max=100):
    mat = sweep_drift_unbounded(params, "delta0", delta0_values)
    mat_view, x_plot, x_max = crop_state_matrix(mat, x_max=x_max)

    vmin, vmax = symmetric_drift_limits(mat_view)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        mat_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, delta0_values.min(), delta0_values.max()],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, label=r"Per-capita drift $(\lambda-\mu)/i$")

    # Zero-drift contour
    ax.contour(
        x_plot,
        delta0_values,
        mat_view,
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )

    # Feasibility boundary for zero drift
    ax.axhline(
        params["r"],
        ls=":",
        color="k",
        lw=2,
        label=r"$\delta_0=r$",
    )

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"$\delta_0$")
    ax.set_title(
        r"Drift sweep over extrinsic death pressure $\delta_0$",
        fontweight="bold",
    )
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_ratio_sweep_unbounded(params, ratios, fixed_c3=1.0, x_max=100):
    policies = []
    drifts = []

    for ratio in ratios:
        p = params.copy()
        p["c3"] = fixed_c3
        p["kappa"] = ratio * fixed_c3

        model, u, policy = solve_unbounded(p)
        i, drift = per_capita_drift_unbounded(model, policy)

        policies.append(policy[1 : model.N])
        drifts.append(drift[1 : model.N])

    policies = np.array(policies)
    drifts = np.array(drifts)

    policy_view, x_plot, x_max = crop_state_matrix(policies, x_max=x_max)
    drift_view, _, _ = crop_state_matrix(drifts, x_max=x_max)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        policy_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, ratios.min(), ratios.max()],
    )

    plt.colorbar(im, ax=ax, label=r"Optimal action $a^*(i)$")

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"$\kappa/c_3$")
    ax.set_title(r"Policy sweep over ratio $\kappa/c_3$", fontweight="bold")

    plt.tight_layout()
    plt.show()

    vmin, vmax = symmetric_drift_limits(drift_view)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        drift_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, ratios.min(), ratios.max()],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, label=r"Per-capita drift $(\lambda-\mu)/i$")

    ax.contour(
        x_plot,
        ratios,
        drift_view,
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"$\kappa/c_3$")
    ax.set_title(r"Drift sweep over ratio $\kappa/c_3$", fontweight="bold")

    plt.tight_layout()
    plt.show()


def simulate_birth_death_unbounded(model, policy, i0=10, T=300, seed=None):
    rng = np.random.default_rng(seed)

    t = 0.0
    i = int(i0)

    times = [t]
    states = [i]

    while t < T and 0 < i < model.N:
        a = float(policy[i])

        lam = model.lam(i, a)
        mu = model.mu(i, a)
        rate = lam + mu

        if rate <= 0:
            break

        t += rng.exponential(1.0 / rate)

        if t > T:
            break

        if rng.random() < lam / rate:
            i += 1
        else:
            i -= 1

        times.append(t)
        states.append(i)

    return np.array(times), np.array(states)


def plot_trajectories_unbounded(params, i0=10, T=300, n_paths=8):
    model, u, opt_policy = solve_unbounded(params)

    baseline_policy = np.full(model.N + 1, model.a_init)
    baseline_policy[0] = 1.0

    aggressive_policy = np.ones(model.N + 1)
    aggressive_policy[0] = 1.0

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for k in range(n_paths):
        t, x = simulate_birth_death_unbounded(
            model,
            opt_policy,
            i0=i0,
            T=T,
            seed=100 + k,
        )

        ax.step(
            t,
            x,
            where="post",
            alpha=0.8,
            lw=1.5,
            label="optimal policy" if k == 0 else None,
        )

    t, x = simulate_birth_death_unbounded(
        model,
        baseline_policy,
        i0=i0,
        T=T,
        seed=999,
    )

    ax.step(
        t,
        x,
        where="post",
        color="k",
        ls="--",
        lw=2,
        label=r"clipped $a_0^*$",
    )

    t, x = simulate_birth_death_unbounded(
        model,
        aggressive_policy,
        i0=i0,
        T=T,
        seed=1234,
    )

    ax.step(
        t,
        x,
        where="post",
        color="k",
        ls=":",
        lw=2,
        label=r"aggressive $a=1$",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Population state $i(t)$")
    ax.set_title(
        "Stochastic trajectories under unbounded linear reward",
        fontweight="bold",
    )
    add_grid(ax)
    ax.legend()

    plt.tight_layout()
    plt.show()


BASE_UNBOUNDED = dict(
    N=800,
    r=0.30,
    delta=0.15,
    delta0=0.05,  # extrinsic death pressure
    alpha=0.51,
    kappa=1.0,
    c3=1.0,
    penalty=1e4,
    n_actions=201,
)

# Dormancy-like / near baseline

# Intermediate

# Growth-dominated


kappa_values = np.linspace(0.1, 10.0, 10)
c3_values = np.linspace(0.1, 5.0, 10)
alpha_values = np.linspace(0.51, 1, 40)
delta0_values = np.linspace(0.0, 0.40, 50)
ratios = np.linspace(0.1, 50.0, 60)

plot_policy_heatmap_unbounded(
    BASE_UNBOUNDED,
    "kappa",
    kappa_values,
    title=r"Policy sweep over deviation penalty $\kappa$",
    x_max=200,
)

plot_policy_heatmap_unbounded(
    BASE_UNBOUNDED,
    "c3",
    c3_values,
    title=r"Policy sweep over linear growth reward $c_3$",
    x_max=200,
)

#     "alpha",
#     alpha_values,
#     title=r"Policy sweep over discount factor $\alpha$",
#     x_max=200,
# )

#     "delta0",
#     title=r"Policy sweep over extrinsic death pressure $\delta_0$",
#     x_max=200,
# )

plot_drift_heatmap_unbounded(
    BASE_UNBOUNDED,
    "kappa",
    kappa_values,
    title=r"Drift sweep over deviation penalty $\kappa$",
    x_max=200,
)

plot_drift_heatmap_unbounded(
    BASE_UNBOUNDED,
    "c3",
    c3_values,
    title=r"Drift sweep over linear growth reward $c_3$",
    x_max=200,
)

#     x_max=200,
# )

#     ratios,
#     fixed_c3=1.0,
#     x_max=200,
# )

BASE_UNBOUNDED = dict(
    N=800,
    r=0.30,
    delta=0.150,
    delta0=0.05,  # extrinsic death pressure
    alpha=0.51,
    penalty=1e4,
    n_actions=201,
)

REGIMES = [
    dict(kappa=50.0, c3=0.5, label=r"strong regularization: $\kappa=50,\ c_3=.5$"),
    dict(kappa=1, c3=1.0, label=r"balanced: $\kappa=1,\ c_3=1$"),
    dict(kappa=1.0, c3=5.0, label=r"growth-dominated: $\kappa=1,\ c_3=5$"),
]


def solve_unbounded(params):
    model = DormancyCTMDP_Unbounded_Fast(params)
    u, policy = model.solve()
    return model, u, policy


def per_capita_drift_unbounded(model, policy):
    i_grid = np.arange(model.N + 1)
    drift = np.full(model.N + 1, np.nan)

    for i in range(1, model.N + 1):
        a = float(policy[i])
        drift[i] = (model.lam(i, a) - model.mu(i, a)) / i

    return i_grid, drift


def add_grid(ax):
    ax.grid(True, alpha=0.3)


def crop_state_matrix(mat, x_max=100):
    x_max = min(int(x_max), mat.shape[1])
    mat_view = mat[:, :x_max]
    x_plot = np.arange(1, x_max + 1)
    return mat_view, x_plot, x_max


def symmetric_drift_limits(mat):
    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def add_regime_ratio_lines(ax, regimes=REGIMES, x_text=1.02):
    ymin, ymax = ax.get_ylim()

    for reg in regimes:
        ratio = reg["kappa"] / reg["c3"]

        if ymin <= ratio <= ymax:
            ax.axhline(
                ratio,
                ls=":",
                lw=1.5,
                color="k",
                alpha=0.8,
            )

            ax.text(
                x_text,
                ratio,
                reg["label"],
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=7,
            )


def add_delta0_feasibility_line(ax, params, delta0_values):
    r = float(params["r"])

    if delta0_values.min() <= r <= delta0_values.max():
        ax.axhline(
            r,
            ls=":",
            color="k",
            lw=2,
            label=r"$\delta_0=r$",
        )
        ax.legend(fontsize=8)


# Solve the three regimes


def solve_regimes(base_params=BASE_UNBOUNDED, regimes=REGIMES):
    results = []

    for reg in regimes:
        params = base_params.copy()
        params["kappa"] = reg["kappa"]
        params["c3"] = reg["c3"]

        model, u, policy = solve_unbounded(params)
        i, drift = per_capita_drift_unbounded(model, policy)

        results.append(
            {
                "params": params,
                "label": reg["label"],
                "model": model,
                "u": u,
                "policy": policy,
                "i": i,
                "drift": drift,
            }
        )

    return results


def plot_unbounded_policy_overlay(results, x_max=100):
    # Font sizes
    title_fs = 20
    label_fs = 17
    tick_fs = 14
    legend_fs = 13

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    for res in results:
        model = res["model"]
        i = res["i"]
        policy = res["policy"]

        mask = (i >= 1) & (i <= min(x_max, model.N - 1))

        ax.plot(
            i[mask],
            policy[mask],
            lw=2,
            label=res["label"],
        )

    model0 = results[0]["model"]

    ax.axhline(
        model0.a_init,
        ls="--",
        color="k",
        label=r"Baseline $a^*$",
    )

    ax.set_xlim(1, min(x_max, model0.N - 1))
    ax.set_ylim(0.34, 1.02)

    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_ylabel(r"Optimal action $\pi^*(i)$", fontsize=label_fs)

    ax.set_title(
        "Unbounded linear-reward model: policy comparison",
        fontsize=title_fs,
        fontweight="bold",
        pad=10,
    )

    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)

    ax.legend(
        fontsize=legend_fs,
        frameon=True,
    )

    plt.tight_layout()
    plt.show()


def plot_unbounded_drift_overlay(results, ylim=None, x_max=100):
    # Font sizes
    title_fs = 20
    label_fs = 17
    tick_fs = 14
    legend_fs = 13

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    for res in results:
        model = res["model"]
        i = res["i"]
        drift = res["drift"]

        mask = (i >= 1) & (i <= min(x_max, model.N - 1))

        ax.plot(
            i[mask],
            drift[mask],
            lw=2,
            label=res["label"],
        )

    model0 = results[0]["model"]

    ax.axhline(0.0, ls="--", color="k", label=r"Baseline $a^*$")

    ax.set_xlim(1, min(x_max, model0.N - 1))
    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_ylabel(r"Per-capita drift $(\lambda-\mu)/i$", fontsize=label_fs)
    ax.set_title(
        "Unbounded linear-reward model: drift comparison",
        fontsize=title_fs,
        fontweight="bold",
        pad=10,
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(fontsize=legend_fs, frameon=True)

    plt.tight_layout()
    plt.show()


def plot_unbounded_value_overlay(results, x_max=100):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    for res in results:
        model = res["model"]
        i = res["i"]
        u = res["u"]

        mask = (i >= 1) & (i <= min(x_max, model.N - 1))

        ax.plot(
            i[mask],
            u[mask],
            lw=2,
            label=res["label"],
        )

    model0 = results[0]["model"]

    ax.set_xlim(1, min(x_max, model0.N - 1))
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Value function $u(i)$")
    ax.set_title("Unbounded linear-reward model: value functions", fontweight="bold")
    add_grid(ax)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def ratio_sweep_unbounded(
    base_params=BASE_UNBOUNDED,
    ratios=np.linspace(0.1, 120.0, 100),
    fixed_c3=1.0,
):
    policy_mat = []
    drift_mat = []

    for ratio in ratios:
        params = base_params.copy()
        params["c3"] = fixed_c3
        params["kappa"] = ratio * fixed_c3

        model, u, policy = solve_unbounded(params)
        i, drift = per_capita_drift_unbounded(model, policy)

        # states 1,...,N-1
        policy_mat.append(policy[1 : model.N])
        drift_mat.append(drift[1 : model.N])

    return np.array(policy_mat), np.array(drift_mat), ratios


def plot_ratio_heatmaps_unbounded(
    base_params=BASE_UNBOUNDED,
    ratios=np.linspace(0.1, 120.0, 100),
    fixed_c3=1.0,
    x_max=100,
):
    policy_mat, drift_mat, ratios = ratio_sweep_unbounded(
        base_params=base_params,
        ratios=ratios,
        fixed_c3=fixed_c3,
    )

    policy_view, x_plot, x_max = crop_state_matrix(policy_mat, x_max=x_max)
    drift_view, _, _ = crop_state_matrix(drift_mat, x_max=x_max)

    # Policy heatmap
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        policy_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, ratios.min(), ratios.max()],
    )

    plt.colorbar(im, ax=ax, label=r"Optimal action $pi^*(i)$")

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"$\kappa/c_3$")
    ax.set_title(r"Policy structure over ratio $\kappa/c_3$", fontweight="bold")

    add_regime_ratio_lines(ax, REGIMES)

    plt.tight_layout()
    plt.show()

    # Drift heatmap
    vmin, vmax = symmetric_drift_limits(drift_view)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        drift_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, ratios.min(), ratios.max()],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, label=r"Per-capita drift $(\lambda-\mu)/i$")

    # Zero-drift contour
    ax.contour(
        x_plot,
        ratios,
        drift_view,
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"$\kappa/c_3$")
    ax.set_title(r"Drift structure over ratio $\kappa/c_3$", fontweight="bold")

    add_regime_ratio_lines(ax, REGIMES)

    plt.tight_layout()
    plt.show()


def delta0_sweep_unbounded(
    base_params=BASE_UNBOUNDED,
    delta0_values=np.linspace(0.0, 0.40, 60),
):
    policy_mat = []
    drift_mat = []

    for delta0 in delta0_values:
        params = base_params.copy()
        params["delta0"] = float(delta0)

        # Use a representative regime unless kappa/c3 are already supplied.
        params.setdefault("kappa", 10.0)
        params.setdefault("c3", 1.0)

        model, u, policy = solve_unbounded(params)
        i, drift = per_capita_drift_unbounded(model, policy)

        policy_mat.append(policy[1 : model.N])
        drift_mat.append(drift[1 : model.N])

    return np.array(policy_mat), np.array(drift_mat), delta0_values


def plot_delta0_drift_heatmap_unbounded(
    base_params=BASE_UNBOUNDED,
    delta0_values=np.linspace(0.0, 0.40, 60),
    x_max=100,
):
    policy_mat, drift_mat, delta0_values = delta0_sweep_unbounded(
        base_params=base_params,
        delta0_values=delta0_values,
    )

    drift_view, x_plot, x_max = crop_state_matrix(drift_mat, x_max=x_max)

    vmin, vmax = symmetric_drift_limits(drift_view)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        drift_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, delta0_values.min(), delta0_values.max()],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, label=r"Per-capita drift $(\lambda-\mu)/i$")

    # Zero-drift contour
    ax.contour(
        x_plot,
        delta0_values,
        drift_view,
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )

    # Feasibility boundary delta0 = r
    add_delta0_feasibility_line(ax, base_params, delta0_values)

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"$\delta_0$")
    ax.set_title(
        r"Drift structure over extrinsic death pressure $\delta_0$",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()


def simulate_birth_death_unbounded(model, policy, i0=10, T=300, seed=None):
    rng = np.random.default_rng(seed)

    t = 0.0
    i = int(i0)

    times = [t]
    states = [i]

    while t < T and 0 < i < model.N:
        a = float(policy[i])

        lam = model.lam(i, a)
        mu = model.mu(i, a)
        rate = lam + mu

        if rate <= 0:
            break

        t += rng.exponential(1.0 / rate)

        if t > T:
            break

        if rng.random() < lam / rate:
            i += 1
        else:
            i -= 1

        times.append(t)
        states.append(i)

    return np.array(times), np.array(states)


def plot_unbounded_trajectories(results, i0=10, T=300, n_paths=5):
    for res in results:
        model = res["model"]
        policy = res["policy"]

        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

        for k in range(n_paths):
            t, x = simulate_birth_death_unbounded(
                model,
                policy,
                i0=i0,
                T=T,
                seed=1000 + 17 * k,
            )

            ax.step(
                t,
                x,
                where="post",
                alpha=0.85,
                lw=1.5,
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Population state $i(t)$")
        ax.set_title(res["label"], fontweight="bold")
        add_grid(ax)

        plt.tight_layout()
        plt.show()


results_unbounded = solve_regimes()
xmax = 500
plot_unbounded_policy_overlay(results_unbounded, x_max=xmax)
plot_unbounded_drift_overlay(results_unbounded, x_max=xmax)
plot_unbounded_value_overlay(results_unbounded, x_max=xmax)

plot_unbounded_drift_overlay(results_unbounded, ylim=(-0.02, 0.32), x_max=xmax)

plot_ratio_heatmaps_unbounded(
    base_params=BASE_UNBOUNDED,
    ratios=np.linspace(0.1, 120.0, 50),
    fixed_c3=1.0,
    x_max=xmax,
)

delta0_values = np.linspace(0.0, 0.40, 60)

plot_delta0_drift_heatmap_unbounded(
    base_params={**BASE_UNBOUNDED, "kappa": 10.0, "c3": 1.0},
    delta0_values=delta0_values,
    x_max=xmax,
)

plot_unbounded_trajectories(results_unbounded, i0=10, T=300, n_paths=5)
# %% [markdown]
# ## Constrained linear-reward model

# %%


def get_Lambda(model):
    if hasattr(model, "Lambda"):
        return model.Lambda

    if hasattr(model, "global_uniformization_rate"):
        return model.global_uniformization_rate()

    lam_attr = getattr(model, "lam", None)
    if lam_attr is not None and not callable(lam_attr):
        return lam_attr

    raise AttributeError("Could not determine LP uniformization rate Lambda.")


def add_grid(ax):
    ax.grid(True, alpha=0.3)


def symmetric_drift_limits(mat):
    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def crop_state_matrix(mat, x_max=100):
    x_max = min(int(x_max), mat.shape[1])
    mat_view = mat[:, :x_max]
    x_plot = np.arange(1, x_max + 1)
    return mat_view, x_plot, x_max


def solve_constrained(params, beta):
    model = TumorActorLP(params, beta)
    res = model.solve()

    if not res.success:
        print(f"LP failed for beta={beta}: {res.message}")
        return model, res, None, None

    mixed_policy = model.extract_policy(res.x)

    if hasattr(model, "expected_action_policy"):
        a_exp = model.expected_action_policy(res.x)
    else:
        a_exp = expected_action_from_policy(
            mixed_policy, model.N, default_a=model.a_init
        )

    return model, res, mixed_policy, a_exp


def expected_action_from_policy(policy, N, default_a=0.5):
    a_exp = np.full(N + 1, default_a, dtype=float)

    for i in range(N + 1):
        if i in policy:
            a_exp[i] = sum(float(a) * float(p) for a, p in policy[i])

    a_exp[0] = 1.0
    return np.clip(a_exp, 0.0, 1.0)


def per_capita_drift_constrained(model, a_exp):
    i = np.arange(model.N + 1)
    drift = np.full(model.N + 1, np.nan)

    for k in range(1, model.N + 1):
        a = float(a_exp[k])
        drift[k] = (model.lam(k, a) - model.mu(k, a)) / k

    return i, drift


def budget_used_constrained(model, res):
    if not res.success:
        return np.nan

    n_actions = get_n_actions(model)
    Lambda = get_Lambda(model)

    x = res.x.reshape((model.N + 1, n_actions))
    total = 0.0

    for i in range(model.N + 1):
        for a_idx, a in enumerate(model.actions):
            if hasattr(model, "regulatory_cost"):
                cost = model.regulatory_cost(i, a)
            else:
                cost = model.kappa * i * (a - model.a_star) ** 2

            total += x[i, a_idx] * cost / (model.alpha + Lambda)

    return total


def find_zero_crossings(i, drift):
    i = np.asarray(i, dtype=float)
    drift = np.asarray(drift, dtype=float)

    crossings = []

    for j in range(len(i) - 1):

        if i[j] <= 0 or i[j + 1] <= 0:
            continue

        if not np.isfinite(drift[j]) or not np.isfinite(drift[j + 1]):
            continue

        if drift[j] == 0:
            crossings.append(float(i[j]))

        elif drift[j] * drift[j + 1] < 0:
            crossings.append(0.5 * (float(i[j]) + float(i[j + 1])))

    return crossings


def dormant_width_constrained(i, drift, eps=0.05):
    mask = (i > 0) & np.isfinite(drift)
    return int(np.sum(np.abs(drift[mask]) <= eps))


def dormant_bounds_constrained(i, drift, eps=0.05):
    mask = (i > 0) & np.isfinite(drift) & (np.abs(drift) <= eps)
    states = i[mask]

    if len(states) == 0:
        return np.nan, np.nan

    return states.min(), states.max()


# Single beta plots


def plot_policy_constrained(params, beta):
    model, res, policy, a_exp = solve_constrained(params, beta)

    if a_exp is None:
        return

    # Raw biological drift
    i_full, drift = per_capita_drift_constrained(model, a_exp)

    # Occupancy-supported interpretation
    (
        a_masked,
        drift_masked,
        occupancy,
        support,
        occ_tol,
    ) = mask_constrained_solution(
        model,
        res,
        a_exp,
        drift,
    )

    i = np.arange(1, model.N)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(i, a_masked[1 : model.N], lw=2)

    ax.axhline(model.a_init, ls="--", color="k", label=r"$Baseline\ a^*$")

    ax.set_xlim(1, model.N - 1)
    ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Expected optimal action $\mathbb{E}[\pi\mid i]$")

    ax.set_title(
        rf"Constrained model: optimal policy, $\beta={beta}$", fontweight="bold"
    )

    add_grid(ax)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_drift_constrained(params, beta):
    model, res, policy, a_exp = solve_constrained(params, beta)

    if a_exp is None:
        return

    # Raw biological drift
    i, drift = per_capita_drift_constrained(model, a_exp)

    # Occupancy-supported interpretation
    (
        a_masked,
        drift_masked,
        occupancy,
        support,
        occ_tol,
    ) = mask_constrained_solution(
        model,
        res,
        a_exp,
        drift,
    )

    crossings = find_zero_crossings(i, drift_masked)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(i[1 : model.N], drift_masked[1 : model.N], lw=2)

    ax.axhline(0.0, ls="--", color="k")

    for x in crossings:
        if 1 <= x <= model.N - 1:
            ax.axvline(x, ls=":", color="k")

    ax.set_xlim(1, model.N - 1)

    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Per-capita drift $(\lambda-\mu)/i$")

    ax.set_title(
        rf"Constrained model: induced drift, $\beta={beta}$", fontweight="bold"
    )

    add_grid(ax)

    plt.tight_layout()
    plt.show()


# Overlay plots for multiple beta


def solve_beta_grid(
    params,
    beta_values,
    occ_rel_tol=1e-8,
    occ_abs_tol=1e-12,
):
    results = []

    for beta in beta_values:

        model, res, policy, a_exp = solve_constrained(params, beta)

        if a_exp is None:
            continue

        # Raw biological drift
        i, drift = per_capita_drift_constrained(model, a_exp)

        # Occupancy-supported interpretation
        (
            a_masked,
            drift_masked,
            occupancy,
            support,
            occ_tol,
        ) = mask_constrained_solution(
            model,
            res,
            a_exp,
            drift,
            rel_tol=occ_rel_tol,
            abs_tol=occ_abs_tol,
        )

        used = budget_used_constrained(model, res)

        crossings = find_zero_crossings(i, drift_masked)

        results.append(
            {
                "beta": beta,
                "model": model,
                "res": res,
                "policy": policy,
                # Keep raw action for simulation if needed
                "a_exp_raw": a_exp,
                # Use masked quantities for interpretation/plots
                "a_exp": a_masked,
                "i": i,
                "drift": drift_masked,
                "occupancy": occupancy,
                "support": support,
                "occ_tol": occ_tol,
                "budget_used": used,
                "crossings": crossings,
            }
        )

    return results


def plot_policy_overlay_constrained(results):
    if len(results) == 0:
        print("No successful LP results to plot.")
        return

    # Font sizes
    title_fs = 20
    label_fs = 17
    tick_fs = 14
    legend_fs = 13

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    N = results[0]["model"].N

    for out in results:
        i = out["i"]
        a_exp = out["a_exp"]

        ax.plot(
            i[1:N],
            a_exp[1:N],
            lw=2,
            label=rf"$\beta={out['beta']:.3g}$",
        )

    a_init = results[0]["model"].a_init
    ax.axhline(a_init, ls="--", color="k", label=r"Baseline $a^*$")

    ax.set_xlim(1, 100)
    ax.set_ylim(0.38, 1.02)

    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_ylabel(
        r"Expected optimal action $\mathbb{E}[\pi \mid i]$",
        fontsize=label_fs,
    )
    ax.set_title(
        "Constrained model: policy comparison",
        fontsize=title_fs,
        fontweight="bold",
        pad=10,
    )

    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(fontsize=legend_fs, frameon=True)

    plt.tight_layout()
    plt.show()


def plot_drift_overlay_constrained(results, ylim=None):
    if len(results) == 0:
        print("No successful LP results to plot.")
        return

    # Font sizes
    title_fs = 20
    label_fs = 17
    tick_fs = 14
    legend_fs = 13

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    N = results[0]["model"].N

    for out in results:
        i = out["i"]
        drift = out["drift"]

        ax.plot(
            i[1:N],
            drift[1:N],
            lw=2,
            label=rf"$\beta={out['beta']:.3g}$",
        )

    ax.axhline(0.0, ls="--", color="k")

    ax.set_xlim(1, 100)

    ax.set_ylim(-0.05, 0.5)
    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_ylabel(
        r"Per-capita drift $(\lambda-\mu)/i$",
        fontsize=label_fs,
    )
    ax.set_title(
        "Constrained model: drift comparison",
        fontsize=title_fs,
        fontweight="bold",
        pad=10,
    )

    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(fontsize=legend_fs, frameon=True)

    plt.tight_layout()
    plt.show()


# Beta sweep heatmaps


def beta_sweep_constrained(
    params,
    beta_values,
    eps=0.05,
    occ_rel_tol=1e-8,
    occ_abs_tol=1e-12,
):
    N = int(params["N"])

    policy_mat = []
    drift_mat = []
    occupancy_mat = []

    widths = []
    lower_bounds = []
    upper_bounds = []
    first_crossings = []
    budget_used = []

    for beta in beta_values:

        model, res, policy, a_exp = solve_constrained(params, beta)

        if a_exp is None:

            policy_mat.append(np.full(N - 1, np.nan))

            drift_mat.append(np.full(N - 1, np.nan))

            occupancy_mat.append(np.full(N - 1, np.nan))

            widths.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
            first_crossings.append(np.nan)
            budget_used.append(np.nan)

            continue

        # Raw expected action and biological drift

        i, drift = per_capita_drift_constrained(model, a_exp)

        # Occupancy support

        (
            a_masked,
            drift_masked,
            occupancy,
            support,
            occ_tol,
        ) = mask_constrained_solution(
            model,
            res,
            a_exp,
            drift,
            rel_tol=occ_rel_tol,
            abs_tol=occ_abs_tol,
        )

        # Derived quantities use ONLY supported states

        crossings = find_zero_crossings(i, drift_masked)

        lo, hi = dormant_bounds_constrained(i, drift_masked, eps=eps)

        width = dormant_width_constrained(i, drift_masked, eps=eps)

        # Store only interior states 1,...,N-1

        policy_mat.append(a_masked[1 : model.N])

        drift_mat.append(drift_masked[1 : model.N])

        occupancy_mat.append(occupancy[1 : model.N])

        widths.append(width)

        lower_bounds.append(lo)
        upper_bounds.append(hi)

        first_crossings.append(crossings[0] if len(crossings) > 0 else np.nan)

        budget_used.append(budget_used_constrained(model, res))

    return {
        "beta_values": np.array(beta_values),
        "policy_mat": np.array(policy_mat),
        "drift_mat": np.array(drift_mat),
        "occupancy_mat": np.array(occupancy_mat),
        "widths": np.array(widths),
        "lower_bounds": np.array(lower_bounds),
        "upper_bounds": np.array(upper_bounds),
        "first_crossings": np.array(first_crossings),
        "budget_used": np.array(budget_used),
    }


def plot_beta_heatmaps_constrained(params, beta_values, eps=0.05, x_max=100):
    sweep = beta_sweep_constrained(params, beta_values, eps=eps)

    policy_view, x_plot, x_max = crop_state_matrix(sweep["policy_mat"], x_max=x_max)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        policy_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, beta_values.min(), beta_values.max()],
    )

    plt.colorbar(im, ax=ax, label=r"Expected action $\mathbb{E}[a\mid i]$")

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Constraint budget $\beta$")
    ax.set_title(r"Constrained model: policy sweep over $\beta$", fontweight="bold")

    plt.tight_layout()
    plt.show()

    drift_view, x_plot, x_max = crop_state_matrix(sweep["drift_mat"], x_max=x_max)
    vmin, vmax = symmetric_drift_limits(drift_view)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        drift_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, beta_values.min(), beta_values.max()],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, label=r"Per-capita drift $(\lambda-\mu)/i$")

    #     x_plot,
    #     beta_values,
    #     drift_view,
    #     levels=[0.0],
    #     colors="k",
    #     linewidths=1.2,
    # )

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Constraint budget $\beta$")
    ax.set_title(r"Constrained model: drift sweep over $\beta$", fontweight="bold")

    plt.tight_layout()
    plt.show()

    return sweep


# Delta0 sweep at fixed beta


def delta0_sweep_constrained(
    params,
    delta0_values,
    beta,
    eps=0.05,
    occ_rel_tol=1e-8,
    occ_abs_tol=1e-12,
):
    N = int(params["N"])

    policy_mat = []
    drift_mat = []
    occupancy_mat = []

    widths = []
    first_crossings = []

    for delta0 in delta0_values:

        p = params.copy()
        p["delta0"] = float(delta0)

        model, res, policy, a_exp = solve_constrained(p, beta)

        if a_exp is None:
            policy_mat.append(np.full(N - 1, np.nan))
            drift_mat.append(np.full(N - 1, np.nan))
            occupancy_mat.append(np.full(N - 1, np.nan))

            widths.append(np.nan)
            first_crossings.append(np.nan)

            continue

        # Raw biological drift
        i, drift = per_capita_drift_constrained(model, a_exp)

        # Occupancy-supported interpretation
        (
            a_masked,
            drift_masked,
            occupancy,
            support,
            occ_tol,
        ) = mask_constrained_solution(
            model,
            res,
            a_exp,
            drift,
            rel_tol=occ_rel_tol,
            abs_tol=occ_abs_tol,
        )

        crossings = find_zero_crossings(i, drift_masked)

        policy_mat.append(a_masked[1 : model.N])

        drift_mat.append(drift_masked[1 : model.N])

        occupancy_mat.append(occupancy[1 : model.N])

        widths.append(dormant_width_constrained(i, drift_masked, eps=eps))

        first_crossings.append(crossings[0] if len(crossings) > 0 else np.nan)

    return {
        "delta0_values": np.array(delta0_values),
        "policy_mat": np.array(policy_mat),
        "drift_mat": np.array(drift_mat),
        "occupancy_mat": np.array(occupancy_mat),
        "widths": np.array(widths),
        "first_crossings": np.array(first_crossings),
        "beta": beta,
    }


def plot_delta0_drift_heatmap_constrained(
    params, delta0_values, beta, eps=0.05, x_max=100
):
    sweep = delta0_sweep_constrained(params, delta0_values, beta=beta, eps=eps)

    drift_view, x_plot, x_max = crop_state_matrix(sweep["drift_mat"], x_max=x_max)
    vmin, vmax = symmetric_drift_limits(drift_view)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        drift_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, delta0_values.min(), delta0_values.max()],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, label=r"Per-capita drift $(\lambda-\mu)/i$")

    ax.contour(
        x_plot,
        delta0_values,
        drift_view,
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )

    # Feasibility boundary: zero drift attainable only if delta0 <= r.
    ax.axhline(
        params["r"],
        ls=":",
        color="k",
        lw=2,
        label=r"$\delta_0=r$",
    )

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"$\delta_0$")
    ax.set_title(
        rf"Constrained model: drift sweep over $\delta_0$, $\beta={beta}$",
        fontweight="bold",
    )
    ax.legend()

    plt.tight_layout()
    plt.show()

    return sweep


# Dormancy metrics vs beta


def plot_dormancy_metrics_constrained(sweep, eps=0.05):
    beta_values = sweep["beta_values"]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(beta_values, sweep["widths"], marker="o", lw=2)
    ax.set_xlabel(r"Constraint budget $\beta$")
    ax.set_ylabel(rf"Number of states with $|(\lambda-\mu)/i|\leq {eps}$")
    ax.set_title("Constrained model: dormant interval width", fontweight="bold")
    add_grid(ax)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(beta_values, sweep["first_crossings"], marker="o", lw=2)
    ax.set_xlabel(r"Constraint budget $\beta$")
    ax.set_ylabel("First drift zero-crossing")
    ax.set_title("Constrained model: regulating state vs budget", fontweight="bold")
    add_grid(ax)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(beta_values, sweep["lower_bounds"], marker="o", lw=2, label="lower bound")
    ax.plot(beta_values, sweep["upper_bounds"], marker="o", lw=2, label="upper bound")
    ax.set_xlabel(r"Constraint budget $\beta$")
    ax.set_ylabel("Dormant interval bounds")
    ax.set_title("Constrained model: near-zero-drift region", fontweight="bold")
    add_grid(ax)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(beta_values, sweep["budget_used"], marker="o", lw=2, label="used")
    ax.plot(beta_values, beta_values, ls="--", color="k", label="available")
    ax.set_xlabel(r"Constraint budget $\beta$")
    ax.set_ylabel("Budget used")
    ax.set_title("Constraint usage", fontweight="bold")
    add_grid(ax)
    ax.legend()
    plt.tight_layout()
    plt.show()


# Stochastic trajectories from extracted deterministic policy


def simulate_birth_death_constrained(model, a_exp, i0=1, T=300, seed=None):
    rng = np.random.default_rng(seed)

    t = 0.0
    i = int(i0)

    times = [t]
    states = [i]

    while t < T and 0 < i < model.N:
        a = float(a_exp[i])

        lam = model.lam(i, a)
        mu = model.mu(i, a)
        rate = lam + mu

        if rate <= 0:
            break

        t += rng.exponential(1.0 / rate)

        if t > T:
            break

        if rng.random() < lam / rate:
            i += 1
        else:
            i -= 1

        times.append(t)
        states.append(i)

    return np.array(times), np.array(states)


def plot_trajectories_constrained(results, i0=1, T=300, n_paths=8):
    for out in results:
        model = out["model"]
        a_exp = out["a_exp"]

        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

        for k in range(n_paths):
            t, x = simulate_birth_death_constrained(
                model,
                a_exp,
                i0=i0,
                T=T,
                seed=100 + 19 * k,
            )

            ax.step(t, x, where="post", alpha=0.8, lw=1.5)

        crossings = out["crossings"]

        if len(crossings) > 0:
            ax.axhline(crossings[0], ls=":", color="k", label="first zero crossing")

        ax.set_xlabel("Time")
        ax.set_ylabel("Population state $i(t)$")
        ax.set_title(
            rf"Constrained trajectories, $\beta={out['beta']:.3g}$", fontweight="bold"
        )
        add_grid(ax)
        ax.legend()

        plt.tight_layout()
        plt.show()


BASE_CONSTRAINED = dict(
    N=200,
    r=0.30,
    delta=0.15,
    delta0=0.05,  # extrinsic death pressure
    alpha=0.51,
    kappa=2.0,
    c3=1.0,
    penalty=1e4,
    n_a=101,
    initial_state=10,
)


beta_examples = [0.01, 0.38, 1.1, 2, 3, 4, 5.0]

results_constrained = solve_beta_grid(BASE_CONSTRAINED, beta_examples)

plot_policy_overlay_constrained(results_constrained)
plot_drift_overlay_constrained(results_constrained, ylim=(-0.05, 0.3))

beta_values = np.linspace(0.005, 5.0, 20)
sweep_constrained = plot_beta_heatmaps_constrained(
    BASE_CONSTRAINED,
    beta_values,
    eps=0.05,
    x_max=100,
)

plot_dormancy_metrics_constrained(sweep_constrained, eps=0.05)

# # Delta0 drift sweep at a representative fixed beta

#     beta=delta0_sweep_beta,
#     eps=0.05,
#     x_max=100,
# )

import pandas as pd

# N-truncation diagnostics


def discounted_state_occupancy_constrained(model, res):
    if not res.success:
        return None, None

    n_actions = get_n_actions(model)
    Lambda = get_Lambda(model)

    x = res.x.reshape((model.N + 1, n_actions))
    occ_action = x / (model.alpha + Lambda)
    occ_state = occ_action.sum(axis=1)

    return occ_state, occ_action


def occupancy_tail_diagnostics(model, res, tail_m=10, tail_frac_start=0.90):
    occ_state, _ = discounted_state_occupancy_constrained(model, res)

    if occ_state is None:
        return {
            "occ_total": np.nan,
            "occ_mean_state": np.nan,
            "boundary_occ_frac": np.nan,
            "last_m_occ_frac": np.nan,
            "upper_occ_frac": np.nan,
        }

    total = np.sum(occ_state)

    if total <= 0 or not np.isfinite(total):
        return {
            "occ_total": np.nan,
            "occ_mean_state": np.nan,
            "boundary_occ_frac": np.nan,
            "last_m_occ_frac": np.nan,
            "upper_occ_frac": np.nan,
        }

    states = np.arange(model.N + 1)

    last_m_start = max(0, model.N - int(tail_m) + 1)
    upper_start = int(np.ceil(tail_frac_start * model.N))

    return {
        "occ_total": total,
        "occ_mean_state": np.sum(states * occ_state) / total,
        "boundary_occ_frac": occ_state[model.N] / total,
        "last_m_occ_frac": np.sum(occ_state[last_m_start : model.N + 1]) / total,
        "upper_occ_frac": np.sum(occ_state[upper_start : model.N + 1]) / total,
    }


def N_truncation_test_constrained(
    params,
    beta,
    N_values=(80, 100, 120, 160, 200),
    eps=0.05,
    x_compare=75,
    tail_m=10,
    tail_frac_start=0.90,
):
    N_values = sorted([int(N) for N in N_values])
    x_compare = int(min(x_compare, min(N_values) - 1))

    raw_results = []

    for N in N_values:
        p = params.copy()
        p["N"] = int(N)

        model, res, policy, a_exp = solve_constrained(p, beta)

        if a_exp is None:
            raw_results.append(
                {
                    "N": N,
                    "params": p,
                    "model": model,
                    "res": res,
                    "policy": None,
                    "a_exp": None,
                    "i": np.arange(N + 1),
                    "drift": np.full(N + 1, np.nan),
                    "success": False,
                }
            )
            continue

        i, drift = per_capita_drift_constrained(model, a_exp)

        raw_results.append(
            {
                "N": N,
                "params": p,
                "model": model,
                "res": res,
                "policy": policy,
                "a_exp": a_exp,
                "i": i,
                "drift": drift,
                "success": True,
            }
        )

    successful = [out for out in raw_results if out["success"]]

    if len(successful) == 0:
        print("No successful LP solves in N-truncation test.")
        return {
            "beta": beta,
            "N_values": np.array(N_values),
            "x_compare": x_compare,
            "states_common": np.arange(1, x_compare + 1),
            "results": raw_results,
            "summary": pd.DataFrame(),
        }

    # Use largest successful N as reference.
    ref = successful[-1]
    states_common = np.arange(1, x_compare + 1)

    ref_policy = ref["a_exp"][states_common]
    ref_drift = ref["drift"][states_common]

    rows = []

    for out in raw_results:
        N = out["N"]
        model = out["model"]
        res = out["res"]

        row = {
            "beta": beta,
            "N": N,
            "success": out["success"],
        }

        if not out["success"]:
            row.update(
                {
                    "budget_used": np.nan,
                    "budget_frac": np.nan,
                    "first_crossing": np.nan,
                    "dormant_width_full": np.nan,
                    "dormant_width_common": np.nan,
                    "policy_Linf_vs_ref": np.nan,
                    "policy_RMSE_vs_ref": np.nan,
                    "drift_Linf_vs_ref": np.nan,
                    "drift_RMSE_vs_ref": np.nan,
                    "occ_total": np.nan,
                    "occ_mean_state": np.nan,
                    "boundary_occ_frac": np.nan,
                    "last_m_occ_frac": np.nan,
                    "upper_occ_frac": np.nan,
                }
            )
            rows.append(row)
            continue

        a_common = out["a_exp"][states_common]
        d_common = out["drift"][states_common]

        policy_diff = a_common - ref_policy
        drift_diff = d_common - ref_drift

        crossings = find_zero_crossings(out["i"], out["drift"])

        budget_used = budget_used_constrained(model, res)
        tail_diag = occupancy_tail_diagnostics(
            model,
            res,
            tail_m=tail_m,
            tail_frac_start=tail_frac_start,
        )

        row.update(
            {
                "budget_used": budget_used,
                "budget_frac": budget_used / beta if beta > 0 else np.nan,
                "first_crossing": crossings[0] if len(crossings) > 0 else np.nan,
                "dormant_width_full": dormant_width_constrained(
                    out["i"], out["drift"], eps=eps
                ),
                "dormant_width_common": int(np.sum(np.abs(d_common) <= eps)),
                "policy_Linf_vs_ref": np.nanmax(np.abs(policy_diff)),
                "policy_RMSE_vs_ref": np.sqrt(np.nanmean(policy_diff**2)),
                "drift_Linf_vs_ref": np.nanmax(np.abs(drift_diff)),
                "drift_RMSE_vs_ref": np.sqrt(np.nanmean(drift_diff**2)),
            }
        )

        row.update(tail_diag)
        rows.append(row)

    summary = pd.DataFrame(rows)

    return {
        "beta": beta,
        "N_values": np.array(N_values),
        "x_compare": x_compare,
        "states_common": states_common,
        "results": raw_results,
        "reference_N": ref["N"],
        "summary": summary,
    }


def plot_N_truncation_overlay_constrained(
    trunc,
    kind="drift",
    x_max=None,
    ylim=None,
):
    results = [out for out in trunc["results"] if out["success"]]

    if len(results) == 0:
        print("No successful LP results to plot.")
        return

    if x_max is None:
        x_max = trunc["x_compare"]

    x_max = int(min(x_max, trunc["x_compare"]))
    states = np.arange(1, x_max + 1)

    title_fs = 20
    label_fs = 17
    tick_fs = 14
    legend_fs = 12

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    for out in results:
        N = out["N"]

        if kind == "policy":
            y = out["a_exp"][states]
            ax.plot(states, y, lw=2, label=rf"$N={N}$")

        elif kind == "drift":
            y = out["drift"][states]
            ax.plot(states, y, lw=2, label=rf"$N={N}$")

        else:
            raise ValueError("kind must be either 'policy' or 'drift'.")

    if kind == "policy":
        a_init = results[0]["model"].a_init
        ax.axhline(a_init, ls="--", color="k", label=r"Baseline $a^*$")
        ax.set_ylabel(
            r"Expected optimal action $\mathbb{E}[\pi \mid i]$",
            fontsize=label_fs,
        )
        ax.set_ylim(-0.02, 1.02)
        title = rf"N-truncation test: policy, $\beta={trunc['beta']}$"

    else:
        ax.axhline(0.0, ls="--", color="k")
        ax.set_ylabel(
            r"Per-capita drift $(\lambda-\mu)/i$",
            fontsize=label_fs,
        )
        title = rf"N-truncation test: drift, $\beta={trunc['beta']}$"

    ax.set_xlim(1, x_max)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=10)

    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(fontsize=legend_fs, frameon=True)

    plt.tight_layout()
    plt.show()


def plot_N_truncation_diagnostics_constrained(trunc):
    df = trunc["summary"].copy()
    df = df[df["success"] == True]

    if len(df) == 0:
        print("No successful LP results to plot.")
        return

    title_fs = 18
    label_fs = 15
    tick_fs = 13
    legend_fs = 12

    # Policy/drift errors versus largest-N reference
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)

    ax.plot(df["N"], df["policy_Linf_vs_ref"], marker="o", lw=2, label="policy")
    ax.plot(df["N"], df["drift_Linf_vs_ref"], marker="o", lw=2, label="drift")

    ax.set_xlabel("Truncation level $N$", fontsize=label_fs)
    ax.set_ylabel(r"$L^\infty$ error vs largest $N$", fontsize=label_fs)
    ax.set_title("Interior solution error", fontsize=title_fs, fontweight="bold")
    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(fontsize=legend_fs)

    plt.tight_layout()
    plt.show()

    # Occupancy near artificial boundary
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)

    ax.plot(df["N"], df["boundary_occ_frac"], marker="o", lw=2, label=r"state $N$")
    ax.plot(df["N"], df["last_m_occ_frac"], marker="o", lw=2, label="last states")
    ax.plot(df["N"], df["upper_occ_frac"], marker="o", lw=2, label="upper tail")

    ax.set_xlabel("Truncation level $N$", fontsize=label_fs)
    ax.set_ylabel("Fraction of discounted occupancy", fontsize=label_fs)
    ax.set_title(
        "Upper-boundary occupancy diagnostic", fontsize=title_fs, fontweight="bold"
    )
    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(fontsize=legend_fs)

    plt.tight_layout()
    plt.show()

    # First zero crossing and dormant width
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)

    ax.plot(
        df["N"], df["first_crossing"], marker="o", lw=2, label="first zero crossing"
    )
    ax.plot(
        df["N"],
        df["dormant_width_common"],
        marker="o",
        lw=2,
        label="common dormant width",
    )

    ax.set_xlabel("Truncation level $N$", fontsize=label_fs)
    ax.set_ylabel("State / number of states", fontsize=label_fs)
    ax.set_title(
        "Dormancy metrics under N-truncation", fontsize=title_fs, fontweight="bold"
    )
    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(fontsize=legend_fs)

    plt.tight_layout()
    plt.show()


def N_truncation_beta_grid_constrained(
    params,
    beta_values,
    N_values=(80, 100, 120, 160, 200),
    eps=0.05,
    x_compare=75,
    tail_m=10,
    tail_frac_start=0.90,
):
    tests = {}
    all_rows = []

    for beta in beta_values:
        trunc = N_truncation_test_constrained(
            params,
            beta=beta,
            N_values=N_values,
            eps=eps,
            x_compare=x_compare,
            tail_m=tail_m,
            tail_frac_start=tail_frac_start,
        )

        tests[beta] = trunc

        df = trunc["summary"].copy()
        all_rows.append(df)

    summary_all = pd.concat(all_rows, ignore_index=True)

    return tests, summary_all


N_values = [100, 200, 400, 500, 1000, 1500]

beta_test = 0.32

trunc = N_truncation_test_constrained(
    BASE_CONSTRAINED,
    beta=beta_test,
    N_values=N_values,
    eps=0.05,
    x_compare=75,
    tail_m=10,
    tail_frac_start=0.90,
)

trunc["summary"]
plot_N_truncation_overlay_constrained(
    trunc,
    kind="policy",
    x_max=75,
)

plot_N_truncation_overlay_constrained(
    trunc,
    kind="drift",
    x_max=75,
    ylim=(-0.05, 0.50),
)

plot_N_truncation_diagnostics_constrained(trunc)
# %% [markdown]
# ## Constrained quadratic-reward model

# %%


def get_n_actions_v2(model):
    return int(getattr(model, "n_actions", getattr(model, "n_a", len(model.actions))))


def get_Lambda_v2(model):
    if hasattr(model, "Lambda"):
        return model.Lambda

    if hasattr(model, "global_uniformization_rate"):
        return model.global_uniformization_rate()

    lam_attr = getattr(model, "lam", None)
    if lam_attr is not None and not callable(lam_attr):
        return lam_attr

    raise AttributeError("Could not determine LP uniformization rate Lambda.")


def add_grid(ax):
    ax.grid(True, alpha=0.3)


def symmetric_drift_limits(mat):
    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def crop_state_matrix(mat, x_max=100):
    x_max = min(int(x_max), mat.shape[1])
    mat_view = mat[:, :x_max]
    x_plot = np.arange(1, x_max + 1)
    return mat_view, x_plot, x_max


def solve_constrained_v2(params, beta):
    model = TumorActorLP_v2(params, beta)
    res = model.solve()

    if not res.success:
        print(f"LP_v2 failed for beta={beta}: {res.message}")
        return model, res, None, None

    mixed_policy = model.extract_policy(res.x)

    if hasattr(model, "expected_action_policy"):
        a_exp = model.expected_action_policy(res.x)
    else:
        a_exp = expected_action_from_policy_v2(
            mixed_policy,
            model.N,
            default_a=model.a_init,
        )

    return model, res, mixed_policy, a_exp


def expected_action_from_policy_v2(policy, N, default_a=0.5):
    a_exp = np.full(N + 1, default_a, dtype=float)

    for i in range(N + 1):
        if i in policy:
            a_exp[i] = sum(float(a) * float(p) for a, p in policy[i])

    a_exp[0] = 1.0
    return np.clip(a_exp, 0.0, 1.0)


def per_capita_drift_constrained_v2(model, a_exp):
    i = np.arange(model.N + 1)
    drift = np.full(model.N + 1, np.nan)

    for k in range(1, model.N + 1):
        a = float(a_exp[k])
        drift[k] = (model.lam(k, a) - model.mu(k, a)) / k

    return i, drift


def reward_profile_v2(model, a_exp):
    i = np.arange(model.N + 1)
    rew = np.full(model.N + 1, np.nan)

    for k in range(1, model.N + 1):
        rew[k] = model.reward(k, float(a_exp[k]))

    return i, rew


def budget_used_constrained_v2(model, res):
    if not res.success:
        return np.nan

    if hasattr(model, "discounted_regulatory_cost"):
        return model.discounted_regulatory_cost(res.x)

    n_actions = get_n_actions_v2(model)
    Lambda = get_Lambda_v2(model)

    x = res.x.reshape((model.N + 1, n_actions))
    total = 0.0

    for i in range(model.N + 1):
        for a_idx, a in enumerate(model.actions):
            if hasattr(model, "regulatory_cost"):
                cost = model.regulatory_cost(i, a)
            else:
                cost = model.kappa * i * (a - model.a_star) ** 2

            total += x[i, a_idx] * cost / (model.alpha + Lambda)

    return float(total)


def reward_value_constrained_v2(model, res):
    if not res.success:
        return np.nan

    if hasattr(model, "discounted_reward_value"):
        return model.discounted_reward_value(res.x)

    n_actions = get_n_actions_v2(model)
    Lambda = get_Lambda_v2(model)

    x = res.x.reshape((model.N + 1, n_actions))
    total = 0.0

    for i in range(model.N + 1):
        for a_idx, a in enumerate(model.actions):
            total += x[i, a_idx] * model.reward(i, a) / (model.alpha + Lambda)

    return float(total)


def find_zero_crossings_v2(i, drift):
    mask = i > 0
    ii = i[mask]
    dd = drift[mask]

    crossings = []

    for j in range(len(ii) - 1):
        if np.isnan(dd[j]) or np.isnan(dd[j + 1]):
            continue

        if dd[j] == 0:
            crossings.append(ii[j])
        elif dd[j] * dd[j + 1] < 0:
            crossings.append(0.5 * (ii[j] + ii[j + 1]))

    return crossings


def dormant_width_constrained_v2(i, drift, eps=0.05):
    mask = (i > 0) & np.isfinite(drift)
    return int(np.sum(np.abs(drift[mask]) <= eps))


def dormant_bounds_constrained_v2(i, drift, eps=0.05):
    mask = (i > 0) & np.isfinite(drift) & (np.abs(drift) <= eps)
    states = i[mask]

    if len(states) == 0:
        return np.nan, np.nan

    return states.min(), states.max()


# Single beta plots


def plot_policy_constrained_v2(params, beta, x_max=100):
    model, res, policy, a_exp = solve_constrained_v2(params, beta)

    if a_exp is None:
        return None

    i_full, drift = per_capita_drift_constrained_v2(model, a_exp)

    (
        a_masked,
        drift_masked,
        occupancy,
        support,
        occ_tol,
    ) = mask_constrained_solution(
        model,
        res,
        a_exp,
        drift,
    )

    x_max = min(int(x_max), model.N - 1)
    i = np.arange(1, x_max + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(i, a_masked[1 : x_max + 1], lw=2)

    ax.axhline(model.a_init, ls="--", color="k", label=r"Baseline $a^*$")

    ax.set_xlim(1, x_max)
    ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Expected optimal action $\mathbb{E}[\pi\mid i]$")

    ax.set_title(rf"Constrained Quadratic: policy, $\beta={beta}$", fontweight="bold")

    add_grid(ax)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return model, res, policy, a_exp


def plot_drift_constrained_v2(params, beta, x_max=100, ylim=None):
    model, res, policy, a_exp = solve_constrained_v2(params, beta)

    if a_exp is None:
        return None

    i, drift = per_capita_drift_constrained_v2(model, a_exp)

    (
        a_masked,
        drift_masked,
        occupancy,
        support,
        occ_tol,
    ) = mask_constrained_solution(
        model,
        res,
        a_exp,
        drift,
    )

    crossings = find_zero_crossings_v2(i, drift_masked)

    x_max = min(int(x_max), model.N - 1)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(i[1 : x_max + 1], drift_masked[1 : x_max + 1], lw=2)

    ax.axhline(0.0, ls="--", color="k")

    for x in crossings:
        if 1 <= x <= x_max:
            ax.axvline(x, ls=":", color="k")

    ax.set_xlim(1, x_max)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Per-capita drift $(\lambda-\mu)/i$")

    ax.set_title(rf"Constrained Quadratic: drift, $\beta={beta}$", fontweight="bold")

    add_grid(ax)

    plt.tight_layout()
    plt.show()

    return model, res, policy, a_exp


def plot_reward_profile_constrained_v2(params, beta, x_max=100):

    model, res, policy, a_exp = solve_constrained_v2(params, beta)

    if a_exp is None:
        return None

    i, drift = per_capita_drift_constrained_v2(model, a_exp)

    _, reward_inst = reward_profile_v2(model, a_exp)

    (
        a_masked,
        drift_masked,
        occupancy,
        support,
        occ_tol,
    ) = mask_constrained_solution(
        model,
        res,
        a_exp,
        drift,
    )

    reward_masked = np.asarray(reward_inst, dtype=float).copy()

    reward_masked[~support] = np.nan

    x_max = min(int(x_max), model.N - 1)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(i[1 : x_max + 1], reward_masked[1 : x_max + 1], lw=2)

    ax.axhline(0.0, ls="--", color="k")

    ax.set_xlim(1, x_max)

    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Instantaneous reward $r_e(i,a(i))$")

    ax.set_title(rf"Constrained Quadratic: reward, $\beta={beta}$", fontweight="bold")

    add_grid(ax)

    plt.tight_layout()
    plt.show()

    return model, res, policy, a_exp


# Overlay plots for multiple beta


def solve_beta_grid_v2(
    params,
    beta_values,
    verbose=True,
    occ_rel_tol=1e-8,
    occ_abs_tol=1e-12,
):
    results = []

    for beta in beta_values:

        model, res, policy, a_exp = solve_constrained_v2(params, beta)

        if a_exp is None:
            continue

        # Raw quantities

        i, drift = per_capita_drift_constrained_v2(model, a_exp)

        _, reward_inst = reward_profile_v2(model, a_exp)

        # Occupancy-supported interpretation

        (
            a_masked,
            drift_masked,
            occupancy,
            support,
            occ_tol,
        ) = mask_constrained_solution(
            model,
            res,
            a_exp,
            drift,
            rel_tol=occ_rel_tol,
            abs_tol=occ_abs_tol,
        )

        reward_masked = np.asarray(reward_inst, dtype=float).copy()

        reward_masked[~support] = np.nan

        # Derived quantities

        crossings = find_zero_crossings_v2(i, drift_masked)

        used = budget_used_constrained_v2(model, res)

        value = reward_value_constrained_v2(model, res)

        # Store

        results.append(
            {
                "beta": beta,
                "model": model,
                "res": res,
                "policy": policy,
                # Keep raw expected action for simulation if needed
                "a_exp_raw": a_exp,
                # Occupancy-supported quantities for figures/analysis
                "a_exp": a_masked,
                "i": i,
                "drift": drift_masked,
                "reward_inst": reward_masked,
                "occupancy": occupancy,
                "support": support,
                "occ_tol": occ_tol,
                "budget_used": used,
                "reward_value": value,
                "crossings": crossings,
                "preferred_size_a_star": model.preferred_size(model.a_star),
                "fallback": getattr(res, "used_unconstrained_fallback", False),
            }
        )

        if verbose:
            print(
                f"beta={beta:.4g}, "
                f"budget_used={used:.4g}, "
                f"reward_value={value:.4g}, "
                f"supported_states={np.sum(support)}, "
                f"fallback="
                f"{getattr(res, 'used_unconstrained_fallback', False)}"
            )

    return results


def plot_policy_overlay_constrained_v2(results, x_max=100, ylim=(-0.02, 1.02)):
    if len(results) == 0:
        print("No successful LP_v2 results to plot.")
        return

    title_fs = 20
    label_fs = 17
    tick_fs = 14
    legend_fs = 13

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    N = results[0]["model"].N
    x_max = min(int(x_max), N - 1)

    for out in results:
        i = out["i"]
        a_exp = out["a_exp"]

        ax.plot(
            i[1 : x_max + 1],
            a_exp[1 : x_max + 1],
            lw=2,
            label=rf"$\beta={out['beta']:.3g}$",
        )

    a_init = results[0]["model"].a_init
    ax.axhline(a_init, ls="--", color="k", label=r"Baseline $a^*$")

    ax.set_xlim(1, x_max)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_ylabel(
        r"Expected optimal action $\mathbb{E}[\pi \mid i]$",
        fontsize=label_fs,
    )
    ax.set_title(
        "Constrained Quadratic: policy comparison",
        fontsize=title_fs,
        fontweight="bold",
        pad=10,
    )

    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(
        fontsize=legend_fs,
        frameon=True,
        loc="center right",
    )

    plt.tight_layout()
    plt.show()


def plot_drift_overlay_constrained_v2(results, x_max=100, ylim=None):
    if len(results) == 0:
        print("No successful LP_v2 results to plot.")
        return

    title_fs = 20
    label_fs = 17
    tick_fs = 14
    legend_fs = 13

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    N = results[0]["model"].N
    x_max = min(int(x_max), N - 1)

    for out in results:
        i = out["i"]
        drift = out["drift"]

        ax.plot(
            i[1 : x_max + 1],
            drift[1 : x_max + 1],
            lw=2,
            label=rf"$\beta={out['beta']:.3g}$",
        )

    ax.axhline(0.0, ls="--", color="k")

    ax.set_xlim(1, x_max)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_ylabel(
        r"Per-capita drift $(\lambda-\mu)/i$",
        fontsize=label_fs,
    )
    ax.set_title(
        "Constrained Quadratic: drift comparison",
        fontsize=title_fs,
        fontweight="bold",
        pad=10,
    )

    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(
        fontsize=legend_fs,
        frameon=True,
        loc="center right",
    )

    plt.tight_layout()
    plt.show()


def plot_reward_overlay_constrained_v2(results, x_max=100, ylim=None):
    if len(results) == 0:
        print("No successful LP_v2 results to plot.")
        return

    title_fs = 20
    label_fs = 17
    tick_fs = 14
    legend_fs = 13

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    N = results[0]["model"].N
    x_max = min(int(x_max), N - 1)

    for out in results:
        i = out["i"]
        reward_inst = out["reward_inst"]

        ax.plot(
            i[1 : x_max + 1],
            reward_inst[1 : x_max + 1],
            lw=2,
            label=rf"$\beta={out['beta']:.3g}$",
        )

    ax.axhline(0.0, ls="--", color="k")

    pref = results[0].get("preferred_size_a_star", np.nan)
    if np.isfinite(pref) and 1 <= pref <= x_max:
        ax.axvline(pref, ls=":", color="k", label=rf"$i_{{pref}}\approx {pref:.1f}$")

    ax.set_xlim(1, x_max)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_ylabel(r"Instantaneous reward $r_e(i,a(i))$", fontsize=label_fs)
    ax.set_title(
        "Constrained Quadratic: reward comparison",
        fontsize=title_fs,
        fontweight="bold",
        pad=10,
    )

    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(fontsize=legend_fs, frameon=True)

    plt.tight_layout()
    plt.show()


# Beta sweep heatmaps


def beta_sweep_constrained_v2(
    params,
    beta_values,
    eps=0.05,
    verbose=False,
    occ_rel_tol=1e-8,
    occ_abs_tol=1e-12,
):
    N = int(params["N"])

    policy_mat = []
    drift_mat = []
    reward_mat = []
    occupancy_mat = []

    widths = []
    lower_bounds = []
    upper_bounds = []
    first_crossings = []

    budget_used = []
    reward_values = []
    fallback_flags = []

    for beta in beta_values:

        model, res, policy, a_exp = solve_constrained_v2(params, beta)

        if a_exp is None:

            policy_mat.append(np.full(N - 1, np.nan))

            drift_mat.append(np.full(N - 1, np.nan))

            reward_mat.append(np.full(N - 1, np.nan))

            occupancy_mat.append(np.full(N - 1, np.nan))

            widths.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
            first_crossings.append(np.nan)

            budget_used.append(np.nan)
            reward_values.append(np.nan)
            fallback_flags.append(False)

            continue

        # Raw quantities

        i, drift = per_capita_drift_constrained_v2(model, a_exp)

        _, reward_inst = reward_profile_v2(model, a_exp)

        # Occupancy support

        (
            a_masked,
            drift_masked,
            occupancy,
            support,
            occ_tol,
        ) = mask_constrained_solution(
            model,
            res,
            a_exp,
            drift,
            rel_tol=occ_rel_tol,
            abs_tol=occ_abs_tol,
        )

        # Reward profile is also only scientifically
        # interpretable where the optimized policy is supported.
        reward_masked = np.asarray(reward_inst, dtype=float).copy()

        reward_masked[~support] = np.nan

        # Derived drift quantities

        crossings = find_zero_crossings_v2(i, drift_masked)

        lo, hi = dormant_bounds_constrained_v2(i, drift_masked, eps=eps)

        width = dormant_width_constrained_v2(i, drift_masked, eps=eps)

        used = budget_used_constrained_v2(model, res)

        value = reward_value_constrained_v2(model, res)

        # Store interior supported quantities

        policy_mat.append(a_masked[1 : model.N])

        drift_mat.append(drift_masked[1 : model.N])

        reward_mat.append(reward_masked[1 : model.N])

        occupancy_mat.append(occupancy[1 : model.N])

        widths.append(width)
        lower_bounds.append(lo)
        upper_bounds.append(hi)

        first_crossings.append(crossings[0] if len(crossings) > 0 else np.nan)

        budget_used.append(used)
        reward_values.append(value)

        fallback_flags.append(getattr(res, "used_unconstrained_fallback", False))

        if verbose:

            n_supported = int(np.sum(support))

            print(
                f"beta={beta:.4g}, "
                f"budget_used={used:.4g}, "
                f"reward_value={value:.4g}, "
                f"supported_states={n_supported}, "
                f"occ_tol={occ_tol:.3e}, "
                f"fallback="
                f"{getattr(res, 'used_unconstrained_fallback', False)}"
            )

    return {
        "beta_values": np.array(beta_values),
        "policy_mat": np.array(policy_mat),
        "drift_mat": np.array(drift_mat),
        "reward_mat": np.array(reward_mat),
        "occupancy_mat": np.array(occupancy_mat),
        "widths": np.array(widths),
        "lower_bounds": np.array(lower_bounds),
        "upper_bounds": np.array(upper_bounds),
        "first_crossings": np.array(first_crossings),
        "budget_used": np.array(budget_used),
        "reward_values": np.array(reward_values),
        "fallback_flags": np.array(fallback_flags, dtype=bool),
    }


def plot_beta_heatmaps_constrained_v2(params, beta_values, eps=0.05, x_max=100):
    sweep = beta_sweep_constrained_v2(params, beta_values, eps=eps)

    policy_view, x_plot, x_max = crop_state_matrix(sweep["policy_mat"], x_max=x_max)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        policy_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, beta_values.min(), beta_values.max()],
    )

    plt.colorbar(im, ax=ax, label=r"Expected action $\mathbb{E}[a\mid i]$")

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Constraint budget $\beta$")
    ax.set_title(r"Constrained Quadratic: policy sweep over $\beta$", fontweight="bold")

    plt.tight_layout()
    plt.show()

    drift_view, x_plot, x_max = crop_state_matrix(sweep["drift_mat"], x_max=x_max)
    vmin, vmax = symmetric_drift_limits(drift_view)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        drift_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, beta_values.min(), beta_values.max()],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, label=r"Per-capita drift $(\lambda-\mu)/i$")

    # Uncomment to draw the zero-drift contour.
    #     x_plot,
    #     beta_values,
    #     drift_view,
    #     levels=[0.0],
    #     colors="k",
    #     linewidths=1.2,
    # )

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Constraint budget $\beta$")
    ax.set_title(r"Constrained Quadratic: drift sweep over $\beta$", fontweight="bold")

    plt.tight_layout()
    plt.show()

    reward_view, x_plot, x_max = crop_state_matrix(sweep["reward_mat"], x_max=x_max)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        reward_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, beta_values.min(), beta_values.max()],
    )

    plt.colorbar(im, ax=ax, label=r"Instantaneous reward $r_e(i,a(i))$")

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"Constraint budget $\beta$")
    ax.set_title(r"Constrained Quadratic: reward sweep over $\beta$", fontweight="bold")

    plt.tight_layout()
    plt.show()

    return sweep


# Delta0 sweep at fixed beta


def delta0_sweep_constrained_v2(
    params,
    delta0_values,
    beta,
    eps=0.05,
    occ_rel_tol=1e-8,
    occ_abs_tol=1e-12,
):
    N = int(params["N"])

    policy_mat = []
    drift_mat = []
    reward_mat = []
    occupancy_mat = []

    widths = []
    first_crossings = []

    for delta0 in delta0_values:

        p = params.copy()
        p["delta0"] = float(delta0)

        model, res, policy, a_exp = solve_constrained_v2(p, beta)

        if a_exp is None:
            policy_mat.append(np.full(N - 1, np.nan))
            drift_mat.append(np.full(N - 1, np.nan))
            reward_mat.append(np.full(N - 1, np.nan))
            occupancy_mat.append(np.full(N - 1, np.nan))

            widths.append(np.nan)
            first_crossings.append(np.nan)

            continue

        # Raw quantities

        i, drift = per_capita_drift_constrained_v2(model, a_exp)

        _, reward_inst = reward_profile_v2(model, a_exp)

        # Occupancy-supported interpretation

        (
            a_masked,
            drift_masked,
            occupancy,
            support,
            occ_tol,
        ) = mask_constrained_solution(
            model,
            res,
            a_exp,
            drift,
            rel_tol=occ_rel_tol,
            abs_tol=occ_abs_tol,
        )

        reward_masked = np.asarray(reward_inst, dtype=float).copy()

        reward_masked[~support] = np.nan

        # Derived quantities

        crossings = find_zero_crossings_v2(i, drift_masked)

        width = dormant_width_constrained_v2(i, drift_masked, eps=eps)

        # Store interior supported quantities

        policy_mat.append(a_masked[1 : model.N])

        drift_mat.append(drift_masked[1 : model.N])

        reward_mat.append(reward_masked[1 : model.N])

        occupancy_mat.append(occupancy[1 : model.N])

        widths.append(width)

        first_crossings.append(crossings[0] if len(crossings) > 0 else np.nan)

    return {
        "delta0_values": np.array(delta0_values),
        "policy_mat": np.array(policy_mat),
        "drift_mat": np.array(drift_mat),
        "reward_mat": np.array(reward_mat),
        "occupancy_mat": np.array(occupancy_mat),
        "widths": np.array(widths),
        "first_crossings": np.array(first_crossings),
        "beta": beta,
    }


def plot_delta0_drift_heatmap_constrained_v2(
    params, delta0_values, beta, eps=0.05, x_max=100
):
    sweep = delta0_sweep_constrained_v2(params, delta0_values, beta=beta, eps=eps)

    drift_view, x_plot, x_max = crop_state_matrix(sweep["drift_mat"], x_max=x_max)
    vmin, vmax = symmetric_drift_limits(drift_view)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    im = ax.imshow(
        drift_view,
        aspect="auto",
        origin="lower",
        extent=[1, x_max, delta0_values.min(), delta0_values.max()],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, label=r"Per-capita drift $(\lambda-\mu)/i$")

    ax.contour(
        x_plot,
        delta0_values,
        drift_view,
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )

    ax.axhline(
        params["r"],
        ls=":",
        color="k",
        lw=2,
        label=r"$\delta_0=r$",
    )

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$")
    ax.set_ylabel(r"$\delta_0$")
    ax.set_title(
        rf"Constrained Quadratic: drift sweep over $\delta_0$, $\beta={beta}$",
        fontweight="bold",
    )
    ax.legend()

    plt.tight_layout()
    plt.show()

    return sweep


# Dormancy / budget / reward metrics vs beta


def plot_dormancy_metrics_constrained_v2(sweep, eps=0.05):
    beta_values = sweep["beta_values"]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(beta_values, sweep["widths"], marker="o", lw=2)
    ax.set_xlabel(r"Constraint budget $\beta$")
    ax.set_ylabel(rf"Number of states with $|(\lambda-\mu)/i|\leq {eps}$")
    ax.set_title("Constrained Quadratic: dormant interval width", fontweight="bold")
    add_grid(ax)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(beta_values, sweep["first_crossings"], marker="o", lw=2)
    ax.set_xlabel(r"Constraint budget $\beta$")
    ax.set_ylabel("First drift zero-crossing")
    ax.set_title("Constrained Quadratic: regulating state vs budget", fontweight="bold")
    add_grid(ax)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(beta_values, sweep["lower_bounds"], marker="o", lw=2, label="lower bound")
    ax.plot(beta_values, sweep["upper_bounds"], marker="o", lw=2, label="upper bound")
    ax.set_xlabel(r"Constraint budget $\beta$")
    ax.set_ylabel("Dormant interval bounds")
    ax.set_title("Constrained Quadratic: near-zero-drift region", fontweight="bold")
    add_grid(ax)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(beta_values, sweep["budget_used"], marker="o", lw=2, label="used")
    ax.plot(beta_values, beta_values, ls="--", color="k", label="available")
    ax.set_xlabel(r"Constraint budget $\beta$")
    ax.set_ylabel("Budget used")
    ax.set_title("Constrained Quadratic: constraint usage", fontweight="bold")
    add_grid(ax)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(beta_values, sweep["reward_values"], marker="o", lw=2)
    ax.set_xlabel(r"Constraint budget $\beta$")
    ax.set_ylabel("Discounted reward value")
    ax.set_title("Constrained Quadratic: achieved reward", fontweight="bold")
    add_grid(ax)
    plt.tight_layout()
    plt.show()


# Stochastic trajectories


def simulate_birth_death_constrained_v2(model, a_exp, i0=1, T=300, seed=None):
    rng = np.random.default_rng(seed)

    t = 0.0
    i = int(i0)

    times = [t]
    states = [i]

    while t < T and 0 < i < model.N:
        a = float(a_exp[i])

        lam = model.lam(i, a)
        mu = model.mu(i, a)
        rate = lam + mu

        if rate <= 0:
            break

        t += rng.exponential(1.0 / rate)

        if t > T:
            break

        if rng.random() < lam / rate:
            i += 1
        else:
            i -= 1

        times.append(t)
        states.append(i)

    return np.array(times), np.array(states)


def plot_trajectories_constrained_v2(results, i0=1, T=300, n_paths=8):
    for out in results:
        model = out["model"]
        a_exp = out.get("a_exp_raw", out["a_exp"])

        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

        for k in range(n_paths):
            t, x = simulate_birth_death_constrained_v2(
                model,
                a_exp,
                i0=i0,
                T=T,
                seed=100 + 19 * k,
            )

            ax.step(t, x, where="post", alpha=0.8, lw=1.5)

        crossings = out["crossings"]

        if len(crossings) > 0:
            ax.axhline(crossings[0], ls=":", color="k", label="first zero crossing")

        ax.set_xlabel("Time")
        ax.set_ylabel("Population state $i(t)$")
        ax.set_title(
            rf"Constrained v2 trajectories, $\beta={out['beta']:.3g}$",
            fontweight="bold",
        )
        add_grid(ax)

        if len(crossings) > 0:
            ax.legend()

        plt.tight_layout()
        plt.show()


# N-truncation diagnostics


def discounted_state_occupancy_constrained_v2(model, res):
    if not res.success:
        return None, None

    n_actions = get_n_actions_v2(model)
    Lambda = get_Lambda_v2(model)

    x = res.x.reshape((model.N + 1, n_actions))
    occ_action = x / (model.alpha + Lambda)
    occ_state = occ_action.sum(axis=1)

    return occ_state, occ_action


def occupancy_tail_diagnostics_v2(model, res, tail_m=10, tail_frac_start=0.90):
    occ_state, _ = discounted_state_occupancy_constrained_v2(model, res)

    if occ_state is None:
        return {
            "occ_total": np.nan,
            "occ_mean_state": np.nan,
            "boundary_occ_frac": np.nan,
            "last_m_occ_frac": np.nan,
            "upper_occ_frac": np.nan,
        }

    total = np.sum(occ_state)

    if total <= 0 or not np.isfinite(total):
        return {
            "occ_total": np.nan,
            "occ_mean_state": np.nan,
            "boundary_occ_frac": np.nan,
            "last_m_occ_frac": np.nan,
            "upper_occ_frac": np.nan,
        }

    states = np.arange(model.N + 1)

    last_m_start = max(0, model.N - int(tail_m) + 1)
    upper_start = int(np.ceil(tail_frac_start * model.N))

    return {
        "occ_total": total,
        "occ_mean_state": np.sum(states * occ_state) / total,
        "boundary_occ_frac": occ_state[model.N] / total,
        "last_m_occ_frac": np.sum(occ_state[last_m_start : model.N + 1]) / total,
        "upper_occ_frac": np.sum(occ_state[upper_start : model.N + 1]) / total,
    }


def N_truncation_test_constrained_v2(
    params,
    beta,
    N_values=(80, 100, 120, 160, 200),
    eps=0.05,
    x_compare=75,
    tail_m=10,
    tail_frac_start=0.90,
):
    N_values = sorted([int(N) for N in N_values])
    x_compare = int(min(x_compare, min(N_values) - 1))

    raw_results = []

    for N in N_values:
        p = params.copy()
        p["N"] = int(N)

        model, res, policy, a_exp = solve_constrained_v2(p, beta)

        if a_exp is None:
            raw_results.append(
                {
                    "N": N,
                    "params": p,
                    "model": model,
                    "res": res,
                    "policy": None,
                    "a_exp": None,
                    "i": np.arange(N + 1),
                    "drift": np.full(N + 1, np.nan),
                    "reward_inst": np.full(N + 1, np.nan),
                    "success": False,
                }
            )
            continue

        i, drift = per_capita_drift_constrained_v2(model, a_exp)
        _, reward_inst = reward_profile_v2(model, a_exp)

        raw_results.append(
            {
                "N": N,
                "params": p,
                "model": model,
                "res": res,
                "policy": policy,
                "a_exp": a_exp,
                "i": i,
                "drift": drift,
                "reward_inst": reward_inst,
                "success": True,
            }
        )

    successful = [out for out in raw_results if out["success"]]

    if len(successful) == 0:
        print("No successful LP_v2 solves in N-truncation test.")
        return {
            "beta": beta,
            "N_values": np.array(N_values),
            "x_compare": x_compare,
            "states_common": np.arange(1, x_compare + 1),
            "results": raw_results,
            "summary": pd.DataFrame(),
        }

    ref = successful[-1]
    states_common = np.arange(1, x_compare + 1)

    ref_policy = ref["a_exp"][states_common]
    ref_drift = ref["drift"][states_common]

    rows = []

    for out in raw_results:
        N = out["N"]
        model = out["model"]
        res = out["res"]

        row = {
            "beta": beta,
            "N": N,
            "success": out["success"],
        }

        if not out["success"]:
            row.update(
                {
                    "budget_used": np.nan,
                    "budget_frac": np.nan,
                    "reward_value": np.nan,
                    "first_crossing": np.nan,
                    "dormant_width_full": np.nan,
                    "dormant_width_common": np.nan,
                    "policy_Linf_vs_ref": np.nan,
                    "policy_RMSE_vs_ref": np.nan,
                    "drift_Linf_vs_ref": np.nan,
                    "drift_RMSE_vs_ref": np.nan,
                    "occ_total": np.nan,
                    "occ_mean_state": np.nan,
                    "boundary_occ_frac": np.nan,
                    "last_m_occ_frac": np.nan,
                    "upper_occ_frac": np.nan,
                    "solve_method": "failed",
                    "used_unconstrained_fallback": False,
                }
            )
            rows.append(row)
            continue

        a_common = out["a_exp"][states_common]
        d_common = out["drift"][states_common]

        policy_diff = a_common - ref_policy
        drift_diff = d_common - ref_drift

        crossings = find_zero_crossings_v2(out["i"], out["drift"])

        budget_used = budget_used_constrained_v2(model, res)
        reward_value = reward_value_constrained_v2(model, res)
        tail_diag = occupancy_tail_diagnostics_v2(
            model,
            res,
            tail_m=tail_m,
            tail_frac_start=tail_frac_start,
        )

        row.update(
            {
                "budget_used": budget_used,
                "budget_frac": budget_used / beta if beta > 0 else np.nan,
                "reward_value": reward_value,
                "first_crossing": crossings[0] if len(crossings) > 0 else np.nan,
                "dormant_width_full": dormant_width_constrained_v2(
                    out["i"], out["drift"], eps=eps
                ),
                "dormant_width_common": int(np.sum(np.abs(d_common) <= eps)),
                "policy_Linf_vs_ref": np.nanmax(np.abs(policy_diff)),
                "policy_RMSE_vs_ref": np.sqrt(np.nanmean(policy_diff**2)),
                "drift_Linf_vs_ref": np.nanmax(np.abs(drift_diff)),
                "drift_RMSE_vs_ref": np.sqrt(np.nanmean(drift_diff**2)),
                "solve_method": getattr(res, "solve_method", "unknown"),
                "used_unconstrained_fallback": getattr(
                    res, "used_unconstrained_fallback", False
                ),
            }
        )

        row.update(tail_diag)
        rows.append(row)

    summary = pd.DataFrame(rows)

    return {
        "beta": beta,
        "N_values": np.array(N_values),
        "x_compare": x_compare,
        "states_common": states_common,
        "results": raw_results,
        "reference_N": ref["N"],
        "summary": summary,
    }


def plot_N_truncation_overlay_constrained_v2(
    trunc,
    kind="drift",
    x_max=None,
    ylim=None,
):
    results = [out for out in trunc["results"] if out["success"]]

    if len(results) == 0:
        print("No successful LP_v2 results to plot.")
        return

    if x_max is None:
        x_max = trunc["x_compare"]

    x_max = int(min(x_max, trunc["x_compare"]))
    states = np.arange(1, x_max + 1)

    title_fs = 20
    label_fs = 17
    tick_fs = 14
    legend_fs = 12

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    for out in results:
        N = out["N"]

        if kind == "policy":
            y = out["a_exp"][states]
            ax.plot(states, y, lw=2, label=rf"$N={N}$")

        elif kind == "drift":
            y = out["drift"][states]
            ax.plot(states, y, lw=2, label=rf"$N={N}$")

        elif kind == "reward":
            y = out["reward_inst"][states]
            ax.plot(states, y, lw=2, label=rf"$N={N}$")

        else:
            raise ValueError("kind must be 'policy', 'drift', or 'reward'.")

    if kind == "policy":
        a_init = results[0]["model"].a_init
        ax.axhline(a_init, ls="--", color="k", label=r"Baseline $a^*$")
        ax.set_ylabel(
            r"Expected optimal action $\mathbb{E}[\pi \mid i]$",
            fontsize=label_fs,
        )
        if ylim is None:
            ylim = (-0.02, 1.02)
        title = rf"N-truncation test: v2 policy, $\beta={trunc['beta']}$"

    elif kind == "drift":
        ax.axhline(0.0, ls="--", color="k")
        ax.set_ylabel(
            r"Per-capita drift $(\lambda-\mu)/i$",
            fontsize=label_fs,
        )
        title = rf"N-truncation test: v2 drift, $\beta={trunc['beta']}$"

    else:
        ax.axhline(0.0, ls="--", color="k")
        ax.set_ylabel(r"Instantaneous reward $r_e(i,a(i))$", fontsize=label_fs)
        title = rf"N-truncation test: v2 reward, $\beta={trunc['beta']}$"

    ax.set_xlim(1, x_max)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=10)

    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(fontsize=legend_fs, frameon=True)

    plt.tight_layout()
    plt.show()


def plot_N_truncation_diagnostics_constrained_v2(trunc):
    df = trunc["summary"].copy()
    df = df[df["success"] == True]

    if len(df) == 0:
        print("No successful LP_v2 results to plot.")
        return

    title_fs = 18
    label_fs = 15
    tick_fs = 13
    legend_fs = 12

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)

    ax.plot(df["N"], df["policy_Linf_vs_ref"], marker="o", lw=2, label="policy")
    ax.plot(df["N"], df["drift_Linf_vs_ref"], marker="o", lw=2, label="drift")

    ax.set_xlabel("Truncation level $N$", fontsize=label_fs)
    ax.set_ylabel(r"$L^\infty$ error vs largest $N$", fontsize=label_fs)
    ax.set_title("v2 interior solution error", fontsize=title_fs, fontweight="bold")
    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(fontsize=legend_fs)

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)

    ax.plot(df["N"], df["boundary_occ_frac"], marker="o", lw=2, label=r"state $N$")
    ax.plot(df["N"], df["last_m_occ_frac"], marker="o", lw=2, label="last states")
    ax.plot(df["N"], df["upper_occ_frac"], marker="o", lw=2, label="upper tail")

    ax.set_xlabel("Truncation level $N$", fontsize=label_fs)
    ax.set_ylabel("Fraction of discounted occupancy", fontsize=label_fs)
    ax.set_title(
        "v2 upper-boundary occupancy diagnostic", fontsize=title_fs, fontweight="bold"
    )
    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(fontsize=legend_fs)

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)

    ax.plot(
        df["N"], df["first_crossing"], marker="o", lw=2, label="first zero crossing"
    )
    ax.plot(
        df["N"],
        df["dormant_width_common"],
        marker="o",
        lw=2,
        label="common dormant width",
    )

    ax.set_xlabel("Truncation level $N$", fontsize=label_fs)
    ax.set_ylabel("State / number of states", fontsize=label_fs)
    ax.set_title(
        "v2 dormancy metrics under N-truncation", fontsize=title_fs, fontweight="bold"
    )
    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(fontsize=legend_fs)

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)

    ax.plot(df["N"], df["budget_used"], marker="o", lw=2, label="budget used")
    ax.plot(df["N"], df["reward_value"], marker="o", lw=2, label="reward value")

    ax.set_xlabel("Truncation level $N$", fontsize=label_fs)
    ax.set_ylabel("Value", fontsize=label_fs)
    ax.set_title("v2 budget and reward stability", fontsize=title_fs, fontweight="bold")
    ax.tick_params(axis="both", labelsize=tick_fs)

    add_grid(ax)
    ax.legend(fontsize=legend_fs)

    plt.tight_layout()
    plt.show()


def N_truncation_beta_grid_constrained_v2(
    params,
    beta_values,
    N_values=(80, 100, 120, 160, 200),
    eps=0.05,
    x_compare=75,
    tail_m=10,
    tail_frac_start=0.90,
):
    tests = {}
    all_rows = []

    for beta in beta_values:
        trunc = N_truncation_test_constrained_v2(
            params,
            beta=beta,
            N_values=N_values,
            eps=eps,
            x_compare=x_compare,
            tail_m=tail_m,
            tail_frac_start=tail_frac_start,
        )

        tests[beta] = trunc
        all_rows.append(trunc["summary"].copy())

    summary_all = pd.concat(all_rows, ignore_index=True)

    return tests, summary_all


# Example parameters and usage

BASE_CONSTRAINED_V2 = dict(
    N=200,
    r=0.30,
    delta=0.15,
    delta0=0.05,
    alpha=0.51,
    kappa=1.3,
    c1=0.01,
    c2=0.0,
    c3=0.75,
    penalty=1e4,
    n_a=101,
    initial_state=10,
)

# With c2 = 0:
#   r_e(i) = c3*i - c1*i^2
#   i_pref = c3/(2*c1)
# With c1=0.005 and c3=1.0, i_pref = 100.
# This keeps the preferred-size effect visible in the first 100 states
# while N remains a computational truncation.

beta_examples_v2 = [0.05, 0.30, 0.5, 0.75, 1, 1.2, 1.5]
beta_values_v2 = np.linspace(0.005, 2.0, 20)


results_constrained_v2 = solve_beta_grid_v2(
    BASE_CONSTRAINED_V2,
    beta_examples_v2,
)

plot_policy_overlay_constrained_v2(
    results_constrained_v2,
    x_max=100,
)

plot_drift_overlay_constrained_v2(
    results_constrained_v2,
    x_max=100,
    ylim=(-0.3, 0.40),
)

plot_reward_overlay_constrained_v2(
    results_constrained_v2,
    x_max=100,
)

sweep_constrained_v2 = plot_beta_heatmaps_constrained_v2(
    BASE_CONSTRAINED_V2,
    beta_values_v2,
    eps=0.05,
    x_max=100,
)

plot_dormancy_metrics_constrained_v2(
    sweep_constrained_v2,
    eps=0.05,
)


# N-truncation test

N_values_v2 = [100, 200, 400, 500, 1000, 1500]
beta_test_v2 = 0.30

trunc_v2 = N_truncation_test_constrained_v2(
    BASE_CONSTRAINED_V2,
    beta=beta_test_v2,
    N_values=N_values_v2,
    eps=0.05,
    x_compare=75,
    tail_m=10,
    tail_frac_start=0.90,
)

display(trunc_v2["summary"])

plot_N_truncation_overlay_constrained_v2(
    trunc_v2,
    kind="policy",
    x_max=75,
)

plot_N_truncation_overlay_constrained_v2(
    trunc_v2,
    kind="drift",
    x_max=75,
    ylim=(-0.05, 0.50),
)

plot_N_truncation_overlay_constrained_v2(
    trunc_v2,
    kind="reward",
    x_max=75,
)

plot_N_truncation_diagnostics_constrained_v2(trunc_v2)
# %% [markdown]
# ## Logistic comparison

# %%
# Logistic-like drift comparison
# Quadratic vs Constrained Quadratic

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Basic helpers


def add_grid(ax):
    ax.grid(True, alpha=0.3)


def get_n_actions_v2(model):
    return int(getattr(model, "n_actions", getattr(model, "n_a", len(model.actions))))


def get_Lambda_v2(model):
    if hasattr(model, "Lambda"):
        return model.Lambda

    if hasattr(model, "global_uniformization_rate"):
        return model.global_uniformization_rate()

    lam_attr = getattr(model, "lam", None)
    if lam_attr is not None and not callable(lam_attr):
        return lam_attr

    raise AttributeError("Could not determine LP uniformization rate Lambda.")


def find_zero_crossings(i, drift):
    i = np.asarray(i, dtype=float)
    drift = np.asarray(drift, dtype=float)

    crossings = []

    for j in range(len(i) - 1):

        if i[j] <= 0 or i[j + 1] <= 0:
            continue

        if not np.isfinite(drift[j]) or not np.isfinite(drift[j + 1]):
            continue

        if drift[j] == 0:
            crossings.append(float(i[j]))

        elif drift[j] * drift[j + 1] < 0:
            crossings.append(0.5 * (float(i[j]) + float(i[j + 1])))

    return crossings


def first_zero_crossing(i, drift):
    crossings = find_zero_crossings(i, drift)

    if len(crossings) == 0:
        return np.nan

    return crossings[0]


# Occupancy support for constrained LP


def occupancy_supported_mask_constrained_quad(
    curve,
    *,
    rel_tol=1e-8,
    abs_tol=1e-12,
    keep_first_contiguous_block=True,
):
    model = curve["model"]
    res = curve.get("res", None)

    # Non-LP models do not have a constrained occupancy measure.
    if res is None or not getattr(res, "success", False):
        return np.ones(model.N + 1, dtype=bool)

    nA = get_n_actions_v2(model)
    x = np.asarray(res.x, dtype=float).reshape((model.N + 1, nA))
    x = np.maximum(x, 0.0)

    state_mass = x.sum(axis=1)

    max_mass = np.nanmax(state_mass)
    tol = max(abs_tol, rel_tol * max_mass)

    supported = state_mass > tol

    # Never interpret state 0 for per-capita drift.
    supported[0] = False

    if not keep_first_contiguous_block:
        return supported

    # Keep only the first contiguous supported block after state 0.
    # This removes artificial far-state/default-policy segments.
    mask = np.zeros_like(supported, dtype=bool)

    started = False

    for i in range(1, model.N + 1):
        if supported[i]:
            started = True
            mask[i] = True
        elif started:
            break

    return mask


def make_drift_for_fit(curve, *, use_occupancy_cut=True):
    drift = np.asarray(curve["drift"], dtype=float).copy()

    if use_occupancy_cut and curve.get("is_constrained_lp", False):
        occ_mask = occupancy_supported_mask_constrained_quad(curve)
        drift[~occ_mask] = np.nan

    return drift


# Logistic reference fitting


def fit_logistic_reference_from_zero_crossing(
    i,
    drift,
    *,
    x_max=100,
    fit_min=1,
    fit_max=None,
):
    i = np.asarray(i, dtype=float)
    drift = np.asarray(drift, dtype=float)

    if fit_max is None:
        fit_max = x_max

    K = first_zero_crossing(i, drift)

    if not np.isfinite(K) or K <= 0:
        return {
            "rho": np.nan,
            "K": np.nan,
            "g_log": np.full_like(i, np.nan, dtype=float),
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
        }

    mask = (i >= fit_min) & (i <= fit_max) & np.isfinite(drift)

    ii = i[mask]
    dd = drift[mask]

    if len(ii) < 3:
        return {
            "rho": np.nan,
            "K": K,
            "g_log": np.full_like(i, np.nan, dtype=float),
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
        }

    phi = 1.0 - ii / K
    denom = np.sum(phi**2)

    if denom <= 0:
        rho = np.nan
    else:
        rho = float(np.sum(dd * phi) / denom)

    g_log = rho * (1.0 - i / K)

    fitted = rho * phi
    residual = dd - fitted

    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((dd - np.mean(dd)) ** 2))

    if ss_tot > 0:
        r2 = float(1.0 - ss_res / ss_tot)
    else:
        r2 = np.nan

    return {
        "rho": rho,
        "K": K,
        "g_log": g_log,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def fit_logistic_reference_linear(
    i,
    drift,
    *,
    x_max=100,
    fit_min=1,
    fit_max=None,
):
    i = np.asarray(i, dtype=float)
    drift = np.asarray(drift, dtype=float)

    if fit_max is None:
        fit_max = x_max

    mask = (i >= fit_min) & (i <= fit_max) & np.isfinite(drift)

    ii = i[mask]
    dd = drift[mask]

    if len(ii) < 3:
        return {
            "rho": np.nan,
            "K": np.nan,
            "g_log": np.full_like(i, np.nan, dtype=float),
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        }

    slope, intercept = np.polyfit(ii, dd, deg=1)

    rho = float(intercept)

    if slope < 0 and rho > 0:
        K = float(-rho / slope)
    else:
        K = np.nan

    if np.isfinite(K):
        g_log = rho * (1.0 - i / K)
    else:
        g_log = intercept + slope * i

    fitted = intercept + slope * ii
    residual = dd - fitted

    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((dd - np.mean(dd)) ** 2))

    if ss_tot > 0:
        r2 = float(1.0 - ss_res / ss_tot)
    else:
        r2 = np.nan

    return {
        "rho": rho,
        "K": K,
        "g_log": g_log,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "slope": float(slope),
        "intercept": float(intercept),
    }


# Model solvers


def solve_quadratic_for_logistic_comparison(params):
    model = DormancyCTMDP_DiscountedPI_V2(params)
    out = model.solve()

    if isinstance(out, tuple):
        policy = out[-1]
    else:
        policy = out

    i = np.arange(model.N + 1)
    drift = np.full(model.N + 1, np.nan)

    for k in range(1, model.N + 1):
        a = float(policy[k])
        drift[k] = (model.lam(k, a) - model.mu(k, a)) / k

    return {
        "label": "Quadratic",
        "model": model,
        "policy": policy,
        "i": i,
        "drift": drift,
        "is_constrained_lp": False,
    }


def solve_constrained_quadratic_for_logistic_comparison(params, beta):
    model = TumorActorLP_v2(params, beta)
    res = model.solve()

    i = np.arange(model.N + 1)
    drift = np.full(model.N + 1, np.nan)

    if not res.success:
        print(f"Constrained quadratic LP failed for beta={beta}: {res.message}")

        return {
            "label": rf"Constrained quadratic, $\beta={beta}$",
            "model": model,
            "res": res,
            "a_exp": None,
            "i": i,
            "drift": drift,
            "is_constrained_lp": True,
        }

    if hasattr(model, "expected_action_policy"):
        a_exp = model.expected_action_policy(res.x)
    else:
        policy = model.extract_policy(res.x)
        a_exp = np.full(model.N + 1, model.a_init, dtype=float)

        for k in range(model.N + 1):
            if k in policy:
                a_exp[k] = sum(float(a) * float(p) for a, p in policy[k])

        a_exp[0] = 1.0

    for k in range(1, model.N + 1):
        a = float(a_exp[k])
        drift[k] = (model.lam(k, a) - model.mu(k, a)) / k

    return {
        "label": rf"Constrained quadratic, $\beta={beta}$",
        "model": model,
        "res": res,
        "a_exp": a_exp,
        "i": i,
        "drift": drift,
        "is_constrained_lp": True,
    }


# Plotting and fit table


def plot_logistic_drift_comparison(
    curves,
    *,
    x_max=100,
    fit_min=1,
    fit_max=None,
    fit_method="zero_crossing",
    use_occupancy_cut=True,
    legend_inside_right=True,
    show_fit_table=True,
):
    if fit_max is None:
        fit_max = x_max

    title_fs = 22
    label_fs = 18
    tick_fs = 14
    legend_fs = 11

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=150)

    rows = []

    for curve in curves:
        label = curve["label"]
        i = np.asarray(curve["i"], dtype=float)
        drift = np.asarray(curve["drift"], dtype=float)

        drift_for_fit = make_drift_for_fit(
            curve,
            use_occupancy_cut=use_occupancy_cut,
        )

        if fit_method == "zero_crossing":
            fit = fit_logistic_reference_from_zero_crossing(
                i,
                drift_for_fit,
                x_max=x_max,
                fit_min=fit_min,
                fit_max=fit_max,
            )
        elif fit_method == "linear":
            fit = fit_logistic_reference_linear(
                i,
                drift_for_fit,
                x_max=x_max,
                fit_min=fit_min,
                fit_max=fit_max,
            )
        else:
            raise ValueError("fit_method must be 'zero_crossing' or 'linear'.")

        rho = fit["rho"]
        K = fit["K"]

        rows.append(
            {
                "model": label,
                "fit_method": fit_method,
                "rho_logistic": rho,
                "K_logistic": K,
                "RMSE": fit["rmse"],
                "MAE": fit["mae"],
                "R2": fit["r2"],
            }
        )

        # Plot model drift.
        mask_plot = (i >= 1) & (i <= x_max) & np.isfinite(drift)

        if use_occupancy_cut and curve.get("is_constrained_lp", False):
            occ_mask = occupancy_supported_mask_constrained_quad(curve)
            mask_plot = mask_plot & occ_mask

        ax.plot(
            i[mask_plot],
            drift[mask_plot],
            lw=3,
            label=label,
        )

        # Plot fitted logistic reference only on the same support/window.
        if np.any(np.isfinite(fit["g_log"])):
            if np.isfinite(rho) and np.isfinite(K):
                logistic_label = (
                    rf"Logistic ref. for {label}: " rf"$\rho={rho:.3g}$, $K={K:.1f}$"
                )
            else:
                logistic_label = rf"Logistic ref. for {label}"

            ax.plot(
                i[mask_plot],
                fit["g_log"][mask_plot],
                lw=2.3,
                ls="--",
                label=logistic_label,
            )

        # Mark fitted carrying capacity.
        if np.isfinite(K) and 1 <= K <= x_max:
            ax.axvline(K, ls=":", lw=1.6)

    ax.axhline(0.0, ls="--", color="k", lw=1.6)

    ax.set_xlim(1, x_max)
    ax.set_xlabel("Population state $i$", fontsize=label_fs)
    ax.set_ylabel(r"Per-capita drift $g(i)$", fontsize=label_fs)

    ax.set_title(
        "Logistic-like comparison of induced drift",
        fontsize=title_fs,
        fontweight="bold",
        pad=10,
    )

    ax.tick_params(axis="both", labelsize=tick_fs)
    add_grid(ax)

    if legend_inside_right:
        ax.legend(
            fontsize=legend_fs,
            frameon=True,
            loc="upper right",
        )
    else:
        ax.legend(
            fontsize=legend_fs,
            frameon=True,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )

    plt.tight_layout()
    plt.show()

    fit_df = pd.DataFrame(rows)

    print("\nFitted logistic reference parameters:")
    for _, row in fit_df.iterrows():
        print(
            f"{row['model']}: "
            f"rho = {row['rho_logistic']:.6g}, "
            f"K = {row['K_logistic']:.6g}, "
            f"RMSE = {row['RMSE']:.6g}, "
            f"R2 = {row['R2']:.6g}"
        )

    if show_fit_table:
        display(fit_df)

    return fit_df


def logistic_trajectory(t, i0, rho, K):
    t = np.asarray(t, dtype=float)

    if i0 <= 0 or not np.isfinite(rho) or not np.isfinite(K) or K <= 0:
        return np.full_like(t, np.nan, dtype=float)

    return K / (1.0 + ((K - i0) / i0) * np.exp(-rho * t))


def simulate_birth_death_from_policy_for_logistic(
    model,
    action_array,
    *,
    i0=1,
    T=300,
    seed=None,
):
    rng = np.random.default_rng(seed)

    t = 0.0
    i = int(i0)

    times = [t]
    states = [i]

    while t < T and 0 < i < model.N:
        a = float(action_array[i])

        lam = model.lam(i, a)
        mu = model.mu(i, a)
        rate = lam + mu

        if rate <= 0:
            break

        t += rng.exponential(1.0 / rate)

        if t > T:
            break

        if rng.random() < lam / rate:
            i += 1
        else:
            i -= 1

        times.append(t)
        states.append(i)

    return np.array(times), np.array(states)


def plot_logistic_trajectory_reference_constrained_quadratic(
    curve,
    fit_row,
    *,
    beta=None,
    i0=1,
    T=300,
    n_paths=25,
    n_time=400,
    y_max=None,
):
    model = curve["model"]
    a_exp = curve.get("a_exp", None)

    if a_exp is None:
        print("No action policy available for trajectory simulation.")
        return None

    rho = float(fit_row["rho_logistic"])
    K = float(fit_row["K_logistic"])

    t_grid = np.linspace(0, T, n_time)
    mean_states = np.zeros_like(t_grid)

    all_paths = []

    for k in range(n_paths):
        t, x = simulate_birth_death_from_policy_for_logistic(
            model,
            a_exp,
            i0=i0,
            T=T,
            seed=1000 + 17 * k,
        )

        x_interp = np.interp(t_grid, t, x, left=x[0], right=x[-1])
        mean_states += x_interp
        all_paths.append((t, x))

    mean_states /= n_paths

    logistic_ref = logistic_trajectory(t_grid, i0=i0, rho=rho, K=K)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)

    for t, x in all_paths[: min(n_paths, 10)]:
        ax.step(t, x, where="post", alpha=0.35, lw=1.2)

    ax.plot(
        t_grid,
        mean_states,
        lw=3,
        label="Mean stochastic trajectory",
    )

    ax.plot(
        t_grid,
        logistic_ref,
        lw=2.5,
        ls="--",
        label=rf"Logistic reference: $\rho={rho:.3g}$, $K={K:.1f}$",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Population state $i(t)$")

    if beta is None:
        title = "Constrained quadratic: trajectory vs logistic reference"
    else:
        title = (
            rf"Constrained quadratic: trajectory vs logistic reference, $\beta={beta}$"
        )

    ax.set_title(title, fontweight="bold")

    if y_max is not None:
        ax.set_ylim(0, y_max)

    add_grid(ax)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.show()

    return {
        "rho": rho,
        "K": K,
        "t_grid": t_grid,
        "mean_states": mean_states,
        "logistic_ref": logistic_ref,
        "paths": all_paths,
    }


# Parameters

QUADRATIC_PARAMS_FOR_LOGISTIC = dict(
    N=600,
    r=0.30,
    delta=0.15,
    delta0=0.05,
    alpha=0.51,
    kappa=1.3,
    c1=0.01,
    c2=0.0,
    c3=0.75,
    penalty=1e4,
    n_a=101,
    initial_state=10,
)

CONSTRAINED_QUADRATIC_PARAMS_FOR_LOGISTIC = dict(
    N=600,
    r=0.30,
    delta=0.15,
    delta0=0.05,
    alpha=0.51,
    kappa=1.3,
    c1=0.01,
    c2=0.0,
    c3=0.75,
    penalty=1e4,
    n_a=101,
    initial_state=10,
)


# Run comparison

beta_for_logistic = 1.2

quad_curve = solve_quadratic_for_logistic_comparison(QUADRATIC_PARAMS_FOR_LOGISTIC)

constrained_quad_curve = solve_constrained_quadratic_for_logistic_comparison(
    CONSTRAINED_QUADRATIC_PARAMS_FOR_LOGISTIC,
    beta=beta_for_logistic,
)

fit_df = plot_logistic_drift_comparison(
    [
        quad_curve,
        constrained_quad_curve,
    ],
    x_max=100,
    fit_min=1,
    fit_max=100,
    fit_method="zero_crossing",
    use_occupancy_cut=True,
    legend_inside_right=True,
    show_fit_table=True,
)


# Optional trajectory plot for constrained quadratic

# Grab the fitted logistic row for the constrained quadratic curve.
constrained_fit_row = fit_df[
    fit_df["model"].astype(str).str.contains("Constrained", case=False, regex=False)
].iloc[0]

trajectory_out = plot_logistic_trajectory_reference_constrained_quadratic(
    constrained_quad_curve,
    constrained_fit_row,
    beta=beta_for_logistic,
    i0=10,
    T=300,
    n_paths=25,
    n_time=400,
    y_max=140,
)
# %% [markdown]
# ## Numerical convergence

# %%
# ACTION-GRID CONVERGENCE STUDY — PLOTS ONLY

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Grid refinement

ACTION_GRID_SIZES = [26, 51, 101, 201]

OCC_ABS_TOL = 1e-14
OCC_REL_TOL = 1e-10


def action_spacing(n_actions):
    return 1.0 / (n_actions - 1)


# Replace these with the exact parameters used in your figures.

SHARED_DISCOUNTED = dict(
    N=500,
    r=0.20,
    delta=0.25,
    delta0=0.05,
    alpha=0.51,
    penalty=1.0e4,
)

PARAM_DISCOUNTED_I = dict(
    **SHARED_DISCOUNTED,
    L=35,
    kappa1=1.0,
    kappa2=0.5,
)

PARAM_DISCOUNTED_QUADRATIC = dict(
    **SHARED_DISCOUNTED,
    c1=0.005,
    c2=0.02,
    kappa=1.2,
    c3=1.0,
)

PARAM_UNBOUNDED = dict(
    N=800,
    r=0.20,
    delta=0.25,
    delta0=0.05,
    alpha=0.51,
    penalty=1.0e4,
    kappa=1.0,
    c3=0.8,
)

PARAM_CONSTRAINED_LINEAR = dict(
    N=400,
    r=0.20,
    delta=0.25,
    delta0=0.05,
    alpha=0.51,
    penalty=1.0e4,
    kappa=1.2,
    c3=1.0,
    initial_state=10,
)

BETA_LINEAR = 0.35

PARAM_CONSTRAINED_QUADRATIC = dict(
    N=400,
    r=0.20,
    delta=0.25,
    delta0=0.05,
    alpha=0.51,
    penalty=1.0e4,
    kappa=1.3,
    c1=0.01,
    c2=0.0,
    c3=0.75,
    initial_state=10,
)

BETA_QUADRATIC = 0.37


def relative_error(x, x_ref):
    denom = max(1.0, abs(float(x_ref)))
    return abs(float(x) - float(x_ref)) / denom


def relative_linf(x, x_ref):

    x = np.asarray(x, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float)

    return float(np.max(np.abs(x - x_ref)) / max(1.0, np.max(np.abs(x_ref))))


def interior_policy_errors(policy, policy_ref):

    policy = np.asarray(policy)
    policy_ref = np.asarray(policy_ref)

    # Exclude i=0 and i=N
    diff = np.abs(policy[1:-1] - policy_ref[1:-1])

    return {
        "policy_linf": float(np.max(diff)),
        "policy_mean_abs": float(np.mean(diff)),
    }


def lp_state_occupancy(model, x):

    X = np.asarray(x).reshape(model.N + 1, model.n_actions)

    X = np.maximum(X, 0.0)

    return np.sum(X, axis=1)


def lp_policy_errors(
    model,
    x,
    model_ref,
    x_ref,
):

    occ = lp_state_occupancy(model, x)
    occ_ref = lp_state_occupancy(model_ref, x_ref)

    a = model.expected_action_policy(x)
    a_ref = model_ref.expected_action_policy(x_ref)

    max_occ = max(
        np.max(occ),
        np.max(occ_ref),
        OCC_ABS_TOL,
    )

    threshold = max(
        OCC_ABS_TOL,
        OCC_REL_TOL * max_occ,
    )

    mask = (occ > threshold) & (occ_ref > threshold)

    # Ignore artificial boundaries
    mask[0] = False
    mask[-1] = False

    if not np.any(mask):

        return {
            "policy_linf": np.nan,
            "policy_mean_abs": np.nan,
            "occupancy_weighted_mean": np.nan,
            "occupancy_coverage": 0.0,
        }

    diff = np.abs(a[mask] - a_ref[mask])

    weights = occ_ref[mask]

    weighted_mean = np.sum(weights * diff) / np.sum(weights)

    total_ref_occ = np.sum(occ_ref[1:-1])

    coverage = np.sum(occ_ref[mask]) / total_ref_occ if total_ref_occ > 0 else np.nan

    return {
        "policy_linf": float(np.max(diff)),
        "policy_mean_abs": float(np.mean(diff)),
        "occupancy_weighted_mean": float(weighted_mean),
        "occupancy_coverage": float(coverage),
    }


# FAST POLICY ITERATION FOR UNBOUNDED MODEL


def evaluate_unbounded_policy(model, policy):

    N = model.N

    lower = np.zeros(N + 1)
    diag = np.zeros(N + 1)
    upper = np.zeros(N + 1)
    rhs = np.zeros(N + 1)

    for i in range(N + 1):

        a = float(policy[i])

        lam_i = model.lam(i, a)
        mu_i = model.mu(i, a)

        diag[i] = model.alpha + lam_i + mu_i

        if i == 0:

            rhs[i] = -model.penalty

        else:

            rhs[i] = model.c3 * i - model.kappa * i * (a - model.a_star) ** 2

        if i > 0:
            lower[i] = -mu_i

        if i < N:
            upper[i] = -lam_i

    # Thomas algorithm
    cp = np.zeros(N + 1)
    dp = np.zeros(N + 1)

    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]

    for i in range(1, N + 1):

        den = diag[i] - lower[i] * cp[i - 1]

        if abs(den) < 1e-14:
            raise RuntimeError(f"Near singular solve at state {i}")

        if i < N:
            cp[i] = upper[i] / den

        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / den

    V = np.zeros(N + 1)

    V[N] = dp[N]

    for i in range(N - 1, -1, -1):

        V[i] = dp[i] - cp[i] * V[i + 1]

    return V


def solve_unbounded_policy_iteration(
    model,
    max_iter=100,
):

    N = model.N
    actions = model.actions

    policy = np.full(N + 1, model.a_init)

    policy[0] = 1.0

    for iteration in range(max_iter):

        V = evaluate_unbounded_policy(model, policy)

        new_policy = np.zeros(N + 1)
        new_policy[0] = 1.0

        for i in range(1, N + 1):

            Vm = V[i - 1]

            Vp = V[i + 1] if i < N else V[i]

            a = actions

            lam = model.lam(i, a)
            mu = model.mu(i, a)

            reward = model.c3 * i - model.kappa * i * (a - model.a_star) ** 2

            vals = (reward + lam * Vp + mu * Vm) / (model.alpha + lam + mu)

            new_policy[i] = actions[np.argmax(vals)]

        change = np.max(np.abs(new_policy - policy))

        policy = new_policy

        if change == 0:
            break

    V = evaluate_unbounded_policy(model, policy)

    return V, policy


def run_deterministic(
    model_class,
    params,
    n_actions,
):

    p = dict(params)
    p["n_actions"] = n_actions

    model = model_class(p)

    result = model.solve()

    if isinstance(result, tuple):

        value, policy = result

    else:

        policy = np.asarray(result)

        if hasattr(model, "V"):
            value = model.V.copy()

        elif hasattr(model, "u"):
            value = model.u.copy()

        else:
            value = None

    return {
        "model": model,
        "policy": np.asarray(policy),
        "value": value,
    }


def run_unbounded(
    params,
    n_actions,
):

    p = dict(params)
    p["n_actions"] = n_actions

    model = DormancyCTMDP_Unbounded_Fast(p)

    value, policy = solve_unbounded_policy_iteration(model)

    return {
        "model": model,
        "policy": policy,
        "value": value,
    }


def run_lp(
    model_class,
    params,
    beta,
    n_actions,
):

    p = dict(params)
    p["n_actions"] = n_actions

    model = model_class(p, beta=beta)

    result = model.solve()

    if not result.success:
        raise RuntimeError(result.message)

    x = result.x

    return {
        "model": model,
        "x": x,
        "policy": model.expected_action_policy(x),
        "objective": model.discounted_reward_value(x),
        "regulatory_cost": model.discounted_regulatory_cost(x),
    }


MODEL_SPECS = [
    {
        "key": "discounted_I",
        "label": "Threshold Cost",
        "solver": "deterministic",
        "class": DormancyCTMDP_DiscountedPI,
        "params": PARAM_DISCOUNTED_I,
    },
    {
        "key": "discounted_quadratic",
        "label": "Discounted quadratic",
        "solver": "deterministic",
        "class": DormancyCTMDP_DiscountedPI_V2,
        "params": PARAM_DISCOUNTED_QUADRATIC,
    },
    {
        "key": "unbounded_linear",
        "label": "Unbounded linear reward",
        "solver": "unbounded",
        "params": PARAM_UNBOUNDED,
    },
    {
        "key": "constrained_linear",
        "label": "Constrained linear reward",
        "solver": "lp",
        "class": TumorActorLP,
        "params": PARAM_CONSTRAINED_LINEAR,
        "beta": BETA_LINEAR,
    },
    {
        "key": "constrained_quadratic",
        "label": "Constrained quadratic reward",
        "solver": "lp",
        "class": TumorActorLP_v2,
        "params": PARAM_CONSTRAINED_QUADRATIC,
        "beta": BETA_QUADRATIC,
    },
]


all_runs = {}

for spec in MODEL_SPECS:

    print("\n" + "=" * 60)
    print(spec["label"])
    print("=" * 60)

    runs = {}

    for n_actions in ACTION_GRID_SIZES:

        print(f"n_a = {n_actions}, " f"h = {action_spacing(n_actions):.4f}")

        if spec["solver"] == "deterministic":

            run = run_deterministic(
                spec["class"],
                spec["params"],
                n_actions,
            )

        elif spec["solver"] == "unbounded":

            run = run_unbounded(
                spec["params"],
                n_actions,
            )

        else:

            run = run_lp(
                spec["class"],
                spec["params"],
                spec["beta"],
                n_actions,
            )

        runs[n_actions] = run

    all_runs[spec["key"]] = runs


reference_n = ACTION_GRID_SIZES[-1]

rows = []

for spec in MODEL_SPECS:

    runs = all_runs[spec["key"]]
    ref = runs[reference_n]

    for n_actions in ACTION_GRID_SIZES:

        run = runs[n_actions]

        row = {
            "model_key": spec["key"],
            "model_label": spec["label"],
            "n_actions": n_actions,
            "h": action_spacing(n_actions),
        }

        if spec["solver"] in [
            "deterministic",
            "unbounded",
        ]:

            row.update(
                interior_policy_errors(
                    run["policy"],
                    ref["policy"],
                )
            )

            row["occupancy_weighted_mean"] = np.nan

            row["occupancy_coverage"] = np.nan

            row["regulatory_cost"] = np.nan

            row["objective_relative_error"] = relative_linf(
                run["value"][1:-1],
                ref["value"][1:-1],
            )

        else:

            row.update(
                lp_policy_errors(
                    run["model"],
                    run["x"],
                    ref["model"],
                    ref["x"],
                )
            )

            row["objective_relative_error"] = relative_error(
                run["objective"],
                ref["objective"],
            )

            row["regulatory_cost"] = run["regulatory_cost"]

        rows.append(row)


metrics = pd.DataFrame(rows)

display(metrics)


# FIGURE 1:

for spec in MODEL_SPECS:

    fig, ax = plt.subplots(figsize=(7, 5))

    for n_actions in ACTION_GRID_SIZES:

        policy = all_runs[spec["key"]][n_actions]["policy"]

        states = np.arange(len(policy))

        ax.plot(
            states[1:-1],
            policy[1:-1],
            linewidth=1.6,
            label=(rf"$n_a={n_actions}$ " rf"$(h={action_spacing(n_actions):g})$"),
        )

    ax.set_xlabel("Tumor population $i$")

    ax.set_ylabel("Optimal action $a(i)$")

    ax.set_title(spec["label"])

    ax.set_ylim(-0.02, 1.02)

    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()


# FIGURE 2:

fig, ax = plt.subplots(figsize=(10, 7))

shift_factors = [0.94, 0.97, 1.00, 1.03, 1.06]
markers = ["o", "s", "^", "D", "v"]
linestyles = ["-", "--", "-.", ":", "-"]

for spec, shift, marker, ls in zip(MODEL_SPECS, shift_factors, markers, linestyles):

    d = metrics[metrics["model_key"] == spec["key"]]

    d = d[d["n_actions"] != reference_n]

    valid = np.isfinite(d["policy_linf"]) & (d["policy_linf"] > 0)

    h_plot = d.loc[valid, "h"] * shift

    ax.loglog(
        h_plot,
        d.loc[valid, "policy_linf"],
        marker=marker,
        linestyle=ls,
        linewidth=2.2,
        markersize=8,
        label=spec["label"],
    )


ax.set_xlabel(r"Action-grid spacing $h$", fontsize=24, labelpad=10)

ax.set_ylabel(r"$\|\pi_h-\pi_{\mathrm{ref}}\|_\infty$", fontsize=24, labelpad=10)

ax.set_title("Action-grid convergence", fontsize=28, fontweight="bold", pad=15)

ax.tick_params(axis="both", which="major", labelsize=18)

ax.tick_params(axis="both", which="minor", labelsize=16)

ax.legend(
    fontsize=16,
    loc="upper left",
    frameon=True,
    edgecolor="black",
    framealpha=1.0,
    fancybox=False,
)
ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.show()

# FIGURE 3:

fig, ax = plt.subplots(figsize=(7, 5))

for spec in MODEL_SPECS:

    d = metrics[metrics["model_key"] == spec["key"]]

    d = d[d["n_actions"] != reference_n]

    if spec["solver"] == "lp":

        y = d["occupancy_weighted_mean"]

    else:

        y = d["policy_mean_abs"]

    valid = np.isfinite(y) & (y > 0)

    ax.loglog(
        d.loc[valid, "h"],
        y.loc[valid],
        marker="o",
        linewidth=1.6,
        label=spec["label"],
    )

ax.set_xlabel(r"Action-grid spacing $h$")

ax.set_ylabel("Mean policy discrepancy")

ax.set_title("Mean action-grid error")

ax.legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.show()


# FIGURE 4:

fig, ax = plt.subplots(figsize=(7, 5))

for spec in MODEL_SPECS:

    d = metrics[metrics["model_key"] == spec["key"]]

    d = d[d["n_actions"] != reference_n]

    valid = np.isfinite(d["objective_relative_error"]) & (
        d["objective_relative_error"] > 0
    )

    ax.loglog(
        d.loc[valid, "h"],
        d.loc[valid, "objective_relative_error"],
        marker="o",
        linewidth=1.6,
        label=spec["label"],
    )

ax.set_xlabel(r"Action-grid spacing $h$")

ax.set_ylabel("Relative value/objective error")

ax.set_title("Convergence of value and objective")

ax.legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.show()


# FIGURE 5:

fig, ax = plt.subplots(figsize=(7, 5))

for spec in MODEL_SPECS:

    if spec["solver"] != "lp":
        continue

    d = metrics[metrics["model_key"] == spec["key"]]

    ax.plot(
        d["h"],
        d["regulatory_cost"],
        marker="o",
        linewidth=1.6,
        label=spec["label"],
    )

    ax.axhline(
        spec["beta"],
        linestyle="--",
        linewidth=1,
    )

ax.set_xlabel(r"Action-grid spacing $h$")

ax.set_ylabel("Discounted regulatory cost")

ax.set_title("Budget constraint under grid refinement")

ax.legend(frameon=False)

plt.tight_layout()
plt.show()


display(
    metrics[metrics["n_actions"] == 101][
        [
            "model_label",
            "policy_linf",
            "policy_mean_abs",
            "occupancy_weighted_mean",
            "objective_relative_error",
            "regulatory_cost",
        ]
    ]
)
# STATE-SPACE TRUNCATION CONVERGENCE
#
# Compare each finite-state solution against the largest-N
# solution for that model.
#
# Deterministic/unbounded models:
#     compare policy over a common interior state range.
#
# Constrained LP models:
#     compare only states with meaningful occupancy under
#     BOTH the N-solution and the reference solution.


# Occupancy tolerances for LP state-space comparisons

STATE_OCC_REL_TOL = 1e-8
STATE_OCC_ABS_TOL = 1e-12


STATE_SPACE_SPECS = [
    {
        "key": "discounted_I",
        "label": "Threshold model",
        "solver": "deterministic",
        "class": DormancyCTMDP_DiscountedPI,
        "params": PARAM_DISCOUNTED_I,
        "n_actions": 101,
        # Current manuscript-scale model uses N=80.
        "N_values": [80, 100, 150, 200, 300],
        # Compare safely away from every truncation boundary.
        "compare_max": 75,
    },
    {
        "key": "discounted_quadratic",
        "label": "Discounted quadratic",
        "solver": "deterministic",
        "class": DormancyCTMDP_DiscountedPI_V2,
        "params": PARAM_DISCOUNTED_QUADRATIC,
        "n_actions": 101,
        "N_values": [80, 100, 150, 200, 300],
        "compare_max": 75,
    },
    {
        "key": "unbounded_linear",
        "label": "Unbounded linear reward",
        "solver": "unbounded",
        "params": PARAM_UNBOUNDED,
        "n_actions": 201,
        "N_values": [400, 600, 800, 1000, 1500],
        "compare_max": 300,
    },
    {
        "key": "constrained_linear",
        "label": "Constrained linear reward",
        "solver": "lp",
        "class": TumorActorLP,
        "params": PARAM_CONSTRAINED_LINEAR,
        "beta": BETA_LINEAR,
        "n_actions": 101,
        "N_values": [100, 200, 400, 500, 1000, 1500],
        # Same interior window used in your N-truncation diagnostics.
        "compare_max": 75,
    },
    {
        "key": "constrained_quadratic",
        "label": "Constrained quadratic reward",
        "solver": "lp",
        "class": TumorActorLP_v2,
        "params": PARAM_CONSTRAINED_QUADRATIC,
        "beta": BETA_QUADRATIC,
        "n_actions": 101,
        "N_values": [100, 200, 400, 500, 1000, 1500],
        "compare_max": 75,
    },
]


def state_space_policy_errors(
    policy,
    policy_ref,
    compare_max,
):

    policy = np.asarray(policy, dtype=float)

    policy_ref = np.asarray(policy_ref, dtype=float)

    max_common = min(
        int(compare_max),
        len(policy) - 2,
        len(policy_ref) - 2,
    )

    if max_common < 1:
        return {
            "policy_linf": np.nan,
            "policy_mean_abs": np.nan,
        }

    states = np.arange(1, max_common + 1)

    diff = np.abs(policy[states] - policy_ref[states])

    return {
        "policy_linf": float(np.max(diff)),
        "policy_mean_abs": float(np.mean(diff)),
    }


def state_space_lp_policy_errors(
    run,
    ref,
    compare_max,
    rel_tol=STATE_OCC_REL_TOL,
    abs_tol=STATE_OCC_ABS_TOL,
):

    model = run["model"]
    model_ref = ref["model"]

    x = np.asarray(run["x"], dtype=float)

    x_ref = np.asarray(ref["x"], dtype=float)

    # State occupancies

    occ = lp_state_occupancy(model, x)

    occ_ref = lp_state_occupancy(model_ref, x_ref)

    # Expected policies

    policy = model.expected_action_policy(x)

    policy_ref = model_ref.expected_action_policy(x_ref)

    # Common interior range

    max_common = min(
        int(compare_max),
        model.N - 1,
        model_ref.N - 1,
    )

    if max_common < 1:
        return {
            "policy_linf": np.nan,
            "policy_mean_abs": np.nan,
            "occupancy_weighted_mean": np.nan,
            "occupancy_coverage": 0.0,
        }

    states = np.arange(1, max_common + 1)

    # Occupancy threshold for EACH solution separately

    tol = max(abs_tol, rel_tol * np.max(occ))

    tol_ref = max(abs_tol, rel_tol * np.max(occ_ref))

    support = (occ[states] > tol) & (occ_ref[states] > tol_ref)

    if not np.any(support):
        return {
            "policy_linf": np.nan,
            "policy_mean_abs": np.nan,
            "occupancy_weighted_mean": np.nan,
            "occupancy_coverage": 0.0,
        }

    supported_states = states[support]

    # Policy discrepancy

    diff = np.abs(policy[supported_states] - policy_ref[supported_states])

    # Use reference occupancy for weighted discrepancy.
    weights = occ_ref[supported_states]

    weighted_mean = np.sum(weights * diff) / np.sum(weights)

    # Fraction of reference occupied mass in the comparison
    # window represented by jointly supported states.
    ref_support = occ_ref[states] > tol_ref

    ref_mass = np.sum(occ_ref[states][ref_support])

    joint_mass = np.sum(occ_ref[supported_states])

    coverage = joint_mass / ref_mass if ref_mass > 0 else np.nan

    return {
        "policy_linf": float(np.max(diff)),
        "policy_mean_abs": float(np.mean(diff)),
        "occupancy_weighted_mean": float(weighted_mean),
        "occupancy_coverage": float(coverage),
    }


def run_state_space_model(
    spec,
    N,
):

    params = dict(spec["params"])

    params["N"] = int(N)

    if spec["solver"] == "deterministic":

        return run_deterministic(
            spec["class"],
            params,
            spec["n_actions"],
        )

    elif spec["solver"] == "unbounded":

        return run_unbounded(
            params,
            spec["n_actions"],
        )

    elif spec["solver"] == "lp":

        return run_lp(
            spec["class"],
            params,
            spec["beta"],
            spec["n_actions"],
        )

    else:

        raise ValueError(f"Unknown solver type: {spec['solver']}")


state_space_runs = {}

for spec in STATE_SPACE_SPECS:

    print("\n" + "=" * 60)

    print("State-space convergence:", spec["label"])

    print("=" * 60)

    runs = {}

    for N in spec["N_values"]:

        print(f"N = {N}")

        t0 = time.perf_counter()

        run = run_state_space_model(spec, N)

        elapsed = time.perf_counter() - t0

        print(f"    done in {elapsed:.2f} s")

        runs[int(N)] = run

    state_space_runs[spec["key"]] = runs


state_rows = []

for spec in STATE_SPACE_SPECS:

    runs = state_space_runs[spec["key"]]

    N_reference = max(runs.keys())

    ref = runs[N_reference]

    for N in sorted(runs.keys()):

        run = runs[N]

        row = {
            "model_key": spec["key"],
            "model_label": spec["label"],
            "solver": spec["solver"],
            "N": int(N),
            "N_reference": int(N_reference),
            "compare_max": int(spec["compare_max"]),
        }

        if spec["solver"] == "lp":

            errors = state_space_lp_policy_errors(
                run,
                ref,
                compare_max=spec["compare_max"],
            )

            row.update(errors)

        else:

            errors = state_space_policy_errors(
                run["policy"],
                ref["policy"],
                compare_max=spec["compare_max"],
            )

            row.update(errors)

            row["occupancy_weighted_mean"] = np.nan

            row["occupancy_coverage"] = np.nan

        state_rows.append(row)


state_metrics = pd.DataFrame(state_rows)

display(state_metrics)

#
# State-space convergence of the optimal policy.
# Exact-zero discrepancies are displayed at a small
# visualization-only floor so they can appear on a log axis.

fig, ax = plt.subplots(figsize=(10, 7))

markers = [
    "o",
    "s",
    "^",
    "D",
    "v",
]

linestyles = [
    "-",
    "--",
    "-.",
    ":",
    "-",
]


# Actual N values used in all calculations are unchanged.
x_offsets = {
    "discounted_I": -20,
    "discounted_quadratic": 20,
    "unbounded_linear": 0,
    "constrained_linear": 0,
    "constrained_quadratic": 0,
}


# Exact-zero discrepancies cannot be shown on a logarithmic axis.
# This floor is used ONLY for plotting.
VISUAL_FLOOR = 1e-16


for spec, marker, ls in zip(
    STATE_SPACE_SPECS,
    markers,
    linestyles,
):

    d = state_metrics[state_metrics["model_key"] == spec["key"]].copy()

    # Remove largest-N reference solution.
    #
    # Its discrepancy is zero by definition and is not an
    # independent convergence point.
    d = d[d["N"] != d["N_reference"]]

    # Keep finite discrepancies, INCLUDING exact zeros.
    valid = np.isfinite(d["policy_linf"])

    # Visualization-only horizontal offset
    x_plot = d.loc[valid, "N"].to_numpy(dtype=float) + x_offsets[spec["key"]]

    # True numerical discrepancies
    y_true = d.loc[valid, "policy_linf"].to_numpy(dtype=float)

    # Visualization-only floor for exact zeros
    y_plot = np.where(y_true == 0.0, VISUAL_FLOOR, y_true)

    ax.semilogy(
        x_plot,
        y_plot,
        marker=marker,
        linestyle=ls,
        linewidth=2.2,
        markersize=8,
        label=spec["label"],
    )


ax.set_xlabel(
    r"State-space truncation $N$",
    fontsize=24,
    labelpad=10,
)

ax.set_ylabel(
    r"$\|\pi_N-\pi_{\mathrm{ref}}\|_\infty$",
    fontsize=24,
    labelpad=10,
)

ax.set_title(
    "State-space truncation convergence",
    fontsize=28,
    fontweight="bold",
    pad=15,
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=18,
)

ax.tick_params(
    axis="both",
    which="minor",
    labelsize=16,
)

ax.legend(
    fontsize=16,
    loc="best",
    frameon=True,
    edgecolor="black",
    framealpha=1.0,
    fancybox=False,
)

ax.grid(
    True,
    which="both",
    linestyle="--",
    linewidth=0.8,
    alpha=0.5,
)

plt.tight_layout()
plt.show()
# EXTINCTION-PENALTY CONVERGENCE
#
# Compare optimal policies as the extinction-state penalty
# is increased.
#
# Reference:
#     P_ext = 1e5
#
# Constrained LP models are compared only over states with
# meaningful discounted occupancy under BOTH solutions.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PENALTY_VALUES = np.array(
    [
        1e1,
        1e2,
        1e3,
        1e4,
        1e5,
        1e6,
        1e7,
    ],
    dtype=float,
)

PENALTY_REFERENCE = 1e5


PENALTY_SPECS = [
    {
        "key": "discounted_I",
        "label": "Threshold Model",
        "solver": "deterministic",
        "class": DormancyCTMDP_DiscountedPI,
        "params": PARAM_DISCOUNTED_I,
        "compare_max": 90,
        "n_actions": 101,
    },
    {
        "key": "discounted_quadratic",
        "label": "Discounted quadratic",
        "solver": "deterministic",
        "class": DormancyCTMDP_DiscountedPI_V2,
        "params": PARAM_DISCOUNTED_QUADRATIC,
        "compare_max": 90,
        "n_actions": 101,
    },
    {
        "key": "unbounded_linear",
        "label": "Unbounded linear reward",
        "solver": "unbounded",
        "class": DormancyCTMDP_Unbounded_Fast,
        "params": PARAM_UNBOUNDED,
        "compare_max": 720,
        "n_actions": 201,
    },
    {
        "key": "constrained_linear",
        "label": "Constrained linear reward",
        "solver": "lp",
        "class": TumorActorLP,
        "params": PARAM_CONSTRAINED_LINEAR,
        "compare_max": 1350,
        "n_actions": 101,
        "beta": BETA_LINEAR,
    },
    {
        "key": "constrained_quadratic",
        "label": "Constrained quadratic reward",
        "solver": "lp",
        "class": TumorActorLP_v2,
        "params": PARAM_CONSTRAINED_QUADRATIC,
        "compare_max": 450,
        "n_actions": 101,
        "beta": BETA_QUADRATIC,
    },
]


# Same conceptual criterion used for the occupancy masking:
#
# d(i) > max(
#     abs_tol,
#     rel_tol * max_j d(j)
# )

OCC_REL_TOL = 1e-8
OCC_ABS_TOL = 1e-12


def lp_state_occupancy(model, x):

    X = np.asarray(x, dtype=float).reshape(model.N + 1, model.n_actions)

    X = np.maximum(X, 0.0)

    return X.sum(axis=1)


def run_deterministic_at_penalty(
    model_class,
    base_params,
    penalty,
    n_actions,
):

    p = dict(base_params)

    p["penalty"] = float(penalty)
    p["n_actions"] = int(n_actions)

    model = model_class(p)

    result = model.solve()

    if isinstance(result, tuple):

        value, policy = result

    else:

        policy = np.asarray(result, dtype=float)

        if hasattr(model, "V"):

            value = model.V.copy()

        elif hasattr(model, "u"):

            value = model.u.copy()

        else:

            value = None

    return {
        "model": model,
        "policy": np.asarray(policy, dtype=float),
        "value": value,
    }


def run_unbounded_at_penalty(
    base_params,
    penalty,
    n_actions,
):

    p = dict(base_params)

    p["penalty"] = float(penalty)
    p["n_actions"] = int(n_actions)

    model = DormancyCTMDP_Unbounded_Fast(p)

    # Uses the same policy-iteration solver as your
    # action-grid/state-space convergence analysis.
    value, policy = solve_unbounded_policy_iteration(model)

    return {
        "model": model,
        "policy": np.asarray(policy, dtype=float),
        "value": np.asarray(value, dtype=float),
    }


def run_lp_at_penalty(
    model_class,
    base_params,
    beta,
    penalty,
    n_actions,
):

    p = dict(base_params)

    p["penalty"] = float(penalty)
    p["n_actions"] = int(n_actions)

    model = model_class(p, beta=float(beta))

    res = model.solve()

    if res is None or not res.success:

        if res is None:
            message = "unknown LP failure"
        else:
            message = res.message

        raise RuntimeError(f"LP failed for penalty={penalty:g}: " f"{message}")

    x = np.asarray(res.x, dtype=float)

    policy = model.expected_action_policy(x)

    return {
        "model": model,
        "res": res,
        "x": x,
        "policy": np.asarray(policy, dtype=float),
        "objective": model.discounted_reward_value(x),
        "regulatory_cost": model.discounted_regulatory_cost(x),
    }


def deterministic_penalty_error(
    policy,
    policy_ref,
    compare_max,
):

    p = np.asarray(policy, dtype=float)

    p_ref = np.asarray(policy_ref, dtype=float)

    diff = np.abs(p[1 : compare_max + 1] - p_ref[1 : compare_max + 1])

    return {
        "policy_linf": float(np.max(diff)),
        "policy_mean_abs": float(np.mean(diff)),
    }


def lp_penalty_error(
    model,
    x,
    model_ref,
    x_ref,
    compare_max,
):

    occ = lp_state_occupancy(model, x)

    occ_ref = lp_state_occupancy(model_ref, x_ref)

    a = model.expected_action_policy(x)

    a_ref = model_ref.expected_action_policy(x_ref)

    states = np.arange(1, compare_max + 1)

    occ_common = occ[states]
    occ_ref_common = occ_ref[states]

    a_common = a[states]
    a_ref_common = a_ref[states]

    # Use each solution's own scale when determining
    # whether the state is meaningfully occupied.
    tol = max(OCC_ABS_TOL, OCC_REL_TOL * np.max(occ))

    tol_ref = max(OCC_ABS_TOL, OCC_REL_TOL * np.max(occ_ref))

    mask = (occ_common > tol) & (occ_ref_common > tol_ref)

    if not np.any(mask):

        return {
            "policy_linf": np.nan,
            "policy_mean_abs": np.nan,
            "occupancy_weighted_mean": np.nan,
            "occupancy_coverage": 0.0,
        }

    diff = np.abs(a_common[mask] - a_ref_common[mask])

    weights = occ_ref_common[mask]

    weighted_mean = np.sum(weights * diff) / np.sum(weights)

    total_ref_occ = np.sum(occ_ref_common)

    if total_ref_occ > 0:

        coverage = np.sum(occ_ref_common[mask]) / total_ref_occ

    else:

        coverage = np.nan

    return {
        "policy_linf": float(np.max(diff)),
        "policy_mean_abs": float(np.mean(diff)),
        "occupancy_weighted_mean": float(weighted_mean),
        "occupancy_coverage": float(coverage),
    }


penalty_runs = {}


for spec in PENALTY_SPECS:

    print()
    print("=" * 70)
    print(spec["label"])
    print("=" * 70)

    runs = {}

    for penalty in PENALTY_VALUES:

        print(f"P_ext = {penalty:9.1e}", end=" ... ")

        if spec["solver"] == "deterministic":

            run = run_deterministic_at_penalty(
                spec["class"],
                spec["params"],
                penalty,
                spec["n_actions"],
            )

        elif spec["solver"] == "unbounded":

            run = run_unbounded_at_penalty(
                spec["params"],
                penalty,
                spec["n_actions"],
            )

        elif spec["solver"] == "lp":

            run = run_lp_at_penalty(
                spec["class"],
                spec["params"],
                spec["beta"],
                penalty,
                spec["n_actions"],
            )

        else:

            raise ValueError("Unknown solver type.")

        runs[float(penalty)] = run

        print("done")

    penalty_runs[spec["key"]] = runs


rows = []


for spec in PENALTY_SPECS:

    runs = penalty_runs[spec["key"]]

    ref = runs[float(PENALTY_REFERENCE)]

    compare_max = min(int(spec["compare_max"]), int(ref["model"].N) - 1)

    for penalty in PENALTY_VALUES:

        run = runs[float(penalty)]

        row = {
            "model_key": spec["key"],
            "model_label": spec["label"],
            "penalty": float(penalty),
            "penalty_reference": float(PENALTY_REFERENCE),
            "compare_max": compare_max,
        }

        if spec["solver"] in [
            "deterministic",
            "unbounded",
        ]:

            errors = deterministic_penalty_error(
                run["policy"],
                ref["policy"],
                compare_max,
            )

            row.update(errors)

            row["occupancy_weighted_mean"] = np.nan

            row["occupancy_coverage"] = np.nan

        else:

            errors = lp_penalty_error(
                run["model"],
                run["x"],
                ref["model"],
                ref["x"],
                compare_max,
            )

            row.update(errors)

        rows.append(row)


penalty_metrics = pd.DataFrame(rows)

display(penalty_metrics)


# Exclude the reference itself because it is zero by definition.
penalty_summary = penalty_metrics[
    penalty_metrics["penalty"] != PENALTY_REFERENCE
].copy()

display(
    penalty_summary[
        [
            "model_label",
            "penalty",
            "policy_linf",
            "policy_mean_abs",
            "occupancy_weighted_mean",
            "occupancy_coverage",
        ]
    ]
)


fig, ax = plt.subplots(figsize=(10, 7))


markers = [
    "o",
    "s",
    "^",
    "D",
    "v",
]

linestyles = [
    "-",
    "--",
    "-.",
    ":",
    "-",
]


# Exact zeros cannot be shown on log scale.
# This value is FOR VISUALIZATION ONLY.
VISUAL_FLOOR = 1e-16


# Slight horizontal separation on the log x-axis,
# analogous to your action-grid visualization.
x_factors = {
    "discounted_I": 0.91,
    "discounted_quadratic": 0.95,
    "unbounded_linear": 1.00,
    "constrained_linear": 1.05,
    "constrained_quadratic": 1.10,
}


for spec, marker, ls in zip(
    PENALTY_SPECS,
    markers,
    linestyles,
):

    d = penalty_metrics[penalty_metrics["model_key"] == spec["key"]].copy()

    # Reference discrepancy is zero by definition.
    d = d[d["penalty"] != d["penalty_reference"]]

    valid = np.isfinite(d["policy_linf"])

    P_true = d.loc[valid, "penalty"].to_numpy(dtype=float)

    y_true = d.loc[valid, "policy_linf"].to_numpy(dtype=float)

    # Visualization-only x displacement
    P_plot = P_true * x_factors[spec["key"]]

    # Visualization-only floor for exact zeros
    y_plot = np.where(y_true == 0.0, VISUAL_FLOOR, y_true)

    ax.loglog(
        P_plot,
        y_plot,
        marker=marker,
        linestyle=ls,
        linewidth=2.2,
        markersize=8,
        label=spec["label"],
    )


ax.set_xlabel(
    r"Extinction penalty $P_{\rm ext}$",
    fontsize=24,
    labelpad=10,
)

ax.set_ylabel(
    r"$\|\pi_{P_{\rm ext}}-\pi_{\rm ref}\|_\infty$",
    fontsize=24,
    labelpad=10,
)

ax.set_title(
    "Extinction-penalty convergence",
    fontsize=28,
    fontweight="bold",
    pad=15,
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=18,
)

ax.tick_params(
    axis="both",
    which="minor",
    labelsize=14,
)

ax.legend(
    fontsize=15,
    loc="best",
    frameon=True,
    edgecolor="black",
    framealpha=1.0,
    fancybox=False,
)

ax.grid(
    True,
    which="both",
    linestyle="--",
    linewidth=0.8,
    alpha=0.5,
)

plt.tight_layout()
plt.show()
