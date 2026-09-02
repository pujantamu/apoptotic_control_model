import numpy as np

from .analysis import mask_unoccupied, per_capita_drift
from .models import (
    DormancyCTMDP_DiscountedPI,
    DormancyCTMDP_DiscountedPI_V2,
    DormancyCTMDP_Unbounded_Fast,
    TumorActorLP,
    TumorActorLP_v2,
)


def solve_threshold(params):
    model = DormancyCTMDP_DiscountedPI(params)
    value, policy = model.solve()
    return model, value, policy


def solve_quadratic(params):
    model = DormancyCTMDP_DiscountedPI_V2(params)
    policy = model.solve()
    return model, model.V.copy(), policy


def solve_unbounded(params):
    model = DormancyCTMDP_Unbounded_Fast(params)
    value, policy = solve_unbounded_policy_iteration(model)
    return model, value, policy


def evaluate_unbounded_policy(model, policy):
    lower = np.zeros(model.N + 1)
    diagonal = np.zeros(model.N + 1)
    upper = np.zeros(model.N + 1)
    right_hand_side = np.zeros(model.N + 1)

    for i in range(model.N + 1):
        action = float(policy[i])
        birth = model.lam(i, action)
        death = model.mu(i, action)
        diagonal[i] = model.alpha + birth + death
        right_hand_side[i] = (
            -model.penalty
            if i == 0
            else model.c3 * i - model.kappa * i * (action - model.a_star) ** 2
        )
        if i > 0:
            lower[i] = -death
        if i < model.N:
            upper[i] = -birth

    c_prime = np.zeros(model.N + 1)
    d_prime = np.zeros(model.N + 1)
    c_prime[0] = upper[0] / diagonal[0]
    d_prime[0] = right_hand_side[0] / diagonal[0]

    for i in range(1, model.N + 1):
        denominator = diagonal[i] - lower[i] * c_prime[i - 1]
        if abs(denominator) < 1e-14:
            raise RuntimeError(f"Near-singular solve at state {i}")
        if i < model.N:
            c_prime[i] = upper[i] / denominator
        d_prime[i] = (right_hand_side[i] - lower[i] * d_prime[i - 1]) / denominator

    value = np.zeros(model.N + 1)
    value[-1] = d_prime[-1]
    for i in range(model.N - 1, -1, -1):
        value[i] = d_prime[i] - c_prime[i] * value[i + 1]
    return value


def solve_unbounded_policy_iteration(model, max_iter=100):
    policy = np.full(model.N + 1, model.a_init)
    policy[0] = 1.0

    for _ in range(max_iter):
        value = evaluate_unbounded_policy(model, policy)
        new_policy = np.zeros(model.N + 1)
        new_policy[0] = 1.0

        for i in range(1, model.N + 1):
            previous_value = value[i - 1]
            next_value = value[i + 1] if i < model.N else value[i]
            actions = model.actions
            birth = model.lam(i, actions)
            death = model.mu(i, actions)
            reward = model.c3 * i - model.kappa * i * (actions - model.a_star) ** 2
            candidate = (reward + birth * next_value + death * previous_value) / (
                model.alpha + birth + death
            )
            new_policy[i] = actions[np.argmax(candidate)]

        if np.array_equal(new_policy, policy):
            policy = new_policy
            break
        policy = new_policy

    model.u = evaluate_unbounded_policy(model, policy)
    model.policy = policy
    return model.u.copy(), model.policy.copy()


def _solve_lp(model):
    result = model.solve()
    if not result.success:
        raise RuntimeError(result.message)
    policy = model.expected_action_policy(result.x)
    drift = per_capita_drift(model, policy)
    policy_plot, drift_plot, occupancy, support = mask_unoccupied(
        model, result, policy, drift
    )
    return {
        "model": model,
        "result": result,
        "policy": policy,
        "policy_plot": policy_plot,
        "drift": drift,
        "drift_plot": drift_plot,
        "occupancy": occupancy,
        "support": support,
    }


def solve_constrained(params, beta):
    return _solve_lp(TumorActorLP(params, beta=beta))


def solve_constrained_quadratic(params, beta):
    return _solve_lp(TumorActorLP_v2(params, beta=beta))


def baseline_policy(params):
    a_star = (params["delta"] + params.get("delta0", 0.0)) / (
        params["r"] + params["delta"]
    )
    policy = np.full(params["N"] + 1, np.clip(a_star, 0.0, 1.0))
    policy[0] = 1.0
    return policy
