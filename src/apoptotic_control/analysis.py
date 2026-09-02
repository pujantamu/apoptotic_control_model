import numpy as np


def per_capita_drift(model, policy):
    policy = np.asarray(policy, dtype=float)
    drift = np.full(model.N + 1, np.nan)
    for i in range(1, model.N + 1):
        drift[i] = model.drift(i, policy[i])
    return drift


def state_occupancy(model, result):
    x = np.asarray(result.x, dtype=float).reshape(model.N + 1, model.n_actions)
    return np.maximum(x, 0.0).sum(axis=1)


def occupied_states(model, result, rel_tol=1e-8, abs_tol=1e-12):
    occupancy = state_occupancy(model, result)
    cutoff = max(abs_tol, rel_tol * occupancy.max())
    mask = occupancy > cutoff
    mask[0] = False
    return occupancy, mask


def mask_unoccupied(model, result, *arrays, rel_tol=1e-8, abs_tol=1e-12):
    occupancy, mask = occupied_states(model, result, rel_tol, abs_tol)
    masked = []
    for array in arrays:
        values = np.asarray(array, dtype=float).copy()
        values[~mask] = np.nan
        masked.append(values)
    return (*masked, occupancy, mask)


def first_zero_crossing(states, drift):
    states = np.asarray(states)
    drift = np.asarray(drift)
    valid = np.isfinite(drift)
    x = states[valid]
    y = drift[valid]
    if len(y) < 2:
        return np.nan
    crossings = np.where(y[:-1] * y[1:] <= 0)[0]
    if len(crossings) == 0:
        return np.nan
    j = crossings[0]
    if y[j + 1] == y[j]:
        return float(x[j])
    return float(x[j] - y[j] * (x[j + 1] - x[j]) / (y[j + 1] - y[j]))


def near_zero_width(drift, eps=0.02, start=1, stop=None):
    drift = np.asarray(drift, dtype=float)
    stop = len(drift) if stop is None else min(stop, len(drift))
    values = drift[start:stop]
    return int(np.sum(np.isfinite(values) & (np.abs(values) <= eps)))


def sweep(model_class, params, name, values):
    policies = []
    drifts = []
    for value in values:
        run_params = dict(params, **{name: float(value)})
        model = model_class(run_params)
        solved = model.solve()
        policy = solved[1] if isinstance(solved, tuple) else solved
        policies.append(policy)
        drifts.append(per_capita_drift(model, policy))
    return np.asarray(policies), np.asarray(drifts)
