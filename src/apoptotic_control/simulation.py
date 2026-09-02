import numpy as np


def gillespie(model, policy, i0=10, final_time=80.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    t = 0.0
    state = int(i0)
    times = [t]
    states = [state]

    while t < final_time and state > 0:
        action = float(policy[state])
        birth = model.lam(state, action)
        death = model.mu(state, action)
        rate = birth + death
        if rate <= 0:
            break

        t += rng.exponential(1.0 / rate)
        if t > final_time:
            break
        if rng.random() < birth / rate:
            state = min(state + 1, model.N)
        else:
            state -= 1
        times.append(t)
        states.append(state)

    return np.asarray(times), np.asarray(states)


def ensemble(model, policy, i0=10, final_time=80.0, n_paths=1000, n_time=501, seed=123):
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, final_time, n_time)
    paths = np.zeros((n_paths, n_time))
    for k in range(n_paths):
        event_time, states = gillespie(model, policy, i0, final_time, rng)
        indices = np.searchsorted(event_time, time, side="right") - 1
        paths[k] = states[np.clip(indices, 0, len(states) - 1)]
    return time, paths


def path_summary(paths):
    return {
        "q25": np.quantile(paths, 0.25, axis=0),
        "median": np.quantile(paths, 0.50, axis=0),
        "q75": np.quantile(paths, 0.75, axis=0),
    }
