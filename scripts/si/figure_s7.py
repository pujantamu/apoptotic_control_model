import argparse

import matplotlib.pyplot as plt
import numpy as np

from apoptotic_control.models import DormancyCTMDP_Unbounded_Fast
from apoptotic_control.parameters import FIGURE_S7
from apoptotic_control.plotting import finish, panel_label, use_paper_style
from apoptotic_control.solvers import (
    solve_constrained,
    solve_constrained_quadratic,
    solve_quadratic,
    solve_threshold,
    solve_unbounded_policy_iteration,
)


CASES = (
    {
        "key": "threshold",
        "kind": "threshold",
        "action_label": "Threshold Cost",
        "state_label": "Threshold model",
        "penalty_label": "Threshold Model",
        "n_actions": 101,
        "compare_max": 75,
        "penalty_compare_max": 90,
    },
    {
        "key": "quadratic",
        "kind": "quadratic",
        "action_label": "Discounted quadratic",
        "state_label": "Discounted quadratic",
        "penalty_label": "Discounted quadratic",
        "n_actions": 101,
        "compare_max": 75,
        "penalty_compare_max": 90,
    },
    {
        "key": "unbounded",
        "kind": "unbounded",
        "action_label": "Unbounded linear reward",
        "state_label": "Unbounded linear reward",
        "penalty_label": "Unbounded linear reward",
        "n_actions": 201,
        "compare_max": 300,
        "penalty_compare_max": 720,
    },
    {
        "key": "constrained",
        "kind": "constrained",
        "action_label": "Constrained linear reward",
        "state_label": "Constrained linear reward",
        "penalty_label": "Constrained linear reward",
        "n_actions": 101,
        "compare_max": 75,
        "penalty_compare_max": 1350,
    },
    {
        "key": "constrained_quadratic",
        "kind": "constrained_quadratic",
        "action_label": "Constrained quadratic reward",
        "state_label": "Constrained quadratic reward",
        "penalty_label": "Constrained quadratic reward",
        "n_actions": 101,
        "compare_max": 75,
        "penalty_compare_max": 450,
    },
)

STATE_SIZES = {
    "threshold": (80, 100, 150, 200, 300),
    "quadratic": (80, 100, 150, 200, 300),
    "unbounded": (400, 600, 800, 1000, 1500),
    "constrained": (100, 200, 400, 500, 1000, 1500),
    "constrained_quadratic": (100, 200, 400, 500, 1000, 1500),
}


def solve_case(case, *, n_actions=None, size=None, penalty=None):
    params = dict(FIGURE_S7[case["key"]])
    beta = params.pop("beta", None)
    if n_actions is not None:
        params["n_actions"] = int(n_actions)
    if size is not None:
        params["N"] = int(size)
    if penalty is not None:
        params["penalty"] = float(penalty)

    kind = case["kind"]
    if kind == "threshold":
        model, value, policy = solve_threshold(params)
        return {"model": model, "value": value, "policy": policy}
    if kind == "quadratic":
        model, value, policy = solve_quadratic(params)
        return {"model": model, "value": value, "policy": policy}
    if kind == "unbounded":
        model = DormancyCTMDP_Unbounded_Fast(params)
        value, policy = solve_unbounded_policy_iteration(model)
        return {"model": model, "value": value, "policy": policy}

    if kind == "constrained":
        run = solve_constrained(params, beta)
    else:
        run = solve_constrained_quadratic(params, beta)
    return {
        "model": run["model"],
        "result": run["result"],
        "policy": run["policy"],
        "occupancy": run["occupancy"],
    }


def deterministic_error(policy, reference, states):
    difference = np.abs(np.asarray(policy)[states] - np.asarray(reference)[states])
    return float(np.max(difference))


def action_grid_lp_error(run, reference):
    occupancy = run["occupancy"]
    occupancy_ref = reference["occupancy"]
    scale = max(occupancy.max(), occupancy_ref.max(), 1e-14)
    cutoff = max(1e-14, 1e-10 * scale)
    support = (occupancy > cutoff) & (occupancy_ref > cutoff)
    support[0] = False
    support[-1] = False
    if not np.any(support):
        return np.nan
    return float(
        np.max(np.abs(run["policy"][support] - reference["policy"][support]))
    )


def state_space_lp_error(run, reference, compare_max):
    states = np.arange(1, compare_max + 1)
    occupancy = run["occupancy"]
    occupancy_ref = reference["occupancy"]
    cutoff = max(1e-12, 1e-8 * occupancy.max())
    cutoff_ref = max(1e-12, 1e-8 * occupancy_ref.max())
    support = (occupancy[states] > cutoff) & (occupancy_ref[states] > cutoff_ref)
    if not np.any(support):
        return np.nan
    states = states[support]
    return deterministic_error(run["policy"], reference["policy"], states)


def penalty_lp_error(run, reference, compare_max):
    states = np.arange(1, compare_max + 1)
    occupancy = run["occupancy"]
    occupancy_ref = reference["occupancy"]
    cutoff = max(1e-12, 1e-8 * occupancy.max())
    cutoff_ref = max(1e-12, 1e-8 * occupancy_ref.max())
    support = (occupancy[states] > cutoff) & (occupancy_ref[states] > cutoff_ref)
    if not np.any(support):
        return np.nan
    states = states[support]
    return deterministic_error(run["policy"], reference["policy"], states)


def action_grid_study(quick=False):
    grids = (26, 51, 101) if quick else (26, 51, 101, 201)
    output = {}
    for case in CASES:
        runs = [solve_case(case, n_actions=n) for n in grids]
        reference = runs[-1]
        errors = []
        for run in runs[:-1]:
            if case["kind"].startswith("constrained"):
                error = action_grid_lp_error(run, reference)
            else:
                states = np.arange(1, run["model"].N)
                error = deterministic_error(run["policy"], reference["policy"], states)
            errors.append(error)
        output[case["action_label"]] = (
            np.array([1.0 / (n - 1) for n in grids[:-1]]),
            np.asarray(errors),
        )
    return output


def state_space_study(quick=False):
    output = {}
    for case in CASES:
        sizes = STATE_SIZES[case["key"]]
        if quick:
            sizes = sizes[:3]
        runs = [
            solve_case(case, n_actions=case["n_actions"], size=size) for size in sizes
        ]
        reference = runs[-1]
        errors = []
        for run in runs[:-1]:
            compare_max = min(
                case["compare_max"], run["model"].N - 1, reference["model"].N - 1
            )
            if case["kind"].startswith("constrained"):
                error = state_space_lp_error(run, reference, compare_max)
            else:
                states = np.arange(1, compare_max + 1)
                error = deterministic_error(run["policy"], reference["policy"], states)
            errors.append(error)
        output[case["state_label"]] = (np.asarray(sizes[:-1]), np.asarray(errors))
    return output


def penalty_study(quick=False):
    penalties = np.array((1e2, 1e4, 1e5)) if quick else np.logspace(1, 7, 7)
    reference_penalty = 1e5
    output = {}
    for case in CASES:
        runs = {
            float(penalty): solve_case(
                case, n_actions=case["n_actions"], penalty=penalty
            )
            for penalty in penalties
        }
        reference = runs.get(reference_penalty)
        if reference is None:
            reference = solve_case(
                case, n_actions=case["n_actions"], penalty=reference_penalty
            )
        x_values = []
        errors = []
        compare_max = min(
            case["penalty_compare_max"], reference["model"].N - 1
        )
        for penalty in penalties:
            if penalty == reference_penalty:
                continue
            run = runs[float(penalty)]
            if case["kind"].startswith("constrained"):
                error = penalty_lp_error(run, reference, compare_max)
            else:
                states = np.arange(1, compare_max + 1)
                error = deterministic_error(run["policy"], reference["policy"], states)
            x_values.append(penalty)
            errors.append(error)
        output[case["penalty_label"]] = (np.asarray(x_values), np.asarray(errors))
    return output


def compute(quick=False):
    return action_grid_study(quick), state_space_study(quick), penalty_study(quick)


def plot(data):
    use_paper_style()
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 4)
    axes = (
        fig.add_subplot(grid[0, :2]),
        fig.add_subplot(grid[0, 2:]),
        fig.add_subplot(grid[1, 1:3]),
    )
    markers = ("o", "s", "^", "D", "v")
    linestyles = ("-", "--", "-.", ":", "-")

    shifts = (0.94, 0.97, 1.00, 1.03, 1.06)
    for (label, (x, y)), shift, marker, linestyle in zip(
        data[0].items(), shifts, markers, linestyles
    ):
        valid = np.isfinite(y) & (y > 0)
        axes[0].loglog(
            x[valid] * shift,
            y[valid],
            marker=marker,
            linestyle=linestyle,
            label=label,
        )

    offsets = (-20, 20, 0, 0, 0)
    for (label, (x, y)), offset, marker, linestyle in zip(
        data[1].items(), offsets, markers, linestyles
    ):
        valid = np.isfinite(y)
        values = np.where(y[valid] == 0.0, 1e-16, y[valid])
        axes[1].semilogy(
            x[valid] + offset,
            values,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )

    factors = (0.91, 0.95, 1.00, 1.05, 1.10)
    for (label, (x, y)), factor, marker, linestyle in zip(
        data[2].items(), factors, markers, linestyles
    ):
        valid = np.isfinite(y)
        values = np.where(y[valid] == 0.0, 1e-16, y[valid])
        axes[2].loglog(
            x[valid] * factor,
            values,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )

    axes[0].set(
        title="Action-grid convergence",
        xlabel="Action-grid spacing $h$",
        ylabel=r"$\|\pi_h-\pi_{\mathrm{ref}}\|_\infty$",
    )
    axes[1].set(
        title="State-space truncation convergence",
        xlabel="State-space truncation $N$",
        ylabel=r"$\|\pi_N-\pi_{\mathrm{ref}}\|_\infty$",
    )
    axes[2].set(
        title="Extinction-penalty convergence",
        xlabel=r"Extinction penalty $P_{\mathrm{ext}}$",
        ylabel=r"$\|\pi_{P_{\mathrm{ext}}}-\pi_{\mathrm{ref}}\|_\infty$",
    )
    for label, ax in zip("ABC", axes):
        panel_label(ax, f"{label}.")
        ax.title.set_fontweight("bold")
        ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(frameon=True, edgecolor="black")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    finish(plot(compute(args.quick)), "figures/supplementary/Figure_S7.png")
