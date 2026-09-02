import argparse

import matplotlib.pyplot as plt
import numpy as np

from apoptotic_control.parameters import FIGURE_1
from apoptotic_control.plotting import finish, panel_label, use_paper_style
from apoptotic_control.solvers import (
    solve_constrained,
    solve_constrained_quadratic,
    solve_quadratic,
    solve_threshold,
    solve_unbounded,
)

CASES = {
    "Threshold model": ("threshold", {**FIGURE_1["shared"], **FIGURE_1["threshold"]}),
    "Discounted quadratic": (
        "quadratic",
        {**FIGURE_1["shared"], **FIGURE_1["quadratic"]},
    ),
    "Unbounded linear reward": (
        "unbounded",
        {**FIGURE_1["shared"], **FIGURE_1["unbounded"]},
    ),
    "Constrained linear reward": (
        "constrained",
        {**FIGURE_1["shared"], **FIGURE_1["constrained"]},
    ),
    "Constrained quadratic reward": (
        "constrained_quadratic",
        {**FIGURE_1["shared"], **FIGURE_1["constrained_quadratic"]},
    ),
}


def solve_case(kind, params):
    params = dict(params)
    beta = params.pop("beta", None)
    if kind == "threshold":
        model, value, policy = solve_threshold(params)
        return model, policy, value, None
    if kind == "quadratic":
        model, value, policy = solve_quadratic(params)
        return model, policy, value, None
    if kind == "unbounded":
        model, value, policy = solve_unbounded(params)
        return model, policy, value, None
    if kind == "constrained":
        run = solve_constrained(params, beta)
    else:
        run = solve_constrained_quadratic(params, beta)
    return run["model"], run["policy_plot"], None, run["support"]


def policy_error(run, reference, stop=75):
    _, policy, _, support = run
    _, policy_ref, _, support_ref = reference
    stop = min(stop, len(policy) - 1, len(policy_ref) - 1)
    states = np.arange(1, stop + 1)
    mask = np.isfinite(policy[states]) & np.isfinite(policy_ref[states])
    if support is not None:
        mask &= support[states] & support_ref[states]
    if not np.any(mask):
        return np.nan
    return float(np.max(np.abs(policy[states][mask] - policy_ref[states][mask])))


def action_grid_study(quick=False):
    grids = (26, 51, 101) if quick else (26, 51, 101, 201)
    output = {}
    for label, (kind, base) in CASES.items():
        runs = [solve_case(kind, dict(base, n_actions=n)) for n in grids]
        output[label] = (
            np.array([1 / (n - 1) for n in grids[:-1]]),
            [policy_error(run, runs[-1]) for run in runs[:-1]],
        )
    return output


def state_space_study(quick=False):
    sizes = (80, 120, 200) if quick else (80, 100, 150, 200, 300)
    output = {}
    for label, (kind, base) in CASES.items():
        if kind in {"constrained", "constrained_quadratic"} and not quick:
            model_sizes = (100, 200, 400, 500)
        elif kind == "unbounded" and not quick:
            model_sizes = (400, 600, 800, 1000)
        else:
            model_sizes = sizes
        runs = [solve_case(kind, dict(base, N=n)) for n in model_sizes]
        output[label] = (
            np.array(model_sizes[:-1]),
            [policy_error(run, runs[-1]) for run in runs[:-1]],
        )
    return output


def penalty_study(quick=False):
    penalties = np.array((1e2, 1e4, 1e5)) if quick else np.logspace(1, 7, 7)
    reference_penalty = 1e5
    output = {}
    for label, (kind, base) in CASES.items():
        reference = solve_case(kind, dict(base, penalty=reference_penalty))
        output[label] = (
            penalties,
            [
                policy_error(solve_case(kind, dict(base, penalty=p)), reference)
                for p in penalties
            ],
        )
    return output


def compute(quick=False):
    return action_grid_study(quick), state_space_study(quick), penalty_study(quick)


def plot(data):
    use_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[1, 1].axis("off")
    for ax, study in zip((axes[0, 0], axes[0, 1], axes[1, 0]), data):
        for label, (x, y) in study.items():
            y = np.maximum(np.asarray(y, dtype=float), 1e-16)
            ax.plot(x, y, marker="o", label=label)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
        ax.legend()
    axes[0, 0].set(
        xscale="log",
        title="Action-grid convergence",
        xlabel="Action-grid spacing $h$",
        ylabel=r"$\|\pi-\pi_{ref}\|_\infty$",
    )
    axes[0, 1].set(
        title="State-space truncation convergence",
        xlabel="State-space truncation $N$",
        ylabel=r"$\|\pi-\pi_{ref}\|_\infty$",
    )
    axes[1, 0].set(
        xscale="log",
        title="Extinction-penalty convergence",
        xlabel=r"Extinction penalty $P_{ext}$",
        ylabel=r"$\|\pi-\pi_{ref}\|_\infty$",
    )
    for label, ax in zip("ABC", (axes[0, 0], axes[0, 1], axes[1, 0])):
        panel_label(ax, f"{label}.")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    finish(plot(compute(args.quick)), "figures/supplementary/Figure_S7.png")
