import argparse

import matplotlib.pyplot as plt
import numpy as np

from apoptotic_control.analysis import near_zero_width, per_capita_drift
from apoptotic_control.parameters import FIGURE_1
from apoptotic_control.plotting import finish, panel_label, use_paper_style
from apoptotic_control.simulation import ensemble, path_summary
from apoptotic_control.solvers import (
    baseline_policy,
    solve_constrained,
    solve_constrained_quadratic,
    solve_quadratic,
    solve_threshold,
    solve_unbounded,
)


def parameters(name):
    return {**FIGURE_1["shared"], **FIGURE_1[name]}


def compute(n_paths=1000):
    threshold = solve_threshold(parameters("threshold"))
    quadratic = solve_quadratic(parameters("quadratic"))
    unbounded = solve_unbounded(parameters("unbounded"))

    p_linear = parameters("constrained")
    linear = solve_constrained(p_linear, p_linear.pop("beta"))
    p_quadratic = parameters("constrained_quadratic")
    constrained_quadratic = solve_constrained_quadratic(
        p_quadratic, p_quadratic.pop("beta")
    )

    runs = {
        "Baseline $a^*$": (threshold[0], baseline_policy(parameters("threshold"))),
        "Threshold cost": (threshold[0], threshold[2]),
        "Quadratic cost": (quadratic[0], quadratic[2]),
        "Unbounded cost": (unbounded[0], unbounded[2]),
        "Constrained cost": (linear["model"], linear["policy_plot"]),
        "Constrained quadratic cost": (
            constrained_quadratic["model"],
            constrained_quadratic["policy_plot"],
        ),
    }
    simulation_policies = {
        **runs,
        "Constrained cost": (linear["model"], linear["policy"]),
        "Constrained quadratic cost": (
            constrained_quadratic["model"],
            constrained_quadratic["policy"],
        ),
    }

    summaries = {}
    for j, (label, (model, policy)) in enumerate(simulation_policies.items()):
        time, paths = ensemble(
            model,
            policy,
            i0=10,
            final_time=80,
            n_paths=n_paths,
            n_time=501,
            seed=123 + 1000 * j,
        )
        summaries[label] = (time, path_summary(paths))

    return runs, summaries


def plot(runs, summaries):
    use_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = dict(zip(runs, ["0.45", "C0", "C1", "C2", "C3", "C4"]))

    widths = []
    for label, (model, policy) in runs.items():
        states = np.arange(model.N + 1)
        style = ":" if label.startswith("Baseline") else "-"
        axes[0, 0].plot(states, policy, style, color=colors[label], label=label)
        drift = per_capita_drift(model, policy)
        axes[0, 1].plot(states[1:], drift[1:], color=colors[label], label=label)
        widths.append(near_zero_width(drift, eps=0.02, start=1, stop=100))

    for label, (time, summary) in summaries.items():
        axes[1, 0].plot(time, summary["median"], color=colors[label], label=label)
        axes[1, 0].fill_between(
            time, summary["q25"], summary["q75"], color=colors[label], alpha=0.18
        )

    axes[1, 1].bar(list(runs), widths, edgecolor="black")
    axes[0, 0].set(title="Optimal policy", xlabel="State $i$", ylabel=r"$\pi^*(i)$")
    axes[0, 1].set(
        title="Normalized drift", xlabel="State $i$", ylabel=r"$(\lambda-\mu)/i$"
    )
    axes[1, 0].set(
        title="Stochastic trajectories",
        xlabel="Time",
        ylabel="Population size",
        ylim=(0, 100),
    )
    axes[1, 1].set(
        title=r"Near-zero-drift region for $i<100$",
        ylabel=r"Number of states with $|g(i)|\leq 0.02$",
    )
    axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_xlim(1, 99)
    axes[0, 1].set_xlim(1, 99)
    axes[1, 1].tick_params(axis="x", rotation=20)

    for label, ax in zip("ABCD", axes.flat):
        panel_label(ax, f"{label}.")
        ax.grid(alpha=0.25)
    for ax in axes.flat[:3]:
        ax.legend()

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    runs, summaries = compute(n_paths=50 if args.quick else 1000)
    finish(plot(runs, summaries), "figures/main/Figure_1.png")


if __name__ == "__main__":
    main()
