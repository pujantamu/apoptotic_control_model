import argparse

import matplotlib.pyplot as plt
import numpy as np

from apoptotic_control.analysis import per_capita_drift
from apoptotic_control.models import DormancyCTMDP_Unbounded_Fast
from apoptotic_control.parameters import (
    FIGURE_3,
    FIGURE_3_MAX_ITER,
    FIGURE_3_REGIMES,
)
from apoptotic_control.plotting import finish, panel_label, use_paper_style
from apoptotic_control.solvers import baseline_policy


def compute(quick=False):
    base = dict(FIGURE_3)
    max_iter = FIGURE_3_MAX_ITER
    if quick:
        base["N"] = 300
        max_iter = 400
    runs = []
    for kappa, c3, label in FIGURE_3_REGIMES:
        model = DormancyCTMDP_Unbounded_Fast(dict(base, kappa=kappa, c3=c3))
        _, policy = model.solve(max_iter=max_iter)
        runs.append((label, model, policy))
    return runs


def plot(runs):
    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, model, policy in runs:
        states = np.arange(model.N + 1)
        axes[0].plot(states[1:501], policy[1:501], linewidth=2, label=label)
        drift = per_capita_drift(model, policy)
        axes[1].plot(states[1:501], drift[1:501], linewidth=2, label=label)

    baseline = baseline_policy(runs[0][1].p)
    states = np.arange(runs[0][1].N + 1)
    axes[0].plot(states[1:501], baseline[1:501], "k--", label="Baseline $a^*$")
    axes[1].axhline(0, color="black", linestyle="--", label="Baseline $a^*$")
    axes[0].set(
        title="Unbounded linear-reward model: policy comparison",
        xlabel="Population state $i$",
        ylabel=r"Optimal action $\pi^*(i)$",
    )
    axes[1].set(
        title="Unbounded linear-reward model: drift comparison",
        xlabel="Population state $i$",
        ylabel=r"$(\lambda-\mu)/i$",
    )
    axes[0].set_ylim(0.34, 1.02)
    axes[1].set_ylim(-0.02, 0.32)
    for label, ax in zip("AB", axes):
        panel_label(ax, f"{label}.")
        ax.grid(alpha=0.25)
        ax.legend()
        ax.title.set_fontweight("bold")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    finish(plot(compute(args.quick)), "figures/main/Figure_3.png")


if __name__ == "__main__":
    main()
