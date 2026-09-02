import argparse

import matplotlib.pyplot as plt
import numpy as np

from apoptotic_control.parameters import FIGURE_S5
from apoptotic_control.plotting import (
    finish,
    heatmap,
    panel_label,
    symmetric_limits,
    use_paper_style,
)
from apoptotic_control.solvers import solve_constrained


def compute(quick=False):
    beta = np.linspace(0.005, 5.0, 6 if quick else 20)
    policies = []
    drifts = []
    for value in beta:
        result = solve_constrained(FIGURE_S5, value)
        policies.append(result["policy_plot"][1:101])
        drifts.append(result["drift_plot"][1:101])
    return beta, np.asarray(policies), np.asarray(drifts)


def plot(data):
    beta, policies, drifts = data
    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    states = np.arange(1, 101)
    image = heatmap(axes[0], policies, states, beta)
    fig.colorbar(image, ax=axes[0], label=r"$\mathrm{E}[\pi^*\mid i]$")
    limits = symmetric_limits(drifts)
    image = heatmap(
        axes[1], drifts, states, beta, cmap="RdBu_r", vmin=limits[0], vmax=limits[1]
    )
    fig.colorbar(image, ax=axes[1], label=r"$(\lambda-\mu)/i$")
    for label, ax, title in zip("AB", axes, ("Policy", "Normalized drift")):
        ax.set(title=title, xlabel="Population state $i$", ylabel=r"Budget $\beta$")
        panel_label(ax, f"{label}.")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    finish(plot(compute(args.quick)), "figures/supplementary/Figure_S5.png")
