import argparse

import matplotlib.pyplot as plt
import numpy as np

from apoptotic_control.analysis import per_capita_drift
from apoptotic_control.parameters import FIGURE_S4
from apoptotic_control.plotting import (
    finish,
    heatmap,
    panel_label,
    symmetric_limits,
    use_paper_style,
)
from apoptotic_control.solvers import solve_unbounded


def sweep(name, values):
    policies = []
    drifts = []
    for value in values:
        model, _, policy = solve_unbounded(dict(FIGURE_S4, **{name: float(value)}))
        policies.append(policy[1:201])
        drifts.append(per_capita_drift(model, policy)[1:201])
    return np.asarray(policies), np.asarray(drifts)


def compute(quick=False):
    count = 5 if quick else 10
    kappa = np.linspace(0.1, 10.0, count)
    c3 = np.linspace(0.1, 5.0, count)
    return kappa, c3, sweep("kappa", kappa), sweep("c3", c3)


def plot(data):
    kappa, c3, kappa_data, c3_data = data
    use_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    states = np.arange(1, 201)
    panels = (
        (kappa_data[0], kappa, "viridis", r"$\kappa_1$", "Policy"),
        (kappa_data[1], kappa, "RdBu_r", r"$\kappa_1$", "Drift"),
        (c3_data[0], c3, "viridis", r"$c_3$", "Policy"),
        (c3_data[1], c3, "RdBu_r", r"$c_3$", "Drift"),
    )
    for label, ax, panel in zip("ABCD", axes.flat, panels):
        values, parameters, cmap, ylabel, title = panel
        limits = symmetric_limits(values) if cmap == "RdBu_r" else (None, None)
        image = heatmap(
            ax, values, states, parameters, cmap=cmap, vmin=limits[0], vmax=limits[1]
        )
        fig.colorbar(image, ax=ax)
        ax.set(title=title, xlabel="Population state $i$", ylabel=ylabel)
        panel_label(ax, f"{label}.")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    finish(plot(compute(args.quick)), "figures/supplementary/Figure_S4.png")
