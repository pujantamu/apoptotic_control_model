import argparse

import matplotlib.pyplot as plt
import numpy as np

from apoptotic_control.parameters import (
    FIGURE_4_BETAS,
    FIGURE_4_CONSTRAINED,
    FIGURE_4_CONSTRAINED_QUADRATIC,
    FIGURE_4_QUADRATIC_BETAS,
)
from apoptotic_control.plotting import finish, panel_label, use_paper_style
from apoptotic_control.solvers import solve_constrained, solve_constrained_quadratic


def compute(quick=False):
    linear_betas = FIGURE_4_BETAS[:3] if quick else FIGURE_4_BETAS
    quadratic_betas = (
        FIGURE_4_QUADRATIC_BETAS[:3] if quick else FIGURE_4_QUADRATIC_BETAS
    )
    linear = [
        (beta, solve_constrained(FIGURE_4_CONSTRAINED, beta)) for beta in linear_betas
    ]
    quadratic = [
        (beta, solve_constrained_quadratic(FIGURE_4_CONSTRAINED_QUADRATIC, beta))
        for beta in quadratic_betas
    ]
    return linear, quadratic


def plot(data):
    linear, quadratic = data
    use_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for row, runs in enumerate((linear, quadratic)):
        for beta, run in runs:
            states = np.arange(run["model"].N + 1)
            axes[row, 0].plot(
                states, run["policy_plot"], linewidth=2, label=rf"$\beta={beta:g}$"
            )
            axes[row, 1].plot(
                states, run["drift_plot"], linewidth=2, label=rf"$\beta={beta:g}$"
            )
        axes[row, 0].axhline(
            runs[0][1]["model"].a_init,
            color="black",
            linestyle="--",
            label=r"Baseline $a^*$",
        )

    titles = (
        "Constrained model: policy comparison",
        "Constrained model: drift comparison",
        "Constrained quadratic: policy comparison",
        "Constrained quadratic: drift comparison",
    )
    ylabels = (
        r"Expected optimal action $\mathbb{E}[\pi\mid i]$",
        r"Per-capita drift $(\lambda-\mu)/i$",
        r"Expected optimal action $\mathbb{E}[\pi\mid i]$",
        r"Per-capita drift $(\lambda-\mu)/i$",
    )
    for label, ax, title, ylabel in zip("ABCD", axes.flat, titles, ylabels):
        ax.set(title=title, xlabel="Population state $i$", ylabel=ylabel, xlim=(1, 100))
        ax.grid(alpha=0.25)
        ax.legend()
        panel_label(ax, f"{label}.")
        ax.title.set_fontweight("bold")
    axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_ylim(0.38, 1.02)
    axes[0, 1].set_ylim(-0.05, 0.30)
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 1].set_ylim(-0.30, 0.40)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    finish(plot(compute(args.quick)), "figures/main/Figure_4.png")


if __name__ == "__main__":
    main()
