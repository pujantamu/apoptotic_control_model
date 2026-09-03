import matplotlib.pyplot as plt
import numpy as np

from apoptotic_control.analysis import per_capita_drift
from apoptotic_control.parameters import FIGURE_S1
from apoptotic_control.plotting import finish, panel_label, use_paper_style
from apoptotic_control.solvers import solve_threshold


def solve(**changes):
    return solve_threshold(dict(FIGURE_S1, **changes))


def plot():
    use_paper_style()
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    states = np.arange(1, FIGURE_S1["N"])
    a_star = (FIGURE_S1["delta"] + FIGURE_S1["delta0"]) / (
        FIGURE_S1["r"] + FIGURE_S1["delta"]
    )

    for value in (0.1, 1.0, 10.0, 50.0):
        model, _, policy = solve(kappa2=value)
        axes[0, 0].plot(states, policy[states], label=rf"$\kappa_2={value:g}$")
        drift = per_capita_drift(model, policy)
        axes[0, 1].plot(states, drift[states], label=rf"$\kappa_2={value:g}$")

    for value in (100.0, 1.0, 0.01):
        _, _, policy = solve(kappa1=value, kappa2=1.0)
        axes[1, 0].plot(states, policy[states], label=rf"$\kappa_1={value:g}$")

    for value in (10.0, 1.0, 0.1, 0.01):
        _, _, policy = solve(alpha=value)
        axes[1, 1].plot(states, policy[states], label=rf"$\alpha={value:g}$")

    for r, delta in ((0.1, 1.1), (0.3, 0.9), (0.9, 0.3), (1.1, 0.1)):
        _, _, policy = solve(r=r, delta=delta)
        axes[2, 0].plot(states, policy[states], label=rf"$r={r:g},\ \delta={delta:g}$")

    for value in (25.0, 50.0, 125.0):
        _, _, policy = solve(L=value)
        line = axes[2, 1].plot(
            states,
            policy[states],
            label=rf"$L={value:g}$ ($N={FIGURE_S1['N']}$)",
        )[0]
        axes[2, 1].axvline(value, color=line.get_color(), linestyle=":", alpha=0.7)

    titles = (
        r"Policy slices across $\kappa_2$",
        r"Drift slices across $\kappa_2$",
        r"Cost Ratio Regime: $\kappa_1$ vs $\kappa_2$",
        "Temporal Horizon Regime: Myopic vs Long-Horizon",
        "Drift Regime: Growth vs Decay Biased",
        "Capacity Regime: Threshold Position",
    )
    ylabels = (
        r"$\pi^*(i)$",
        r"Normalized net drift $(\lambda-\mu)/i$",
        r"$\pi^*(i)$",
        r"$\pi^*(i)$",
        r"$\pi^*(i)$",
        r"$\pi^*(i)$",
    )
    axes[0, 0].axhline(
        a_star, color="black", linestyle="--", linewidth=0.8, label=r"Baseline $a^*$"
    )
    axes[1, 0].axhline(
        a_star, color="black", linestyle="--", linewidth=0.8, label=r"Baseline $a^*$"
    )
    for ax in axes.flat[:5]:
        ax.axvline(
            FIGURE_S1["L"],
            color="0.6",
            linestyle=":",
            linewidth=0.8,
            label=r"$L$ threshold",
        )
    axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=0.8)

    for label, ax, title, ylabel in zip("ABCDEF", axes.flat, titles, ylabels):
        ax.set(title=title, xlabel="State $i$", ylabel=ylabel, xlim=(1, 200))
        ax.grid(alpha=0.25)
        ax.legend()
        panel_label(ax, f"{label}.")
        ax.title.set_fontweight("bold")
    axes[2, 1].set_ylim(0.25, 1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.parse_args()
    finish(plot(), "figures/supplementary/Figure_S1.png")
