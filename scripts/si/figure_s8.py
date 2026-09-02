import matplotlib.pyplot as plt
import numpy as np

from apoptotic_control.analysis import first_zero_crossing, per_capita_drift
from apoptotic_control.parameters import (
    FIGURE_S8_BETA,
    FIGURE_S8_CONSTRAINED,
    FIGURE_S8_QUADRATIC,
)
from apoptotic_control.plotting import finish, use_paper_style
from apoptotic_control.solvers import solve_constrained_quadratic, solve_quadratic


def logistic_fit(states, drift):
    capacity = first_zero_crossing(states, drift)
    mask = (states >= 1) & (states <= 100) & np.isfinite(drift)
    basis = 1.0 - states[mask] / capacity
    rho = np.sum(drift[mask] * basis) / np.sum(basis**2)
    return rho, capacity, rho * (1.0 - states / capacity)


def compute(quick=False):
    quadratic_params = dict(FIGURE_S8_QUADRATIC)
    constrained_params = dict(FIGURE_S8_CONSTRAINED)
    if quick:
        quadratic_params["N"] = 200
        constrained_params["N"] = 200

    model, _, policy = solve_quadratic(quadratic_params)
    states = np.arange(model.N + 1)
    drift = per_capita_drift(model, policy)

    constrained = solve_constrained_quadratic(constrained_params, FIGURE_S8_BETA)
    constrained_drift = constrained["drift_plot"]
    return (
        states,
        ("Quadratic", drift, logistic_fit(states, drift)),
        (
            rf"Constrained quadratic, $\beta={FIGURE_S8_BETA:g}$",
            constrained_drift,
            logistic_fit(states, constrained_drift),
        ),
    )


def plot(data):
    states, *curves = data
    use_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, drift, (rho, capacity, reference) in curves:
        mask = (states >= 1) & (states <= 100) & np.isfinite(drift)
        line = ax.plot(states[mask], drift[mask], label=label)[0]
        ax.plot(
            states[mask],
            reference[mask],
            "--",
            color=line.get_color(),
            label=rf"Logistic fit: $\rho={rho:.3g}$, $K={capacity:.1f}$",
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set(
        title="Logistic-like comparison of induced drift",
        xlabel="Population state $i$",
        ylabel=r"Per-capita drift $g(i)$",
        xlim=(1, 100),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    finish(plot(compute(args.quick)), "figures/supplementary/Figure_S8.png")
