import matplotlib.pyplot as plt
import numpy as np

from apoptotic_control.analysis import first_zero_crossing, sweep
from apoptotic_control.models import DormancyCTMDP_DiscountedPI_V2
from apoptotic_control.parameters import FIGURE_S2
from apoptotic_control.plotting import finish, heatmap, panel_label, use_paper_style


def regulating_states(name, values):
    _, drifts = sweep(DormancyCTMDP_DiscountedPI_V2, FIGURE_S2, name, values)
    states = np.arange(FIGURE_S2["N"] + 1)
    return np.array([first_zero_crossing(states, drift) for drift in drifts])


def compute(quick=False):
    count = 12 if quick else 40
    c2 = np.linspace(0.0, 0.10, count)
    kappa = np.linspace(0.2, 10.0, count)
    c3 = np.linspace(0.2, 5.0, count)
    alpha = np.linspace(0.01, 0.30, count)
    c1 = np.linspace(0.001, 0.03, count)
    return {
        "c2": (c2, sweep(DormancyCTMDP_DiscountedPI_V2, FIGURE_S2, "c2", c2)[0]),
        "kappa": (
            kappa,
            sweep(DormancyCTMDP_DiscountedPI_V2, FIGURE_S2, "kappa", kappa)[0],
        ),
        "c3": (c3, sweep(DormancyCTMDP_DiscountedPI_V2, FIGURE_S2, "c3", c3)[0]),
        "alpha": (
            alpha,
            sweep(DormancyCTMDP_DiscountedPI_V2, FIGURE_S2, "alpha", alpha)[0],
        ),
        "horizon_c1": (c1, regulating_states("c1", c1)),
        "horizon_c3": (c3, regulating_states("c3", c3)),
    }


def plot(data):
    use_paper_style()
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    states = np.arange(1, 101)
    for ax, key, ylabel, title in (
        (axes[0, 0], "c2", r"$c_2$", r"Proliferation-size penalty $c_2$"),
        (axes[0, 1], "kappa", r"$\kappa_1$", r"Deviation penalty $\kappa_1$"),
        (axes[1, 0], "c3", r"$c_3$", r"Growth reward $c_3$"),
        (axes[1, 1], "alpha", r"$\alpha$", r"Discount rate $\alpha$"),
    ):
        values, policies = data[key]
        image = heatmap(ax, policies[:, 1:101], states, values)
        fig.colorbar(image, ax=ax, label=r"$\pi^*(i)$")
        ax.set(title=title, xlabel="Population state $i$", ylabel=ylabel)

    for ax, key, xlabel in (
        (axes[2, 0], "horizon_c1", r"$c_1$"),
        (axes[2, 1], "horizon_c3", r"$c_3$"),
    ):
        values, horizon = data[key]
        ax.plot(values, horizon, marker="o", markersize=3)
        ax.set(xlabel=xlabel, ylabel="Interior regulating state")
        ax.grid(alpha=0.25)

    for label, ax in zip("ABCDEF", axes.flat):
        panel_label(ax, f"{label}.")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    finish(plot(compute(args.quick)), "figures/supplementary/Figure_S2.png")
