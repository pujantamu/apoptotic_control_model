import matplotlib.pyplot as plt
import numpy as np

from apoptotic_control.analysis import sweep
from apoptotic_control.models import (
    DormancyCTMDP_DiscountedPI,
    DormancyCTMDP_DiscountedPI_V2,
)
from apoptotic_control.parameters import FIGURE_2_QUADRATIC, FIGURE_2_THRESHOLD
from apoptotic_control.plotting import (
    finish,
    heatmap,
    panel_label,
    symmetric_limits,
    use_paper_style,
)


def compute():
    kappa2 = np.linspace(0.1, 12.0, 40)
    c1 = np.linspace(0.001, 0.03, 40)
    threshold = sweep(DormancyCTMDP_DiscountedPI, FIGURE_2_THRESHOLD, "kappa2", kappa2)
    quadratic = sweep(DormancyCTMDP_DiscountedPI_V2, FIGURE_2_QUADRATIC, "c1", c1)
    return kappa2, c1, threshold, quadratic


def plot(data):
    kappa2, c1, threshold, quadratic = data
    use_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    states = np.arange(1, 101)

    panels = (
        (threshold[0][:, 1:101], kappa2, "viridis", None, None),
        (
            threshold[1][:, 1:101],
            kappa2,
            "RdBu_r",
            *symmetric_limits(threshold[1][:, 1:101]),
        ),
        (quadratic[0][:, 1:101], c1, "viridis", None, None),
        (
            quadratic[1][:, 1:101],
            c1,
            "RdBu_r",
            *symmetric_limits(quadratic[1][:, 1:101]),
        ),
    )
    titles = (
        "Threshold model: optimal policy",
        "Threshold model: normalized drift",
        "Quadratic cost model: optimal policy",
        "Quadratic cost model: normalized drift",
    )
    ylabels = (r"$\kappa_2$", r"$\kappa_2$", r"$c_1$", r"$c_1$")

    for label, ax, panel, title, ylabel in zip(
        "ABCD", axes.flat, panels, titles, ylabels
    ):
        values, parameter, cmap, vmin, vmax = panel
        image = heatmap(ax, values, states, parameter, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(image, ax=ax)
        ax.set(title=title, xlabel="Population state $i$", ylabel=ylabel)
        panel_label(ax, f"{label}.")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    finish(plot(compute()), "figures/main/Figure_2.png")
