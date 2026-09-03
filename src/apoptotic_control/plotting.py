import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def finish(fig, output):
    output = Path(output)
    if root := os.environ.get("APOPTOTIC_OUTPUT_ROOT"):
        output = Path(root) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.15)
    return output


def panel_label(ax, label):
    ax.text(-0.12, 1.03, label, transform=ax.transAxes, weight="bold", size=15)


def heatmap(ax, values, states, parameters, *, cmap="viridis", vmin=None, vmax=None):
    return ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        extent=[states[0], states[-1], parameters[0], parameters[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


def symmetric_limits(values):
    bound = np.nanmax(np.abs(values))
    return -bound, bound


def use_paper_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )
