"""Plot measured current containment and axial D95 at two source populations."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def plot(smaller, larger, target, first_count=None):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.3, 3.8), constrained_layout=True)
    for path, count, color in ((smaller, first_count, "#b46026"), (larger, None, "#146678")):
        with np.load(path) as particles:
            keep = slice(None) if count is None else particles["ray_id"] < count
            weight = particles["weight"][keep]
            xy = particles["phase"][keep, :2]
            centre = np.average(xy, axis=0, weights=weight)
            diameter = 2e9 * np.linalg.norm(xy - centre, axis=1)
            order = np.argsort(diameter)
            cumulative = np.cumsum(weight[order]) / weight.sum()
            left.step(diameter[order], cumulative, where="post", color=color, lw=1.5)
            key = "focus_D95_nm" if count is None else f"focus_D95_nm_{count}"
            z, d95 = particles["focus_offset_nm"], particles[key]
            near = abs(z) <= 22.0
            right.plot(z[near], d95[near], lw=1.6, color=color, label=f"{len(weight):,} originals")
            at_min = np.argmin(d95)
            right.scatter(z[at_min], d95[at_min], color=color, zorder=3, s=19)
    left.set(
        xscale="log",
        xlim=(0.05, 110),
        ylim=(0, 1.02),
        xlabel="Centred containment diameter (nm)",
        ylabel="Fraction of transmitted current",
        title="At the specimen upper surface",
    )
    left.axhline(0.95, ls="--", color=".38", lw=0.9)
    left.text(0.058, 0.925, "95%", va="top", fontsize=8, color=".28")
    left.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    right.axvline(0, ls="--", color=".38", lw=0.9)
    right.set(
        xlim=(-22, 22),
        xlabel="Axial distance from upper surface (nm)",
        ylabel="D95 (nm)",
        title="Through-focus probe diameter",
    )
    right.legend(frameon=False)
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color=".9", lw=0.6)
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smaller", type=Path)
    parser.add_argument("larger", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--first-count", type=int, help="Read the specified nested original cohort from the first file")
    args = parser.parse_args()
    plot(args.smaller, args.larger, args.output, args.first_count)
