"""Plot measured Si dark-field currents and the fixed registered native line."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_map(path):
    with np.load(path) as data:
        if "channel_current_a" in data:
            return data["coords_nm"], data["channel_current_a"][1] * 1e9
        if not np.array_equal(data["x_nm"], data["y_nm"]):
            raise ValueError("The reference acquisition is not on a square grid")
        return data["x_nm"], data["df_a"] * 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--map", dest="maps", nargs=2, metavar=("LABEL", "NPZ"), action="append", required=True
    )
    args = parser.parse_args()
    cases = [(name, *read_map(path)) for name, path in args.maps]
    widths = [np.diff(coordinate).mean() for _, coordinate, _ in cases]
    if not np.allclose(widths, 0.020):
        raise ValueError("Expected native 20 pm scan pitch")
    fig, axes = plt.subplots(
        2,
        len(cases),
        squeeze=False,
        figsize=(3.2 * len(cases), 5.6),
        gridspec_kw={"height_ratios": [1.22, 1.0]},
        layout="constrained",
    )
    limit = max(np.max(data) for _, _, data in cases)
    for col, (label, coord, data) in enumerate(cases):
        ax = axes[0, col]
        edge = [coord[0] - 0.01, coord[-1] + 0.01]
        im = ax.imshow(
            data,
            extent=[*edge, *edge],
            interpolation="nearest",
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=limit,
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Stage x (nm)")
        if col == 0:
            ax.set_ylabel("Stage y (nm)")
        ix = np.searchsorted(coord, 0.0)
        ax.plot([coord[ix]] * 2, [-0.25, 0.12], "-", color="cyan", lw=1)
        ax = axes[1, col]
        keep = (coord >= -0.26) & (coord <= 0.13)
        for site in (-0.13609255925, 0.0):
            ax.axvline(site, color=".6", lw=1, linestyle="--")
        ax.plot(coord[keep], data[keep, ix], "o-", ms=3.5, color="#384e78")
        ax.set_xlim(-0.26, 0.13)
        ax.set_ylim(0.0, limit * 1.03)
        ax.set_xlabel("Stage y (nm)")
        ax.grid(axis="y", alpha=0.22)
        if col == 0:
            ax.set_ylabel("Annular DF current (nA)")
    fig.colorbar(im, ax=list(axes[0]), label="Annular DF current (nA)", shrink=0.78, pad=0.025)
    fig.savefig(args.output, dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    main()
