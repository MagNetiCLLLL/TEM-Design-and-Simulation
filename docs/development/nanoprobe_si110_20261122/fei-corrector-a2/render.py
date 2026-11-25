"""Plot absolute-current histograms at the physical surface and the measured focal planes."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent


def render(folder):
    import matplotlib as mpl

    with np.load(folder / "probe-planes.npz") as f:
        before = f["before_xy_um"]
        after = [f[key][:, :2] * 1e6 for key in ("surface", "D95", "rms")]
        w = f["weights_uA"] * 1e6  # pA
        through = f["profiles"]
    report = json.loads((folder / "evidence.json").read_text())
    ws = report["waists"]
    metrics = report["after"]
    planes = ws["planes"]
    surface = ws["specimen_z_mm"]
    labels = [
        "Before tuning  ·  specimen",
        "After tuning  ·  specimen",
        "After tuning  ·  D95 minimum",
        "After tuning  ·  radial RMS minimum",
    ]
    positions = [surface, surface, planes["D95"]["z_mm"], planes["rms"]["z_mm"]]
    diam = [
        report["before"]["D95_um"],
        metrics["D95_um"],
        planes["D95"]["widths_um"][0],
        planes["rms"]["widths_um"][0],
    ]
    # Identical physical axes and current scale for every panel. All arrivals enter each metric.
    half = max(0.50, np.ceil(1.35 * max(diam) * 0.5 / 0.10) * 0.10)
    pixel = 0.025
    edges = np.linspace(-half, half, round(2 * half / pixel) + 1)
    points = [before, *after]
    maps = [np.histogram2d(*q.T, bins=(edges, edges), weights=w)[0].T for q in points]
    norm = mpl.colors.Normalize(vmin=0, vmax=max(h.max() for h in maps))
    fig = Figure(figsize=(9.6, 8.1), layout="constrained")
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05])
    fig.suptitle(
        "HPol A2 adjustment  |  current at the measured planes",
        fontweight="bold",
        fontsize=15,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    for ax, q, h, label, z, d in zip(axes, points, maps, labels, positions, diam):
        im = ax.imshow(
            h,
            origin="lower",
            extent=[-half, half, -half, half],
            interpolation="nearest",
            norm=norm,
            cmap="magma",
        )
        centre = np.average(q, axis=0, weights=w)
        ax.add_patch(
            Circle(centre, d / 2, fill=False, ec="#83d7d1", lw=1.1, ls=(0, (3, 3)))
        )
        ax.set_title(label, loc="left", fontweight="bold", fontsize=11, pad=20)
        ax.text(
            0,
            1.013,
            f"z = {z:.5f} mm    |    D95 = {d*1000:.1f} nm",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.text(
            0.025,
            0.025,
            f"{100*h.sum()/w.sum():.2f}% of current shown",
            color="white",
            fontsize=7.5,
            transform=ax.transAxes,
        )
        ax.set(xlabel="Laboratory x (µm)", ylabel="Laboratory y (µm)", aspect="equal")
    bar = fig.colorbar(im, cax=fig.add_subplot(grid[:, 2]))
    bar.set_label(f"Current per {pixel*1000:.0f} × {pixel*1000:.0f} nm pixel (pA)")
    fig.savefig(folder / "probe-waists.png", dpi=165)
    fig = Figure(figsize=(9.2, 4.2), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    dz = (through[:, 0] - surface) * 1000
    extent = max(65.0, 1.8 * max(abs(planes[k]["offset_um"]) for k in ("D95", "rms")))
    keep = abs(dz) <= extent
    ax.plot(
        dz[keep], through[keep, 1] * 1000, color="#154e77", lw=1.6, label="D95 diameter"
    )
    ax.plot(
        dz[keep],
        2 * through[keep, 2] * 1000,
        color="#b75a20",
        lw=1.6,
        label="Twice radial RMS radius",
    )
    ax.axvline(0, color="#333", lw=0.9, ls="--", label="Specimen upper surface")
    for index, key, color in [(0, "D95", "#154e77"), (1, "rms", "#b75a20")]:
        p = planes[key]
        height = p["widths_um"][index] * 1000 * (1 if index == 0 else 2)
        ax.scatter([p["offset_um"]], [height], color=color, s=35, zorder=4)
        ax.annotate(
            f"{p['offset_um']:+.2f} µm",
            (p["offset_um"], height),
            xytext=(6, 13),
            textcoords="offset points",
            color=color,
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
        )
    ax.set(
        xlabel="Distance from specimen upper surface (µm, downstream positive)",
        ylabel="Diameter (nm)",
        xlim=(-extent, extent),
    )
    ax.set_title(
        "Final objective focus  |  same transmitted primaries",
        fontweight="bold",
        loc="left",
    )
    ax.grid(alpha=0.18)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.savefig(folder / "axial-focus.png", dpi=165)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", type=Path, default=HERE)
    render(parser.parse_args().folder)
