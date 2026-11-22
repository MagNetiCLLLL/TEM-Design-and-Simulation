"""Plot the archived classical landing current at the fixed specimen upper surface."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
CASES = (
    ("before", "Before tuning", "#58687d"),
    ("rms", "RMS focus", "#007c75"),
    ("d95", "D95 focus", "#b44c28"),
)
ANNULUS_UM = (2.0, 6.0)


@dataclass
class Probe:
    label: str
    colour: str
    xy_um: np.ndarray
    current_na: np.ndarray
    d95_um: float


def read_probes():
    evidence = HERE / "evidence"
    summary = json.loads((evidence / "comparison.json").read_text(encoding="utf-8"))
    source_na = summary["gun"]["gun_exit_A"] * 1e9
    probes = []
    for key, label, colour in CASES:
        run = summary["runs"][key]
        file = evidence / run["trajectory_file"]
        if sha256(file.read_bytes()).hexdigest() != run["trajectory_sha256"]:
            raise ValueError(f"Unexpected specimen checkpoint for {label}")
        with np.load(file, allow_pickle=False) as data:
            if not np.any(data["z_m"] == summary["specimen_upper_surface_z_m"]):
                raise ValueError("Missing specimen surface in the observation grid")
            live = data["loss"] == -1
            np.testing.assert_equal(live.sum(), run["survived"])
            xy = data["phase"][live, :2] * 1e6
            current = data["weights"][live] * source_na
        if not np.isfinite(xy).all() or not np.all(current > 0):
            raise ValueError("A surviving particle has invalid coordinates or current")
        np.testing.assert_allclose(current.sum() / 1000, run["surface"]["transmitted_uA"], rtol=1e-12)
        centred = xy - np.average(xy, axis=0, weights=current)
        radius = np.hypot(*centred.T)
        order = np.argsort(radius)
        index = np.searchsorted(np.cumsum(current[order]), 0.95 * current.sum())
        diameter = 2 * radius[order[index]]
        np.testing.assert_allclose(diameter, run["surface"]["D95_nm"] / 1000, rtol=1e-10)
        probes.append(Probe(label, colour, xy, current, diameter))
    return probes, 1000 * summary["specimen_upper_surface_z_m"]


def landing_current_density(xy_um, current_na, half_width, pitch):
    edges = np.linspace(-half_width, half_width, round(2 * half_width / pitch) + 1)
    bins, _, _ = np.histogram2d(*xy_um.T, bins=(edges, edges), weights=current_na)
    inside = np.all(np.abs(xy_um) <= half_width, axis=1)
    np.testing.assert_allclose(bins.sum(), current_na[inside].sum(), rtol=1e-12)
    return bins.T / pitch**2, 100 * bins.sum() / current_na.sum()


def plot_shapes(probes, surface_mm):
    views = ((72, 1.0, [-60, -30, 0, 30, 60]), (8, 0.25, [-8, -4, 0, 4, 8]))
    maps = [[landing_current_density(p.xy_um, p.current_na, extent, pitch) for p in probes]
            for extent, pitch, _ in views]
    grids = [grid for row in maps for grid, _ in row]
    norm = LogNorm(vmin=min(g[g > 0].min() for g in grids), vmax=max(g.max() for g in grids))
    cmap = LinearSegmentedColormap.from_list(
        "landing_current", matplotlib.colormaps["inferno"](np.linspace(0.16, 0.99, 256))
    )
    cmap.set_bad("#0a0c12")

    fig = plt.figure(figsize=(11.1, 8.0), facecolor="white")
    layout = fig.add_gridspec(
        2,
        4,
        width_ratios=(1, 1, 1, 0.052),
        left=0.073,
        right=0.908,
        bottom=0.105,
        top=0.825,
        wspace=0.15,
        hspace=0.22,
    )
    fig.text(
        0.073,
        0.965,
        "Probe current density at the specimen upper surface",
        fontsize=18,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.073,
        0.915,
        f"Classical particle landing positions     ·     z = {surface_mm:.3f} mm"
        "     ·     300 keV     ·     100 µm C2 aperture",
        color="#454c59",
        fontsize=10.5,
    )
    for i, (extent, pitch, ticks) in enumerate(views):
        for j, probe in enumerate(probes):
            ax = fig.add_subplot(layout[i, j])
            density, shown = maps[i][j]
            image = ax.imshow(
                np.ma.masked_equal(density, 0),
                origin="lower",
                interpolation="nearest",
                extent=(-extent, extent, -extent, extent),
                cmap=cmap,
                norm=norm,
                aspect="equal",
            )
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            if j == 0:
                ax.set_ylabel("y (µm)")
            else:
                ax.tick_params(labelleft=False)
            if i == 0:
                ax.set_title(f"{probe.label}\nD95 {probe.d95_um:.2f} µm", pad=8, fontsize=11.5)
            else:
                ax.set_xlabel("x (µm)")
            ax.text(
                0.024,
                0.024,
                f"{shown:.2f}% in view",
                transform=ax.transAxes,
                fontsize=8,
                color="white",
                va="bottom",
                bbox=dict(fc="#0a0c12", ec="none", alpha=0.85, pad=2),
            )
    colour_ax = fig.add_subplot(layout[:, 3])
    colourbar = fig.colorbar(
        image, cax=colour_ax, ticks=[0.5, 2, 10, 50, 200, 1000], format=FuncFormatter(lambda x, _: f"{x:g}")
    )
    colourbar.set_label("Current density (nA/µm²)  ·  logarithmic scale", labelpad=7)
    fig.text(
        0.073,
        0.024,
        "Bin width is 1 µm in the upper row and 0.25 µm in the central enlargement."
        "   Black pixels contain zero recorded hits.",
        fontsize=9,
        color="#454c59",
    )
    fig.savefig(HERE / "specimen-probe-shape.png", dpi=210)
    plt.close(fig)


def plot_profiles(probes):
    fig, (radial, azimuth, modes) = plt.subplots(1, 3, figsize=(12.0, 3.95))
    fig.subplots_adjust(left=0.073, right=0.987, bottom=0.18, top=0.70, wspace=0.33)
    fig.text(
        0.073,
        0.965,
        "Radial extent and angular structure of the probe",
        fontsize=17,
        fontweight="bold",
        va="top",
    )
    # Include every surviving tail in the radial profile, even outside both displayed fields.
    outer = max(np.hypot(*p.xy_um.T).max() for p in probes)
    edges = np.r_[0.0, np.geomspace(0.12, np.ceil(outer), 146)]
    angle_edges = np.arange(0, 361, 5)
    multiplicity = np.arange(1, 7)
    for i, probe in enumerate(probes):
        x, y = probe.xy_um.T
        r = np.hypot(x, y)
        current, _ = np.histogram(r, bins=edges, weights=probe.current_na)
        np.testing.assert_allclose(current.sum(), probe.current_na.sum(), rtol=1e-12)
        density = current / (np.pi * np.diff(edges**2))
        radial.plot(
            0.5 * (edges[1:] + edges[:-1]),
            np.ma.masked_equal(density, 0),
            lw=1.35,
            color=probe.colour,
            label=probe.label,
        )
        ring = (r >= ANNULUS_UM[0]) & (r < ANNULUS_UM[1])
        theta = np.arctan2(y[ring], x[ring])
        weights = probe.current_na[ring]
        sectors, _ = np.histogram(np.degrees(theta) % 360, bins=angle_edges, weights=weights)
        np.testing.assert_allclose(sectors.sum(), weights.sum(), rtol=1e-12)
        azimuth.stairs(
            sectors / np.diff(angle_edges), angle_edges, baseline=None, color=probe.colour, linewidth=1.4
        )
        harmonics = np.abs(np.average(np.exp(1j * theta[:, None] * multiplicity), axis=0, weights=weights))
        modes.bar(multiplicity + (i - 1) * 0.24, harmonics, width=0.23, color=probe.colour)
    radial.set(
        xscale="log",
        yscale="log",
        xlabel="Radius from optical axis (µm)",
        ylabel="Mean current density (nA/µm²)",
        title="All specimen hits",
    )
    azimuth.set(
        xlim=(0, 360),
        ylim=(0, None),
        xticks=[0, 90, 180, 270, 360],
        xlabel="Azimuth from +x toward +y (°)",
        ylabel="Current (nA/degree)",
        title="Current in the 2–6 µm annulus",
    )
    modes.set(
        xlabel="Angular harmonic m",
        ylabel="Relative amplitude",
        xticks=multiplicity,
        ylim=(0, 0.8),
        title="Angular symmetry in the same annulus",
    )
    for ax in (radial, azimuth, modes):
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        *radial.get_legend_handles_labels(),
        loc="center left",
        bbox_to_anchor=(0.065, 0.815),
        ncols=3,
        frameon=False,
        columnspacing=2,
    )
    fig.savefig(HERE / "specimen-probe-profiles.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10})
    spots, plane_mm = read_probes()
    plot_shapes(spots, plane_mm)
    plot_profiles(spots)
