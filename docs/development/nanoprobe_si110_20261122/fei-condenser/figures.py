"""Render the measured column paths, landing currents, pupil angles and axial focus."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = ["#554b97", "#167f83", "#cf623d", "#497bb0"]
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "font.family": "DejaVu Sans",
        "savefig.dpi": 165,
    }
)


def open_case(name):
    folder = HERE / name
    rec = json.loads((folder / "result.json").read_text())
    with np.load(folder / "trajectories.npz") as d:
        data = dict(d)
    return rec, data


def lenses(axis, which):
    for a, b, n in which:
        axis.axvspan(a, b, color="#64748b", alpha=0.10, lw=0)
        axis.text(
            (a + b) / 2, 0.96, n, transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=8
        )
    axis.axhline(0, color="#737880", lw=0.6)
    axis.grid(axis="y", alpha=0.18)


def rays(name, filename="ray-diagram.png"):
    rec, d = open_case(name)
    g = np.load(HERE / name / "gun-paths.npz")
    np.testing.assert_array_equal(d["selected_ids"], g["selected_ids"])
    z = np.r_[g["z_mm"], d["z_mm"][1:]]
    xy = np.concatenate((g["xy_m"].transpose(1, 0, 2), d["xy_m"][1:]), axis=0) * 1e3
    survived = np.isin(d["selected_ids"], d["ids"][d["loss"] == -1])
    fig, (full, zoom, rad) = plt.subplots(
        3, 1, figsize=(10, 8), layout="constrained", height_ratios=[1, 1, 0.8]
    )
    fig.suptitle(
        f"Nanoprobe Normal   |   C1 {rec['c1_A']:+g} A    C2 {rec['c2_A']:+g} A    gun lens +50 V",
        x=0.06,
        ha="left",
        fontweight="bold",
        fontsize=15,
    )
    for ax in (full, zoom):
        for j in range(len(survived)):
            color = COLORS[j % len(COLORS)] if survived[j] else "#9ba2a8"
            ax.plot(z, xy[:, j, 0], lw=0.55, alpha=0.58 if survived[j] else 0.23, color=color)
        ax.set_ylabel("Ray x (mm)")
    lenses(
        full,
        [
            (0, 36, "Gun"),
            (450, 550, "C1"),
            (550, 750, "C2"),
            (830, 1000, "C3"),
            (1070, 1106.4, "ADL"),
            (1129.5, 1379.0, "Corrector"),
            (1380, 1420, "Scan"),
            (1512.5, 1596.5, "Mini / OL"),
        ],
    )
    full.set_xlim(0, 1600)
    full.set_title("A   Tracked paths from the physical tip", fontsize=11)
    full.axvline(765, color="#29374c", lw=0.9, ls=":")
    lens = [
        (1129.5, 1159.5, "HP2"),
        (1153.6, 1179.1, "TL22"),
        (1237.2, 1262.7, "TL21"),
        (1285.85, 1315.85, "HP1"),
        (1353.6, 1379.0, "TL12"),
        (1380, 1395, "Upper scan"),
        (1405, 1420, "Lower scan"),
        (1512.5, 1532.5, "Mini"),
        (1534.5, 1596.5, "OL"),
    ]
    # Stagger neighbouring hardware names in the zoom to keep both visible.
    lenses(zoom, [])
    for j, (a, b, label) in enumerate(lens):
        zoom.axvspan(a, b, color="#64748b", alpha=0.10, lw=0)
        zoom.text(
            (a + b) / 2,
            0.99 - (0.09 if j in (1, 5) else 0),
            label,
            transform=zoom.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )
    zoom.set_title("B   Corrector to specimen upper surface", fontsize=11)
    zoom.set_xlim(1120, 1601)
    extent = np.nanmax(np.abs(xy[(z >= 1120) & (z <= 1601), :, 0]))
    zoom.set_ylim(-1.22 * extent, 1.22 * extent)
    zz = d["z_mm"]
    r = 1000 * np.sqrt(d["stats_m"][:, 0] + d["stats_m"][:, 1])
    rad.plot(zz, r, color="#223b55", lw=1.7, label="Live current-weighted RMS radius")
    required = [(1179.10, 1237.20), (1420.0, 1512.5)]
    for ax in (zoom, rad):
        for i, (a, b) in enumerate(required):
            ax.axvspan(
                a,
                b,
                color="#77c386",
                alpha=0.20,
                lw=0,
                label="Required waist interval" if ax == rad and i == 0 else None,
            )
        ax.axvline(1599.1999975, color="#bc5145", lw=1.0, ls="--")
    for root in rec["waists"]["live"]["radial_mm"]:
        if root > 1120:
            h = np.interp(root, zz, r)
            rad.scatter(root, h, s=27, color="#ae503a", zorder=4)
            rad.annotate(
                f"{root:.1f}",
                (root, h),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    rad.set_xlim(1120, 1601)
    rad.set_ylim(bottom=0, top=r[zz > 1120].max() * 1.25)
    rad.set_title("C   Two-dimensional beam envelope", fontsize=11)
    rad.set_ylabel("RMS radius (mm)")
    rad.set_xlabel("Distance from tip z (mm)")
    rad.legend(loc="upper left", frameon=False, fontsize=8)
    rad.grid(axis="y", alpha=0.2)
    fig.savefig(HERE / filename)
    plt.close(fig)


def landings(names):
    items = [open_case(n) for n in names]
    maps = []
    for rec, d in items:
        live = d["loss"] == -1
        p = d["surface_phase"][live, :2] * 1e6
        w = d["weights_uA"][live] * 1e3
        width = 1.24 * max(rec["surface"]["D95_um"] / 2, np.max(abs(np.average(p, weights=w, axis=0))))
        bins = np.linspace(-width, width, 49)
        H, *_ = np.histogram2d(*p.T, bins=(bins, bins), weights=w)
        maps.append((bins, H / np.diff(bins)[0] ** 2))
    positive = np.concatenate([a[a > 0] for _, a in maps])
    norm = LogNorm(vmin=positive.min(), vmax=positive.max())
    ncols = 2 if len(items) == 4 else len(items)
    nrows = (len(items) + ncols - 1) // ncols
    fig, grid = plt.subplots(
        nrows, ncols, figsize=(3.8 * ncols + 0.9, 3.7 * nrows + 0.7), layout="constrained", squeeze=False
    )
    axes = grid.ravel()
    fig.suptitle(
        "Primary landing current at the specimen upper surface",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    for ax, (rec, d), (bins, H) in zip(axes, items, maps):
        ax.set_facecolor("#f1f3f5")
        im = ax.pcolormesh(
            bins, bins, np.ma.masked_where(H.T == 0, H.T), cmap="magma", norm=norm, shading="flat"
        )
        s = rec["surface"]
        ax.add_patch(
            Circle(s["centroid_um"], s["D95_um"] / 2, fill=False, ec="#16867b", lw=1.25, ls="--")
        )
        ax.plot(0, 0, "+", c="#03877c", ms=7)
        ax.set_aspect("equal")
        ax.set_title(
            f"C1 {rec['c1_A']:+g} A   C2 {rec['c2_A']:+g} A\nD95 {s['D95_um']:.0f} µm   ·   α95 {s['alpha95_mrad']:.1f} mrad",
            fontsize=10,
        )
        ax.set_xlabel("x (µm)")
        ax.tick_params(labelsize=8)
    for ax in grid[:, 0]:
        ax.set_ylabel("y (µm)")
    fig.colorbar(im, ax=axes, shrink=0.62, label="nA / µm²")
    fig.savefig(HERE / "specimen-probes.png")
    plt.close(fig)
    fig, (pupil, focus) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    r, d = items[-1]
    live = d["loss"] == -1
    rim = (
        np.maximum(
            np.linalg.norm(d["aperture_phase"][live, :2], axis=1),
            np.linalg.norm(d["aperture_rear_phase"][live, :2], axis=1),
        )
        * 1e6
    )
    vel = d["surface_phase"][live, 2:4]
    w = d["weights_uA"][live]
    chief = np.r_[np.average(vel, axis=0, weights=w), 1.0]
    chief /= np.linalg.norm(chief)
    dirs = np.column_stack((vel, np.ones(len(vel))))
    ang = 1e3 * np.arctan2(np.linalg.norm(np.cross(dirs, chief), axis=1), dirs @ chief)
    pupil.scatter(rim, ang, s=3 + 10 * np.sqrt(w / w.max()), alpha=0.55, color=COLORS[1])
    pupil.axvspan(47.5, 50, color="#77c386", alpha=0.23)
    pupil.axhline(30, color="#a9463a", ls="--", lw=1)
    pupil.set(
        xlabel="Maximum radius at the two C2 foil faces (µm)",
        ylabel="Specimen polar angle (mrad)",
        xlim=(0, 50.5),
    )
    pupil.set_title("A   Aperture rays reaching the specimen", fontsize=11)
    for j, (r, d) in enumerate(items):
        z, y = d["focus_upstream"].T
        focus.plot(z, y, lw=1.6, color=COLORS[j], label=f"{r['c1_A']:+g} / {r['c2_A']:+g} A")
    focus.set_yscale("log")
    focus.axvline(1599.1999975, color="#a9463a", ls="--", lw=1)
    focus.set(xlabel="Distance from tip z (mm)", ylabel="D95 (µm)")
    focus.set_title("B   D95 from Mini entry to specimen", fontsize=11)
    focus.legend(fontsize=8, frameon=False)
    fig.savefig(HERE / "aperture-and-focus.png")
    plt.close(fig)


def upstream(name):
    rec, rays = open_case(name)
    folder = HERE / name
    c1_image = json.loads((folder / "c1-image.json").read_text())["images"][0]
    with np.load(folder / "gun-paths.npz") as gun:
        gun_z = gun["z_mm"]
        gun_rms = gun["rms_m"][:, :, 2] * 1000
    fig, (gun_axis, cond) = plt.subplots(1, 2, figsize=(10.4, 4.1), layout="constrained")
    fig.suptitle(
        f"Gun crossover and condenser waists   |   {rec['c1_A']:+g} / {rec['c2_A']:+g} A",
        x=0.07,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    window = (gun_z >= 16) & (gun_z <= 32)
    gun_axis.plot(gun_z[window], gun_rms[0, window], color=COLORS[0], label="All gun-exit IDs")
    gun_axis.plot(gun_z[window], gun_rms[1, window], color=COLORS[1], label="Final specimen IDs")
    gun_axis.axvline(24, ls=":", color="#182d37", label="Gun aperture (24 mm)")
    gun_axis.axvline(25.26, ls="--", color=COLORS[2], label="Axial point image (25.26 mm)")
    gun_axis.set(
        xlabel="Distance from tip (mm)", ylabel="2-D RMS radius (mm)", title="Gun near the 4 mm aperture"
    )
    gun_axis.legend(frameon=False, fontsize=8)
    z = rays["z_mm"]
    used = z <= 845
    for key, color, label in [
        ("stats_m", COLORS[0], "Live beam"),
        ("fixed_stats_m", COLORS[1], "Final specimen IDs"),
    ]:
        widths = np.sqrt(rays[key][:, :2].sum(axis=1)) * 1000
        cond.plot(z[used], widths[used], color=color, label=label)
    for a, b in [(450, 550), (550, 750), (830, 845)]:
        cond.axvspan(a, b, color="#64748b", alpha=0.08, lw=0)
    cond.axvspan(750, 830, color="#77c386", alpha=0.22, lw=0)
    cond.axvline(765, ls=":", color="#182d37", label="100 µm C2 aperture")
    cond.axvline(c1_image["z_mm"], ls="--", color=COLORS[2], label="C1 paraxial source image")
    cond.set(
        xlabel="Distance from tip (mm)",
        ylabel="2-D RMS radius (mm)",
        xlim=(450, 845),
        title="C1 image and the C2–C3 interval",
    )
    cond.set_yscale("log")
    cond.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in (gun_axis, cond):
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(HERE / "gun-and-condensers.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--compare-relay", help="Case for a separate crossover-priority ray diagram")
    args = parser.parse_args()
    landings(args.cases)
    rays(args.cases[-1])
    if args.compare_relay:
        rays(args.compare_relay, "relay-ray-diagram.png")
        upstream(args.compare_relay)
