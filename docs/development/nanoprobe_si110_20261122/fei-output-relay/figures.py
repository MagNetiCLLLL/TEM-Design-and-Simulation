"""Draw actual tip-to-specimen trajectories, specimen current, and check the three planes."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
CASE = HERE / "final"
COLORS = ["#4e4998", "#e27737", "#147e81", "#5682b1"]
PENS = ["#b44b32", "#17699a", "#358457"]
plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "savefig.dpi": 170,
    }
)


def read(folder, n):
    return json.loads((folder / n).read_text())


def optical_hardware(ax, stations, size=8):
    for a, b, label, y in stations:
        ax.axvspan(a, b, color="#9baebb", alpha=0.15, lw=0, zorder=-4)
        ax.text(
            (a + b) / 2,
            y,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=size,
            color="#334853",
        )
    ax.axhline(0, lw=0.6, color="#999", zorder=-3)
    ax.grid(axis="y", lw=0.5, alpha=0.24)


def ray_diagram():
    r = read(CASE, "result.json")
    with np.load(CASE / "gun-paths.npz") as g, np.load(CASE / "trajectories.npz") as d:
        np.testing.assert_array_equal(g["selected_ids"], d["selected_ids"])
        z = np.r_[g["z_mm"], d["z_mm"][1:]]
        xy = np.concatenate((g["xy_m"].transpose(1, 0, 2), d["xy_m"][1:])) * 1000
        ok = np.isin(d["selected_ids"], d["ids"][d["loss"] == -1])
        keep = (d["z_mm"] > 1340) & (d["z_mm"] < 1535)
        zz = d["z_mm"][keep]
        rms = np.sqrt(d["stats_m"][keep, :2].sum(axis=1)) * 1e6
    pivot = read(CASE, "pivot.json")["z_mm"]
    front = read(CASE, "front.json")["all_fronts"]
    front = next((v["z_mm"] for v in front if 1420 < v["z_mm"] < 1512.5), None)
    waist = next((z for z in r["waists"]["live"]["radial_mm"] if 1420 < z < 1512.5), None)
    fig, axes = plt.subplots(
        4, 1, figsize=(11.6, 10.3), layout="constrained", height_ratios=[1, 1, 1.12, 0.8]
    )
    fig.suptitle(
        "Nanoprobe Normal    |    Physical-tip trajectories to the specimen",
        fontsize=16,
        fontweight="bold",
        ha="left",
        x=0.065,
    )
    for ax, dim, zoom in ((axes[0], 0, False), (axes[1], 1, False), (axes[2], 0, True)):
        for j, live in enumerate(ok):
            ax.plot(
                z,
                xy[:, j, dim] * (1000 if zoom else 1),
                lw=0.62 if live else 0.5,
                color=COLORS[j % 4] if live else "#969b9f",
                alpha=0.64 if live else 0.27,
            )
        ax.set_ylabel(("x" if dim == 0 else "y") + (" (µm)" if zoom else " (mm)"))
        if zoom:
            optical_hardware(
                ax,
                [
                    (1353.6, 1379, "TL12", 0.98),
                    (1380, 1395, "Upper AC", 0.98),
                    (1405, 1420, "Lower AC", 0.83),
                    (1512.5, 1532.5, "Mini", 0.98),
                ],
                8,
            )
            ax.set_xlim(1340, 1535)
            m = (z > 1340) & (z < 1535)
            lim = np.nanmax(abs(xy[m][:, ok, dim])) * 1100
            ax.set_ylim(-lim, lim * 1.2)
        else:
            optical_hardware(
                ax,
                [
                    (0, 36, "Gun", 0.98),
                    (450, 550, "C1", 0.98),
                    (550, 750, "C2", 0.98),
                    (830, 1000, "C3", 0.98),
                    (1070, 1106.4, "ADL", 0.98),
                    (1129.5, 1379, "Corrector", 0.98),
                    (1380, 1420, "Scan", 0.98),
                    (1512.5, 1596.5, "Mini / OL", 0.98),
                ],
            )
            ax.set_xlim(0, 1602)
            ax.axvline(765, color="#374e60", lw=0.9, ls=":")
            ax.axvline(1599.2, color="#a74431", lw=0.9)
    axes[0].set_title(
        f'Gun lens +50 V   ·   C1 −34 A   ·   C2 +9 A   ·   C2 aperture Ø100 µm   ·   TL12 {r["tl12_A"]:+.3f} A   ·   objective {r["objective_A"]:+.4f} A',
        fontsize=10,
    )
    axes[2].set_title("TL12 to Mini Condenser   |   zero scan command", fontsize=11)
    axes[3].plot(zz, rms, color="#292644", lw=2, label="Current-weighted RMS radius")
    axes[3].set_xlim(1340, 1535)
    axes[3].set_ylabel("RMS radius (µm)")
    axes[3].set_xlabel("Axial position from the emitting tip, z (mm)")
    handles = []
    for v, c, sty, label in zip(
        (waist, pivot, front), PENS, ("--", ":", "-."), ("Radial waist", "AC pivot", "Mini + OL front")
    ):
        if v is None:
            continue
        for ax in axes[2:]:
            ax.axvline(v, color=c, ls=sty, lw=1.65)
        handles.append(Line2D([], [], color=c, ls=sty, lw=1.65, label=f"{label} {v:.2f} mm"))
    axes[3].legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.savefig(HERE / "column-rays.png")
    plt.close(fig)


def probe():
    r = read(CASE, "result.json")
    f = read(CASE, "focus.json")
    fig, (ax, ar, af) = plt.subplots(
        1, 3, figsize=(13, 4.65), layout="constrained", width_ratios=[1, 1.02, 1.12]
    )
    fig.suptitle(
        "Specimen upper surface    |    " + f'z = {f["specimen_z_mm"]:.3f} mm',
        fontsize=15,
        fontweight="bold",
        ha="left",
        x=0.04,
    )
    with np.load(CASE / "trajectories.npz") as d:
        ok = d["loss"] == -1
        w = d["weights_uA"][ok] * 1e6
        xy = d["surface_phase"][ok, :2] * 1e6
        v = d["surface_phase"][ok, 2:4]
        ap = (
            np.maximum(
                np.linalg.norm(d["aperture_phase"][ok, :2], axis=1),
                np.linalg.norm(d["aperture_rear_phase"][ok, :2], axis=1),
            )
            * 1e6
        )
    centre = np.average(xy, weights=w, axis=0)
    d95 = r["surface"]["D95_um"]
    lim = max(1.28 * d95 / 2, np.quantile(np.max(abs(xy - centre), axis=1), 0.995))
    ext = (centre[0] - lim, centre[0] + lim, centre[1] - lim, centre[1] + lim)
    hb = ax.hexbin(
        *xy.T,
        C=w,
        reduce_C_function=np.sum,
        gridsize=40,
        extent=ext,
        mincnt=1,
        cmap="magma",
        norm=LogNorm(),
        linewidths=0,
    )
    ax.set(xlim=ext[:2], ylim=ext[2:], aspect="equal", xlabel="x (µm)", ylabel="y (µm)")
    ax.add_patch(Circle(centre, d95 / 2, fill=False, edgecolor="#3e8c9d", lw=1.35))
    ax.plot(*centre, "+", c="#288f9c")
    ax.set_title(f"Landing current   ·   D95 = {d95:.2f} µm", fontsize=10)
    cb = fig.colorbar(hb, ax=ax, location="bottom", pad=0.04, shrink=0.83)
    cb.set_label("pA per occupied bin")
    chief = np.r_[np.average(v, weights=w, axis=0), 1.0]
    unit = np.column_stack((v, np.ones(len(w))))
    angle = 1e3 * np.arctan2(np.linalg.norm(np.cross(unit, chief), axis=1), unit @ chief)
    rim = ap >= 47.5
    inward = np.einsum("ij,ij->i", xy - centre, v - chief[:2]) < 0
    ar.scatter(ap[~rim], angle[~rim], s=6, lw=0, c="#8c9cab", alpha=0.46)
    for mask, marker, color, label in (
        (rim & inward, "o", "#217968", "Rim · inward"),
        (rim & ~inward, "x", "#bc4d2c", "Rim · outward"),
    ):
        ar.scatter(ap[mask], angle[mask], s=20, marker=marker, lw=1, c=color, alpha=0.9, label=label)
    ar.axhline(30.0, color="#236f91", lw=1, ls="--", label="30 mrad")
    ar.set(
        xlim=(0, 50.6),
        ylim=(0, max(32.0, np.max(angle) * 1.05)),
        xlabel="Radius at C2 aperture (µm)",
        ylabel="Semi-angle at specimen (mrad)",
    )
    ar.set_title("Actual transmitted pupil", fontsize=10)
    ar.legend(frameon=False, fontsize=8)
    with np.load(CASE / "through-focus.npz") as d:
        z = d["z_mm"] - f["specimen_z_mm"]
        af.plot(z, d["D95_um"], c="#4f4a89", label="D95")
        af.plot(z, 2 * d["rms_um"], c="#be6536", label="2 × RMS radius")
    af.axvline(0.0, ls="--", c="#333", lw=1, label="Specimen top")
    af.set(xlabel="Offset from specimen top (mm)", ylabel="Diameter (µm)", xlim=(-10, 10))
    af.set_title("Through focus   ·   vacuum continuation", fontsize=10)
    af.legend(frameon=False, fontsize=8)
    for a in (ar, af):
        a.grid(axis="y", alpha=0.2)
    fig.savefig(HERE / "pupil-and-focus.png")
    plt.close(fig)

    dx = max(1.0, round(d95 / 32, 1))
    bound = np.ceil(np.max(abs(xy)) / dx) * dx + dx
    edges = np.arange(-bound, bound + 0.5 * dx, dx)
    hist, _, _ = np.histogram2d(*xy.T, bins=[edges, edges], weights=w)
    np.testing.assert_allclose(hist.sum(), w.sum(), rtol=1e-12)
    density = np.ma.masked_equal(hist.T / dx**2, 0)
    pos = density.compressed()
    norm = LogNorm(vmin=pos.min(), vmax=pos.max())
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.35), layout="constrained")
    fig.suptitle(
        f"Specimen-plane probe    |    D95 = {d95:.2f} µm",
        x=0.07,
        ha="left",
        fontweight="bold",
        fontsize=15,
    )
    cmap = plt.get_cmap("inferno").with_extremes(bad="#090912")
    for ax, a, title in zip(axes, (bound, 0.25 * d95), ("Full landing distribution", "Central region")):
        im = ax.pcolormesh(edges, edges, density, cmap=cmap, norm=norm, shading="flat", rasterized=True)
        ax.set(aspect="equal", xlabel="x (µm)", ylabel="y (µm)")
        ctr = np.zeros(2) if ax is axes[0] else centre
        ax.set_xlim(ctr[0] - a, ctr[0] + a)
        ax.set_ylim(ctr[1] - a, ctr[1] + a)
        ax.add_patch(Circle(centre, d95 / 2, fill=False, edgecolor="#64d8df", lw=1.4))
        ax.plot(*centre, marker="+", color="#64d8df", ms=8)
        ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=axes, shrink=0.83, location="bottom", aspect=40, pad=0.04)
    cb.set_label(
        f"Landing current density (pA/µm²)   ·   {dx:g} µm square bins\n"
        f"{len(w):,} weighted rays   ·   {w.sum()/1000:.2f} nA   ·   circle encloses 95% of current"
    )
    fig.savefig(HERE / "specimen-probe.png", bbox_inches="tight", pad_inches=0.17)
    plt.close(fig)


if __name__ == "__main__":
    ray_diagram()
    probe()
