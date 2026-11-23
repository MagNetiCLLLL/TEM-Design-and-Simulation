"""Render saved particle paths, measured relay planes and specimen incident current."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from plot_normal_objective_rays import (
    COLORS,
    EVIDENCE,
    FOCUS,
    HERE,
    INK,
    LOST,
    PIVOT,
    WAIST,
    instrument_parts,
    line_paths,
    read_json,
    shade,
    station_label,
)


def render(tag, scan):
    r = read_json(EVIDENCE / f"joint-{tag}.json")
    p = read_json(scan)
    with np.load(EVIDENCE / f"joint-{tag}-particles.npz") as f:
        z = f["track_z_mm"]
        x = f["tracks"][:, :, 0] * 1e6
        bands = f["track_band"]
        live = f["surface_loss"] == -1
        xy = f["surface_phase"][live, :2] * 1e6
        amps = r["emitted_A"] * f["weight"][live] * 1e9
        rms_z = f["z_mm"]
        rms = f["moments"][:, 0]
    assert np.isclose(amps.sum(), 1000 * r["surface"]["specimen_uA"])
    subj = "Objective-only focus trial" if tag == "objective-focus" else "Relay alignment"
    f_plane = min(r["focal_mm"]["mini_and_objective"], key=lambda v: abs(v - 1490.0))
    ds = r["waists_mm"]["downstream_of_lower_scan"]["radial"]
    pivot_candidates = p["scan"]["candidates"]
    closest = (
        min(pivot_candidates, key=lambda row: abs(row["z_mm"] - f_plane)) if pivot_candidates else None
    )
    pivot = closest["z_mm"] if closest else None
    start, end = r["gun_exit_mm"], r["specimen_top_mm"]
    parts = instrument_parts()
    fig = plt.figure(figsize=(12.7, 7.8), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.08, 1, 0.75])
    top, detail, envelope = (fig.add_subplot(gs[i]) for i in range(3))
    fig.suptitle(f"{subj}   |   Nanoprobe Normal, 300 kV", fontsize=17, color=INK, weight="bold")
    top.set_title(
        f"Gun exit to specimen upper surface     ·     C2 aperture Ø100 µm"
        f"     ·     TL12 {r['TL12_A']:+.4f} A     ·     Objective {r['objective_A']:+.6f} A",
        loc="left",
        fontsize=10,
    )
    line_paths(top, z, x, bands, (start, end), scale=0.001)
    top.set_ylabel("Ray x (mm)")
    for key, label in (
        ("condenser_lens_1", "C1"),
        ("condenser_lens_2", "C2"),
        ("condenser_lens_3", "C3"),
        ("adapter_lens", "ADL"),
        ("ac_deflector", "AC"),
        ("mini_condenser", "Mini"),
    ):
        shade(top, parts[key], start, label)
    for key in (
        "probe_hp2_hexapole",
        "probe_tl22_lens",
        "probe_tl21_lens",
        "probe_hp1_hexapole",
        "probe_tl12_lens",
    ):
        shade(top, parts[key], start)
    station_label(top, 1225, "Probe corrector")
    station_label(top, end - 20, "Objective", 0.83)
    top.axvline(765.0, color=INK, lw=0.75, ls=":")
    station_label(top, 765.0, "C2 foil", 0.84)
    top.axvline(end, color=INK, lw=1)
    top.set_xticks(
        [450, 600, 800, 1000, 1200, 1400, end],
        ["450", "600", "800", "1000", "1200", "1400", f"{end:.1f}"],
    )
    line_paths(detail, z, x, bands, (1338.0, end))
    detail.set_ylabel("Ray x (µm)")
    detail.set_title("TL12 through the specimen   ·   zero scan command", loc="left", fontsize=10)
    for key, label in (
        ("probe_tl12_lens", "TL12"),
        ("ac_deflector", None),
        ("condenser_stigmator", "Stigmator (off)"),
        ("mini_condenser", "Mini Condenser"),
        ("objective_upper_pole", "Upper pole"),
    ):
        shade(detail, parts[key], start, label)
    for zz, label in zip([1387.5, 1412.5], ["Upper scan", "Lower scan"]):
        detail.axvline(zz, ls=":", lw=0.9, color="#647381")
        station_label(detail, zz, label, 0.95)
    detail.axvline(end, color=INK, lw=1)
    keep = (rms_z >= 1338) & (rms_z <= end)
    envelope.plot(rms_z[keep], rms[keep], lw=1.65, color=INK)
    envelope.set(xlim=(1338, end), ylim=(0, None), ylabel="2D RMS radius (µm)", xlabel="Column z (mm)")
    envelope.spines[["top", "right"]].set_visible(False)
    envelope.grid(axis="y", color="#e1e6e8", lw=0.6)
    for ax in (detail, envelope):
        ax.axvline(f_plane, ls="--", lw=1.4, color=FOCUS)
        if pivot is not None:
            ax.axvline(pivot, ls=":", lw=2, color=PIVOT)
        for waist in ds:
            if waist < end - 0.001:
                ax.axvline(waist, ls="-.", lw=1.1, color=WAIST)
    handles = [
        Line2D([], [], ls="--", color=FOCUS, label=f"Two-lens front focus  {f_plane:.2f} mm"),
        Line2D([], [], ls="-.", color=WAIST, label="Measured radial waist"),
    ]
    if pivot is not None:
        residual = closest["position_singular_nm_per_urad"][0]
        label = "Driven pivot" if residual < 1e-6 else "Minimum scan displacement"
        handles.insert(1, Line2D([], [], ls=":", color=PIVOT, lw=2, label=f"{label}  {pivot:.2f} mm"))
    envelope.legend(handles=handles, loc="best", fontsize=8, framealpha=0.9)
    colors = [
        Line2D([], [], color=c, lw=1.5, label=s)
        for c, s in zip(COLORS, ("0–0.35 R", "0.35–0.65 R", "0.65–0.90 R", "0.90–1.00 R"), strict=True)
    ]
    colors.append(Line2D([], [], color=LOST, ls="--", lw=1, label="Intercepted at foil"))
    top.legend(
        handles=colors,
        loc="lower left",
        ncol=5,
        fontsize=7,
        framealpha=0.9,
        title="Entry radius at C2 foil (R = 50 µm)",
        title_fontsize=7,
        alignment="left",
    )
    fig.savefig(HERE / f"joint-{tag}-rays.png", dpi=165, facecolor="white")
    plt.close(fig)

    s = r["surface"]
    fig, (im, enc) = plt.subplots(
        1, 2, figsize=(10.5, 4.75), width_ratios=[1.18, 1], layout="constrained"
    )
    fig.suptitle(
        f"{subj}   |   Current at the specimen upper surface", fontsize=16, weight="bold", color=INK
    )
    half = float(np.ceil(np.max(np.abs(xy)) * 1.015))
    half = max(half, 0.001)
    hist, xx, yy = np.histogram2d(*xy.T, bins=103, range=[[-half, half], [-half, half]], weights=amps)
    assert np.isclose(hist.sum(), amps.sum())
    dens = hist.T / ((xx[1] - xx[0]) * (yy[1] - yy[0]))
    np.savez_compressed(
        EVIDENCE / f"joint-{tag}-current-density.npz",
        x_edges_um=xx,
        y_edges_um=yy,
        current_A_yx=hist.T * 1e-9,
        density_A_m2_yx=dens * 1e3,
        specimen_top_mm=r["specimen_top_mm"],
        source_sha256=r["source_digest"],
        particle_archive_sha256=r["particles_sha256"],
    )
    art = im.imshow(
        dens,
        extent=[-half, half, -half, half],
        origin="lower",
        cmap="inferno",
        norm=PowerNorm(0.5, vmin=0, vmax=dens.max()),
        interpolation="nearest",
    )
    im.add_patch(
        Circle(s["centroid_um"], s["D95_um"] / 2, fill=False, edgecolor="#a6e3f0", lw=1.1, ls="--")
    )
    im.plot(0, 0, "+", markersize=7, color="white", mew=0.8)
    im.set(xlabel="Specimen x (µm)", ylabel="Specimen y (µm)", aspect="equal")
    fig.colorbar(art, ax=im, shrink=0.82, label="Incident current density (nA/µm²)")
    radius = np.linalg.norm(xy - s["centroid_um"], axis=1)
    order = np.argsort(radius)
    enc.plot(radius[order], np.cumsum(amps[order]) * 0.001, color=FOCUS, lw=1.65)
    enc.axvline(s["D95_um"] / 2, ls="--", color=WAIST, lw=1)
    enc.axhline(0.95 * s["specimen_uA"], ls=":", color=WAIST, lw=1)
    enc.set(
        xlabel="Radius from current centroid (µm)",
        ylabel="Enclosed current (µA)",
        xlim=(0, None),
        ylim=(0, s["specimen_uA"] * 1.07),
    )
    enc.spines[["top", "right"]].set_visible(False)
    enc.grid(color="#e2e6e8", lw=0.6)
    enc.text(
        0.97,
        0.12,
        f"D95  {s['D95_um']:.3f} µm\nRMS radius  {s['rms_um']:.3f} µm\n"
        f"Transmitted  {s['specimen_uA']:.4f} µA\n{len(amps):,} of {r['emitted']:,} packets",
        transform=enc.transAxes,
        ha="right",
        fontsize=10,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.87),
    )
    fig.savefig(HERE / f"joint-{tag}-probe.png", dpi=175, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tag")
    p.add_argument("scan", type=Path)
    args = p.parse_args()
    render(args.tag, args.scan)
