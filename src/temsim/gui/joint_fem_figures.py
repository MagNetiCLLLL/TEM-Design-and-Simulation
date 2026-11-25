"""Render the mechanical ray paths and weighted discrete landings from one execution."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle


def render(folder, result):
    render_condenser(folder, result)
    with np.load(folder / "trajectories.npz") as data:
        z = data["z_mm"]
        paths = data["paths_mm"]
        ids = data["selected_ids"]
        live = np.isin(ids, data["sample_ids"])
        xy = data["sample_xy_um"]
        weights = data["sample_weights_uA"] * 1000
    figure = Figure(figsize=(11, 7.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1, gridspec_kw={"height_ratios": [1, 1, 0.95]})
    figure.suptitle("Nanoprobe Normal   |   Joint FEM trajectories", fontweight="bold")
    palette = [
        "#0072b2",
        "#e69f00",
        "#009e73",
        "#d55e00",
        "#8064af",
        "#9d4423",
        "#cc79a7",
        "#206872",
    ]
    arrivals = np.flatnonzero(live)
    drawn = arrivals[np.linspace(0, len(arrivals) - 1, min(len(arrivals), 160), dtype=int)]
    for ax, dim, zoom in zip(axes, [0, 1, 0], [False, False, True]):
        for j in np.r_[np.flatnonzero(~live), drawn]:
            ax.plot(
                z,
                paths[j, :, dim] * (1000 if zoom else 1),
                color=palette[j % len(palette)] if live[j] else "#9ba6ae",
                alpha=0.92 if live[j] else 0.4,
                lw=1.05 if live[j] else 0.7,
            )
        ax.axhline(0, color="#4a5963", lw=0.45)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel(("x" if dim == 0 else "y") + (" (µm)" if zoom else " (mm)"))
        if zoom:
            ax.set_xlim(1125, z[-1] + 1)
            local = paths[live][:, (z >= 1125), dim] * 1000
            limit = max(1, np.nanmax(abs(local))) * 1.24
            ax.set_ylim(-limit, limit * 1.18)
            hardware = [
                (1129.5, 1159.5, "HP2"),
                (1153.6, 1179.1, "TL22"),
                (1237.2, 1262.7, "TL21"),
                (1285.85, 1315.85, "HP1"),
                (1353.6, 1379, "TL12"),
                (1380, 1395, "Upper scan"),
                (1405, 1420, "Lower scan"),
                (1512.5, 1532.5, "Mini"),
                (1534.5, 1596.5, "Objective"),
            ]
        else:
            ax.set_xlim(0, z[-1] + 2)
            hardware = [
                (0, 36, "Gun"),
                (450, 550, "C1"),
                (550, 750, "C2"),
                (830, 1000, "C3"),
                (1070, 1106.4, "ADL"),
                (1129.5, 1379, "Corrector"),
                (1380, 1420, "Scan"),
                (1512.5, 1596.5, "Mini / OL"),
            ]
        for i, (lo, hi, name) in enumerate(hardware):
            ax.axvspan(lo, hi, color="#6f92aa", alpha=0.09)
            ax.text(
                (lo + hi) / 2,
                0.96 if not zoom or i % 2 == 0 else 0.83,
                name,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
            )
        ax.axvline(z[-1], color="#a64d33", lw=1)
        if zoom:
            for image in result.get("full_column_source_images", ()):
                position = image["z_mm"]
                if 1125 < position < 1512.5:
                    ax.axvline(position, color="#495663", ls="--", lw=1)
                    ax.text(position, 0.02, f"{position:.1f} mm", rotation=90,
                            transform=ax.get_xaxis_transform(), ha="right", va="bottom",
                            fontsize=7, color="#384652")
            relay = result.get("measured_relay")
            if relay:
                hp2, hp1 = relay["HP2_HP1_effective_midplanes_mm"]
                ax.axvline(hp2, color="#086b68", ls="-.", lw=1,
                           label="Effective HP midplanes")
                ax.axvline(hp1, color="#086b68", ls="-.", lw=1)
                ax.axvline(relay["objective_radial_coma_zero_mm"], color="#714b8d",
                           ls=":", lw=1.3, label="Objective radial-coma zero")
                waists = [z0 for z0 in result["specimen_cohort_waists_mm"]["r"]
                          if 1125 < z0 < 1512.5]
                ax.plot(waists, [0.20] * len(waists), "v", color="#a14724",
                        transform=ax.get_xaxis_transform(), label="Finite-current waist")
                ax.legend(loc="lower right", ncol=3, fontsize=7, framealpha=.9)
            ax.set_title(
                "Corrector and scan region  ·  dashed lines mark paraxial source images",
                loc="left", fontsize=9,
            )
    axes[0].set_title(
        f"{result['emitted']:,} emitted   ·   {len(drawn)} of {len(arrivals)} specimen rays shown   ·   gray rays end at an intercept",
        fontsize=10,
    )
    axes[-1].set_xlabel("Distance from emitting tip (mm)")
    figure.savefig(folder / "column-rays.png", dpi=155)
    figure = Figure(figsize=(6.4, 5.6), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    centre = np.average(xy, axis=0, weights=weights)
    ax.add_patch(
        Circle(
            centre,
            result["D95_um"] / 2,
            fill=False,
            color="#57808a",
            lw=1.15,
            label="D95",
        )
    )
    pts = ax.scatter(
        *xy.T, c=weights, cmap="viridis", s=75, edgecolors="#20282d", lw=0.7
    )
    ax.plot(*centre, marker="+", color="#2d5563", ms=11, label="Current centroid")
    ax.set_aspect("equal", "datalim")
    ax.margins(0.09)
    ax.legend(loc="best", fontsize=8)
    ax.set(
        xlabel="x (µm)",
        ylabel="y (µm)",
        title=f"Specimen landings at {z[-1]:.3f} mm\n{len(xy)} arrivals   ·   D95 {result['D95_um']:.2f} µm",
    )
    figure.colorbar(pts, ax=ax, shrink=0.8, label="Current represented per ray (nA)")
    figure.savefig(folder / "specimen-landings.png", dpi=155)


def render_condenser(folder, result):
    """Show the accepted illumination and both current-weighted envelopes over C1/C2."""
    with np.load(folder / "trajectories.npz") as data:
        z = data["relay_z_mm"]
        xy = data["relay_display_xy_um"]
        live = data["relay_live_rms_um"]
        accepted = data["relay_C2_rms_um"]
    relay = result["condenser_relay"]
    figure = Figure(figsize=(10, 7.4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(
        3, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.85]}
    )
    figure.suptitle("C1 and C2 source images", fontweight="bold")
    axes[0].set_title(
        f"{relay['C2_transmitted']:,} primaries through C2   ·   "
        f"{relay['C2_transmitted_nA']:.2f} nA   ·   {len(xy)} trajectories displayed",
        fontsize=10,
    )
    for dim, ax in enumerate(axes[:2]):
        for ray in xy:
            ax.plot(z, ray[:, dim], color="#007a85", alpha=0.30, lw=0.8)
        ax.axhline(0, color="#283b49", lw=0.55)
        ax.set_ylabel(("x" if dim == 0 else "y") + " (µm)")
    axes[2].plot(z, live[:, 2], color="#7b5a87", label="All live current", lw=1.65)
    axes[2].plot(
        z, accepted[:, 2], color="#007a85", label="C2-transmitted current", lw=1.65
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Radial RMS (µm)")
    axes[2].set_xlabel("Distance from emitting tip (mm)")
    axes[2].legend(loc="upper left", frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        if relay.get("interlens_interval_mm"):
            ax.axvspan(
                *relay["interlens_interval_mm"], color="#d8eae0", alpha=0.6, zorder=0
            )
        ax.axvline(relay["C2_foil_exit_z_mm"], color="#708393", lw=0.9, ls=":")
        for img in relay["source_images"]:
            if 450 < img["z_mm"] < 835:
                ax.axvline(img["z_mm"], color="#b84c31", ls="--", lw=1.05)
    for img in relay["source_images"]:
        if 450 < img["z_mm"] < 835:
            axes[1].annotate(
                f"{img['z_mm']:.2f} mm\n|M| {max(img['principal_magnifications']):.3f}",
                (img["z_mm"], 0.94),
                xycoords=("data", "axes fraction"),
                xytext=(4, 0),
                textcoords="offset points",
                color="#a4422c",
                va="top",
                fontsize=8,
            )
    axes[-1].set_xlim(450, 835)
    figure.savefig(folder / "condenser-relay.png", dpi=155)
