"""Render the registered illumination field intervals and the per-element inventory."""

from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
ROUND, HEX, UNIT, INK = "#155c79", "#795588", "#a05a28", "#233743"


def bar_panel(ax, display, components, legend_range):
    for j, (label, key) in enumerate(display):
        c = components[key]
        color = UNIT if not c["active"] else HEX if "m²" in c["units"] else ROUND
        for frac, height, alpha in (("0.1", .48, .24), ("0.5", .24, 1)):
            for lo, hi in c["bands_mm"][frac]:
                ax.broken_barh([(lo, hi-lo)], (j-height/2, height), color=color, alpha=alpha,
                               linewidth=0, zorder=2)
        ax.plot(c["center_mm"], j, marker="|", ms=10, markeredgewidth=1.05, color=INK, zorder=3)
    ax.set_xlim(*legend_range)
    ax.set_ylim(len(display)-.45, -.55)
    ax.set_yticks(range(len(display)), [label for label, _ in display], fontsize=9)
    ax.set_xlabel("Column z (mm)", color=INK)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", colors=INK, length=3)
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9aa8af")
    ax.grid(axis="x", color="#dee5e8", linewidth=.7, zorder=0)


def draw(c):
    fig, axes = plt.subplots(1, 2, figsize=(13.1, 8.6), gridspec_kw={"width_ratios": [1, 1.04]})
    fig.subplots_adjust(left=.108, right=.98, bottom=.135, top=.835, wspace=.38)
    fig.text(.04, .96, "Where the fields act along the illumination column", fontsize=19, weight="bold", color=INK)
    fig.text(.041, .918, "300 kV Nanoprobe Normal   ·   Each channel is normalized to its own peak on the incident path.",
             fontsize=10, color="#546770")
    names = ["C1","C2","C3","ADL","HP2","TL22","TL21","HP1","HPol X","HPol Y","TL12","TL11 / Mini","Objective"]
    bar_panel(axes[0], [(k, k) for k in names], c, (450, 1602))
    axes[0].set_title("Excited lenses and corrector channels", loc="left", pad=12, fontsize=12, color=INK)
    detail = [(k, k + (" X" if (k+" X") in c and k != "HPol" else "")) for k in
              ("Beam deflector upper", "Beam deflector lower", "ADL", "QPH2", "HP2", "DPH2", "TL22",
               "DP22", "HPC", "QPC", "DP21", "TL21", "QPH1", "HP1", "DPH1", "HPol X", "HPol Y",
               "QPol", "DP11", "TL12", "Upper scan", "Lower scan")]
    bar_panel(axes[1], detail, c, (1005, 1455))
    axes[1].set_title("Corrector and scan region", loc="left", pad=12, fontsize=12, color=INK)
    fig.legend(handles=[
        Patch(facecolor=ROUND, label="Round lens"), Patch(facecolor=HEX, label="Excited sextupole"),
        Patch(facecolor=UNIT, label="Unpowered winding, unit-current vacuum field (X)"),
        Line2D([], [], marker="|", linestyle="none", markersize=10, color=INK, label="Mechanical center")],
        loc="lower center", bbox_to_anchor=(.5,.044), ncol=4, frameon=False, fontsize=9)
    fig.text(.5, .028, "Light bands exceed 10% of the channel peak. Dark bands exceed 50%. "
             "The incident profile ends at the specimen entrance.", ha="center", color="#465964", fontsize=9)
    fig.savefig(HERE/"illumination-field-ranges.png", dpi=160, facecolor="white")
    plt.close(fig)


def intervals(c, threshold):
    return ", ".join(f"{a:.1f}–{b:.1f}" for a,b in c["bands_mm"][threshold])


def inventory(r):
    c, stations = r["components"], r["stations"]
    target = HERE/"field-range-inventory.md"
    rows = ["# Illumination field inventory", "",
            "Column z is in mm from the model origin. The 300 kV gun exit is at 450 mm. "
            "The incident calculation ends at 1599.1999975 mm. Values at that endpoint describe only the incident side. "
            "The original excitation and the 100 µm C2 aperture define the working point. "
            "[Mechanical interpretation and primary references](field-range-audit.md)", "",
            "Each interval is a connected stretch above the stated fraction of its own channel maximum. "
            "The thresholds are reporting choices. Weak fields continue outside the reported intervals. "
            "A round channel uses on-axis longitudinal field, a dipole uses transverse field, and a multipole uses "
            "its leading transverse coefficient at finite radius. The last column lists other excited channels above "
            "10% of their respective maxima at this component's center. Percentages from different rows cannot be added.", "",
            "| Channel | Center | Mechanical interval | Field used | ≥50% | ≥10% | Other excited channels at center |",
            "| --- | ---: | --- | --- | --- | --- | --- |"]
    for key in sorted(c, key=lambda k: c[k]["center_mm"]):
        s = c[key]
        adjacent = stations[key]["active_profile_fraction_at_center"]
        near = ", ".join(f"{k} {100*v:.0f}%" for k,v in sorted(adjacent.items(),key=lambda x:-x[1]) if v>=.1) or "—"
        a, b = s["mechanical_span_mm"]
        method = ("Off, per A in vacuum" if not s["active"] else "Material response to current" if "bias_A" in s
                  else "Material field" if key == "Objective" else "Vacuum winding field")
        rows.append(f"| {key} | {s['center_mm']:.2f} | {a:.1f}–{b:.1f} | {method} | {intervals(s,'0.5')} | "
                    f"{intervals(s,'0.1')} | {near} |")
    rows.extend(["", "The paired saddle stations list their shared assembly envelope. The center marks each winding station. "
                 "Unpowered channels show the response to 1 A in the individual X or Y vacuum winding with physical turns. "
                 "For HP1, HP2, HPol X/Y and TL21/TL22 the percent bands use the local current response of the nonlinear material "
                 "model. The static round-lens fields for C1–C3, ADL, TL12 and Mini use the signed vacuum winding groups. "
                 "The objective uses its single material-field supply. [Profiles and solver checks](evidence/field-footprints.json)", "",
                 "## Passive and optical reference planes", "",
                 "| Station | z (mm) | Excited channels above 10% of own peak |",
                 "| --- | ---: | --- |"])
    for key in ("C2 aperture","C3 aperture","Virtual scan pivot","Two-lens front focus","Specimen entrance"):
        s = stations[key]
        near = ", ".join(f"{k} {100*v:.0f}%" for k,v in s["active_profile_fraction_at_center"].items() if v>=.1) or "—"
        rows.append(f"| {key} | {s['z_mm']:.3f} | {near} |")
    rows.extend(["", "DP12 uses the same four upper/lower AC winding channels. Its logical offset has no additional field interval. "
                 "The physical condenser apertures intercept particles at the foil. The virtual pivot and combined-field front focal "
                 "plane are optical reference planes. The sleeve, return plates and pole pieces modify the response of their magnetic circuit.", "",
                 "## Objective field near and behind the specimen", "",
                 "The isolated objective solution extends beyond the incident integration domain. The entries below give its signed "
                 "axial field at the corresponding hardware centers. The post-specimen imaging and projection systems require their own operating state.", "",
                 "| Station | z (mm) | Objective Bz (T) |", "| --- | ---: | ---: |"])
    for k, s in r["objective_field_at_downstream_stations"].items():
        rows.append(f"| {k} | {s['center_mm']:.3f} | {s['objective_Bz_T']:.5f} |")
    ol = {"bands_mm": r["objective_incident_and_outgoing_bands_mm"]}
    rows += ["", f"Over the full column module the objective exceeds 50% of its peak at {intervals(ol,'0.5')} mm "
             f"and 10% at {intervals(ol,'0.1')} mm. The incident-only bands above use the incident maximum.", "",
             "The installed outgoing windings have the following unit-current vacuum responses over the specimen-to-module-end interval. "
             "A separate imaging state is needed to assign their actual excitation.", "",
             "| Outgoing winding | Center | ≥50% | ≥10% |", "| --- | ---: | --- | --- |"]
    for key, s in r["outgoing_unit_windings"].items():
        rows.append(f"| {key} | {s['center_mm']:.2f} | {intervals(s,'0.5')} | {intervals(s,'0.1')} |")
    rows += ["", "The physical objective aperture is downstream of the specimen in this column. Its location differs from the "
             "illumination-side coma-free plane. Descan shares the physical image/diffraction deflector in this configuration. "
             "The source upstream of the frozen gun exit, energy filter, post-column detector and optional image correctors have "
             "no working-point field attribution in this incident trace.", ""]
    target.write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    report = json.loads((EVIDENCE/"field-footprints.json").read_text())
    path = EVIDENCE/"field-footprint-profiles.npz"
    if sha256(path.read_bytes()).hexdigest() != report["profile_file_sha256"]:
        raise ValueError("Profiles and field inventory belong to different runs")
    with np.load(path) as f:
        np.testing.assert_array_equal(f["names"], list(report["components"]))
        for k, values in zip(f["names"], f["amplitude"], strict=True):
            np.testing.assert_allclose(values.max(), report["components"][k]["peak"])
    draw(report["components"])
    inventory(report)
