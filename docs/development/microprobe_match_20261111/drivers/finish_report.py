"""Create original scientific figures and an English evidence report from archived observations."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT / "docs/development/microprobe_match_20261111"


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    results = load(DEST / "results.json")
    family = results["family"]
    colors = ["#0072B2", "#009E73", "#D55E00", "#9A4F96"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True,
                             gridspec_kw={"height_ratios": [2.1, 1.]})
    for entry, color in zip(family, colors):
        case = DEST / "column" / entry["run"] / f"trial-{entry['index']:02d}"
        with np.load(case / "beam-envelope.npz") as data:
            z, rms, r95 = data["z_mm"], data["rms_radius_um"], data["d95_um"] / 2
            axes[0].semilogy(z, rms, color=color, lw=1.1,
                             label=f"D95 = {entry['surface']['d95_um']:.3f} um; RMS radius")
            axes[0].semilogy(z, r95, color=color, lw=.8, ls="--", alpha=.65)
            mask = z >= 1550
            axes[1].plot(z[mask], rms[mask], color=color, lw=1.3)
            axes[1].plot(z[mask], r95[mask], color=color, ls="--", lw=.9, alpha=.65)
    # Use executed assembly coordinates rather than schematic coordinates.
    import sys
    sys.path[:0] = [str(ROOT / "src")]
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.profile_io import read_profile, apply_profile_values
    state = default_state(); catalog = AssemblyCatalog()
    selection, values = read_profile(DEST / family[1]["profile"])
    catalog.apply(state, catalog.normalise_selection(selection)); apply_profile_values(state, values)
    keys = [("condenser_lens_1", "C1"), ("condenser_lens_2", "C2"),
            ("condenser_lens_3", "C3"), ("adapter_lens", "ADL"),
            ("probe_hp2_hexapole", "HP2"), ("probe_tl22_lens", "TL22"),
            ("probe_tl21_lens", "TL21"), ("probe_hp1_hexapole", "HP1"),
            ("probe_tl12_lens", "TL12"), ("mini_condenser", "Mini")]
    parts = {part.key: part for part in (*state.lenses, *state.corrector_elements)}
    devices = [(parts[key].z_mm, label) for key, label in keys] + [(1400., "AC"), (1599.2, "Specimen")]
    for index, (z, label) in enumerate(devices):
        axes[0].axvline(z, color=".75", lw=.65, zorder=0)
        axes[0].text(z, .96 if index % 2 else .84, label, rotation=90, fontsize=8,
                     va="top", ha="right", transform=axes[0].get_xaxis_transform())
    for ax in axes:
        ax.grid(alpha=.18); ax.set_xlabel("Absolute axial position z (mm)")
        ax.set_ylabel("Current-weighted radius (um)")
    axes[0].set_xlim(450, 1601); axes[0].legend(loc="lower left", fontsize=8, ncol=2)
    axes[0].set_title("Measured Microprobe envelopes: solid RMS radius, dashed R95 = D95/2")
    axes[1].axvspan(state.ac_deflector.upper_z_mm, state.ac_deflector.lower_z_mm, alpha=.1)
    axes[1].axvline(1584.3462305713263, color=".3", ls=":", lw=1., label="Canonical front-focal relation")
    axes[1].axvline(state.sample.upper_surface_z_mm, color=".3", lw=1., label="Specimen upper surface")
    axes[1].set_xlim(1550, 1600); axes[1].set_title("Mini–Objective interval: source waist before broad specimen illumination")
    axes[1].legend(fontsize=8)
    fig.savefig(DEST / "beam-envelopes.png", dpi=180)
    fig.savefig(DEST / "beam-envelopes.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), constrained_layout=True)
    for entry, color, ax in zip(family, colors, axes):
        case = DEST / "column" / entry["run"] / f"trial-{entry['index']:02d}"
        with np.load(case / "native-selected-phase.npz") as data:
            row = np.flatnonzero(data["z_mm"] == entry["surface"]["z_mm"])[0]
            keep = data["alive"][row] & (data["weights"] > 0)
            x, y = data["x_m"][row, keep] * 1e6, data["y_m"][row, keep] * 1e6
            ax.scatter(x, y, s=4, color=color, alpha=.55, linewidths=0)
            ax.add_patch(plt.Circle(entry["surface"]["centroid_xy_um"], entry["surface"]["d95_um"] / 2,
                                   fill=False, color=".25", lw=.8, ls="--"))
        s = entry["surface"]
        ax.set_title(f"D95 {s['d95_um']:.3f} um\nalpha95 {s['alpha95_mrad']:.4f} mrad", fontsize=10)
        ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_xlabel("x (um)"); ax.grid(alpha=.15)
        ax.text(.03, .03, f"{s['count']} arrivals\n{s['current_a']*1e6:.3f} uA", transform=ax.transAxes, fontsize=9)
    axes[0].set_ylabel("y (um)")
    fig.suptitle("Specimen upper surface: actual particles, absolute currents, no restored aperture losses")
    fig.savefig(DEST / "illumination-footprints.png", dpi=180)
    fig.savefig(DEST / "illumination-footprints.svg")
    plt.close(fig)

    rows = []
    controls = []
    waists = []
    for entry in family:
        s = entry["surface"]
        label = f"{s['d95_um']:.3f}"
        rows.append(f"| [{label}]({entry['profile']}) | {s['radial_rms_um']:.6f} | {s['alpha95_mrad']:.6f} | {s['alpha_rms_mrad']:.6f} | {s['current_a']*1e6:.6f} | {100*s['current_a']/1e-5:.4f}% | {s['count']} / {s['effective_count']:.0f} |")
        c = entry["controls"]
        controls.append(f"| {label} | {c['condenser_lens_2']['percent']:.12f} | {c['condenser_lens_3']['percent']:.12f} | {c['probe_hp2_hexapole']['strength_m3']:.6f} | {c['probe_hp1_hexapole']['strength_m3']:.6f} |")
        for w in sorted(entry["waists"], key=lambda item:item["rms_minimum"]["z_mm"]):
            if w["population"] != "actual_population": continue
            r, d = w["rms_minimum"], w["d95_minimum"]
            bracket = w["rms_native_bracket_mm"]
            waists.append(f"| {label} | {w['interval']} | {r['z_mm']:.6f} [{bracket[0]:.6f}, {bracket[-1]:.6f}] | {r['rms_radius_um']*1000:.3f} | {r['d95_um']*1000:.3f} | {r['alpha95_mrad']:.6f} / {r['alpha_rms_mrad']:.6f} | {d['z_mm']:.6f} | {d['d95_um']*1000:.3f} |")
    (DEST / "measured-tables.md").write_text(
        "# Measured finite-source results\n\nAll sizes and angles are measured about the actual current centroid/chief. D95 contains 95% of the current that actually reaches that plane.\n\n"
        "| Specimen D95 (um) / profile | RMS radius (um) | alpha95 (mrad) | Angular RMS (mrad) | Current (uA) | Emitted-current transmission | Arrivals / effective count |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n" + "\n".join(rows) +
        "\n\n## Coupled controls\n\n| Specimen D95 (um) | C2 (%) | C3 (%) | HP2 (m^-3) | HP1 (m^-3) |\n| --- | --- | --- | --- | --- |\n" + "\n".join(controls) +
        "\n\n## Source-image observations\n\nThe table uses all current alive at each plane. The JSON additionally retains the common specimen-surviving cohort, all native RMS/D95 brackets and original particle IDs. A bracket describes executed observation spacing, not a confidence interval. RMS and D95 minima are located independently over their optical interval.\n\n"
        "| Specimen D95 (um) | Interval | RMS-minimum z and native bracket (mm) | RMS radius (nm) | D95 at RMS minimum (nm) | alpha95 / angular RMS at RMS minimum (mrad) | D95-minimum z (mm) | Minimum D95 (nm) |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n" + "\n".join(waists) + "\n", encoding="utf8")
    print("Created two original scientific figures and complete measured tables")


if __name__ == "__main__":
    main()
