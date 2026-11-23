"""Export compact numerical evidence and the field/location figures for this audit."""

from hashlib import sha256
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from circuit_audit import ADL, ARCHIVE, HERE, LOWER_MM, ROUND, TL12, UPPER_MM, WORK, audit_inputs
from field_profiles import load_solution, sample_delta
from temsim.corrector_field_cli import _save
from temsim.module_manifest import read_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT


CHANNELS = {
    ADL: "ADL",
    "probe_hp2_hexapole": "HP2",
    ROUND[1]: "TL22",
    ROUND[2]: "TL21",
    "probe_hp1_hexapole": "HP1",
    "probe_hpol_hexapole_x": "HPol X",
    "probe_hpol_hexapole_y": "HPol Y",
    TL12: "TL12",
}
OTHER_STATIONS = {
    "probe_qph2_quadrupole": "QPH2",
    "probe_hpc_hexapole": "HPC",
    "probe_qpc_quadrupole": "QPC",
    "probe_qph1_quadrupole": "QPH1",
    "probe_qpol_quadrupole": "QPol",
    "probe_dph2_coils": "DPH2",
    "probe_dp22_coils": "DP22",
    "probe_dp21_coils": "DP21",
    "probe_dph1_coils": "DPH1",
    "probe_dp11_coils": "DP11",
    "upper_scan": "Upper AC scan",
    "lower_scan": "Lower AC scan",
}
COLOR = "#126b77"
PURPLE = "#8346a3"
FOLDER = WORK / "profiles-5-170"
ASSEMBLED = WORK / "profiles-5-170-edge"


def crossings(z, y, fraction):
    amp = np.abs(y)
    threshold = fraction * max(amp)
    above = amp >= threshold
    i = np.flatnonzero(above[1:] != above[:-1])
    ends = z[i] + (threshold - amp[i]) * (z[i + 1] - z[i]) / (amp[i + 1] - amp[i])
    return np.r_[z[:1] if above[0] else [], ends, z[-1:] if above[-1] else []].reshape(-1, 2).tolist()


def metrics(z, y):
    amplitude = np.abs(y)
    i = np.argmax(amplitude)
    return dict(
        peak=float(amplitude[i]),
        peak_z_mm=float(z[i]),
        bands_mm={str(t): crossings(z, y, t) for t in (0.5, 0.1, 0.01)},
        scan_magnitude=np.interp([UPPER_MM, LOWER_MM], z, amplitude).tolist(),
        scan_percent=(100 * np.interp([UPPER_MM, LOWER_MM], z, amplitude) / amplitude[i]).tolist(),
    )


def scan_field_integral(z, field):
    """Fraction of the isolated round lens's axial Bz-squared integral between scan stations."""
    inside = (z > UPPER_MM) & (z < LOWER_MM)
    local = np.r_[UPPER_MM, z[inside], LOWER_MM]
    if not z[0] < UPPER_MM < LOWER_MM < z[-1]:
        raise ValueError("The axial solution must include both scan stations")
    numerator = np.trapezoid(np.interp(local, z, field) ** 2, local)
    return float(numerator / np.trapezoid(field**2, z))


def marker(ax):
    for label, z in [("Upper scan", UPPER_MM), ("Lower scan", LOWER_MM)]:
        ax.axvline(z, color="#5d6680", ls=":", lw=1.2)
        ax.text(
            z + 0.8,
            0.94,
            label,
            rotation=90,
            ha="left",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
            color="#3e465a",
        )


def prepare():
    recipe, _, archive_hash = audit_inputs()
    evidence = {}
    data = {}
    stats = {}
    stations = {}
    doc = read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    parts = {p["key"]: p for p in doc["parts"]}
    for key in list(CHANNELS) + ["total"]:
        with np.load(ASSEMBLED / (key + ".npz")) as f:
            z = f["z_mm"]
            y = f["bz_t"] if key in ROUND or key == "total" else f["s3_t_m2"]
            data[key] = y
            evidence[key] = {
                k: f[k].tolist() for k in ("check_z_mm", "native_air_bz_t", "material_volume_check_bz_t")
            }
            evidence[key]["receipt"] = json.loads(str(f["receipt"]))
            if key != "total":
                part = key[:-2] if key.startswith("probe_hpol") else key
                stations[key] = parts[part]["local_center_z_mm"] + 450
            stats[key] = metrics(z, y)
            if key not in (*ROUND, "total"):
                stats[key]["radius_test_fraction_of_peak"] = float(
                    max(abs(y - f["s3_inner_t_m2"])) / max(abs(y))
                )
    data["z_mm"] = z
    for spacing, padding in ((5.0, 170.0), (4.0, 210.0)):
        case = f"{spacing:g}-{padding:g}"
        path = WORK / f"profiles-{case}" / "tl12-only.npz"
        if not path.is_file():
            sol = load_solution(WORK / (case + "-tl12-only.npz"))
            r = json.loads(str(sol["report"]))
            sample_delta(
                sol, path, dict(currents=r["currents_A"], mesh_id=r["mesh_id"], residual=r["residual"])
            )
        with np.load(path) as f:
            np.testing.assert_array_equal(z, f["z_mm"])
            key = "tl12_only_" + case
            data[key] = f["bz_t"]
            stats[key] = metrics(z, f["bz_t"])
            evidence[key] = {
                k: f[k].tolist() for k in ("check_z_mm", "native_air_bz_t", "material_volume_check_bz_t")
            }
        with np.load(WORK / f"profiles-{case}" / "tl12-edge-load.npz") as f:
            key = "tl12_edge_load_" + case
            data[key] = f["bz_t"]
            stats[key] = metrics(z, f["bz_t"])
            evidence[key] = {
                k: f[k].tolist() for k in ("check_z_mm", "native_air_bz_t", "material_volume_check_bz_t")
            }
            evidence[key]["receipt"] = json.loads(str(f["receipt"]))
    for key in CHANNELS:
        with np.load(FOLDER / (key + ".npz")) as f:
            y = f["bz_t"] if key in ROUND else f["s3_t_m2"]
            data["centroid_load_" + key] = y
            stats["centroid_load_" + key] = metrics(z, y)
            evidence["centroid_load_" + key] = {
                k: f[k].tolist() for k in ("check_z_mm", "native_air_bz_t", "material_volume_check_bz_t")
            }
    section_stats = {}
    for file in sorted((WORK / "sections").glob("*.npz")):
        with np.load(file) as f:
            rec = json.loads(str(f["receipt"]))
            section_stats[file.stem] = dict(receipt=rec, **metrics(f["z_mm"], f["bz_t"]))
            if file.stem == TL12 + "-10-0.58-270":
                section = dict(f)
                data["section_z_mm"] = f["z_mm"]
                data["section_bz_t"] = f["bz_t"]
                data["winding_bz_t"] = f["vacuum_t"]
            if file.stem.endswith("-vacuum"):
                section_stats[file.stem]["max_absolute_air_FEM_error_over_analytic_peak"] = float(
                    max(abs(f["vacuum_t"] - f["bz_t"])) / max(f["vacuum_t"])
                )
    isolated_round_increments = {}
    for key, bias in [(ROUND[1], -145.0), (ROUND[2], 330.0)]:
        step = 0.01 * abs(bias)
        base = WORK / "sections" / f"{key}-{bias:g}-0.82-170.npz"
        perturbed = WORK / "sections" / f"{key}-{bias + step:g}-0.82-170.npz"
        with np.load(base) as a, np.load(perturbed) as b:
            np.testing.assert_array_equal(a["z_mm"], b["z_mm"])
            diff = (b["bz_t"] - a["bz_t"]) / step
            isolated_round_increments[key] = {"bias_A": bias, "step_A": step, **metrics(a["z_mm"], diff)}
    with np.load(ARCHIVE / "field-footprint-profiles.npz") as f:
        prior = f["amplitude"][list(f["names"]).index("TL12")]
        stats["archive_three_vacuum_groups"] = metrics(f["z_mm"], prior)
    stats["matched_single_winding"] = metrics(data["section_z_mm"], data["winding_bz_t"])
    joints = json.loads((HERE / "material-contacts.json").read_text())
    packs = json.loads((HERE / "winding-clearances.json").read_text())
    boundary_checks = json.loads((ASSEMBLED / "induction.json").read_text())
    observation = {}
    mark = dict(stations)
    for key in OTHER_STATIONS:
        if key in parts:
            mark[key] = parts[key]["local_center_z_mm"] + 450
    mark.update(upper_scan=UPPER_MM, lower_scan=LOWER_MM)
    for key, location in mark.items():
        observation[key] = dict(
            z_mm=location,
            total_corrector_axis_bz_t=float(np.interp(location, z, data["total"])),
            relative_to_each_coil_peak_percent={
                name: float(100 * np.interp(location, z, abs(data[k])) / max(abs(data[k])))
                for k, name in CHANNELS.items()
            },
        )
    result = dict(
        scope="Probe corrector ADL through TL12. Static 300 kV Nanoprobe Normal column geometry.",
        frame="Column z in mm, increasing downstream. Module origin +450 mm.",
        archive_sha256=archive_hash,
        section=section_stats,
        isolated_round_increments=isolated_round_increments,
        profiles=stats,
        checks=evidence,
        stages_mm=stations,
        field_at_stations=observation,
        isolated_TL12_scan_integral=dict(
            full_interval_mm=[float(section["z_mm"][0]), float(section["z_mm"][-1])],
            scan_stations_mm=[UPPER_MM, LOWER_MM],
            fraction_of_axial_Bz_squared={
                key: scan_field_integral(section["z_mm"], section[key]) for key in ("bz_t", "vacuum_t")
            },
        ),
        reference=boundary_checks,
        centroid_reference=json.loads((FOLDER / "induction.json").read_text()),
        edge_load_materials=json.loads((FOLDER / "edge-induction.json").read_text()),
        geometry=dict(
            solids=len(recipe["solids"]),
            contact_pairs=len(joints["contacts"]),
            isolated_sleeves=joints["isolated_solids"],
            winding_packs=len(packs),
            packs_intersecting_magnetic_material=sum(bool(p["intersections_mm3"]) for p in packs),
            minimum_pack_to_steel_clearance_mm=min(p["clearance_mm"] for p in packs),
        ),
        units="Joint coil differences: round T/A; hexapole complex T/m^2/A at r=1.2 mm; total and isolated TL12 T.",
    )
    _save(HERE / "axial-profiles.npz", **data)
    result["profile_sha256"] = sha256((HERE / "axial-profiles.npz").read_bytes()).hexdigest()
    (HERE / "magnetic-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_ranges(stats, stations, observation)
    return data, stats, stations, section


def write_ranges(stats, stations, observation):
    lines = [
        "# Field ranges in the assembled probe corrector",
        "",
        "Column z increases towards the specimen. Round lenses use the on-axis longitudinal field. "
        "HP2, HP1 and the two HPol channels use the magnitude of the complex transverse sextupole "
        "coefficient at a radius of 1.2 mm. Each interval is relative to the same drive's own maximum. "
        "The curves are sampled every 0.5 mm. Weak fringe fields continue outside the listed bands. "
        "[Excitations, material laws and numerical checks](README.md) · "
        "[All signed profiles and phase information](axial-profiles.npz)",
        "",
        "| Drive | Mechanical center (mm) | Peak z (mm) | At least 50% (mm) | At least 10% (mm) |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for key, name in CHANNELS.items():
        m = stats[key]
        bands = [", ".join(f"{lo:.1f}–{hi:.1f}" for lo, hi in m["bands_mm"][t]) for t in ("0.5", "0.1")]
        lines.append(f"| {name} | {stations[key]:.2f} | {m['peak_z_mm']:.1f} | {bands[0]} | {bands[1]} |")
    lines += [
        "",
        "## Fields present at the winding centers",
        "",
        "The third column is the signed total axial field generated by this corrector at the chosen "
        "currents. The last column measures how strongly another excited channel acts at that location, "
        "as a percentage of its own peak response to a current change. A dash means that each of the "
        "other channels falls below 10% of its own peak. The percentage is independent of the signed "
        "total axial field. Steerers and trims at zero drive are included as receiving locations.",
        "",
        "| Winding center | z (mm) | Total axial field (mT) | Other responses reaching 10% of their own peak |",
        "| --- | ---: | ---: | --- |",
    ]
    names = {**CHANNELS, **OTHER_STATIONS}
    for key, row in sorted(observation.items(), key=lambda item: item[1]["z_mm"]):
        name = names[key]
        if name == "HPol Y":
            continue
        self_names = {"HPol X", "HPol Y"} if name == "HPol X" else {name}
        others = sorted(row["relative_to_each_coil_peak_percent"].items(), key=lambda item: -item[1])
        influences = ", ".join(f"{ch} {p:.0f}%" for ch, p in others if ch not in self_names and p >= 10)
        field_mt = 1000 * row["total_corrector_axis_bz_t"]
        lines.append(
            f"| {'HPol' if name == 'HPol X' else name} | {row['z_mm']:.2f} | {field_mt:+.3f} | "
            f"{influences or '—'} |"
        )
    (HERE / "field-ranges.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_tl12(data, section):
    fig, (top, ax) = plt.subplots(
        2,
        1,
        figsize=(9.4, 6.9),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.12]},
        layout="constrained",
    )
    fig.suptitle("TL12 magnetic return and field at the scan stations", fontsize=15, weight="bold")
    zz, rr, flux = section["z_mm"], section["r_mm"], section["poloidal_flux_wb"]
    material = "#c6d5d9"
    shapes = json.loads(str(section["section_zr_mm"]))
    for shape in shapes.values():
        s = np.asarray(shape)
        top.fill(s[:, 0], s[:, 1], facecolor=material, edgecolor="#415e65", lw=0.85)
    top.add_patch(
        Rectangle((1354.87, 39.165), 22.86, 7.65, facecolor="#c77739", edgecolor="#84480d", alpha=0.95)
    )
    top.contour(
        zz, rr, flux, levels=np.linspace(0.045, 0.96, 15) * flux.max(), colors="#1c747d", linewidths=0.8
    )
    top.contour(
        zz,
        rr,
        flux,
        levels=np.array([0.002, 0.007, 0.016, 0.03]) * flux.max(),
        colors="#43858d",
        linewidths=0.65,
        linestyles="dashed",
    )
    top.set(ylim=(0, 89), ylabel="Radius (mm)", title="Isolated round-lens FEM, 19 turns at +10 A")
    top.legend(
        handles=[
            Patch(facecolor=material, edgecolor="#415e65", label="ARMCO return circuit"),
            Patch(facecolor="#c77739", label="Winding"),
            Line2D([], [], color="#1c747d", lw=1, label="Poloidal flux contours"),
        ],
        loc="upper left",
        fontsize=8,
    )
    marker(top)
    ax.plot(
        data["z_mm"],
        1000 * 10 * data[TL12],
        color=COLOR,
        lw=2.5,
        label="Assembled 3-D FEM, change with TL12 on",
    )
    ax.plot(
        zz, 1000 * section["bz_t"], color=PURPLE, ls="--", label="Isolated TL12, axisymmetric FEM", lw=1.7
    )
    ax.plot(
        zz, 1000 * section["vacuum_t"], color="#bb702c", lw=1.7, label="Same winding and current in vacuum"
    )
    ax.fill_betweenx([-1, 14], 1359.3, 1373.3, color="#dce8eb", alpha=0.55, zorder=0)
    ax.scatter(
        [UPPER_MM, LOWER_MM],
        np.interp([UPPER_MM, LOWER_MM], data["z_mm"], 10000 * data[TL12]),
        s=23,
        color=COLOR,
        zorder=4,
    )
    for station in (UPPER_MM, LOWER_MM):
        value = np.interp(station, data["z_mm"], 10000 * data[TL12])
        ax.annotate(
            f"{value:.2f} mT",
            (station, value),
            xytext=(9, 17),
            textcoords="offset points",
            fontsize=8,
            color=COLOR,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5),
            arrowprops=dict(arrowstyle="-", color=COLOR, linewidth=0.6),
        )
    ax.set(
        xlim=(1320, 1438),
        ylim=(-0.25, 12.9),
        xlabel="Column z (mm)",
        ylabel="Axial field (mT)",
        title="Signed on-axis field produced by 190 ampere-turns",
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.grid(axis="y", alpha=0.22)
    marker(ax)
    fig.savefig(HERE / "tl12-field.png", dpi=155)
    plt.close(fig)


def plot_locations(data, stations):
    z = data["z_mm"]
    fig, ax = plt.subplots(figsize=(10.2, 5), layout="constrained")
    for i, (key, name) in enumerate(CHANNELS.items()):
        base = len(CHANNELS) - 1 - i
        y = abs(data[key])
        y /= max(y)
        ax.fill_between(z, base, base + 0.82 * y, color=COLOR if key in ROUND else PURPLE, alpha=0.88)
        ax.vlines(stations[key], base - 0.05, base + 0.86, color="#313b44", lw=0.85, ls=":")
    ax.set_yticks(range(len(CHANNELS)), list(reversed(CHANNELS.values())))
    ax.set(
        xlim=(1050, 1438),
        ylim=(-0.12, 8.04),
        xlabel="Column z (mm)",
        title="Individual excitation responses in the assembled probe corrector",
    )
    ax.legend(
        handles=[
            Patch(color=COLOR, label="On-axis axial field"),
            Patch(color=PURPLE, label="Transverse sextupole coefficient"),
        ],
        loc="upper right",
        framealpha=0.97,
        fontsize=8,
        bbox_to_anchor=(0.77, 0.97),
    )
    ax.grid(axis="x", alpha=0.2)
    marker(ax)
    fig.text(
        0.5,
        -0.015,
        "Each row is normalized to its own peak. Short dotted marks identify mechanical centers.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(HERE / "corrector-field-locations.png", dpi=155, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data, stats, stations, section = prepare()
    plot_tl12(data, section)
    plot_locations(data, stations)
