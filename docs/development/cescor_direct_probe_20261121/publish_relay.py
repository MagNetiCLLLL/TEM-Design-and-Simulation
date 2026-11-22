"""Archive matched specimen and transfer measurements on absolute current and position scales."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np

from relay_transfer import HERE, HP1, HP2, OUT, SURFACE, TL21, TL22, WORK
from column_current import OWNER, ACTIVE


def read(name):
    return json.loads((WORK / (name + ".json")).read_text(encoding="utf-8"))


def magnet(path):
    with np.load(path) as arch:
        j = json.loads(str(arch["report"]))
        z = 450 + arch["z_local_m"] * 1e3
        xy = arch["xy_m"]
        centre = len(xy) // 2
        b = arch["B_zyx_T"][:, centre, centre, 2] * 1000
    peaks = {}
    for key, position in [("TL21", TL21), ("TL22", TL22)]:
        keep = abs(z - position * 1e3) <= 20.0
        i = np.argmax(abs(b[keep]))
        peaks[key] = dict(z_mm=float(z[keep][i]), Bz_mT=float(b[keep][i]))
    return dict(
        field_file=path.name,
        sha256=sha256(path.read_bytes()).hexdigest(),
        peaks=peaks,
        **{
            k: j[k]
            for k in (
                "currents_A",
                "turns",
                "tetrahedra",
                "nodes",
                "material_size_mm",
                "mesh_id",
                "material_peaks",
                "residual",
                "circuit_sha256",
            )
        },
    )


def cropped(row):
    kept = {
        k: row[k]
        for k in (
            "currents_A",
            "objective_At",
            "specimen",
            "source_digest",
            "emitted",
            "survived",
            "current_ledger",
            "entrance",
            "relay",
            "central_relay",
            "field_sha256",
        )
    }
    for k in ("rim", "rim_sensitivity", "rim_azimuth_check"):
        kept[k] = {i: v for i, v in row[k].items() if i not in ("azimuth_deg", "angle_mrad")}
    kept["physical_edge_cohorts"] = row["physical_edge_cohorts"]
    return kept


def save_figure(cases, folder):
    fig = plt.figure(figsize=(11.2, 7.9), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.18, 0.95], width_ratios=[1, 1, 0.055])
    ax = [fig.add_subplot(gs[0, i]) for i in range(2)]
    cb = fig.add_subplot(gs[0, 2])
    extent = max(0.3, max(row["specimen"]["D95_nm"] for _, row, _, _ in cases) * 0.00065)
    edges = np.linspace(-extent, extent, 141)
    area = (edges[1] - edges[0]) ** 2
    hist = []
    fig.suptitle("CESCOR relay and specimen probe", fontsize=19, fontweight="bold")
    for label, row, phase, weights in cases:
        hist.append(
            np.histogram2d(*(phase[:, :2] * 1e6).T, bins=[edges, edges], weights=weights * 15000)[0].T / area
        )
    norm = LogNorm(vmin=min(x[x > 0].min() for x in hist), vmax=max(x.max() for x in hist))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#100e19")
    for a, h, (label, row, _, _) in zip(ax, hist, cases):
        im = a.pcolormesh(edges, edges, np.ma.masked_less_equal(h, 0), norm=norm, cmap=cmap, rasterized=True)
        a.add_patch(plt.Circle((0, 0), 0.05, fill=False, color="#39e2d2", lw=1.15))
        a.set(aspect="equal", xlabel="Sample x (µm)", ylabel="Sample y (µm)")
        a.set_title(f"{label}   ·   D95 {row['specimen']['D95_nm']*.001:.3g} µm", loc="left", fontsize=12)
    bar = fig.colorbar(im, cax=cb)
    bar.set_label("Landing current density (nA/µm²)")
    colors = ["#376cb0", "#bc4d2e"]
    a = fig.add_subplot(gs[1, 0])
    for color, (label, row, p, w) in zip(colors, cases):
        radius = np.linalg.norm(p[:, :2], axis=1) * 1e9
        order = np.argsort(radius)
        a.step(radius[order], np.cumsum(w[order]) * 15, where="post", label=label, color=color, lw=1.6)
    a.axvline(50, color="#327b73", ls=":", lw=1.0)
    a.set(
        xscale="log",
        xlabel="Radius from optical axis (nm)",
        ylabel="Enclosed current (µA)",
        ylim=(0, 15.7),
        xlim=(1, max(1000, extent * 3000)),
    )
    a.set_title("Current around the fixed axis", loc="left", fontsize=12)
    a.legend(frameon=False, fontsize=9)
    a.grid(which="major", alpha=0.21)
    b = fig.add_subplot(gs[1, 1:])
    for offset, color, (_, row, _, _) in zip([-0.12, 0.12], colors, cases):
        x = np.arange(3) + offset
        stat = np.array(
            [row["relay"][k]["singular_values_p05_p50_p95_mm_per_rad"] for k in ["0-1", "1-2", "0-2"]]
        )[:, :, 0]
        b.errorbar(
            x,
            stat[:, 1],
            yerr=np.stack([stat[:, 1] - stat[:, 0], stat[:, 2] - stat[:, 1]]),
            fmt="o",
            color=color,
            capsize=4,
            ms=6,
        )
    b.set_xticks(np.arange(3), ["HP2 → HP1", "HP1 → pupil", "HP2 → pupil"])
    b.set(xlim=(-0.45, 2.45), yscale="log", ylabel="Landing shift for 1 µrad (nm)")
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.set_title("Stronger transverse response   ·   median and 5–95%", loc="left", fontsize=12)
    b.grid(axis="y", which="major", alpha=0.23)
    fig.savefig(folder / "relay-conjugacy-probe.png", dpi=176, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def publish(args):
    folder = HERE / "evidence"
    cases = []
    archive = {}
    records = {}
    for label, name in [("Before", args.before), ("After", args.after)]:
        j = read(name)
        with np.load(WORK / (name + ".npz")) as f:
            p = f["phase"]
            w = f["weights"]
            ids = f["ids"]
            loss = f["loss"]
            energy = f["energy_offset_eV"]
        if archive:
            for k, array in [("source_id", ids), ("weight", w), ("energy_offset_eV", energy)]:
                np.testing.assert_array_equal(archive[k], array)
        else:
            archive.update(source_id=ids, weight=w, energy_offset_eV=energy)
        live = loss == -1
        np.testing.assert_allclose(w[live].sum() * 15, j["specimen"]["transmitted_uA"], rtol=1e-12)
        np.testing.assert_allclose(
            15000 * w[live & (np.linalg.norm(p[:, :2], axis=1) <= 50e-9)].sum(),
            j["specimen"]["axis_50nm_uA"] * 1000,
            rtol=1e-12,
        )
        archive[label.lower() + "_phase"] = p
        archive[label.lower() + "_loss"] = loss
        cases.append((label, j, p[live], w[live]))
        row = cropped(j)
        ax = read(name + "-axial")
        row["focus"] = {
            k: ax[k]
            for k in [
                "surface",
                "C2_foil",
                "D95_minimum_bracket_nm",
                "RMS_minimum_bracket_nm",
                "waists_mm",
                "common_waists_mm",
            ]
        }
        core = read(name + "-core")
        row["focus"]["fixed_axis_core"] = {
            k: core[k] for k in ["radius_nm", "maximum_nA", "offsets_at_max_um", "surface_nA"]
        }
        row["source_effective"] = w[live].sum() ** 2 / np.sum(w[live] ** 2)
        records[label.lower()] = row
    paths = [OUT / "a2-cubic-both-fine-field.npz", OUT / "relay30-fine-field.npz"]
    paths += [WORK / (p + "-field.npz") for p in ("m160", "m330", "m370", "m420", "m330-log", "m330-32")]
    records["nonlinear_fields"] = [magnet(f) for f in paths]
    for k in ("before", "after"):
        row = records[k]
        field = next(j for j in records["nonlinear_fields"] if j["sha256"] == row["field_sha256"])
        row["field_file"] = field["field_file"]
        row["corrector_windings_A"] = field["currents_A"]
        row["background_windings_A"] = {
            owner: np.array(row["currents_A"])[OWNER[ACTIVE] == owner].tolist()
            for owner in dict.fromkeys(OWNER[ACTIVE])
        }
        row["objective_current_A"] = row["objective_At"] / 1160.0
    records["surface_z_mm"] = 1000 * SURFACE
    records["hexapole_centres_mm"] = [1000 * HP2, 1000 * HP1]
    previous = read("historical")
    records["previous_percentile_setting"] = {
        key: previous[key]
        for key in (
            "specimen",
            "source_digest",
            "emitted",
            "survived",
            "objective_At",
            "field_sha256",
            "rim",
            "current_ledger",
        )
    }
    records["objective_pivots"] = []
    for name in args.pivot:
        j = read(name)
        records["objective_pivots"].append(
            {
                k: j[k]
                for k in [
                    "objective_At",
                    "entry_m",
                    "scan_um",
                    "ring_mrad",
                    "tilt_mrad",
                    "step_mm",
                    "roots",
                    "perpendicular_median",
                    "perpendicular_p10_p90",
                ]
                if k in j
            }
        )
    records["sensitivity"] = {}
    for name in args.check:
        j = read(name)
        records["sensitivity"][name] = dict(
            specimen=j["specimen"],
            emitted=j["emitted"],
            survived=j["survived"],
            field_sha256=j["field_sha256"],
            edge=j["rim_azimuth_check"]["angle_mrad_min_median_max"],
            max_step_mm=j.get("max_step_mm"),
            source_digest=j["source_digest"],
            relay=j.get("relay"),
            central_relay=j.get("central_relay"),
        )
    records["source_current_A"] = 15e-6
    records["C2_aperture_diameter_um"] = 100.0
    records["matched_names"] = [args.before, args.after]
    records["measurement"] = {
        "core_axis": "Fixed installed optical axis",
        "core_radius_nm": 50.0,
        "D95_centre": "Surviving current centroid",
        "edge": "Maximum mechanical specimen angle over 128 azimuths approaching the limiting C2 foil face from inside",
        "relay": "Position response to symmetric laboratory slope perturbations at each source plane; rows x,y; columns tx,ty",
        "phase_columns": ["x_m", "y_m", "dx_dz", "dy_dz", "time_s"],
    }
    records["landing_file"] = "relay-conjugacy-landings.npz"
    np.savez_compressed(folder / records["landing_file"], **archive)
    (folder / "relay-conjugacy.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_figure(cases, HERE)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", default="comparison-before")
    p.add_argument("--after", required=True)
    p.add_argument("--pivot", nargs="+", required=True)
    p.add_argument("--check", nargs="*", default=[])
    publish(p.parse_args())
