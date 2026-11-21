"""Publish the compact, measured results and the comparison figure."""

import csv
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

from column import CACHE, HERE, POINT, SURFACE, metrics
from fields import CASES

LABELS = {
    "narrow": "Narrow tip",
    "photo": "Photo profile",
    "wide": "Wide tip",
    "photo_iron_tube": "Photo + iron tube",
}
ORDER = tuple(LABELS)
COLORS = ["#b05d12", "#165381", "#23734e", "#884290"]
STYLES = ["--", "-", "-.", ":"]


def read(name):
    return json.loads((CACHE / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = {}
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), layout="constrained")
    contour, field_ax, focus, enc = axes.flat
    inset = field_ax.inset_axes([0.61, 0.52, 0.36, 0.38])
    for i, case in enumerate(ORDER):
        iron = case == "photo_iron_tube"
        primary = "jointfine"
        mesh_tag = "jointmedium"
        step_tag = "jointhalf"
        probefile = f"probe-{case}-{primary}"
        p = read(probefile + ".json")
        mesh = read(f"probe-{case}-{mesh_tag}.json")
        step = read(f"probe-{case}-{step_tag}.json")
        fm = {
            t: read(f"{case}-3000-{level}-p{pad}.json")
            for t, level, pad in [
                ("base", "base", 8),
                ("fine", "fine", 8),
                ("finer", "finer", 8),
                ("expanded_base", "base", 9),
                ("interface", "interface", 8),
                ("reference", "reference", 8),
            ]
        }
        fid = fm["reference"]
        with np.load(CACHE / (probefile + ".npz")) as d:
            live = d["loss_step"] == -1
            weights = d["weight"]
            phase = d["phase"]
            f_z = (d["focus_z_m"] - SURFACE) * 1e6
            d95 = d["focus_D95_RMS_nm_core_nA"][:, 0] * 0.001
            ids = d["ray_id"]
            assert len(np.unique(ids)) == 32768 and len(weights) == 32768 and weights.sum() == 1.0
            by_count = {}
            for n in [4096, 8192, 16384, 32768]:
                ok = live & (np.arange(len(ids)) < n)
                by_count[str(n)] = metrics(phase[ok], weights[ok], 1e-5)
            r = np.hypot(phase[live, 0], phase[live, 1])
            current = {
                str(nm): float(1e4 * weights[live][r <= nm * 1e-9].sum()) for nm in [0.05, 1, 10, 50, 100]
            }
            sort = np.argsort(r)
            radius_nm = r[sort] * 1e9
            enclosed_uA = 10 * np.cumsum(weights[live][sort])
        surf = p["surface"]
        assert 29.7 <= surf["alpha95_mrad"] <= 30.3 and surf["transmitted_uA"] >= 9
        assert p["D95_waist_offset_nm"] == 0 and p["focus_counts_stable"]
        assert max(abs(p["D95_surface_minus_min_nm"]), abs(step["surface"]["D95_nm"] - surf["D95_nm"])) < 0.1
        assert p["max_surviving_radius_mm"] <= 2.88 and p["D95_waist_bracket_nm"] == [-0.5, 0.0, 0.5]
        assert [len(c) for c in p["x_y_waist_brackets_mm"]["C2_to_C3"]] == [1, 1]
        assert p["x_y_waist_brackets_mm"]["C3_to_ADL"] == [[], []]
        assert all(
            z[0] >= SURFACE * 1000
            for direction in p["x_y_waist_brackets_mm"]["Mini_to_sample"]
            for z in direction
        )
        older_tag = "anglepair" if iron else "verified"
        data[case] = dict(
            primary=p,
            adjacent_field_mesh_refocused=mesh,
            half_step_fixed_controls=step,
            original_short_refinement_window=read(f"probe-{case}-{older_tag}.json"),
            common_excitation_fields=fm,
            source_prefix_metrics=by_count,
            axis_disk_current_nA=current,
            output_archive=probefile + ".npz",
            output_sha256=digest(CACHE / (probefile + ".npz")),
        )
        rows.append(
            dict(
                case=case,
                face_OD_mm=CASES[case][0],
                knee_OD_mm=CASES[case][1],
                Bz_sample_mT_at_3000_At=1000 * fid["B_sample_T"],
                Bz_peak_mT=1000 * fid["peak_axis_T"],
                Bz2_integral_T2_m=fid["B2_integral_T2m"],
                connector_axial_flux_uWb=1e6 * fid["joint_flux_Wb"],
                optical_excitation_At=p["ni"],
                alpha95_mrad=surf["alpha95_mrad"],
                transmitted_uA=surf["transmitted_uA"],
                D95_um=surf["D95_nm"] * 0.001,
                D90_um=surf["D90_nm"] * 0.001,
                RMS_radius_um=surf["rms_radius_nm"] * 0.001,
                D95_waist_offset_nm=p["D95_waist_offset_nm"],
                RMS_waist_offset_um=p["rms_waist_offset_nm"] * 0.001,
                core50pm_nA=surf["core_50pm_nA"],
                core100nm_nA=current["100"],
                adjacent_mesh_refocused_D95_um=mesh["surface"]["D95_nm"] * 0.001,
            )
        )
        style = dict(color=COLORS[i], linestyle=STYLES[i], lw=1.8)
        section = np.array(fid["section_mm"])
        if not iron:
            start = np.flatnonzero(section[:, 1] == 36.0)[0]
            section = section[start : start + 4]
            contour.plot(section[:, 0] - SURFACE * 1000, section[:, 1], **style)
        with np.load(CACHE / f"{case}-3000-reference-p8.npz") as g:
            bz = make_interp_spline(g["z"], 2 * g["coeff"][0])
            z = np.linspace(-150, 120, 3200)
            field_ax.plot(z, 1000 * bz(SURFACE + z * 0.001), label=LABELS[case], **style)
            z = np.linspace(-6, 6, 193)
            if not iron:
                inset.plot(z, 1000 * bz(SURFACE + z * 0.001), **style)
        focus.plot(f_z, d95, **style)
        # Quantiles and fixed-radius currents use all packets. Decimation is only for plotting.
        inds = np.unique(np.r_[np.arange(min(300, len(r))), np.linspace(0, len(r) - 1, 1000, dtype=int)])
        enc.step(radius_nm[inds], enclosed_uA[inds], where="post", **style)
    for ax in [field_ax, focus]:
        ax.axvline(0, c="#606570", lw=0.8, alpha=0.65)
    contour.set(
        title="Upper pole nose",
        xlabel="Axial position relative to specimen (mm)",
        ylabel="Radius (mm)",
        ylim=(0, 40),
    )
    contour.axhline(2.88, c="#777", lw=0.8, ls=":", label="Bore radius")
    contour.text(-25.0, 3.55, "Bore radius 2.88 mm", fontsize=8, color="#555")
    field_ax.set(
        title="Objective field at +3,000 ampere-turns",
        xlabel="Axial position relative to specimen (mm)",
        ylabel="Axial field (mT)",
    )
    inset.set(xlim=(-6, 6), ylim=(15.6, 17.9))
    inset.set_title("Tip profiles near gap", fontsize=8)
    inset.tick_params(labelsize=7)
    focus.set(
        title="D95 through the specimen plane",
        xlabel="Defocus from specimen entrance (µm)",
        ylabel="Current-weighted D95 (µm)",
        xlim=(-45, 70),
        ylim=(5.6, 13.8),
    )
    enc.set(
        title="Current inside a disk on the optical axis",
        xlabel="Disk radius (nm)",
        ylabel="Enclosed current (µA)",
        xscale="log",
        yscale="log",
        xlim=(0.05, 10000),
        ylim=(0.00025, 12),
    )
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    handles, labels = field_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    fig.savefig(HERE / "comparison.png", dpi=170)
    plt.close(fig)
    with (HERE / "comparison.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    gun = json.loads((POINT / "gun.json").read_text())
    with np.load(CACHE / "electric.npz") as ef:
        electrostatic_digest = str(ef["digest"])
    archive = dict(
        gun_sha256=gun["sha256"],
        gun_plane_m=0.45,
        emitted_current_A=1e-5,
        nominal_energy_eV=300000,
        energy_FWHM_eV=0.30,
        source_count=32768,
        minimum_transmission_uA=9,
        aperture_diameter_um=100,
        core_radius_pm=50,
        electrostatic_digest=electrostatic_digest,
        original_currents_sha256=digest(POINT / "currents.json"),
        multipliers_in_order=["C2", "C3", "Mini", "ADL", "TL22", "TL21", "TL12", "C1"],
        cases=data,
        parked_control=read("transport-check.json"),
        matched_cpu_check=read("matched-transport-check.json"),
    )
    (HERE / "results.json").write_text(json.dumps(archive, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
