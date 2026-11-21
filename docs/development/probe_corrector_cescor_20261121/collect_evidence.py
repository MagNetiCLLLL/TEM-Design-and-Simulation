"""Archive the executed comparison, preserving currents, source IDs and measured focal planes."""

from hashlib import sha256
import json
from shutil import copyfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from compare import ARCHIVE, HERE, POINT, WORK, metrics


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def sextupole_coefficients(data):
    """Resolve the complex transverse order on a ring in each physical pole interval."""
    xy = data["xy_m"]
    z = data["z_local_m"]
    interpolate = RegularGridInterpolator((z, xy, xy), data["B_zyx_T"])
    angle = 2 * np.pi * np.arange(120) / 120
    result = {}
    for name, center in (("HP1", 0.85085), ("HP2", 0.69450)):
        lo, hi = center - 0.015, center + 0.015
        axial = np.r_[lo, z[(z > lo) & (z < hi)], hi]
        result[name] = {}
        for radius in (0.0012, 0.0018, 0.0024):
            points = np.stack(
                np.broadcast_arrays(axial[:, None], radius * np.sin(angle), radius * np.cos(angle)), axis=-1
            )
            b = interpolate(points)
            harmonic = ((b[..., 0] + 1j * b[..., 1]) * np.exp(2j * angle)).mean(axis=1) / radius**2
            integrated = np.trapezoid(harmonic, axial)
            result[name][f"{radius*1000:g}_mm"] = dict(
                peak_T_m2=float(abs(harmonic).max()),
                integral_real_T_m=float(integrated.real),
                integral_imag_T_m=float(integrated.imag),
            )
    return result


def collect():
    evidence = HERE / "evidence"
    evidence.mkdir(exist_ok=True)
    reference = json.loads((POINT / "currents.json").read_text())
    driver_keys = [
        "condenser_lens_2",
        "condenser_lens_3",
        "mini_condenser",
        "adapter_lens",
        "probe_tl22_lens",
        "probe_tl21_lens",
        "probe_tl12_lens",
        "condenser_lens_1",
    ]
    drives = np.array(reference["current_a"])
    owners = np.array(reference["owner_per_group"])
    study = dict(
        aperture_um=100.0,
        minimum_transmitted_uA=9.0,
        emitted_uA=10.0,
        source_count=32768,
        reference_energy_keV=300.0,
        gun_exit_sha256=digest(POINT / "gun.npz"),
        electrostatic_grid_sha256=json.loads((ARCHIVE / "study.json").read_text())[
            "electrostatic_grid_sha256"
        ],
        sample_upper_surface_global_mm=1599.1999975,
        fields={},
        probes={},
        archive_baseline=json.loads((ARCHIVE / "after/focus.json").read_text()),
    )
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    colours = ["#007873", "#6a7581", "#ba592b"]
    cases = [
        ("material", "fine-field", "material-final", "Ni–Fe pins + iron returns"),
        ("vacuum_same_current", "vac-field", "vacuum-final", "Vacuum, same current"),
        (
            "vacuum_equal_integral",
            "vacuum-equal-field",
            "equal-final",
            "Vacuum, matched field integral",
        ),
        ("coarser_mesh", "field", "coarse-final", "Coarser mesh"),
        ("half_transport_step", "fine-field", "step-final", "Half step"),
    ]
    reference_source = None
    axial_profiles = {}
    for index, (name, grid, probe, label) in enumerate(cases):
        table = WORK / (grid + ".npz")
        with np.load(table) as data:
            r = json.loads(str(data["report"]))
            if r["vacuum_only"]:
                r.update(material_peaks={}, iterations=0, residual=0.0)
            study["fields"][name] = {
                **r,
                "sha256": digest(table),
                "sextupole": sextupole_coefficients(data),
            }
            j = len(data["xy_m"]) // 2
            if index < 3:
                z = data["z_local_m"] + 0.45
                bz = data["B_zyx_T"][:, j, j, 2]
                axial_profiles[f"{name}_global_z_m"] = z
                axial_profiles[f"{name}_Bz_T"] = bz
                ax.plot(
                    1000 * z,
                    1000 * bz,
                    color=colours[index],
                    lw=1.7,
                    label=label,
                )
        row = json.loads((WORK / (probe + ".json")).read_text())
        arrays = WORK / (probe + ".npz")
        with np.load(arrays) as data:
            live = data["loss"] == -1
            source = tuple(data[k] for k in ("ids", "weights", "energy_offset_eV"))
            if reference_source is None:
                reference_source = source
            for current, original in zip(source, reference_source):
                np.testing.assert_array_equal(current, original)
            assert len(np.unique(source[0])) == 32768 and np.all(source[1] > 0)
            np.testing.assert_allclose(source[1].sum(), 1, rtol=1e-12)
            p, w = data["phase"][live], data["weights"][live]
            result = metrics(p, w, 1e-5)
            row["surface"].update(result)
            if row["peak_core_nA"] == 0:
                row["first_max_core_offset_nm"] = None
            row["source_id_sha256"] = sha256(data["ids"].tobytes()).hexdigest()
            row["surface_array_sha256"] = digest(arrays)
            prefixes = {}
            for n in (8192, 16384, 32768):
                present = live[:n]
                prefix = metrics(data["phase"][:n][present], data["weights"][:n][present], 1e-5)
                prefixes[str(n)] = {k: prefix[k] for k in ("D95_nm", "alpha95_mrad", "core_count")}
            row["source_prefix_shape_at_fixed_controls"] = prefixes
            row["effective_survivor_count"] = float(w.sum() ** 2 / (w @ w))
            row["background_signed_groups_A"] = {
                k: (drives[owners == k] * scale).tolist()
                for k, scale in zip(driver_keys, row["controls"])
                if k not in ("probe_tl22_lens", "probe_tl21_lens")
            }
            if index < 3:
                copyfile(arrays, evidence / (name + ".npz"))
                center = np.average(p[:, :2], axis=0, weights=w)
                radius = np.linalg.norm(p[:, :2] - center, axis=1)
                order = np.argsort(radius)
                bx.plot(
                    1e6 * radius[order],
                    10 * np.cumsum(w[order]),
                    color=colours[index],
                    lw=1.5,
                    ls=("-", "--", ":")[index],
                )
        study["probes"][name] = row
    for k, z in (("HP2", 1144.5), ("TL22", 1166.35), ("TL21", 1249.95), ("HP1", 1300.85)):
        ax.axvline(z, lw=0.65, color="#999999", ls=":")
        ax.text(z, 1.01, k, transform=ax.get_xaxis_transform(), ha="center", size=8)
    ax.set(xlim=(1110, 1330), xlabel="Column z (mm)", ylabel="Axial field on beam axis (mT)")
    ax.legend(fontsize=8, loc="lower right")
    bx.set(
        xlim=(0, 75),
        ylim=(0, 10.1),
        xlabel="Radius from current centroid (µm)",
        ylabel="Enclosed current at specimen (µA)",
    )
    for a in (ax, bx):
        a.grid(alpha=0.18)
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "field-and-probe.png", dpi=175)
    plt.close(fig)
    np.savez_compressed(evidence / "axial-fields.npz", **axial_profiles)
    study["objective_field"] = json.loads((WORK / "column/armco-36000-gap-fine-p8.json").read_text())
    (evidence / "comparison.json").write_text(json.dumps(study, indent=2), encoding="utf8")


if __name__ == "__main__":
    collect()
