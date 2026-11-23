"""Compare the signed axial field at fixed current, and quantify numerical sensitivity."""

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from condenser_pair_study import EV, OUT, Pair
from condenser_source_fields import sha, write
from condenser_source_search import transfer


def run():
    p = Pair()
    f = p.f
    point = json.loads((OUT / "point.json").read_text())
    z = np.linspace(0.310, 1.075, 3061)
    basis = f.round_axis(z)
    common = f.objective(z)[:, 0].copy()
    gun = z < 0.45
    common[gun] += f.far_corrector(z[gun])[:, 0, 2]
    common[~gun] += np.interp(z[~gun], f.zjoint, f.bjoint[:, 2])
    result = {}
    profiles = []
    for label, amp in (("matched", [-41.0, 19.5]), ("tuned", point["series_A"])):
        path, d = p.field(amp, 1.5, 550)
        _, fine = p.field(amp, 1.0, 800.0)
        _, coarse = p.field(amp, 2.5, 550)
        _, expanded = p.field(amp, 1.5, 800.0)
        q = f.current(*amp)
        vacuum = q @ basis + common
        y = vacuum + np.interp(z, d["z"], d["induced_T"])
        profiles.append((vacuum, y))

        def peaks(a):
            rows = []
            for lo, hi, func in ((0.45, 0.55, np.argmin), (0.55, 0.75, np.argmax)):
                sel = np.flatnonzero((z >= lo) & (z <= hi))
                k = sel[func(a[sel])]
                rows.append(dict(z_mm=float(z[k] * 1000), Bz_T=float(a[k])))
            return rows

        pc = q[:12] @ f.round_axis(d["fem_z"])[:12] + np.interp(d["fem_z"], d["z"], d["induced_T"])
        inside = (d["fem_z"] > 0.43) & (d["fem_z"] < 0.775)
        maxdelta = lambda a, b: float(np.max(abs(a["induced_T"] - b["induced_T"])))
        roots, im, mag, gap = p.prediction(amp, d)
        counts_ref, im_ref, mag_ref, gap_ref = p.prediction(amp, fine)
        nodes, mid, momentum, electric, first, second, static = p.optics
        vacuum_axis = static + amp[0] * first + amp[1] * second
        source_planes = {}
        for name, field in (
            ("vacuum", vacuum_axis),
            ("material", vacuum_axis + np.interp(mid, d["z"], d["induced_T"])),
        ):
            b = transfer(nodes, momentum, electric, field, True)[-1][:, 1]
            crosses = np.flatnonzero(b[1:] * b[:-1] < 0)
            positions = nodes[crosses] - b[crosses] * np.diff(nodes)[crosses] / np.diff(b)[crosses]
            source_planes[name] = (positions * 1000).tolist()
        result[label] = dict(
            series_A=amp,
            peak_vacuum=peaks(vacuum),
            peak_material=peaks(y),
            comparison_planes_mm=[490, 500, 540, 550, 560, 620, 650, 750],
            plane_vacuum_T=np.interp([0.49, 0.5, 0.54, 0.55, 0.56, 0.62, 0.65, 0.75], z, vacuum).tolist(),
            plane_material_T=np.interp([0.49, 0.5, 0.54, 0.55, 0.56, 0.62, 0.65, 0.75], z, y).tolist(),
            residual=float(d["relative_residual"]),
            peak_iron_T=float(d["peak_material_T"]),
            fine_peak_iron_T=float(fine["peak_material_T"]),
            nodes=int(d["nodes"]),
            fine_nodes=int(fine["nodes"]),
            coarse_to_tracking_max_delta_T=maxdelta(coarse, d),
            tracking_to_wider_max_delta_T=maxdelta(d, expanded),
            tracking_to_fine_wider_max_delta_T=maxdelta(d, fine),
            direct_FEM_bore_vs_open_recovery_max_T=float(np.max(abs(pc[inside] - d["fem_bore_T"][inside]))),
            independent_finer_counts=counts_ref.tolist(),
            prediction_counts=roots.tolist(),
            paraxial_source_conjugates_mm=source_planes,
            prediction_image_mm=1000 * im,
            prediction_gap_mm=1000 * gap if roots[2] else None,
            fine_image_mm=1000 * im_ref,
            fine_gap_mm=1000 * gap_ref if counts_ref[2] else None,
            field_digest=sha(path),
            curve_digests=d["curve_digests"].tolist(),
        )
        shutil.copyfile(path, EV / f"{label}-material-field.npz")
    result["parts"] = json.loads(str(d["parts_json"]))
    result["fixed_remote_axis_max_T_on_iron_z"] = float(
        np.max(abs((f.current(0, 0)[12:] @ basis[12:] + common)[(z >= 0.45) & (z <= 0.75)]))
    )
    result["scope"] = (
        "C1 and C2 windings with their installed poles and outer returns, including incident C3 windings. Independent downstream material fields retained. Unspecified carrier mu_r=1."
    )
    write(EV / "axial-comparison.json", result)
    np.savez_compressed(
        EV / "axial-profiles.npz",
        z_mm=z * 1000,
        matched_vacuum_T=profiles[0][0],
        matched_iron_T=profiles[0][1],
        tuned_iron_T=profiles[1][1],
    )
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10.5, 6.1), height_ratios=[2.3, 1], sharex=True, layout="constrained"
    )
    for a in (ax, ax2):
        a.axhline(0, color="#75808b", lw=0.7)
        a.grid(axis="y", alpha=0.18)
        for lo, hi in ((490, 540), (560, 620)):
            a.axvspan(lo, hi, color="#b2c7d2", alpha=0.25, lw=0)
        a.spines[["right", "top"]].set_visible(False)
    for y, title, color, ls in [
        (profiles[0][0], "Vacuum windings at −41 / +19.5 A", "#515a69", "--"),
        (profiles[0][1], "Poles and returns at −41 / +19.5 A", "#ad442b", "-"),
        (profiles[1][1], "Poles and returns at −30 / +16 A", "#007a81", "-"),
    ]:
        ax.plot(z * 1000, y * 1000, c=color, ls=ls, label=title, lw=2.15)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_ylabel("Signed axial field  $B_z$ (mT)")
    ax.set_title("C1 lens and C2 lens  |  axial field along the optical axis", loc="left", weight="bold")
    ax2.plot(z * 1000, (profiles[0][1] - profiles[0][0]) * 1000, color="#ad442b", lw=1.9)
    ax2.set(
        ylabel="Material contribution\n(mT)",
        xlabel="Distance from the effective source  z (mm)",
        xlim=(375, 840),
    )
    ax2.text(
        0.988,
        0.96,
        "Same currents · shading marks the two pole pieces",
        transform=ax2.transAxes,
        va="top",
        ha="right",
        fontsize=9,
    )
    fig.savefig(EV / "axial-field.png", dpi=165)
    print(
        json.dumps(
            {
                k: {
                    x: result[k][x]
                    for x in (
                        "peak_vacuum",
                        "peak_material",
                        "tracking_to_fine_wider_max_delta_T",
                        "tracking_to_wider_max_delta_T",
                        "fine_image_mm",
                        "fine_gap_mm",
                    )
                }
                for k in ("matched", "tuned")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
