"""Independent full-source measurements and figures from completed trajectory archives."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from temsim.physics.condenser_relay import beam_widths
from temsim.joint_fem import input_file
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import AsinhNorm
from matplotlib.figure import Figure
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "checkpoint"

REGIONS = [
    (0, 36.5, "Gun"),
    (450, 550, "C1"),
    (550, 750, "C2"),
    (830, 1000, "C3"),
    (1070, 1106.4, "ADL"),
    (1129.5, 1379, "Corrector"),
    (1380, 1420, "Scan"),
    (1512.5, 1532.5, "Mini"),
    (1534.5, 1596.5, "Obj."),
]
BLUE, RED = "#276782", "#bd5835"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    """Index the saved display paths by original source identity."""
    run = dict(
        folder=path,
        profile=json.loads((path / "profile.json").read_text()),
        result=json.loads((path / "result.json").read_text()),
    )
    keys = dict(
        z="z_mm",
        paths="paths_mm",
        ids="selected_ids",
        w="emitter_weights_uA",
        sample="sample_ids",
        gun="gun_exit_ids",
        c2="C2_transmitted_ids",
        xy="sample_xy_um",
        w_sample="sample_weights_uA",
    )
    with np.load(path / "trajectories.npz") as ar:
        run.update({key: ar[name] for key, name in keys.items()})
        np.testing.assert_array_equal(
            run["gun"], np.flatnonzero(ar["gun_events"] == -1)
        )
        np.testing.assert_array_equal(
            run["sample"], run["gun"][ar["column_loss"] == -1]
        )
    assert run["result"]["field_sha256"] == run["profile"]["solution_sha256"]
    assert len(run["w"]) == 100000 and np.all(run["w"] > 0)
    np.testing.assert_array_equal(run["w"][run["sample"]], run["w_sample"])
    order = np.argsort(run["ids"])
    rows = order[np.searchsorted(run["ids"][order], run["sample"])]
    np.testing.assert_array_equal(run["ids"][rows], run["sample"])
    run["sample_rows"] = rows
    return run


def metrology(run, focus_path):
    path = run["folder"]
    surface = run["z"][-1]
    dense = path / "downstream-window.npz"
    if dense.is_file():
        with np.load(dense) as ar:
            z = ar["z_mm"]
            ids = ar["source_ids"]
            paths = ar["paths_mm"]
            np.testing.assert_array_equal(run["w"][ids], ar["weights_uA"])
            present = np.isfinite(paths[:, 0]).all(axis=1)
            assert np.array_equal(np.isfinite(paths).all(axis=2).any(axis=1), present)
            ids, paths = ids[present], paths[present]
    else:
        # When dense live-window paths are absent, use the complete C2 survivors and
        # check the recovered roots against the native full live-current inventory.
        idx = np.flatnonzero(np.isin(run["ids"], run["c2"]))
        retained = run["ids"][idx]
        np.testing.assert_array_equal(np.sort(retained), np.sort(run["c2"]))
        sel = run["z"] >= 1176.0 - 1e-7
        z, ids, paths = run["z"][sel], retained, run["paths"][idx][:, sel]
    _, roots = beam_widths(z * 0.001, paths * 0.001, run["w"][ids])
    for key, root in roots.items():
        reported = [
            val
            for val in run["result"]["column_live_waists_mm"][key]
            if z[0] < val < z[-1]
        ]
        np.testing.assert_allclose(root, reported, rtol=0.0, atol=0.0008)
    with np.load(focus_path) as ar:
        curve = np.column_stack(
            (ar["z_mm"], ar["D95_um"], ar["rms_um"], ar["rms_xy_um"])
        )
    zz = curve[:, 0]
    assert np.all(np.diff(zz) > 0)
    focus = {}
    for j, quantity in enumerate(("D95", "r", "x", "y"), start=1):
        size = curve[:, j]
        idx = int(np.argmin(size))
        is_inside = 0 < idx < len(zz) - 1
        fitted = zz[idx]
        if quantity != "D95" and is_inside:
            t = zz[idx - 1 : idx + 2] - zz[idx]
            a, b, _ = np.polyfit(t, size[idx - 1 : idx + 2] ** 2, 2)
            if a > 0 and t[0] <= -b / (2 * a) <= t[-1]:
                fitted = zz[idx] - b / (2 * a)
        focus[quantity] = dict(
            z_mm=fitted,
            offset_from_specimen_mm=fitted - surface,
            sampled_minimum_um=size[idx],
            sampled_z_mm=zz[idx],
            bracketed=is_inside,
            bracket_mm=zz[max(0, idx - 1) : idx + 2].tolist(),
        )
    return {
        "field_sha256": run["result"]["field_sha256"],
        "all_downstream_xyr_stationary_planes_mm": roots,
        "specimen_z_mm": float(surface),
        "terminal_focus": focus,
    }, curve


def block_sensitivity(old, new):
    """Omit matching original-source ID blocks; these are sensitivity ranges."""
    rows = []
    for group in range(20):
        values = []
        for run in (old, new):
            used = run["sample"] // 5000 != group
            xy = run["xy"][used]
            w = run["w_sample"][used]
            centre = np.average(xy, axis=0, weights=w)
            r = np.linalg.norm(xy - centre, axis=1)
            order = np.argsort(r)
            diameter = (
                2 * r[order[np.searchsorted(np.cumsum(w[order]), 0.95 * w.sum())]]
            )
            disk = w[np.linalg.norm(xy, axis=1) <= 0.05].sum() * 1e6
            values.append(np.array([diameter, disk]))
        rows.append(values[1] - values[0])
    deltas = np.array(rows)
    return {
        "groups": len(rows),
        "block_size_emitted_ids": 5000,
        "D95_delta_min_max_um": [float(deltas[:, 0].min()), float(deltas[:, 0].max())],
        "axis_current_delta_min_max_pA": [
            float(deltas[:, 1].min()),
            float(deltas[:, 1].max()),
        ],
        "groups_with_lower_D95": int((deltas[:, 0] < 0).sum()),
        "groups_with_higher_axis_current": int((deltas[:, 1] > 0).sum()),
    }


def read_case(path, reference):
    run = load(path)
    focus_path = path / "through-focus.npz"
    if not focus_path.is_file():
        profile_path = CHECKPOINT / (
            "baseline-profile.json" if reference else "profile.json"
        )
        expected = json.loads(profile_path.read_text(encoding="utf-8"))
        if run["profile"]["solution_sha256"] != expected["solution_sha256"]:
            raise ValueError(
                f"Archived through-focus observations belong to a different magnetic field: {path}"
            )
        focus_path = CHECKPOINT / (
            "baseline-through-focus.npz" if reference else "through-focus.npz"
        )
    report, curve = metrology(run, focus_path)
    weights, xy = run["w_sample"], run["xy"]
    centroid = np.average(xy, axis=0, weights=weights)
    radius = np.linalg.norm(xy - centroid, axis=1)
    order = np.argsort(radius)
    d95 = (
        2
        * radius[
            order[np.searchsorted(np.cumsum(weights[order]), 0.95 * weights.sum())]
        ]
    )
    np.testing.assert_allclose(d95, run["result"]["D95_um"], rtol=1e-12)
    surface_row = np.argmin(abs(curve[:, 0] - report["specimen_z_mm"]))
    np.testing.assert_allclose(
        curve[surface_row, 0], report["specimen_z_mm"], rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(curve[surface_row, 1], d95, rtol=1e-9)
    assert run["profile"]["c2_aperture_um"] == 100 and np.isclose(run["z"][0], 0.0)
    assert np.isclose(run["z"][-1], run["profile"]["specimen_z_mm"], rtol=0, atol=1e-8)
    tracking = run["profile"]["assets"]["tracking"]
    input_file(run["profile"], "tracking")
    disk = {}
    for label, centre in [("axis", np.zeros(2)), ("centroid", centroid)]:
        selected = weights[np.linalg.norm(xy - centre, axis=1) <= 0.05]
        disk[label] = {
            "pA": selected.sum() * 1e6,
            "packets": selected.size,
            "effective_packets": (
                selected.sum() ** 2 / (selected @ selected) if selected.size else 0.0
            ),
        }
    return (
        run,
        dict(
            case=path.name,
            raw_trace_sha256=sha256(path / "trajectories.npz"),
            tracking_sha256=tracking["sha256"],
            field_sha256=report["field_sha256"],
            emitted=len(run["w"]),
            gun_survivors=len(run["gun"]),
            passing_C2=len(run["c2"]),
            specimen_arrivals=len(weights),
            effective_arrivals=weights.sum() ** 2 / (weights @ weights),
            D95_um=d95,
            RMS_um=np.sqrt(np.average(radius**2, weights=weights)),
            centroid_um=centroid.tolist(),
            disk_radius_nm=50,
            disk_currents=disk,
            current_nA=weights.sum() * 1000,
            specimen_z_mm=report["specimen_z_mm"],
            waists_mm={
                axis: {
                    region: [float(z) for z in roots if lo < z < hi]
                    for region, (lo, hi) in {
                        "TL22_TL21": (1179.1, 1237.2),
                        "postscan": (1420.0, 1512.5),
                    }.items()
                }
                for axis, roots in report[
                    "all_downstream_xyr_stationary_planes_mm"
                ].items()
            },
            through_focus=report["terminal_focus"],
            objective_A=run["profile"]["excitation_A"][
                "objective_lens/physical-pack-4"
            ],
            objective_in_common_supply=(
                run["profile"]["excitation_A"]["objective_lens/physical-pack-4"]
                == run["profile"]["excitation_A"]["objective_lens/physical-pack-5"]
            ),
        ),
        curve,
    )


def draw_column(run, metric, curves, filename):
    fig = Figure(figsize=(12.3, 7.7), layout="constrained")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.95])
    z, selected = run["z"], run["sample_rows"]
    paths = run["paths"]
    at_c3 = paths[selected, np.argmin(np.abs(z - 1000))]
    r = np.linalg.norm(at_c3, axis=1)
    ids = []
    for azimuth in np.array_split(np.argsort(np.arctan2(at_c3[:, 1], at_c3[:, 0])), 8):
        group = azimuth[np.argsort(r[azimuth])]
        ids.extend(group[np.linspace(0, len(group) - 1, min(6, len(group)), dtype=int)])
    rays = paths[selected[ids]]
    losses = paths[~np.isin(run["ids"], run["sample"])]
    roots = [v for group in metric["waists_mm"]["r"].values() for v in group]
    palette = [BLUE, "#84639a", "#408769", RED, "#bf8c42", "#646e9d"]
    for dim in range(2):
        ax = fig.add_subplot(gs[dim, :])
        for p in losses:
            ax.plot(z, p[:, dim], color="#a9b4b9", alpha=0.33, lw=0.45)
        for i, p in enumerate(rays):
            ax.plot(z, p[:, dim], color=palette[i % len(palette)], alpha=0.77, lw=0.68)
        h = np.nanmax(np.abs(rays[:, :, dim]))
        ax.set(
            ylim=(-1.06 * h, 1.32 * h),
            xlim=(z[0], z[-1]),
            ylabel=("x" if dim == 0 else "y") + " (mm)",
        )
        for lo, hi, name in REGIONS:
            ax.axvspan(lo, hi, color="#707d88", alpha=0.10, lw=0, zorder=-2)
            ax.text(
                (lo + hi) / 2,
                0.96 if name != "Obj." else 0.82,
                name,
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                ha="center",
                va="top",
                color="#263546",
            )
        ax.axvline(765, color="#865a8a", ls="--", lw=0.8)
        ax.text(
            765,
            0.88,
            "C2 aperture",
            ha="center",
            va="top",
            fontsize=8,
            color="#865a8a",
            transform=ax.get_xaxis_transform(),
        )
        for root in roots:
            ax.axvline(root, color=BLUE, lw=0.8, ls=":")
        ax.grid(axis="y", alpha=0.14)
        tick = [*range(0, 1500, 200), z[-1]]
        ax.set_xticks(tick, [*map(str, range(0, 1500, 200)), f"{z[-1]:.1f}"])
        if dim == 0:
            ax.set_title(
                f"Emitting tip to specimen upper surface    |    {len(rays)} sampled arrivals and {len(losses)} sampled intercepted paths",
                loc="left",
                fontsize=10,
            )
        else:
            ax.set_xlabel("Distance from the tip (mm)")
    inset = fig.add_subplot(gs[2, 0])
    sel = z >= 1165
    for i, p in enumerate(rays):
        inset.plot(
            z[sel],
            p[sel, 0] * 1000,
            color=palette[i % len(palette)],
            lw=0.67,
            alpha=0.76,
        )
    for root in roots:
        inset.axvline(root, color=BLUE, lw=0.9, ls=":")
        inset.text(
            root,
            0.04,
            f"{root:.3f} mm",
            ha="center",
            fontsize=8,
            transform=inset.get_xaxis_transform(),
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.0),
            color=BLUE,
        )
    inset.set(
        xlim=(z[sel][0], z[-1]), xlabel="Distance from the tip (mm)", ylabel="x (µm)"
    )
    inset.set_title("Corrector and post-scan waists", loc="left", fontsize=11)
    focus = fig.add_subplot(gs[2, 1])
    for label, col, curve in curves:
        delta = (curve[:, 0] - z[-1]) * 1000
        zone = abs(delta) <= 105
        focus.plot(delta[zone], curve[zone, 1], color=col, lw=1.7, label=label)
        i = np.argmin(curve[zone, 1])
        focus.scatter(delta[zone][i], curve[zone, 1][i], s=16, color=col)
    focus.axvline(0, color="#4c5660", ls=":", lw=0.8)
    focus.legend(frameon=False, fontsize=8)
    focus.set(
        xlim=(-105, 105), xlabel="Axial offset from specimen (µm)", ylabel="D95 (µm)"
    )
    focus.set_title("Full beam through focus", loc="left", fontsize=11)
    fig.suptitle(
        f"Nanoprobe Normal   |   {curves[-1][0]} full-column trajectories",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(filename, dpi=170)


def draw_probe(runs, metrics, labels, filename):
    n = len(runs)
    fig = Figure(figsize=(3.95 * n + 1.15, 7.45), layout="constrained")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1] * n + [0.055])
    lim = max(0.15, np.ceil(max(np.abs(v["xy"]).max() for v in runs) * 10) / 10)
    for row, (half, bins, mult, unit) in enumerate(
        [(lim, 63, 1.0, "µm"), (0.225, 35, 1000.0, "nm")]
    ):
        edges = np.linspace(-half, half, bins + 1)
        densities = []
        for run in runs:
            hist = np.histogram2d(
                *run["xy"].T, bins=(edges, edges), weights=run["w_sample"] * 1e6
            )[0].T
            if row == 0:
                np.testing.assert_allclose(hist.sum(), run["w_sample"].sum() * 1e6)
            densities.append(hist / np.diff(edges)[0] ** 2)
        peak = max(v.max() for v in densities)
        norm = AsinhNorm(linear_width=peak * 0.002, vmin=0, vmax=peak)
        for col, (density, metric, label) in enumerate(zip(densities, metrics, labels)):
            ax = fig.add_subplot(gs[row, col])
            layer = ax.imshow(
                density,
                origin="lower",
                extent=np.array([-half, half, -half, half]) * mult,
                cmap="magma",
                interpolation="nearest",
                norm=norm,
            )
            centroid = np.array(metric["centroid_um"]) * mult
            if row == 0:
                ax.add_patch(
                    Circle(
                        centroid,
                        mult * metric["D95_um"] / 2,
                        fc="none",
                        ec="#68d6d4",
                        ls="--",
                        lw=1.15,
                    )
                )
                ttl = f"{label}   ·   D95 {metric['D95_um']:.4f} µm"
            else:
                ax.add_patch(Circle((0, 0), 50, fc="none", ec="#70e2df", lw=1.45))
                ax.add_patch(
                    Circle(centroid, 50, fc="none", ec="white", ls="--", lw=1.1)
                )
                ttl = f"{label}   ·   {metric['disk_currents']['axis']['pA']:.2f} pA in axis disk"
            ax.set(xlabel=f"x ({unit})", ylabel=f"y ({unit})")
            ax.set_title(ttl, fontsize=10, loc="left")
        cb = fig.colorbar(layer, cax=fig.add_subplot(gs[row, n]))
        cb.set_label("Current density (pA/µm²)", fontsize=9)
        ticks = peak * np.array([0, 0.01, 0.1, 1.0])
        cb.set_ticks(ticks, labels=[f"{v:.2g}" for v in ticks])
    fig.suptitle(
        f"Specimen upper surface    |    z = {metrics[0]['specimen_z_mm']:.4f} mm",
        fontsize=13,
        weight="bold",
    )
    fig.supxlabel(
        "Top: dashed 95%-current circle.   Detail: fixed-axis 50 nm disk (solid), centroid disk (dashed).",
        fontsize=8,
    )
    fig.savefig(filename, dpi=175)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Directory produced by replaying baseline-profile.json with temsim.joint_fem.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=CHECKPOINT,
        help="Final trajectory archive or a replay of profile.json.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory for the two PNG figures and paired measurements.",
    )
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    old, before, before_curve = read_case(args.baseline, reference=True)
    new, after, after_curve = read_case(args.candidate, reference=False)
    np.testing.assert_array_equal(old["w"], new["w"])
    assert before["tracking_sha256"] == after["tracking_sha256"]
    controls_before, controls_after = (v["profile"]["excitation_A"] for v in (old, new))
    active = controls_before.keys() | controls_after.keys()
    changes = {
        key: [controls_before.get(key, 0.0), controls_after.get(key, 0.0)]
        for key in sorted(active)
        if controls_before.get(key, 0.0) != controls_after.get(key, 0.0)
    }
    allowed = {
        "probe_hp1_hexapole",
        "probe_hp2_hexapole",
        "probe_hpol_hexapole_x",
        "probe_hpol_hexapole_y",
        "probe_hpc_hexapole_x",
        "probe_hpc_hexapole_y",
        "objective_lens/physical-pack-4",
        "objective_lens/physical-pack-5",
    }
    assert changes.keys() <= allowed and after["objective_in_common_supply"]
    report = dict(
        baseline=before,
        adjusted=after,
        identical_tracking_inputs=True,
        identical_survivor_ids={
            key: bool(np.array_equal(old[key], new[key]))
            for key in ("gun", "c2", "sample")
        },
        changed_currents_A=changes,
        D95_reduction_percent=100 * (1 - after["D95_um"] / before["D95_um"]),
        axis_disk_delta_pA=after["disk_currents"]["axis"]["pA"]
        - before["disk_currents"]["axis"]["pA"],
        matched_source_block_sensitivity=block_sensitivity(old, new),
    )
    (args.out / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    draw_column(
        new,
        after,
        [("Baseline", BLUE, before_curve), ("Adjusted", RED, after_curve)],
        args.out / "full-column.png",
    )
    draw_probe(
        [old, new],
        [before, after],
        ["Baseline", "Adjusted"],
        args.out / "sample-probe.png",
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in after.items()
                if k in ("D95_um", "waists_mm", "disk_currents", "current_nA")
            }
        )
    )


if __name__ == "__main__":
    main()
