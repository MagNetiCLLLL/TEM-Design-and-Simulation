"""Check actual landing weights, finite sampling and magnetic interpolation along rays."""

import argparse
import hashlib
import json

import numpy as np
from scipy.interpolate import CubicSpline

from column_trace import values, wq
from gun_images import GUN
from volume import BASE, CACHE, HERE, WORK, Vacuum, basis, sha


def compare_field(path, table, z, xy):
    # Use sites on tracked trajectories in each part of the column and retain
    # the largest sampled radii in addition to stratified ordinary locations.
    rng = np.random.default_rng(29416)
    sites = []
    for lo, hi in [
        (450, 550),
        (550, 765),
        (765, 1070),
        (1070, 1237.2),
        (1237.2, 1380),
        (1380, 1512.5),
        (1512.5, 1599.21),
    ]:
        iz, ir = np.where(np.isfinite(xy[:, :, 0]) & ((z >= lo) & (z < hi))[:, None])
        r = np.linalg.norm(xy[iz, ir], axis=1)
        chosen = np.unique(np.r_[rng.choice(len(iz), 100, replace=False), np.argsort(r)[-20:]])
        sites.append(np.column_stack((xy[iz[chosen], ir[chosen]], z[iz[chosen]] * 0.001)))
    points = np.vstack(sites)
    direct = Vacuum(path).symmetric(points)
    with np.load(table) as data:
        radii = data["scales"]
        degree = int(data["degree"])
        predicted = np.zeros_like(direct)
        remaining = np.ones(len(points))
        r = np.linalg.norm(points[:, :2], axis=1)
        for j, (scale, layer) in enumerate(zip(radii, data["layers"], strict=True)):
            blend = np.clip((scale - r) / (0.16 * scale), 0.0, 1.0)
            weight = remaining * (blend**2 * (3 - 2 * blend) if j < len(radii) - 1 else 1.0)
            remaining -= weight
            cc = CubicSpline(data["z_m"], layer, axis=0)(points[:, 2])
            predicted += weight[:, None] * np.einsum(
                "pi,pic->pc", basis(points[:, :2] / scale, degree), cc
            )
        np.testing.assert_allclose(remaining, 0.0, atol=1e-14)
    errors = np.linalg.norm(predicted - direct, axis=1)
    worst = int(np.argmax(errors))
    return {
        "sites": len(points),
        "sampled_radius_max_mm": 1e3 * max(r),
        "p95_error_T": np.quantile(errors, 0.95),
        "max_error_T": errors[worst],
        "worst_z_mm": 1e3 * points[worst, 2],
        "worst_radius_mm": 1e3 * r[worst],
    }


def audit(name):
    folder = HERE / name
    rec = json.loads((folder / "result.json").read_text())
    with np.load(CACHE / rec["gun_packet"]) as gun, np.load(folder / "trajectories.npz") as rays:
        n = len(gun["ids"])
        ids = rays["ids"]
        w = rays["weights_uA"]
        survive = rays["loss"] == -1
        top = rays["surface_phase"]
        np.testing.assert_array_equal(gun["ids"], np.arange(n))
        assert n >= 1000 and np.all(gun["weight_uA"] > 0)
        np.testing.assert_allclose(gun["weight_uA"].sum(), 15, rtol=1e-12)
        np.testing.assert_array_equal(ids, gun["ids"][gun["label"] == 0])
        np.testing.assert_allclose(w, gun["weight_uA"][ids], rtol=1e-15)
        assert np.all(rays["loss"] >= -1) and np.isfinite(top[survive]).all()
        ledger = rec["current_ledger_A"]
        np.testing.assert_allclose(w[survive].sum() * 1e-6, ledger["specimen_A"], rtol=1e-12)
        np.testing.assert_allclose(
            ledger["specimen_A"] + sum(ledger["intercepted_A"].values()), ledger["gun_exit_A"], rtol=1e-12
        )
        fractions = {}
        for count in (n // 4, n // 2, n):
            use = survive & (ids < count)
            fractions[count] = values(top[use], w[use] * (n / count))
        positions = top[survive, :2]
        slopes = top[survive, 2:4]
        arrivals = w[survive]
        xy = positions - np.average(positions, weights=arrivals, axis=0)
        chief_slope = np.average(slopes, weights=arrivals, axis=0)
        outward = np.sum(xy * (slopes - chief_slope), axis=1) > 0
        outward_fraction = arrivals[outward].sum() / arrivals.sum()
        chief = np.r_[chief_slope, 1.0]
        directions = np.column_stack((slopes, np.ones(len(slopes))))
        angles = 1e3 * np.arctan2(np.linalg.norm(np.cross(directions, chief), axis=1), directions @ chief)
        at_foil = (
            np.maximum(
                np.linalg.norm(rays["aperture_phase"][survive, :2], axis=1),
                np.linalg.norm(rays["aperture_rear_phase"][survive, :2], axis=1),
            )
            * 1e6
        )
        assert np.all(at_foil <= 50.0000001)
        rings = {}
        for lower in (45.0, 47.5, 49.0, 49.5):
            ring = at_foil >= lower
            rings[str(lower)] = {
                "survivors": int(ring.sum()),
                "sample_nA": 1e3 * arrivals[ring].sum(),
                "median_angle_mrad": wq(angles[ring], arrivals[ring], 0.5) if ring.any() else None,
            }
        z = rays["z_mm"]
        radius_um = 1e6 * np.sqrt(rays["stats_m"][:, :2].sum(axis=1))
        stations = {str(s): np.interp(s, z, radius_um) for s in [1144.5, 1420, 1512.5, 1599.1999975]}
        field = compare_field(WORK / (rec["field"] + ".npz"), CACHE / rec["field_table"], z, rays["xy_m"])
    result = {
        "originals": n,
        "gun_exits": len(ids),
        "physical_current_balanced": True,
        "prefixes": fractions,
        "live_RMS_radius_um_at_z_mm": stations,
        "field_on_paths": field,
        "outward_specimen_current_fraction": outward_fraction,
        "rim_lower_radius_um": rings,
    }
    (folder / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, n, field, flush=True)


def compare(cases):
    windows = [(450, 650), (750, 830), (1179.1, 1237.2), (1420, 1512.5)]
    adjustable = {"condenser_lens_1/physical-pack-0", "condenser_lens_2/physical-pack-1"}
    comparison = {"waist_windows_mm": windows, "operating_points": {}}
    reference = None
    stops = json.loads((GUN / "anode-mesh15-50V.json").read_text())["stops"]
    gun_labels = ["gun_exit", "reverse"] + [stop["key"] for stop in stops]
    for name in cases:
        folder = HERE / name
        rec = json.loads((folder / "result.json").read_text())
        checks = json.loads((folder / "validation.json").read_text())
        with np.load(CACHE / rec["gun_packet"]) as packet:
            event = packet["label"]
            assert event.min() >= 0 and event.max() < len(gun_labels)
            current = np.bincount(event, weights=packet["weight_uA"], minlength=len(gun_labels))
            np.testing.assert_allclose(current.sum(), 15.0, rtol=1e-12)
            gun_ledger = {}
            for label, value in zip(gun_labels, current, strict=True):
                gun_ledger[label] = gun_ledger.get(label, 0.0) + value
            np.testing.assert_allclose(
                gun_ledger["gun_exit"] * 1e-6, rec["current_ledger_A"]["gun_exit_A"], rtol=1e-12
            )
        field_path = WORK / (rec["field"] + ".npz")
        field_hash = sha(field_path)
        with np.load(field_path) as solution:
            receipt = json.loads(str(solution["report"]))
            recipe_hash = hashlib.sha256(str(solution["recipe"]).encode()).hexdigest()
        with np.load(CACHE / rec["field_table"]) as table:
            for layer in json.loads(str(table["layer_receipts"])):
                assert layer["solution_sha256"] == field_hash and layer["bias_A"] == receipt["excitation"]
        if field_path == BASE:
            with np.load(GUN / "gun-volume-field.npz") as gun:
                assert str(gun["volume_sha256"]) == field_hash
        else:
            with np.load(CACHE / (field_path.stem + "-gun.npz")) as gun:
                assert json.loads(str(gun["receipt"]))["solution_sha256"] == field_hash
        fixed = {k: v for k, v in receipt["excitation"].items() if k not in adjustable}
        controls = {
            "fixed_currents_A": fixed,
            "mesh_id": receipt["mesh_id"],
            "circuit_sha256": receipt["circuit_sha256"],
            "geometry_and_sources_sha256": recipe_hash,
        }
        if reference is None:
            reference = controls
        assert controls == reference and receipt["residual"] <= 2e-8
        s = rec["surface"]
        roots = rec["waists"]["live"]["radial_mm"]
        counts = [sum(low < z < high for z in roots) for low, high in windows]
        extra = [z for z in roots if all(not (lo < z < hi) for lo, hi in windows)]
        comparison["operating_points"][name] = {
            "C1_A": rec["c1_A"],
            "C2_A": rec["c2_A"],
            "originals": checks["originals"],
            "specimen_primaries": s["count"],
            "gun_first_hit_uA": gun_ledger,
            "column_first_hit_A": rec["current_ledger_A"],
            "D95_um": s["D95_um"],
            "specimen_nA": s["sample_nA"],
            "alpha95_mrad": s["alpha95_mrad"],
            "aperture_rim": rec["rim_outer_5pct"],
            "positive_radial_current_fraction": checks["outward_specimen_current_fraction"],
            "live_waists_mm": roots,
            "waist_counts": counts,
            "outside_windows_mm": extra,
            "intermediate_windows_pass": counts == [1, 1, 1, 1] and not extra,
            "upstream_min_D95_z_mm_and_value_um": rec["focus_upstream_min_mm_D95um"],
            "sample_z_mm": 1599.1999975,
            "sample_RMS_slope_mrad": rec["surface_rms_radius_slope_mrad"],
            "field_receipt_residual": receipt["residual"],
            "solution": rec["field"],
            "field_sha256": field_hash,
        }
    comparison.update(reference)
    with (
        np.load(HERE / "angle-control" / "trajectories.npz") as coarse,
        np.load(HERE / "angle-step" / "trajectories.npz") as fine,
    ):
        np.testing.assert_array_equal(
            coarse["ids"][coarse["loss"] == -1], fine["ids"][fine["loss"] == -1]
        )
    (HERE / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("cases", nargs="+")
    cli.add_argument("--compare", action="store_true", help="Compare the existing per-case checks")
    args = cli.parse_args()
    if args.compare:
        compare(args.cases)
    else:
        for case in args.cases:
            audit(case)
