"""Archive compact independently measured widths, focus coordinates and Si imaging evidence."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def observed_best(rows, metric, target, *, maximum=False):
    values = np.array([row[metric] for row in rows])
    best = values.max() if maximum else values.min()
    z = np.array([row["z_mm"] for row in rows])
    hit = np.flatnonzero(values == best)
    at_surface = values[np.flatnonzero(z == target).item()]
    return dict(
        value=float(best),
        planes_mm=z[hit].tolist(),
        offsets_nm=((z[hit] - target) * 1e6).tolist(),
        first_or_last_plane=bool(hit[0] == 0 or hit[-1] == len(rows) - 1),
        target_fractional_difference=float(abs(at_surface - best) / best) if best else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("scan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()
    metrics = json.loads((args.trial / "metrics.json").read_text())
    scan = json.loads((args.scan / "report.json").read_text())
    surface = metrics["specimen_upper_surface_mm"]
    if metrics["input_digest"] != scan["input_digest"] or surface != scan["entrance_mm"]:
        raise ValueError("Optical measurement and scan have different source inputs")
    original = metrics["population"]["positive_current_count"]
    if original != scan["original_positive"] or original < 1000:
        raise ValueError("Invalid original particle population")
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.trial / "surface.npz") as phase:
        live = phase["alive"] & (phase["weight"] > 0)
        if (
            str(phase["input_digest"]) != metrics["input_digest"]
            or live.sum() != scan["surviving"]
            or float(phase["z_mm"]) != surface
        ):
            raise ValueError("Entrance phase and Si scan differ")
        weights = phase["weight"][live]
        x, y = phase["x_m"][live], phase["y_m"][live]
        radii = np.hypot(x - np.average(x, weights=weights), y - np.average(y, weights=weights))
        from temsim.physics.beam_statistics import _weighted_quantile

        median = _weighted_quantile(radii, weights, 0.5)
    rows = metrics.pop("full_profile")
    selected = [r for r in rows if 1522.5 <= r["z_mm"] <= surface + 2.0]
    metrics["native_Mini_centre_to_surface_plus_2mm"] = dict(
        first_z_mm=selected[0]["z_mm"],
        last_z_mm=selected[-1]["z_mm"],
        count=len(selected),
        d95=observed_best(selected, "D95_nm", surface),
        rms=observed_best(selected, "rms_radius_nm", surface),
        axis_core_50pm=observed_best(selected, "core_50pm_current_a", surface, maximum=True),
    )
    metrics["surface"]["D50_nm"] = 2e9 * median
    files = {
        "entrance.npz": args.trial / "surface.npz",
        "si110.json": args.scan / "report.json",
        "si110.npz": args.scan / "scan.npz",
        "native-profile.csv": args.scan / f"prefix-{original}" / "native-profile.csv",
    }
    if args.recipe:
        files["currents.json"] = args.recipe
    metrics["sha256"] = {name: digest(path) for name, path in files.items()}
    keys = (
        "z_mm",
        "surviving",
        "current_a",
        "rms_radius_nm",
        "D95_nm",
        "alpha95_mrad",
        "core_50pm_current_a",
        "radial_covariance_m_rad",
    )
    for key, path in files.items():
        shutil.copyfile(path, args.output / key)
    np.savez_compressed(
        args.output / "through-focus.npz", **{k: [r.get(k, np.nan) for r in rows] for k in keys}
    )
    metrics["sha256"]["through-focus.npz"] = digest(args.output / "through-focus.npz")
    (args.output / "optics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
