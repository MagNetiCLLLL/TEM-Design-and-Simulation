"""Summarize completed full-source Microprobe observations without retracing."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def native_tails(arrays, row, current):
    columns = np.flatnonzero(arrays["alive"][row] & (arrays["weights"] > 0.))
    w = arrays["weights"][columns]
    phase = np.column_stack([arrays[key][row, columns] for key in ("x_m", "y_m", "tx_rad", "ty_rad")])
    center = np.average(phase[:, :2], weights=w, axis=0)
    radius = np.linalg.norm(phase[:, :2] - center, axis=1)
    direction = np.column_stack([phase[:, 2:], np.ones(len(columns))])
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    chief = np.average(direction, weights=w, axis=0)
    chief /= np.linalg.norm(chief)
    angle = np.arctan2(np.linalg.norm(np.cross(direction, chief), axis=1), np.clip(direction @ chief, -1., 1.))
    records = {}
    for label, values in (("largest_centroid_radii", radius), ("largest_mechanical_angles", angle)):
        records[label] = [dict(source_id=int(arrays["source_id"][columns[j]]),
            original_source_current_a=float(current * w[j]), centroid_radius_um=float(radius[j] * 1e6),
            mechanical_angle_mrad=float(angle[j] * 1e3),
            phase_xy_um_slopes_mrad=(phase[j] * [1e6, 1e6, 1e3, 1e3]).tolist())
            for j in np.argsort(values)[-5:][::-1]]
    records.update(maximum_centroid_radius_um=float(radius.max() * 1e6),
        maximum_mechanical_angle_mrad=float(angle.max() * 1e3),
        finite_positive_phases=bool(np.isfinite(phase).all()),
        no_particles_deleted_or_reweighted=True)
    return records


def audit(run_dir):
    output = dict(scope="Postprocessing of completed native classical observations; no new propagation",
        source_count_note="Original positive source count is separate from surviving and effective counts",
        size_note="RMS is a weighted second-moment radius; D95 is twice the weighted 95% containment radius",
        angle_note="Mechanical direction cone about the current-weighted chief, not canonical slope",
        topology_note="A region representative is a native mesh point; retain its bracket as resolution",
        trials=[])
    for case in sorted(run_dir.glob("trial-??")):
        if not (case / "report.json").is_file():
            continue
        receipt = json.loads((case / "report.json").read_text(encoding="utf-8"))
        with np.load(case / "observations.npz", allow_pickle=False) as arrays:
            surface = receipt["surface"]
            row = int(np.searchsorted(arrays["z_mm"], surface["z_mm"]))
            if arrays["z_mm"][row] != surface["z_mm"]:
                raise ValueError("Actual sample surface is absent")
            tails = native_tails(arrays, row, receipt["execution"]["source_current_a"])
            tails["maximum_radius_to_r95_ratio"] = tails["maximum_centroid_radius_um"] / (.5 * surface["d95_um"])
            tails["maximum_angle_to_alpha95_ratio"] = tails["maximum_mechanical_angle_mrad"] / surface["alpha95_mrad"]
        topology = {}
        for label, populations in receipt.get("topology", {}).items():
            topology[label] = {kind: dict(status=entry["status"], unresolved=entry["unresolved"],
                regions=[{name: region[name] for name in
                    ("representative_z_mm", "bracket_mm", "current_phase_space") if name in region}
                    for region in entry["regions"]]) for kind, entry in populations.items()}
        diameter = [entry["d95_um"] for entry in receipt["axial_profile"]]
        output["trials"].append(dict(index=receipt["index"], input_digest=receipt["execution"]["input_digest"],
            original_source_count=receipt["execution"]["original_source_count"],
            emitted_positive_current_points=receipt["execution"]["emitted_positive_current_points"],
            source_identity_sha256=receipt["execution"]["source_identity_sha256"],
            original_source_current_a=receipt["execution"]["source_current_a"],
            surface=surface, controls=receipt["controls"], per_stage_losses=receipt["per_stage_losses"],
            axial_d95_range_um=[min(diameter), max(diameter)],
            axial_d95_span_relative_to_surface=float((max(diameter)-min(diameter))/surface["d95_um"]),
            surface_tails=tails, topology=topology,
            observations_sha256=sha256((case/"observations.npz").read_bytes()).hexdigest()))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = audit(args.run_dir)
    output = args.output or args.run_dir / "family-audit.json"
    output.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    for trial in data["trials"]:
        surface, tails = trial["surface"], trial["surface_tails"]
        print(trial["index"], "D95_um", surface["d95_um"], "alpha95_mrad", surface["alpha95_mrad"],
            "max_radius/r95", tails["maximum_radius_to_r95_ratio"],
            "max_angle/alpha95", tails["maximum_angle_to_alpha95_ratio"])


if __name__ == "__main__":
    main()
