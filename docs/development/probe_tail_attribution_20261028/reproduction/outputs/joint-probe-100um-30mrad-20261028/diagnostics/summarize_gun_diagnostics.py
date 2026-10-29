"""Offline original-current phase/shape differences from exact gun observations."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("gun_tail_diagnostic", Path(__file__).with_name("run_gun_tail_diagnostic.py"))
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def phase_changes(before, after):
    for key in ("observed_z_mm", "launch_weight", "launch_ray_id"):
        if diagnostic.array_hash(before[key]) != diagnostic.array_hash(after[key]):
            raise ValueError("Different exact planes or original source identities/weights")
    weights, ids = before["launch_weight"], before["launch_ray_id"]
    rows = []
    for index, z in enumerate(before["observed_z_mm"]):
        left = before["observed_alive"][index] & (weights > 0)
        right = after["observed_alive"][index] & (weights > 0)
        common = left & right
        row = dict(z_mm=float(z), before_positive_survivors=int(left.sum()),
            after_positive_survivors=int(right.sum()), common_positive_survivors=int(common.sum()),
            added_active_ids=ids[right & ~left].tolist(), removed_active_ids=ids[left & ~right].tolist(),
            common_original_current_fraction=float(weights[common].sum()/weights.sum()),
            scope="Phase differences use only the explicitly reported common positive-current cohort")
        if not common.any():
            rows.append(dict(row, status="NOT_EVALUATED"))
            continue
        deltas = {}
        for name in diagnostic.PHASE:
            a, b = before[f"observed_{name}"][index], after[f"observed_{name}"][index]
            if not np.isfinite(a[left]).all() or not np.isfinite(b[right]).all():
                raise ValueError("Nonfinite actual positive-current phase cannot be omitted")
            deltas[name] = b[common]-a[common]
        position_nm = np.hypot(deltas["x_m"], deltas["y_m"]) * 1e9
        slope_urad = np.hypot(deltas["tx_rad"], deltas["ty_rad"]) * 1e6
        selected = weights[common]/weights[common].sum()
        id306 = np.flatnonzero(ids[common] == 306)
        row.update(status="MEASURED", maximum_position_delta_nm=float(position_nm.max()),
            weighted_rms_position_delta_nm=float(np.sqrt(np.sum(selected*position_nm**2))),
            maximum_slope_delta_urad=float(slope_urad.max()),
            weighted_rms_slope_delta_urad=float(np.sqrt(np.sum(selected*slope_urad**2))),
            maximum_kinetic_energy_delta_ev=float(np.max(np.abs(deltas["kinetic_energy_ev"]))),
            maximum_flight_time_delta_fs=float(np.max(np.abs(deltas["flight_time_s"]))*1e15),
            id306_position_delta_nm=float(position_nm[id306[0]]) if len(id306) else None,
            id306_slope_delta_urad=float(slope_urad[id306[0]]) if len(id306) else None)
        rows.append(row)
    keys_changed = before["blocked_key"] != after["blocked_key"]
    a, b = before["blocked_z_mm"], after["blocked_z_mm"]
    z_changed = ~(np.equal(a, b) | (np.isnan(a) & np.isnan(b)))
    return dict(exact_observed_planes=rows,
        stop_key_changed_ids=ids[keys_changed].tolist(), stop_z_changed_ids=ids[z_changed].tolist())


def summarize(directory):
    directory = Path(directory).resolve()
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") not in ("COMPLETE", "FAILED", "CANCELLED"):
        raise ValueError("Only a finished execution can be summarized")
    completed = []
    for ordinal, run in enumerate(report["runs"], 1):
        if run["status"] != "COMPLETE":
            continue
        folder = f"step-{ordinal}" if report["mode"] == "step" else f"grid-{run['cells_per_bore']}"
        path = directory / folder / "particles.npz"
        if diagnostic.file_hash(path) != run["raw_file_sha256"]:
            raise ValueError("Raw exact-plane artifact checksum changed")
        with np.load(path, allow_pickle=False) as archive:
            keep = [key for key in archive.files if key.startswith(("observed_", "launch_")) or key in ("blocked_key", "blocked_z_mm")]
            arrays = {key: archive[key].copy() for key in keep}
        completed.append((folder, run, arrays))
    comparisons = []
    for (left_name, left, before), (right_name, right, after) in zip(completed[:-1], completed[1:]):
        if left["source_array_fingerprints"] != right["source_array_fingerprints"]:
            raise ValueError("Original full source changed between observations")
        comparisons.append(dict(before=left_name, after=right_name, **phase_changes(before, after)))
    summary = dict(schema="temsim-offline-gun-phase-comparison-v1", model="classical_particles",
        execution_status=report["status"], qualification="NOT_QUALIFIED", source_count=769,
        report_path=str(report_path), report_sha256=diagnostic.file_hash(report_path), mode=report["mode"],
        comparisons=comparisons,
        plane_shapes=[dict(run=folder, step_mm=row["trace_step_mm"], cells_per_bore=row["cells_per_bore"],
                           measurements=row["measurements"]) for folder, row, _ in completed],
        limitations=["All measured shapes use the full native positive-current population at each plane.",
            "Only phase differences use the explicit intersection cohort; added/lost IDs are retained.",
            "No source, coordinate, energy, stop, or optical-control refit is performed.",
            "A gun-only difference does not identify its final focused specimen consequence.",
            "No physical-model, global-convergence, or probe qualification is inferred."])
    destination = directory / "offline-phase-comparison.json"
    if destination.exists():
        raise FileExistsError(destination)
    diagnostic.write_json(destination, summary)
    print(destination)
    for pair in comparisons:
        print(pair["before"], pair["after"], pair["exact_observed_planes"][-1])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
