"""Read-only exact gun-exit replay against the original fine-column archive."""
import importlib.util
import json
from pathlib import Path
import numpy as np

SPEC = importlib.util.spec_from_file_location("gun_tail_diagnostic", Path(__file__).with_name("run_gun_tail_diagnostic.py"))
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)
from temsim.instrument_snapshot import InstrumentSnapshot, capture_instrument_snapshot
from temsim.physics.gun_field_environment import instrument_gun_field_context
from temsim.physics.closed_gun_field import closed_field_request
from temsim.physics.particle_sections import gun_dependency_signature
from validate_joint_probe import _restore_exact
from solve_joint_probe import snapshot_changes


def main():
    root = diagnostic.ROOT
    baseline = root / "outputs/joint-probe-100um-30mrad-20261028/seed769-native-d95-waist-h005"
    observed = root / "outputs/probe-tail-attribution-20261028/gun-step769"
    before_report = json.loads((baseline / "report.json").read_text(encoding="utf-8"))
    after_report = json.loads((observed / "report.json").read_text(encoding="utf-8"))
    if before_report["execution_status"] != "COMPLETE" or after_report["status"] != "COMPLETE":
        raise ValueError("Exact replay requires two complete executed records")
    old_snapshot = InstrumentSnapshot.from_dict(json.loads((baseline / "inputs.json").read_text(encoding="utf-8")))
    new_snapshot = InstrumentSnapshot.from_dict(json.loads((observed / "step-1/inputs.json").read_text(encoding="utf-8")))
    with np.load(baseline / "particles.npz", allow_pickle=False) as old, np.load(observed / "step-1/particles.npz", allow_pickle=False) as new:
        pairs = [("gun_exit_" + key, "exit_" + key) for key in ("x_m", "y_m", "tx_rad", "ty_rad", "alive", "flight_time_s")]
        pairs += [("source_id", "exit_ray_id"), ("weights", "exit_weight"),
                  ("source_energy_offset_ev", "exit_energy_offset_ev"),
                  ("gun_blocked_key", "blocked_key"), ("gun_blocked_z_mm", "blocked_z_mm")]
        pairs += [("emission_" + key, "emission_" + key) for key in ("ray_id", "position_m", "direction", "normal")]
        rows = {}
        for left, right in pairs:
            a, b = old[left], new[right]
            exact = a.shape == b.shape and a.dtype == b.dtype and (np.array_equal(a, b, equal_nan=True)
                if a.dtype.kind in "fc" else np.array_equal(a, b))
            finite = np.isfinite(a) & np.isfinite(b) if a.dtype.kind in "fc" and a.shape == b.shape else None
            rows[left] = dict(exact=bool(exact), baseline_sha256=diagnostic.array_hash(a), observed_sha256=diagnostic.array_hash(b),
                maximum_absolute_difference=float(np.max(np.abs(a[finite]-b[finite]), initial=0.)) if finite is not None else None)
        state = _restore_exact(new_snapshot)
        actual_k = float(state.electron_gun.nominal_exit_energy_ev) + new["exit_energy_offset_ev"]
        expected_k = old["gun_exit_kinetic_energy_ev"]
        rows["gun_exit_kinetic_energy_ev"] = dict(exact=bool(np.array_equal(actual_k, expected_k, equal_nan=True)),
            baseline_sha256=diagnostic.array_hash(expected_k), observed_sha256=diagnostic.array_hash(actual_k))
    identities = []
    for snapshot in (old_snapshot, new_snapshot):
        state = _restore_exact(snapshot)
        signature = gun_dependency_signature(state)
        with instrument_gun_field_context(state) as magnetic:
            request = closed_field_request(state.electron_gun)
            identities.append(dict(gun_dependency_signature=signature,
                electric_request_sha256=diagnostic.digest(request),
                magnetic_physical_identity=magnetic.physical_identity,
                magnetic_numerical_identity=magnetic.numerical_identity))
        if capture_instrument_snapshot(state).digest != snapshot.digest:
            raise ValueError("Read-only field capture changed original inputs")
    field = after_report["runs"][0]["field_identity"]
    field_matches_receipt = identities[1]["electric_request_sha256"] == field["request_sha256"] and (
        identities[1]["magnetic_physical_identity"] == field["magnetic"]["physical_identity"] and
        identities[1]["magnetic_numerical_identity"] == field["magnetic"]["numerical_identity"])
    old_signature_matches = identities[0]["gun_dependency_signature"] == before_report["receipt"]["gun_dependency_signature"]
    result = dict(schema="temsim-exact-gun-observer-replay-v1", status="PASS" if all(row["exact"] for row in rows.values())
        and identities[0] == identities[1] and field_matches_receipt and old_signature_matches else "FAIL",
        qualification="NOT_QUALIFIED", scope="Observer non-perturbation at unchanged gun settings; no new propagation",
        baseline=str(baseline), observed=str(observed / "step-1"), arrays=rows, identities=identities,
        identity_matches_saved_observation=field_matches_receipt, original_signature_matches_receipt=old_signature_matches,
        complete_graph_changes=snapshot_changes(old_snapshot.to_dict()["graph"], new_snapshot.to_dict()["graph"]),
        source_files_sha256={str(path): diagnostic.file_hash(path) for path in
            (baseline / "particles.npz", baseline / "inputs.json", baseline / "report.json",
             observed / "step-1/particles.npz", observed / "step-1/inputs.json", observed / "report.json")})
    destination = observed / "exact-observer-replay.json"
    if destination.exists():
        raise FileExistsError(destination)
    diagnostic.write_json(destination, result)
    print(result["status"], result["complete_graph_changes"], destination)
    print({key: value for key, value in rows.items() if not value["exact"]})
    return result


if __name__ == "__main__":
    main()
