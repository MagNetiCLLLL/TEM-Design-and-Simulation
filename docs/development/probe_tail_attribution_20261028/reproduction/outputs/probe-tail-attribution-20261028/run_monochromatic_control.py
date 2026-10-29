"""Declared monochromatic source control with fixed optics and original IDs."""
from pathlib import Path
from hashlib import sha256
import json
import shutil
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "agent.md").is_file())
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from measure_probe_waist import measure
from validate_joint_probe import load_candidate
from temsim.instrument_snapshot import capture_instrument_snapshot

BASE = ROOT / "outputs/joint-probe-100um-30mrad-20261028/seed769-native-d95-waist-h005"
OUT = ROOT / "outputs/probe-tail-attribution-20261028/mono769-h005"


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [v for k in sorted(a.keys() | b.keys()) for v in differences(a.get(k), b.get(k), prefix + "/" + k)]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [v for i, (x, y) in enumerate(zip(a, b)) for v in differences(x, y, prefix + "/" + str(i))]
    return [] if a == b else [dict(path=prefix, before=a, after=b)]


def ahash(value):
    a = np.ascontiguousarray(value)
    return sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def main():
    state, identity = load_candidate(BASE / "inputs.json")
    before = capture_instrument_snapshot(state).to_dict()
    assert state.step_mm == .05 and state.electron_gun.emitter.ray_count == 769
    emitted_before = state.electron_gun.emitter.emit()
    assert state.electron_gun.emitter.energy_spread_fwhm_ev == .3
    state.electron_gun.emitter.energy_spread_fwhm_ev = 0.
    emitted_after = state.electron_gun.emitter.emit()
    fields = ("x_m", "y_m", "tx_rad", "ty_rad", "weight", "ray_id")
    assert all(np.array_equal(getattr(emitted_before,k), getattr(emitted_after,k)) for k in fields)
    assert np.array_equal(emitted_after.energy_offset_ev, np.zeros(769))
    changed = capture_instrument_snapshot(state).to_dict()
    delta = differences(before, changed)
    graph = [row for row in delta if row["path"].startswith("/graph/")]
    assert len(graph) == 1 and graph[0]["path"].endswith("/energy_spread_fwhm_ev"), delta
    assert {row["path"].split("/")[1] for row in delta} <= {"graph", "digest"}, delta
    baseline = json.loads((BASE / "report.json").read_text())
    assert baseline["execution_status"] == "COMPLETE"
    # Preserve the baseline's actual grid; add fine points near the sample too.
    extra = sorted(set(baseline["parameters"]["extra_z_offset_nm"] + [round(i*.1, 8) for i in range(-30,31)]))
    observation_source = (ROOT / "scripts/measure_probe_waist.py").read_bytes()
    receipt = dict(type="physical source-energy control", baseline=str(BASE), explicit_input_delta=delta,
        source_spatial_identity_equal=True, original_source_hashes={k:ahash(getattr(emitted_before,k)) for k in fields},
        mean_launch_kinetic_energy_ev=state.electron_gun.emitter.emission_energy_ev,
        no_retuning=True, no_size_acceptance_cap=True,
        interpretation="Whole gun+column energy effect; does not isolate downstream magnetic chromaticity")
    report = measure(state, OUT, extra_z_offset_nm=extra, threads=8, time_budget_s=1800,
        candidate={**identity, "control_experiment": receipt})
    (OUT / "measure_probe_waist.py").write_bytes(observation_source)
    shutil.copyfile(__file__, OUT / "run_monochromatic_control.py")
    (OUT / "control-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if report["execution_status"] != "COMPLETE":
        raise RuntimeError(report["errors"])
    with np.load(BASE / "particles.npz", allow_pickle=False) as old, np.load(OUT / "particles.npz", allow_pickle=False) as new:
        same = {k:np.array_equal(old[k],new[k]) for k in old.files if k.startswith("emission_") or k in ("source_id","weights")}
        assert all(same.values()), same
        energy = new["gun_exit_kinetic_energy_ev"]
        receipt["executed_source_arrays_equal"] = same
        receipt["gun_exit_energy_ev"] = dict(minimum=float(energy.min()), maximum=float(energy.max()),
            mean=float(energy.mean()), std=float(energy.std()))
        receipt["observer_archive_matches"] = sha256(observation_source).hexdigest() == report["observer_sha256"]
        assert receipt["observer_archive_matches"]
        s = report["analysis"]["surface"]
        i = int(np.flatnonzero(new["z_mm"] == s["z_mm"])[0])
        p = int(np.flatnonzero(new["source_id"] == 306)[0])
        receipt["id306_surface_radius_nm"] = float(np.hypot(new["x_m"][i,p]-s["centroid_x_m"], new["y_m"][i,p]-s["centroid_y_m"])*1e9)
    (OUT / "control-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(status=report["execution_status"], elapsed_s=report["elapsed_s"],
        surface={k:v for k,v in s.items() if k != "source_ids"}, d95_waist=report["analysis"]["d95_waist"],
        energy=receipt["gun_exit_energy_ev"], id306_radius_nm=receipt["id306_surface_radius_nm"])), flush=True)


if __name__ == "__main__":
    main()
