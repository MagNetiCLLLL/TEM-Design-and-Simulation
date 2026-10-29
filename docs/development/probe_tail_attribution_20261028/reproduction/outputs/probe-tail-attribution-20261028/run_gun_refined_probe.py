"""Propagate a complete unchanged source after explicit gun precision changes."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import shutil
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "agent.md").is_file())
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(Path(__file__).parent)]
from measure_probe_waist import measure
from validate_joint_probe import load_candidate
from temsim.instrument_snapshot import capture_instrument_snapshot
from run_monochromatic_control import differences

BASE = ROOT / "outputs/joint-probe-100um-30mrad-20261028/seed769-native-d95-waist-h005"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gun-step-mm", type=float, default=.1, choices=(.1,.05))
    parser.add_argument("--cells-per-bore", type=int, default=16, choices=(16,20,24))
    args = parser.parse_args()
    state, identity = load_candidate(BASE / "inputs.json")
    before = capture_instrument_snapshot(state).to_dict()
    launch = state.electron_gun.emitter.emit()
    assert state.electron_gun.trace_step_mm == .2
    assert state.electron_gun._gun_field_cells_per_bore == 16
    state.electron_gun.trace_step_mm = args.gun_step_mm
    state.electron_gun._gun_field_cells_per_bore = args.cells_per_bore
    delta = differences(before, capture_instrument_snapshot(state).to_dict())
    graph = [v for v in delta if v["path"].startswith("/graph/")]
    expected = {"trace_step_mm"} | ({"_gun_field_cells_per_bore"} if args.cells_per_bore != 16 else set())
    assert {v["path"].rsplit("/",1)[-1] for v in graph} == expected, delta
    assert len(graph) == len(expected), delta
    assert {v["path"].split("/")[1] for v in delta} <= {"graph","digest"}, delta
    after_launch = state.electron_gun.emitter.emit()
    fields = ("x_m","y_m","tx_rad","ty_rad","weight","ray_id","energy_offset_ev")
    assert all(np.array_equal(getattr(launch,k),getattr(after_launch,k)) for k in fields)
    baseline = json.loads((BASE / "report.json").read_text())
    observer = (ROOT / "scripts/measure_probe_waist.py").read_bytes()
    receipt = dict(type="gun numerical precision, full-column response", baseline=str(BASE),
        explicit_input_delta=delta, emission_including_energy_exactly_equal=True,
        no_retuning=True, column_step_mm=state.step_mm, gun_step_mm=args.gun_step_mm,
        cells_per_bore=args.cells_per_bore, numerical_convergence="Single declared comparison; not full qualification")
    report = measure(state, args.output, extra_z_offset_nm=baseline["parameters"]["extra_z_offset_nm"],
        threads=4, time_budget_s=1800, cancel_file=args.output / "cancel",
        candidate={**identity, "precision_control": receipt})
    (args.output / "measure_probe_waist.py").write_bytes(observer)
    shutil.copyfile(__file__, args.output / "run_gun_refined_probe.py")
    (args.output / "precision-control.json").write_text(json.dumps(receipt, indent=2)+"\n")
    if report["execution_status"] != "COMPLETE":
        raise RuntimeError(report["errors"])
    assert sha256(observer).hexdigest() == report["observer_sha256"]
    with np.load(BASE / "particles.npz", allow_pickle=False) as old, np.load(args.output / "particles.npz", allow_pickle=False) as new:
        same = {k:np.array_equal(old[k],new[k]) for k in old.files if k.startswith("emission_") or k in ("source_id","weights")}
        assert all(same.values()), same
        s = report["analysis"]["surface"]
        i = int(np.flatnonzero(new["z_mm"] == s["z_mm"])[0])
        p = int(np.flatnonzero(new["source_id"] == 306)[0])
        receipt["executed_source_arrays_equal"] = same
        receipt["id306_surface_radius_nm"] = float(np.hypot(new["x_m"][i,p]-s["centroid_x_m"],new["y_m"][i,p]-s["centroid_y_m"])*1e9)
    (args.output / "precision-control.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(dict(status=report["execution_status"], elapsed_s=report["elapsed_s"],
        surface={k:v for k,v in s.items() if k!="source_ids"}, d95_waist=report["analysis"]["d95_waist"],
        id306_radius_nm=receipt["id306_surface_radius_nm"])),flush=True)


if __name__ == "__main__":
    main()
