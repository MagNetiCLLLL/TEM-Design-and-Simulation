"""Repeat the unchanged 769-electron candidate at finer column steps."""
from pathlib import Path
import json
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from measure_probe_waist import measure
from validate_joint_probe import load_candidate
from temsim.instrument_snapshot import capture_instrument_snapshot

OUT = ROOT / "outputs/joint-probe-100um-30mrad-20261028"
baseline_path = OUT / "seed769-native-d95-waist-h02"
baseline = json.loads((baseline_path / "report.json").read_text())
assert baseline["execution_status"] == "COMPLETE", baseline["errors"]
minimum = baseline["analysis"]["d95_waist"]["representative_offset_nm"]
# Add actual checkpoints around the coarse observed minimum, not interpolated data.
center = round(minimum)
extra = [round(center + j * .1, 8) for j in range(-25, 26)]
assert all(abs(v) <= 100 for v in extra)


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [entry for key in sorted(left.keys() | right.keys())
                for entry in differences(left.get(key), right.get(key), prefix + "/" + key)]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [entry for i, (a, b) in enumerate(zip(left, right))
                for entry in differences(a, b, prefix + "/" + str(i))]
    return [] if left == right else [{"path": prefix, "before": left, "after": right}]


runs = [(baseline_path, baseline)]
for step, suffix in ((.05, "h005"), (.025, "h0025")):
    state, candidate = load_candidate(OUT / "seed769-h02/best.json")
    original = capture_instrument_snapshot(state).to_dict()
    assert state.step_mm == .2 and state.electron_gun.emitter.ray_count == 769
    state.step_mm = step
    changed = capture_instrument_snapshot(state).to_dict()
    delta = differences(original, changed)
    graph_changes = [item for item in delta if item["path"].startswith("/graph/")]
    assert len(graph_changes) == 1 and graph_changes[0]["path"].endswith("/step_mm"), delta
    assert {item["path"].split("/")[1] for item in delta} <= {"graph", "digest"}, delta
    destination = OUT / ("seed769-native-d95-waist-" + suffix)
    report = measure(state, destination, extra_z_offset_nm=extra,
        threads=8, time_budget_s=1800, candidate={**candidate, "explicit_input_delta": delta})
    shutil.copyfile(ROOT / "scripts/measure_probe_waist.py", destination / "measure_probe_waist.py")
    (destination / "step-only-input-delta.json").write_text(json.dumps(delta, indent=2) + "\n")
    print(json.dumps({"step_mm": step, "execution_status": report["execution_status"],
        "elapsed_s": report["elapsed_s"], "surface": report.get("analysis", {}).get("surface"),
        "d95_waist": report.get("analysis", {}).get("d95_waist")}), flush=True)
    if report["execution_status"] != "COMPLETE":
        raise RuntimeError(report["errors"])
    runs.append((destination, report))

data = [np.load(path / "particles.npz", allow_pickle=False) for path, _ in runs]
source_keys = [key for key in data[0].files if key.startswith("emission_") or
               key in ("source_id", "weights", "gun_exit_kinetic_energy_ev")]
source_equal = all(np.array_equal(data[0][key], item[key]) for item in data[1:] for key in source_keys)
summary = {"scope": "Classical fixed-source and fixed-control column-step diagnostic; not full qualification",
           "source_arrays_exactly_equal": source_equal, "runs": [], "fine_comparison": {}}
for (path, report), arrays in zip(runs, data):
    analysis = report["analysis"]
    i = int(np.flatnonzero(arrays["z_mm"] == analysis["surface"]["z_mm"])[0])
    p = int(np.flatnonzero(arrays["source_id"] == 306)[0])
    s = analysis["surface"]
    radius = np.hypot(arrays["x_m"][i,p] - s["centroid_x_m"], arrays["y_m"][i,p] - s["centroid_y_m"]) * 1e9
    summary["runs"].append({"report": str(path / "report.json"), "surface": {k:v for k,v in s.items() if k != "source_ids"},
        "d95_waist": analysis["d95_waist"], "rms_waist_diagnostic": analysis["rms_waist_diagnostic"],
        "id306_surface_radius_nm": radius, "current_membership_stable": analysis["positive_current_membership_stable"]})
a,b = (item["analysis"] for _,item in runs[-2:])
for key in ("d90_nm", "d95_nm", "alpha95_mrad", "current_a", "rms_radius_nm"):
    summary["fine_comparison"][key] = {"absolute_change": b["surface"][key]-a["surface"][key],
        "relative_change": b["surface"][key]/a["surface"][key]-1}
summary["fine_comparison"]["observed_d95_waist_shift_nm"] = b["d95_waist"]["representative_offset_nm"]-a["d95_waist"]["representative_offset_nm"]
(OUT / "diagnostics/native-d95-waist-step-comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"comparison": str(OUT / "diagnostics/native-d95-waist-step-comparison.json"),
                  "source_arrays_exactly_equal": source_equal, "fine_comparison": summary["fine_comparison"]}), flush=True)
