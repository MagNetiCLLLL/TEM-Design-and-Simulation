"""Original-source high-angle trials with exact in-process gun dependency reuse.

Each JSON-line request is a complete independent finite-source physical trial.
Only production gun traces with the same dependency signature can be reused.
No clipped particle is restored and no source weight is renormalized.
"""
from pathlib import Path
from hashlib import sha256
import json
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
import study_two_condenser_nanoprobe as study
from temsim.cpu_resources import numerical_job
from temsim.profile_io import save_profile

cache = {}
observers = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in
    (Path(__file__), ROOT/"scripts/study_two_condenser_nanoprobe.py",
     ROOT/"scripts/measure_probe_waist.py", ROOT/"scripts/audit_joint_probe_fields.py")}
print("READY_FOR_JSON_TRIAL", flush=True)
for line in sys.stdin:
    if not line.strip():
        continue
    request = json.loads(line)
    if request.get("stop"):
        print("CLOSED", flush=True)
        break
    name = request["name"]
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Use a unique local trial directory name")
    out = HERE/name
    out.mkdir(exist_ok=False)
    started = perf_counter()
    count, step = int(request["count"]), float(request["step"])
    with numerical_job(2):
        state, selection = study.prepare_state(count, step, aperture_um=70.)
        study.set_controls(state, request["controls"])
        lenses = study.installed_lenses(state)
        projector = request.get("projector_controls", {
            "diffraction_lens": 13.59281212137673,
            "intermediate_lens": 8.51783304453163,
            "projector_lens_1": 32.1132572718703,
            "projector_lens_2": 9.826874422815507})
        for key, value in projector.items():
            if key not in {"diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2"}:
                raise ValueError("Unsupported projector control")
            lenses[key].percent = float(value)
            lenses[key].validate()
        focus, intervals, planes = study.observation_planes(state, 150., 2., True)
        fine = state.sample.upper_surface_z_mm + np.linspace(-10., 10., 81)*1e-6
        focus = np.unique(np.r_[focus, fine])
        planes = np.unique(np.r_[planes, fine])
        request.update(expected_c2_diameter_mm=.070, minimum_alpha95_mrad=10.,
            requested_planes_mm=planes.tolist(), focus_planes_mm=focus.tolist(),
            topology_intervals=intervals, checkpoint_population_budget=4096,
            observer_sha256=observers)
        study._json(out/"request.json", request)
        print("RUNNING "+name+" count="+str(count)+" step="+str(step), flush=True)
        arrays, info, snapshot = study.execute(state, planes, cache,
            expected_c2_diameter_mm=.070, checkpoint_ray_count=4096)
        study._json(out/"inputs.json", snapshot.to_dict())
        save_profile(out/"input.toml", state, selection)
        study._npz(out/"observations.npz", arrays)
        result = study.summarize(state, arrays, info, focus, intervals)
        result.update(status="COMPLETED_CLASSICAL_TWO_CONDENSER_OBSERVATION", execution=info,
            observer_sha256=observers,
            controls={key: float(lenses[key].percent) for key in
                ("condenser_lens_1", "condenser_lens_2", "mini_condenser", "objective_lens")},
            projector_controls=projector,
            observations_sha256=sha256((out/"observations.npz").read_bytes()).hexdigest(),
            elapsed_s=perf_counter()-started,
            scope="Full original source; actual 70 um finite contacts; common observation mesh; no waves")
        result["alpha95_requirement"] = dict(minimum_mrad=10., achieved_mrad=result.get("alpha95_mrad"),
            passed=bool(result.get("alpha95_mrad", 0.) >= 10.),
            definition="95% of actual specimen current about its weighted chief direction")
        study._json(out/"report.json", result)
        print("COMPLETED "+name+" "+json.dumps({
            key:result.get(key) for key in ("d95_nm", "rms_radius_nm", "alpha95_mrad",
                "surviving_positive_count", "sample_current_a", "direct_surface_focus")}), flush=True)
        print("NATIVE_FOCUS "+json.dumps({key:result.get("native_focus", {}).get(key) for key in
            ("d95_waist", "rms_waist_diagnostic")}), flush=True)
        print("READY_FOR_JSON_TRIAL", flush=True)
