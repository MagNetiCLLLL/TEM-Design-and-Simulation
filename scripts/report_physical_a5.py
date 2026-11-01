"""Assemble measured A5 evidence without changing or accepting an optical state."""
from argparse import ArgumentParser
import csv
from hashlib import sha256
import json

import numpy as np

from calibrate_physical_a5 import (ARCHIVE, ROOT, INPUTS, DRIVES, apply, read_json,
    _json, gallery, verify_archive, assess_trial, coefficient_vector)
from reproduce_probe_baseline import load_baseline
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.working_point import WorkingPointCheckpoint


def observable_row(rec):
    sample = rec["at_specimen"]
    return dict(name=rec["name"], a5_mm=float(np.linalg.norm(rec["fits"][0]["terms_m"]["A5"])*1000),
        a5_quarter_mm=float(np.linalg.norm(rec["fits"][1]["terms_m"]["A5"])*1000),
        d95_nm=sample["d95_nm"], rms_radius_nm=sample["rms_radius_nm"],
        core_50pm_ua=sample["core_50pm_a"]*1e6, alpha95_mrad=sample["alpha95_mrad"],
        transmitted_ua=sample["current_a"]*1e6, particles=sample["surviving_rays"],
        outer_current_ua=rec["tails"]["outside_1nm"]["current_a"]*1e6,
        outer_m6_ua=rec["tails"]["outside_1nm"]["absolute_sixth_a"]*1e6,
        outer_shell_m6_sum_ua=rec["tails"]["outer_shell_absolute_sum_a"]*1e6,
        inner_m3_ua=sample["threefold_tail"]["harmonic_current_a"]*1e6,
        waist_offset_nm=rec["at_crossover"]["offset_from_surface_nm"] if rec["at_crossover"] else None,
        rms_minus_d95_z_nm=rec["separation"]["resolved_delta_z_nm"] if rec["separation"] else None,
        step_mm=rec["maximum_step_mm"], image=rec["name"]+"/incident-probe.png")


def integration_comparison(coarse, fine):
    a, b = observable_row(coarse), observable_row(fine)
    if coarse["controls"] != fine["controls"] or coarse["source_hashes"] != fine["source_hashes"]:
        raise ValueError("Integration checks must share controls and original source")
    delta = {key: b[key]-a[key] for key in a if isinstance(a[key], (int, float))}
    delta.pop("step_mm")
    # These are checks of step sensitivity, not statistical confidence bounds.
    delta_coeff = coefficient_vector(fine["fits"][0])-coefficient_vector(coarse["fits"][0])
    return dict(coarse=coarse["name"], fine=fine["name"], fine_minus_coarse=delta,
        coefficient_delta_pm=delta_coeff.tolist(), max_coefficient_change_pm=float(abs(delta_coeff).max()),
        scope="One halved integration step; fixed source, not particle-count convergence")


def report(name):
    base = read_json(ARCHIVE/"baseline/result.json")
    trial = read_json(ARCHIVE/name/"result.json")
    fine = read_json(ARCHIVE/(name+"-fine")/"result.json")
    base_fine = read_json(ARCHIVE/"baseline-fine/result.json")
    topology = read_json(ARCHIVE/name/"topology.json")
    if topology["input_digest"] != trial["input_digest"]:
        raise ValueError("Topology refers to another candidate")
    gates = assess_trial(name, native_focus=True)
    fine_gates = assess_trial(name+"-fine", native_focus=True)
    numerical = dict(baseline=integration_comparison(base, base_fine),
                     candidate=integration_comparison(trial, fine))
    _json(ARCHIVE/"integration-comparison.json", numerical)
    requested_keys = ("fixed_population", "original_transmission", "angle", "a2_protected", "c3_protected",
        "outer_sixfold_improvement", "outer_current", "outer_shells", "d95", "rms", "core_current", "native_focus")
    requested = {key: bool(gates["gates"][key] and fine_gates["gates"][key]) for key in requested_keys}
    requested["a5_decreases_both_pupils_and_steps"] = max(gates["a5_fraction"]+fine_gates["a5_fraction"]) < 1
    requested["normal_topology"] = bool(topology["passed"])
    for rec in (gates, fine_gates):
        rec["final_qualification"] = "REVIEW_CANDIDATE_ONLY_STRICT_GATES_FAILED"
        rec["integration_evidence"] = "../integration-comparison.json"
        rec["topology_evidence"] = "../"+name+"/topology.json"
        _json(ARCHIVE/rec["name"]/"gates.json", rec)
    state, _, _, _ = load_baseline(INPUTS)
    apply(state, trial["controls"])
    state.step_mm = trial["maximum_step_mm"]
    snapshot = capture_instrument_snapshot(state)
    if snapshot.digest != trial["input_digest"]:
        raise ValueError("Candidate package input differs from the measured setting")
    package = ARCHIVE/name/"review-candidate-inputs.temwp"
    cp = WorkingPointCheckpoint(snapshot, {}, state.sample.upper_surface_z_mm, "inputs-only:"+snapshot.digest,
        dict(label="A5 coupled review candidate - NOT ACCEPTED", package_kind="INSTRUMENT_INPUTS_ONLY",
             validation_status="INPUTS_ONLY", phase_status="NOT_COMPUTED", quality="NOT_COMPUTED",
             source_representation="classical tip particles", correction_acceptance="NOT_ACCEPTED",
             evidence="docs/development/probe_a5_physical_20261101/README.md"))
    if not package.exists():
        cp.write_package(package)
    restored = WorkingPointCheckpoint.read_package(package)
    if restored.digest != cp.digest or capture_instrument_snapshot(restored.compatible_state()).digest != snapshot.digest:
        raise ValueError("Candidate input package does not restore exactly")
    gallery()
    verification = verify_archive()
    results = sorted((read_json(f) for f in ARCHIVE.glob("*/result.json")), key=lambda rec: rec["utc"])
    rows = [observable_row(rec) for rec in results]
    with (ARCHIVE/"trial-ledger.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    images = ["# Every executed specimen-current image", "",
        "Chronological native current images. All use the same source, fixed axes and absolute color scale.",
        "Response perturbations are uncompensated diagnostic trials, not accepted working points.", ""]
    for rec in results:
        images.extend(["## "+rec["name"], "", f"![{rec['name']}]({rec['name']}/incident-probe.png)", "",
                       f"[Measurement]({rec['name']}/result.json) | [Input recipe]({rec['name']}/input-recipe.json)", ""])
    (ARCHIVE/"IMAGE_HISTORY.md").write_text("\n".join(images), encoding="utf-8")
    audit = read_json(ARCHIVE/"driver-audit.json")
    physical = []
    for row in audit["drivers"]:
        inc = trial["controls"].get(row["channel"], 0)*row["step"]
        physical.append(dict(channel=row["channel"], coordinate=row["coordinate"], unit=row["unit"],
            baseline=row["value"], change=inc, candidate=row["value"]+inc))
    result = dict(schema="physical-a5-study-outcome-v1", candidate=name,
        status="COUPLED_REVIEW_CANDIDATE_NOT_ACCEPTED", accepted_working_point="unchanged original A5 baseline",
        requested_scope_measured_checks=requested, requested_scope_pass=all(requested.values()),
        strict_study_pass=bool(gates["all_scalar_and_probe_gates"] and fine_gates["all_scalar_and_probe_gates"]),
        failed_strict_gates=[key for key, val in gates["gates"].items() if not val],
        scope_note="1% A5 and 5 pm all-other-term gates are investigator study targets, not requirements supplied by the user. Their failures remain visible.",
        baseline=observable_row(base), candidate_measured=observable_row(trial), candidate_fine=observable_row(fine),
        original_source=trial["source_digest"], driver_changes=physical,
        archive_trials=verification["completed_trials"], input_package_restore_exact=True,
        candidate_package_sha256=sha256(package.read_bytes()).hexdigest(),
        limitations=["D4/C5 coupling exceeds the independent-A5 budget", "Inner threefold tail increases",
            "D95/RMS focus separation increases", "No verified winding-current map or independent OEM supply rank",
            "No source-population convergence or coherent/atomic-image validation", "Local search is not a global feasibility proof"],
        implementation_sha256={str(file.relative_to(ROOT)): sha256(file.read_bytes()).hexdigest()
            for file in [ROOT/"scripts"/n for n in ("calibrate_physical_a5.py", "design_native_a5.py", "polish_native_a5.py", "report_physical_a5.py")]})
    _json(ARCHIVE/"outcome.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("name", default="native-mid-polished", nargs="?")
    report(parser.parse_args().name)
