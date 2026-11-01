"""Package measured joint-search evidence without replacing accepted inputs."""
from argparse import ArgumentParser
import csv
from hashlib import sha256
import json
import numpy as np

from optimize_joint_probe import (ARCHIVE, PRIOR, ROOT, LABELS, configure, assess,
    summarize, physical)
from report_physical_a5 import observable_row, integration_comparison
from temsim.working_point import WorkingPointCheckpoint
from temsim.instrument_snapshot import capture_instrument_snapshot


def report(name):
    configure()
    summarize()
    verification=physical.verify_archive()
    base=physical.read_json(ARCHIVE/"baseline/result.json")
    selected=physical.read_json(ARCHIVE/name/"result.json")
    fine=physical.read_json(ARCHIVE/(name+"-fine")/"result.json")
    topology=physical.read_json(ARCHIVE/name/"topology.json")
    assessment=assess(name)
    fine_assessment=assess(name+"-fine")
    if topology["input_digest"] != selected["input_digest"]:
        raise ValueError("The topology check belongs to another input")
    numerical=integration_comparison(selected,fine)
    differences=numerical["fine_minus_coarse"]
    limits=dict(d95_nm=.001,rms_radius_nm=.001,alpha95_mrad=.001,
        core_50pm_ua=.005,inner_m3_ua=.005,outer_m6_ua=.005,outer_shell_m6_sum_ua=.005)
    numerical["comparison_limits"]=limits
    numerical["within_step_budget"]=bool(all(abs(differences[key])<=limit+1e-12 for key,limit in limits.items())
        and numerical["max_coefficient_change_pm"] <= .1
        and differences["particles"] == 0 and abs(differences["transmitted_ua"]) < 1e-10)
    physical._json(ARCHIVE/"integration-comparison.json",numerical)
    passed=bool(assessment["all_measured_goals"] and fine_assessment["all_measured_goals"]
                and topology["passed"] and numerical["within_step_budget"])
    state,_,_,_=physical.load_baseline(physical.INPUTS)
    parts=physical.objects(state)
    drive_values=[dict(channel=label,component=key,coordinate=attribute,unit=unit,
        baseline=float(getattr(parts[key],attribute)),
        change=float(selected["controls"].get(label,0.)*step))
        for label,key,attribute,step,unit in physical.DRIVES]
    physical.apply(state,selected["controls"])
    for row in drive_values:
        row["candidate"]=float(getattr(parts[row["component"]],row["coordinate"]))
    state.step_mm=selected["maximum_step_mm"]
    snapshot=capture_instrument_snapshot(state)
    if snapshot.digest != selected["input_digest"]:
        raise ValueError("The selected settings cannot be reproduced")
    package=ARCHIVE/name/"joint-candidate-inputs.temwp"
    checkpoint=WorkingPointCheckpoint(snapshot,{},state.sample.upper_surface_z_mm,"inputs-only:"+snapshot.digest,
        dict(label="Joint C3 C5 A5 fixed-source candidate - original baseline preserved",
             package_kind="INSTRUMENT_INPUTS_ONLY",validation_status="INPUTS_ONLY",
             phase_status="NOT_COMPUTED",quality="NOT_COMPUTED",source_representation="classical tip particles",
             study_status="MEASURED_FIXED_SOURCE_GOALS_PASSED" if passed else "REVIEW_CANDIDATE_GOALS_NOT_ALL_PASSED"))
    if not package.exists(): checkpoint.write_package(package)
    restored=WorkingPointCheckpoint.read_package(package)
    if restored.digest != checkpoint.digest or capture_instrument_snapshot(restored.compatible_state()).digest != snapshot.digest:
        raise ValueError("The candidate package does not restore exactly")
    trials=sorted([physical.read_json(f) for f in ARCHIVE.glob("*/result.json")],key=lambda rec:rec["utc"])
    rows=[observable_row(rec) for rec in trials]
    eligible=[]
    objective_keys=("d95", "rms", "core", "inner_m3", "inner_shell_m3", "outer_m6", "outer_shell_m6", "outer_current")
    for trial in trials:
        if trial["name"] == "baseline" or trial["maximum_step_mm"] != .05: continue
        a=assess(trial["name"])
        if a["all_measured_goals"]:
            values=np.array([a["ratios"][k]*(-1 if k=="core" else 1) for k in objective_keys])
            eligible.append((trial["name"],values,a["ratios"]))
    frontier=[dict(name=n,ratios=r) for n,v,r in eligible
        if not any(np.all(other<=v) and np.any(other<v) for name2,other,_ in eligible if name2!=n)]
    physical._json(ARCHIVE/"measured-pareto-candidates.json",dict(scope="Executed coarse trials meeting all goals; no global-optimum claim",candidates=frontier))
    for rec in (assessment,fine_assessment):
        rec["topology_and_step_qualification"]="MEASURED_SEE_INTEGRATION_COMPARISON_AND_TOPOLOGY"
        physical._json(ARCHIVE/rec["name"]/"joint-assessment.json",rec)
    with (ARCHIVE/"trial-ledger.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    pages=["# Joint-search specimen-current image history", "",
        "Every image is native unified-field propagation of the same original 2000 positive-current particles.",
        "Focus suffixes are separately executed shared-Objective changes. Initial or failed candidates remain visible.", ""]
    for rec in trials:
        sample=rec["at_specimen"]
        waist=rec["at_crossover"]
        pages.extend(["## "+rec["name"],"",
            f"D95 {sample['d95_nm']:.6f} nm; RMS radius {sample['rms_radius_nm']:.6f} nm; core {sample['core_50pm_a']*1e6:.3f} uA.",
            f"Measured crossover offset: {waist['offset_from_surface_nm']:.3f} nm." if waist else "Crossover unresolved.","",
            f"![{rec['name']}]({rec['name']}/incident-probe.png)","",
            f"[Measurements]({rec['name']}/result.json) | [Exact input recipe]({rec['name']}/input-recipe.json)",""])
    (ARCHIVE/"IMAGE_HISTORY.md").write_text("\n".join(pages),encoding="utf-8")
    outcome=dict(schema="joint-probe-search-outcome-v1",selected=name,
        status="FIXED_SOURCE_MEASURED_GOALS_PASSED" if passed else "NO_SELECTED_CANDIDATE_PASSES_ALL_GOALS",
        original_accepted_point="UNCHANGED",global_optimum="NOT_PROVEN",all_goals_passed=passed,
        baseline=observable_row(base),selected_measurement=observable_row(selected),fine_measurement=observable_row(fine),
        ratios=assessment["ratios"],hard_checks=assessment["hard_checks"],
        selection_reason="Balance of measured core current, RMS and shell-resolved tails; alternate nondominated settings retained",
        driver_changes=drive_values,
        geometric_coefficients_m=dict(baseline=base["fits"][0]["terms_m"],candidate=selected["fits"][0]["terms_m"]),
        assessment=assessment,topology=topology,numerical_check=numerical,
        candidate_restore_exact=True,package_sha256=sha256(package.read_bytes()).hexdigest(),
        measured_pareto_candidates=frontier,
        executions=verification["completed_trials"],source_digest=base["source_digest"],
        source_scope="Identical original 2000 particles; no source-count convergence or coherent calculation",
        implementation_sha256={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest()
            for f in (ROOT/"scripts/optimize_joint_probe.py",ROOT/"scripts/report_joint_probe.py")})
    physical._json(ARCHIVE/"outcome.json",outcome)
    print(json.dumps({k:outcome[k] for k in ("selected","status","executions","ratios","hard_checks")},indent=2))


if __name__ == "__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("name")
    report(parser.parse_args().name)
