"""Summarize particle-only concentration at one fixed plane and independent refinements."""
import math

import numpy as np


def fixed_target_assessment(reference, candidate, focus_rows, *, count_refinement=None,
                            step_refinement=None, angle_target_mrad=30., angle_tolerance_mrad=.3,
                            diameter="d50"):
    """Compare native rows. Callers verify snapshot, source/field lineage and numerical controls."""
    if diameter not in {"d50", "d95"}:
        raise ValueError("Select d50 or d95 for the through-focus objective")
    size_key = diameter + "_nm"
    metrics = ("d50_nm", "d95_nm", "core_50pm_a", "alpha95_mrad", "current_a")
    for row in (reference, candidate, *focus_rows, count_refinement, step_refinement):
        if row is None:
            continue
        if not all(math.isfinite(row[k]) and row[k] >= 0 for k in metrics) or not math.isfinite(row["z_mm"]):
            raise ValueError("Fixed-target comparison requires finite nonnegative measured observables")
    z = candidate["z_mm"]
    if any(row["z_mm"] != z for row in (reference,count_refinement,step_refinement) if row is not None):
        raise ValueError("Fixed-target comparisons must share the exact physical target")
    if not all(candidate[k] > 0 for k in metrics) or not reference["current_a"] > 0:
        raise ValueError("The declared working point requires positive current, angle and measurable size")
    at = [row for row in focus_rows if row["z_mm"] == z]
    if len(at) != 1 or any(at[0][k] != candidate[k] for k in metrics):
        raise ValueError("Through-focus profile must include the actual target measurement")
    focus_z = np.array([row["z_mm"] for row in focus_rows])
    if np.any(np.diff(focus_z) <= 0):
        raise ValueError("Through-focus planes must be in physical order")
    current_feasible = candidate["current_a"] >= reference["current_a"]*(1-1e-12)
    angle_feasible = abs(candidate["alpha95_mrad"]-angle_target_mrad) <= angle_tolerance_mrad
    smaller = candidate["d95_nm"] < reference["d95_nm"]
    if diameter == "d50":
        smaller = smaller and candidate["d50_nm"] < reference["d50_nm"]
    brighter = candidate["core_50pm_a"] > reference["core_50pm_a"]
    peak = max(row["core_50pm_a"] for row in focus_rows)
    smallest = min(row[size_key] for row in focus_rows)
    peak_ratio = candidate["core_50pm_a"]/peak
    size_ratio = candidate[size_key]/smallest if smallest > 0 else None
    best_core = [row["z_mm"] for row in focus_rows if row["core_50pm_a"] == peak]
    best_size = [row["z_mm"] for row in focus_rows if row[size_key] == smallest]
    interior = all(focus_z[0] < plane < focus_z[-1] for plane in (*best_core,*best_size))
    stable_population = all("source_ids" in row and "source_ids" in at[0]
                            and row["source_ids"] == at[0]["source_ids"] for row in focus_rows)
    focus_status = "INCONCLUSIVE"
    exact = near = None
    if interior and stable_population and size_ratio is not None:
        exact = z in best_core and z in best_size
        if diameter == "d95":
            near = peak_ratio >= .99 and size_ratio <= 1.01
            focus_status = "PASS" if exact else "FAIL"
        else:
            near = peak_ratio >= .95 and size_ratio <= 1.05
            focus_status = "PASS" if near else "FAIL"
    axial = None
    if all("rms_radius_nm" in row for row in focus_rows):
        from measure_probe_waist import observed_minimum, waist_separation
        rms_values = [row["rms_radius_nm"] for row in focus_rows]
        if not all(math.isfinite(v) and v >= 0 for v in rms_values):
            raise ValueError("RMS waist assessment requires finite nonnegative radii")
        rms = observed_minimum(focus_z, rms_values, z)
        d95 = observed_minimum(focus_z, [row["d95_nm"] for row in focus_rows], z)
        support = all(row.get("surviving_rays", 0) >= 16 and
                      row.get("effective_rays", 0) >= 16*(1-32*np.finfo(float).eps) for row in focus_rows)
        axial = dict(rms=rms, d95=d95, separation=waist_separation(rms, d95, stable=stable_population,
                    support=support), role="Secondary minimization objective; no required zero separation")
    def convergence(other):
        if other is None:
            return dict(status="NOT_EVALUATED")
        budgets = dict(d95_nm=.05, core_50pm_a=.10, current_a=.01)
        if diameter == "d50":
            budgets["d50_nm"] = .05
        changes = {key: abs(other[key]/candidate[key]-1) for key in budgets}
        angle_shift = abs(other["alpha95_mrad"]-candidate["alpha95_mrad"])
        passed = all(changes[k] <= budgets[k] for k in budgets) and angle_shift <= angle_tolerance_mrad
        return dict(status="PASS" if passed else "FAIL", relative_changes=changes,
                    relative_budgets=budgets, angle_change_mrad=angle_shift)
    return dict(fixed_target_performance="PASS" if current_feasible and angle_feasible and smaller and brighter else "FAIL",
        focus_diameter=diameter, both_observed_extrema_at_target=exact, within_focus_allowance=near,
        axial_waists=axial,
        focus_status=focus_status, target_fraction_of_best_core_current=peak_ratio,
        best_core_planes_mm=best_core,
        **{f"target_over_best_{diameter}": size_ratio, f"best_{diameter}_planes_mm": best_size},
        focus_population_stable=stable_population, best_planes_bounded=interior,
        angle_feasible=angle_feasible, transmission_preserved=current_feasible,
        source_refinement=convergence(count_refinement),step_refinement=convergence(step_refinement),
        full_qualification="NOT_QUALIFIED",
        scope="Fixed-target relative metrics and numerical checks. Field, topology and aberration evidence are separate.")


def _read_survey(path):
    import json
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("schema") != "unified-crossover-geometric-cs-v1"
            or report["field_audit"].get("passed") is not True
            or report["original_positive_current_count"] < 1000):
        raise ValueError("A completed original-source unified particle survey is required")
    target = report["planes_mm"]["S_target"]
    rows = report.get("plane_metrics")
    if rows is None:
        rows = report["profile"]["planes"]
    rows = [dict(row, core_50pm_a=row["axis_core_50pm_current_a"]) for row in rows
            if abs(row["z_mm"]-target) <= .00010000001]
    return report, next(row for row in rows if row["z_mm"] == target), rows


def main():
    from argparse import ArgumentParser
    from pathlib import Path
    import json
    from temsim.instrument_snapshot import InstrumentSnapshot
    from solve_joint_probe import snapshot_changes
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--count-refinement", type=Path)
    parser.add_argument("--step-refinement", type=Path)
    parser.add_argument("--diameter", choices=("d95", "d50"), default="d95",
                        help="Native D95 with exact joint extrema (default), or historical D50 allowance")
    args = parser.parse_args()
    reference, base, _ = _read_survey(args.reference)
    report, candidate, profile = _read_survey(args.candidate)

    def snapshot(path, expected):
        value = InstrumentSnapshot.from_dict(json.loads(path.with_name("inputs.json").read_text(encoding="utf-8")))
        if value.digest != expected["input_digest"]:
            raise ValueError("Survey input file does not match its execution receipt")
        return value

    parent = snapshot(args.candidate, report)
    original = snapshot(args.reference, reference)
    if not (original.implementation == parent.implementation
            and report["physical_source_ids_sha256"] == reference["physical_source_ids_sha256"]
            and report["source_weights_sha256"] == reference["source_weights_sha256"]
            and report["original_current_a"] == reference["original_current_a"]
            and report["pupil"] == reference["pupil"]):
        raise ValueError("Operating-point comparison requires the same implementation, current and reference family")
    with (np.load(args.reference.with_name("rays.npz"), allow_pickle=False) as ref_rays,
          np.load(args.candidate.with_name("rays.npz"), allow_pickle=False) as candidate_rays):
        for key in ("source_id", "weights", "gun_exit_xy_slopes", "gun_exit_kinetic_energy_ev",
                    "gun_exit_flight_time_s", "launch_pencil"):
            if not np.array_equal(ref_rays[key], candidate_rays[key], equal_nan=True):
                raise ValueError(f"Reference source or diagnostic pencil changed ({key})")
    refined_foci = {}
    def refinement(path, kind):
        if path is None:
            return None
        other, row, other_profile = _read_survey(path)
        after = snapshot(path, other)
        if (parent.implementation != after.implementation or parent.external_inputs != after.external_inputs
                or report["pupil"] != other["pupil"] or report["planes_mm"] != other["planes_mm"]
                or report["original_current_a"] != other["original_current_a"]):
            raise ValueError("Refinement changed physical inputs or diagnostic references")
        changes = snapshot_changes(dict(parent.to_dict()["graph"]), dict(after.to_dict()["graph"]))
        root = parent.graph["root"]["ref"]
        allowed = {f"/nodes/{root}/attributes/step_mm"}
        if kind == "count":
            gun_ref = parent.graph["nodes"][root]["attributes"]["electron_gun"]["ref"]
            emitter_ref = parent.graph["nodes"][gun_ref]["attributes"]["emitter"]["ref"]
            allowed = {f"/nodes/{emitter_ref}/attributes/ray_count"}
        if any(change["path"] not in allowed for change in changes):
            raise ValueError("Refinement changed undeclared controls")
        equal_grid = (report["maximum_observation_spacing_mm"] == other["maximum_observation_spacing_mm"]
                      and [r["z_mm"] for r in profile] == [r["z_mm"] for r in other_profile])
        smaller_step = other["actual_maximum_step_mm"] < .99*report["actual_maximum_step_mm"]
        same_step = math.isclose(other["actual_maximum_step_mm"],report["actual_maximum_step_mm"],rel_tol=1e-6)
        more = other["original_positive_current_count"] > report["original_positive_current_count"]
        if not equal_grid or not ((more and same_step) if kind=="count" else
                (smaller_step and other["physical_source_ids_sha256"] == report["physical_source_ids_sha256"]
                 and other["source_weights_sha256"] == report["source_weights_sha256"])):
            raise ValueError("Not an independent numerical refinement")
        details = fixed_target_assessment(base, row, other_profile, diameter=args.diameter)
        refined_foci[kind] = {key: details[key] for key in (
            "focus_status", "both_observed_extrema_at_target", "within_focus_allowance",
            "target_fraction_of_best_core_current", f"target_over_best_{args.diameter}",
            "axial_waists", "angle_feasible", "transmission_preserved")}
        return row

    assessment = fixed_target_assessment(base, candidate, profile, diameter=args.diameter,
        count_refinement=refinement(args.count_refinement, "count"),
        step_refinement=refinement(args.step_refinement, "step"))
    assessment["refined_focus_observations"] = refined_foci
    print(json.dumps(assessment, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
