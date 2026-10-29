"""Offline exact-plane comparison of completed gun-precision probe profiles."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "agent.md").is_file())
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from temsim.benchmark_suite import atomic_json
from temsim.physics.beam_statistics import transverse_beam_statistics
from solve_joint_probe import snapshot_changes

BASE = ROOT / "outputs/joint-probe-100um-30mrad-20261028/seed769-native-d95-waist-h005"
HERE = Path(__file__).resolve().parent
CASES = dict(original=BASE, gun_fine=HERE / "poly769-gunstep01-grid16-h005",
             grid20=HERE / "poly769-gunstep01-grid20-h005", grid24=HERE / "poly769-gunstep01-grid24-h005")
PHASE = ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def load(directory):
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report["execution_status"] != "COMPLETE" or report["receipt"]["status"] != "COMPLETE":
        raise ValueError("All profile inputs must have completed actual execution")
    hashes = {name: sha(directory / name) for name in ("inputs.json", "particles.npz", "report.json")}
    if any(hashes[name] != report["files_sha256"][name] for name in ("inputs.json", "particles.npz")):
        raise ValueError("A profile input changed after execution")
    with np.load(directory / "particles.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    weights = arrays["weights"]
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid retained original current weights")
    native = []
    for index, z in enumerate(arrays["z_mm"]):
        active = arrays["alive"][index] & (weights > 0)
        if not active.any() or any(not np.isfinite(arrays[key][index, active]).all() for key in PHASE):
            raise ValueError("Nonfinite or missing positive-current phase cannot be silently omitted")
        if np.any(arrays["kinetic_energy_ev"][index, active] <= 0) or np.any(arrays["flight_time_s"][index, active] < 0):
            raise ValueError("Invalid retained energy or clock")
        stats = transverse_beam_statistics(*(arrays[key][index] for key in PHASE[:4]), alive=active, weights=weights)
        native.append(dict(z_mm=float(z), d95_nm=float(2e9 * stats.radius_95_m),
            rms_radius_nm=float(1e9 * stats.radius_rms_m), alpha95_mrad=float(1e3 * stats.convergence_95_rad),
            current_a=float(report["receipt"]["source_current_a"] * float(weights[active].sum()/weights.sum())),
            positive_survivors=int(active.sum())))
    requested = np.asarray(report["parameters"]["requested_planes_mm"])
    index = np.searchsorted(arrays["z_mm"], requested)
    if not np.array_equal(arrays["z_mm"][index], requested):
        raise ValueError("A requested actual observation plane is missing")
    minimum = min(native[i]["d95_nm"] for i in index)
    if minimum != report["analysis"]["d95_waist"]["minimum"]:
        raise ValueError("Native raw D95 minimum disagrees with the saved measurement")
    summary = dict(directory=str(directory), source_files_sha256=hashes,
        input_digest=report["input_digest"], implementation=report["implementation"],
        surface=report["analysis"]["surface"], d95_waist=report["analysis"]["d95_waist"],
        surface_gate=report["analysis"]["surface_gate"], elapsed_s=report["elapsed_s"],
        native_raw_minimum_exactly_matches_saved=True, total_retained_planes=len(native),
        requested_profile_planes=len(requested))
    inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    return dict(report=report, arrays=arrays, native=native, requested=requested, summary=summary, inputs=inputs)


def compare(before, after):
    a, b = before["arrays"], after["arrays"]
    invariant = [key for key in a if key.startswith("emission_") or key in ("weights", "source_id")]
    if any(not np.array_equal(a[key], b[key]) for key in invariant):
        raise ValueError("Original emission coordinates/directions, IDs or absolute weights changed")
    if not np.array_equal(before["requested"], after["requested"]):
        raise ValueError("Requested native profile planes differ")
    common, ia, ib = np.intersect1d(a["z_mm"], b["z_mm"], return_indices=True)
    rows = []
    for z, i, j in zip(common, ia, ib):
        left, right = before["native"][i], after["native"][j]
        positive = a["weights"] > 0
        old, new = a["alive"][i] & positive, b["alive"][j] & positive
        both = old & new
        delta = np.hypot(b["x_m"][j, both]-a["x_m"][i, both], b["y_m"][j, both]-a["y_m"][i, both])*1e9
        rows.append(dict(z_mm=float(z), exact_same_positive_population=bool(np.array_equal(old, new)),
            common_positive_survivors=int(both.sum()), added_ids=a["source_id"][new & ~old].tolist(),
            lost_ids=a["source_id"][old & ~new].tolist(),
            maximum_common_particle_xy_change_nm=float(delta.max()) if len(delta) else None,
            observables={key: dict(before=left[key], after=right[key], delta=right[key]-left[key],
                relative_delta=(right[key]-left[key])/abs(left[key]) if left[key] else None)
                for key in ("d95_nm", "rms_radius_nm", "alpha95_mrad", "current_a")}))
    wa = before["summary"]["d95_waist"]
    wb = after["summary"]["d95_waist"]
    a_bounds, b_bounds = wa["neighbour_bracket_offset_nm"], wb["neighbour_bracket_offset_nm"]
    offset_delta = wb["representative_offset_nm"] - wa["representative_offset_nm"]
    bracket_separation = max(abs(a_bounds[0]-b_bounds[1]), abs(a_bounds[1]-b_bounds[0]))
    return dict(all_common_native_planes=rows, common_plane_count=len(common),
        unmatched_before_planes_mm=np.setdiff1d(a["z_mm"], common).tolist(),
        unmatched_after_planes_mm=np.setdiff1d(b["z_mm"], common).tolist(),
        same_original_emission_ids_and_weights=True,
        complete_input_graph_changes=snapshot_changes(before["inputs"]["graph"], after["inputs"]["graph"]),
        observed_minimum_d95_delta_nm=wb["minimum"]-wa["minimum"],
        observed_minimum_d95_relative_delta=wb["minimum"]/wa["minimum"]-1.,
        observed_minimum_offset_delta_nm=offset_delta,
        before_neighbour_bracket_offset_nm=a_bounds, after_neighbour_bracket_offset_nm=b_bounds,
        maximum_separation_between_reported_brackets_nm=bracket_separation,
        prior_axial_comparison_budget_nm=.25,
        observed_representative_delta_within_prior_axial_budget=abs(offset_delta) <= .25,
        entire_reported_brackets_within_prior_axial_budget=bracket_separation <= .25,
        bracket_scope="Separation of observed neighbouring samples; not a rigorous continuous-argmin error bound")


def main():
    output = HERE / "gun-precision-full-column-comparison.json"
    if output.exists():
        raise FileExistsError(output)
    cases = {key: load(path) for key, path in CASES.items()}
    pairs = (("original", "gun_fine"), ("gun_fine", "grid20"), ("grid20", "grid24"), ("gun_fine", "grid24"))
    report = dict(schema="temsim-gun-precision-full-column-comparison-v1", status="MEASURED", qualification="NOT_QUALIFIED",
        model="classical_particles", source_count=769, column_step_mm=.05, maximum_d95_cap_nm=None,
        cases={key: value["summary"] for key, value in cases.items()},
        comparisons={left+"_to_"+right: compare(cases[left], cases[right]) for left, right in pairs},
        limitations=["All comparisons use actual native phases at exact common Z; no interpolation or coordinate rescaling.",
            "Gun maximum-step refinement changes gun integration only.",
            "Cells-per-bore refinement changes the shared full-instrument electrostatic discretization, including its column field.",
            "There is no source-count convergence, pure chromatic attribution, OEM model validation or focus qualification.",
            "The 0.25 nm axial budget is reported explicitly and is not loosened to accept a measured grid change.",
            "A nearby observed sample minimum is not an exact continuous argmin."])
    atomic_json(output, report)
    print(output)
    for key, value in report["comparisons"].items():
        print(key, {name: value[name] for name in ("common_plane_count", "observed_minimum_d95_delta_nm",
              "observed_minimum_offset_delta_nm", "after_neighbour_bracket_offset_nm")})


if __name__ == "__main__":
    main()
