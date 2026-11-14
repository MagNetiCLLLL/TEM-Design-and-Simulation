"""Run declared ideal Gaussian magnetic-lens and passive-tube experiments.

Every finite-source partition transports at least 1024 distinct positive-
current original particles. Larger populations reuse the unchanged completed
prefix trajectories, with explicit new population-current normalization.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from ideal_lens_beam import (array_hash, field_order_evidence, gaussian_field,
    phase_statistics, propagate, source_population, spherical_fit)
from ideal_lens_wall import classify_cylindrical_wall
from temsim.calculation_manifest import solver_source_identity


def write_json(path, value):
    def default(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=default, allow_nan=False)+"\n", encoding="utf-8")


def run_case(request, reference_report, reference_path, output):
    started = perf_counter()
    peak, sigma = float(request["peak_t"]), float(request["sigma_mm"])*1e-3
    reference_case = next(row for row in reference_report["cases"]
        if row["peak_t"] == peak and row["sigma_mm"] == request["sigma_mm"])
    reference = reference_case["variants"]["refined8"]
    ref = reference["coefficients"]
    outward = np.asarray(reference["pupil"]["outward_pupil_slopes_per_initial_position_m1"])
    initial_slopes = np.asarray(reference["pupil"]["initial_mechanical_slopes_per_position_m1"])
    conversion = abs(float(reference["pupil"]["radial_slope_per_m"]))
    radius = (float(request["source_radius_mm"])*1e-3 if "source_radius_mm" in request
              else float(request["paraxial_edge_mrad"])*1e-3/conversion)
    edge = conversion*radius
    start, image = -8*sigma, float(ref["image_z_m"])
    window = max(.1*sigma, 2*abs(ref["cs_mm"]*1e-3)*edge**2)
    if image-window <= start:
        raise ValueError("Declared through-focus window reaches the source plane")
    end = image+window
    focus_planes = image+np.linspace(-window, window, 129)
    planes = np.unique(np.r_[np.linspace(start, end, 65), focus_planes, image])
    field = gaussian_field(peak, sigma, int(request.get("field_order", 9)))
    source = source_population(2048, radius, initial_slopes)
    smaller = source_population(1024, radius, initial_slopes)
    if not all(np.array_equal(value[:1024], smaller[key]) for key, value in source.items()):
        raise ValueError("Source growth changed the original prefix")
    output.mkdir(parents=True, exist_ok=False)
    drivers = {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in
        (Path(__file__), ROOT/"scripts/ideal_lens_beam.py", ROOT/"scripts/ideal_lens_wall.py")}
    request = dict(request, reference_sha256=sha256(reference_path.read_bytes()).hexdigest(),
        source_schema="ideal-uniform-disk-halton235-v1", source_radius_mm=radius*1e3,
        source_current_a=1e-6, energy_ev=300000., source_counts=[1024, 2048],
        source_boundary="Uniform area disk; inverse CDF r=R sqrt(Halton2), azimuth=2pi Halton3; training=Halton5<0.5",
        input_slopes="Fixed paraxial zero-Larmor entrance matrix; not reinitialized from each nonlinear potential",
        start_m=start, paraxial_image_m=image, end_m=end, focus_window_m=window,
        observer_sha256=drivers, implementation=solver_source_identity(),
        tube_role="Passive absorbing cylinder; prescribed magnetic field is unchanged by its diameter",
        qualification="Synthetic monochromatic zero-emittance source, not an FEG or specimen-wave qualification")
    write_json(output/"request.json", request)
    write_json(output/"field.json", field.to_dict())
    np.savez_compressed(output/"source.npz", **source, combined_population_weights=np.full(2048, 1/2048))
    partition_records, phases, wall_parts = [], [], {float(d): [] for d in request["tube_diameters_mm"]}
    for part in range(2):
        lo, hi = part*1024, (part+1)*1024
        print(f"TRACE {request['name']} original IDs {lo}..{hi-1}", flush=True)
        solution, receipt = propagate(field, source["initial_phase"][lo:hi], energy_ev=300000.,
            start_m=start, end_m=end, sigma_m=sigma,
            step_over_sigma=float(request.get("step_over_sigma", 1/16)))
        phase = solution(planes).T.reshape(len(planes), 1024, 5)
        phases.append(phase)
        np.savez_compressed(output/f"characteristics-{part}.npz", z_m=planes, phase=phase,
            integration_z_m=solution.ts,
            source_id=source["source_id"][lo:hi], kinetic_energy_ev=source["kinetic_energy_ev"][lo:hi],
            historical_population_weights=np.full(1024, 1/1024 if part == 0 else 1/2048))
        receipts = {}
        for diameter in wall_parts:
            wall = classify_cylindrical_wall(solution, diameter*.5e-3,
                source["source_id"][lo:hi], state_width=5)
            arrays = {key: value for key, value in wall.items() if isinstance(value, np.ndarray)}
            np.savez_compressed(output/f"wall-{diameter:g}mm-part{part}.npz", **arrays)
            if not wall["classification_complete"]:
                raise ValueError("Unresolved continuous wall contact; preserved raw characteristics, no physical classification claimed")
            receipts[str(diameter)] = {key: value for key, value in wall.items() if not isinstance(value, np.ndarray)}
            wall_parts[diameter].append(wall)
        partition_records.append(dict(original_id_interval=[lo, hi], transport=receipt, wall_receipts=receipts,
            source_prefix_reused=(part == 1), newly_transported_positive_count=1024,
            initial_phase_sha256=array_hash(source["initial_phase"][lo:hi]),
            retained_phase_sha256=array_hash(phase)))
        del solution
    phase = np.concatenate(phases, axis=1)
    at_image = int(np.flatnonzero(planes == image)[0])
    focus_indexes = np.searchsorted(planes, focus_planes)
    populations = []
    for count in (1024, 2048):
        weights = np.full(count, 1/count)
        p = phase[:, :count]
        fit = spherical_fit(source["initial_phase"][:count], p[at_image], source["training"][:count],
            outward, edge, ref["cs_mm"])
        views = []
        for diameter, partitions in wall_parts.items():
            hit = np.concatenate([row["hit_z_m"] for row in partitions])[:count]
            status = np.concatenate([row["status"] for row in partitions])[:count]
            hit = np.where(np.isnan(hit), np.inf, hit)
            stats = []
            for index in focus_indexes:
                row = phase_statistics(p[index], planes[index] < hit, weights, 1e-6)
                row.update(z_m=float(planes[index]), offset_from_paraxial_nm=float((planes[index]-image)*1e9))
                stats.append(row)
            minimum = int(np.argmin([row.get("d95_nm", np.inf) for row in stats]))
            image_stats = phase_statistics(p[at_image], image < hit, weights, 1e-6)
            blocked = np.isfinite(hit) & (hit <= image)
            loss = float(weights[blocked].sum()*1e-6)
            balance = image_stats["current_a"]+loss-1e-6
            if abs(balance) > 1e-18:
                raise ValueError("Absolute physical current ledger does not close")
            views.append(dict(tube_diameter_mm=diameter, at_paraxial_image=image_stats,
                emitted_positive_count=count, original_packet_current_a=1e-6/count,
                intercepted_before_image=int(blocked.sum()), intercepted_current_a=loss,
                entrance_intercepted=int(np.count_nonzero(status == "outside_at_entrance")),
                downstream_wall_contacts=int(np.count_nonzero(status == "wall_contact")),
                absolute_current_balance_residual_a=balance,
                native_d95_minimum=stats[minimum],
                native_minimum_bounded=0 < minimum < len(stats)-1,
                neighbour_offset_bracket_nm=[stats[max(0, minimum-1)]["offset_from_paraxial_nm"], stats[min(len(stats)-1, minimum+1)]["offset_from_paraxial_nm"]],
                focus_planes=stats, intrinsic_cs_reference_mm=ref["cs_mm"],
                full_declared_pupil_physically_admitted=not bool(blocked.any()),
                sphere_fit_from_clipped_population=False))
        populations.append(dict(original_positive_count=count, positive_weight_count=count, weights_sum=float(weights.sum()),
            source_phase_hash=array_hash(source["initial_phase"][:count]), vacuum_fit=fit, tube_views=views))
    earlier, grown = populations
    comparison = dict(source_prefix_phase_reused_exactly=True, source_prefix_size=1024,
        added_original_positive_count=1024, historical_prefix_raw_rewritten=False,
        normalization="Original 1024 record retains weight 1/1024. New combined 2048 source has weight 1/2048 per original ID; blocked current is not restored.",
        tube_checks=[])
    for a, b in zip(earlier["tube_views"], grown["tube_views"]):
        x, y = a["at_paraxial_image"], b["at_paraxial_image"]
        relative = y["d95_nm"]/x["d95_nm"]-1 if "d95_nm" in x and "d95_nm" in y else None
        comparison["tube_checks"].append(dict(tube_diameter_mm=a["tube_diameter_mm"], relative_d95_change=relative,
            d95_count_gate_passed=relative is not None and abs(relative) <= .01))
    comparison["relative_cs_fit_change"] = grown["vacuum_fit"]["cs_fit_mm"]/earlier["vacuum_fit"]["cs_fit_mm"]-1
    comparison["cs_fit_count_gate_passed"] = abs(comparison["relative_cs_fit_change"]) <= .001
    observed_radius = float(np.linalg.norm(phase[:, :, :2], axis=2).max())
    path_radius = max(float(np.max(row["counterfactual_max_radius_upper_bound_m"]))
        for partitions in wall_parts.values() for row in partitions)
    report = dict(schema="ideal-lens-finite-source-study-v1", request=request,
        intrinsic_cs_mm=ref["cs_mm"], reference_cc_mm=ref["cc_mm"],
        paraxial_image_z_m=image, image_field_fraction=float(np.exp(-.5*(image/sigma)**2)),
        source_radius_mm=radius*1e3, paraxial_edge_slope_mrad=edge*1e3,
        maximum_observed_characteristic_radius_mm=observed_radius*1e3,
        dense_characteristic_radius_upper_bound_mm=path_radius*1e3,
        field_order_evidence=field_order_evidence(peak, sigma, max(radius, path_radius)),
        partitions=partition_records, populations=populations, source_count_comparison=comparison,
        observations=dict(z_m=planes.tolist(), focus_indices=focus_indexes.tolist(), image_index=at_image),
        elapsed_s=perf_counter()-started, source_and_characteristics_scope="Retained vacuum characteristics after wall contact are counterfactual. Physical views use only pre-contact prefixes.")
    write_json(output/"report.json", report)
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    write_json(output/"files.json", hashes)
    print(f"DONE {request['name']} {report['elapsed_s']:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", action="append", help="Run only named declared cases")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    for request in json.loads(args.requests.read_text()):
        if args.name and request["name"] not in args.name:
            continue
        if Path(request["name"]).name != request["name"]:
            raise ValueError("Case name must be a local directory name")
        run_case(request, reference, args.reference, args.output/request["name"])


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
