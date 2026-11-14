"""Study intrinsic aberrations of geometry-driven, nonmagnetic copper coils.

The same complete Biot-Savart field transports both nested source partitions.
No Gaussian fit, empirical Cs kick, pole material, thermal qualification or
specimen/wave model is used. Every newly transported partition has 1024
distinct original positive-current particles.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from ideal_lens_beam import array_hash, phase_statistics, propagate, source_population, spherical_fit
from ideal_lens_wall import classify_cylindrical_wall
from temsim.calculation_manifest import solver_source_identity
from temsim.physics.annular_coil_potential import AnnularCoilPotential
from temsim.physics.round_lens_aberration_benchmark import paraxial_focus_reference, third_order_reference


def write_json(path, value):
    def default(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=default,
                                    allow_nan=False) + "\n", encoding="utf-8")


def make_field(request, quadrature_order=None):
    return AnnularCoilPotential(source_id="ideal-copper-coil",
        inner_radius_m=float(request["inner_diameter_mm"]) * .0005,
        outer_radius_m=float(request["outer_diameter_mm"]) * .0005,
        length_m=float(request["length_mm"]) * .001,
        ampere_turns=float(request["ampere_turns"]),
        quadrature_order=int(quadrature_order or request.get("quadrature_order", 12)))


def transport_implementation():
    paths = ("src/temsim/physics/annular_coil_potential.py", "src/temsim/physics/vector_field_transport.py",
             "src/temsim/physics/relativistic_lorentz.py", "src/temsim/physics/lens_field_provider.py",
             "scripts/ideal_lens_beam.py", "scripts/ideal_lens_wall.py")
    return dict(files={name: sha256((ROOT / name).read_bytes()).hexdigest() for name in paths},
                numpy=np.__version__, scipy=scipy.__version__)


def reference_for(request):
    """Independent order-by-order Lorentz ODE, not a fit to finite particles."""
    start, search = float(request.get("start_m", -.2)), float(request.get("focus_search_end_m", 5.))
    step = min(float(request["inner_diameter_mm"]) * .0005,
               float(request["length_mm"]) * .001) / 8
    result = {}
    for name, order, spacing in (("primary", int(request.get("quadrature_order", 12)), step),
                                 ("quadrature_refined", 24, step),
                                 ("step_refined", 24, step / 2)):
        field = make_field(request, order)
        print(f"REFERENCE {request['name']} {name}", flush=True)
        focus = paraxial_focus_reference(field, energy_ev=300000., z_start_m=start,
            focus_search_end_m=search, maximum_step_m=spacing)
        coefficients = third_order_reference(field, energy_ev=300000., z_start_m=start,
            focus_search_end_m=search, maximum_step_m=spacing)
        result[name] = dict(focus=focus, coefficients=coefficients, maximum_step_m=spacing,
                            quadrature_order=order, field_digest=field.digest)
    q, s, p = (result[name]["coefficients"] for name in ("quadrature_refined", "step_refined", "primary"))
    result["checks"] = dict(relative_cs_quadrature_change=abs(p["cs_mm"] / q["cs_mm"] - 1),
        relative_cs_step_change=abs(q["cs_mm"] / s["cs_mm"] - 1),
        image_quadrature_difference_m=abs(p["image_z_m"] - q["image_z_m"]),
        image_step_difference_m=abs(q["image_z_m"] - s["image_z_m"]),
        qualification="Numerical sensitivity of the prescribed copper-coil model; not a measured-lens calibration")
    return result


def run_case(request, reference, output, *, resume_transport=False):
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=resume_transport)
    field = make_field(request)
    focus = reference["primary"]["focus"]
    coeff = reference["step_refined"]["coefficients"]
    initial_matrix = np.asarray(focus["initial_slopes_per_position_m1"])
    outward = -np.asarray(focus["image_slopes_per_initial_position_m1"])
    edge = np.tan(float(request.get("edge_angle_mrad", 10.)) * .001)
    radius = edge / abs(float(focus["radial_slope_per_m"]))
    # All primary tubes admit the declared entrance pupil. A second smaller
    # tube is a physical interception experiment, not a change to the field.
    tube = float(request.get("tube_diameter_mm", 5.76)) * .001
    if radius >= field.inner_radius_m:
        raise ValueError("Declared source pupil reaches the current-bearing copper")
    start, image = float(request.get("start_m", -.2)), float(focus["image_z_m"])
    scale = max(field.outer_radius_m, field.length_m / 2)
    window = max(.01 * scale, 2 * abs(coeff["cs_mm"] * .001) * edge**2)
    if image - window <= start:
        raise ValueError("Through-focus interval reaches the source plane")
    end = image + window
    focus_planes = image + np.linspace(-window, window, 65)
    planes = np.unique(np.r_[np.linspace(start, end, 65), focus_planes, image])
    source = source_population(2048, radius, initial_matrix)
    smaller = source_population(1024, radius, initial_matrix)
    if not all(np.array_equal(value[:1024], smaller[key]) for key, value in source.items()):
        raise ValueError("Source growth changed the original source prefix")
    recorded = dict(request, source_schema="ideal-uniform-disk-halton235-v1",
        energy_ev=300000., source_current_a=1e-6, source_counts=[1024, 2048], source_radius_m=radius,
        start_m=start, paraxial_image_m=image, end_m=end, focus_window_m=window,
        integration_length_scale_m=scale,
        integration_note="Legacy helper argument sigma_m is only a step/split scale here; no Gaussian field is used",
        transport_implementation=transport_implementation(),
        implementation=solver_source_identity(), field_digest=field.digest,
        drivers={p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), ROOT / "scripts/ideal_lens_beam.py", ROOT / "scripts/ideal_lens_wall.py")},
        qualification="Ideal monochromatic zero-emittance source; not an installed instrument or coherent probe",
        current_rule="Each original in the N-particle population has 1 microampere/N. Intercepted current is never restored.",
        pupil_rule="Same declared 10 mrad paraxial edge angle across cases, with separately recorded source radii")
    if resume_transport:
        if (output / "report.json").exists():
            raise ValueError("Completed results are immutable; use another output directory")
        original = json.loads((output / "request.json").read_text(encoding="utf-8"))
        if original.get("transport_implementation") != transport_implementation():
            raise ValueError("Retained transport physics or environment changed, or the older record lacks explicit implementation binding")
        if any(original.get(key) != value for key, value in request.items()):
            raise ValueError("Retained transport belongs to different requested inputs")
        if (original["field_digest"] != field.digest
                or json.loads((output / "field.json").read_text()) != json.loads(json.dumps(field.to_dict()))):
            raise ValueError("Retained field does not match the current immutable coil")
        for driver in ("scripts/ideal_lens_beam.py", "scripts/ideal_lens_wall.py"):
            retained_hash = original["drivers"].get(driver, original["drivers"].get(str(Path(driver))))
            if retained_hash != sha256((ROOT / driver).read_bytes()).hexdigest():
                raise ValueError("Transport or continuous-wall implementation changed")
        saved_source = np.load(output / "source.npz")
        if any(not np.array_equal(saved_source[key], value) for key, value in source.items()):
            raise ValueError("Retained source IDs, positions, directions, energies or times changed")
        recorded = original
    else:
        write_json(output / "request.json", recorded)
        write_json(output / "field.json", field.to_dict())
        write_json(output / "reference.json", reference)
        np.savez_compressed(output / "source.npz", **source,
                            combined_population_weights=np.full(2048, 1 / 2048))
    phases, walls, receipts = [], [], []
    for part in range(2):
        lo, hi = 1024 * part, 1024 * (part + 1)
        print(f"TRACE {request['name']} original IDs {lo}..{hi-1}", flush=True)
        if resume_transport:
            saved = np.load(output / f"characteristics-{part}.npz")
            if not np.array_equal(saved["z_m"], planes) or not np.array_equal(saved["source_id"], source["source_id"][lo:hi]):
                raise ValueError("Retained planes or source partition changed")
            phase = saved["phase"]
            wall = dict(np.load(output / f"wall-{part}.npz"))
            wall["classification_complete"] = not bool(wall["unresolved_mask"].any())
            receipt = dict(reused_completed_transport=True, integration_z_m=saved["integration_z_m"],
                observer_recovery="Completed arrays recovered after a statistics-only failure; no transport rerun",
                original_driver_sha256=original["drivers"].get("scripts/study_copper_coil_aberrations.py",
                    original["drivers"].get(str(Path("scripts/study_copper_coil_aberrations.py")))))
        else:
            solution, receipt = propagate(field, source["initial_phase"][lo:hi], energy_ev=300000.,
                start_m=start, end_m=end, sigma_m=scale,
                step_over_sigma=float(request.get("step_over_scale", 1 / 8)))
            phase = solution(planes).T.reshape(len(planes), 1024, 5)
            wall = classify_cylindrical_wall(solution, tube / 2, source["source_id"][lo:hi], state_width=5)
        if not wall["classification_complete"]:
            raise ValueError("Unresolved continuous wall contact; no physical current result claimed")
        if not resume_transport:
            np.savez_compressed(output / f"characteristics-{part}.npz", z_m=planes, phase=phase,
                integration_z_m=solution.ts, source_id=source["source_id"][lo:hi],
                historical_population_weights=np.full(1024, 1 / 1024 if part == 0 else 1 / 2048))
            np.savez_compressed(output / f"wall-{part}.npz",
                **{key: value for key, value in wall.items() if isinstance(value, np.ndarray)})
        receipts.append(dict(original_id_interval=[lo, hi], transport=receipt,
            newly_transported_original_positive_count=1024,
            source_phase_sha256=array_hash(source["initial_phase"][lo:hi]),
            retained_phase_sha256=array_hash(phase),
            wall={key: value for key, value in wall.items() if not isinstance(value, np.ndarray)}))
        phases.append(phase)
        walls.append(wall)
        write_json(output / f"partition-{part}-receipt.json", receipts[-1])
        if not resume_transport:
            del solution
    phase = np.concatenate(phases, axis=1)
    # The wall classifier deliberately retains NaN for a ray that never hit.
    # Convert only the observer's comparison coordinate; preserve raw arrays.
    hit = np.concatenate([np.where(wall["status"] == "survived", np.inf, wall["hit_z_m"]) for wall in walls])
    if np.isnan(hit).any():
        raise ValueError("A nonsurviving ray has no classified wall-hit coordinate")
    at_image = int(np.flatnonzero(planes == image)[0])
    populations = []
    for count in (1024, 2048):
        weights = np.full(count, 1 / count)
        fit = spherical_fit(source["initial_phase"][:count], phase[at_image, :count],
            source["training"][:count], outward, edge, coeff["cs_mm"])
        stats = phase_statistics(phase[at_image, :count], image < hit[:count], weights, 1e-6)
        blocked_current = float(weights[hit[:count] <= image].sum() * 1e-6)
        balance = stats["current_a"] + blocked_current - 1e-6
        if abs(balance) > 1e-18:
            raise ValueError("Absolute physical current ledger does not close")
        views = []
        for index in np.searchsorted(planes, focus_planes):
            views.append(dict(z_m=float(planes[index]), offset_from_paraxial_nm=float((planes[index]-image)*1e9),
                **phase_statistics(phase[index, :count], planes[index] < hit[:count], weights, 1e-6)))
        minimum = min(range(len(views)), key=lambda i: views[i].get("d95_nm", np.inf))
        populations.append(dict(original_positive_count=count, effective_at_image=stats["effective"],
            surviving_at_image=stats["surviving_positive"], at_paraxial_image=stats, vacuum_fit=fit,
            intercepted_current_a=blocked_current, absolute_current_balance_residual_a=balance,
            lowest_sampled_d95=views[minimum], minimum_is_interior=0 < minimum < len(views)-1,
            sampled_focus_planes=views,
            minimum_note="Lowest of 65 declared planes, not a continuous optimum"))
    p, q = populations
    comparison = dict(source_prefix_preserved_exactly=True, original_prefix_count=1024,
        appended_original_positive_count=1024, earlier_transport_reused_without_rewriting=True,
        relative_d95_change=(q["at_paraxial_image"]["d95_nm"] / p["at_paraxial_image"]["d95_nm"] - 1
                            if "d95_nm" in p["at_paraxial_image"] and "d95_nm" in q["at_paraxial_image"] else None),
        relative_cs_fit_change=q["vacuum_fit"]["cs_fit_mm"] / p["vacuum_fit"]["cs_fit_mm"] - 1)
    # Compare the actually represented beam cylinder against a higher coil
    # quadrature. This is a sampled numerical check, not an error certificate.
    maximum_radius = float(np.linalg.norm(phase[:, :, :2], axis=2).max())
    zz, rr = np.meshgrid(np.linspace(-4 * scale, 4 * scale, 65), np.linspace(0., maximum_radius, 9))
    points = np.column_stack((rr.ravel(), np.zeros(rr.size), zz.ravel()))
    refined = make_field(request, 24)
    b, br = field.field_at_global_positions_t(points), refined.field_at_global_positions_t(points)
    peak = abs(float(field.axis_bz_t(0.)))
    field_check = dict(sampled_cylinder_radius_m=maximum_radius, grid_shape=[9, 65],
        primary_quadrature_order=field.quadrature_order, refined_quadrature_order=24,
        max_absolute_difference_t=float(np.linalg.norm(b - br, axis=1).max()),
        max_difference_over_axis_peak=float(np.linalg.norm(b - br, axis=1).max() / peak))
    report = dict(schema="copper-coil-intrinsic-aberration-study-v1", request=recorded,
        statistics_driver_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        completed_transport_reused_after_observer_failure=resume_transport,
        on_axis_centre_field_t=peak, intrinsic_cs_mm=coeff["cs_mm"], reference_cc_mm=coeff["cc_mm"],
        populations=populations, source_count_comparison=comparison, quadrature_check=field_check,
        partitions=receipts, elapsed_seconds=perf_counter()-started,
        timing_scope="Statistics-only recovery; transport duration was not retained" if resume_transport else "Complete case including transport and observers",
        distinctions={"RMS": "Current-weighted radial RMS about the centroid",
            "D95": "Diameter containing 95% of the current actually surviving at that plane",
            "alpha95": "95% current angular radius about the mean direction, arctan converted",
            "intrinsic_Cs": "Independent third-order perturbation coefficient of the unblocked lens; wall clipping does not redefine it"})
    write_json(output / "report.json", report)
    print(f"DONE {request['name']} Cs={coeff['cs_mm']:.9g} mm D95={q['at_paraxial_image'].get('d95_nm', 'NO_TRANSMISSION')} nm", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", action="append")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--resume-transport", action="store_true",
                        help="Reanalyse complete retained partitions without rewriting or rerunning transport")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for request in json.loads(args.requests.read_text(encoding="utf-8")):
        name = request["name"]
        if args.name and name not in args.name:
            continue
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Case name must be a local directory name")
        reference_path = args.output / f"reference-{name}.json"
        if reference_path.exists():
            record = json.loads(reference_path.read_text(encoding="utf-8"))
            if record["request"] != request:
                raise ValueError("Existing reference belongs to different frozen inputs")
            reference = record["reference"]
        else:
            reference = reference_for(request)
            write_json(reference_path, dict(request=request, reference=reference))
        if not args.reference_only:
            run_case(request, reference, args.output / name, resume_transport=args.resume_transport)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
