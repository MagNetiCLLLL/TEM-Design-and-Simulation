"""Validate a frozen joint-probe candidate using original-current particles.

This command measures the exact specimen upper surface. It never fits controls,
refocuses a candidate, installs a preset, or substitutes a wave-optical probe.
Two integration steps establish a bounded step comparison, not source-count
convergence. Use a new output directory for every independent observation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path


SCHEMA = "temsim-joint-probe-validation-v1"
PHYSICAL_LIMITS = dict(target_alpha95_mrad=30., alpha95_relative_tolerance=.01,
    search_alpha95_relative_target=.001, maximum_d95_nm=.7,
    focus_tolerance_nm=1., minimum_surviving_and_effective_rays=16,
    c2_aperture_diameter_um=100., minimum_current_a=None, maximum_rms_nm=None)
NUMERICAL_LIMITS = dict(alpha95_relative=.0025, d95_relative=.02,
    rms_relative=.02, current_relative=.02, focus_absolute_nm=.25)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if hasattr(value, "tolist"):
        return _clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_clean(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _restore_exact(snapshot):
    """Restore supported immutable input graphs without pickle/deepcopy."""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = snapshot.restore()
    if capture_instrument_snapshot(state).digest != snapshot.digest:
        raise ValueError("Restoring the complete solver snapshot changed its exact input graph")
    return state


def load_candidate(path):
    """Restore the complete solver snapshot, checking its paired public state.

    Plain standalone state JSON remains an explicit input format. It is not a
    complete graph: solver candidate directories and pointers use inputs.json
    so dynamic quality/source attributes and derived planes remain bound.
    """
    from temsim.optics.column import State
    from temsim.instrument_snapshot import InstrumentSnapshot
    path = Path(path).resolve()
    pointer_identity = None
    if path.is_dir():
        path = path / ("inputs.json" if (path / "inputs.json").is_file()
                       else "state.json" if (path / "state.json").is_file() else "best.json")
    entry_bytes = path.read_bytes()
    payload = json.loads(entry_bytes.decode("utf-8"))
    if isinstance(payload, dict) and "directory" in payload:
        directory = Path(payload["directory"])
        resolved = (path.parent / directory).resolve()
        if directory.is_absolute() or not resolved.is_relative_to(path.parent):
            raise ValueError("Best-candidate pointer must remain within its run directory")
        hashes = payload.get("files_sha256")
        if not isinstance(hashes, dict) or not {"state.json", "inputs.json"} <= hashes.keys():
            raise ValueError("Solver pointer must pin the complete snapshot and paired state hashes")
        for name, expected in hashes.items():
            if not isinstance(name, str) or Path(name).name != name or not isinstance(expected, str):
                raise ValueError("Invalid solver-candidate file hash entry")
            file = resolved / name
            digest = sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Solver-candidate file checksum mismatch: {name}")
        pointer_identity = dict(path=str(path), file_sha256=sha256(entry_bytes).hexdigest(),
            verified_files_sha256=hashes)
        path = resolved / "inputs.json"
    elif path.name == "candidate.json":
        path = path.with_name("inputs.json")
    elif path.name == "state.json" and path.with_name("inputs.json").is_file():
        path = path.with_name("inputs.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    snapshot_identity = None
    if isinstance(payload, dict) and {"graph", "external_inputs", "implementation", "digest"} <= payload.keys():
        snapshot = InstrumentSnapshot.from_dict(payload)
        state = _restore_exact(snapshot)
        snapshot_identity = snapshot.digest
        paired = path.with_name("state.json")
        if paired != path and paired.is_file():
            paired_payload = json.loads(paired.read_text(encoding="utf-8"))
            # JSON represents tuple-valued crystallographic coordinates as
            # lists. Compare the exact serialized values, without nonfinite
            # sanitization or numeric/boolean coercion.
            if _digest(state.to_dict()) != _digest(paired_payload):
                raise ValueError("Complete solver snapshot disagrees with its paired state settings")
    else:
        state = State.from_dict(payload)
        if _digest(state.to_dict()) != _digest(payload):
            raise ValueError("Restoring the candidate changed its saved physical settings")
    return state, dict(path=str(path.resolve()), file_sha256=sha256(path.read_bytes()).hexdigest(),
        restoration_kind="complete_snapshot" if snapshot_identity is not None else "standalone_serialized_state",
        snapshot_identity=snapshot_identity, solver_pointer=pointer_identity)


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value))


def physical_gates(observation, *, surface_z_mm):
    """Independently check saved observables; absent evidence cannot pass."""
    surface = observation.get("surface", {})
    stats = surface.get("statistics", {})
    topology = observation.get("topology", {})
    corrector = observation.get("corrector", {})
    chain = observation.get("crossover_chain", {})
    # The condenser summary can return before evaluating direct focus when an
    # upstream interval is unresolved. The whole-capture chain owns this proof.
    direct_focus = chain.get("direct_focus")
    direct_focus = direct_focus if isinstance(direct_focus, dict) else {}
    direct_focus_complete = (
        direct_focus.get("schema") == "direct-focus-finite-region-topology-v2"
        and direct_focus.get("requirement") == "direct_focus_without_extra_pre_specimen_waist"
        and isinstance(direct_focus.get("failures"), list)
        and ((direct_focus.get("status") == "PASS"
              and direct_focus.get("passed") is True and direct_focus["failures"] == [])
             or (direct_focus.get("status") == "FAIL"
                 and direct_focus.get("passed") is False and bool(direct_focus["failures"]))))
    def scaled(mapping, name, scale=1.):
        value = mapping.get(name)
        return value * scale if _number(value) else None
    metrics = dict(alpha95_mrad=scaled(stats, "convergence_95_rad", 1e3),
        d95_nm=scaled(stats, "radius_95_m", 2e9),
        rms_radius_nm=scaled(stats, "radius_rms_m", 1e9),
        local_focus_offset_nm=scaled(surface, "local_waist_offset_nm"),
        covariance_focus_offset_nm=scaled(stats, "waist_offset_m", 1e9),
        current_a=observation.get("sample_current_a"),
        surviving_rays=stats.get("surviving_rays"),
        effective_rays=surface.get("effective_rays"),
        source_current_fraction=stats.get("surviving_fraction"))
    checks = {}
    checks["exact_upper_surface"] = (surface.get("surface_z_mm") == surface_z_mm
        and _number(surface_z_mm))
    angle = metrics["alpha95_mrad"]
    checks["alpha95"] = _number(angle) and abs(angle / 30. - 1.) <= .01
    diameter = metrics["d95_nm"]
    checks["d95"] = _number(diameter) and 0 <= diameter < .7
    for key in ("local_focus_offset_nm", "covariance_focus_offset_nm"):
        checks[key] = _number(metrics[key]) and abs(metrics[key]) <= 1.
    curvature = surface.get("variance_second_derivative")
    checks["positive_variance_curvature"] = _number(curvature) and curvature > 0
    checks["rms_observed"] = _number(metrics["rms_radius_nm"]) and metrics["rms_radius_nm"] >= 0
    checks["positive_original_current"] = (_number(metrics["current_a"]) and metrics["current_a"] > 0
        and _number(metrics["source_current_fraction"]) and 0 < metrics["source_current_fraction"] <= 1.)
    checks["sample_support"] = (_number(metrics["surviving_rays"]) and metrics["surviving_rays"] >= 16
        and _number(metrics["effective_rays"]) and metrics["effective_rays"] >= 16 * (1 - 32 * 2.220446049250313e-16))
    checks["source_identity"] = isinstance(surface.get("source_population_identity"), str) and bool(surface["source_population_identity"])
    checks["condenser_topology"] = topology.get("passed") is True
    checks["direct_focus_topology"] = direct_focus_complete and direct_focus.get("status") == "PASS"
    checks["corrector_finite_regions"] = corrector.get("waist_detector") == "finite_region" and corrector.get("waist_evidence_passed") is True
    checks["required_whole_capture_topology"] = chain.get("required_topology_satisfied_at_sampling") is True
    checks["loss_attribution"] = chain.get("unattributed_population_change_count") == 0
    checks["observer_acceptance"] = observation.get("fixed_sampling_pass") is True and observation.get("failures") == []
    unresolved = any(not checks[key] for key in ("source_identity", "corrector_finite_regions",
        "required_whole_capture_topology", "loss_attribution")) or topology.get("status") == "UNRESOLVED" or not direct_focus_complete
    return dict(status="PASS" if all(checks.values()) else "UNRESOLVED" if unresolved else "FAIL",
        passed=all(checks.values()), checks=checks, metrics=metrics,
        failed_checks=[key for key, passed in checks.items() if not passed],
        direct_focus_evidence=dict(source="crossover_chain.direct_focus",
            schema=direct_focus.get("schema"), status=direct_focus.get("status", "NOT_EVALUATED"),
            complete=direct_focus_complete, passed=checks["direct_focus_topology"],
            failures=direct_focus.get("failures")),
        plane_kind="specimen_upper_surface", plane_z_mm=surface_z_mm,
        physical_pupil_edge=dict(status="NOT_EVALUATED", role="Independent zero-current boundary diagnostic; alpha95 is a current quantile"),
        focus_scope="Positive local radial variance curvature and radial covariance focus; finite source-image topology is checked separately")


def compare_observations(coarse, fine, *, require_same_source=True):
    """Compare immutable inputs and bounded numerical changes independently of physical PASS."""
    reasons = []
    if coarse.get("extra_capture_planes_mm", []) != fine.get("extra_capture_planes_mm", []):
        reasons.append("Different explicit extra capture planes")
    for row in (coarse, fine):
        if row.get("execution_status") != "COMPLETE":
            reasons.append("Both observations must execute completely")
    for key in ("fixed_candidate_identity", "solver_identity", "external_input_identity"):
        if not coarse.get(key) or coarse.get(key) != fine.get(key):
            reasons.append(f"Missing or differing {key}")
    if require_same_source and (not coarse.get("source_identity") or coarse.get("source_identity") != fine.get("source_identity")):
        reasons.append("Different or missing complete-source identity")
    for key in ("plane_kind", "plane_z_mm"):
        if coarse.get("physical", {}).get(key) is None or coarse.get("physical", {}).get(key) != fine.get("physical", {}).get(key):
            reasons.append(f"Different or missing {key}")
    changes = {}
    left, right = (row.get("physical", {}).get("metrics", {}) for row in (coarse, fine))
    for key, budget in (("alpha95_mrad", .0025), ("d95_nm", .02),
                        ("rms_radius_nm", .02), ("current_a", .02),
                        ("local_focus_offset_nm", .25), ("covariance_focus_offset_nm", .25)):
        a, b = left.get(key), right.get(key)
        absolute = key.endswith("focus_offset_nm")
        difference = abs(a - b) if _number(a) and _number(b) else None
        error = (difference if absolute else difference / abs(b) if b else 0. if difference == 0. else None) if difference is not None else None
        passed = error is not None and error <= budget
        changes[key] = dict(coarse=a, fine=b, absolute_difference=difference,
            error=error, error_kind="absolute_nm" if absolute else "relative_to_finer", limit=budget, passed=passed)
        if not passed:
            reasons.append(f"Missing or out-of-budget {key}")
    return dict(status="PASS" if not reasons else "FAIL", passed=not reasons,
        reasons=reasons, changes=changes,
        scope="Bounded numerical comparison; neither optical settings nor source weights were fitted")


def _candidate_identity(state):
    """Bind the complete input graph; only explicit step/N are normalized."""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    normalized = _restore_exact(capture_instrument_snapshot(state))
    normalized.step_mm = .05
    normalized.electron_gun.emitter.ray_count = 16
    snapshot = capture_instrument_snapshot(normalized).to_dict()
    return _digest({"graph": snapshot["graph"], "external_inputs": snapshot["external_inputs"]})


def validate_state(state, output_directory, *, rays=None, steps=(.05, .025),
                   spacing_mm=1., extra_capture_planes_mm=(), candidate=None, observer=None, audit=None):
    """Run detached, unchanged candidate controls at the explicitly selected discretizations."""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    if observer is None:
        from qualify_finite_field_nanoprobe import observe as observer
    if audit is None:
        from audit_joint_probe_fields import audit
    count = state.electron_gun.emitter.ray_count if rays is None else rays
    if isinstance(count, bool) or not isinstance(count, int) or count < 16:
        raise ValueError("Use at least 16 complete source particles")
    steps = tuple(steps)
    if len(steps) not in (1, 2) or any(not _number(step) or step <= 0 for step in steps):
        raise ValueError("Use one or two finite positive integration steps")
    if len(steps) == 2 and steps[1] != steps[0] / 2:
        raise ValueError("The independent fine step must equal half the coarse step")
    if not _number(spacing_mm) or not 0 < spacing_mm <= 1.:
        raise ValueError("Checkpoint spacing must be positive and at most 1 mm")
    try:
        extra_capture_planes_mm = tuple(extra_capture_planes_mm)
    except TypeError as error:
        raise ValueError("Extra capture planes must be finite physical coordinates") from error
    if any(not _number(value) for value in extra_capture_planes_mm):
        raise ValueError("Extra capture planes must be finite physical coordinates")
    if any(not float(state.electron_gun.exit_plane_z_mm) <= value <= float(state.sample.upper_surface_z_mm)
           for value in extra_capture_planes_mm):
        raise ValueError("Extra capture planes must lie between gun exit and specimen upper surface")
    extra_capture_planes_mm = tuple(sorted(set(map(float, extra_capture_planes_mm))))
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = capture_instrument_snapshot(state)
    report = dict(schema=SCHEMA, utc=datetime.now(timezone.utc).isoformat(), candidate=candidate,
        execution_status="RUNNING", passed=False, physical_limits=PHYSICAL_LIMITS,
        numerical_limits=NUMERICAL_LIMITS, requested_rays=count, steps_mm=steps,
        extra_capture_planes_mm=list(extra_capture_planes_mm),
        original_snapshot_identity=original.digest, fixed_candidate_identity=_candidate_identity(state),
        source_count_convergence=dict(status="NOT_EVALUATED", required_successive_refinements=2,
            note="Regenerate the complete source at each N; freeze every optical control"),
        qualification="NOT_QUALIFIED_SOURCE_COUNT_CONVERGENCE_NOT_ESTABLISHED",
        scope="Classical incident particles at the exact specimen upper surface; no scattering, wave probe, pupil-edge, or driven-scan qualification",
        runs=[], errors=[])
    destination = output / "report.json"
    _write(destination, report)
    try:
        for index, step in enumerate(steps):
            detached = _restore_exact(original)
            detached.step_mm = float(step)
            detached.electron_gun.emitter.ray_count = count
            snapshot = capture_instrument_snapshot(detached)
            row = dict(step_mm=step, rays=count, execution_status="RUNNING",
                extra_capture_planes_mm=list(extra_capture_planes_mm),
                fixed_candidate_identity=_candidate_identity(detached),
                input_snapshot_identity=snapshot.digest, solver_identity=snapshot.to_dict()["implementation"],
                external_input_identity=_digest(snapshot.to_dict()["external_inputs"]))
            report["runs"].append(row)
            _write(destination, report)
            row["field_audit"] = audit(detached)
            if row["field_audit"].get("passed") is not True or row["field_audit"].get("status") != "PASS":
                raise ValueError("C2 aperture and no-added-aberrations audit did not pass")
            prefix = output / f"step-{index + 1}.json"
            extra_options = (dict(extra_capture_planes_mm=extra_capture_planes_mm)
                if extra_capture_planes_mm else {})
            observation = observer(detached, spacing_mm=spacing_mm, target_mrad=30.,
                require_equal_crossover_angles=False, maximum_diameter95_nm=.7, output=prefix, **extra_options)
            if capture_instrument_snapshot(detached).digest != snapshot.digest:
                raise ValueError("Observation changed candidate inputs")
            if observation.get("solver_identity") != row["solver_identity"]:
                raise ValueError("Observation implementation differs from captured inputs")
            if observation.get("snapshot_digest") != snapshot.digest or observation.get("rays") != count or observation.get("step_mm") != step:
                raise ValueError("Observation does not match its exact requested inputs")
            if observation.get("extra_capture_planes_mm", []) != list(extra_capture_planes_mm):
                raise ValueError("Observation does not match its explicit extra capture planes")
            row.update(execution_status="COMPLETE", observation_file=prefix.name,
                source_identity=observation.get("surface", {}).get("source_population_identity"),
                physical=physical_gates(observation, surface_z_mm=float(detached.sample.upper_surface_z_mm)),
                topology=observation.get("topology"), corrector=observation.get("corrector"),
                observer_failures=observation.get("failures"),
                elapsed_s=observation.get("elapsed_s"))
            _write(destination, report)
        report["step_comparison"] = (compare_observations(*report["runs"]) if len(steps) == 2 else
            dict(status="NOT_EVALUATED", passed=False, reasons=["Single-step observation only"]))
        report["fixed_sampling_physical_pass"] = all(row["physical"]["passed"] for row in report["runs"])
        report["passed"] = report["fixed_sampling_physical_pass"] and report["step_comparison"]["passed"]
        report["execution_status"] = "COMPLETE"
    except (Exception, KeyboardInterrupt) as error:
        report["execution_status"] = "CANCELLED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        if report["runs"] and report["runs"][-1]["execution_status"] == "RUNNING":
            report["runs"][-1]["execution_status"] = report["execution_status"]
        report["errors"].append(dict(type=type(error).__name__, message=str(error)))
    finally:
        report["original_state_unchanged"] = capture_instrument_snapshot(state).digest == original.digest
        if not report["original_state_unchanged"]:
            report.update(passed=False, execution_status="FAILED")
            report["errors"].append(dict(type="StateMutation", message="The original candidate inputs changed"))
        _write(destination, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True,
        help="Full state.json, snapshot inputs.json, candidate directory, candidate.json or best.json")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--rays", type=int, help="Fresh complete source count; default preserves the candidate count")
    parser.add_argument("--step-mm", type=float, default=.05)
    parser.add_argument("--single-step", action="store_true", help="Observe one step without numerical acceptance")
    parser.add_argument("--spacing-mm", type=float, default=1.)
    parser.add_argument("--extra-capture-z-mm", type=float, action="append", default=[],
        help="Additional exact physical checkpoint in mm; repeat for multiple coordinates")
    args = parser.parse_args(argv)
    try:
        state, candidate = load_candidate(args.candidate)
        steps = (args.step_mm,) if args.single_step else (args.step_mm, args.step_mm / 2)
        report = validate_state(state, args.output, rays=args.rays, steps=steps,
            spacing_mm=args.spacing_mm, extra_capture_planes_mm=args.extra_capture_z_mm, candidate=candidate)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Validation failed before execution: {error}\n")
    print(json.dumps(dict(execution=report["execution_status"], passed=report["passed"],
        qualification=report["qualification"], report=str(args.output.resolve() / "report.json"))))
    if report["execution_status"] != "COMPLETE":
        return 2
    return 0 if report["passed"] or args.single_step else 1


if __name__ == "__main__":
    raise SystemExit(main())
