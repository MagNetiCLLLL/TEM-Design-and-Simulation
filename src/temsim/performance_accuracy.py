"""Read-only performance and accuracy evidence for completed calculations.

No source is sampled, no field is propagated, and no lens is adjusted here.
Repeatability is distinct from numerical convergence and physical qualification.
All scalar limits are explicit; raw particle arrays always require exact identity.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from temsim.immutable_json import json_digest, thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot, physical_identity_graph


OBSERVER_IDENTITY = "temsim-performance-accuracy-observer-v3"
OBSERVATION_SCHEMA = "temsim-performance-accuracy-observation-v1"
REPORT_SCHEMA = "temsim-performance-accuracy-report-v1"
METRICS = ("alpha95_mrad", "d95_nm", "rms_radius_nm", "current_a")
_PHASE = ("x_m", "y_m", "tx_rad", "ty_rad")
_NUMERICAL_NAMES = frozenset({
    "step_mm", "trace_step_mm", "drift_step_mm", "history_step_mm",
    "acceleration_backend", "ray_count", "maximum_modes", "energy_samples",
    "mode_tail_tolerance", "radial_nodes", "axial_nodes", "apex_cells_per_radius",
    "outer_radius_factor", "wave_grid_size", "wave_slice_thickness_angstrom",
    "wave_frozen_phonon_configurations", "wave_frozen_phonon_seed",
})


def _safe(value):
    """Strict JSON: unknown/nonfinite scalar data become null, never a pass."""
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _number(value, *, nonnegative=False):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and (not nonnegative or number >= 0) else None


def _array(value):
    if value is None:
        raise ValueError("Missing retained array")
    array = np.asarray(value)
    if array.dtype.kind not in "biufc" or array.dtype.hasobject:
        raise ValueError("Expected a numeric retained array")
    return array


def array_fingerprint(value):
    """Hash exact dtype, shape and bytes, including stopped/unknown entries."""
    array = _array(value)
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return dict(sha256=digest.hexdigest(), shape=list(array.shape), dtype=array.dtype.str,
                bytes=int(array.nbytes), nonfinite_count=int(np.count_nonzero(~np.isfinite(array))))


def _validate_ledger_array(value, shape, live, name):
    array = _array(value)
    if (array.shape != shape or array.dtype.kind not in "fiu" or np.isinf(array).any()
            or np.any(~np.isfinite(array[live]))
            or (name == "kinetic_energy_ev" and np.any(array[np.isfinite(array)] <= 0))
            or (name == "flight_time_s" and np.any(array[np.isfinite(array)] < 0))):
        raise ValueError(f"Invalid retained {name}: require matching shape and physical finite values on live particles")
    return array


def _require_equal_arrays(first, second, label):
    first, second = _array(first), _array(second)
    if (first.dtype != second.dtype or first.shape != second.shape
            or not np.array_equal(first, second, equal_nan=True)):
        raise ValueError(f"Retained {label} arrays differ in dtype, shape or exact values")


def _validated_incident_section(simulation, branch):
    """Read an executed section without admitting it to a new calculation."""
    saved = getattr(simulation, "section_checkpoint", None)
    if saved is None:
        return None
    from temsim.physics.particle_sections import validate_section_checkpoint
    validate_section_checkpoint(saved)
    section = saved.segments[0]
    column = section.branch
    suffix = np.asarray(branch.z) >= float(column.z[0])
    _require_equal_arrays(np.asarray(branch.z)[suffix], column.z, "section planes")
    for name in ("x", "y", "tx", "ty", "flight_time_s", "kinetic_energy_ev"):
        _require_equal_arrays(np.asarray(getattr(branch, name))[suffix], getattr(column, name),
                              f"section {name}")
    for name in ("alive", "blocked_z", "ray_weight", "source_ray_id"):
        _require_equal_arrays(getattr(branch, name), getattr(column, name), f"section {name}")
    if tuple(branch.blocked_key) != tuple(column.blocked_key):
        raise ValueError("Section physical stop labels differ from the incident history")
    gun = simulation.gun_trace
    for name in (*_PHASE, "weight", "ray_id", "alive", "energy_offset_ev", "flight_time_s"):
        _require_equal_arrays(getattr(gun.exit_bundle, name), getattr(saved.gun_trace.exit_bundle, name),
                              f"section source {name}")
    cp = simulation.incident_checkpoints
    for name in ("z_mm", *_PHASE, "flight_time_s", "kinetic_energy_ev"):
        _require_equal_arrays(getattr(cp, name), getattr(section.checkpoints, name), f"section checkpoint {name}")
    _require_equal_arrays(simulation.incident_plan.z_mm, section.plan.z_mm, "section plan")
    return section


def _proven_gun_prefix(simulation, branch):
    """Prove the exact displayed gun prefix whose energy was not retained.

    Production concatenates unquantized gun history and the executed column.
    Coordinates below a nominal boundary alone are insufficient evidence.
    """
    gun = simulation.gun_trace
    plan, cp = simulation.incident_plan, simulation.incident_checkpoints
    start = _number(np.asarray(plan.z_mm)[0])
    gun_z, z = _array(gun.z_mm), _array(branch.z)
    if (start is None or gun_z.ndim != 1 or not len(gun_z)
            or not np.isfinite(gun_z).all() or np.any(np.diff(gun_z) <= 0)
            or gun_z[-1] != start or np.asarray(cp.z_mm)[0] != start):
        raise ValueError("Unknown gun energy requires matching executed gun, plan and checkpoint boundaries")
    prefix, gun_prefix = z < start, gun_z < start
    if not prefix.any() or not np.any(z == start):
        raise ValueError("Unknown gun energy requires an exact retained gun-to-column boundary")
    _require_equal_arrays(z[prefix], gun_z[gun_prefix], "gun prefix planes")
    for name, gun_name in (("x", "x_m"), ("y", "y_m"), ("tx", "tx_rad"),
                          ("ty", "ty_rad"), ("flight_time_s", "flight_time_s")):
        _require_equal_arrays(np.asarray(getattr(branch, name))[prefix],
                              np.asarray(getattr(gun, gun_name))[gun_prefix], f"gun prefix {name}")
    return prefix, dict(status="PARTIAL_RETAINED", column_start_z_mm=start,
        unknown_prefix_rows=int(np.count_nonzero(prefix)),
        unknown_prefix_range_mm=[float(z[prefix][0]), float(z[prefix][-1])],
        proof="Exact dtype and value match to executed gun history; exact column plan/checkpoint boundary",
        scope="Gun-prefix kinetic energies were not retained; no interpolation or nominal-energy substitution")


def _input_identity(state, result):
    identity = dict(status="UNRESOLVED", observer=OBSERVER_IDENTITY,
                    inputs_digest=None, physical_inputs_digest=None,
                    numerical_inputs_digest=None, implementation=None)
    try:
        snapshot = capture_instrument_snapshot(state)
        graph = thaw_json(physical_identity_graph(snapshot.graph))
        external = thaw_json(snapshot.external_inputs)
        inputs = dict(graph=graph, external_inputs=external)
        physical = deepcopy(graph)
        numerical = {}
        # Only explicitly named numerical controls are separated. Unrecognized
        # fields remain in the physical identity, conservatively preventing reuse.
        for index, node in enumerate(physical.get("nodes", ())):
            attributes = node.get("attributes", {})
            for name in tuple(attributes):
                if name in _NUMERICAL_NAMES:
                    numerical[f"{index}:{node.get('type', '')}:{name}"] = attributes[name]
                    attributes[name] = {"numerical_control": name}
        identity.update(status="CAPTURED", inputs_digest=json_digest(inputs),
            physical_inputs_digest=json_digest(dict(graph=physical, external_inputs=external)),
            numerical_inputs_digest=json_digest(numerical), numerical_inputs=numerical,
            observer_implementation=snapshot.implementation,
            physical_identity_scope="All captured inputs except the listed numerical controls; unknown fields remain bound",
            full_input_identity_scope="Physical and numerical inputs plus external contents; implementation recorded separately")
        captured = getattr(getattr(result, "calculation_manifest", None), "instrument_snapshot", None)
        retained_cp = getattr(getattr(result, "simulation", None), "incident_checkpoints", None)
        retained_code = getattr(getattr(retained_cp, "unified_state", None), "implementation_digest", None)
        identity.update(implementation=None, implementation_basis="NOT_RECORDED")
        if captured is not None:
            expected = json_digest(dict(graph=physical_identity_graph(captured.graph),
                                        external_inputs=captured.external_inputs))
            identity["manifest_inputs_match"] = expected == identity["inputs_digest"]
            identity.update(implementation=captured.implementation,
                executed_implementation=captured.implementation, implementation_basis="Captured calculation manifest")
            if expected != identity["inputs_digest"]:
                identity.update(status="MISMATCH", reason="Supplied inputs differ from the captured calculation manifest")
            if retained_code and retained_code != captured.implementation:
                identity.update(status="MISMATCH", reason="Manifest and retained checkpoint implementation identities differ")
        elif retained_code:
            identity.update(implementation=retained_code, executed_implementation=retained_code,
                            implementation_basis="Executed unified incident checkpoint")
        if any(not row.get("sha256") for row in external):
            identity.update(status="UNRESOLVED", reason="An external input has no captured content identity")
    except (AttributeError, KeyError, TypeError, ValueError, OSError) as error:
        identity["reason"] = str(error)
    return identity


def _performance(result):
    cache_hit = bool(getattr(result, "cache_hit", False))
    supplied = getattr(result, "performance", None) or {}
    calculated = sorted(getattr(result, "calculated_products", ()))
    reused = sorted(getattr(result, "reused_products", ()))
    if cache_hit:
        reused = sorted(set(reused).union(calculated))
        calculated = []
    stages = []
    backends = []
    if not cache_hit:
        for row in supplied.get("stages", ()):
            if isinstance(row, Mapping) and row.get("stage"):
                seconds = _number(row.get("seconds"), nonnegative=True)
                if seconds is not None:
                    stages.append(dict(stage=str(row["stage"]), seconds=seconds))
        backends = [_safe(row) for row in supplied.get("backend_stages", ()) if isinstance(row, Mapping)]
    return dict(cache_hit=cache_hit, calculated_products=calculated, reused_products=reused,
        stages=stages, backend_stages=backends,
        pipeline_seconds=None if cache_hit else _number(supplied.get("pipeline_seconds"), nonnegative=True),
        timing_status="REUSED_NO_NEW_PHYSICS" if cache_hit else "RECORDED" if stages else "NOT_RECORDED",
        timing_scope="Current call only; inherited cached product times are excluded; caller measures wall time separately",
        inherited_timing_ignored=cache_hit and bool(supplied),
        last_cuda_column_execution=(_safe(supplied.get("last_cuda_column_execution"))
            if not cache_hit and "column" in calculated and "column" not in reused else None))


def _population(simulation):
    report = dict(status="UNRESOLVED", issues=[], fingerprints={}, source_current_denominator=None)
    branch = getattr(simulation, "incident", None)
    source = getattr(getattr(simulation, "gun_trace", None), "exit_bundle", None)
    try:
        weight = _array(getattr(branch, "ray_weight", None))
        ids = _array(getattr(branch, "source_ray_id", None))
        alive = _array(getattr(branch, "alive", None))
        z = _array(getattr(branch, "z", None))
        phase = [_array(getattr(branch, name, None)) for name in ("x", "y", "tx", "ty")]
        blocked = _array(getattr(branch, "blocked_z", None))
        if (weight.ndim != 1 or weight.dtype.kind not in "fiu" or not np.isfinite(weight).all()
                or np.any(weight < 0) or not math.isfinite(float(weight.sum())) or weight.sum() <= 0
                or ids.shape != weight.shape or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or len(np.unique(ids)) != len(ids) or alive.shape != weight.shape or alive.dtype.kind != "b"
                or z.ndim != 1 or len(z) < 1 or not np.isfinite(z).all() or np.any(np.diff(z) <= 0)
                or blocked.shape != weight.shape or blocked.dtype.kind not in "fiu" or np.isinf(blocked).any()
                or any(a.shape != (len(z), len(weight)) or a.dtype.kind not in "fiu" for a in phase)):
            raise ValueError("Invalid incident particle shapes, identities, weights, masks or planes")
        live = (weight > 0)[None, :] & ((np.isnan(blocked)[None, :] & alive[None, :])
                                      | (z[:, None] < blocked[None, :]))
        if any(np.any(~np.isfinite(a[live])) for a in phase):
            raise ValueError("Nonfinite positive-current phase before a physical stop")
        arrays = {"source_ids": ids, "weights": weight, "terminal_alive": alive,
                  "z_mm": z, "blocked_z_mm": blocked, "observed_alive": live}
        arrays.update(dict(zip(_PHASE, phase)))
        section = _validated_incident_section(simulation, branch)
        for name in ("flight_time_s", "kinetic_energy_ev"):
            values = getattr(branch, name, None)
            if values is not None:
                required = live
                if (name == "kinetic_energy_ev" and np.shape(values) == live.shape
                        and np.any(np.isnan(np.asarray(values)[live]))):
                    prefix, coverage = _proven_gun_prefix(simulation, branch)
                    required = live & ~prefix[:, None]
                    report["energy_coverage"] = coverage
                    arrays["kinetic_energy_required"] = required
                arrays[name] = _validate_ledger_array(values, live.shape, required, name)
        report["fingerprints"] = {name: array_fingerprint(a) for name, a in arrays.items()}
        blocked_keys = getattr(branch, "blocked_key", None)
        if (not isinstance(blocked_keys, (list, tuple)) or len(blocked_keys) != len(weight)
                or any(value is not None and not isinstance(value, str) for value in blocked_keys)):
            raise ValueError("Missing or invalid per-particle physical stop labels")
        report["fingerprints"]["blocked_keys"] = dict(sha256=json_digest(list(blocked_keys)),
                                                      count=len(blocked_keys), encoding="exact-json")
        report.update(ray_count=len(weight), surviving_positive_current_rays=int(np.sum(alive & (weight > 0))),
            source_current_denominator=float(weight.sum()), surviving_weight=float(weight[alive].sum()),
            source_identity_basis="Executed gun exit IDs and original absolute weights; no renormalization")
        if source is None:
            raise ValueError("Executed gun source is unavailable; source identity cannot be certified")
        source_arrays = {name: _array(getattr(source, name, None))
                         for name in (*_PHASE, "weight", "ray_id", "alive", "energy_offset_ev")}
        if any(a.shape != weight.shape for a in source_arrays.values()):
            raise ValueError("Executed gun source population does not match the incident columns")
        if (not np.array_equal(ids, source_arrays["ray_id"])
                or not np.array_equal(weight, source_arrays["weight"])):
            raise ValueError("Incident source IDs or original weights differ from the executed gun")
        if source_arrays["alive"].dtype.kind != "b":
            raise ValueError("Executed source mask must be boolean")
        if np.any(alive & ~source_arrays["alive"] & (weight > 0)):
            raise ValueError("A stopped source particle reappears at the incident endpoint")
        source_live = source_arrays["alive"] & (weight > 0)
        endpoint_live = source_live & (np.isnan(blocked) | (z[-1] < blocked))
        if np.any((alive != endpoint_live) & (weight > 0)):
            raise ValueError("Incident endpoint survival disagrees with the source and physical stop ledger")
        if any(np.any(~np.isfinite(source_arrays[key][source_live])) for key in (*_PHASE, "energy_offset_ev")):
            raise ValueError("Executed current-carrying gun source is nonfinite")
        for name in ("flight_time_s", "kinetic_energy_ev"):
            values = getattr(source, name, None)
            if values is not None:
                source_arrays[name] = _validate_ledger_array(values, weight.shape, source_live, name)
        report["source_fingerprints"] = {key: array_fingerprint(a) for key, a in source_arrays.items()}
        cp = getattr(simulation, "incident_checkpoints", None)
        report["checkpoint_fingerprints"] = {}
        if cp is not None:
            checkpoint_arrays = {key: _array(getattr(cp, key)) for key in ("z_mm", *_PHASE)}
            cp_z = checkpoint_arrays["z_mm"]
            if (cp_z.ndim != 1 or cp_z.dtype.kind not in "fiu" or not len(cp_z) or not np.isfinite(cp_z).all()
                    or np.any(np.diff(cp_z) <= 0)):
                raise ValueError("Invalid retained checkpoint planes")
            cp_live = source_live[None, :] & (np.isnan(blocked)[None, :] | (cp_z[:, None] < blocked[None, :]))
            if any(checkpoint_arrays[name].shape != cp_live.shape or checkpoint_arrays[name].dtype.kind not in "fiu"
                   or np.any(~np.isfinite(checkpoint_arrays[name][cp_live])) for name in _PHASE):
                raise ValueError("Invalid retained checkpoint phase arrays")
            for name in ("flight_time_s", "kinetic_energy_ev"):
                values = getattr(cp, name, None)
                if values is not None:
                    checkpoint_arrays[name] = _validate_ledger_array(values, cp_live.shape, cp_live, name)
            report["checkpoint_fingerprints"] = {key: array_fingerprint(a) for key, a in checkpoint_arrays.items()}
        report["status"] = "VALID"
        return report, (branch, source, weight, ids, section)
    except (AttributeError, TypeError, ValueError, IndexError, OverflowError) as error:
        report["issues"].append(str(error))
        return report, None


def _observations(state, simulation, data):
    unavailable = dict(status="NOT_EVALUATED", reason="No valid retained source and incident population")
    observables = dict(status="NOT_EVALUATED", **{key: None for key in METRICS})
    evidence = dict(focus=unavailable.copy(), topology=unavailable.copy())
    if data is None:
        return observables, evidence
    branch, source, weight, ids, section = data
    try:
        from temsim.physics.beam_statistics import transverse_beam_statistics
        from temsim.physics.beam_current import effective_source_current_a
        end = float(state.sample.upper_surface_z_mm)
        cp = getattr(simulation, "incident_checkpoints", None)
        if cp is None:
            from types import SimpleNamespace
            cp = SimpleNamespace(z_mm=branch.z, **dict(zip(_PHASE, (branch.x, branch.y, branch.tx, branch.ty))))
            basis = "Exact retained incident plane; display history is not topology evidence"
        else:
            basis = "Exact full-precision incident checkpoint"
        z = np.asarray(cp.z_mm, float)
        matches = np.flatnonzero(z == end)
        plane_kind = "specimen_upper_surface"
        if len(matches) != 1 and getattr(simulation, "incident_checkpoints", None) is not None:
            reference = _number(getattr(state.sample, "z_mm", None))
            plan_z = np.asarray(getattr(getattr(simulation, "incident_plan", None), "z_mm", ()))
            if (reference is not None and plan_z.ndim == 1 and len(plan_z)
                    and plan_z[-1] == reference and z[-1] == reference
                    and np.asarray(branch.z)[-1] == reference):
                reference_rows = np.flatnonzero(z == reference)
                if len(reference_rows) == 1:
                    end, matches = reference, reference_rows
                    plane_kind = "sample_reference_plane"
                    basis = ("Exact incident optical-reference checkpoint at sample.z_mm; "
                             "not a specimen upper-surface focus or specimen-scattering measurement")
        if len(matches) != 1 and section is not None and float(section.branch.z[-1]) < end:
            end = float(section.branch.z[-1])
            if (_number(getattr(simulation, "metrics", {}).get("section_target_z_mm")) != end
                    or float(section.plan.z_mm[-1]) != end or float(section.checkpoints.z_mm[-1]) != end):
                raise ValueError("Section target, plan and exact terminal checkpoint must agree")
            cp = section.checkpoints
            z = np.asarray(cp.z_mm, float)
            matches = np.flatnonzero(z == end)
            plane_kind = "section_terminal"
            basis = "Exact validated section-terminal checkpoint; local beam statistics, specimen not reached"
        if len(matches) != 1:
            observables["reason"] = "The specimen upper surface is not an exact retained plane"
            return observables, evidence
        i = int(matches[0])
        blocked = np.asarray(branch.blocked_z)
        masks = ((np.asarray(source.alive, bool) & (weight > 0))[None, :]
                 & (np.isnan(blocked)[None, :] | (z[:, None] < blocked[None, :])))
        arrays = [np.asarray(getattr(cp, key)) for key in _PHASE]
        if any(a.shape != masks.shape for a in arrays) or any(np.any(~np.isfinite(a[masks])) for a in arrays):
            raise ValueError("Captured phase is nonfinite or misaligned on current-carrying checkpoints")
        # A later stop does not remove current that reached this exact plane.
        # Endpoint survival is checked against the physical stop ledger above.
        alive = masks[i]
        if not alive.any():
            observables.update(status="NO_TRANSMISSION", current_a=0., transmission=0.,
                               plane_z_mm=end, plane_kind=plane_kind, basis=basis)
            return observables, evidence
        stats = transverse_beam_statistics(*(a[i] for a in arrays), alive=alive, weights=weight)
        current = effective_source_current_a(state)
        if not math.isfinite(current) or current < 0:
            raise ValueError("The captured source current is not finite and nonnegative")
        observables.update(status="MEASURED", plane_z_mm=end, plane_kind=plane_kind, basis=basis,
            alpha95_mrad=stats.convergence_95_mrad, d95_nm=2*stats.radius_95_m*1e9,
            rms_radius_nm=stats.radius_rms_m*1e9, current_a=current*stats.surviving_fraction,
            transmission=stats.surviving_fraction, effective_source_current_a=current,
            mean_x_m=stats.mean_x_m, mean_y_m=stats.mean_y_m,
            mean_tx_rad=stats.mean_tx_rad, mean_ty_rad=stats.mean_ty_rad,
            paraxial_waist_offset_nm=stats.waist_offset_m*1e9,
            angle_definition="Current-weighted alpha95 relative to the chief direction; not a physical pupil edge",
            radius_definition="Current-weighted transverse radius about the centroid; raw coordinates are unchanged")
        if plane_kind in {"section_terminal", "sample_reference_plane"}:
            reason = ("The bounded section has not reached the specimen; local beam statistics do not qualify a specimen probe"
                      if plane_kind == "section_terminal" else
                      "The sample reference plane is an incident optical reference; specimen upper-surface focus, "
                      "topology qualification and specimen scattering are not evaluated by these local statistics")
            evidence.update(focus=dict(status="NOT_EVALUATED", reason=reason),
                            topology=dict(status="NOT_EVALUATED", reason=reason))
            return observables, evidence
        evidence["focus"] = dict(status="NOT_EVALUATED", reason="Three exact upstream focus planes are required; an extrapolated waist is diagnostic only")
        focus_planes = np.array([end-.0002, end-.0001, end])
        selected = [np.flatnonzero(z == plane) for plane in focus_planes]
        if getattr(simulation, "incident_checkpoints", None) is not None and all(len(row) == 1 for row in selected):
            from temsim.optics.surface_probe_focus import local_waist_from_radii
            radii = [transverse_beam_statistics(*(a[int(row[0])] for a in arrays),
                     alive=alive, weights=weight).radius_rms_m for row in selected]
            offset, curvature = local_waist_from_radii(radii, 1e-7)
            evidence["focus"] = dict(status="MEASURED" if math.isfinite(offset) else "UNRESOLVED",
                local_waist_offset_nm=_number(offset), variance_second_derivative=_number(curvature),
                planes_mm=focus_planes.tolist(), scope="Read-only exact-plane observation; no declared target-angle or size qualification")
        if getattr(simulation, "incident_checkpoints", None) is not None:
            from temsim.optics.condenser_topology import evaluate, is_applicable
            if is_applicable(state):
                evidence["topology"] = evaluate(state, cp, masks, weight, maximum_spacing_mm=.5, source_ids=ids)
            else:
                evidence["topology"] = dict(status="NOT_EVALUATED", reason="Required C2/C3/ADL geometry is not applicable")
    except (AttributeError, KeyError, TypeError, ValueError, FloatingPointError) as error:
        observables.setdefault("reason", str(error))
        if observables["status"] != "MEASURED":
            observables["status"] = "UNRESOLVED"
        evidence["observation_error"] = str(error)
    return observables, evidence


def observe_calculation(state, result):
    """Observe a returned CalculationResult without propagation or input edits."""
    simulation = getattr(result, "simulation", None)
    population, data = _population(simulation)
    observables, evidence = _observations(state, simulation, data)
    physical = "FAIL" if evidence["topology"].get("status") == "FAIL" else "UNRESOLVED"
    metrics = getattr(simulation, "metrics", {}) or {}
    numerical = dict(status="NOT_RUN", scope="No independent numerical refinement is executed by this observer")
    retained = metrics.get("numerical_qualification")
    if isinstance(retained, Mapping):
        numerical["retained_evidence"] = _safe(retained)
        numerical["retained_evidence_role"] = "Historical supplied evidence; not rerun or promoted to full qualification"
    return _safe(dict(schema=OBSERVATION_SCHEMA, identity=_input_identity(state, result),
        execution=dict(status="COMPLETE" if simulation is not None else "INCOMPLETE",
            scope="A result object was returned; this is not numerical or physical acceptance"),
        performance=_performance(result), population=population, observables=observables,
        qualification=dict(numerical=numerical, physical=dict(status=physical,
            scope="Independent target, size, focus and convergence acceptance remains required", **evidence))))


def _tolerances(values):
    result = {key: dict(atol=0., rtol=0.) for key in METRICS}
    if values is not None:
        if not isinstance(values, Mapping) or set(values)-set(METRICS):
            raise ValueError("Declare tolerances only for known unit-bearing observables")
        for key, row in values.items():
            if not isinstance(row, Mapping) or set(row) != {"atol", "rtol"}:
                raise ValueError("Each observable tolerance needs explicit atol and rtol")
            normalized = {name: _number(row[name], nonnegative=True) for name in ("atol", "rtol")}
            if any(value is None for value in normalized.values()):
                raise ValueError("Observable tolerances must be finite and nonnegative")
            result[key] = normalized
    return result


def compare_observations(reference, candidate, *, metric_tolerances=None):
    """Compare exact raw arrays and scalars under identical captured inputs."""
    tolerances = _tolerances(metric_tolerances)
    failures = []
    if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
        return dict(status="UNRESOLVED", failures=["Missing observation"], metric_tolerances=tolerances)
    for label, observation in (("reference", reference), ("candidate", candidate)):
        if (observation.get("schema") != OBSERVATION_SCHEMA
                or observation.get("execution", {}).get("status") != "COMPLETE"
                or observation.get("identity", {}).get("status") != "CAPTURED"
                or observation.get("population", {}).get("status") != "VALID"
                or observation.get("observables", {}).get("status") != "MEASURED"):
            failures.append(f"Incomplete or invalid {label} evidence")
    for key in ("inputs_digest", "physical_inputs_digest", "numerical_inputs_digest"):
        first, second = reference.get("identity", {}).get(key), candidate.get("identity", {}).get(key)
        if not first or first != second:
            failures.append(f"Mismatched or missing {key}")
    first_plane, second_plane = reference.get("observables", {}), candidate.get("observables", {})
    if (first_plane.get("plane_kind") not in {"specimen_upper_surface", "section_terminal", "sample_reference_plane"}
            or first_plane.get("plane_kind") != second_plane.get("plane_kind")
            or _number(first_plane.get("plane_z_mm")) is None
            or _number(first_plane.get("plane_z_mm")) != _number(second_plane.get("plane_z_mm"))):
        failures.append("Mismatched or missing observation plane kind or exact axial position")
    for key in ("fingerprints", "source_fingerprints"):
        first, second = reference.get("population", {}).get(key), candidate.get("population", {}).get(key)
        if not first or first != second:
            failures.append(f"Mismatched or missing exact {key}")
    first_cp = reference.get("population", {}).get("checkpoint_fingerprints")
    if first_cp != candidate.get("population", {}).get("checkpoint_fingerprints"):
        failures.append("Mismatched exact checkpoint fingerprints")
    differences = {}
    for key, limits in tolerances.items():
        first = _number(reference.get("observables", {}).get(key))
        second = _number(candidate.get("observables", {}).get(key))
        allowed = None if first is None else limits["atol"]+limits["rtol"]*abs(first)
        delta = None if first is None or second is None else abs(second-first)
        passed = delta is not None and delta <= allowed
        differences[key] = dict(reference=first, candidate=second, absolute_difference=delta,
                                allowed_difference=allowed, passed=passed)
        if not passed:
            failures.append(f"Missing or out-of-budget {key}")
    return dict(status="PASS" if not failures else "FAIL", failures=failures,
        metric_tolerances=tolerances, observables=differences,
        implementations=[reference.get("identity", {}).get("implementation"), candidate.get("identity", {}).get("implementation")],
        scope="Same-input repeatability only; exact raw arrays required even with scalar tolerances; not physical or convergence qualification")


def _statistics(values):
    finite = [number for value in values if (number := _number(value, nonnegative=True)) is not None]
    return dict(count=len(finite), missing_count=len(values)-len(finite),
        median=float(np.median(finite)) if finite else None,
        minimum=min(finite) if finite else None, maximum=max(finite) if finite else None,
        standard_deviation=float(np.std(finite)) if finite else None)


def summarize_runs(runs, *, metric_tolerances=None):
    """Group timings by case and phase; never count a missing run as passing."""
    limits = _tolerances(metric_tolerances)
    rows = [_safe(dict(row)) for row in runs]
    groups, cases = {}, {}
    for row in rows:
        case = str(row.get("case_id", "unspecified"))
        phase = str(row.get("phase", "unspecified"))
        groups.setdefault((case, phase), []).append(row)
        cases.setdefault(case, []).append(row)
    summaries = []
    for (case, phase), members in sorted(groups.items()):
        complete = [row for row in members
            if row.get("execution_status") not in {"FAILED", "CANCELLED", "INCOMPLETE"}
            and isinstance(row.get("observation"), Mapping)
            and row["observation"].get("execution", {}).get("status") == "COMPLETE"]
        summaries.append(dict(case_id=case, phase=phase, runs=len(members),
            complete_runs=len(complete), failed_or_incomplete_runs=len(members)-len(complete),
            elapsed_seconds=_statistics([row.get("elapsed_seconds") for row in complete]),
            peak_rss_bytes=_statistics([(row.get("memory") or {}).get("peak_rss_bytes") for row in complete]),
            peak_vram_bytes=_statistics([(row.get("memory") or {}).get("peak_vram_bytes") for row in complete])))
    repeatability = []
    for case, members in sorted(cases.items()):
        comparisons = [compare_observations(members[0].get("observation"), row.get("observation"),
                                            metric_tolerances=limits) for row in members[1:]]
        failures = [i for i, row in enumerate(members) if row.get("execution_status") in {"FAILED", "CANCELLED", "INCOMPLETE"}
                    or not isinstance(row.get("observation"), Mapping)
                    or row["observation"].get("execution", {}).get("status") != "COMPLETE"]
        status = ("FAIL" if failures or any(row["status"] != "PASS" for row in comparisons)
                  else "PASS" if comparisons else "NOT_EVALUATED")
        repeatability.append(dict(case_id=case, status=status, failed_run_indices=failures,
                                  comparisons=comparisons, run_count=len(members)))
    status = ("FAIL" if any(row["status"] == "FAIL" for row in repeatability) else
              "PASS" if repeatability and all(row["status"] == "PASS" for row in repeatability) else "NOT_EVALUATED")
    physical_failures = [dict(case_id=row.get("case_id"), phase=row.get("phase"), index=index)
        for index, row in enumerate(rows) if isinstance(row.get("observation"), Mapping)
        and row["observation"].get("qualification", {}).get("physical", {}).get("status") == "FAIL"]
    return dict(schema=REPORT_SCHEMA, observer=OBSERVER_IDENTITY, runs=rows, groups=summaries,
        repeatability=dict(status=status, cases=repeatability), metric_tolerances=limits,
        numerical_qualification="NOT_RUN", physical_qualification="FAIL" if physical_failures else "UNRESOLVED",
        physical_failures=physical_failures,
        scope="Timing and exact same-input repeatability; independent numerical and physical qualification are separate",
        memory_scope="Caller-provided measured peak RSS/VRAM; missing measurements remain null")


def compare_reports(reference, candidate):
    """Strict before/after comparison; code may differ, inputs and budgets may not."""
    if reference.get("schema") != REPORT_SCHEMA or candidate.get("schema") != REPORT_SCHEMA:
        raise ValueError("Unsupported performance/accuracy report schema")
    if reference.get("metric_tolerances") != candidate.get("metric_tolerances"):
        return dict(status="INCOMPARABLE", reason="Declared accuracy budgets differ")
    for report in (reference, candidate):
        if (report.get("repeatability", {}).get("status") == "FAIL"
                or any(not isinstance(row.get("observation"), Mapping)
                       or row.get("execution_status") in {"FAILED", "CANCELLED", "INCOMPLETE"}
                       for row in report.get("runs", ()))):
            return dict(status="FAIL", reason="A failed or missing run cannot be removed from a before/after comparison")
    if any(report.get("repeatability", {}).get("status") != "PASS" for report in (reference, candidate)):
        return dict(status="UNRESOLVED", reason="Both reports require complete internal repeatability checks")
    before_groups = {(row["case_id"], row["phase"]): row for row in reference.get("groups", ())}
    after_groups = {(row["case_id"], row["phase"]): row for row in candidate.get("groups", ())}
    if (not before_groups or set(before_groups) != set(after_groups)
            or any(before_groups[key]["runs"] != after_groups[key]["runs"] for key in before_groups)):
        return dict(status="INCOMPARABLE", reason="Matching case/phase inventories and run counts are required")
    before = {row["case_id"]: row for row in reference.get("runs", ()) if row.get("observation")}
    after = {row["case_id"]: row for row in candidate.get("runs", ()) if row.get("observation")}
    if not before or set(before) != set(after):
        return dict(status="INCOMPARABLE", reason="Complete matching case inventories are required")
    comparisons = {key: compare_observations(before[key]["observation"], after[key]["observation"],
                    metric_tolerances=reference["metric_tolerances"]) for key in sorted(before)}
    timing = []
    for key in sorted(before_groups):
        old = _number(before_groups[key]["elapsed_seconds"].get("median"), nonnegative=True)
        new = _number(after_groups[key]["elapsed_seconds"].get("median"), nonnegative=True)
        timing.append(dict(case_id=key[0], phase=key[1], before_median_seconds=old,
            after_median_seconds=new, speedup=old/new if old is not None and new is not None and new > 0 else None,
            accuracy_passed=comparisons[key[0]]["status"] == "PASS"))
    return dict(status="PASS" if all(row["status"] == "PASS" for row in comparisons.values()) else "FAIL",
                cases=comparisons, performance=timing,
                scope="Same physical/numerical inputs and explicit budgets; implementation identities remain visible")


def format_report(report):
    """Render the shared report as readable English text without hiding limits."""
    lines = ["Performance and accuracy report", "",
             f"Same-input repeatability: {report.get('repeatability', {}).get('status', 'NOT_EVALUATED')}",
             f"Numerical qualification: {report.get('numerical_qualification', 'NOT_RUN')}",
             f"Physical qualification: {report.get('physical_qualification', 'UNRESOLVED')}",
             "Raw array acceptance: exact source IDs, weights, masks, phase and retained checkpoints.",
             "Scalar tolerances: " + json.dumps(report.get("metric_tolerances", {}), sort_keys=True), ""]
    for group in report.get("groups", ()):
        timing = group["elapsed_seconds"]
        lines.append(f"{group['case_id']} / {group['phase']}: {group['runs']} runs; "
            f"median {timing['median']} s; standard deviation {timing['standard_deviation']} s; "
            f"peak RSS median {group['peak_rss_bytes']['median']} bytes; "
            f"peak VRAM median {group['peak_vram_bytes']['median']} bytes")
    representatives = {}
    for index, row in enumerate(report.get("runs", ())):
        case = str(row.get("case_id", "unspecified"))
        representatives.setdefault(case, None)
        if isinstance(row.get("observation"), Mapping):
            representatives[case] = (index, row)
    def scalar(value):
        number = _number(value)
        return "unknown" if number is None else f"{number:.8g}"
    for case, representative in sorted(representatives.items()):
        lines.append("")
        if representative is None:
            lines.append(f"{case} measurements: NOT_EVALUATED; no retained observation")
            lines.append("  Focus: NOT_EVALUATED; no retained observation")
            lines.append("  Topology: NOT_EVALUATED; no retained observation")
            continue
        index, row = representative
        observation = row["observation"]
        values = observation.get("observables", {})
        lines.append(f"{case} measurements: {values.get('status', 'NOT_EVALUATED')} "
            f"(representative retained run {index+1}, phase {row.get('phase', 'unspecified')})")
        lines.append(f"  Plane: {values.get('plane_kind', 'NOT_EVALUATED')}; z_mm={scalar(values.get('plane_z_mm'))}")
        lines.append(f"  alpha95_mrad={scalar(values.get('alpha95_mrad'))}; "
            f"D95_nm={scalar(values.get('d95_nm'))}; "
            f"RMS_radius_nm={scalar(values.get('rms_radius_nm'))}; "
            f"current_a={scalar(values.get('current_a'))}")
        if values.get("reason"):
            lines.append("  Measurement reason: "+str(values["reason"]))
        physical = observation.get("qualification", {}).get("physical", {})
        for key, label in (("focus", "Focus"), ("topology", "Topology")):
            evidence = physical.get(key, {})
            detail = f"  {label}: {evidence.get('status', 'NOT_EVALUATED')}"
            if key == "focus" and "local_waist_offset_nm" in evidence:
                detail += "; local_waist_offset_nm="+scalar(evidence["local_waist_offset_nm"])
            reasons = [str(evidence[name]) for name in ("reason", "message") if evidence.get(name)]
            if not reasons and evidence.get("scope"):
                reasons.append(str(evidence["scope"]))
            if reasons:
                detail += "; "+"; ".join(dict.fromkeys(reasons))
            lines.append(detail)
    for row in report.get("repeatability", {}).get("cases", ()):
        lines.append(f"{row['case_id']} repeatability: {row['status']}")
        for comparison in row.get("comparisons", ()):
            lines.extend("  "+message for message in comparison.get("failures", ()))
    lines.extend(("", str(report.get("scope", "")), str(report.get("memory_scope", ""))))
    return "\n".join(lines)+"\n"


def write_report(report, path):
    """Write strict JSON to the explicitly requested report path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_safe(report), indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    return destination
