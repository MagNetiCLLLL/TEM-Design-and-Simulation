"""Audit the executed joint-probe field plan without propagating particles.

``audit(state, plan=None)`` returns strict JSON or raises ValueError. With no
plan, production plan construction runs on a restored private input snapshot;
this can resolve field providers but never traces a gun or a particle. Passing
the executed segment's plan avoids even that construction. A PASS establishes
the declared model and action ownership, not optical or hardware qualification.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from temsim.immutable_json import thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.simulation_modes import is_unified
from temsim.physics.core import AxialPropagationPlan, build_propagation_plan
from temsim.physics.instrument_magnetic import active_column_events, _component_kick_events
from temsim.physics.plan_contract import validate_plan_contract
from temsim.physics.potential_snapshot import _canonical_owner
from temsim.physics.posed_wave_aperture import PosedWaveAperture
from temsim.physics.unified_field_runtime import UnifiedFieldRuntime, unified_source_selection

SCHEMA = "temsim-joint-probe-field-audit-v1"
C2_KEY = "condenser_aperture_2"
_EXTRA_ARRAYS = ("cs_kick_m3", "thin_power_m1", "thin_rotation_rad", "kick_x_rad", "kick_y_rad")
_EXTRA_LISTS = ("posed_spherical_kicks", "intrinsic_aberration_kicks", "thin_actions", "mapped_fields")
_CONTROL_FIELDS = ("z_mm", "percent", "polarity", "strength_m3", "strength_m2", "orientation_rad",
    "strength_x_percent", "strength_y_percent", "effective_length_mm", "mechanical_length_mm",
    "mechanical_clear_bore_diameter_mm", "maximum_strength_m3", "maximum_strength_m2")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(not isinstance(value, (bool, np.bool_)), f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    _require(math.isfinite(result), f"{name} must be a finite number")
    return result


def _inputs(state):
    _require(is_unified(state), "Joint probe audit requires unified field optics")
    gun = state.electron_gun
    _require(gun.source_representation == "classical_particles", "A physical classical_particles source is required")
    for flag in ("wave_enabled", "stem_wave_enabled", "stem_fourdstem_enabled"):
        _require(not getattr(state.sample, flag, False), f"Coherent input is not admitted: sample.{flag}")
    apertures = [item for item in state.apertures if item.key == C2_KEY]
    _require(len(apertures) == 1, "Exactly one physical C2 aperture is required")
    aperture = apertures[0]
    for name in ("installed", "enabled", "inserted"):
        _require(getattr(aperture, name, True) is True, f"C2 aperture must be {name}")
    _require(_finite(aperture.diameter_mm, "C2 diameter") == .100, "C2 diameter must be exactly 0.100 mm")
    _require(_finite(aperture.radius_mm, "C2 radius") == .050, "C2 actual radius must be exactly 0.050 mm")
    start = _finite(gun.exit_plane_z_mm, "Physical gun exit")
    end = _finite(state.sample.upper_surface_z_mm, "Specimen upper surface")
    _require(start < end, "The specimen upper surface must follow the physical gun exit")
    return aperture, start, end


def _plan_actions(plan, runtime, start, end):
    _require(type(plan) is AxialPropagationPlan, "An exact production AxialPropagationPlan is required")
    _require(plan.solver_signature == "unified-potential-rk4-v1:" + runtime.digest,
        "The executed solver must be the captured unified Lorentz solver")
    z = np.asarray(plan.z_mm)
    _require(z.ndim == 1 and len(z) >= 2 and np.isfinite(z).all() and np.all(np.diff(z) > 0)
        and z[0] == start and z[-1] == end, "Plan must span the actual gun exit to the exact specimen upper surface")
    contract = thaw_json(plan.physics_contract)
    validate_plan_contract(contract)
    _require(contract["mode"] == "unified" and list(contract["interval_mm"]) == [start, end],
        "Plan contract must own the full unified incident interval")
    _require(contract["unified_capture"] == {"runtime": runtime.digest,
        "field": runtime.snapshot.numerical_digest, "geometry": runtime.geometry.digest},
        "Plan contract is not bound to the actual field and geometry capture")
    arrays = {}
    for name in _EXTRA_ARRAYS:
        values = np.asarray(getattr(plan, name))
        _require(values.shape == z.shape and np.isfinite(values).all() and not np.any(values != 0),
            f"Additional empirical or thin action is forbidden: {name}")
        arrays[name] = {"count": int(values.size), "nonzero_count": 0}
    lists = {}
    for name in _EXTRA_LISTS:
        values = getattr(plan, name)
        _require(isinstance(values, (tuple, list)) and not values, f"Additional action list is forbidden: {name}")
        lists[name] = 0
    _require(plan.electric_field is None and plan.electric_field_identity is None,
        "The captured runtime must be the sole electric-field owner")
    records = contract["interactions"]["records"]
    actual = {row.source_id: row for row in runtime.snapshot.contributions}
    _require(len(records) == len(actual) and len({row["interaction_id"] for row in records}) == len(actual),
        "Executed interaction inventory does not match captured potentials")
    for record in records:
        source = actual.get(record["interaction_id"])
        _require(source is not None and record["representation"] == "field"
            and record["effect_family"] == "electromagnetic_potential"
            and record["source_digest"] == source.potential.digest
            and set(record["covered_effects"]) == {owner+":"+source.kind for owner in source.owner_ids},
            "An interaction is missing, duplicated, empirical, or not bound to its actual potential")
    return {"additional_arrays": arrays, "additional_lists": lists, "extra_electric_provider": False,
        "interaction_count": len(records), "interval_mm": [start, end], "node_count": int(len(z)),
        "legacy_input_coefficients": "Preserved; inactive in this field-owned interval"}


def _field_ownership(state, runtime):
    metadata = thaw_json(runtime.metadata)
    for flag in ("empirical_aberrations_applied", "equivalent_image_maps_applied"):
        _require(metadata.get(flag) is False, f"Captured field ownership flag must explicitly be false: {flag}")
    _require(metadata.get("round_radial_order") == 3, "The native high-order round-field model must remain active")
    rows, identities, covered = [], set(), set()
    magnetic_owners = set()
    for contribution in runtime.snapshot.contributions:
        _require(contribution.source_id not in identities, "Duplicate captured potential source")
        identities.add(contribution.source_id)
        for owner in contribution.owner_ids:
            key = (contribution.kind, _canonical_owner(runtime.snapshot.scene, owner))
            _require(key not in covered, "Duplicate physical field ownership")
            covered.add(key)
        if contribution.kind == "magnetic":
            magnetic_owners.update(contribution.owner_ids)
        rows.append({"source_id": contribution.source_id, "owner_ids": list(contribution.owner_ids),
            "kind": contribution.kind, "potential_type": type(contribution.potential).__name__,
            "potential_digest": contribution.potential.digest})
    lenses, auxiliaries = unified_source_selection(state)
    components = {item.key: item for group in ("stigmators", "corrector_elements", "deflectors")
        for item in getattr(state, group, ())}
    required_owners = {key: [key] for key in lenses}
    for key in auxiliaries:
        # Native deflector controls own separately captured physical coils.
        # Require every declared coil, including zero-current coils; accepting
        # any owner with a common prefix would hide an omitted second coil.
        events, _ = _component_kick_events(components[key], metadata.get("emission_time_s", 0.))
        required_owners[key] = [f"{key}:{index}" for index in range(len(events))] if events else [key]
    expected = {owner for group in required_owners.values() for owner in group}
    _require(expected <= magnetic_owners, "An enabled physical magnetic field owner is missing: "
        + ", ".join(sorted(expected - magnetic_owners)))
    _require(any(row["kind"] == "electric" and "instrument_electric" in row["owner_ids"] for row in rows),
        "The retained instrument electric potential is missing")
    return {"metadata": metadata, "sources": rows, "selected_round_lenses": list(lenses),
        "selected_auxiliaries": list(auxiliaries), "required_magnetic_owners": required_owners,
        "duplicate_ownership": False,
        "natural_field_aberrations_retained": True}


def _c2_geometry(aperture, runtime, start, end):
    inventory = [row for row in runtime.geometry.aperture_inventory if row["key"] == C2_KEY]
    _require(len(inventory) == 1 and all(inventory[0][key] is True for key in ("installed", "enabled", "inserted")),
        "The actual runtime did not capture an inserted C2 aperture")
    # The physical host also contains its wider mechanical body bore. Only
    # the finite plate primitive proves the inserted optical opening.
    pieces = [(host, piece) for host in runtime.geometry.boundaries for piece in host.pieces
        if piece.key == C2_KEY and type(piece) is PosedWaveAperture]
    _require(len(pieces) == 1, "Exactly one actual finite C2 aperture boundary is required")
    host, piece = pieces[0]
    _require(type(piece) is PosedWaveAperture and piece.radius_mm == .050,
        "The actual C2 boundary must have a 0.050 mm circular radius")
    faces = piece.event_z_mm()
    _require(len(faces) >= 2 and min(faces) > start and max(faces) < end,
        "The physical C2 plate must lie inside the audited incident interval")
    return {"key": C2_KEY, "diameter_mm": float(aperture.diameter_mm), "radius_mm": float(aperture.radius_mm),
        "actual_boundary_radius_mm": float(piece.radius_mm), "installed": True, "enabled": True, "inserted": True,
        "host_key": host.host_key, "local_start_z_mm": float(piece.local_start_z_mm),
        "local_end_z_mm": float(piece.local_end_z_mm), "event_z_mm": [float(v) for v in faces],
        "origin_global_m": list(piece.registration.origin_global_m),
        "rotation_local_to_global": thaw_json(piece.registration.rotation_local_to_global)}


def _controls(state):
    result = {}
    for group in ("lenses", "corrector_elements", "stigmators"):
        result[group] = []
        for item in getattr(state, group):
            row = {"key": item.key, "enabled": bool(item.enabled), "model_type": type(item).__name__}
            for name in _CONTROL_FIELDS:
                if hasattr(item, name):
                    row[name] = _finite(getattr(item, name), item.key + "." + name)
            result[group].append(row)
    return result


def audit(state, plan=None) -> dict:
    """Prove model/action/geometry ownership; never retune or propagate.

    A supplied plan must describe the entire physical gun-exit to sample-upper-
    surface interval and be bound to exactly these current instrument inputs.
    Missing evidence is a ValueError, never a passing incomplete audit.
    """
    aperture, start, end = _inputs(state)
    original = capture_instrument_snapshot(state)
    mode = "provided_executed_plan" if plan is not None else "private_production_plan_construction"
    work = state
    if plan is None:
        work = original.restore()
        _require(capture_instrument_snapshot(work).digest == original.digest,
            "Restoring inputs changed their identity; refusing silent normalization")
        aperture, start, end = _inputs(work)
        plan = build_propagation_plan(work, start, end, active_column_events(work))
    runtime = getattr(plan, "unified_runtime", None)
    _require(type(runtime) is UnifiedFieldRuntime, "An exact captured unified field runtime is required")
    runtime.require_state(work)
    _require(runtime.source_identity == original.digest, "Plan source identity differs from the original inputs")
    actions = _plan_actions(plan, runtime, start, end)
    ownership = _field_ownership(work, runtime)
    c2 = _c2_geometry(aperture, runtime, start, end)
    gun = work.electron_gun
    source = {"source_representation": gun.source_representation,
        "beam_voltage_kv": _finite(work.beam_voltage_kv, "Beam voltage"),
        "nominal_exit_energy_ev": _finite(gun.nominal_exit_energy_ev, "Nominal gun exit energy"),
        "emitted_current_a": _finite(gun.emitted_current_a, "Emitted current"),
        "current_scope": "Input emitted current; actual surviving current and kinetic energies require executed particles"}
    from temsim.physics.beam_current import column_current_limit_percent
    source["column_current_limit_percent"] = column_current_limit_percent(work)
    emitter = getattr(gun, "emitter", None)
    if emitter is not None:
        for name in ("ray_count", "energy_spread_fwhm_ev", "virtual_source_fwhm_nm"):
            if hasattr(emitter, name):
                source[name] = _finite(getattr(emitter, name), "Emitter " + name)
    report = {"schema": SCHEMA, "status": "PASS", "passed": True,
        "scope": "Declared physical model and executed plan action ownership; not optical qualification",
        "execution_model": {"mode": "unified", "source": "classical_particles", "audit_method": mode,
            "solver_signature": plan.solver_signature, "particle_propagation_performed_by_audit": False},
        "c2_aperture": c2, "plan_actions": actions, "field_ownership": ownership, "controls": _controls(work),
        "source": source, "identities": {"input_digest": original.digest, "runtime_digest": runtime.digest,
            "potential_digest": runtime.snapshot.numerical_digest, "geometry_digest": runtime.geometry.digest,
            "plan_signature": plan.signature},
        "limitations": ["Native engineering fields are not OEM field calibration",
            "Natural nonlinear round and multipole fields remain active; they were not zeroed",
            "No particle trace, focus, sampling, topology or experimental qualification is supplied by this audit"]}
    _require(capture_instrument_snapshot(work).digest == original.digest
        and capture_instrument_snapshot(state).digest == original.digest, "Audit changed physical instrument inputs")
    # Enforce a portable JSON result before callers persist a passing proof.
    return json.loads(json.dumps(report, allow_nan=False))
