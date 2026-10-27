"""P0 ownership metadata for the existing shared column operator.

This module never solves a field, changes an optical setting or applies an
action. It records the decisions made by the current compiler. Numerical
arrays and frozen providers remain authoritative for execution; scene input
identity alone is not a field or cache identity.
"""
from collections.abc import Mapping
from dataclasses import asdict, replace
import math
from numbers import Real

import numpy as np

from temsim.immutable_json import freeze_json, thaw_json, json_digest


SCHEMA = "shared-column-physics-contract-v1"


def reject_repeated_physical_events(captured, remaining):
    """A finite coil consumed from the event list cannot also be a thin kick.

    Anonymous external probes are deliberately not deduplicated by position
    or strength: two independently owned drives can have identical values.
    Only an exact extra copy of a consumed, identified coil is rejected.
    """
    for field in captured:
        event = (field.event_z_mm, field.event_dx_rad, field.event_dy_rad)
        # A zero kick has no second action and is common at coincident planes.
        if (event[1] or event[2]) and any(tuple(row) == event for row in remaining):
            raise ValueError(f"Duplicate physical action for {field.key}: finite magnetic coil and thin kick")


def _round_effects(key):
    from temsim.optics.intrinsic_aberrations import INTRINSIC_TERMS
    return (f"{key}:round", f"{key}:Cs", f"{key}:Cc",
            *(f"{key}:{term.field}" for term in INTRINSIC_TERMS))


def capture_plan_contract(state, *, z_mm, numerical_digest, mapped_fields,
                          mapped_keys, dipoles, events, image_events, posed_cs,
                          intrinsic, include_spherical, include_hexapole,
                          electric_field, provider_metadata=()):
    """Capture detached identities and validate ownership before execution."""
    from temsim.physics.physical_scene import capture_physical_scene
    from temsim.physics.interaction_registry import InteractionRecord, InteractionRegistry
    from temsim.physics.core import _support_bounds_mm
    from temsim.physics.lens_field_provider import FrozenMappedField
    from temsim.optics.equivalent_image_lenses import IMAGE_LENS_KEYS, equivalent_image_lenses_enabled
    from temsim.component_keys import CONDENSER_LENS_KEYS

    scene = capture_physical_scene(state)
    lower, upper = float(z_mm[0]), float(z_mm[-1])
    interval = (lower, upper)
    records = []
    reference_planes = {}
    metadata = {row["key"]: row for row in provider_metadata}
    from temsim.physics.posed_aberrations import kick_indices
    executed_kick_planes = {id(kick): float(z_mm[index]) for index, kicks in
                            kick_indices((*posed_cs, *intrinsic), z_mm).items() for kick in kicks}

    def owner(key):
        return scene.owner_for(key) or f"unresolved:{key}"

    def add(key, family, role, span, coverage, *, source=None, drives=(),
            residual=None, suffix="", circuit=None, reference_plane=None):
        lo, hi = max(lower, float(span[0])), min(upper, float(span[1]))
        if hi < lo or (hi == lo and span[1] > span[0]):
            return
        interaction_id = f"{role}:{key}:{family}{suffix}"
        records.append(InteractionRecord(
            interaction_id=interaction_id, owner_id=owner(key),
            effect_family=family, representation=role, interval_mm=(lo, hi),
            covered_effects=tuple(coverage), source_digest=source or numerical_digest,
            logical_drive_ids=tuple(drives or (key,)), circuit_id=circuit,
            residual_provenance=residual))
        if reference_plane is not None:
            reference_planes[interaction_id] = float(reference_plane)

    # Nonlinear total-field metadata is taken from the resolved provider BEFORE
    # freezing. A zero member map is not another executing magnetic source.
    for field in mapped_fields:
        key = field.lens_key
        info = metadata.get(key, {})
        if info.get("model_status") == "included_in_joint_nonlinear_field":
            add(key, "joint_field_member", "observation", field.field_support_mm,
                (), source=field.fingerprint)
            continue
        component_key = getattr(field, "component_key", key)
        is_dipole = getattr(field, "event_z_mm", None) is not None
        if is_dipole:
            family, coverage = "dipole", (f"{key}:dipole",)
        elif hasattr(field, "polynomial_terms"):
            degrees = {int(x)+int(y) for x, y, _ in field.polynomial_terms}
            family = "multipole"
            coverage = tuple(f"{component_key}:{name}" for degree, name in
                             ((1, "dipole"), (2, "quadrupole"), (3, "hexapole")) if degree in degrees)
        else:
            family = "round"
            channels = tuple(info.get("joint_channels", ())) or (key,)
            coverage = tuple(effect for channel in channels for effect in
                             (_round_effects(channel) if isinstance(field, FrozenMappedField)
                              else (f"{channel}:round",)))
        add(component_key, family, "field", field.field_support_mm, coverage,
            source=field.fingerprint, drives=getattr(field, "drive_keys", ()) or tuple(info.get("joint_channels", ())),
            suffix=f":{key}" if component_key != key else "",
            circuit=info.get("circuit_id"))

    reduced = (equivalent_image_lenses_enabled(state)
               and lower >= float(getattr(getattr(state, "sample", None), "z_mm", math.inf)))
    for lens in getattr(state, "lenses", ()):
        key = lens.key
        if not getattr(lens, "enabled", True) or key in mapped_keys or key in metadata:
            continue
        if reduced and key in IMAGE_LENS_KEYS:
            continue
        native = state.condenser_system[key] if key in CONDENSER_LENS_KEYS else lens
        support = _support_bounds_mm(native) or interval
        add(key, "round", "field", support, (f"{key}:round",))

    for component in (*getattr(state, "stigmators", ()), *getattr(state, "corrector_elements", ())):
        key = component.key
        if not getattr(component, "enabled", False) or key in mapped_keys:
            continue
        support = _support_bounds_mm(component) or interval
        if hasattr(component, "quadrupole_tensor_m2") or hasattr(component, "quadrupole_strength_m2"):
            add(key, "quadrupole", "field", support, (f"{key}:quadrupole",))
        if include_hexapole and (hasattr(component, "hexapole_strength_components_m3")
                                  or hasattr(component, "hexapole_strength_m3")):
            add(key, "hexapole", "field", support, (f"{key}:hexapole",))

    for field in dipoles:
        key = field.key.rsplit(":", 1)[0]
        add(key, "dipole", "field", field.field_support_mm, (f"{field.key}:dipole",),
            drives=field.drive_keys, suffix=f":{field.key}", source=json_digest(asdict(field)))
    for event in image_events:
        # This legacy reduction replaces only linear round-lens transport. Its
        # configured extra Cs/Cc remain separate residuals, as before P0.
        add(event.key, "round", "reduced", interval, (f"{event.key}:round",),
            source=json_digest(asdict(event)), reference_plane=event.z_mm)
    if include_spherical:
        from temsim.lens_pose import has_lens_pose
        from temsim.optics.magnetic_lens_aberration import spherical_aberration_mm
        from temsim.optics.lens_focal_length import focal_length_mm
        native_mapped = {key for key, info in metadata.items() if info.get("is_mapped")}
        for lens in getattr(state, "lenses", ()):
            if (not getattr(lens, "enabled", True) or lens.key in native_mapped
                    or has_lens_pose(state, lens.key)):
                continue
            cs = spherical_aberration_mm(lens, state.beam_voltage_kv)
            if cs is None or not cs:
                continue
            try:
                focal = float(focal_length_mm(lens, state.beam_voltage_kv))
            except Exception:
                continue  # The existing scalar Cs compiler also omits this case.
            if not math.isfinite(focal) or focal <= 0:
                continue
            tolerance = 32*np.finfo(float).eps*max(1., abs(lower), abs(upper))
            if not lower-tolerance <= lens.z_mm <= upper+tolerance:
                continue
            plane = float(z_mm[int(np.argmin(abs(z_mm-lens.z_mm)))])
            add(lens.key, "Cs", "residual", (plane, plane), (f"{lens.key}:Cs",),
                residual="legacy additional Cs/f^4 thin action; not fitted from the executing field",
                reference_plane=lens.z_mm)
    for kick in posed_cs:
        if id(kick) not in executed_kick_planes:
            continue
        plane = executed_kick_planes[id(kick)]
        add(kick.lens_key, "Cs", "residual", (plane, plane), (f"{kick.lens_key}:Cs",),
            residual="legacy additional Cs in the placed lens frame", source=json_digest(asdict(kick)),
            reference_plane=kick.z_mm)
    for kick in intrinsic:
        if id(kick) not in executed_kick_planes:
            continue
        plane = executed_kick_planes[id(kick)]
        chromatic = kick.chromatic_reference_energy_ev is not None
        families = ("Cc",) if chromatic else tuple(name for name, value in kick.coefficients
                                                 if value and name.endswith("_mm"))
        for family in families:
            add(kick.lens_key, family, "residual", (plane, plane), (f"{kick.lens_key}:{family}",),
                residual=("legacy additional Cc; energy dependence of main fields retained" if chromatic
                          else "user-declared intrinsic polynomial; excludes legacy Cs"),
                source=json_digest(asdict(kick)), reference_plane=kick.z_mm)

    for index, event in enumerate(events):
        if event[1] or event[2]:
            # The tuple API has no source ID. Preserve independent caller
            # actions explicitly, without claiming physical-source equivalence.
            key = f"external_event:{index}"
            add(key, "slope_kick", "reduced", (event[0], event[0]), (key,),
                source=json_digest(tuple(map(float, event))))
    if electric_field is not None:
        add("electron_gun", "electric", "field", interval, ("electron_gun:electric",),
            source=electric_field.numerical_identity or numerical_digest)
    from temsim.physics.instrument_magnetic import gun_magnetic_support_edges_mm
    edges = gun_magnetic_support_edges_mm(state)
    if edges:
        add("electron_gun", "magnetic", "field", (min(edges), max(edges)), ("electron_gun:magnetic",))

    registry = InteractionRegistry(tuple(records))
    payload = {
        "schema": SCHEMA, "mode": "legacy_compatible", "interval_mm": interval,
        "scene": scene.to_dict(), "interactions": registry.to_dict(),
        "reference_planes_mm": reference_planes,
        "numerical_digest": numerical_digest,
        "scope": "shared column operator; geometry is declared here and clipped by existing boundary executors",
        "outside_operator": ("gun emission and source coherence", "material scattering and absorption",
                             "detector response", "energy-filter transfer and receiver imaging"),
        "limitations": ("No new Maxwell field or potential solver", "No field-derived Cs/Cc qualification",
                        "Native source digests conservatively identify the complete compiled plan, not independent source cache keys",
                        "Anonymous event tuples have no independently verified physical source identity"),
    }
    return _seal(payload)


def _seal(payload):
    result = dict(payload)
    result.pop("digest", None)
    result["digest"] = json_digest(result)
    return freeze_json(result)


def validate_plan_contract(contract):
    """Reject corrupt metadata and duplicate actions, including restored plans."""
    if contract is None:
        return  # Explicit compatibility path for hand-built/historical plans.
    from temsim.physics.physical_scene import PhysicalSceneSnapshot
    from temsim.physics.interaction_registry import InteractionRegistry
    if not isinstance(contract, Mapping) or contract.get("schema") != SCHEMA:
        raise ValueError("Unknown physical plan contract")
    if contract.get("mode") not in {"legacy_compatible", "ideal_thin", "unified"}:
        raise ValueError("Unknown physical ownership mode")
    payload = dict(contract)
    digest = payload.pop("digest", None)
    if digest != json_digest(payload):
        raise ValueError("Physical plan contract checksum mismatch")
    PhysicalSceneSnapshot.from_dict(contract["scene"])
    registry = InteractionRegistry.from_dict(contract["interactions"])
    if contract.get("mode") == "unified":
        identities = contract.get("unified_capture")
        if (not isinstance(identities, Mapping) or set(identities) != {"runtime", "field", "geometry"}
                or any(not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value)
                       for value in identities.values())
                or any(row.representation != "field" for row in registry.records)):
            raise ValueError("Unified ownership requires captured identities and exclusively field actions")
    if not isinstance(contract.get("reference_planes_mm"), Mapping) or not all(
            isinstance(key, str) and not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
            for key, value in contract["reference_planes_mm"].items()):
        raise ValueError("Physical reference planes must be finite values in mm")
    if set(contract["reference_planes_mm"]) - {row.interaction_id for row in registry.records}:
        raise ValueError("Physical reference plane has no owning interaction")
    lo, hi = contract["interval_mm"]
    if (any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in (lo, hi)) or hi < lo):
        raise ValueError("Physical plan contract interval must be finite and ordered")
    if any(row.interval_mm[0] < lo or row.interval_mm[1] > hi for row in registry.records):
        raise ValueError("Owned interaction lies outside the captured plan interval")


def contract_for_reference(contract):
    """Transform ownership alongside the existing first-order counterfactual."""
    if contract is None:
        return None
    from temsim.physics.interaction_registry import InteractionRegistry
    validate_plan_contract(contract)
    registry = InteractionRegistry.from_dict(contract["interactions"])
    keep = tuple(row for row in registry.records if row.effect_family not in {"Cs", "hexapole"}
                 and (row.representation != "residual" or row.effect_family in {"c1_mm", "Cc"}))
    payload = thaw_json(contract)
    payload["interactions"] = InteractionRegistry(keep).to_dict()
    keys = {row.interaction_id for row in keep}
    payload["reference_planes_mm"] = {key: value for key, value in contract["reference_planes_mm"].items() if key in keys}
    payload["counterfactual"] = "first_order_reference"
    payload["parent_contract_digest"] = contract["digest"]
    return _seal(payload)


def contract_for_slice(contract, lower, upper):
    """Retain shared scene identity and record the after-action checkpoint span."""
    if contract is None:
        return None
    from temsim.physics.interaction_registry import InteractionRegistry
    validate_plan_contract(contract)
    registry = InteractionRegistry.from_dict(contract["interactions"])
    records = []
    for row in registry.records:
        lo, hi = row.interval_mm
        if hi <= lower or lo > upper:
            continue
        records.append(replace(row, interval_mm=(max(lower, lo), min(upper, hi))))
    payload = thaw_json(contract)
    payload["interactions"] = InteractionRegistry(tuple(records)).to_dict()
    keys = {row.interaction_id for row in records}
    payload["reference_planes_mm"] = {key: value for key, value in contract["reference_planes_mm"].items() if key in keys}
    payload["interval_mm"] = (float(lower), float(upper))
    payload["parent_contract_digest"] = contract["digest"]
    payload["entrance_actions"] = "already executed at checkpoint"
    return _seal(payload)
