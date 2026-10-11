"""Explicit, opt-in geometry ownership for modelled corrector coil inserts.

An optical channel keeps its drive and key; its declared mechanical host owns
placement and finite field length. These are not extra deflector drives.
Documents without these declarations retain their historical coordinates.
"""
from __future__ import annotations

import math
from numbers import Real

from temsim.component_representation import SHARED_DEFLECTOR_HOSTS, non_material_role


COIL_INSERT_ROLE = "deflector_coil_insert"
_GEOMETRY = (
    "local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "length_mm",
    "optical_reference_local_z_mm", "effective_thickness_mm",
    "mechanical_outer_diameter_mm", "mechanical_clear_bore_diameter_mm",
    "vacuum_inner_diameter_mm",
)
_POSE = ("offset_x_mm", "offset_y_mm", "offset_z_mm",
         "rotation_x_mrad", "rotation_y_mrad", "rotation_z_mrad")


def field_hardware_host_key(part):
    """Return only an explicitly declared modelled host, never infer one."""
    if str(part.get("key", "")) in SHARED_DEFLECTOR_HOSTS:
        return ""
    return str(part.get("physical_host_key", "")) if non_material_role(part) == "control_channel" else ""


def bound_field_owner(part, field):
    """Name the authoritative row for UI edits of a derived coordinate/value."""
    field = str(field)
    host = field_hardware_host_key(part)
    if host and field in {*_GEOMETRY, *_POSE, "model_3d", "parent_key"}:
        return host
    if (part.get("optical_reference_host_key") and field in {
            "local_start_z_mm", "local_center_z_mm", "local_end_z_mm",
            "optical_reference_local_z_mm"}):
        return str(part["optical_reference_host_key"])
    return ""


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def resolve_field_hardware_parts(document):
    """Resolve host references on a copy, idempotently, before all consumers.

    ``optical_reference_host_key`` fixes an insert or control channel to its host's
    optical reference plus ``optical_reference_offset_mm`` (default zero).
    Material dimensions remain those of the insert, not those of the HP.
    """
    original = document.get("parts", ())
    if not any(field_hardware_host_key(p) or p.get("optical_reference_host_key") for p in original):
        return document
    rows = {str(p["key"]): p for p in original}
    if len(rows) != len(original):
        raise ValueError("Duplicate part key in field hardware bindings")
    resolved, visiting = {}, set()

    def resolve(key):
        if key in resolved:
            return resolved[key]
        if key in visiting:
            raise ValueError(f"Cyclic field hardware binding: {key}")
        if key not in rows:
            raise ValueError(f"Missing field hardware host: {key}")
        visiting.add(key)
        part = dict(rows[key])
        reference_host = part.get("optical_reference_host_key")
        if reference_host:
            optical_channel = non_material_role(part) == "control_channel"
            if not optical_channel and (part.get("mechanical_part_role") != COIL_INSERT_ROLE
                                        or part.get("mechanical_only") is not True):
                raise ValueError(f"{key}: optical reference binding requires a coil insert or control channel")
            if optical_channel and field_hardware_host_key(part):
                raise ValueError(f"{key}: bind the physical insert, not both insert and channel")
            host = resolve(str(reference_host))
            if non_material_role(host):
                raise ValueError(f"{key}: optical reference host must be a physical optical component")
            centre = _finite(host.get("optical_reference_local_z_mm"), f"{reference_host}.optical_reference_local_z_mm")
            centre += _finite(part.get("optical_reference_offset_mm", 0.), f"{key}.optical_reference_offset_mm")
            length = _finite(part.get("length_mm"), f"{key}.length_mm")
            if length <= 0.:
                raise ValueError(f"{key}: anchored insert needs a positive length")
            # A channel binding establishes the optical Z relation only. It
            # does not establish a material host, winding layout or extra field.
            # Length is independently editable; endpoints follow the anchored
            # midpoint. Never reuse stored, possibly stale derived endpoints
            # after the HP has moved or an editor has resized the insert.
            part["local_start_z_mm"] = centre - length / 2.
            part["local_end_z_mm"] = centre + length / 2.
            part["local_center_z_mm"] = part["optical_reference_local_z_mm"] = centre
        host_key = field_hardware_host_key(part)
        if host_key:
            host = resolve(host_key)
            if (part.get("physical_host_status") != "modelled"
                    or host.get("mechanical_part_role") != COIL_INSERT_ROLE
                    or host.get("mechanical_only") is not True or non_material_role(host)):
                raise ValueError(f"{key}: physical host must be a declared modelled mechanical coil insert")
            # Binding is explicit, so an old independent placement cannot be
            # silently discarded or added to the physical host a second time.
            transform = part.get("model_3d", {}).get("transform", {})
            own_pose = [part.get(name, 0.) for name in _POSE]
            own_pose += list(transform.get("offset_mm", ())) + list(transform.get("rotation_deg", ()))
            if any(_finite(value, f"{key}.pose") != 0. for value in own_pose):
                raise ValueError(f"{key}: edit the physical host pose ({host_key}), not its channel")
            for name in _GEOMETRY:
                part[name] = _finite(host.get(name), f"{host_key}.{name}")
            if not 0. < part["effective_thickness_mm"] <= part["length_mm"]:
                raise ValueError(f"{host_key}: effective coil thickness must fit the mechanical length")
            part["parent_key"] = host_key
        visiting.remove(key)
        resolved[key] = part
        return part

    result = dict(document)
    result["parts"] = [resolve(str(part["key"])) for part in original]
    return result
