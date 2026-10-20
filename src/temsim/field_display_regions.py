"""Captured physical observation envelopes, independent of field providers.

These envelopes include the beam bore. They limit where a selected field sum
is displayed; they neither mask individual contributions before summation nor
change transport, material, vacuum or field-validity boundaries. No mesh,
provider evaluation, file read or field solve is needed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType, SimpleNamespace

import numpy as np

from temsim.component_representation import SHARED_DEFLECTOR_HOSTS, non_material_role


def _array(value):
    values = np.asarray(value, dtype=float)
    return np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)


def _points(value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Observation points must be finite N by 3 global XYZ metres")
    return points


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class FieldDisplayRegion:
    """One outer cylinder/box, with the exact captured affine placement.

    Native cylinders use declared mechanical envelopes, not inferred material
    occupancy. Explicit CAD primitives retain their box/elliptic-cylinder
    shape. Inner bores and cutting holes are intentionally part of the
    observation volume.
    """

    component_key: str
    label: str
    shape: str
    bounds_m: np.ndarray = field(repr=False, compare=False)
    origin_m: np.ndarray = field(repr=False, compare=False)
    inverse_linear: np.ndarray = field(repr=False, compare=False)
    half_extents_m: np.ndarray = field(repr=False, compare=False)
    note: str = ""
    axial_radius_profile_m: tuple[tuple[float, float], ...] = ()

    def contains(self, points_m):
        points = _points(points_m)
        # Local geometry, not the global AABB, decides membership after a tilt.
        local = np.einsum("...j,ij->...i", points - self.origin_m,
                          self.inverse_linear, optimize=False)
        scaled = local / self.half_extents_m
        tolerance = 32 * np.finfo(float).eps
        axial = np.abs(scaled[:, 2]) <= 1. + tolerance
        if self.shape == "box":
            return axial & np.all(np.abs(scaled[:, :2]) <= 1. + tolerance, axis=1)
        if self.axial_radius_profile_m:
            profile = np.asarray(self.axial_radius_profile_m)
            radius = np.interp(local[:, 2], profile[:, 0], profile[:, 1])
            return axial & (np.sum(local[:, :2] ** 2, axis=1) <= radius ** 2 * (1. + tolerance))
        return axial & (np.sum(scaled[:, :2] ** 2, axis=1) <= 1. + tolerance)


@dataclass(frozen=True)
class FieldDisplayRegions:
    regions: tuple[FieldDisplayRegion, ...] = ()
    source_keys: tuple[str, ...] = ()
    component_source_keys: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({}))
    component_region_keys: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({}))
    notes: tuple[str, ...] = ()
    component_notes: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({}))

    def source_keys_for_component(self, key):
        """Return complete captured source groups, never a divided joint field."""
        return self.component_source_keys.get(str(key), ())

    def regions_for_component(self, key=None):
        """None observes all physical envelopes; a key observes only that part."""
        if key is None:
            return self.regions
        keys = self.component_region_keys.get(str(key), ())
        return tuple(region for region in self.regions if region.component_key in keys)

    def mask(self, points_m, component_key=None):
        points = _points(points_m)
        result = np.zeros(len(points), dtype=bool)
        for region in self.regions_for_component(component_key):
            result |= region.contains(points)
        return result

    def bounds_for_component(self, key=None):
        regions = self.regions_for_component(key)
        if not regions:
            return None
        return _array((np.min([r.bounds_m[0] for r in regions], axis=0),
                       np.max([r.bounds_m[1] for r in regions], axis=0)))


_MAGNETIC_HARDWARE = frozenset({
    "magnetic_lens_assembly", "magnetic_pole_piece", "magnetic_excitation_coil",
    "magnetic_lens_yoke", "magnetic_lens_housing", "c1_c2_pole_piece_cartridge",
})
_CUSTOM_ROLES = frozenset({"custom_mechanical", "custom_mechanical_copy"})


def _host_key(row):
    key = str(row["key"])
    host = SHARED_DEFLECTOR_HOSTS.get(key)
    if host:
        return host
    if (non_material_role(row) == "control_channel"
            and row.get("physical_host_status") == "modelled"):
        return str(row.get("physical_host_key", ""))
    return ""


def _component_sources(rows, sources):
    """Use explicit ownership; navigation parents alone never create a field."""
    from temsim.magnetic_circuits import optical_owner

    direct = {}
    for source in sources:
        if getattr(source, "model", "") == "reference_zero":
            continue
        for key in getattr(source, "component_keys", ()):
            direct.setdefault(str(key), set()).add(str(source.key))
    reverse_hosts = {}
    for key, row in rows.items():
        host = _host_key(row)
        if host and host in rows and not non_material_role(rows[host]):
            reverse_hosts.setdefault(host, set()).update(direct.get(key, ()))
    mapping = {}
    for key, row in rows.items():
        role = non_material_role(row)
        if (row.get("mechanical_part_role") in _CUSTOM_ROLES
                or role == "virtual_reference"
                or row.get("mechanical_profile") == "objective_electronics_sleeve"):
            mapping[key] = ()
            continue
        selected = set(direct.get(key, ())) | reverse_hosts.get(key, set())
        host = _host_key(row)
        if host and host in rows:
            selected.update(direct.get(host, ()))
            selected.update(reverse_hosts.get(host, ()))
        if not role and row.get("mechanical_profile") in _MAGNETIC_HARDWARE:
            owners = {str(row.get("field_source_key", "")), *map(str, row.get("magnetic_lens_keys", ()))}
            owner = optical_owner(row, rows)
            if owner:
                owners.add(owner)
            for owner in owners:
                selected.update(direct.get(owner, ()))
        # Keep the source order, which also prevents duplicate shared fields.
        mapping[key] = tuple(str(source.key) for source in sources if str(source.key) in selected)
    return mapping


def _region(part, row, by_key):
    from temsim.cad_vacuum_tube import PROFILE
    from temsim.energy_filter_model_3d import supports_energy_filter_part
    from temsim.lens_pose import mechanical_part_transform_mm, without_lens_rigid_cad
    from temsim.part_model_3d import _extent
    from temsim.part_model_features import _transform

    if non_material_role(row) or row.get("virtual_only"):
        return None, "No independent physical observation envelope for this control/reference."
    if row.get("branch_path_only") or supports_energy_filter_part(row):
        return None, "Bent-path physical envelope is not available in this straight-column observation index."
    shape_row = without_lens_rigid_cad(row)
    model = shape_row.get("model_3d", {})
    base = model.get("base", {})
    kind = base.get("kind", "existing")
    centre = np.array((0., 0., _number(row.get("local_center_z_mm"), "local_center_z_mm")))
    profile = ()
    if kind == "step":
        return None, "Imported STEP outer envelope is unavailable without its captured shape; no field-support substitute is used."
    if kind in {"box", "elliptic_cylinder"}:
        half = .5 * np.array([_number(base.get(name), name)
                              for name in ("width_mm", "height_mm", "length_mm")])
        primitive_centre = centre
        shape = "box" if kind == "box" else "elliptic_cylinder"
        note = "Explicit CAD outer primitive, including internal holes, with captured affine placement."
    elif kind == "existing":
        start, end = _extent(row)
        diameter = _number(row.get("mechanical_outer_diameter_mm"), "mechanical_outer_diameter_mm")
        half = np.array((.5 * diameter, .5 * diameter, .5 * (end - start)))
        primitive_centre = np.array((0., 0., .5 * (start + end)))
        shape = "elliptic_cylinder"
        note = "Declared mechanical outer cylinder including bore; an observation envelope, not material occupancy or OEM geometry."
        if PROFILE in row:
            nodes = np.asarray(row[PROFILE], dtype=float)
            if (nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) < 2
                    or not np.isfinite(nodes).all() or np.any(np.diff(nodes[:, 0]) <= 0)
                    or np.any(nodes[:, 1] <= 0) or np.any(nodes[:, 1] > diameter)
                    or abs(nodes[0, 0] - start) > 1e-9 or abs(nodes[-1, 0] - end) > 1e-9):
                raise ValueError("Physical outer profile must contain positive ordered Z/diameter pairs covering its axial extent")
            half[:2] = .5 * np.max(nodes[:, 1])
            profile = tuple((float(z - primitive_centre[2]) * 1e-3, float(od) * .5e-3)
                            for z, od in nodes)
            note = "Captured piecewise-linear outer tube profile including bore; openings do not remove the observation volume."
    else:
        return None, f"Unsupported physical outer-envelope base: {kind}."
    if not np.isfinite(half).all() or np.any(half <= 0):
        return None, "No positive three-dimensional physical envelope is declared."
    # Exactly the same transform order as native CAD: local shape about the
    # part centre, then mechanical placement, then resolved module origin.
    affine, offset = _transform(shape_row)
    rotation, translation = mechanical_part_transform_mm(row, by_key)
    affine = np.asarray(affine, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if (affine.shape != (3, 3) or offset.shape != (3,)
            or not np.isfinite(affine).all() or not np.isfinite(offset).all()):
        raise ValueError("Physical CAD transform must be finite")
    linear = rotation @ affine
    inverse = np.linalg.inv(linear)
    if not np.isfinite(inverse).all():
        raise ValueError("Physical CAD inverse transform must be finite")
    shift = _number(part.center_z_mm, "resolved centre") - centre[2]
    origin_mm = rotation @ (affine @ (primitive_centre - centre) + centre + offset) + translation
    origin_mm += (0., 0., shift)
    origin, half_m = origin_mm * 1e-3, half * 1e-3
    # Tight support-function AABB of the transformed filled cylinder/box.
    if profile:
        nodes = np.asarray(profile)
        axial_offsets = nodes[:, 0, None] * linear[:, 2]
        radial_extents = nodes[:, 1, None] * np.hypot(linear[:, 0], linear[:, 1])
        bounds = np.stack((origin + np.min(axial_offsets - radial_extents, axis=0),
                           origin + np.max(axial_offsets + radial_extents, axis=0)))
        extent = .5 * (bounds[1] - bounds[0])
    elif shape == "box":
        extent = np.abs(linear) @ half_m
    else:
        extent = np.hypot(linear[:, 0] * half_m[0], linear[:, 1] * half_m[1])
        extent += np.abs(linear[:, 2]) * half_m[2]
    if not profile:
        bounds = np.array((origin - extent, origin + extent))
    if not np.isfinite(origin).all() or not np.isfinite(extent).all() or np.any(extent <= 0):
        raise ValueError("Placed observation envelope must remain finite with positive extent")
    if model.get("features"):
        note += " Subtractive features do not remove the internal observation volume."
    return FieldDisplayRegion(str(part.key), str(row.get("name", part.key)), shape,
                              _array(bounds), _array(origin),
                              _array(inverse), _array(half_m), note, profile), ""


def build_field_display_regions(state, sources):
    """Capture only resolved geometry and source ownership, without sampling.

    Selecting a passive body can limit a total-field view to its envelope,
    while ``source_keys_for_component`` remains empty. Missing physical shape
    never falls back to a field's numerical support. Optical-axis tilts retain
    the existing Mechanical Layout semantics: they do not rotate material.
    """
    sources = tuple(sources)
    source_keys = tuple(str(source.key) for source in sources)
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("Field source keys must be unique")
    assembly = getattr(state, "_resolved_assembly", None)
    if assembly is None:
        return FieldDisplayRegions(source_keys=source_keys, notes=(
            "Physical observation envelopes unavailable: no captured resolved assembly; current files and field support are not substituted.",))
    parts = tuple(assembly.parts)
    if len({part.key for part in parts}) != len(parts):
        raise ValueError("Assembly part keys must be unique")
    contexts = {module.key: {p.key: _plain(p.data) for p in module.parts}
                for module in getattr(assembly, "modules", ())}
    rows = {part.key: _plain(part.data) for part in parts}
    for part in parts:
        contexts.setdefault(part.module_key, {})[part.key] = rows[part.key]
    notes = [
        "Display masks are unions of captured physical outer envelopes, including bores; apply only after the chosen complete field sum.",
        "Masks never extend a provider's valid domain or alter fields, vacuum boundaries or particle propagation.",
        "Mechanical placement matches Physical Layout; optical-axis tilt does not rotate material, while CAD shape rotations and offsets do.",
    ]
    # Physical Layout adds explicitly independent tube segments outside
    # assembly.parts. Equivalent numerical component bores are not material.
    from temsim.cad_stage_holder import physical_liner_rows
    try:
        liners = [row for row in physical_liner_rows(assembly) if row["key"] not in rows]
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        liners = []
        notes.append(f"Independent physical tube envelopes unavailable: {error}")
    if liners:
        liner_context = {}
        for row in liners:
            key = row["key"]
            rows[key] = _plain(row)
            liner_context[key] = rows[key]
        # Synthetic tube rows already carry global column coordinates.
        parts += tuple(SimpleNamespace(key=row["key"], module_key="@physical_tubes",
                                       center_z_mm=row["local_center_z_mm"]) for row in liners)
        contexts["@physical_tubes"] = liner_context
    source_mapping = _component_sources(rows, sources)
    for row in liners:
        source_mapping[row["key"]] = ()
    regions, region_mapping, component_notes = [], {}, {}
    for part in parts:
        row = rows[part.key]
        try:
            region, reason = _region(part, row, contexts[part.module_key])
        except (ValueError, TypeError, KeyError, OverflowError, np.linalg.LinAlgError) as error:
            region, reason = None, str(error)
        if region is None:
            region_mapping[part.key] = ()
            component_notes[part.key] = (reason,)
            notes.append(f"{part.key}: {reason}")
        else:
            regions.append(region)
            region_mapping[part.key] = (part.key,)
            component_notes[part.key] = (region.note,)
    # Non-material shared channels can observe their declared real host, but
    # optical-reference-only associations do not establish a material envelope.
    for key, row in rows.items():
        host = _host_key(row)
        if host and host in rows:
            region_mapping[key] = region_mapping.get(host, ())
            component_notes[key] = (*component_notes[key], f"Physical observation host: {host}.")
    return FieldDisplayRegions(tuple(regions), source_keys, MappingProxyType(source_mapping),
                               MappingProxyType(region_mapping), tuple(dict.fromkeys(notes)),
                               MappingProxyType(component_notes))


__all__ = ["FieldDisplayRegion", "FieldDisplayRegions", "build_field_display_regions"]
