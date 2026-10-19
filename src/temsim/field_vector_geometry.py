"""Captured Physical Layout surfaces for component field-vector overlays.

Only the existing resolved-assembly mesh builder owns shape and placement.
Its global column XYZ millimetres are converted once to right-handed XYZ
metres (+Z downstream), matching ``FieldVectorScene``. A material mesh does
not extend a field's valid domain and does not imply a coil-resolved solve.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from temsim.assembly_model_3d import assembly_model_from_assembly
from temsim.component_representation import SHARED_DEFLECTOR_HOSTS, non_material_role, representation_note
from temsim.part_model_3d import TriangleMesh, _segments


def _immutable_array(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _mesh_metres(mesh):
    # All geometry coordinates, including optional semantic edge points, cross
    # the same unit boundary. Faces, colour and approximation labels are kept.
    edges = tuple(MappingProxyType({**edge, "vertices": _immutable_array(
        np.asarray(edge["vertices"], dtype=float) * 1e-3)}) for edge in mesh.edges)
    return replace(mesh, vertices=_immutable_array(mesh.vertices * 1e-3), edges=edges)


@dataclass(frozen=True)
class FieldVectorGeometry:
    meshes: tuple[TriangleMesh, ...] = ()
    source_mesh_keys: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))
    notes: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    source_notes: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))

    def select(self, source_keys=None):
        """Return the union once, without rebuilding geometry or fields."""
        keys = tuple(self.source_mesh_keys) if source_keys is None else tuple(dict.fromkeys(source_keys))
        unknown = set(keys) - set(self.source_mesh_keys)
        if unknown:
            raise ValueError(f"Unknown field source keys: {', '.join(sorted(map(str, unknown)))}")
        parts = {part for key in keys for part in self.source_mesh_keys[key]}
        return tuple(mesh for mesh in self.meshes if mesh.key in parts)


def _runtime_geometry(state):
    """Only geometry consumed by the existing builders, from this snapshot."""
    from temsim.energy_filter_model_3d import energy_filter_render_values

    values = {}
    for aperture in getattr(state, "apertures", ()):
        values[str(aperture.key)] = {
            name: getattr(aperture, name) for name in ("radius_mm", "offset_x_mm", "offset_y_mm")
            if hasattr(aperture, name)
        }
    values.update(energy_filter_render_values(state))
    gun = getattr(state, "electron_gun", None)
    if getattr(gun, "type_key", None) == "cold_feg":
        from temsim.tip_emission_view import emission_view_values
        emitter = gun.emitter
        surface = emitter.surface_model
        values["feg_tip"] = {
            "tip_surface_model": surface.to_dict() if surface is not None else None,
            "tip_analytic_emission": emission_view_values(emitter),
        }
    return values


def _physical_keys(component_keys, parts):
    selected, notes = set(), []
    for key in component_keys:
        key = str(key)
        row = parts.get(key)
        if row is None:
            continue
        role = non_material_role(row.data)
        if role:
            host = SHARED_DEFLECTOR_HOSTS.get(key)
            if not host and row.data.get("physical_host_status") == "modelled":
                host = row.data.get("physical_host_key")
            if host and host in parts and not non_material_role(parts[host].data):
                selected.add(str(host))
                notes.append(f"{key}: shared material host {host}; no duplicate channel body.")
            else:
                notes.append(f"{key}: {representation_note(row.data)}")
        else:
            selected.add(key)
    # This is exactly the declared descendant/shared-body relation used by
    # part_model_from_document(include_children=True), restricted to active
    # resolved parts. Optical-reference bindings alone are not material hosts.
    while True:
        descendants = {key for key, part in parts.items() if (
            part.parent_key in selected or selected.intersection(part.data.get("magnetic_lens_keys", ())))
            and not non_material_role(part.data)}
        if descendants <= selected:
            break
        selected.update(descendants)
    return selected, notes


def build_field_vector_geometry(state, sources, *, angular_segments=16):
    """Build once from the field result's captured assembly, never live TOML.

    Missing captured geometry stays unavailable. Shared circuits/channels map
    to the union of their physical hosts. Optical tilt is not applied again:
    the existing mechanical builder owns physical pose semantics, including
    the distinction between optical-axis tilt and material placement.
    """
    count = _segments(angular_segments)
    sources = tuple(sources)
    if len({str(source.key) for source in sources}) != len(sources):
        raise ValueError("Field source keys must be unique")
    assembly = getattr(state, "_resolved_assembly", None)
    if assembly is None:
        note = "Physical geometry unavailable: this field snapshot has no captured resolved assembly; current files are not substituted."
        empty = MappingProxyType({str(source.key): () for source in sources})
        return FieldVectorGeometry(source_mesh_keys=empty, notes=(note,),
                                   source_notes=MappingProxyType({key: (note,) for key in empty}))
    parts = {part.key: part for part in assembly.parts}
    model = assembly_model_from_assembly(assembly, runtime_values=_runtime_geometry(state), angular_segments=count)
    available = {mesh.key for mesh in model.meshes}
    mapping, source_notes = {}, {}
    for source in sources:
        selected, messages = _physical_keys(source.component_keys, parts)
        keys = selected & available
        mapping[str(source.key)] = tuple(sorted(keys))
        for message in model.notes:
            if any(message.startswith(key + ":") for key in selected):
                messages.append(message)
        messages.extend(message for message in model.errors if any(message.startswith(key + ":") for key in selected))
        if not keys:
            messages.append(f"{source.key}: no independently modelled material geometry is available for this captured source.")
        source_notes[str(source.key)] = tuple(dict.fromkeys(messages))
    used = {key for keys in mapping.values() for key in keys}
    meshes = tuple(_mesh_metres(mesh) for mesh in model.meshes if mesh.key in used)
    notes = (
        "Physical models reuse the captured Physical Layout meshes and configured material colours; global right-handed XYZ metres, +Z downstream.",
        "Material placement uses the existing mechanical pose. Optical-axis tilt does not rotate housings, poles or coils. Shared material hosts are drawn once.",
        "Declared mechanical children are included as assembly context; displaying their material does not select their independent field contributions.",
        "Configured / reconstructed geometry is not measured OEM CAD. Approximation labels are retained; model surfaces do not imply valid fields inside materials or outside the field provider domain.",
    )
    return FieldVectorGeometry(meshes, MappingProxyType(mapping), notes, model.errors, MappingProxyType(source_notes))
