"""Immutable input metadata for shared column geometry and optical ownership.

This P0 record does not solve fields, construct a State, resolve a layout, or
replace a numerical propagation identity. It describes the already-resolved
geometry and current runtime inputs. External files are not loaded; arrays are
content-addressed rather than embedded. Executed fields, source populations,
solver approximations and checkpoint phase references remain separate owners.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import sha256
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from temsim.component_representation import SHARED_DEFLECTOR_HOSTS, non_material_role
from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.instrument_snapshot import _RUNTIME_NAMES
from temsim.magnetic_circuits import circuit_inventory
from temsim.optics.field_hardware_binding import field_hardware_host_key


PHYSICAL_SCENE_SCHEMA = "physical-scene-metadata-v1"
_SCOPE = "Resolved geometry and declared optical inputs; not a complete numerical or executed-field identity"
_COLLECTIONS = ("lenses", "apertures", "stigmators", "deflectors",
                "corrector_elements", "recording_planes")
_SINGLE_COMPONENTS = ("sample", "camera", "nanopulser", "energy_filter")
_SETTINGS = (
    "beam_voltage_kv", "simulation_mode", "illumination_mode", "nanoprobe_range",
    "projector_mode", "equivalent_image_lenses_enabled", "chromatic_aberration_enabled",
    "simulation_time_s", "lens_intrinsic_aberrations", "probe_aberrations", "image_aberrations",
    "lens_field_map_descriptors", "component_placements", "energy_filter_installed",
    "corrector_mode", "image_corrector_design", "column_mode", "objective_coupled",
)
_ANCHORS = ("_upper_objective_package_resolved_positions_mm", "_ac_downstream_resolved_positions_mm",
            "_module_optical_offsets_mm", "_module_gun_offsets_mm")
_PRESENTATION = frozenset({"colour", "color", "label", "name"})


def _type_name(value):
    return type(value).__module__ + ":" + type(value).__qualname__


def _attributes(value):
    """Read stored data only, never enumerate/evaluate arbitrary properties."""
    result = dict(vars(value)) if hasattr(value, "__dict__") else {}
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            if item.name not in result:
                result[item.name] = getattr(value, item.name)
    return {key: item for key, item in result.items()
            if key not in _RUNTIME_NAMES and key not in _PRESENTATION}


class _Capture:
    def __init__(self):
        self.unresolved = []
        self.active = set()

    def unknown(self, path, value, reason):
        note = {"path": path, "type": _type_name(value), "reason": reason}
        self.unresolved.append(note)
        return {"unresolved": note}

    def value(self, value, path):
        if value is None or isinstance(value, (str, bool, int, np.bool_, np.integer)):
            return value.item() if isinstance(value, np.generic) else value
        if isinstance(value, (float, np.floating)):
            number = float(value)
            return number if math.isfinite(number) else {"float": number.hex()}
        if isinstance(value, Path):
            return {"path": str(value)}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.fields is not None:
                return self.unknown(path, value, "Object or structured array is not captured")
            return {"array_sha256": sha256(value.tobytes(order="C")).hexdigest(),
                    "dtype": value.dtype.str, "shape": list(value.shape)}
        if callable(value):
            return self.unknown(path, value, "Callable behavior is not an input record")
        identity = id(value)
        if identity in self.active:
            return self.unknown(path, value, "Cyclic input graph is outside this metadata scope")
        self.active.add(identity)
        try:
            if isinstance(value, Mapping):
                if any(not isinstance(key, str) for key in value):
                    return self.unknown(path, value, "Metadata mappings require string keys")
                return {key: self.value(item, path+"."+key) for key, item in sorted(value.items())}
            if isinstance(value, (list, tuple)):
                return [self.value(item, f"{path}[{index}]") for index, item in enumerate(value)]
            if isinstance(value, (set, frozenset)):
                result = [self.value(item, path+"[]") for item in value]
                return sorted(result, key=json_digest)
            attributes = _attributes(value)
            if attributes or is_dataclass(value) or isinstance(value, SimpleNamespace):
                if not (type(value).__module__.startswith("temsim.") or isinstance(value, SimpleNamespace)):
                    self.unknown(path, value, "Custom provider behavior is not qualified by stored attributes")
                return {"type": _type_name(value), "attributes": self.value(attributes, path)}
            return self.unknown(path, value, "Opaque input has no declared stored parameters")
        finally:
            self.active.remove(identity)


def _component_key(component, fallback):
    key = component.get("key", fallback) if isinstance(component, Mapping) else getattr(component, "key", fallback)
    if not isinstance(key, str) or not key:
        raise ValueError("Scene components require a nonempty string key")
    return key


def _geometry_attributes(component):
    attributes = dict(component) if isinstance(component, Mapping) else _attributes(component)
    # These are declared lengths/poses, not inferred solids. Excitation and
    # material activation are intentionally independent of geometry existence.
    return {key: value for key, value in attributes.items() if (
        key.endswith(("_mm", "_um", "_nm")) and key not in {"cs_mm", "cc_mm"}
        or key.endswith(("_mrad", "_rad", "_deg")) and any(
            token in key for token in ("rotation", "orientation", "tilt", "alpha", "beta"))
        or key in {"key", "installed", "parent_key", "model_3d", "material_regions",
                   "physical_host_key", "physical_host_status", "optical_reference_host_key",
                   "_field_hardware_host_key"})}


@dataclass(frozen=True, slots=True)
class PhysicalSceneSnapshot:
    """Detached metadata; ``identity_complete`` never certifies a field model."""
    payload: Mapping
    digest: str = field(init=False)
    geometry_digest: str = field(init=False)

    def __post_init__(self):
        payload = freeze_json(self.payload)
        required = {"scope", "geometry", "components", "component_order", "settings", "gun_inputs",
                    "ownership", "circuits", "unresolved_inputs", "identity_complete"}
        if not isinstance(payload, Mapping) or set(payload) != required or payload["scope"] != _SCOPE:
            raise ValueError("Invalid physical-scene metadata payload")
        if type(payload["identity_complete"]) is not bool or payload["identity_complete"] != (not payload["unresolved_inputs"]):
            raise ValueError("Physical-scene completeness disagrees with unresolved inputs")
        if not isinstance(payload["geometry"], Mapping) or not isinstance(payload["ownership"], Mapping):
            raise ValueError("Physical-scene geometry and ownership must be mappings")
        for key, row in payload["ownership"].items():
            if not key or not isinstance(row, Mapping) or not {"owner_key", "basis", "role"} <= set(row):
                raise ValueError("Invalid physical-scene ownership row")
            owner = row["owner_key"]
            if owner is not None and (not isinstance(owner, str) or not owner):
                raise ValueError("Invalid physical-scene owner key")
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "geometry_digest", json_digest(payload["geometry"]))
        object.__setattr__(self, "digest", json_digest({"schema": PHYSICAL_SCENE_SCHEMA, "payload": payload}))

    @property
    def identity_complete(self):
        return self.payload["identity_complete"]

    def owner_for(self, key: str) -> str | None:
        """Declared host/action owner, never evidence of an independent solid.

        Unbound control channels, virtual references and unknown keys have no
        physical owner. Ordinary runtime-only components own their action key;
        the ownership row explicitly states that their body is not established.
        """
        row = self.payload["ownership"].get(key)
        return row["owner_key"] if row is not None else None

    def to_dict(self):
        return {"schema": PHYSICAL_SCENE_SCHEMA, "payload": thaw_json(self.payload),
                "digest": self.digest, "geometry_digest": self.geometry_digest}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping) or set(data) != {"schema", "payload", "digest", "geometry_digest"}:
            raise ValueError("Invalid physical-scene record")
        if data["schema"] != PHYSICAL_SCENE_SCHEMA:
            raise ValueError("Unsupported physical-scene schema")
        result = cls(data["payload"])
        if result.digest != data["digest"] or result.geometry_digest != data["geometry_digest"]:
            raise ValueError("Physical-scene content digest mismatch")
        return result


def capture_physical_scene(state) -> PhysicalSceneSnapshot:
    """Capture existing inputs without layout resolution, field solves or mutation.

    This bounded inventory is supplementary provenance. Unlisted State controls,
    external asset contents and numerical method identities remain the existing
    working-point/field/propagation contracts' responsibility.
    """
    capture = _Capture()
    components, runtime_geometry, runtime_rows, component_order = {}, {}, {}, {}
    for name in (*_COLLECTIONS, *_SINGLE_COMPONENTS):
        value = getattr(state, name, None)
        members = (() if value is None else (value,)) if name in _SINGLE_COMPONENTS else (value or ())
        component_order[name] = []
        for index, component in enumerate(members):
            key = _component_key(component, name if name in _SINGLE_COMPONENTS else f"{name}[{index}]")
            component_order[name].append(key)
            encoded = capture.value(component, "components."+key)
            if key in components:
                if components[key]["inputs"] != encoded:
                    raise ValueError(f"Conflicting runtime components share scene key {key}")
                components[key]["collections"].append(name)
            else:
                components[key] = {"collections": [name], "inputs": encoded}
                attributes = dict(component) if isinstance(component, Mapping) else _attributes(component)
                runtime_rows[key] = {"key": key, **{name: attributes[name] for name in (
                    "layout_role", "mechanical_profile", "physical_host_key", "physical_host_status",
                    "_field_hardware_host_key") if name in attributes}}
                runtime_geometry[key] = capture.value(_geometry_attributes(component), "geometry.runtime."+key)
    assembly = getattr(state, "_resolved_assembly", None)
    parts, part_rows = {}, {}
    for part in getattr(assembly, "parts", ()):
        data = dict(part) if isinstance(part, Mapping) else dict(part.data)
        key = _component_key(part, data.get("key"))
        if key in parts:
            raise ValueError(f"Duplicate resolved scene part {key}")
        data["key"] = key
        if not isinstance(part, Mapping):
            data["parent_key"] = part.parent_key
        part_rows[key] = data
        global_geometry = ({name: getattr(part, name) for name in (
            "module_key", "source_file", "start_z_mm", "center_z_mm", "end_z_mm", "length_mm", "parent_key")
            if hasattr(part, name)} if not isinstance(part, Mapping) else {})
        parts[key] = {"definition": capture.value(data, "geometry.parts."+key),
                      "resolved": capture.value(global_geometry, "geometry.resolved."+key)}
    ownership = {}
    for key in sorted(set(parts) | set(components)):
        row = {**runtime_rows.get(key, {"key": key}), **part_rows.get(key, {})}
        role = non_material_role(row)
        host = SHARED_DEFLECTOR_HOSTS.get(key) or field_hardware_host_key(row) or row.get("_field_hardware_host_key")
        if host:
            basis = "declared physical host"
            if host not in parts and host not in components:
                capture.unknown("ownership."+key, row, "Declared physical host is absent from this snapshot")
            elif non_material_role({**runtime_rows.get(host, {"key": host}), **part_rows.get(host, {})}):
                capture.unknown("ownership."+key, row, "Declared physical host is a non-material channel/reference")
        elif role:
            host, basis = None, "no independent physical host declared"
        else:
            host = key
            basis = "resolved material/reference component" if key in parts else "runtime action owner; material body not established"
        ownership[key] = {"owner_key": host, "basis": basis, "role": role or "component"}
    circuits = capture.value(circuit_inventory(tuple(part_rows.values())), "circuits") if parts else []
    geometry = {"parts": parts, "part_order": list(parts), "runtime": runtime_geometry,
                "vacuum_bore_segments": capture.value(getattr(assembly, "vacuum_bore_segments", ()), "geometry.vacuum_bores"),
                "vacuum_liner_segments": capture.value(getattr(assembly, "vacuum_liner_segments", ()), "geometry.vacuum_liners"),
                "anchors": capture.value({name: getattr(state, name) for name in _ANCHORS if hasattr(state, name)}, "geometry.anchors")}
    settings = capture.value({name: getattr(state, name) for name in _SETTINGS if hasattr(state, name)}, "settings")
    gun = capture.value(getattr(state, "electron_gun", None), "gun_inputs")
    payload = {"scope": _SCOPE, "geometry": geometry, "components": components,
               "component_order": component_order, "settings": settings,
               "gun_inputs": gun, "ownership": ownership, "circuits": circuits,
               "unresolved_inputs": sorted(capture.unresolved, key=lambda row: (row["path"], row["reason"])),
               "identity_complete": not capture.unresolved}
    return PhysicalSceneSnapshot(payload)
