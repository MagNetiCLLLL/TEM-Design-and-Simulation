"""Fixed entrance plane and validated downstream-only specimen thickness edits."""
from __future__ import annotations

from dataclasses import is_dataclass
from hashlib import sha256
import math
from pathlib import Path

import numpy as np

from temsim import input_io
from temsim.immutable_json import thaw_json, json_digest


def _validate_faces(sample, assembly):
    parts = {part.key: part for part in assembly.parts}
    lower = max(parts[key].start_z_mm for key in ("sample_stage",))
    upper = min(parts[key].end_z_mm for key in ("sample_stage",))
    if "objective_upper_pole" in parts:
        lower = max(lower, parts["objective_upper_pole"].end_z_mm)
    if "objective_lower_pole" in parts:
        upper = min(upper, parts["objective_lower_pole"].start_z_mm)
    holder = parts.get("sample_holder")
    if holder is not None:
        half = float(holder.data["holder_tip_thickness_mm"])*.5
        lower, upper = max(lower, holder.center_z_mm-half), min(upper, holder.center_z_mm+half)
    # Retain the module's existing mechanical-coordinate tolerance. The
    # actual slab faces are checked in addition to its mechanical marker.
    tolerance = 1e-9
    if sample.upper_surface_z_mm < lower-tolerance or sample.lower_surface_z_mm > upper+tolerance:
        raise ValueError("The complete specimen thickness must remain inside the fixed holder, stage and Objective pole gap")


def anchored_sample_assembly(state, assembly, configuration, *, preserve_operating_parameters=True):
    """Resolve only the foil placement into a private archived column TOML."""
    from temsim.column.module_assembly import resolve_module_assembly
    from temsim.part_model_document import PartModelDocument
    sample = state.sample
    part = assembly.part("sample")
    # An explicit hardware installation chooses its new reference. Ordinary
    # catalog reloads must not be mistaken for user foil placement; an
    # intentional Place action first assigns sample.z_mm explicitly.
    if not preserve_operating_parameters:
        object.__setattr__(sample, "z_mm", float(part.center_z_mm))
        object.__setattr__(sample, "upper_surface_reference_z_mm", float(part.center_z_mm)-sample.thickness_nm*.5e-6)
    upper = sample.ensure_upper_surface_reference()
    target = upper+float(sample.thickness_nm)*.5e-6
    object.__setattr__(sample, "z_mm", target)
    _validate_faces(sample, assembly)
    tolerance = 4*abs(float(np.spacing(target)))
    if math.isclose(target, part.center_z_mm, rel_tol=0, abs_tol=tolerance):
        return assembly
    if getattr(state, "_resolved_assembly", None) is None:
        state._resolved_assembly = assembly
    archive = input_io.capture_input_archive(state)
    input_io.bind_archive(state, archive)
    column_path = Path(assembly.root)/dict(assembly.selected_module_paths)["column"]
    with input_io.input_scope(state):
        document = PartModelDocument(column_path)
        document.place_component("sample", float(part.data["local_center_z_mm"])+(target-part.center_z_mm),
                                 include_children=False)
        document.validate()
        content = document.independent_copy_text().encode("utf-8")
    payload = thaw_json(archive)
    original = input_io.archive_for(archive).original_path(column_path)
    row = next(row for row in payload["files"] if Path(row["path"]) == original)
    row.update(content_hex=content.hex(), sha256=sha256(content).hexdigest())
    payload["digest"] = json_digest({key: value for key, value in payload.items() if key != "digest"})
    input_io.bind_archive(state, payload)
    with input_io.input_scope(state):
        result = resolve_module_assembly(configuration, root=assembly.root)
    old = {item.key: item for item in assembly.parts}
    for item in result.parts:
        if item.key == "sample":
            continue
        original_part = old[item.key]
        if (item.data != original_part.data or item.start_z_mm != original_part.start_z_mm
                or item.center_z_mm != original_part.center_z_mm or item.end_z_mm != original_part.end_z_mm):
            raise ValueError("A thickness edit cannot move or change another physical part: "+item.key)
    if abs(result.part("sample").center_z_mm-target) > tolerance:
        raise ValueError("The archived sample placement did not preserve its fixed entrance")
    return result


def _mutable_graph(root):
    """Save references and container contents for atomic in-place rollback."""
    saved, seen = [], set()
    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            saved.append((value, dict(value), "dict"))
            for child in value.values():visit(child)
        elif isinstance(value, list):
            saved.append((value, list(value), "list"))
            for child in value:visit(child)
        elif isinstance(value, tuple):
            for child in value:visit(child)
        elif isinstance(value, set):
            saved.append((value, set(value), "set"))
        elif (type(value).__module__.startswith("temsim.") and hasattr(value, "__dict__")
              and not (is_dataclass(value) and type(value).__dataclass_params__.frozen)):
            saved.append((value, dict(vars(value)), "object"))
            for child in vars(value).values():visit(child)
    visit(root)
    return saved


@input_io.using_state_inputs
def set_sample_thickness(state, thickness_nm):
    """Atomically extend/retract the lower face, preserving the saved upper face.

    Normal layout and archived component placement validate the complete slab.
    Failed edits restore original objects, controls, archive and cached inputs;
    requested thickness is never clipped to a permitted value.
    """
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.design_sweep_execution import _runtime_operating_values
    if isinstance(thickness_nm, (bool, np.bool_)):
        raise ValueError("Specimen thickness must be finite and positive")
    value = float(thickness_nm)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Specimen thickness must be finite and positive")
    saved = _mutable_graph(state)
    old_thickness = float(state.sample.thickness_nm)
    old_upper = float(state.sample.upper_surface_z_mm)
    controls = _runtime_operating_values(state)
    try:
        state.sample.thickness_nm = value
        apply_physical_layout_to_state(state, preserve_operating_parameters=True)
        actual = _runtime_operating_values(state)
        # Thickness itself is the only user scalar changed by this operation.
        if "sample" in actual:
            actual["sample"]["thickness_nm"] = controls["sample"].get("thickness_nm", old_thickness)
        if actual != controls:
            raise ValueError("A specimen thickness edit changed an unrelated operating control")
        if abs(state.sample.upper_surface_z_mm-old_upper) > 4*abs(float(np.spacing(old_upper))):
            raise ValueError("A thickness edit moved the fixed specimen upper surface")
    except BaseException:
        for obj, previous, kind in reversed(saved):
            if kind == "object":obj.__dict__.clear();obj.__dict__.update(previous)
            elif kind == "list":obj[:]=previous
            else:obj.clear();obj.update(previous)
        raise
    return {"old_thickness_nm":old_thickness, "new_thickness_nm":value,
            "upper_surface_z_mm":state.sample.upper_surface_z_mm,
            "lower_surface_z_mm":state.sample.lower_surface_z_mm,
            "sample_center_z_mm":state.sample.z_mm,
            "policy":"Fixed upper surface; thickness changes extend only the downstream face"}


__all__ = ("set_sample_thickness", "anchored_sample_assembly")
