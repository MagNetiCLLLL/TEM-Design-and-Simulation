"""Narrow metadata compatibility for an unchanged executed particle boundary.

Only initializing the specimen entrance/lattice references is permitted. This helper
does not restore an old solver, move matter, change source ancestry, or relax
the physical input comparison used by a new wave calculation.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from temsim.immutable_json import json_digest, thaw_json


_SAMPLE_TYPE = "temsim.optics.model:Sample"
_ANCHOR = "upper_surface_reference_z_mm"
_LATTICE_REFERENCE = "lattice_reference_thickness_nm"


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"Specimen entrance metadata migration requires finite {name}")
    return float(value)


def compare_wave_input_graphs(original_graph, current_graph):
    """Return an exact/migrated receipt, rejecting any other graph change.

    Input graphs are never modified. Canonical JSON comparison preserves
    numeric types, signed zero, reference topology and the order of schema
    lists. Old missing/None entrance/lattice references may become the old
    actual upper surface/current thickness, within four floating-point spacings.
    Actual specimen centre and thickness must remain exactly unchanged.
    """
    if json_digest(original_graph) == json_digest(current_graph):
        return {"schema": "wave-input-graph-compatibility-v1",
                "physical_inputs_unchanged": True, "metadata_migrations": []}
    original, current = deepcopy(thaw_json(original_graph)), deepcopy(thaw_json(current_graph))
    if not isinstance(original, dict) or not isinstance(current, dict):
        raise ValueError("Wave input graphs differ from the captured physical inputs")
    old_nodes, new_nodes = original.get("nodes"), current.get("nodes")
    if (not isinstance(old_nodes, list) or not isinstance(new_nodes, list)
            or len(old_nodes) != len(new_nodes)):
        raise ValueError("Wave input object topology differs from the captured physical inputs")
    migrations = []
    for index, (old, new) in enumerate(zip(old_nodes, new_nodes)):
        if (not isinstance(old, Mapping) or not isinstance(new, Mapping)
                or old.get("type") != _SAMPLE_TYPE or new.get("type") != _SAMPLE_TYPE):
            continue
        old_attributes, new_attributes = old.get("attributes"), new.get("attributes")
        if not isinstance(old_attributes, dict) or not isinstance(new_attributes, dict):
            continue
        for field in (_ANCHOR, _LATTICE_REFERENCE):
            previous, value = old_attributes.get(field), new_attributes.get(field)
            if previous is not None or value is None:
                continue
            center = _finite_real(old_attributes.get("z_mm"), "centre coordinate")
            thickness = _finite_real(old_attributes.get("thickness_nm"), "thickness")
            if thickness < 0:
                raise ValueError("Specimen entrance metadata cannot migrate a negative thickness")
            upper = center-thickness*.5e-6
            expected = upper if field == _ANCHOR else thickness
            value = _finite_real(value, "entrance/lattice reference")
            tolerance = 4*abs(float(np.spacing(expected)))
            if not math.isfinite(expected) or not math.isclose(value, expected, rel_tol=0., abs_tol=tolerance):
                raise ValueError("New specimen reference metadata differs from the old actual upper surface or lattice thickness")
            old_fields, new_fields = old.get("fields"), new.get("fields")
            if (not isinstance(old_fields, list) or not isinstance(new_fields, list)
                    or old_fields.count(field) > 1 or new_fields.count(field) != 1):
                raise ValueError("Specimen reference metadata has an unsupported field schema")
            row = {"node_index": index, "model_type": _SAMPLE_TYPE, "field": field,
                "previous_value": None, "previous_attribute_present": field in old_attributes,
                "actual_center_z_mm": center, "actual_thickness_nm": thickness,
                "scope": "Reference metadata initialized; actual geometry, original lattice and all other inputs unchanged"}
            if field == _ANCHOR:
                row.update(new_value_mm=value, old_actual_upper_surface_z_mm=upper,
                           round_trip_tolerance_mm=tolerance)
            else:
                row.update(new_reference_thickness_nm=value, old_lattice_origin_depth_nm=thickness*.5,
                           new_lattice_origin_depth_nm=value*.5, round_trip_tolerance_nm=tolerance)
            migrations.append(row)
            # Remove only admitted metadata on detached comparison copies.
            old_attributes.pop(field, None)
            new_attributes.pop(field)
            old["fields"] = [name for name in old_fields if name != field]
            new["fields"] = [name for name in new_fields if name != field]
    if not migrations or json_digest(original) != json_digest(current):
        raise ValueError("Wave inputs differ from the captured physical inputs beyond entrance-reference metadata")
    return {"schema": "wave-input-graph-compatibility-v1",
            "physical_inputs_unchanged": True, "metadata_migrations": migrations}


__all__ = ("compare_wave_input_graphs",)
