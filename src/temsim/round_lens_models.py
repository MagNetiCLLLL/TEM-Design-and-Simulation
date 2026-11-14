"""Explicit round-lens source choices and optional Gaussian length scaling.

This module only reads resolved metadata. It does not solve or sample a field,
change a lens excitation, or mutate a saved Gaussian calibration.
"""
from __future__ import annotations

import math
from numbers import Real


ROUND_LENS_FIELD_MODELS = {
    "gaussian": "Gaussian calibration / saved lens models",
    "copper_coil": "Copper coils in vacuum (ignore pole pieces)",
}
GAUSSIAN_GEOMETRY_COUPLINGS = {
    "fixed_calibration": "Fixed Gaussian calibration",
    "axial_length": "Link Gaussian widths to coil length",
}


def round_lens_model_description(state):
    model = validate_round_lens_field_model(getattr(state, "round_lens_field_model", "gaussian"))
    coupling = validate_gaussian_geometry_coupling(getattr(state, "gaussian_geometry_coupling", "fixed_calibration"))
    if model == "copper_coil":
        return ("Round lenses use the declared copper coil dimensions and ampere-turns in vacuum. "
                "Pole pieces, iron and saturation are ignored. Native corrector multipoles, stigmators and deflectors are retained. "
                "This source currently supports classical particles; coherent propagation is unavailable.")
    if coupling == "axial_length":
        return ("Gaussian profile widths and internal offsets follow actual/reference current-bearing coil length. "
                "Optical centres, Gaussian amplitudes and excitation stay fixed; the combined Objective peak may change through overlap. "
                "This is an explicit engineering length rule, not a coil or material field solution. Saved mapped lens recipes remain independent.")
    return ("Gaussian calibration and saved per-lens models retain their existing field law. "
            "Mechanical edits do not automatically reconstruct Gaussian fields. Native non-round field sources are retained.")


def validate_round_lens_field_model(value):
    if not isinstance(value, str) or value not in ROUND_LENS_FIELD_MODELS:
        raise ValueError("Round-lens field model must be gaussian or copper_coil")
    return value


def validate_gaussian_geometry_coupling(value):
    if not isinstance(value, str) or value not in GAUSSIAN_GEOMETRY_COUPLINGS:
        raise ValueError("Gaussian geometry coupling must be fixed_calibration or axial_length")
    return value


def validate_round_lens_selection(model, coupling, simulation_mode):
    model = validate_round_lens_field_model(model)
    coupling = validate_gaussian_geometry_coupling(coupling)
    if simulation_mode != "unified" and (model == "copper_coil" or coupling == "axial_length"):
        raise ValueError("Copper coils and Gaussian length linking require shared Finite-Field Optics. Select Gaussian with fixed calibration before changing the execution model.")
    return model, coupling


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive length")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive length")
    return result


def _coil_intervals(part, by_key):
    from temsim.magnetic_geometry import objective_layer_intervals_mm
    parent = by_key.get(part.parent_key)
    if parent is not None:
        intervals = objective_layer_intervals_mm(parent.data, parent.start_z_mm,
                                                 "magnetic_excitation_coil")
        if intervals:
            return intervals
    if "material_intervals_mm" in part.data:
        offset = float(part.start_z_mm)-float(part.data["local_start_z_mm"])
        return tuple((float(start)+offset, float(end)+offset)
                     for start, end in part.data["material_intervals_mm"])
    return ((float(part.start_z_mm), float(part.end_z_mm)),)


def gaussian_geometry_metadata(state, lens_key):
    """Return a pure length-rule receipt; fixed calibration reads no geometry.

    The rule scales each native optical profile's Gaussian offsets and widths
    about its unchanged optical centre. Amplitudes and excitations are held.
    Objective upper/lower profile centres are fixed, so their combined peak
    can change through overlap. Neither peak-squared integral nor optical
    focus is renormalized.
    """
    coupling = validate_gaussian_geometry_coupling(getattr(state, "gaussian_geometry_coupling", "fixed_calibration"))
    result = {"schema": "gaussian-coil-length-rule-v1", "mode": coupling,
        "lens_key": str(lens_key), "width_scale": 1.,
        "rule": "Scale native a (internal Gaussian offsets and widths); retain optical centres, term amplitudes, excitation and polarity",
        "qualification": "Explicit engineering length rule, not a coil/pole/material field solution or a focus correction"}
    if coupling == "fixed_calibration":
        return result
    from temsim.magnetic_circuits import is_custom_mechanical_part, optical_owner
    by_key = {part.key: part for part in getattr(getattr(state, "_resolved_assembly", None), "parts", ())}
    owner = by_key.get(str(lens_key))
    if owner is None:
        raise ValueError(f"{lens_key}: Gaussian length linking requires installed resolved lens geometry")
    reference = _positive(owner.data.get("gaussian_reference_coil_length_mm"),
                          f"{lens_key}.gaussian_reference_coil_length_mm")
    coils = [part for part in by_key.values()
             if part.data.get("mechanical_profile") == "magnetic_excitation_coil"
             and not is_custom_mechanical_part(part)
             and part.data.get("field_source_key", optical_owner(part, by_key)) == str(lens_key)]
    if not coils:
        raise ValueError(f"{lens_key}: Gaussian length linking requires an explicitly owned coil")
    records, lengths = [], []
    for part in coils:
        intervals = _coil_intervals(part, by_key)
        previous = -math.inf
        for start, end in intervals:
            if not math.isfinite(start) or not math.isfinite(end) or end <= start or start < previous:
                raise ValueError(f"{part.key}: current-bearing intervals must be ordered, disjoint and finite")
            lengths.append(end-start)
            previous = end
        records.append({"key": part.key, "current_bearing_intervals_mm": [list(row) for row in intervals]})
    actual = _positive(math.fsum(lengths), f"{lens_key} current-bearing coil length")
    scale = actual/reference
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{lens_key}: Gaussian coil-length ratio must be finite and positive")
    result.update(actual_coil_length_mm=actual, reference_coil_length_mm=reference,
                  width_scale=scale, coils=records)
    return result


def gaussian_geometry_scale(state, lens_key):
    return gaussian_geometry_metadata(state, lens_key)["width_scale"]


def _clear_live_providers(state):
    state._runtime_lens_field_provider_cache = {}


def _preflight_native_gaussian_lengths(state):
    """Check declared native profiles only; mapped recipes own their geometry.

    This selection check never loads a map or solves a field. The actual
    resolver still validates its chosen source before evaluating a field.
    """
    from types import SimpleNamespace
    candidate = SimpleNamespace(_resolved_assembly=getattr(state, "_resolved_assembly", None),
                                gaussian_geometry_coupling="axial_length")
    descriptors = getattr(state, "lens_field_map_descriptors", {})
    bindings = getattr(state, "_lens_field_map_bindings", {})
    for lens in state.lenses:
        if not lens.enabled:
            continue
        recipe = descriptors.get(lens.key, {})
        if (recipe and recipe.get("solver") not in {"analytic", "analytical"}) or lens.key in bindings:
            continue
        gaussian_geometry_metadata(candidate, lens.key)


def set_round_lens_field_model(state, value):
    from temsim.simulation_modes import mode_key
    value, _ = validate_round_lens_selection(value, getattr(state, "gaussian_geometry_coupling", "fixed_calibration"), mode_key(state))
    if value == getattr(state, "round_lens_field_model", "gaussian"):
        return False
    if value == "gaussian" and getattr(state, "gaussian_geometry_coupling", "fixed_calibration") == "axial_length":
        _preflight_native_gaussian_lengths(state)
    state.round_lens_field_model = value
    _clear_live_providers(state)
    return True


def set_gaussian_geometry_coupling(state, value):
    from temsim.simulation_modes import mode_key
    _, value = validate_round_lens_selection(getattr(state, "round_lens_field_model", "gaussian"), value, mode_key(state))
    if value == getattr(state, "gaussian_geometry_coupling", "fixed_calibration"):
        return False
    if value == "axial_length" and getattr(state, "round_lens_field_model", "gaussian") == "gaussian":
        _preflight_native_gaussian_lengths(state)
    state.gaussian_geometry_coupling = value
    _clear_live_providers(state)
    return True
