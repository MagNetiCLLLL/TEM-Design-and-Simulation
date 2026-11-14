"""Explicit provenance and current normalization for air-core round lenses.

These checks do not solve a magnetic field or qualify an operating preset.
Old documents without this optional metadata retain their existing behavior;
the air-core executor separately requires an explicit reference for each owner.
"""
from __future__ import annotations

import math
from urllib.parse import urlsplit

from temsim.magnetic_geometry import objective_layer_intervals_mm


ROUND_LENS_PROFILES = frozenset({"magnetic_lens_assembly", "integrated_magnetic_lens_channel"})
REFERENCE_FIELDS = frozenset({
    "air_core_reference_ampere_turns", "air_core_reference_status",
    "air_core_reference_source", "air_core_reference_source_urls",
})
REFERENCE_STATUSES = frozenset({
    "engineering_reference_not_oem_not_particle_qualified",
    "manufacturer_documented", "measured_calibration",
})


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def current_intervals_mm(coil, parent):
    """Return authoritative winding intervals in the document's local frame.

    Objective layers use the same split material intervals as CAD and the
    geometry field binding, rather than the historical child display envelope.
    """
    intervals = objective_layer_intervals_mm(parent, parent["local_start_z_mm"], "magnetic_excitation_coil")
    if intervals:
        return intervals
    return tuple(coil.get("material_intervals_mm", ((coil["local_start_z_mm"], coil["local_end_z_mm"]),)))


def validate_coil_references(parts):
    """Validate complete optional round-owner metadata and explicit coil shares."""
    for owner in parts:
        key = str(owner["key"])
        if "gaussian_reference_coil_length_mm" in owner:
            _positive(owner["gaussian_reference_coil_length_mm"], f"{key}.gaussian_reference_coil_length_mm")
        present = REFERENCE_FIELDS.intersection(owner)
        if not present:
            continue
        if owner.get("mechanical_profile") not in ROUND_LENS_PROFILES:
            raise ValueError(f"{key}: air-core round-lens references cannot be assigned to a multipole or deflector")
        missing = REFERENCE_FIELDS.difference(owner)
        if missing:
            raise ValueError(f"{key}: incomplete air-core reference metadata: {', '.join(sorted(missing))}")
        _positive(owner["air_core_reference_ampere_turns"], f"{key}.air_core_reference_ampere_turns")
        if not isinstance(owner["air_core_reference_status"], str) or owner["air_core_reference_status"] not in REFERENCE_STATUSES:
            raise ValueError(f"{key}: invalid air-core reference status")
        if not isinstance(owner["air_core_reference_source"], str) or not owner["air_core_reference_source"].strip():
            raise ValueError(f"{key}: air-core reference source must be nonempty")
        urls = owner["air_core_reference_source_urls"]
        if not isinstance(urls, list) or not urls or any(not isinstance(url, str) or
                urlsplit(url).scheme not in {"http", "https"} or not urlsplit(url).netloc for url in urls):
            raise ValueError(f"{key}: air-core reference source URLs must be nonempty HTTP(S) links")
        coils = [part for part in parts if part.get("mechanical_profile") == "magnetic_excitation_coil"
                 and part.get("field_source_key") == key]
        if not coils:
            raise ValueError(f"{key}: explicit current-bearing round-coil geometry is required")
        shares = []
        for coil in coils:
            name = str(coil["key"])
            inner = _positive(coil["mechanical_inner_diameter_mm"], f"{name}.mechanical_inner_diameter_mm")
            outer = _positive(coil["mechanical_outer_diameter_mm"], f"{name}.mechanical_outer_diameter_mm")
            if outer <= inner:
                raise ValueError(f"{name}: winding outer diameter must exceed inner diameter")
            shares.append(_positive(coil.get("coil_ampere_turn_fraction"), f"{name}.coil_ampere_turn_fraction"))
            intervals = current_intervals_mm(coil, owner)
            if not intervals:
                raise ValueError(f"{name}: at least one current-bearing interval is required")
            if any(not isinstance(interval, (tuple, list)) or len(interval) != 2 or
                   any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in interval)
                   for interval in intervals):
                raise ValueError(f"{name}: current-bearing intervals require two numeric endpoints")
            previous_end = -math.inf
            for start, end in sorted(intervals):
                if not (math.isfinite(start) and math.isfinite(end) and start < end and start >= previous_end):
                    raise ValueError(f"{name}: current-bearing intervals must be finite, positive and disjoint")
                previous_end = end
        if not math.isclose(math.fsum(shares), 1., rel_tol=0., abs_tol=1e-12):
            raise ValueError(f"{key}: coil ampere-turn fractions must sum to one")
