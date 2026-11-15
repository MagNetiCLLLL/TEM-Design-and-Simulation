"""Explicit provenance and current normalization for air-core round lenses.

These checks do not solve a magnetic field or qualify an operating preset.
Old documents without this optional metadata retain their existing behavior;
the air-core executor separately requires an explicit reference for each owner.
"""
from __future__ import annotations

import math
from urllib.parse import urlsplit

from temsim.magnetic_geometry import objective_layer_intervals_mm
from temsim.coil_winding import OWNER_CURRENT_FIELD


ROUND_LENS_PROFILES = frozenset({"magnetic_lens_assembly", "integrated_magnetic_lens_channel"})
REFERENCE_FIELDS = frozenset({
    "air_core_reference_ampere_turns", "air_core_reference_status",
    "air_core_reference_source", "air_core_reference_source_urls",
})
REFERENCE_STATUSES = frozenset({
    "engineering_reference_not_oem_not_particle_qualified",
    "manufacturer_documented", "measured_calibration",
})

# Optional explicit winding/current inputs. Legacy NI records remain readable
# without inventing a turn count, conductor size or electrical current.
def round_coil_operating_point(owner, coils, *, percent=100., polarity=1, enabled=True):
    """Resolve reference/actual I and NI without evaluating a magnetic field.

    Rows may be plain manifest parts or captured ``{key, data}`` records.
    Explicit series windings share the owner's prescribed current. A coil's
    continuous NI is distributed over its unchanged envelope by the field
    adapter, independently of CAD wire spacing.
    """
    def data(row):
        return row.get("data", row)
    definition = data(owner)
    label = str(owner.get("key", definition.get("key", "round lens")))
    if (isinstance(percent, bool) or not isinstance(percent, (int, float))
            or not math.isfinite(percent) or not 0 <= percent <= 100
            or isinstance(polarity, bool) or polarity not in (-1, 1)):
        raise ValueError(f"{label}: prescribed-current excitation requires 0..100 percent and polarity +/-1")
    scale = float(percent)*.01*polarity if enabled else 0.
    if OWNER_CURRENT_FIELD not in definition:
        reference = _positive(definition.get("air_core_reference_ampere_turns"),
                              f"{label}.air_core_reference_ampere_turns")
        actual = reference*float(percent)*.01*polarity if enabled else 0.
        return dict(method="legacy_reference_ampere_turns", reference_ampere_turns=reference,
                    ampere_turns=actual, reference_current_a=None, current_a=None,
                    total_turn_count=None, winding_reference_ampere_turns={})
    current = definition[OWNER_CURRENT_FIELD]
    if (isinstance(current, bool) or not isinstance(current, (int, float))
            or not math.isfinite(current) or current < 0):
        raise ValueError(f"{label}.{OWNER_CURRENT_FIELD} must be finite and nonnegative")
    if not coils:
        raise ValueError(f"{label}: prescribed current requires at least one owned round winding")
    from temsim.coil_winding import validate_winding_definition
    counts = {}
    for coil in coils:
        row = data(coil)
        validate_winding_definition(row)
        key = str(coil.get("key", row.get("key", "winding")))
        if "winding_turn_count" not in row:
            raise ValueError(f"{label}: define turns for every owned winding before prescribing current")
        if key in counts:
            raise ValueError(f"{label}: owned winding identities must be unique")
        counts[key] = int(row["winding_turn_count"])
    total = sum(counts.values())
    try:
        reference = float(current)*total
    except OverflowError as error:
        raise ValueError(f"{label}: winding ampere-turns exceed the finite numerical range") from error
    if not math.isfinite(reference):
        raise ValueError(f"{label}: winding ampere-turns exceed the finite numerical range")
    return dict(method="explicit_winding_turns_and_prescribed_current_v1",
                reference_ampere_turns=reference, ampere_turns=reference*scale,
                reference_current_a=float(current), current_a=float(current)*scale,
                total_turn_count=total,
                winding_reference_ampere_turns={key: float(current)*count for key, count in counts.items()})


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
        if OWNER_CURRENT_FIELD not in owner:
            continue
        if owner.get("mechanical_profile") not in ROUND_LENS_PROFILES:
            raise ValueError(f"{owner['key']}: prescribed round-coil current requires a round-lens owner")
        coils = [part for part in parts if part.get("mechanical_profile") == "magnetic_excitation_coil"
                 and part.get("field_source_key") == owner["key"]]
        round_coil_operating_point(owner, coils)
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
            if OWNER_CURRENT_FIELD not in owner:
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
        if OWNER_CURRENT_FIELD not in owner and not math.isclose(math.fsum(shares), 1., rel_tol=0., abs_tol=1e-12):
            raise ValueError(f"{key}: coil ampere-turn fractions must sum to one")
