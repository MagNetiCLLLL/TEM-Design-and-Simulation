"""Narrow input-only migration for additive round-coil reference metadata.

This comparison never qualifies a historical calculated result. Existing
geometry, controls and reference values must match exactly after TOML parsing.
"""
from copy import deepcopy
import tomllib

from temsim.coil_reference_metadata import (
    REFERENCE_FIELDS, ROUND_LENS_PROFILES, validate_coil_references,
)


def _same_value(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_value(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    return left == right


def additive_coil_parts_metadata_only(old_parts, new_parts):
    """Compare resolved parts with only validated optional metadata additions."""
    if len(old_parts) != len(new_parts):
        return False
    by_key = {part.get("key"): part for part in new_parts if isinstance(part, dict)}
    stripped = deepcopy(new_parts)
    for before, after in zip(old_parts, stripped):
        if not isinstance(before, dict) or not isinstance(after, dict):
            return False
        if before.get("key") != after.get("key"):
            return False
        profile = before.get("mechanical_profile")
        allowed = set()
        if profile in ROUND_LENS_PROFILES:
            allowed.update(REFERENCE_FIELDS)
            allowed.add("gaussian_reference_coil_length_mm")
        elif profile == "magnetic_excitation_coil":
            allowed.add("coil_ampere_turn_fraction")
        additions = after.keys() - before.keys()
        if not additions <= allowed:
            return False
        if "coil_ampere_turn_fraction" in additions:
            owner = by_key.get(after.get("field_source_key"), {})
            if (owner.get("mechanical_profile") not in ROUND_LENS_PROFILES
                    or not REFERENCE_FIELDS.issubset(owner)):
                return False
        for name in additions:
            del after[name]
    if not _same_value(stripped, old_parts):
        return False
    try:
        validate_coil_references(new_parts)
    except (ValueError, TypeError, KeyError, OverflowError):
        return False
    return True


def additive_coil_metadata_only(archived: bytes, current: bytes) -> bool:
    """Admit valid new optional fields on existing round owners and coils only.

    No existing key may change or disappear. New parts, geometry, ports, source
    values and unknown keys are rejected. Archived bytes remain untouched.
    """
    try:
        old = tomllib.loads(archived.decode("utf-8-sig"))
        new = tomllib.loads(current.decode("utf-8-sig"))
        old_parts, new_parts = old.get("parts"), new.get("parts")
        if not isinstance(old_parts, list) or not isinstance(new_parts, list):
            return False
        if not additive_coil_parts_metadata_only(old_parts, new_parts):
            return False
        stripped = deepcopy(new)
        stripped["parts"] = old_parts
        if not _same_value(stripped, old):
            return False
    except (ValueError, TypeError, KeyError, UnicodeError):
        return False
    return True
