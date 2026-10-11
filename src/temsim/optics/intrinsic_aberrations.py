"""Additional intrinsic coefficients owned by individual round lenses.

Lengths are mm and azimuths are degrees in each lens's optical X-Y frame.
Cs and Cc retain their existing lens fields; this map never overrides them.
The thin equivalent screen uses S(q)=-W(q/f), not a second focusing lens.
These are user-declared imperfections, not fitted system residuals.
"""
from collections.abc import Mapping
import math

from temsim.optics.aberration_basis import WAVE_TERMS


INTRINSIC_TERMS = tuple(term for term in WAVE_TERMS if term.name != "C3")
INTRINSIC_FIELDS = tuple(name for term in INTRINSIC_TERMS
                         for name in (term.field, term.azimuth_field) if name)
MODEL_VERSION = "lens-local-intrinsic-polynomial-v1"


def validated_coefficients(row=None):
    """Return a complete independent row, rejecting typos and nonfinite data."""
    row = {} if row is None else row
    if not isinstance(row, Mapping):
        raise ValueError("Lens intrinsic aberrations must be a coefficient mapping")
    unknown = set(row) - set(INTRINSIC_FIELDS)
    if unknown:
        raise ValueError("Unknown intrinsic coefficients: " + ", ".join(sorted(map(str, unknown))))
    result = {}
    for name in INTRINSIC_FIELDS:
        value = row.get(name, 0.)
        if isinstance(value, (str, bool)):
            raise ValueError(f"{name} must be a finite number")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        result[name] = value
    return result


def validated_lens_aberrations(value):
    if not isinstance(value, Mapping):
        raise ValueError("Lens intrinsic aberrations must be keyed by optical lens identity")
    result = {}
    for key, row in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Intrinsic aberrations require a nonempty optical lens key")
        result[key] = validated_coefficients(row)
    return result


def lens_intrinsic_coefficients(state, key):
    return validated_coefficients(getattr(state, "lens_intrinsic_aberrations", {}).get(str(key), {}))


def has_intrinsic_coefficients(row):
    values = validated_coefficients(row)
    return any(values[term.field] != 0. for term in INTRINSIC_TERMS)
