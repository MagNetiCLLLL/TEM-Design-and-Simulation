"""Local CIF phase relative to an entrance-anchored specimen envelope.

The optional reference stores the thickness at which the existing lattice
phase was adopted. Its origin lies reference_thickness/2 below the entrance.
No laboratory-Z subtraction and no source-CIF rewrite is involved.
"""
import math
from numbers import Real


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def lattice_phase_record(thickness_nm, reference_thickness_nm=None):
    thickness = _nonnegative(thickness_nm, "Specimen thickness")
    reference = thickness if reference_thickness_nm is None else _nonnegative(
        reference_thickness_nm, "Lattice reference thickness")
    return {"reference_thickness_nm": reference,
            "origin_depth_below_upper_nm": reference*.5,
            "translation_from_current_center_nm": (reference-thickness)*.5,
            "translation_from_current_center_angstrom": (reference-thickness)*5.,
            "scope": "Original CIF phase fixed relative to the specimen entrance; original CIF unchanged"}


def lattice_translation_z_angstrom(sample):
    """Read a Sample/SpecimenScene without initializing or changing metadata."""
    return lattice_phase_record(sample.thickness_nm,
        getattr(sample, "lattice_reference_thickness_nm", None))["translation_from_current_center_angstrom"]


def lattice_translation_z_nm(sample):
    return lattice_phase_record(sample.thickness_nm,
        getattr(sample, "lattice_reference_thickness_nm", None))["translation_from_current_center_nm"]


def lattice_phase_for_sample(sample):
    return lattice_phase_record(sample.thickness_nm,
                                getattr(sample, "lattice_reference_thickness_nm", None))


__all__ = ("lattice_phase_record", "lattice_phase_for_sample", "lattice_translation_z_angstrom",
           "lattice_translation_z_nm")
