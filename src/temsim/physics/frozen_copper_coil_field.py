"""Immutable copper-coil consumer adapter; display envelopes are not cutoffs.

The exact annular-loop field and full axial tails are delegated unchanged to
the captured potential.  The finite winding envelope only schedules work and
seeds a bounded display.  Display cylinders follow the physical winding bores;
they are not a Gaussian or near-axis approximation of the field.
"""
from dataclasses import dataclass
import inspect
import math
from itertools import product

import numpy as np
from scipy.constants import elementary_charge, mu_0

from temsim.physics.annular_coil_potential import AnnularCoilPotential
from temsim.physics.copper_coil_field_provider import CopperCoilAssemblyPotential
from temsim.physics.lens_field_provider import CoordinateRegistration, FieldMapError, _posed_axial_support


_METHODS = {
    cls: tuple((name, inspect.getattr_static(cls, name)) for name in names)
    for cls, names in (
        (CopperCoilAssemblyPotential, ("field_at_global_positions_t", "sample", "sample_jet", "to_metadata")),
        (AnnularCoilPotential, ("field_at_global_positions_t", "_local", "contains_winding_local",
                               "_sample_local", "_near_axis", "_axis_coefficients", "axial_derivatives",
                               "axial_derivatives_numpy", "to_metadata")),
        (CoordinateRegistration, ("positions_global_to_local_m", "vectors_local_to_global")),
    )
}


def _methods_match(value):
    methods = _METHODS.get(type(value))
    return methods is not None and all(inspect.getattr_static(value, name, None) is original
                                       for name, original in methods)


@dataclass(frozen=True, slots=True)
class FrozenCopperCoilField:
    potential: CopperCoilAssemblyPotential

    def __post_init__(self):
        if type(self.potential) is not CopperCoilAssemblyPotential:
            raise FieldMapError("Frozen ideal-current coil field requires a captured coil potential")

    @classmethod
    def from_provider(cls, provider):
        return cls(provider.potential)

    @property
    def lens_key(self):
        return self.potential.source_id

    @property
    def axial_domain_all_z(self):
        return True

    @property
    def radial_support_m(self):
        return min(coil.inner_radius_m for coil in self.potential.coils)

    @property
    def scale(self):
        return float(any(coil.ampere_turns != 0. for coil in self.potential.coils))

    @property
    def fingerprint(self):
        return self.potential.digest

    @property
    def field_support_mm(self):
        """Finite scheduling/seed envelope; the field does not vanish outside."""
        bounds = [_posed_axial_support(((coil.center_z_m-coil.length_m/2)*1e3,
            (coil.center_z_m+coil.length_m/2)*1e3), coil.registration, coil.outer_radius_m)
            for coil in self.potential.coils]
        return min(row[0] for row in bounds), max(row[1] for row in bounds)

    @property
    def display_bounds_m(self):
        corners = []
        for coil in self.potential.coils:
            r = coil.inner_radius_m
            points = np.array(tuple(product((-r, r), (-r, r),
                (coil.center_z_m-coil.length_m/2, coil.center_z_m+coil.length_m/2))))
            corners.append(points@coil.registration.rotation_array.T+coil.registration.origin_array_m)
        points = np.concatenate(corners)
        return np.array((points.min(axis=0), points.max(axis=0)))

    def display_contains(self, positions_m):
        points = np.asarray(positions_m, dtype=float)
        admitted = np.ones(points.shape[:-1], dtype=bool)
        for coil in self.potential.coils:
            local = coil.registration.positions_global_to_local_m(points)
            radius = np.hypot(local[..., 0], local[..., 1])
            admitted &= radius <= coil.inner_radius_m
            admitted &= ~coil.contains_winding_local(local)
        return admitted

    def maximum_step_mm(self, requested_step_mm, momentum_kg_m_s):
        if not (math.isfinite(requested_step_mm) and requested_step_mm > 0
                and math.isfinite(momentum_kg_m_s) and momentum_kg_m_s > 0):
            raise FieldMapError("Ideal-current coil transport step and momentum must be finite and positive")
        # This conservative estimate resolves source dimensions and a nominal
        # near-axis angular advance. It does not certify quadrature convergence.
        scale_m = min(min(coil.inner_radius_m, coil.length_m) for coil in self.potential.coils)
        peak_t = math.fsum(mu_0*abs(coil.ampere_turns)/(2*coil.inner_radius_m)
                          for coil in self.potential.coils)
        bend_mm = .02*momentum_kg_m_s/(elementary_charge*max(peak_t, 1e-30))*1e3
        return min(float(requested_step_mm), .2*scale_m*1e3, bend_mm)

    def field_at_global_positions_t(self, positions_m):
        return self.potential.field_at_global_positions_t(positions_m)

    def implementation_is_supported(self):
        return (_methods_match(self.potential)
                and all(_methods_match(coil) and _methods_match(coil.registration)
                        for coil in self.potential.coils))


def is_all_z_copper_field(provider):
    """Recognise the admitted exact adapter, not arbitrary callback metadata."""
    from temsim.physics.transverse_coil_field import FrozenTransverseCoilField
    return type(provider) in (FrozenCopperCoilField,FrozenTransverseCoilField)
