"""Fourth-order axial composition for the experimental Gaussian diagnostic.

This expands a physical macro grid into signed midpoint-frozen Strang stages.
Intermediate backtracks are numerical stages, never extra physical apertures,
Cs planes or measurements. The entrance momentum/potential reference is fixed.
"""
from dataclasses import replace
import numpy as np

from temsim.physics.core import fields, skew_quadrupole_field, hexapole_field_components
from temsim.physics.column_wave import _electric_coefficients


def fourth_order_column_stages(state, plan):
    if plan.mapped_fields or plan.posed_spherical_kicks or plan.intrinsic_aberration_kicks:
        raise ValueError('Experimental fourth-order stages require aligned analytic fields')
    w = 1. / (2. - 2.**(1./3.))
    weights = np.array([w, 1.-2.*w, w])
    cumulative = np.r_[0., np.cumsum(weights)]
    fractions = (cumulative[:-1]+cumulative[1:])/2.
    dz_mm = np.diff(plan.z_mm)
    midpoints = (plan.z_mm[:-1,None]+dz_mm[:,None]*fractions).ravel()
    endpoints = (plan.z_mm[:-1,None]+dz_mm[:,None]*cumulative[1:]).ravel()
    endpoints[2::3] = plan.z_mm[1:]
    z = np.r_[plan.z_mm[0], endpoints]
    # Fresh physical provider samples are essential: interpolation of the old
    # midpoint samples would preserve the old second-order field error.
    magnetic, sx, sy = fields(midpoints, state)
    sxy = skew_quadrupole_field(midpoints, state)
    hn, hs = hexapole_field_components(midpoints, state)
    def expand_events(values):
        result = np.zeros(3*len(dz_mm)+1)
        result[::3] = values
        return result
    staged = replace(plan, z_mm=z, step_m=(plan.step_m[:,None]*weights).ravel(),
        midpoint_magnetic_t=magnetic, midpoint_sx_m2=sx, midpoint_sy_m2=sy,
        midpoint_sxy_m2=sxy, midpoint_hex_normal_m3=hn, midpoint_hex_skew_m3=hs,
        cs_kick_m3=expand_events(plan.cs_kick_m3),
        kick_x_rad=expand_events(plan.kick_x_rad), kick_y_rad=expand_events(plan.kick_y_rad),
        # Finite dipoles are constant on each original interval, whose exact
        # endpoints already include every physical coil boundary.
        dipole_bx_t=np.repeat(np.asarray(plan.dipole_bx_t)[1::3],9),
        dipole_by_t=np.repeat(np.asarray(plan.dipole_by_t)[1::3],9))
    return staged, _electric_coefficients(staged)
