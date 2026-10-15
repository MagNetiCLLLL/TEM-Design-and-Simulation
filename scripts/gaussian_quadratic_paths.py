"""Accurate small-matrix quadratic stages for experimental Gaussian replay.

The outer signed Strang stages keep their nonlinear screens. This helper
integrates their linear field continuously using nested fourth-order signed
midpoint quadrature, without propagating any packet during that integration.
Only smooth aligned analytic fields and zero finite dipole drives are admitted.
Discrete Cs, apertures and deflector impulses stay with the outer physical grid.
"""
from dataclasses import replace
import math
import numpy as np

from scripts.experimental_gaussian_beamlets import append_physical_column_step
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.column_wave import _column_transports, _electric_coefficients
from temsim.physics.core import fields, skew_quadrupole_field


def integrated_quadratic_stage_paths(state, plan, energy_kev, *, maximum_step_mm=.0625,
                                    maximum_working_bytes=256*1024**2, retained_bytes=0,
                                    cancelled=lambda: False):
    """Return one accurately lifted linear path per outer signed stage.

    The coordinate path can backtrack outside a physical macro interval.
    Providers must therefore describe smooth linear fields over that extended
    range; impulses and absorbing boundaries must not be inserted here.
    """
    if not math.isfinite(maximum_step_mm) or maximum_step_mm <= 0:
        raise ValueError("Quadratic integration step must be finite and positive")
    if any(getattr(plan, name, ()) for name in (
            'mapped_fields', 'posed_spherical_kicks', 'intrinsic_aberration_kicks')):
        raise ValueError("Integrated quadratic stages require aligned analytic fields")
    if np.any(plan.dipole_bx_t) or np.any(plan.dipole_by_t):
        raise ValueError("Integrated quadratic stages require zero finite dipole drives")
    sizes = np.maximum(1., np.ceil(abs(plan.step_m)*1e3/maximum_step_mm))
    if not np.isfinite(sizes).all() or float(sizes.sum()) > maximum_working_bytes/(3*512):
        raise MemoryError("Quadratic micro-stage count exceeds the diagnostic budget")
    count = sizes.astype(np.int64)
    micro_count = 3*int(count.sum())
    # Vectorized physical provider arrays, electric nodes/gradients/Hessians,
    # and construction temporaries; final paths only retain tiny matrices.
    required = retained_bytes+512*micro_count+1024*len(count)
    if required > maximum_working_bytes:
        raise MemoryError("Quadratic micro-stage arrays exceed the explicit diagnostic budget")
    if cancelled():
        raise InterruptedError("Quadratic stage preparation cancelled")
    w = 1./(2.-2.**(1./3.))
    weights = np.array([w, 1.-2.*w, w])
    cumulative = np.r_[0., np.cumsum(weights)]
    fractions = (cumulative[:-1]+cumulative[1:])/2.
    midpoints = np.empty(micro_count)
    endpoints = np.empty(micro_count)
    distances = np.empty(micro_count)
    index = 0
    for stage, pieces in enumerate(count):
        lower, upper = plan.z_mm[stage:stage+2]
        edges = np.linspace(lower, upper, int(pieces)+1)
        width = np.diff(edges)
        size = 3*int(pieces)
        midpoints[index:index+size] = (edges[:-1,None]+width[:,None]*fractions).ravel()
        values = edges[:-1,None]+width[:,None]*cumulative[1:]
        values[:,-1] = edges[1:]
        endpoints[index:index+size] = values.ravel()
        distances[index:index+size] = (width[:,None]*weights*1e-3).ravel()
        index += size
    magnetic, sx, sy = fields(midpoints, state)
    sxy = skew_quadrupole_field(midpoints, state)
    z = np.r_[plan.z_mm[0], endpoints]
    micro = replace(plan, z_mm=z, step_m=distances,
        midpoint_magnetic_t=magnetic, midpoint_sx_m2=sx, midpoint_sy_m2=sy,
        midpoint_sxy_m2=sxy,
        midpoint_hex_normal_m3=np.zeros(micro_count), midpoint_hex_skew_m3=np.zeros(micro_count),
        cs_kick_m3=np.zeros(micro_count+1), kick_x_rad=np.zeros(micro_count+1), kick_y_rad=np.zeros(micro_count+1),
        dipole_bx_t=np.zeros(3*micro_count), dipole_by_t=np.zeros(3*micro_count))
    electric = _electric_coefficients(micro)
    # Its first node is the original column entrance, so every substage uses
    # the same physical energy and canonical momentum reference.
    paths, current = [], CanonicalPath(1e-3)
    boundaries = np.cumsum(3*count)
    stage = 0
    for index, path in enumerate(_column_transports(micro, energy_kev, electric=electric)):
        if index % 256 == 0 and cancelled():
            raise InterruptedError("Quadratic stage integration cancelled")
        append_physical_column_step(current, path)
        if index+1 == boundaries[stage]:
            paths.append(current)
            current = CanonicalPath(1e-3)
            stage += 1
    return tuple(paths), {
        'schema': 'experimental-integrated-quadratic-stages-v1',
        'method': 'nested fourth-order signed midpoint quadrature; exact lifted-map composition',
        'maximum_quadratic_step_mm': maximum_step_mm,
        'micro_stage_count': micro_count,
        'outer_stage_count': len(count),
        'estimated_working_bytes': required,
        'scope': 'Quadratic field integration only; nonlinear outer splitting convergence remains separate',
    }
