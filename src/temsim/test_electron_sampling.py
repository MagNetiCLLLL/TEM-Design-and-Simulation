"""Stable numerical cell selection, independent of physical field evaluation."""
from __future__ import annotations

from math import sqrt

import numpy as np
from temsim.physics.discrete_gradient import register_jitable
from temsim.physics.relativistic_lorentz import (
    ELECTRON_MASS_KG, ELEMENTARY_CHARGE_C, SPEED_OF_LIGHT_M_PER_S,
)


_REST_ENERGY_EV = ELECTRON_MASS_KG * SPEED_OF_LIGHT_M_PER_S**2 / ELEMENTARY_CHARGE_C


def diagnostic_linear_axial_field(scene):
    """Identify the unmodified piecewise-linear potential shared by both pushers."""
    from temsim.physics.closed_gun_field import ClosedGunField
    from temsim.test_electron_scene import TestElectronScene

    if type(scene) is not TestElectronScene or type(scene.electric_base) is not ClosedGunField:
        return None
    base = scene.electric_base
    if (getattr(scene.diagnostic_fields_at_global_position, "__func__", None)
            is not TestElectronScene.diagnostic_fields_at_global_position
            or getattr(base.interpolate, "__func__", None) is not ClosedGunField.interpolate
            or getattr(scene.electric_provider, "wien_field", None) is not None):
        return None
    return base


@register_jitable(inline="always")
def step_cell_index(axis, value):
    """Resolve a roundoff-sized face tie consistently for step scheduling.

    Only the cell-width budget uses this convention. Particle coordinates,
    interpolation, potential differences and hardware intersections are never
    snapped. The caller still limits motion with the neighbouring cell width.
    """
    index = min(max(np.searchsorted(axis, value, side="right")-1, 0), len(axis)-2)
    if index > 0 and abs(value-axis[index]) <= 8.*abs(np.spacing(axis[index])):
        return index-1
    return index


@register_jitable
def limit_time_at_axial_face(position, momentum, magnetic, potential, axis, voltage, dt):
    """End a straight axial trial within one linear-potential cell."""
    x, y, z = position
    ux, uy, uz = momentum
    if (x != 0. or y != 0. or ux != 0. or uy != 0. or uz == 0.
            or magnetic[0] != 0. or magnetic[1] != 0. or z < 0.):
        return dt

    # Select the next face past a roundoff-sized tie. The coordinates and
    # physical field interpolation remain untouched at a face.
    tie = 16. * abs(np.spacing(z))
    if uz > 0.:
        face = np.searchsorted(axis, z + tie, side="right")
    else:
        face = np.searchsorted(axis, z - tie, side="left") - 1
    if face < 0 or face >= len(axis):
        return dt
    distance = abs(axis[face] - z)
    if distance >= SPEED_OF_LIGHT_M_PER_S * dt:
        return dt

    # Crossing a slope change averages the force over distance instead of
    # time. Inside one cell the force is constant, so the two momenta give
    # the exact relativistic arrival time without subtracting nearby gammas.
    gamma = sqrt(1. + uz * uz)
    end_kinetic = uz * uz / (gamma + 1.) + (voltage[face] - potential) / _REST_ENERGY_EV
    if end_kinetic < 0.:
        return dt  # The particle turns before reaching the face.
    end_momentum = sqrt(end_kinetic * (2. + end_kinetic))
    arrival_time = distance / SPEED_OF_LIGHT_M_PER_S * (gamma + 1. + end_kinetic) / (abs(uz) + end_momentum)
    return min(dt, arrival_time)
