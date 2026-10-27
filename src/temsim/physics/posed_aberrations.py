"""Configured thin spherical-aberration kicks in a displaced lens frame.

This retains the existing Cs/f**4 equivalent thin-lens model; it is not a
high-order field solution. A tilted thin plane is represented at its global
centre Z by straight-line entrance/exit coordinate changes.
"""
from dataclasses import dataclass
import math

import numpy as np

from temsim.physics.lens_field_provider import CoordinateRegistration


@dataclass(frozen=True, slots=True)
class FrozenLensAberrationKick:
    lens_key: str
    local_z_m: float
    strength_m3: float
    registration: CoordinateRegistration

    def __post_init__(self):
        if not math.isfinite(self.local_z_m) or not math.isfinite(self.strength_m3):
            raise ValueError("Lens aberration plane and strength must be finite")

    @property
    def z_mm(self):
        return float((self.registration.rotation_array @ (0., 0., self.local_z_m)
                      + self.registration.origin_array_m)[2] * 1e3)

    def apply(self, x, tx, y, ty):
        if self.strength_m3 == 0:
            return x, tx, y, ty
        from temsim.physics.wave_device import array_module
        xp = next((array_module(value) for value in (x, tx, y, ty)
                   if hasattr(value, "__cuda_array_interface__")), np)
        x, tx, y, ty = (xp.asarray(value, dtype=xp.float64) for value in (x, tx, y, ty))
        points = xp.column_stack((x, y, xp.full_like(x, self.z_mm*1e-3)))
        # Matrix algebra deliberately preserves stopped-ray NaNs.
        rotation = xp.asarray(self.registration.rotation_array)
        origin = xp.asarray(self.registration.origin_array_m)
        local = (points-origin) @ rotation
        direction = xp.column_stack((tx, ty, xp.ones_like(tx))) @ rotation
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            direction = direction/direction[:, 2, None]
            plane = local + (self.local_z_m-local[:, 2, None]) * direction
            radial = self.strength_m3*xp.sum(plane[:, :2]**2, axis=1)
            direction[:, :2] -= radial[:, None]*plane[:, :2]
            outgoing = direction @ rotation.T
            outgoing /= outgoing[:, 2, None]
            global_plane = plane @ rotation.T + origin
            global_plane += (self.z_mm*1e-3-global_plane[:, 2, None])*outgoing
        return global_plane[:, 0], outgoing[:, 0], global_plane[:, 1], outgoing[:, 1]

    def apply_canonical(self, x, tx, y, ty, **kwargs):
        """Small-inclination action shared with coherent Cs propagation."""
        from temsim.physics.posed_aberration_wave import apply_canonical_ray_kick
        return apply_canonical_ray_kick(self, x, tx, y, ty, **kwargs)


def posed_spherical_kicks(state, z0, z1):
    from temsim.lens_pose import has_lens_pose, lens_pose_registration
    from temsim.optics.lens_focal_length import focal_length_mm
    from temsim.optics.magnetic_lens_aberration import spherical_aberration_mm
    from temsim.physics.lens_field_provider import active_mapped_providers
    mapped = {p.lens_key for p in active_mapped_providers(state)}
    result = []
    for lens in state.lenses:
        if not lens.enabled or lens.key in mapped or not has_lens_pose(state, lens.key):
            continue
        cs = spherical_aberration_mm(lens, state.beam_voltage_kv)
        if not cs:
            continue
        focal = float(focal_length_mm(lens, state.beam_voltage_kv))
        if not math.isfinite(focal) or focal <= 0:
            continue
        event = FrozenLensAberrationKick(lens.key, float(lens.z_mm)*1e-3,
            float(cs)*1e-3/(focal*1e-3)**4, lens_pose_registration(state, lens.key))
        if z0-1e-9 <= event.z_mm <= z1+1e-9:
            result.append(event)
    return tuple(result)


def kick_indices(kicks, z_mm, *, include_initial=True):
    result = {}
    for kick in kicks:
        distance = np.abs(z_mm-kick.z_mm)
        coincident = np.flatnonzero(distance <= 8*np.spacing(max(abs(kick.z_mm), 1.)))
        # A transformed event can differ by an ulp from its original plane.
        # Prefer the first coincident node so restart ownership is stable.
        index = int(coincident[0]) if coincident.size else int(np.argmin(distance))
        if (np.isclose(z_mm[index], kick.z_mm, rtol=0., atol=1e-9)
                and (include_initial or index != 0)):
            result.setdefault(index, []).append(kick)
    return result
