"""Nominal axisymmetric material checks for the shared projector iron stack."""

from itertools import combinations
import math

import numpy as np

from temsim.magnetic_circuits import COIL, MAGNETIC_BODIES, radial_profile_mm
from temsim.mechanical_dimension_bindings import mechanical_gap_center_mm


def _sections_intersect(a, b, tol):
    start, end = max(a[0, 0], b[0, 0]), min(a[-1, 0], b[-1, 0])
    if end - start <= tol:
        return False
    z = np.unique(np.clip(np.r_[a[:, 0], b[:, 0]], start, end))
    ab = np.interp(z, a[:, 0], a[:, 2]) - np.interp(z, b[:, 0], b[:, 1])
    ba = np.interp(z, b[:, 0], b[:, 2]) - np.interp(z, a[:, 0], a[:, 1])
    if np.any(np.minimum(ab, ba) > tol):
        return True
    # Between adjacent knots both differences are affine. Their minimum can
    # also peak where they cross, including between two disjoint end sections.
    difference = ab - ba
    i = np.flatnonzero(difference[:-1] * difference[1:] < 0)
    fraction = difference[i] / (difference[i] - difference[i+1])
    return bool(np.any(ab[i] + fraction * (ab[i+1] - ab[i]) > tol))


def validate_projector_iron(parts, geometry):
    """Check installed material profiles and explicit gap datums in module mm."""
    parents = {p['key']: p for p in parts if p.get('mechanical_layer_geometry') == 'independent_annuli'
               and p.get('magnetic_circuit_topology') == 'shared_pole_multi_gap'}
    if not parents:
        return
    tol = 1e-9
    by_key = {p['key']: p for p in parts}
    bore = float(geometry['projector_stack_vacuum_inner_diameter_mm'])
    liner = bore/2 + float(geometry['vacuum_liner_wall_thickness_mm'])
    for key, parent in parents.items():
        if 'pole_gap_center_offset_mm' not in parent:
            continue
        center = mechanical_gap_center_mm(parent)
        gap = float(parent['pole_gap_mm'])
        upper, lower = (by_key.get(f'{key}_{side}_pole', {}) for side in ('upper', 'lower'))
        if (gap <= 0 or upper.get('parent_key') != key or lower.get('parent_key') != key
                or not math.isclose(upper['local_end_z_mm'], center-gap/2, abs_tol=tol, rel_tol=0)
                or not math.isclose(lower['local_start_z_mm'], center+gap/2, abs_tol=tol, rel_tol=0)):
            raise ValueError(f'{key}: the material faces must bound the declared gap about its mechanical datum')

    materials = []
    for p in parts:
        if (p.get('parent_key') not in parents
                or p.get('mechanical_profile') not in MAGNETIC_BODIES | {COIL, 'magnetic_lens_housing'}
                or p.get('model_3d', {}).get('base', {}).get('kind', 'existing') != 'existing'):
            continue
        start, end = p['local_start_z_mm'], p['local_end_z_mm']
        profile = radial_profile_mm(p, end-start)
        if profile is not None:
            sections = [profile + [start, 0., 0.]]
        elif p['mechanical_profile'] == 'magnetic_pole_piece':
            # Alternative native pole recipes have their own contour checks.
            continue
        else:
            ri, ro = p['mechanical_inner_diameter_mm']/2, p['mechanical_outer_diameter_mm']/2
            sections = [np.array([[a, ri, ro], [b, ri, ro]])
                        for a, b in p.get('material_intervals_mm', ((start, end),))]
        for section in sections:
            if section[:, 1].min() < liner - tol:
                raise ValueError(f"{p['key']}: projector material intrudes into the vacuum liner")
            materials.append((p['key'], section))
    for (key_a, a), (key_b, b) in combinations(materials, 2):
        if _sections_intersect(a, b, tol):
            raise ValueError(f'Mechanical radial layers overlap between {key_a} and {key_b} at module Z '
                             f'{max(a[0, 0], b[0, 0]):g}..{min(a[-1, 0], b[-1, 0]):g} mm')
