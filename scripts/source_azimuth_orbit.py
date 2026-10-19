"""Offline qualification helper: common source-azimuth orbit for radial quadrature.

The 5D base order is position radial CDF, position azimuth CDF, direction
azimuth CDF, Young-energy CDF, Boersch-energy CDF. Shift both azimuths by
j/K. The existing +/- ray pair supplies the opposite half of each orbit.
This is a measure-preserving source quadrature, not an optical symmetry:
every expanded source ray still needs its own actual gun/column propagation.
No downstream image rotation, centroid subtraction or loss renormalisation
is permitted. K only controls common-azimuth integration; it does not refine
the remaining variables or certify high-harmonic/coupled aperture responses.

This helper defines an explicit offline qualification source table; it does not change the default production emitter or optical model.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AzimuthOrbit:
    base_points: np.ndarray
    independent_base_index: np.ndarray
    half_orbit_index: np.ndarray
    radial_base_weight: np.ndarray
    metadata: dict


def expand_common_azimuth(base_points, orbit_size):
    """Return M*K/2 radial bases, each with probability 1/(M*K/2).

    A subsequent radial-cell Gauss weight must multiply radial_base_weight;
    the existing two signs each multiply by1/2. If the existing root helper
    already divides its weights by the expanded base count, do not multiply
    by radial_base_weight a second time. Base-major ordering retains the
    exact expanded 5D prefix when independent M grows at fixed K. Solved
    radial-node tables need not have prefixes as boundaries change.
    """
    points=np.asarray(base_points,dtype=np.float64)
    if (points.ndim!=2 or points.shape[1]!=5 or not len(points)
            or not np.isfinite(points).all() or np.any((points<0)|(points>1))):
        raise ValueError('Finite nonempty 5D source CDF bases in [0,1] are required')
    if type(orbit_size) is not int or orbit_size not in (2,4,8,16):
        raise ValueError('Orbit size must be exactly 2, 4, 8 or 16')
    m=len(points);half=orbit_size//2
    parent=np.repeat(np.arange(m,dtype=np.int64),half)
    orbit=np.tile(np.arange(half,dtype=np.int64),m)
    expanded=points[parent].copy()
    if orbit_size!=2:
        expanded[:,[1,2]]=(expanded[:,[1,2]]+orbit[:,None]/orbit_size)%1.
    weights=np.full(len(expanded),1/len(expanded))
    return AzimuthOrbit(expanded,parent,orbit,weights,dict(
        schema='offline-common-source-azimuth-orbit-v1',
        independent_sobol_base_count=m,common_azimuth_orbit_size=orbit_size,
        half_orbits_per_independent_base=half,expanded_radial_base_count=len(expanded),
        source_orientation_count_before_radial_nodes=m*orbit_size,
        zero_weight_anchors_are_not_probability_nodes=True,
        weighting='Each radial cell weight uses 1/(M*K/2), then +/- each uses 1/2; total full-source probability remains one.',
        prefix_scope='Fixed K, five-dimensional base-major prefix only; not final six-dimensional radial tables.',
        scope='Source quadrature proposal only; every ray needs actual unchanged optics. No physical acceptance or convergence certification.'))


def complete_orientation_nodes(orbit):
    """Analytic-test convenience: make the full K orientations explicit."""
    points=np.repeat(orbit.base_points,2,axis=0)
    points[1::2,[1,2]]=(points[1::2,[1,2]]+.5)%1.
    return points,np.repeat(orbit.radial_base_weight/2,2)
