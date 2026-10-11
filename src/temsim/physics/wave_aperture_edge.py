"""Empirical wall-edge error estimates for a round, aperture-selected segment.

This is an admission diagnostic for an explicitly approximate calculation,
not a wave propagator or a rigorous bound for a repeated-mask PDE. The
propagator retains the actual Cartesian mode, including its angular phase.
Here the same complex field is sampled around the physical axis to estimate
the probability of omitted rapidly oscillating wall-edge waves. No source,
optical strength, aperture, physical flux or wave normalization is changed.
"""
from functools import lru_cache
import math

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.optimize import brentq
from scipy.special import jv

from temsim.physics.wave_device import to_host


EMPIRICAL_POWER_MARGIN = 1.1


def _density_rings(wave, radius, count, wavelength_m):
    """Full complex envelope in an origin-zero, scalar-curvature gauge."""
    amplitude = to_host(wave.amplitude)
    basis = np.asarray(wave.basis_m)
    origin = np.asarray(wave.origin_m)
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else np.asarray(wave.curvature_m1)
    tilt = np.zeros(2) if wave.tilt_rad is None else np.asarray(wave.tilt_rad)
    q = float(np.trace(curvature)/2)
    angles = np.arange(count)*2*np.pi/count
    xy = np.asarray(radius)[:, None, None]*np.stack((np.cos(angles), np.sin(angles)), axis=0)[None]
    index = np.einsum('ij,njk->nik', np.linalg.inv(basis), xy-origin[None, :, None])
    index[:, 0] += amplitude.shape[1]//2
    index[:, 1] += amplitude.shape[0]//2
    coordinates = np.stack((index[:, 1], index[:, 0]))
    # Interpolate the slow envelope, never the rapidly oscillating carrier.
    value = map_coordinates(spline_filter(amplitude.real, order=3), coordinates, order=3,
                            mode='constant', cval=0., prefilter=False)
    value = value+1j*map_coordinates(spline_filter(amplitude.imag, order=3), coordinates, order=3,
                                   mode='constant', cval=0., prefilter=False)
    residual = curvature-q*np.eye(2)
    action = (.5*np.einsum('nix,ij,njx->nx', xy, residual, xy)
              +np.einsum('i,nix->nx', tilt-curvature@origin, xy)
              +.5*origin@curvature@origin-tilt@origin)
    return value*np.exp(2j*np.pi/wavelength_m*action)/math.sqrt(abs(np.linalg.det(basis)))


def _round_matrices(paths):
    """Use each short step's positive B to keep the round rotation continuous."""
    result = [np.eye(2)]
    matrix = np.eye(4)
    angle = 0.
    for path in paths:
        if np.any(path.offset):
            raise ValueError('Aperture edge estimate needs a centred, unforced round segment')
        angle += math.atan2(path.matrix[1, 2], path.matrix[0, 2])
        matrix = path.matrix@matrix
        rotation = np.array(((math.cos(angle), -math.sin(angle)),
                             (math.sin(angle), math.cos(angle))))
        scalar = np.array([[np.trace(rotation.T@matrix[:2, :2])/2, np.trace(rotation.T@matrix[:2, 2:])/2],
                           [np.trace(rotation.T@matrix[2:, :2])/2, np.trace(rotation.T@matrix[2:, 2:])/2]])
        reconstructed = np.block([[scalar[0, 0]*rotation, scalar[0, 1]*rotation],
                                  [scalar[1, 0]*rotation, scalar[1, 1]*rotation]])
        if not np.allclose(matrix, reconstructed, atol=1e-10, rtol=1e-8):
            raise ValueError('Aperture edge estimate does not cover non-round optics')
        result.append(scalar)
    return np.asarray(result)


@lru_cache(maxsize=12)
def _edge_geometry(matrix_bytes, count, radius_bytes, q, strength, wavelength, aperture, support_radius):
    matrices = np.frombuffer(matrix_bytes, dtype=np.float64).reshape(count, 2, 2)
    bores = np.frombuffer(radius_bytes, dtype=np.float64)
    final = matrices[-1]
    nodes, pre_caustic_limit = [], float('inf')
    for j, matrix in enumerate(matrices[1:-1], start=1):
        a, b = matrix[0]
        c, d = matrix[1]
        radius = bores[j]
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError('Aperture edge estimate requires finite positive bore radii')
        effective = a+b*q
        if b <= 0 or a/b+q >= 0:
            # Any possible earlier contact is counted separately as a full
            # input-tail error, not silently identified with a small edge.
            roots = []
            for sign in (-1., 1.):
                roots.extend(float(x.real) for x in np.roots((-b*strength, 0., effective, sign*radius))
                             if abs(x.imag) < 1e-12 and x.real > 0)
            if roots:
                pre_caustic_limit = min(pre_caustic_limit, min(roots))
            continue
        if effective >= 0:
            raise ValueError('Aperture edge estimate requires the measured post-focus outward branch')
        reach = lambda r: -effective*r+b*strength*r**3
        if reach(support_radius) < radius:
            continue
        incoming = brentq(lambda r: reach(r)-radius, 0., support_radius, xtol=1e-20)
        jacobian = (effective-b*strength*incoming**2)*(effective-3*b*strength*incoming**2)
        remaining = final@np.linalg.inv(matrix)
        ar, br = remaining[0]
        if br <= 0 or jacobian <= 0:
            raise ValueError('Aperture edge estimate encountered a conjugate or singular edge path')
        slope = (d*radius+incoming)/b
        endpoint = ar*radius+br*slope
        if endpoint < 20*aperture:
            raise ValueError('Wall edges are too close to the accepted stationary region')
        nodes.append((j, incoming, radius, slope, jacobian, br, endpoint))
    if not nodes:
        return np.empty((0, 7)), pre_caustic_limit, 0.
    nodes = np.asarray(nodes)
    indices = nodes[:, 0].astype(int)
    inverse = np.linalg.inv(matrices[indices])
    worst = 0.
    for j in range(1, len(nodes)):
        remaining = np.einsum('ij,njk->nik', matrices[indices[j]], inverse[:j])
        a, b = remaining[:, 0, 0], remaining[:, 0, 1]
        old_radius, slope = nodes[:j, 2], nodes[:j, 3]
        new_radius = nodes[j, 2]
        arrival = a*old_radius+b*slope
        gap = arrival-new_radius
        if np.any(b <= 0) or np.any(gap <= 0):
            raise ValueError('Wall sequence has a returning or stationary edge intercept')
        argument = 2*np.pi/wavelength*old_radius*new_radius/b
        # Absolute row sum of the leading forward/backward angular saddle
        # kernel. This diagnoses the stability of the empirical edge model.
        row = old_radius/np.sqrt(2*np.pi*argument)*(1/gap+1/(arrival+new_radius))
        worst = max(worst, float(np.sum(row)))
    if worst >= .2:
        raise ValueError('Empirical edge feedback is outside its checked weak-coupling regime')
    return nodes, pre_caustic_limit, worst


def estimate_aperture_wall_edges(wave, *, wavelength_m, spherical_m3, paths,
                                plane_z_mm, bore_radii_mm, aperture_radius_m,
                                accepted_probability):
    """Estimate one actual mode; returned powers use the input wave's norm.

    Only use after independently checking the segment is static/round and
    contains no intervening aperture or nonlinear event. The complete physical
    Cs and the final aperture must still be executed by the real propagator.
    Aggregate different modes with their actual weights; a dark mode need not
    satisfy an arbitrary relative error individually.
    """
    if (not math.isfinite(wavelength_m) or wavelength_m <= 0
            or not math.isfinite(spherical_m3) or spherical_m3 <= 0
            or not math.isfinite(aperture_radius_m) or aperture_radius_m <= 0
            or not math.isfinite(accepted_probability) or accepted_probability < 0):
        raise ValueError('Invalid physical inputs to aperture edge estimate')
    matrices = _round_matrices(paths)
    z = np.asarray(plane_z_mm)
    bores = np.asarray(bore_radii_mm, dtype=np.float64)*1e-3
    if len(z) != len(matrices) or bores.shape != (len(matrices),) or np.any(np.diff(z) <= 0):
        raise ValueError('Actual axial nodes, bore radii and executed maps must match')
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else np.asarray(wave.curvature_m1)
    q = float(np.trace(curvature)/2)
    if np.linalg.norm(curvature-q*np.eye(2)) > 1e-8*max(1., abs(q)):
        raise ValueError('Aperture edge stationary estimate needs a round phase carrier')
    xy = to_host(wave.coordinates_m())
    radius = np.sqrt(np.sum(xy*xy, axis=0))
    nodes, early_limit, feedback = _edge_geometry(np.ascontiguousarray(matrices).tobytes(),
        len(matrices), np.ascontiguousarray(bores).tobytes(), q, float(spherical_m3),
        float(wavelength_m), float(aperture_radius_m), float(radius.max()))
    mass = abs(to_host(wave.amplitude))**2
    early_probability = float(np.sum(mass[radius >= early_limit]))
    powers = []
    highest_fraction = 0.
    for angles in (32, 64):
        if not len(nodes):
            powers.append(0.)
            continue
        field = _density_rings(wave, nodes[:, 1], angles, wavelength_m)
        harmonics = np.fft.fft(field, axis=1)/angles
        order = np.abs(np.fft.fftfreq(angles)*angles).astype(int)
        argument = 2*np.pi/wavelength_m*nodes[:, 2]/nodes[:, 5]*aperture_radius_m
        if np.any(argument < 1e4):
            raise ValueError('Wall-edge angular saddles are insufficiently separated')
        jm = jv(order[None, :], argument[:, None])
        lower = jv(order[None, :]-1, argument[:, None])
        upper = jv(order[None, :]+1, argument[:, None])
        disk = np.maximum(0., np.pi*aperture_radius_m**2*(jm*jm-lower*upper))
        contributions = (abs(harmonics)**2*disk
            *(nodes[:, 2]/nodes[:, 6])[:, None]**2/nodes[:, 4, None])
        power = float(np.sum(contributions))
        powers.append(power)
        if angles == 64 and power:
            highest_fraction = float(np.sum(contributions[:, order >= 16])/power)
    convergence = abs(powers[1]-powers[0])/max(powers[1], 1e-300)
    if convergence > .005 or highest_fraction > .001:
        raise ValueError('Actual mode angular wall-edge estimate has not converged')
    edge_probability = EMPIRICAL_POWER_MARGIN*powers[-1]/(1-feedback)**2
    amplitude_error = math.sqrt(edge_probability)+math.sqrt(early_probability)
    intensity_error = 2*amplitude_error*math.sqrt(accepted_probability)+amplitude_error**2
    return dict(schema='empirical-aperture-wall-edge-v1',
        scope='Empirical leading-edge power model with actual-mode angular data and weak boundary feedback; not a rigorous repeated-mask PDE bound or total solver qualification',
        accepted_probability=float(accepted_probability),
        diagonal_edge_probability=powers[-1], empirical_edge_probability=edge_probability,
        pre_caustic_input_tail_probability=early_probability,
        empirical_complex_error_norm=amplitude_error,
        empirical_intensity_error_probability=intensity_error,
        empirical_power_margin=EMPIRICAL_POWER_MARGIN,
        feedback_absolute_row_sum=feedback, physical_wall_samples=len(nodes),
        angular_counts=[32, 64], angular_power_relative_change=convergence,
        high_angular_power_fraction=highest_fraction,
        full_cartesian_mode_retained=True, phase_fitted=False, normalized=False)


def aggregate_aperture_edge_estimates(weighted_records, *, maximum_relative_error=.01):
    """Add independent mode intensities, including weak or locally dark modes."""
    accepted = error = 0.
    for weight, record in weighted_records:
        if not math.isfinite(weight) or weight < 0 or record.get('schema') != 'empirical-aperture-wall-edge-v1':
            raise ValueError('Each mode needs a valid weight and its own empirical edge estimate')
        accepted += weight*record['accepted_probability']
        error += weight*record['empirical_intensity_error_probability']
    if not math.isfinite(maximum_relative_error) or maximum_relative_error <= 0:
        raise ValueError('The declared accepted-intensity tolerance must be positive')
    relative = error/accepted if accepted else (0. if error == 0 else float('inf'))
    return dict(schema='empirical-aperture-wall-mixture-v1', accepted_probability=accepted,
        empirical_intensity_error_probability=error, empirical_relative_intensity_error=relative,
        maximum_relative_error=maximum_relative_error, within_declared_accuracy=relative <= maximum_relative_error,
        scope='Empirical intensity error estimate for this actual incoherent mixture at the selected physical aperture; not rigorous PDE qualification')
