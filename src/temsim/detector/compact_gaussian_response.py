"""Compact physical Gaussian mixtures without a deposition grid.

The sixth-order translation expansion retains the actual weighted centre and
all mixed moments through total degree six. Neither PSF
widths, retained hit weights, nor the physical observation coordinates change.

For the standard 2D Gaussian g and a unit direction v,
||D_v**7 g||_1 = E|He_7(Z)| <= sqrt(7!). Taylor's integral remainder after
degree six is bounded globally by |u|**7 / sqrt(7!). Thus a nonnegative
mixture has truncation bound sum(w*|u|**7)/sqrt(7!), also valid on any ROI.
The reported bound excludes floating-point evaluation/accumulation error.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import ndtr


MAXIMUM_WHITENED_RADIUS = .001
ORDER = 6


@dataclass(frozen=True)
class CompactGaussianResponse:
    probability: np.ndarray
    metadata: dict


def _finite_sum(values):
    if not np.all(np.isfinite(values)):
        return None
    try:
        value = math.fsum(values)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _integrated_hermite(edges, centre, sigma):
    """Integrals of phi(t)*He_p(t), p=0..6, on the requested intervals.

    Positive-tail CDF subtraction uses the reflected lower tail. Gaussian
    derivatives beyond 40 sigma are below float64 range; clipping arguments
    before polynomial evaluation avoids 0*inf and incidental overflow.
    """
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        t = (edges-centre)/sigma
    if not np.all(np.isfinite(t)):
        return None
    lo, hi = t[:-1], t[1:]
    mass = np.where(lo > 0., ndtr(-lo)-ndtr(-hi), ndtr(hi)-ndtr(lo))
    bounded = np.clip(t, -40., 40.)
    with np.errstate(under='ignore'):
        phi = np.exp(-.5*bounded*bounded)/math.sqrt(2.*math.pi)
    phi[np.abs(t) > 40.] = 0.
    h0, h1 = np.ones_like(t), bounded
    polynomials = [h0, h1]
    for n in range(1, ORDER-1):
        polynomials.append(bounded*polynomials[-1]-n*polynomials[-2])
    integrated = [mass]
    for h in polynomials:
        edge_values = phi*h
        integrated.append(edge_values[:-1]-edge_values[1:])
    return integrated


def compact_gaussian_response(positions_mm, weights, x_edges, y_edges, point_spread):
    """Return a bounded compact response, or None for the ordinary fallback.

    Only axis-aligned Gaussian PSFs and positive-weight clouds within a maximum
    whitened expansion radius of 0.001 qualify. Off-axis anisotropy, broader clouds, or
    floating-point parameter extremes fall back without changing input data.
    Invalid numeric records are errors rather than a reason to conceal data.
    Arrays use [y, x]; positions and edges are physical millimetres. The result
    contains probability per emitted electron per pixel, without crop or
    total-flux normalization. Zero-weight records have no physical influence.
    A compact cloud about the viewport midpoint uses that fixed expansion
    origin, preserving nonzero first moments and avoiding changing-CDF-basis
    cancellation between two images. Otherwise its actual weighted centre is
    used. This is an algebraic expansion origin, not physical recentering.
    """
    if str(point_spread.model).strip().lower() != 'gaussian':
        return None
    sx, sy = float(point_spread.sigma_x_mm), float(point_spread.sigma_y_mm)
    angle = float(point_spread.rotation_deg)
    if not all(math.isfinite(value) for value in (sx, sy, angle)) or sx <= 0. or sy <= 0.:
        return None
    if sx != sy:
        axis_angle = math.remainder(angle, 180.)
        if abs(axis_angle) == 90.:
            sx, sy = sy, sx
        elif axis_angle != 0.:
            return None
    # Keep variance and whitening arithmetic in a finite normal range. The
    # regular renderer is responsible for any unsupported extreme settings.
    minimum, maximum = math.sqrt(np.finfo(float).tiny), math.sqrt(np.finfo(float).max)/4.
    if not minimum <= min(sx, sy) <= max(sx, sy) <= maximum:
        return None
    positions = np.asarray(positions_mm, dtype=float)
    w = np.asarray(weights, dtype=float)
    xe, ye = np.asarray(x_edges, dtype=float), np.asarray(y_edges, dtype=float)
    if w.ndim != 1 or positions.shape != (len(w), 2):
        raise ValueError('Compact Gaussian response requires matching Nx2 positions and weights')
    if not np.all(np.isfinite(w)) or np.any(w < 0.):
        raise ValueError('Compact Gaussian response requires finite nonnegative weights')
    if any(edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges))
           or np.any(edges[1:] <= edges[:-1]) for edges in (xe, ye)):
        raise ValueError('Compact Gaussian response requires finite increasing pixel edges')
    selected = w > 0.
    q, w = positions[selected], w[selected]
    if not np.all(np.isfinite(q)):
        raise ValueError('Positive-weight Gaussian positions must be finite')
    shape = (len(ye)-1, len(xe)-1)
    if not len(w):
        return CompactGaussianResponse(np.zeros(shape), dict(order=ORDER,
            centre_mm=None, total_weight=0., central_covariance_mm2=None,
            expansion_centre_mm=None, expansion_centre_selection='no positive hits',
            radius_over_sigma_max=0., global_l1_truncation_bound=0.,
            bound_scope='truncation only; excludes floating-point evaluation'))
    total = _finite_sum(w)
    if total is None or total <= 0.:
        return None
    # Translation by an anchor avoids summing a large common detector offset
    # repeatedly. fsum preserves zero-order flux when records are replicated.
    with np.errstate(over='ignore', invalid='ignore'):
        delta = q-q[0]
    if not np.all(np.isfinite(delta)):
        return None
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        fractions = w/total
        offsets = [_finite_sum(fractions*delta[:, axis]) for axis in range(2)]
        if any(value is None for value in offsets):
            return None
        centre = q[0]+np.asarray(offsets)
    if not np.all(np.isfinite(centre)):
        return None
    viewport_centre = np.array([.5*xe[0]+.5*xe[-1], .5*ye[0]+.5*ye[-1]])
    for selection, expansion_centre in [('viewport midpoint', viewport_centre), ('weighted mean', centre)]:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            u = (q-expansion_centre)/np.array([sx, sy])
            radius = np.hypot(u[:, 0], u[:, 1])
        if np.all(np.isfinite(radius)) and np.max(radius) <= MAXIMUM_WHITENED_RADIUS:
            break
    else:
        return None
    bx = _integrated_hermite(xe, expansion_centre[0], sx)
    by = _integrated_hermite(ye, expansion_centre[1], sy)
    if bx is None or by is None:
        return None
    with np.errstate(under='ignore'):
        bound = math.fsum(w*radius**7)/math.sqrt(math.factorial(ORDER+1))
    powers_x = [np.ones(len(w))]
    powers_y = [np.ones(len(w))]
    for _ in range(ORDER):
        powers_x.append(powers_x[-1]*u[:, 0])
        powers_y.append(powers_y[-1]*u[:, 1])
    response = total*np.outer(by[0], bx[0])
    for p in range(ORDER+1):
        for r in range(ORDER+1-p):
            if p+r == 0:
                continue
            coefficient = math.fsum(w*powers_x[p]*powers_y[r])/math.factorial(p)/math.factorial(r)
            if coefficient:
                response += coefficient*np.outer(by[r], bx[p])
    # No forced clipping or renormalization: an inadmissible numeric result
    # uses the existing renderer, rather than silently changing probability.
    if not np.all(np.isfinite(response)) or np.any(response < 0.):
        return None
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        centred = q-centre
        covariance = [[_finite_sum(fractions*centred[:, i]*centred[:, j])
                       for j in range(2)] for i in range(2)]
    if any(value is None for row in covariance for value in row):
        return None
    return CompactGaussianResponse(response, dict(order=ORDER,
        centre_mm=centre.tolist(), total_weight=total,
        expansion_centre_mm=expansion_centre.tolist(), expansion_centre_selection=selection,
        central_covariance_mm2=covariance, radius_over_sigma_max=float(radius.max()),
        laboratory_sigma_mm=[sx, sy],
        global_l1_truncation_bound=bound,
        bound_scope='global L1 truncation only; excludes floating-point evaluation'))
