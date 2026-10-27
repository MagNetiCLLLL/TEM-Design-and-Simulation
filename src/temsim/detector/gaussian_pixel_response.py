"""Deterministic pixel integrals and bounded, conservative Gaussian mixtures.

Coordinates and widths are millimetres; values are source probability per
pixel. Neither route recentres hits or normalizes a cropped response. The
direct route integrates a conditional normal with 16-point Gauss--Legendre
quadrature on intervals no wider than half either conditional scale. A two-point
rule is used only when its fourth-derivative relative error bound is at most
2e-13, including the polynomial in a compact moment expansion. The
fallback resolves the smallest principal sigma with 24 cells, then integrates
the piecewise bilinear density over each output pixel, including tiny views.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import ndtr
from scipy.sparse import csr_matrix

from temsim.detector.point_spread import apply_point_spread

TAIL_SIGMA = 9.
DIRECT_EVALUATION_BUDGET = 8 * 1024**2
GRID_CELL_BUDGET = 4 * 1024**2
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(16)
_SHORT_NODES, _SHORT_WEIGHTS = np.polynomial.legendre.leggauss(2)
SHORT_INTERVAL_RELATIVE_BUDGET = 2e-13


def _covariance_scales(psf):
    angle = math.radians(psf.rotation_deg)
    c, s = math.cos(angle), math.sin(angle)
    sx, sy = psf.sigma_x_mm, psf.sigma_y_mm
    mx, my = math.hypot(c*sx, s*sy), math.hypot(s*sx, c*sy)
    conditional = min(sx, sy) * (max(sx, sy)/mx)
    slope = (c*sx/mx)*(s*sx) - (s*sy/mx)*(c*sy)
    if not all(math.isfinite(v) for v in (mx, my, conditional, slope)) or conditional <= 0.:
        raise ValueError("Detector Gaussian covariance is not numerically resolved")
    return mx, my, conditional, slope


def _interval_mass(midpoint, width):
    """Stable normal interval mass, including sub-PSF nanometre viewports."""
    half = .5*width
    lo, hi = midpoint-half, midpoint+half
    result = np.where(lo > 0., ndtr(-lo)-ndtr(-hi), ndtr(hi)-ndtr(lo))
    small = (width*(1.+np.abs(midpoint)) <= 1e-3) & (np.abs(midpoint) < 40.)
    t = np.clip(midpoint, -40., 40.)
    t2, w2 = t*t, np.minimum(width, 1.)**2
    series = (1.+(t2-1.)*w2/24.+(t2*t2-6.*t2+3.)*w2*w2/1920.)
    return np.where(small, width*np.exp(-.5*t2)/math.sqrt(2.*math.pi)*series, result)


def _pixel_span(edges, low, high):
    return (max(0, int(np.searchsorted(edges, low, side="right"))-1),
            min(len(edges)-1, int(np.searchsorted(edges, high, side="left"))))


def _short_interval_rule(lo, hi, subdivisions, ylo, yhi, slope, moments):
    """Admit GL2 only with a conservative relative quadrature remainder.

    On a segment of width d, GL2 error/integral is at most
    d**4/4320 * sup|f''''|/inf(f). For f=phi(t)phi(z) P(t,z), z'= -slope,
    derivatives are exact Hermite polynomials. Their absolute-coefficient
    bounds give the numerator and a positive lower bound for P; the Gaussian
    max/min ratio at fixed y is bounded by exp((T+|slope|*Z)*d). These bounds
    also hold after integration over any positive-width y interval. A failed
    or unresolved bound retains the original 16-point rule, never GL2.
    """
    slope = np.float64(slope)
    t = np.maximum(abs(lo), abs(hi))
    z = np.maximum.reduce([abs(ylo-slope*lo), abs(ylo-slope*hi),
                           abs(yhi-slope*lo), abs(yhi-slope*hi)])
    coefficients = np.ones((1, 1)) if moments is None else moments
    degree = len(coefficients)-1
    ht, hz = [np.ones_like(t), t], [np.ones_like(z), z]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for n in range(1, degree+4):
            ht.append(t*ht[-1]+n*ht[-2])
            hz.append(z*hz[-1]+n*hz[-2])
        derivative, variation = np.zeros_like(t), np.zeros_like(t)
        for p, r in zip(*np.nonzero(coefficients)):
            coefficient = abs(coefficients[p, r])
            if p+r:
                variation += coefficient*ht[p]*hz[r]
            derivative += coefficient*sum(math.comb(4, j)*abs(slope)**(4-j)*ht[p+j]*hz[r+4-j]
                                          for j in range(5))
        lower = coefficients[0, 0]-variation
        width = (hi-lo)/subdivisions
        logarithm = (t+abs(slope)*z)*width
        relative = width**4/4320.*derivative/np.where(lower > 0., lower, 1.)*np.exp(np.minimum(logarithm, 700.))
    return ((lower > 0.) & (logarithm < 700.) & np.isfinite(relative)
            & (relative <= SHORT_INTERVAL_RELATIVE_BUDGET))


def _integrated_hermites(midpoint, width, order):
    """Normal/Hermite interval integrals with stable short-interval limits."""
    values = [_interval_mass(midpoint, width)]
    lo, hi = midpoint-.5*width, midpoint+.5*width
    a, b = np.clip(lo, -40., 40.), np.clip(hi, -40., 40.)
    pa, pb = np.exp(-.5*a*a)/math.sqrt(2.*math.pi), np.exp(-.5*b*b)/math.sqrt(2.*math.pi)
    pa, pb = np.where(np.abs(lo) <= 40., pa, 0.), np.where(np.abs(hi) <= 40., pb, 0.)
    ha, hb = [np.ones_like(a), a], [np.ones_like(b), b]
    t = np.clip(midpoint, -40., 40.)
    hm = [np.ones_like(t), t]
    for n in range(1, order+4):
        hm.append(t*hm[-1]-n*hm[-2])
    for n in range(1, order-1):
        ha.append(a*ha[-1]-n*ha[-2])
        hb.append(b*hb[-1]-n*hb[-2])
    small = (width*(1.+np.abs(midpoint)) <= 1e-3) & (np.abs(midpoint) < 40.)
    w2 = np.minimum(width, 1.)**2
    phi = np.exp(-.5*t*t)/math.sqrt(2.*math.pi)
    for n in range(1, order+1):
        series = width*phi*(hm[n]+w2*hm[n+2]/24.+w2*w2*hm[n+4]/1920.)
        values.append(np.where(small, series, pa*ha[n-1]-pb*hb[n-1]))
    return values


def integrated_gaussian_response(positions, weights, xe, ye, psf, *, moments=None):
    """Return direct pixel integrals, or None before exceeding the work budget.

    Exact duplicate positions may share an integral. This is not spatial
    binning: distinct subpixel positions and their weights remain distinct.
    Marginal tails beyond nine sigma have total mass below 5e-19 per unit
    source weight. Quadrature work and temporary arrays are explicitly bounded.
    """
    selected = np.asarray(weights) > 0.
    q, w = np.asarray(positions)[selected], np.asarray(weights)[selected]
    response = np.zeros((len(ye)-1, len(xe)-1))
    if not len(w):
        return response
    q, inverse = np.unique(q, axis=0, return_inverse=True)
    w = np.bincount(inverse, weights=w, minlength=len(q))
    mx, my, conditional, slope = _covariance_scales(psf)
    step = .5/max(1., abs(slope/conditional))
    plans, cost = [], 0
    for position, weight in zip(q, w):
        first, last = _pixel_span(xe, position[0]-TAIL_SIGMA*mx, position[0]+TAIL_SIGMA*mx)
        bottom, top = _pixel_span(ye, position[1]-TAIL_SIGMA*my, position[1]+TAIL_SIGMA*my)
        if first >= last or bottom >= top:
            continue
        lo = np.maximum((xe[first:last]-position[0])/mx, -TAIL_SIGMA)
        hi = np.minimum((xe[first+1:last+1]-position[0])/mx, TAIL_SIGMA)
        subdivisions = np.maximum(1., np.ceil((hi-lo)/step))
        short = _short_interval_rule(lo, hi, subdivisions,
            (ye[bottom]-position[1])/conditional, (ye[top]-position[1])/conditional,
            slope/conditional, moments)
        orders = np.where(short, len(_SHORT_NODES), len(_NODES))
        work = float(np.sum(subdivisions*orders)) * (top-bottom)
        if not math.isfinite(work) or cost+work > DIRECT_EVALUATION_BUDGET:
            return None
        cost += work
        plans.append((position, weight, first, bottom, top, lo, hi, subdivisions.astype(int), short))
    for position, weight, first, bottom, top, lo, hi, subdivisions, short in plans:
        ymid = (.5*ye[bottom:top]+.5*ye[bottom+1:top+1]-position[1])/conditional
        ywidth = np.diff(ye[bottom:top+1])/conditional
        for offset, (a, b, count, use_short) in enumerate(zip(lo, hi, subdivisions, short)):
            edges = np.linspace(a, b, count+1)
            mid, half = .5*(edges[:-1]+edges[1:]), .5*np.diff(edges)
            nodes, node_weights = (_SHORT_NODES, _SHORT_WEIGHTS) if use_short else (_NODES, _WEIGHTS)
            t = (mid[:, None]+half[:, None]*nodes).ravel()
            quadrature = (half[:, None]*node_weights).ravel()*np.exp(-.5*t*t)/math.sqrt(2.*math.pi)
            # Bound each temporary conditional-CDF array as well as total work.
            value = np.zeros(top-bottom)
            for start in range(0, len(t), 256):
                selected_t = t[start:start+256]
                midpoint = ymid[:, None]-(slope/conditional)*selected_t[None, :]
                if moments is None:
                    mass = _interval_mass(midpoint, ywidth[:, None])
                    value += mass @ quadrature[start:start+256]
                else:
                    order = len(moments)-1
                    hy = _integrated_hermites(midpoint, ywidth[:, None], order)
                    hx = [np.ones_like(selected_t), selected_t]
                    for degree in range(1, order):
                        hx.append(selected_t*hx[-1]-degree*hx[-2])
                    for r in range(order+1):
                        coefficient = sum(moments[p, r]*hx[p] for p in range(order+1-r))
                        value += hy[r] @ (quadrature[start:start+256]*coefficient)
            response[bottom:top, first+offset] += weight*value
    return response


def compact_rotated_gaussian_response(positions, weights, xe, ye, psf):
    """Sixth-order compact cloud in Cholesky-whitened physical coordinates."""
    from temsim.detector.compact_gaussian_response import (
        CompactGaussianResponse, ORDER, MAXIMUM_WHITENED_RADIUS, _finite_sum,
    )
    xe, ye = np.asarray(xe, dtype=float), np.asarray(ye, dtype=float)
    if any(edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
           or np.any(np.diff(edges) <= 0.) for edges in (xe, ye)):
        raise ValueError("Compact Gaussian response requires finite increasing pixel edges")
    q, w = np.asarray(positions, dtype=float), np.asarray(weights, dtype=float)
    if w.ndim != 1 or q.shape != (len(w), 2) or not np.isfinite(w).all() or np.any(w < 0.):
        raise ValueError("Compact Gaussian response requires matching positions and nonnegative finite weights")
    selected = w > 0.
    q, w = q[selected], w[selected]
    if not np.isfinite(q).all():
        raise ValueError("Positive-weight Gaussian positions must be finite")
    if not len(w):
        return None
    total = _finite_sum(w)
    if total is None or total <= 0.:
        return None
    fractions = w/total
    offsets = [_finite_sum(fractions*(q[:, axis]-q[0, axis])) for axis in range(2)]
    if any(value is None for value in offsets):
        return None
    centre = q[0]+np.asarray(offsets)
    mx, my, conditional, slope = _covariance_scales(psf)
    viewport = np.array([.5*xe[0]+.5*xe[-1], .5*ye[0]+.5*ye[-1]])
    for selection, origin in (("viewport midpoint", viewport), ("weighted mean", centre)):
        u = np.empty_like(q)
        u[:, 0] = (q[:, 0]-origin[0])/mx
        u[:, 1] = ((q[:, 1]-origin[1])-slope*u[:, 0])/conditional
        radius = np.hypot(u[:, 0], u[:, 1])
        if np.isfinite(radius).all() and radius.max() <= MAXIMUM_WHITENED_RADIUS:
            break
    else:
        return None
    coefficients = np.zeros((ORDER+1, ORDER+1))
    for p in range(ORDER+1):
        for r in range(ORDER+1-p):
            coefficients[p, r] = math.fsum(w*u[:, 0]**p*u[:, 1]**r)/math.factorial(p)/math.factorial(r)
    response = integrated_gaussian_response(origin[None, :], np.ones(1), xe, ye, psf,
                                             moments=coefficients)
    if (response is None and DIRECT_EVALUATION_BUDGET > 0
            and 2*(len(xe)-1)*(len(ye)-1) > DIRECT_EVALUATION_BUDGET):
        # A single compact kernel needs at least two evaluations per output
        # pixel even for GL2. Tile that unavoidable output-sized work instead
        # of degrading its physical response or increasing the per-call budget.
        rows = max(1, DIRECT_EVALUATION_BUDGET//(len(_SHORT_NODES)*(len(xe)-1)))
        response = np.empty((len(ye)-1, len(xe)-1))
        first = 0
        while first < len(ye)-1:
            last = min(len(ye)-1, first+rows)
            tile = integrated_gaussian_response(origin[None, :], np.ones(1), xe,
                ye[first:last+1], psf, moments=coefficients)
            if tile is None:
                if rows == 1:
                    return None
                rows = max(1, rows//2)
                continue
            response[first:last] = tile
            first = last
    if response is None or not np.isfinite(response).all() or np.any(response < 0.):
        return None
    displacement = q-centre
    covariance = [[_finite_sum(fractions*displacement[:, i]*displacement[:, j])
                   for j in range(2)] for i in range(2)]
    return CompactGaussianResponse(response, dict(order=ORDER, centre_mm=centre.tolist(),
        total_weight=total, expansion_centre_mm=origin.tolist(), expansion_centre_selection=selection,
        central_covariance_mm2=covariance, radius_over_sigma_max=float(radius.max()),
        laboratory_sigma_mm=[mx, my], whitening="Cholesky of rotated covariance",
        global_l1_truncation_bound=math.fsum(w*radius**7)/math.sqrt(math.factorial(ORDER+1)),
        marginal_tail_probability_bound=2.*math.erfc(TAIL_SIGMA/math.sqrt(2.))*total,
        bound_scope="global L1 translation truncation only; excludes floating-point and quadrature evaluation"))


def _linear_integral_matrix(edges, centres, pitch):
    """Integrate linear hat functions, without subtracting cumulative sums."""
    rows, columns, values = [], [], []
    for row, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        lo = max(0, int(np.searchsorted(centres, left, side="right"))-1)
        hi = min(len(centres)-1, int(np.searchsorted(centres, right, side="left")))
        for column in range(lo, hi):
            a, b = max(left, centres[column]), min(right, centres[column+1])
            if b <= a:
                continue
            fraction = ((a-centres[column])+.5*(b-a))/pitch
            rows.extend((row, row))
            columns.extend((column, column+1))
            values.extend(((b-a)*(1.-fraction), (b-a)*fraction))
    return csr_matrix((values, (rows, columns)), shape=(len(edges)-1, len(centres)))


def sampled_gaussian_response(positions, weights, xe, ye, psf):
    """PSF-resolved convolution plus conservative integration on output pixels.

    The work grid is independent of display sampling and anchored at zero in
    physical coordinates. Interpolation integrates a positive bilinear density;
    its whole-plane flux and first moment are those of the deposited mixture.
    Nine-sigma support loses tails rather than redistributing them. An excessive
    physical grid is rejected instead of silently coarsening the PSF.
    """
    mx, my, _, _ = _covariance_scales(psf)
    pitch = min(psf.sigma_x_mm, psf.sigma_y_mm)/24.
    if not math.isfinite(pitch) or pitch <= 0.:
        raise ValueError("Detector Gaussian grid pitch is not numerically resolved")
    q, w = np.asarray(positions), np.asarray(weights)
    keep = ((w > 0.) & (q[:, 0] >= xe[0]-TAIL_SIGMA*mx) & (q[:, 0] <= xe[-1]+TAIL_SIGMA*mx)
            & (q[:, 1] >= ye[0]-TAIL_SIGMA*my) & (q[:, 1] <= ye[-1]+TAIL_SIGMA*my))
    q, w = q[keep], w[keep]
    if not len(w):
        return np.zeros((len(ye)-1, len(xe)-1)), (pitch, pitch)
    lower, upper = q.min(axis=0), q.max(axis=0)
    view_low, view_high = np.array([xe[0], ye[0]]), np.array([xe[-1], ye[-1]])
    halo = TAIL_SIGMA*np.array([mx, my])
    low = np.minimum(lower, np.maximum(view_low, lower-halo))
    high = np.maximum(upper, np.minimum(view_high, upper+halo))
    starts, stops = np.floor(low/pitch)-2., np.ceil(high/pitch)+2.
    shape = stops-starts+1.
    kernel_radius = TAIL_SIGMA*(max(psf.sigma_x_mm, psf.sigma_y_mm)/pitch)
    if (not math.isfinite(kernel_radius) or not np.all(np.isfinite(shape))
            or np.prod(shape+2*kernel_radius+2) > GRID_CELL_BUDGET):
        raise ValueError("Detector PSF exceeds the bounded physical response grid; narrow the view or inspect particle arrivals without Detector blur")
    nx, ny = (int(value) for value in shape)
    x, y = (np.arange(nx)+starts[0])*pitch, (np.arange(ny)+starts[1])*pitch
    gx, gy = (q[:, 0]-x[0])/pitch, (q[:, 1]-y[0])/pitch
    ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx-ix, gy-iy
    deposited = np.zeros((ny, nx))
    for ox, oy, fraction in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                              (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        deposited += np.bincount((iy+oy)*nx+ix+ox, weights=w*fraction,
                                 minlength=nx*ny).reshape(ny, nx)
    response = apply_point_spread(deposited, psf, pixel_size_x_mm=pitch,
                                 pixel_size_y_mm=pitch, support_sigma=TAIL_SIGMA,
                                 preserve_tail_loss=True)
    integrate_x = _linear_integral_matrix(xe, x, pitch)
    integrate_y = _linear_integral_matrix(ye, y, pitch)
    probability = (integrate_x @ (integrate_y @ response).T).T / (pitch*pitch)
    return probability, (pitch, pitch)
