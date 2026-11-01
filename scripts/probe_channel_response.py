"""Resolve geometric aberrations in a fixed laboratory outgoing-slope chart.

The coefficients are length coefficients of the gradient-compatible part of
the transverse ray map. They are not measurements of coherent wave phase.
"""
import numpy as np


TERMS = (("C1", 1, 0), ("A1", 1, 2), ("A2", 2, 3), ("B2", 2, 1),
         ("C3", 3, 0), ("S3", 3, 2), ("A3", 3, 4), ("B4", 4, 1),
         ("D4", 4, 3), ("A4", 4, 5), ("C5", 5, 0), ("A5", 5, 6),
         ("C52", 5, 2), ("C54", 5, 4))


def geometric_coefficients(rays, training_count, *, degree=7, omit_a5=False):
    """Fit disjoint training/holdout rays. Chief, training then holdout ordering."""
    rays = np.asarray(rays, float)
    if (rays.ndim != 2 or rays.shape[1] != 4 or not np.isfinite(rays).all()
            or degree < 5 or not 0 < training_count < len(rays)-1):
        raise ValueError("A finite two-dimensional reference family and independent holdout are required")
    train = slice(0, training_count+1)
    hold = slice(training_count+1, None)
    slopes = rays[:, 2:] - rays[0, 2:]
    radius = np.max(np.linalg.norm(slopes[train], axis=1))
    if radius == 0:
        raise ValueError("The reference pupil is degenerate")
    # Complex monomials identify azimuthal harmonics directly. A5 is the
    # unique degree-five conjugate(u)**5 column, independent of radial C5.
    u = (slopes[:, 0] + 1j*slopes[:, 1])/radius
    powers = [(n-k, k) for n in range(degree+1) for k in range(n+1)
              if not (omit_a5 and n == k == 5)]
    matrix = np.column_stack([u**i * u.conj()**j for i, j in powers])
    intercept = (rays[:, 0]-rays[0, 0]) + 1j*(rays[:, 1]-rays[0, 1])
    coef, _, rank, singular = np.linalg.lstsq(matrix[train], intercept[train], rcond=1e-12)
    if rank < len(powers) or singular[0]/singular[-1] > 1e7:
        raise ValueError("The reference pupil does not resolve the fitted harmonics")
    components = {(i+j, i-j): v/radius**(i+j) for (i, j), v in zip(powers, coef)}
    terms = {}
    for name, n, m in TERMS:
        if not m:
            c = components[n, 1].real
        else:
            a, b = (1+m/(n+1))/2, (1-m/(n+1))/2
            minus = components.get((n, 1-m), 0j)
            plus = components.get((n, 1+m), 0j)
            c = (a*minus + b*plus.conjugate())/(a*a+b*b)
        terms[name] = [float(np.real(c)), float(np.imag(c))]
    return dict(terms_m=terms, chief=rays[0].tolist(), slope_radius=radius,
                holdout_rms_pm=float(np.sqrt(np.mean(abs(matrix[hold]@coef-intercept[hold])**2))*1e12),
                condition=float(singular[0]/singular[-1]), degree=degree, a5_omitted=omit_a5)
