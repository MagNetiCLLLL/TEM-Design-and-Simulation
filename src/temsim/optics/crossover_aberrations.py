"""Plane-local geometric spherical aberration of a controlled monochromatic pencil.

The chart uses actual forward mechanical slopes (dx/dz, dy/dz). A positive
round-lens Cs sends marginal rays through focus early, hence their transverse
intercept at the paraxial focus points along the outgoing slope. Retaining
the complete linear term makes an exact field-free drift change C1 alone.
These ray coefficients are not a coherent phase or an intrinsic lens constant.
"""
import numpy as np


# Independent vector fits allow rotations, astigmatism and non-gradient terms.
_POWERS = tuple((degree-j, j) for degree in range(6) for j in range(degree+1))
_AZIMUTH = np.arange(32) * (2*np.pi/32)
_UNIT = np.column_stack((np.cos(_AZIMUTH), np.sin(_AZIMUTH)))


def reference_pencil(radius_m, curvature_m1):
    """A single axial virtual point (parallel illumination at zero curvature)."""
    if not np.isfinite([radius_m, curvature_m1]).all() or radius_m <= 0:
        raise ValueError("A positive launch radius and finite curvature are required")
    radii = np.arange(1, 9)/8
    def rings(r, angles):
        return (r[:, None, None] * np.column_stack((np.cos(angles), np.sin(angles)))).reshape(-1, 2)
    train = rings(radii, np.arange(24)*2*np.pi/24)
    holdout = rings(radii-1/16, (np.arange(24)+.5)*2*np.pi/24)
    xy = radius_m*np.vstack((np.zeros((1, 2)), train, holdout))
    return np.column_stack((xy, curvature_m1*xy)), len(train)


def _design(u):
    return np.column_stack([u[:, 0]**i * u[:, 1]**j for i, j in _POWERS])


def _fit(slopes, positions, training, holdout):
    if np.count_nonzero(training) < len(_POWERS) or not np.any(holdout):
        raise ValueError("Insufficient independent training or holdout support")
    scale = float(np.max(np.linalg.norm(slopes[training], axis=1)))
    if scale <= 0:
        raise ValueError("Degenerate local angle chart")
    basis = _design(slopes/scale)
    vector, _, rank, singular = np.linalg.lstsq(basis[training], positions[training], rcond=1e-12)
    if rank != len(_POWERS) or singular[0]/singular[-1] > 1e7:
        raise ValueError("Unresolved rank or conditioning of the two-dimensional angular chart")
    error = np.linalg.norm(basis@vector - positions, axis=1)
    unit_basis = _design(_UNIT)
    result = dict(local_slope_radius=scale, condition=float(singular[0]/singular[-1]),
                  holdout_rms_m=float(np.sqrt(np.mean(error[holdout]**2))),
                  training_rms_m=float(np.sqrt(np.mean(error[training]**2))),
                  powers_xy=list(_POWERS), scaled_vector_coefficients_m=vector.tolist())
    for n in (1, 3, 5):
        take = np.array([i+j == n for i, j in _POWERS])
        radial = np.sum((unit_basis[:, take]@vector[take])*_UNIT, axis=1)
        result[f"c{n}_m"] = float(np.mean(radial)/scale**n)
    return result


def measure_geometric_cs(launch_xy, output, n_train, *, holdout_budget_m):
    """Azimuthal mean cubic at fixed Z. All input probes must reach this plane."""
    xy, raw = np.asarray(launch_xy, float), np.asarray(output, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or raw.shape != (len(xy), 4):
        raise ValueError("Ordered chief, training and holdout coordinates are required")
    if not 1 < n_train < len(raw)-1 or not np.isfinite(holdout_budget_m) or holdout_budget_m <= 0:
        raise ValueError("Invalid pupil split or holdout error budget")
    if not np.isfinite(raw).all() or not np.isfinite(xy).all():
        return dict(status="INCONCLUSIVE", reason="Pencil intercepted or missing", c3_mm=None)
    radius = np.linalg.norm(xy-xy[0], axis=1)
    index = np.arange(len(raw))
    slopes = raw[:, 2:] - raw[0, 2:]
    train, hold = index <= n_train, index > n_train
    inner = radius <= .75*np.max(radius[train])*(1+1e-13)
    try:
        full = _fit(slopes, raw[:, :2], train, hold)
        small = _fit(slopes, raw[:, :2], train & inner, hold & inner)
    except ValueError as exc:
        return dict(status="INCONCLUSIVE", reason=str(exc), c3_mm=None)
    c3 = full["c3_m"]
    # An empirical resolution indicator, not a statistical confidence interval.
    uncertainty = abs(c3-small["c3_m"]) + max(
        fit["holdout_rms_m"]/fit["local_slope_radius"]**3 for fit in (full, small))
    valid = max(full["holdout_rms_m"], small["holdout_rms_m"]) <= holdout_budget_m
    return dict(status="MEASURED_AT_FIXED_STEP" if valid else "INCONCLUSIVE",
                c3_mm=1e3*c3 if valid else None, fit=full, smaller_pupil_fit=small,
                radius_holdout_sensitivity_mm=1e3*uncertainty, holdout_budget_m=holdout_budget_m,
                sign_at_fixed_step=("positive" if c3>uncertainty else "negative" if c3 < -uncertainty
                                    else "unresolved") if valid else "unresolved",
                chief_xy_slopes=raw[0].tolist(), chart="laboratory outgoing mechanical slopes about chief",
                scope="Cumulative single-point geometric response; full linear term retained")
