"""Separate measured corrector beam waists from theoretical plane conjugacy."""

import numpy as np

from temsim.physics.beam_waist import detect_beam_waist


def measured_corrector_crossovers(
    checkpoints, masks, weights, z_min_mm, z_max_mm, *, minimum_rays=5,
):
    """Find sampled, two-dimensional waists in a clear physical interval.

    Inputs are executed ``PropagationCheckpoints`` (positions in metres,
    mechanical slopes in radians, Z in mm), not a first-order transfer matrix
    or interpolated plot history. One positive-current cohort survives the
    ENTIRE sampled interval; clipping cannot manufacture a minimum by changing
    the population. Every accepted three-plane bracket lies strictly between
    the specified physical boundaries.

    Require a radial RMS minimum and negative-to-positive position/slope
    covariance. Both principal spot variances must also decrease and increase
    across the bracket, rejecting a line focus or pupil conjugacy mistaken for
    a two-dimensional source image. This is a conservative, sampled condition:
    a separated astigmatic pair may have a radial minimum without satisfying it.
    Repeating with a finer Z grid and transport step remains a separate check.
    """
    z = np.asarray(checkpoints.z_mm, float)
    arrays = [np.asarray(getattr(checkpoints, key), float) for key in
              ("x_m", "y_m", "tx_rad", "ty_rad")]
    masks = np.asarray(masks, bool)
    weights = np.asarray(weights, float)
    if (z.ndim != 1 or np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0)
            or weights.ndim != 1 or masks.shape != (z.size, weights.size)
            or any(a.shape != masks.shape for a in arrays)
            or np.any(~np.isfinite(weights)) or np.any(weights < 0)
            or not np.isfinite(weights.sum()) or weights.sum() <= 0
            or not np.isfinite([z_min_mm, z_max_mm]).all()
            or z_min_mm >= z_max_mm or minimum_rays < 2):
        raise ValueError("Invalid measured corrector crossover checkpoints")
    selected = np.flatnonzero((z > z_min_mm) & (z < z_max_mm))
    if selected.size < 3:
        return []
    live = np.all(masks[selected], axis=0) & (weights > 0)
    if np.count_nonzero(live) < minimum_rays:
        return []
    if any(np.any(~np.isfinite(a[np.ix_(selected, live)])) for a in arrays):
        raise ValueError("Nonfinite surviving corrector crossover coordinates")
    w = weights[live] / weights[live].sum()
    x, y, tx, ty = [a[np.ix_(selected, live)] for a in arrays]
    x, y, tx, ty = [a - (a @ w)[:, None] for a in (x, y, tx, ty)]
    xx, xy, yy = (x*x) @ w, (x*y) @ w, (y*y) @ w
    radius_squared = xx + yy
    covariance = (x*tx + y*ty) @ w
    spot_covariance = np.empty((selected.size, 2, 2))
    spot_covariance[:, 0, 0] = xx
    spot_covariance[:, 1, 1] = yy
    spot_covariance[:, 0, 1] = spot_covariance[:, 1, 0] = xy
    principal_variances = np.linalg.eigvalsh(spot_covariance)
    out = []
    for k in range(1, selected.size - 1):
        if not (covariance[k-1] < 0 < covariance[k+1]
                and radius_squared[k] <= radius_squared[k-1]
                and radius_squared[k] <= radius_squared[k+1]):
            continue
        # A constant principal width must not qualify through eigensolver
        # round-off. This is only a machine-precision comparison, not a fitted
        # physical acceptance threshold or an imposed roundness requirement.
        noise = 32*np.finfo(float).eps * np.max(principal_variances[k-1:k+2])
        if not np.all(principal_variances[k] + noise
                      < np.minimum(principal_variances[k-1], principal_variances[k+1])):
            continue
        out.append(dict(
            z_mm=float(z[selected[k]]),
            bracket_mm=[float(z[selected[k-1]]), float(z[selected[k+1]])],
            rms_radius_nm=float(np.sqrt(radius_squared[k])*1e9),
            bracket_rms_radius_nm=(np.sqrt(radius_squared[k-1:k+2])*1e9).tolist(),
            bracket_principal_rms_nm=(np.sqrt(np.maximum(
                principal_variances[k-1:k+2], 0))*1e9).tolist(),
            bracket_covariance_m_rad=covariance[k-1:k+2].tolist(),
            ray_count=int(np.count_nonzero(live)),
            source_current_fraction=float(weights[live].sum()/weights.sum()),
            cohort="same positive-current rays surviving the entire sampled interval",
            status="EXECUTED_AT_FIXED_SAMPLING",
            kind="current-weighted two-dimensional corrector beam waist",
        ))
    return out

def detect_corrector_crossovers(branch, targets, half_window_mm=14.0):
    """Return weighted display markers near historical design targets.

    These markers are not the exact-checkpoint qualification above. Keep the
    historical signature and labels used by saved calculations and the GUI.
    """
    out=[]
    labels=("Before corrector","Inside corrector","Post-scan crossover")
    for label,target in zip(labels,targets):
        candidate = detect_beam_waist(branch, target-half_window_mm, target+half_window_mm)
        if candidate is not None:
            out.append(dict(candidate, name=label, target_z_mm=float(target),
                            verified=True, precision="sampled display history"))
    return out
