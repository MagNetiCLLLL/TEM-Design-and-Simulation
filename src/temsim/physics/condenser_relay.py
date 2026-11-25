"""Conjugate planes of the FEI gun and the C1/C2 source relay, in SI units."""

import numpy as np
from scipy.signal import find_peaks

from temsim.physics.fei_tip_transport import REST_EV, field_at, transport


def _angular_matrix(xy, step):
    return ((xy[::2].astype(float) - xy[1::2]) / (2 * step)).transpose(1, 2, 0)


def _image_planes(z, response):
    """Find an angular-response null in both transverse directions, including lens rotation."""
    norm = np.linalg.norm(response, axis=(1, 2))
    for i in find_peaks(-norm)[0]:
        if not np.all(np.isfinite(norm[i - 1 : i + 2])):
            continue
        before, after = response[i - 1], response[i + 1]
        # A line focus or an extremum in just one projected ray is not a point image.
        if np.any(np.sum(before * after, axis=0) >= 0):
            continue
        best = None
        for j in (i - 1, i):
            change = response[j + 1] - response[j]
            length = np.sum(change * change)
            if not length:
                continue
            t = float(np.clip(-np.sum(response[j] * change) / length, 0, 1))
            residual = np.linalg.svd(response[j] + t * change, compute_uv=False)
            if best is None or residual[0] < best[0][0]:
                best = residual, j, t
        if best is None:
            continue
        residual, j, t = best
        # Resolve the two-axis null to better than one observation interval.
        if residual[0] >= np.linalg.svd(after - before, compute_uv=False)[-1] / 2:
            continue
        yield j, t, z[j] + t * (z[j + 1] - z[j]), residual


def gun_object_and_fan(electric, magnetic, stops):
    """Locate the real gun image of the emitting tip, then perturb that object in x and y."""
    delta, angle = 1e-7, 1e-6
    start_energy = 0.2
    p = np.sqrt(start_energy / REST_EV * (2 + start_energy / REST_EV))
    signs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    tip = np.zeros((4, 6))
    tip[:, 3:5] = signs * p * np.sin(angle)
    tip[:, 5] = p * np.cos(angle)
    gun_z = np.linspace(0, 0.040, 4001)
    _, _, xy = transport(tip, electric, magnetic, stops, gun_z, 4, 0.5)
    image = next(_image_planes(gun_z, _angular_matrix(xy, angle)), None)
    if image is None:
        return None, np.empty((0, 7))
    _, _, z, _ = image
    fan = np.zeros((8, 6))
    fan[:, 2] = z
    fan[:4, :2] = signs * delta
    for j, ray in enumerate(fan):
        energy = start_energy + field_at(*ray[:3], electric)[0]
        p = np.sqrt(energy / REST_EV * (2 + energy / REST_EV))
        ray[5] = p
        if j >= 4:
            ray[3:5] = p * np.sin(angle) * signs[j - 4]
            ray[5] = p * np.cos(angle)
    hits, states, _ = transport(
        fan, electric, magnetic, stops, np.array([0.45]), 0, 0.5
    )
    return float(z), states if np.all(hits == -1) else np.empty((0, 7))


def source_images(z, fan):
    """Measure the point image, its two principal magnifications, and the angular residual."""
    if len(fan) != 8:
        return []
    a = _angular_matrix(fan[:4], 1e-7)
    b = _angular_matrix(fan[4:], 1e-6)
    result = []
    for j, t, position, residual in _image_planes(z, b):
        magnification = np.linalg.svd(a[j] + t * (a[j + 1] - a[j]), compute_uv=False)
        result.append(
            {
                "z_mm": float(1000 * position),
                "principal_magnifications": magnification.tolist(),
                "residual_nm_per_urad": (residual * 1000).tolist(),
            }
        )
    return result


def beam_widths(z, paths, weights):
    """Return simultaneous x/y/r RMS and stationary planes with clipping-safe derivatives."""
    width = np.full((len(z), 3), np.nan)
    slope = np.full((len(z) - 1, 3), np.nan)
    for i in range(len(z)):
        here = paths[:, i]
        live = np.isfinite(here).all(axis=1)
        if not np.any(live):
            continue
        xy = here[live].astype(float)
        w = weights[live]
        xy -= np.average(xy, axis=0, weights=w)
        var = np.average(xy * xy, axis=0, weights=w)
        width[i] = np.sqrt([var[0], var[1], var.sum()])
        if i == len(z) - 1:
            continue
        nxt = paths[:, i + 1]
        common = live & np.isfinite(nxt).all(axis=1)
        if not np.any(common):
            continue
        left = here[common].astype(float)
        right = nxt[common].astype(float)
        w = weights[common]
        left -= np.average(left, axis=0, weights=w)
        right -= np.average(right, axis=0, weights=w)
        deriv = np.average((right - left) * (right + left), axis=0, weights=w)
        slope[i] = np.r_[deriv, deriv.sum()] / (z[i + 1] - z[i])
    mid = (z[1:] + z[:-1]) / 2
    waists = {}
    for k, key in enumerate(("x", "y", "r")):
        d = slope[:, k]
        j = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
        roots = mid[j] - d[j] * (mid[j + 1] - mid[j]) / (d[j + 1] - d[j])
        waists[key] = (roots * 1000).tolist()
    return width, waists
