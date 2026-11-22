"""Match a magnetized finite-source illumination and image the CESCOR pupil planes."""

import json

import numpy as np
from numba import njit, prange

from column_current import FACTOR, POINT, fields, pencil_fields
from column import read_gun
from relay_transfer import FOIL, HP2, HP1, REF, RADIUS, START, SURFACE


GROUPS = (slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 15), slice(15, 18), slice(18, 21))


@njit(cache=True)
def predict(x, base, basis, h, rad, region, ix, kbias, kbasis, emission):
    bz = base + basis @ x
    kg, ks = FACTOR * (kbias + kbasis @ x)
    w = 1.0 + 0j
    # Laboratory t = exp(i theta) (w' + i k w). The beam is born in a field.
    v = emission[0] + 1j * (emission[1] - kg)
    bp = 0.0
    dp = 0.0
    hp = 0.0
    ap = 0.0
    pupil = 0.0
    fa = 0.0
    fc = 0.0
    fb = 0.0
    fd = 0.0
    foil_shear = 0.0
    foil_phase = 1.0 + 0j
    peak = 0.0
    counts = np.zeros(3, np.int64)
    for i in range(len(h)):
        if i == ix[0] or i == ix[1]:
            if abs(w) > pupil:
                pupil = abs(w)
                foil_phase = w
        if i == ix[1]:
            fa = 1.0
            fc = 0.0
            fb = 0.0
            fd = 1.0
            foil_shear = (v / w).real
        if i == ix[2]:
            bp = 0.0
            dp = 1.0
        if i == ix[3]:
            hp = bp
            bp = 0.0
            dp = 1.0
        if i == ix[4]:
            ap = bp
        k = FACTOR * bz[i]
        c = np.cos(k * h[i])
        s = np.sin(k * h[i]) / k if abs(k) > 1e-13 else h[i]
        wn, vn = c * w + s * v, -k * k * s * w + c * v
        if region[i] >= 0 and (w.conjugate() * v).real < 0 and (wn.conjugate() * vn).real >= 0:
            counts[region[i]] += 1
        if i >= ix[1]:
            peak = max(peak, max(abs(w), abs(wn)) / rad[i])
        w, v = wn, vn
        bp, dp = c * bp + s * dp, -k * k * s * bp + c * dp
        if i >= ix[1]:
            fa, fc = c * fa + s * fc, -k * k * s * fa + c * fc
            fb, fd = c * fb + s * fd, -k * k * s * fb + c * fd
    slope = v + 1j * ks * w
    waist_error = RADIUS * (w * slope.conjugate()).real / (pupil * max(abs(slope), 1e-16))
    angle = np.arctan(RADIUS * abs(slope) / pupil)
    return np.array(
        [
            hp,
            ap,
            angle,
            waist_error,
            pupil * emission[2] / RADIUS,
            peak * RADIUS / pupil,
            counts[0],
            counts[1],
            counts[2],
            RADIUS * abs(w) / pupil,
            fb,
            fa + fb * foil_shear,
            RADIUS * (w / foil_phase).real,
            RADIUS * (slope / foil_phase).real,
        ]
    )


@njit(cache=True, parallel=True)
def batch(xs, base, basis, h, rad, region, ix, kbias, kbasis, emission):
    rows = np.empty((len(xs), 14))
    for i in prange(len(xs)):
        rows[i] = predict(xs[i], base, basis, h, rad, region, ix, kbias, kbasis, emission)
    return rows


def data(table, pupil, step, directions=None, objective_ni=None):
    q = np.array(REF["currents_A"])
    planes = [*FOIL, HP2, HP1, pupil, SURFACE]
    m, t, b, c = fields(step, planes, REF["objective_At"] if objective_ni is None else objective_ni, table)
    f = pencil_fields(m, t, b, c)
    z = m.nodes[1::2]
    endpoints = m.z.searchsorted([START, SURFACE]) * 2
    ctr = len(t["xy_m"]) // 2
    axis = np.interp(m.nodes[endpoints] - START, t["z_local_m"], t["B_zyx_T"][:, ctr, ctr, 2])
    kbias = b[endpoints, 0] + axis
    kbasis = np.column_stack([c[g, :, 0][:, endpoints].T @ q[g] for g in GROUPS])
    base = f[:, 0].copy()
    basis = np.column_stack([c[g, 1::2, 0].T @ q[g] for g in GROUPS])
    if directions is not None:
        kbasis = c[:, endpoints, 0].T @ directions
        basis = c[:, 1::2, 0].T @ directions
    wind = json.loads(str(t["report"]))["currents_A"]
    region = np.select(
        [
            (z > m.intervals[0]) & (z < m.intervals[1]),
            (z > m.intervals[1]) & (z < m.intervals[2]),
            (z > m.intervals[3]) & (z < SURFACE - 0.00025),
        ],
        [0, 1, 2],
        default=-1,
    )
    p, weights, *_ = read_gun(POINT)
    pos = p[:, 0] + 1j * p[:, 1]
    slope = p[:, 2] + 1j * p[:, 3]
    slope_per_m = np.vdot(weights * pos, slope) / np.vdot(weights * pos, pos)
    emission = np.array([slope_per_m.real, slope_per_m.imag, np.max(abs(pos))])
    args = (
        np.ascontiguousarray(base),
        np.ascontiguousarray(basis),
        np.diff(m.z),
        m.radius,
        region,
        m.z.searchsorted(planes[:-1]),
        kbias,
        kbasis,
        emission,
    )
    return q, wind, args
