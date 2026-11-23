"""Find the single-image condenser branch from the tip in the accelerating electrostatic field."""

import numpy as np
from numba import njit
from scipy.constants import c, e, m_e

from column_current import FACTOR, WORK
from condenser_source_fields import Fields, OUT, write


REST = m_e * c * c / e
PC = np.sqrt(300000.0 * (300000.0 + 2 * REST))


def axial_nodes(step, end=1.07):
    with np.load(WORK / "column/electric.npz") as f:
        faces = f["z"]
    return np.unique(
        np.r_[
            0.0,
            np.geomspace(1e-10, 1e-4, 250),
            faces[(faces > 0) & (faces < 0.45)],
            np.arange(0.0, end, step),
            0.45,
            0.55,
            0.75,
            0.83,
            1.0,
            end,
        ]
    )


def electrostatic(z, energy):
    mid = (z[1:] + z[:-1]) / 2
    with np.load(WORK / "column/electric.npz") as f:
        zz, r, v = f["z"], f["r"], f["voltage"]
    axis = np.interp(mid, zz, v[0])
    radial = np.interp(mid, zz, (v[1] - v[0]) / r[1] ** 2)
    pc = np.sqrt((energy + axis) * (energy + axis + 2 * REST))
    return PC / pc, 2 * (energy + axis + REST) / (pc * PC) * radial


@njit(cache=True)
def transfer(z, invp, ef, b, want_trace):
    # Canonical state in the Larmor frame is (x, (p/p_ref) dx/dz).
    # Point source condition x(0)=0. The second column gives the real imaging zeros.
    a, b1, g, d = 1.0, 0.0, 0.0, 1.0
    roots = np.zeros(5, np.int64)
    first_z = 0.0
    gain = 0.0
    gap = 0.0
    last = 0.0
    trace = np.zeros((len(z) if want_trace else 1, 4))
    trace[0] = np.array([a, b1, g, d])
    for i in range(len(z) - 1):
        h = z[i + 1] - z[i]
        Q = ef[i] - (FACTOR * b[i]) ** 2 * invp[i]
        t2 = Q * invp[i] * h * h
        if t2 < -1e-8:
            angle = np.sqrt(-t2)
            co = np.cos(angle)
            sh = h * np.sin(angle) / angle
        elif t2 > 1e-8:
            angle = np.sqrt(t2)
            co = np.cosh(angle)
            sh = h * np.sinh(angle) / angle
        else:
            co = 1 + t2 / 2
            sh = h * (1 + t2 / 6)
        an = co * a + sh * invp[i] * g
        bn = co * b1 + sh * invp[i] * d
        gn = sh * Q * a + co * g
        dn = sh * Q * b1 + co * d
        if b1 * bn < 0:
            frac = b1 / (b1 - bn)
            at = z[i] + h * frac
            if at >= 0.4:
                bin = 0 if at < 0.55 else 1 if at < 0.75 else 2 if at < 0.83 else 3 if at < 1.0 else 4
                roots[bin] += 1
                last = at
                if bin == 0:
                    first_z = at
                    gain = abs(a + (an - a) * frac)
                if bin == 2:
                    gap = at
        a, b1, g, d = an, bn, gn, dn
        if want_trace:
            trace[i + 1] = np.array([a, b1, g, d])
    return roots, first_z, gain, gap, trace


def screen(fields):
    z = axial_nodes(0.0004)
    mid = (z[1:] + z[:-1]) / 2
    ep, ef = electrostatic(z, 0.303)
    units = fields.round_axis(mid)
    common = fields.current(0, 0) @ units + fields.objective(mid)[:, 0]
    common += np.where(
        mid < 0.45,
        fields.far_corrector(mid)[:, 0, 2],
        np.interp(mid, fields.zjoint, fields.bjoint[:, 2]),
    )
    u1, u2 = units[:4].sum(axis=0), units[4:8].sum(axis=0)
    matches = []
    for i1 in np.arange(7.0, 100.0, 1.0):
        for i2 in np.arange(4.0, 60.0, 0.5):
            result = transfer(z, ep, ef, common - i1 * u1 + i2 * u2, False)
            counts, where, gain, gap, _ = result
            if tuple(counts) == (1, 0, 1, 0, 0) and 0.45 < where < 0.55 and gain < 1:
                score = ((where - 0.525) / 0.022) ** 2 + ((gap - 0.792) / 0.016) ** 2
                matches.append((score, -i1, i2, where, gain, gap))
    if not matches:
        raise ValueError("No single-image and single-gap branch in the declared series-current search")
    matches.sort()
    print("found", len(matches), "best", matches[:4], flush=True)
    row = dict(
        series_A=list(matches[0][1:3]),
        image_mm=matches[0][3] * 1000,
        mag=matches[0][4],
        gap_mm=matches[0][5] * 1000,
        candidates=len(matches),
    )
    write(OUT / "proposal.json", row)
    return row


if __name__ == "__main__":
    screen(Fields())
