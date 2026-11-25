"""Stationary relativistic Lorentz transport in the saved three-layer FEM bore field (SI)."""

from math import sqrt

import numpy as np
from numba import njit, prange
from scipy.interpolate import CubicSpline

from temsim.physics.fei_tip_transport import REST_EV


@njit
def derivative(p, node, energy, phi0, field, scales, electric, r2, factor):
    x, y, tx, ty = p
    s = x * x + y * y
    radius = sqrt(s)
    bx = by = bz = 0.0
    remaining = 1.0
    for layer in range(len(scales)):
        scale = scales[layer]
        mix = 1.0
        if layer + 1 < len(scales):
            t = min(1.0, max(0.0, (scale - radius) / (0.16 * scale)))
            mix = t * t * (3.0 - 2.0 * t)
        weight = remaining * mix
        remaining *= 1.0 - mix
        if weight > 0.0:
            u, v = x / scale, y / scale
            term = 0
            for degree in range(6):
                for i in range(degree + 1):
                    basis = weight * u**i * v ** (degree - i)
                    bx += basis * field[node, layer, term, 0]
                    by += basis * field[node, layer, term, 1]
                    bz += basis * field[node, layer, term, 2]
                    term += 1
        if remaining == 0.0:
            break
    j = min(len(r2) - 2, max(0, np.searchsorted(r2, s, side="right") - 1))
    v0, dv, ez0, dez = electric[node, j]
    ds = s - r2[j]
    kin = energy + (v0 + ds * dv - phi0)
    pc = sqrt(kin * (kin + 2.0 * REST_EV))
    norm2 = 1.0 + tx * tx + ty * ty
    magnetic = factor * sqrt(norm2) / pc
    electrostatic = -(kin + REST_EV) * norm2 / (pc * pc)
    ez = ez0 + ds * dez
    return (
        tx,
        ty,
        magnetic * (ty * bz - (1.0 + tx * tx) * by + tx * ty * bx)
        + electrostatic * (-2.0 * x * dv - tx * ez),
        magnetic * ((1.0 + ty * ty) * bx - tx * bz - tx * ty * by)
        + electrostatic * (-2.0 * y * dv - ty * ez),
    )


@njit(inline="always")
def advance(p, k, h):
    return (p[0] + h * k[0], p[1] + h * k[1], p[2] + h * k[2], p[3] + h * k[3])


@njit(parallel=True, nogil=True, cache=True)
def trace_column(
    initial, energies, z, aperture, observations, fields, scales, ep, r2, factor
):
    paths = np.full((len(initial), len(observations), 2), np.nan, np.float32)
    endpoint = initial.copy()
    loss = np.full(len(initial), -1)
    for ray in prange(len(initial)):
        x, y, tx, ty = initial[ray]
        p = (x, y, tx, ty)
        s = x * x + y * y
        j = np.searchsorted(r2, s, side="right") - 1
        if j < 0 or j >= len(r2) - 1:
            loss[ray] = -2
            continue
        phi0 = ep[0, j, 0] + (s - r2[j]) * ep[0, j, 1]
        energy = energies[ray]
        obs = 0
        for i in range(len(z) - 1):
            if i == observations[obs]:
                paths[ray, obs, 0], paths[ray, obs, 1] = p[0], p[1]
                obs += 1
            h = z[i + 1] - z[i]
            a = derivative(p, 2 * i, energy, phi0, fields, scales, ep, r2, factor)
            p2 = advance(p, a, h / 2.0)
            b = derivative(p2, 2 * i + 1, energy, phi0, fields, scales, ep, r2, factor)
            p3 = advance(p, b, h / 2.0)
            c = derivative(p3, 2 * i + 1, energy, phi0, fields, scales, ep, r2, factor)
            p4 = advance(p, c, h)
            d = derivative(p4, 2 * i + 2, energy, phi0, fields, scales, ep, r2, factor)
            nxt = advance(
                advance(advance(advance(p, a, h / 6.0), b, h / 3.0), c, h / 3.0),
                d,
                h / 6.0,
            )
            ends = max(p[0] ** 2 + p[1] ** 2, nxt[0] ** 2 + nxt[1] ** 2)
            swept = max(
                ends,
                p2[0] ** 2 + p2[1] ** 2,
                p3[0] ** 2 + p3[1] ** 2,
                p4[0] ** 2 + p4[1] ** 2,
            )
            if not np.isfinite(sum(nxt)) or swept >= r2[-1]:
                loss[ray] = -2
                break
            if ends > aperture[i] ** 2 and aperture[i] <= scales[-1]:
                loss[ray] = i
                break
            if swept > scales[-1] ** 2:
                loss[ray] = -2
                break
            if ends > aperture[i] ** 2:
                loss[ray] = i
                break
            p = nxt
        if loss[ray] == -1:
            endpoint[ray] = p
            paths[ray, -1, 0], paths[ray, -1, 1] = p[0], p[1]
    return endpoint, loss, paths


def propagate(mesh, table, gun_exit):
    z = mesh["z"]
    nodes = np.empty(2 * len(z) - 1)
    nodes[::2], nodes[1::2] = z, (z[:-1] + z[1:]) / 2
    if int(table["degree"]) != 5 or table["layers"].shape[2:] != (21, 3):
        raise ValueError("Unsupported magnetic field interpolation order")
    field = CubicSpline(table["z_m"], table["layers"].swapaxes(0, 1), axis=0)(nodes)
    r = mesh["electric_r"]
    r = r[: np.searchsorted(r, 0.004) + 2]
    r2 = r * r
    voltage = mesh["electric_v"][: len(r)] - 300000.0
    j = mesh["electric_index"]
    bottom = voltage[:, j].T
    delta = voltage[:, j + 1].T - bottom
    phi = bottom + delta * mesh["electric_fraction"][:, None]
    ez = -delta / mesh["electric_dz"][:, None]
    ep = np.stack(
        (
            phi[:, :-1],
            np.diff(phi) / np.diff(r2),
            ez[:, :-1],
            np.diff(ez) / np.diff(r2),
        ),
        axis=-1,
    )
    u = gun_exit[:, 3:6]
    u2 = (u * u).sum(axis=1)
    energy = REST_EV * u2 / (np.sqrt(1.0 + u2) + 1.0)
    initial = np.column_stack((gun_exit[:, :2], u[:, :2] / u[:, 2, None]))
    factor = (
        -2.0
        * float(mesh["reference_factor"])
        * np.sqrt(300000.0 * (300000.0 + 2.0 * REST_EV))
    )
    obs = np.unique(np.r_[0, mesh["observations"], len(z) - 1])
    endpoint, loss, paths = trace_column(
        initial,
        energy,
        z,
        mesh["radius"],
        obs,
        field,
        table["scales"],
        np.ascontiguousarray(ep),
        r2,
        factor,
    )
    if np.any(loss < -1):
        raise ValueError(
            "A ray left the validated field volume before a physical intercept"
        )
    return endpoint, loss, paths, z[obs]
