"""Evaluate the impressed and induced fields of a solved threefold corrector in its beam bore."""

from __future__ import annotations

import json

import numpy as np
from scipy.constants import mu_0

from .probe_corrector_fem import impressed_h


def _sources(corners, moments, order):
    centres = corners.mean(axis=1)
    if order == 1:
        return centres, moments
    if order != 4:
        raise ValueError("Supported tetrahedral volume quadratures have one or four points")
    # Symmetric, degree-two tetrahedral quadrature with four equal weights.
    abscissas = centres[:, None, :] + (corners - centres[:, None, :]) / np.sqrt(5.0)
    return abscissas.reshape(-1, 3), np.repeat(moments / 4, 4, axis=0)


def _induced_cuda(points, sources, moments, *, winding=False, potential=False):
    import cupy as cp

    kernel = cp.RawKernel(
        r"""
    extern "C" __global__ void induced(const double *p, const double *xyz, const double *m,
            const int n, double *out, double factor, int winding) {
        const int j=blockIdx.x, t=threadIdx.x;
        const double x=p[3*j], y=p[3*j+1], z=p[3*j+2];
        double bx=0, by=0, bz=0;
        for (int i=t; i<n; i+=blockDim.x) {
            double dx=x-xyz[3*i], dy=y-xyz[3*i+1], dz=z-xyz[3*i+2];
            double mx=m[3*i], my=m[3*i+1], mz=m[3*i+2];
            double inv=1./(dx*dx+dy*dy+dz*dz), k=inv*sqrt(inv);
            if (winding==2) {
                bx+=k*(dx*mx+dy*my+dz*mz);
            } else if (winding) {
                bx+=k*(my*dz-mz*dy); by+=k*(mz*dx-mx*dz); bz+=k*(mx*dy-my*dx);
            } else {
                double dot=3*(dx*mx+dy*my+dz*mz)*inv;
                bx+=k*(dx*dot-mx); by+=k*(dy*dot-my); bz+=k*(dz*dot-mz);
            }
        }
        __shared__ double s[3][128];
        s[0][t]=bx; s[1][t]=by; s[2][t]=bz; __syncthreads();
        for (int stride=64; stride; stride/=2) {
            if (t<stride) for (int c=0; c<3; c++) s[c][t]+=s[c][t+stride];
            __syncthreads();
        }
        if (!t) for (int c=0; c<3; c++) out[3*j+c]=factor*s[c][0];
    }""",
        "induced",
    )
    xyz, mag = (cp.asarray(np.ascontiguousarray(a, dtype=float)) for a in (sources, moments))
    result = np.empty_like(points)
    # Small launches also keep individual display-GPU kernels bounded.
    for start in range(0, len(points), 512):
        pos = cp.asarray(np.ascontiguousarray(points[start : start + 512], dtype=float))
        b = cp.empty_like(pos)
        factor = (1.0 if winding else mu_0 / (4 * np.pi)) if not potential else 1 / (4 * np.pi)
        kind = 2 if potential else int(winding)
        kernel((len(b),), (128,), (pos, xyz, mag, np.int32(len(mag)), b, factor, np.int32(kind)))
        result[start : start + len(b)] = b.get()
    return result


def vacuum_field(points, solution, *, quadrature=4, backend="cuda"):
    """Return laboratory B in tesla at exterior points in module-local SI coordinates."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    xyz, moments = _sources(solution["corners_m"], solution["moments_Am2"], quadrature)
    if backend == "cuda":
        field = _induced_cuda(points, xyz, moments)
    elif backend == "cpu":
        field = np.zeros_like(points)
        for i, point in enumerate(points):
            r = point - xyz
            inv = 1 / np.einsum("ij,ij->i", r, r)
            scale = 3 * inv * np.einsum("ij,ij->i", moments, r)
            field[i] = mu_0 / (4 * np.pi) * (inv**1.5 @ (r * scale[:, None] - moments))
    else:
        raise ValueError("Unknown field-sampling backend")
    return field + _coil_b(points, solution, backend)


def _coil_b(points, solution, backend="cuda"):
    recipe = json.loads(str(solution["recipe"]))
    report = json.loads(str(solution["report"]))
    currents = dict(report["currents_A"])
    field = np.zeros_like(points)
    if backend == "cuda":
        from .transverse_coil_field import _unit_nodes

        for key, coil in recipe["sources"].items():
            if coil["kind"] == "hexapole" and currents[key]:
                nodes, dl, *_ = _unit_nodes(json.dumps(coil["coils"], sort_keys=True), coil["turns"], 2, 8)
                field += _induced_cuda(points, nodes, currents[key] * dl, winding=True)
                currents[key] = 0.0
    return field + mu_0 * impressed_h(recipe, points, currents)


class AirPotential:
    """Locate points in the actual air tetrahedra and interpolate the continuous FEM solution."""

    def __init__(self, solution):
        from scipy.spatial import cKDTree

        xyz, tet = solution["air_xyz_m"], solution["air_tet"]
        corners = xyz[tet]
        self.origin = corners[:, 0]
        self.inv = np.linalg.inv(corners[:, 1:] - corners[:, :1])
        self.tree = cKDTree(corners.mean(axis=1))
        self.u = solution["air_u_A"][tet]

    def evaluate(self, points, *, gradient=False):
        result = np.empty((len(points), 3)) if gradient else np.empty(len(points))
        for start in range(0, len(points), 512):
            p = points[start : start + 512]
            missing = np.arange(len(p))
            for count in (12, 64, 256, 1024, 4096):
                _, cand = self.tree.query(p[missing], k=min(count, len(self.u)))
                if cand.ndim == 1:
                    cand = cand[:, None]
                diff = p[missing, None, :] - self.origin[cand]
                bary = np.einsum("nki,nkij->nkj", diff, self.inv[cand])
                inside = (bary.min(axis=-1) >= -1e-8) & (bary.sum(axis=-1) <= 1 + 1e-8)
                yes = np.flatnonzero(np.any(inside, axis=1))
                k = np.argmax(inside[yes], axis=1)
                cell = cand[yes, k]
                du = self.u[cell, 1:] - self.u[cell, :1]
                if gradient:
                    value = np.einsum("nij,nj->ni", self.inv[cell], du)
                else:
                    value = self.u[cell, 0] + np.einsum("ni,ni->n", bary[yes, k], du)
                result[start + missing[yes]] = value
                missing = missing[~inside.any(axis=1)]
                if not len(missing):
                    break
            if len(missing):
                raise ValueError("No containing vacuum tetrahedron for a requested aperture sample")
        return result


def harmonic_table(solution, step_m, *, vacuum_only=False, grid_half=12, angles=36):
    """Continue the solved scalar potential harmonically into the current-free electron aperture."""
    from scipy.special import ive

    # A slightly wider sampling cylinder keeps the full 2.88 mm working disk interior.
    R = 0.00378
    z = np.linspace(0, 1.16, round(1.16 / step_m) + 1)
    xy = np.linspace(-0.0036, 0.0036, 2 * grid_half + 1)
    y, x = np.meshgrid(xy, xy, indexing="ij")
    rad = np.hypot(x, y).ravel()
    rho = np.minimum(rad, R)
    theta = np.arctan2(y, x).ravel()
    data = np.zeros((len(z), len(xy) ** 2, 3))
    if not vacuum_only:
        f = AirPotential(solution)
        phi = np.arange(angles) * 2 * np.pi / angles
        p = np.empty((len(z), angles, 3))
        p[:, :, 0] = R * np.cos(phi)
        p[:, :, 1] = R * np.sin(phi)
        p[:, :, 2] = z[:, None]
        corners = solution["corners_m"]
        end0, end1 = corners[:, :, 2].min(), corners[:, :, 2].max()
        air_z = solution["air_xyz_m"][:, 2]
        # Use only points inside the solved air mesh even with short exterior padding.
        reach = np.minimum(0.125, 0.99 * np.array([end0 - air_z.min(), air_z.max() - end1]))
        if np.min(reach) <= 0:
            raise ValueError("The air mesh must extend beyond both ends of the magnetic circuit")
        fringe = np.where(z < end0, reach[0], reach[1])
        dist = np.maximum(end0 - z, z - end1)
        weight = np.clip((fringe - dist) / np.minimum(0.035, fringe / 2), 0, 1)
        weight = weight**2 * (3 - 2 * weight)
        scalar = np.zeros((len(z), angles))
        inside = weight > 0
        scalar[inside] = weight[inside, None] * f.evaluate(p[inside].reshape(-1, 3)).reshape(-1, angles)
        far = weight < 1
        source, moments = _sources(corners, solution["moments_Am2"], 1)
        outer = _induced_cuda(p[far].reshape(-1, 3), source, moments, potential=True)[:, 0].reshape(
            -1, angles
        )
        scalar[far] += (1 - weight[far, None]) * outer
        # Reflect the two distant endpoints before the periodic axial Fourier transform.
        ext = np.r_[scalar, scalar[-2:0:-1]]
        nz = len(ext)
        spec = np.fft.fft2(ext) / angles
        kz = 2 * np.pi * np.fft.fftfreq(nz, d=z[1] - z[0])
        ak = abs(kz)
        # Positive angular modes suffice for a real scalar; retain only physical C3 harmonics.
        for mode in range(0, angles // 2 + 1, 3):
            harmonic = spec[:, mode]
            if mode == angles // 2:
                # The undetermined Nyquist sine has no independent angular sample.
                continue
            for rr in np.unique(rho):
                sel = rho == rr
                arg = ak[1:] * rr
                outer_arg = ak[1:] * R
                den = ive(mode, outer_arg)
                expo = np.exp(arg - outer_arg)
                ratio = np.r_[(rr / R) ** mode, ive(mode, arg) * expo / den]
                dr = np.r_[
                    mode / R * (rr / R) ** (mode - 1) if mode else 0.0,
                    0.5 * ak[1:] * (ive(abs(mode - 1), arg) + ive(mode + 1, arg)) * expo / den,
                ]
                factors = [dr, 1j * mode * ratio / max(rr, 1e-20), 1j * kz * ratio]
                polar = [np.fft.ifft(harmonic * a)[: len(z)] for a in factors]
                phase = np.exp(1j * mode * theta[sel]) * (2 if mode else 1)
                br, bt, bz = [-mu_0 * np.real(v[:, None] * phase) for v in polar]
                data[:, sel, 0] += br * np.cos(theta[sel]) - bt * np.sin(theta[sel])
                data[:, sel, 1] += br * np.sin(theta[sel]) + bt * np.cos(theta[sel])
                data[:, sel, 2] += bz
    for start in range(0, len(z), 32):
        zs = z[start : start + 32]
        xx, yy, zz = np.broadcast_arrays(x.ravel()[None, :], y.ravel()[None, :], zs[:, None])
        pts = np.stack([xx, yy, zz], axis=-1)
        data[start : start + len(zs)] += _coil_b(pts.reshape(-1, 3), solution).reshape(pts.shape)
    return z, xy, data.reshape(len(z), len(xy), len(xy), 3)
