"""Propagate original gun-exit packets, retaining physical aperture losses and absolute weights."""

import numpy as np
import cupy as cp
from scipy.constants import c, e, m_e

from column import FACTOR, HERE, POINT, read_gun


class Tracker:
    def __init__(self, mesh, *, source=None):
        self.mesh = mesh
        r, v, j, l, dz = mesh.electric
        nr = min(len(r), np.searchsorted(r, 0.004) + 2)
        r = r[:nr]
        vs = v[:nr] - 300000.0
        parts = []
        for inds in np.array_split(np.arange(len(j)), max(1, len(j) // 2048)):
            a, b = vs[:, j[inds]].T, vs[:, j[inds] + 1].T
            bottom = a + (b - a) * l[inds, None]
            ez = -(b - a) / dz[inds, None]
            parts.append(
                np.stack(
                    [
                        bottom[:, :-1],
                        np.diff(bottom, axis=1) / np.diff(r * r),
                        ez[:, :-1],
                        np.diff(ez, axis=1) / np.diff(r * r),
                    ],
                    axis=-1,
                )
            )
        coeff = np.concatenate(parts)
        self.e0 = coeff[0]
        self.r2 = r * r
        self.er, self.ep = cp.asarray(self.r2), cp.asarray(np.ascontiguousarray(coeff))
        self.h, self.rad = cp.asarray(np.diff(mesh.z)), cp.asarray(mesh.radius**2)
        self.tables = ()
        self._trace_kernel = cp.RawKernel(
            (HERE / "lorentz.cu").read_text(), "trace", options=("--std=c++11", "--fmad=false")
        )
        self.rest = m_e * c * c / e
        self.pc = np.sqrt(300000.0 * (300000.0 + 2 * self.rest))
        self.input, self.weights, self.ids, self.de, self.current = read_gun(POINT) if source is None else source

    def kernel(self, grid, block, args):
        return self._trace_kernel(grid, block, (*args, *self.tables))

    def run(self, field, count=None, *, keep_history=True, record_probe=False, surface_index=None):
        data = self.input[:count]
        energy = self.de[:count] + 300000.0
        n = len(data)
        assert n >= 1000
        r2 = data[:, 0] ** 2 + data[:, 1] ** 2
        k = np.clip(np.searchsorted(self.r2, r2, side="right") - 1, 0, len(self.r2) - 2)
        p0 = self.e0[k, 0] + (r2 - self.r2[k]) * self.e0[k, 1]
        ph = cp.asarray(data.copy())
        en, phi0 = cp.asarray(energy), cp.asarray(p0)
        loss = cp.full(n, -1, dtype=cp.int32)
        peak = cp.zeros(n)
        margin = cp.full(n, cp.inf)
        stops = self.mesh.saved
        fd = cp.asarray(np.ascontiguousarray(field))
        result = []
        previous = 0
        moments = []
        w = cp.asarray(self.weights[:n]) if record_probe else None
        probes = []
        if record_probe and not np.all(self.weights[:n] == self.weights[0]):
            raise ValueError("Through-focus partition requires equal original weights")
        for stop in stops:
            for i in range(previous, stop, 160):
                self.kernel(
                    ((n + 127) // 128,),
                    (128,),
                    (
                        ph,
                        en,
                        phi0,
                        fd,
                        self.ep,
                        self.er,
                        np.int32(len(self.r2)),
                        self.h,
                        self.rad,
                        loss,
                        peak,
                        margin,
                        np.int32(i),
                        np.int32(min(i + 160, stop)),
                        np.int32(n),
                        self.rest,
                        FACTOR,
                        self.pc,
                    ),
                )
            if keep_history or stop == stops[-1]:
                result.append(ph.get())
            if surface_index is not None and stop == surface_index:
                self.surface_phase, self.surface_loss = ph.get(), loss.get()
            if record_probe:
                weights = w * (loss == -1)
                total = weights.sum()
                xy = ph[:, :2] - cp.sum(weights[:, None] * ph[:, :2], axis=0) / total
                txy = ph[:, 2:4] - cp.sum(weights[:, None] * ph[:, 2:4], axis=0) / total
                widths = cp.sqrt(cp.sum(weights[:, None] * xy**2, axis=0) / total)
                cov = cp.sum(weights[:, None] * xy * txy, axis=0) / total
                moments.append(cp.stack((total, widths[0], widths[1], cov[0], cov[1])))
                alive = loss == -1
                live_count = int(cp.count_nonzero(alive))
                radial = cp.sum(xy**2, axis=1)
                radial = cp.where(alive, radial, cp.inf)
                idx = (19 * live_count + 19) // 20 - 1
                d95 = 2e9 * cp.sqrt(cp.partition(radial, idx)[idx])
                core = (
                    cp.sum(w * (alive & (cp.sum(ph[:, :2] ** 2, axis=1) <= 50e-12**2))) * self.current * 1e9
                )
                probes.append(cp.stack((d95, 1e9 * cp.sqrt(cp.sum(widths**2)), core)))
            previous = stop
        loss = loss.get()
        if np.any(loss < -1):
            raise ValueError(
                f"Unresolved boundary or out-of-grid ray: {np.unique(loss[loss < -1],return_counts=True)}"
            )
        self.moments = cp.asnumpy(cp.stack(moments)) if moments else None
        self.probes = cp.asnumpy(cp.stack(probes)) if probes else None
        return np.array(result), loss, np.sqrt(peak.get()), margin.get()
