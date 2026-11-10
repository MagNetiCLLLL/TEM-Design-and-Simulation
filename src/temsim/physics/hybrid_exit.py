"""Gauge-covariant local diffraction -> positive weighted exit particles.

Gaussian analysis windows form a discrete partition of unity. Their local
spectra retain position/momentum correlations and exactly partition the input
current. This is a finite-resolution, incoherent handoff, not an exact quantum
phase-space distribution. The unwindowed diffraction is retained separately.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.constants import e, h, hbar

from temsim.physics.hybrid_semiclassical import frame_centres, momentum
from temsim.physics.wave_device import to_host
from temsim.immutable_json import freeze_json


def _freeze(values, dtype=None):
    a = np.asarray(values, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class HybridExitParticles:
    position_m: np.ndarray
    direction: np.ndarray
    energy_ev: np.ndarray
    weight: np.ndarray
    mode_index: np.ndarray
    mode_ids: tuple[str, ...]
    record: dict

    def __post_init__(self):
        object.__setattr__(self, "record", freeze_json(self.record))
        object.__setattr__(self, "mode_ids", tuple(self.mode_ids))
        n = len(self.weight)
        for name, shape, dtype in (("position_m", (n, 3), float), ("direction", (n, 3), float),
                ("energy_ev", (n,), float), ("weight", (n,), float), ("mode_index", (n,), np.int64)):
            value = _freeze(getattr(self, name), dtype)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid hybrid exit {name}")
            object.__setattr__(self, name, value)
        if (np.any(self.weight < 0) or np.any(self.energy_ev <= 0)
                or np.any(self.direction[:, 2] <= 0)
                or not np.allclose(np.sum(self.direction**2, axis=1), 1., rtol=0., atol=1e-12)
                or np.any(self.mode_index < 0) or np.any(self.mode_index >= len(self.mode_ids))):
            raise ValueError("Invalid forward particle probability, mode or energy")


def _line_phase(snapshot, centre, points, z_m, order):
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
    if type(snapshot) is ElectromagneticPotentialSnapshot and not snapshot.contributions:
        # An explicitly empty source inventory has only its exact polynomial
        # gauge. This is not a zero-field inference outside a provider domain.
        gauge = snapshot.gauge
        if not np.any(gauge.gradient_t_m) and not np.any(gauge.hessian_t):
            return 0.
        delta = points-centre
        gradient = (gauge.gradient_t_m+gauge.hessian_t@np.r_[centre, z_m])[:2]
        return e/hbar*(delta@gradient+.5*np.einsum("ni,ij,nj->n", delta, gauge.hessian_t[:2, :2], delta))
    delta = points-centre
    nodes, weights = np.polynomial.legendre.leggauss(order)
    integral = np.zeros(len(points))
    for t, w in zip((nodes+1)/2, weights/2):
        xy = centre+t*delta
        potential = snapshot.sample(np.column_stack((xy, np.full(len(xy), z_m))))["A_t_m"][:, :2]
        integral += w*np.einsum("ij,ij->i", potential, delta)
    # q_e=-e: exp(-i q_e integral(A.dl)/hbar) cancels gauge phase.
    return e*integral/hbar


def _partition_factors(plane, centres, sigma, *, xp):
    """Exact separable Gaussian partition on the already admitted orthogonal grid.

    No window, spatial tail or Fourier bin is removed. Row-major centre order
    is checked before using one-dimensional factors in the two-dimensional FFT.
    """
    index = np.rint((centres-plane.origin_m)@np.linalg.inv(plane.basis_m).T).astype(np.int64)
    cx, cy = np.unique(index[:, 0]), np.unique(index[:, 1])
    if (len(cx)*len(cy) != len(centres) or not np.array_equal(index[:, 0], np.tile(cx, len(cy)))
            or not np.array_equal(index[:, 1], np.repeat(cy, len(cx)))):
        raise ValueError("Gaussian windows do not form the declared Cartesian frame")
    lengths = np.linalg.norm(plane.basis_m, axis=0)
    ny, nx = plane.amplitude.shape
    factors = []
    for count, centres_axis, spacing in ((nx, cx, lengths[0]), (ny, cy, lengths[1])):
        delta = (xp.arange(count)[None, :]-count//2-xp.asarray(centres_axis)[:, None])*spacing
        value = xp.exp(-delta**2/(4*sigma*sigma))
        denominator = xp.sum(value**2, axis=0)
        if bool(xp.any(denominator <= 0)):
            raise ValueError("Exit analysis windows do not cover the executed wave")
        factors.append(value/xp.sqrt(denominator))
    return factors[0], factors[1]


def local_diffraction(mode, snapshot, z_m, numerics, *, xp, cancelled=lambda: False, consume_window=None, progress_callback=None):
    plane = mode.plane
    points, frequency, centres, sigma = frame_centres(plane, numerics)
    pixels = plane.amplitude.size
    # Production sampling streams one window; array tests may request the joint table.
    required = (8*len(centres)*pixels if consume_window is None else 0)+256*pixels
    if required > numerics.maximum_working_bytes:
        raise MemoryError("Joint position/diffraction distribution exceeds its memory budget")
    if cancelled():
        raise InterruptedError("Exit window preparation cancelled")
    wx, wy = _partition_factors(plane, centres, sigma, xp=xp)
    full = xp.asarray(plane.full_amplitude(h/float(momentum(mode.energy_kev*1000))))
    probability = np.empty((len(centres), pixels)) if consume_window is None else None
    marginal = np.zeros(pixels)
    worst_phase_error, squared_phase_error = 0., 0.
    for index, centre in enumerate(centres):
        if cancelled():
            raise InterruptedError("Local diffraction calculation cancelled")
        iy, ix = divmod(index, len(wx))
        window = wy[iy, :, None]*wx[ix, None, :]
        phase = _line_phase(snapshot, centre, points, z_m, 5)
        coarse = _line_phase(snapshot, centre, points, z_m, 3)
        # Ignore no region silently: bound the phase difference weighted by
        # this window and the actual executed wave amplitude.
        error = float(xp.linalg.norm(full*window*xp.asarray(phase-coarse).reshape(
            full.shape if np.ndim(phase) else ())))
        worst_phase_error = max(worst_phase_error, error)
        squared_phase_error += error*error
        if error > 1e-4:
            raise ValueError("Exit magnetic line-integral phase is unresolved; refine the wave/window")
        local = full*window*xp.exp(1j*xp.asarray(phase).reshape(full.shape if np.ndim(phase) else ()))
        spectrum = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(local), norm="ortho"))
        row = to_host(abs(spectrum)**2).ravel()
        marginal += row
        if probability is not None:
            probability[index] = row
        if consume_window is not None:
            consume_window(index, row)
        if progress_callback is not None and (index == 0 or (index+1) % 64 == 0 or index+1 == len(centres)):
            progress_callback(index+1, len(centres), f"Local diffraction: {mode.mode_id}")
    total = math.fsum(marginal)
    if not math.isclose(total, 1., rel_tol=0., abs_tol=2e-11):
        raise ValueError("Local diffraction did not partition unit mode probability")
    if math.sqrt(squared_phase_error) > 1e-4:
        raise ValueError("Combined exit magnetic phase is unresolved; refine the wave/window")
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(full.shape[0])),
                         np.fft.fftshift(np.fft.fftfreq(full.shape[1])), indexing="ij")
    edge = (np.maximum(abs(fx), abs(fy)) > .4).ravel()
    tail = float(marginal[edge].sum())
    if tail > 1e-10:
        raise ValueError("Local mechanical diffraction reaches the sampling boundary; refine the wave grid")
    raw = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(full), norm="ortho"))
    return centres, frequency, probability, {
        "raw_canonical_diffraction": to_host(abs(raw)**2)*mode.weight_per_reference_electron,
        "transport_diffraction": marginal.reshape(full.shape)*mode.weight_per_reference_electron,
        "frequency_xy_m1": frequency.reshape((*full.shape, 2)),
        "window_sigma_m": sigma,
        "window_stride_xy_m": (np.linalg.norm(plane.basis_m, axis=0)*numerics.frame_stride).tolist(),
        "window_count": len(centres),
        "window_execution": "Exact separable Gaussian partition; every window and Fourier bin retained",
        "window_nominal_angular_rms_rad": 1/(2*float(momentum(mode.energy_kev*1000)/hbar)*sigma),
        "magnetic_phase_quadrature_error_l2": worst_phase_error,
        "combined_magnetic_phase_quadrature_error_l2": math.sqrt(squared_phase_error),
        "outer_frequency_probability": tail,
        "scope": "Gauge-covariant Gaussian-window spectrogram; finite spatial/angular resolution; no downstream coherence",
        "raw_spectrum_scope": "Canonical Fourier spectrum in the captured gauge; not mechanical angles in magnetic fields",
    }


def sample_exit(checkpoint, snapshot, count, numerics, *, xp, seed=0, cancelled=lambda: False,
                window_sigma_m=None, progress_callback=None):
    numerics.validate()
    if window_sigma_m is not None and (not math.isfinite(window_sigma_m) or window_sigma_m <= 0):
        raise ValueError("Exit analysis window must have a positive finite physical width")
    if cancelled():
        raise InterruptedError("Exit diffraction sampling cancelled")
    modes = tuple(mode for mode in checkpoint.beam.modes if mode.weight_per_reference_electron > 0)
    if type(count) is not int or count < max(1, len(modes)):
        raise ValueError("Exit count must cover every positive-current coherent mode")
    if 256*count > numerics.maximum_working_bytes:
        raise MemoryError("Exit particle arrays exceed their declared memory budget")
    if not modes:
        return HybridExitParticles(np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0),
            np.empty(0, int), (), {"schema": "hybrid-exit-particles-v1", "count": 0,
            "parent_checkpoint_digest": checkpoint.digest, "source_weight": 0., "sampled_weight": 0.,
            "original_source_particles": False, "zero_loss_attenuation_applied_again": False}), ()
    masses = np.array([mode.weight_per_reference_electron for mode in modes])
    allocation = (count-len(modes))*masses/masses.sum()
    counts = 1+np.floor(allocation).astype(int)
    counts[np.argsort(-(allocation-np.floor(allocation)), kind="stable")[:count-counts.sum()]] += 1
    rng = np.random.default_rng(seed)
    positions, directions, energies, weights, labels, spectra = [], [], [], [], [], []
    z_m = checkpoint.plane_z_mm*1e-3
    for index, (mode, n) in enumerate(zip(modes, counts)):
        local_numerics = numerics
        if window_sigma_m is not None:
            spacing = float(max(np.linalg.norm(mode.plane.basis_m, axis=0)))
            local_numerics = replace(numerics, window_sigma_pixels=window_sigma_m/spacing,
                frame_stride=max(1, round(window_sigma_m/spacing)))
        targets = (np.arange(n)+rng.random())/n
        selected_rows, selected_columns = [], []
        cumulative, selected = 0., 0
        def consume(window_index, probability):
            nonlocal cumulative, selected
            cdf = np.cumsum(probability)
            stop = int(np.searchsorted(targets, cumulative+cdf[-1], side="left"))
            if stop > selected:
                selected_columns.extend(np.searchsorted(cdf, targets[selected:stop]-cumulative, side="right"))
                selected_rows.extend([window_index]*(stop-selected))
                selected = stop
            cumulative += float(cdf[-1])
        centres, frequency, _, spectrum = local_diffraction(mode, snapshot, z_m, local_numerics,
            xp=xp, cancelled=cancelled, consume_window=consume, progress_callback=progress_callback)
        if selected != n:
            raise ValueError("Streaming diffraction failed to account for every requested sample")
        rows, columns = np.asarray(selected_rows, int), np.asarray(selected_columns, int)
        xy = centres[rows]
        origin_phi = float(snapshot.sample(np.array((0., 0., z_m)))["phi_v"])
        phi = snapshot.sample(np.column_stack((xy, np.full(n, z_m))))["phi_v"]
        energy = mode.energy_kev*1000+phi-origin_phi
        # The Wilson-line local spectra already represent mechanical momentum.
        transverse = h*frequency[columns]
        transverse_direction = transverse/momentum(energy)[:, None]
        longitudinal2 = 1-np.sum(transverse_direction**2, axis=1)
        if np.any(longitudinal2 <= 0):
            raise ValueError("Local diffraction sampled non-forward momentum; this paraxial handoff is unsupported")
        positions.append(np.column_stack((xy, np.full(n, z_m))))
        directions.append(np.column_stack((transverse_direction, np.sqrt(longitudinal2))))
        energies.append(energy)
        weights.append(np.full(n, mode.weight_per_reference_electron/n))
        labels.append(np.full(n, index))
        spectra.append({**spectrum, "mode_id": mode.mode_id})
    output = HybridExitParticles(np.concatenate(positions), np.concatenate(directions), np.concatenate(energies),
        np.concatenate(weights), np.concatenate(labels), tuple(mode.mode_id for mode in modes),
        {"schema": "hybrid-exit-particles-v1", "seed": seed, "count": count,
         "weight_reference": checkpoint.beam.reference_plane,
         "parent_checkpoint_digest": checkpoint.digest, "original_source_particles": False,
         "time_reference": "static handoff; absolute emission/arrival time not reconstructed",
         "zero_loss_attenuation_applied_again": False,
         "source_weight": float(masses.sum()), "sampled_weight": float(math.fsum(np.concatenate(weights))),
         "sampling": "mode-stratified systematic sampling of gauge-covariant local diffraction",
         "spatial_angular_coarse_graining": True})
    return output, tuple(spectra)
