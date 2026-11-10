"""Herman--Kluk coherent ray propagation in the captured static potentials.

Internal array operators also support NumPy for mathematical unit tests. User
jobs enter through hybrid_pipeline and require CUDA. Canonical transverse
momenta are divided by one mode's reference momentum; action uses that SAME
reference. Gaussian frame coefficients are complex quadrature amplitudes,
never electron counts. No source, physical loss or wave norm is renormalised.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.constants import c, e, hbar, m_e

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.potential_ray import trace_potential_rays
from temsim.physics.wave_device import to_host


def momentum(energy_ev):
    energy = np.asarray(energy_ev, float)*e
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("Hybrid transport requires positive kinetic energy")
    return np.sqrt(energy*(energy+2*m_e*c*c))/c


@dataclass(frozen=True)
class SemiclassicalNumerics:
    frame_stride: int = 4
    window_sigma_pixels: float = 4.
    frame_padding_sigma: float = 5.
    coefficient_cutoff: float = 1e-10
    maximum_packets: int = 65536
    packet_batch: int = 64
    maximum_working_bytes: int = 2*1024**3
    reconstruction_tolerance: float = 2e-3
    norm_tolerance: float = 5e-3
    finite_difference_fraction: float = 1e-3
    symplectic_tolerance: float = 2e-3
    maximum_phase_increment: float = .5
    maximum_step_m: float = 1e-4
    ray_relative_tolerance: float = 1e-10

    def validate(self):
        for name in ("frame_stride", "maximum_packets", "packet_batch", "maximum_working_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("window_sigma_pixels", "frame_padding_sigma", "reconstruction_tolerance",
                     "norm_tolerance", "finite_difference_fraction", "symplectic_tolerance",
                     "maximum_phase_increment", "maximum_step_m", "ray_relative_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 <= self.coefficient_cutoff < 1 or not np.isfinite(self.coefficient_cutoff):
            raise ValueError("Invalid coherent coefficient cutoff")
        if self.maximum_phase_increment >= math.pi/2:
            raise ValueError("Caustic phase increments must be resolved below pi/2")
        if self.finite_difference_fraction >= .1:
            raise ValueError("Tangent probes must be local to their Gaussian packet")
        if self.norm_tolerance >= 1:
            raise ValueError("Lossless norm tolerance must be less than one")
        return self


@dataclass(frozen=True)
class CoherentPackets:
    positions_m: np.ndarray
    canonical_slopes: np.ndarray
    coefficients: np.ndarray
    sigma_m: float
    wave_number_m1: float
    mode_id: str
    record: dict


def lattice(plane):
    basis = np.asarray(plane.basis_m, float)
    lengths = np.linalg.norm(basis, axis=0)
    if not np.allclose(basis.T@basis, np.diag(lengths**2), rtol=1e-12,
                       atol=1e-12*min(lengths)**2):
        raise ValueError("Hybrid Gaussian frames require an orthogonal XY lattice")
    xy = plane.coordinates_m().reshape(2, -1).T
    ny, nx = plane.amplitude.shape
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(ny)),
                         np.fft.fftshift(np.fft.fftfreq(nx)), indexing="ij")
    frequency = np.einsum("ij,jyx->yxi", np.linalg.inv(basis).T, np.stack((fx, fy))).reshape(-1, 2)
    return xy, frequency, lengths


def frame_centres(plane, numerics):
    xy, frequency, lengths = lattice(plane)
    sigma = numerics.window_sigma_pixels*max(lengths)
    pad = int(math.ceil(numerics.frame_padding_sigma*sigma/min(lengths)))
    ny, nx = plane.amplitude.shape
    # Anchor the lattice to the observation origin, not to a rounded padding
    # boundary. Otherwise one extra padding pixel shifts ALL windows when the
    # same physical wave is represented on a finer numerical grid.
    stride = numerics.frame_stride
    def axis(count):
        first = -math.ceil((count//2+pad)/stride)
        last = math.ceil((count-1-count//2+pad)/stride)
        return np.arange(first, last+1)*stride
    yy, xx = np.meshgrid(axis(ny), axis(nx), indexing="ij")
    centres = np.stack((xx, yy), axis=-1).reshape(-1, 2)@plane.basis_m.T+plane.origin_m
    return xy, frequency, centres, sigma


def gaussian_window(points, centre, sigma, *, xp=np):
    delta = points-xp.asarray(centre)
    return xp.exp(-xp.sum(delta*delta, axis=-1)/(4*sigma*sigma))/(math.sqrt(2*math.pi)*sigma)


def decompose_mode(mode, numerics, *, xp, cancelled=lambda: False):
    """Finite Gabor quadrature of the executed full phase, with reconstruction check."""
    numerics.validate()
    if cancelled():
        raise InterruptedError("Coherent frame decomposition cancelled")
    plane = mode.plane
    xy, frequency, centres, sigma = frame_centres(plane, numerics)
    size = plane.amplitude.size
    if 192*size+80*numerics.maximum_packets > numerics.maximum_working_bytes:
        raise MemoryError("Gaussian frame exceeds its explicit memory budget")
    k = float(momentum(mode.energy_kev*1000)/hbar)
    amplitude = xp.asarray(plane.full_amplitude(2*math.pi/k))
    points = xp.asarray(xy)
    area = abs(float(np.linalg.det(plane.basis_m)))
    frame_weight = numerics.frame_stride**2/size
    positions, slopes, coefficients = [], [], []
    discarded_bound, count = 0., 0
    rebuilt = xp.zeros_like(amplitude)
    for centre in centres:
        if cancelled():
            raise InterruptedError("Coherent frame decomposition cancelled")
        window = gaussian_window(points, centre, sigma, xp=xp).reshape(amplitude.shape)
        transform = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(amplitude*window), norm="ortho"))
        overlap = to_host(transform).ravel()*math.sqrt(area*size)
        overlap *= np.exp(2j*np.pi*(frequency@(centre-plane.origin_m)))
        values = frame_weight*overlap
        keep = abs(values) > numerics.coefficient_cutoff
        discarded_bound += float(np.sum(abs(values[~keep])))
        count += int(keep.sum())
        if count > numerics.maximum_packets:
            raise MemoryError("Coherent frame exceeds maximum_packets; refine the declared budget, not the source")
        # Exact discrete synthesis on the ORIGINAL lattice: invert each
        # window's retained Fourier coefficients. This is the same Gaussian
        # sum as evaluate_packets, without an O(packets * pixels) launch check.
        filtered = transform*xp.asarray(keep.reshape(amplitude.shape))
        local = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(filtered), norm="ortho"))
        rebuilt += numerics.frame_stride**2*area*window*local
        positions.append(np.broadcast_to(centre, (keep.sum(), 2)).copy())
        slopes.append(2*np.pi*frequency[keep]/k)
        coefficients.append(values[keep])
    packets = CoherentPackets(np.concatenate(positions), np.concatenate(slopes), np.concatenate(coefficients),
        sigma, k, mode.mode_id, {"quadrature": "finite Gaussian coherent-state frame",
            "discarded_amplitude_l2_bound": discarded_bound, "packet_count": count,
            "packets_are_original_electrons": False})
    error = float(xp.linalg.norm(rebuilt-amplitude))
    if error > numerics.reconstruction_tolerance:
        raise ValueError(f"Coherent frame reconstruction L2 error {error:g} exceeds {numerics.reconstruction_tolerance:g}; refine frame/grid")
    return replace(packets, record={**packets.record, "initial_reconstruction_l2": error})


def evaluate_packets(packets, plane, *, xp, numerics, cancelled=lambda: False):
    points = xp.asarray(plane.coordinates_m().reshape(2, -1).T)
    size, batch = len(points), numerics.packet_batch
    point_batch = min(size, max(1, (numerics.maximum_working_bytes-64*size)//(128*batch)))
    if numerics.maximum_working_bytes < 192*size+128*batch:
        raise MemoryError("Coherent reconstruction exceeds its tile budget")
    output = xp.zeros(size, dtype=xp.complex128)
    k, sigma = packets.wave_number_m1, packets.sigma_m
    normalizer = 1/(math.sqrt(2*math.pi)*sigma)
    for start in range(0, size, point_batch):
        stop = min(start+point_batch, size)
        for first in range(0, len(packets.coefficients), batch):
            if cancelled():
                raise InterruptedError("Coherent reconstruction cancelled")
            sl = slice(first, first+batch)
            delta = points[None, start:stop]-xp.asarray(packets.positions_m[sl])[:, None]
            phase = k*xp.einsum("bpi,bi->bp", delta, xp.asarray(packets.canonical_slopes[sl]))
            envelope = -xp.sum(delta*delta, axis=-1)/(4*sigma*sigma)
            output[start:stop] += xp.sum(xp.asarray(packets.coefficients[sl])[:, None]
                *xp.exp(envelope+1j*phase), axis=0)*normalizer
    return (output*math.sqrt(abs(float(np.linalg.det(plane.basis_m))))).reshape(plane.amplitude.shape)


def lifted_hk_prefactor(maps, gamma_m1, *, maximum_phase_increment=.5, symplectic_tolerance=2e-3):
    """Continuous square root; resolves simultaneous foci without det-angle aliasing.

    M is d(q,P/p_ref)_out/d(q,P/p_ref)_in, shape (planes, rays, 4, 4).
    Relative eigenphases retain both transverse windings at an axial focus.
    """
    maps = np.asarray(maps, float)
    if maps.ndim != 4 or maps.shape[-2:] != (4, 4) or not np.isfinite(maps).all():
        raise ValueError("Invalid canonical tangent maps")
    scales = np.array((1., 1., gamma_m1, gamma_m1))
    scaled = maps*scales[None, None, None, :]/scales[None, None, :, None]
    j = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    residual = np.max(abs(np.swapaxes(scaled, -1, -2)@j@scaled-j))
    if residual > symplectic_tolerance:
        raise ValueError(f"Ray tangent symplectic residual {residual:g}; refine finite differences/integration")
    a, b, cc, d = maps[..., :2, :2], maps[..., :2, 2:], maps[..., 2:, :2], maps[..., 2:, 2:]
    matrix = .5*(a+d-1j*gamma_m1*b+1j*cc/gamma_m1)
    if not np.allclose(matrix[0], np.eye(2), rtol=0., atol=1e-7):
        raise ValueError("Herman--Kluk lift must start with the identity")
    phases = np.zeros(matrix.shape[:2])
    largest = 0.
    for index in range(1, len(matrix)):
        relative = matrix[index]@np.linalg.inv(matrix[index-1])
        increments = np.angle(np.linalg.eigvals(relative))
        largest = max(largest, float(np.max(abs(increments))))
        if largest > maximum_phase_increment:
            raise ValueError("Caustic phase path is undersampled; reduce the axial step")
        phases[index] = phases[index-1]+np.sum(increments, axis=-1)
    magnitude = np.sqrt(abs(np.linalg.det(matrix)))
    return magnitude*np.exp(.5j*phases), {"maximum_symplectic_residual": float(residual),
        "maximum_lift_increment_rad": largest, "continuous_prefactor_phase_rad": .5*phases[-1]}


def canonical_to_ray(snapshot, xy, canonical, z_m, reference_energy_ev, reference_phi_v):
    xyz = np.column_stack((xy, np.full(len(xy), z_m)))
    field = snapshot.sample(xyz)
    energy = reference_energy_ev+field["phi_v"]-reference_phi_v
    p_ref = float(momentum(reference_energy_ev))
    transverse = p_ref*canonical+e*field["A_t_m"][:, :2]
    pz2 = momentum(energy)**2-np.sum(transverse**2, axis=1)
    if np.any(pz2 <= 0):
        raise ValueError("Coherent frame contains non-forward canonical rays")
    return np.column_stack((xy, transverse/np.sqrt(pz2)[:, None])), energy


def transport_packets(packets, snapshot, z_m, energy_ev, reference_phi_v, numerics, *, cancelled=lambda: False):
    """Classical central paths, canonical tangent probes and reduced action."""
    nodes = np.asarray(z_m, float)
    count, k = len(packets.coefficients), packets.wave_number_m1
    if 2400*len(nodes)*min(count, numerics.packet_batch) > numerics.maximum_working_bytes:
        raise MemoryError("Canonical tangent histories exceed their memory budget; add intermediate grids or reduce packet_batch")
    gamma = 1/(2*k*packets.sigma_m**2)
    delta = numerics.finite_difference_fraction*np.array((packets.sigma_m, packets.sigma_m,
        1/(2*k*packets.sigma_m), 1/(2*k*packets.sigma_m)))
    centres, directions, coefficients, receipts = [], [], [], []
    p_ref = float(momentum(energy_ev))
    for first in range(0, count, numerics.packet_batch):
        if cancelled():
            raise InterruptedError("Semiclassical ray propagation cancelled")
        sl = slice(first, first+numerics.packet_batch)
        initial = np.column_stack((packets.positions_m[sl], packets.canonical_slopes[sl]))
        raw = np.repeat(initial[:, None, :], 9, axis=1)
        for axis in range(4):
            raw[:, 1+2*axis, axis] += delta[axis]
            raw[:, 2+2*axis, axis] -= delta[axis]
        flat = raw.reshape(-1, 4)
        launch, energies = canonical_to_ray(snapshot, flat[:, :2], flat[:, 2:], nodes[0], energy_ev, reference_phi_v)
        result = trace_potential_rays(snapshot, launch, nodes, energy_ev=energies,
            maximum_step_m=numerics.maximum_step_m, relative_tolerance=numerics.ray_relative_tolerance,
            position_tolerance_m=min(1e-13, min(delta[:2])*1e-3),
            slope_tolerance_rad=min(1e-11, min(delta[2:])*1e-3), cancelled=cancelled)
        mechanical = result.xy_slopes
        phase_space = mechanical.copy()
        for row, z in enumerate(nodes):
            xyz = np.column_stack((mechanical[row, :, :2], np.full(len(flat), z)))
            field = snapshot.sample(xyz)
            local_energy = energy_ev+field["phi_v"]-reference_phi_v
            pz = momentum(local_energy)/np.sqrt(1+np.sum(mechanical[row, :, 2:]**2, axis=1))
            phase_space[row, :, 2:] = (pz[:, None]*mechanical[row, :, 2:]-e*field["A_t_m"][:, :2])/p_ref
        samples = phase_space.reshape(len(nodes), len(initial), 9, 4)
        maps = np.stack([(samples[:, :, 1+2*a]-samples[:, :, 2+2*a])/(2*delta[a]) for a in range(4)], axis=-1)
        prefactor, receipt = lifted_hk_prefactor(maps, gamma,
            maximum_phase_increment=numerics.maximum_phase_increment, symplectic_tolerance=numerics.symplectic_tolerance)
        ratio = momentum(energies.reshape(-1, 9)[:, 0])/p_ref
        action = ratio*result.reduced_action_m[-1].reshape(-1, 9)[:, 0]+(ratio-1)*(nodes[-1]-nodes[0])
        coefficients.append(packets.coefficients[sl]*prefactor[-1]*np.exp(1j*k*action))
        centres.append(samples[-1, :, 0, :2])
        directions.append(samples[-1, :, 0, 2:])
        receipts.append({**receipt, "reduced_action_m": action})
    return replace(packets, positions_m=np.concatenate(centres), canonical_slopes=np.concatenate(directions),
        coefficients=np.concatenate(coefficients), record={**packets.record,
        "propagator": "Herman--Kluk IVR; finite-difference canonical tangent; continuous Gaussian caustic lift",
        "maximum_symplectic_residual": max(r["maximum_symplectic_residual"] for r in receipts),
        "maximum_lift_increment_rad": max(r["maximum_lift_increment_rad"] for r in receipts),
        "caustic_phase_rad": np.concatenate([r["continuous_prefactor_phase_rad"] for r in receipts]).tolist(),
        "reduced_action_m": np.concatenate([r["reduced_action_m"] for r in receipts]).tolist()})


def propagate_mode(mode, snapshot, start_m, stop_m, output_plane, numerics, *, xp, cancelled=lambda: False):
    """Lossless semiclassical interval; report signed numerical norm error."""
    packets = decompose_mode(mode, numerics, xp=xp, cancelled=cancelled)
    nodes = np.linspace(start_m, stop_m, max(2, math.ceil((stop_m-start_m)/numerics.maximum_step_m)+1))
    phi = float(snapshot.sample(np.array((0., 0., start_m)))["phi_v"])
    transported = transport_packets(packets, snapshot, nodes, mode.energy_kev*1000, phi, numerics, cancelled=cancelled)
    amplitude = evaluate_packets(transported, output_plane, xp=xp, numerics=numerics, cancelled=cancelled)
    norm = float(xp.sum(abs(amplitude)**2))
    if not math.isfinite(norm) or abs(norm-1.) > numerics.norm_tolerance:
        raise ValueError(f"Semiclassical lossless norm {norm:g}; refine frame, integration or output grid; no current was restored")
    # Unit shape is a representation only; the absolute weight retains norm.
    wave = PlaneWave(to_host(amplitude)/math.sqrt(norm), output_plane.basis_m, output_plane.origin_m)
    final_phi = float(snapshot.sample(np.array((0., 0., stop_m)))["phi_v"])
    output = replace(mode, plane=wave, weight_per_reference_electron=mode.weight_per_reference_electron*norm,
        energy_kev=mode.energy_kev+(final_phi-phi)*1e-3, axial_reference=None)
    return output, {**transported.record, "mode_id": mode.mode_id,
        "input_weight": mode.weight_per_reference_electron, "output_weight": output.weight_per_reference_electron,
        "signed_numerical_probability_change": output.weight_per_reference_electron-mode.weight_per_reference_electron,
        "restored_current": False, "sample_feedback_on_external_fields": False}
