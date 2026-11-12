"""Complex128 CUDA operators for a local mixed-state STEM scan.

The periodic reduction is a change of basis of the same finite-grid material
operator used by the independent resident calculation. External fields are
handled separately by the explicitly qualified overlap split in the caller.
"""
from dataclasses import dataclass
import math

import numpy as np
from temsim.physics.material_slice_storage import CompactPeriodicPotential


def exact_kinetic_phase(frequency_squared, wavelength_angstrom, distance_angstrom, *, xp=np):
    """Stable forward angular-spectrum phase with the axial carrier removed."""
    u = wavelength_angstrom**2*frequency_squared
    if not bool(xp.all(xp.isfinite(u))) or bool(xp.any(u >= 1)):
        raise ValueError("The complete represented kinetic lattice exceeds the forward hemisphere")
    return xp.exp((-2j*np.pi*distance_angstrom/wavelength_angstrom)*u/(1+xp.sqrt(1-u)))


def rotate_wave(wave, angle_rad, pixel_size_xy_angstrom, *, xp):
    """Unitary three-Fourier-shear rotation; never interpolate or renormalize."""
    if angle_rad == 0:
        return wave
    ny, nx = wave.shape[-2:]
    dx, dy = pixel_size_xy_angstrom
    x = (xp.arange(nx)-nx//2)*dx
    y = (xp.arange(ny)-ny//2)*dy
    fx, fy = xp.fft.fftfreq(nx, dx), xp.fft.fftfreq(ny, dy)
    a, b = -math.tan(angle_rad/2), math.sin(angle_rad)
    phase_x = xp.exp(-2j*np.pi*a*y[:, None]*fx[None, :])
    phase_y = xp.exp(-2j*np.pi*b*fy[:, None]*x[None, :])
    result = xp.fft.ifft(xp.fft.fft(wave, axis=-1, norm="ortho")*phase_x, axis=-1, norm="ortho")
    result = xp.fft.ifft(xp.fft.fft(result, axis=-2, norm="ortho")*phase_y, axis=-2, norm="ortho")
    return xp.fft.ifft(xp.fft.fft(result, axis=-1, norm="ortho")*phase_x, axis=-1, norm="ortho")


@dataclass(frozen=True)
class ScanPacket:
    modes: np.ndarray
    weights: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    wavelength_angstrom: float

    @property
    def fraction(self):
        return float(np.sum(abs(self.modes)**2*self.weights[None, :]))


def resident_observe(cp, packet, potential, slices, fov_xy, positions_angstrom,
                     intervals, *, interaction, bandwidth_fraction, batch_size,
                     field=None, field_blocks=1, diffraction=False,
                     diffraction_grid_shape=None, diffraction_per_position=False,
                     diagnostics=None,
                     cancelled=lambda: False, progress_callback=None):
    """Independent full-grid mixed-state multislice, including split fields."""
    ny, nx = potential.shape[-2:]
    dx, dy = fov_xy[0]/nx, fov_xy[1]/ny
    fx, fy = cp.meshgrid(cp.fft.fftfreq(nx, dx), cp.fft.fftfreq(ny, dy))
    squared = fx*fx+fy*fy
    cutoff = bandwidth_fraction*min(.5/dx, .5/dy)
    keep = squared <= cutoff**2
    angle = cp.arcsin(cp.minimum(1., packet.wavelength_angstrom*cp.sqrt(squared)))*1000
    masks = [((angle >= lo) & (angle < hi) & keep) for lo, hi in intervals.values()]
    # Full represented rectangle is checked even though the passband is a disk.
    kinetic_cache = {}
    kinetic = []
    for dz in slices:
        key = float(dz)
        if key not in kinetic_cache:
            kinetic_cache[key] = exact_kinetic_phase(squared, packet.wavelength_angstrom, key/2, xp=cp)*keep
        kinetic.append(kinetic_cache[key])
    n = len(positions_angstrom)
    sums = cp.zeros((len(intervals), n), dtype=cp.float64)
    out_power = cp.zeros(n, dtype=cp.float64)
    readout_y, readout_x = (ny, nx) if diffraction_grid_shape is None else tuple(diffraction_grid_shape)
    if readout_y < ny or readout_x < nx or (readout_y-ny) % 2 or (readout_x-nx) % 2:
        raise ValueError("Diffraction zero embedding requires a containing, centre-aligned same-pitch real-space rectangle")
    diffraction_sum = (cp.zeros(((n,) if diffraction_per_position else ())+(readout_y, readout_x), dtype=cp.float64)
                       if diffraction else None)
    depth_indices = np.unique(np.r_[0, np.linspace(1, len(slices), min(9, len(slices))).round().astype(int)])
    edge_max = cp.zeros(len(depth_indices)) if diagnostics is not None else None
    edge_weighted = cp.zeros(len(depth_indices)) if diagnostics is not None else None
    def edge_at(wave, depth, weight):
        if diagnostics is None or depth not in depth_indices:
            return
        index = int(np.flatnonzero(depth_indices == depth)[0])
        total = cp.sum(abs(wave)**2, axis=(-2, -1), dtype=cp.float64)
        edge = (cp.sum(abs(wave[..., :2, :])**2, axis=(-2, -1), dtype=cp.float64)
                +cp.sum(abs(wave[..., -2:, :])**2, axis=(-2, -1), dtype=cp.float64)
                +cp.sum(abs(wave[..., 2:-2, :2])**2, axis=(-2, -1), dtype=cp.float64)
                +cp.sum(abs(wave[..., 2:-2, -2:])**2, axis=(-2, -1), dtype=cp.float64))
        edge_max[index] = cp.maximum(edge_max[index], cp.max(edge/cp.maximum(total, 1e-300)))
        edge_weighted[index] += weight*cp.sum(edge)/n
    mode_count = len(packet.weights)
    boundaries = np.unique(np.rint(np.linspace(0, len(slices), field_blocks+1)).astype(int))
    if len(boundaries) != field_blocks+1:
        raise ValueError("Field split blocks cannot exceed material slice count")
    for mi, weight in enumerate(packet.weights):
        if cancelled():
            raise InterruptedError("Mixed STEM scan cancelled; no partial image published")
        base = cp.zeros((ny, nx), dtype=cp.complex128)
        base[cp.asarray(packet.gy % ny), cp.asarray(packet.gx % nx)] = cp.asarray(packet.modes[:, mi])
        for begin in range(0, n, batch_size):
            if cancelled():
                raise InterruptedError("Mixed STEM scan cancelled; no partial image published")
            end = min(n, begin+batch_size)
            xy = cp.asarray(positions_angstrom[begin:end])
            shift = cp.exp(-2j*np.pi*(xy[:, 0, None, None]*fx+xy[:, 1, None, None]*fy))
            spectrum = base[None, :, :]*shift
            wave = cp.fft.ifft2(spectrum, norm="ortho")
            edge_at(wave, 0, weight)
            for block in range(field_blocks):
                lo, hi = boundaries[block:block+2]
                distance = float(np.sum(slices[lo:hi]))
                if field is not None:
                    wave = field(wave, distance/2, (dx, dy), cp)
                    spectrum = cp.fft.fft2(wave, norm="ortho")
                for si in range(lo, hi):
                    # Stream one grating. Retaining NZ full complex128 grids
                    # would make realistic 10 nm refinements exceed 24 GiB.
                    layer = potential.slice_at(si) if isinstance(potential,CompactPeriodicPotential) else potential[si]
                    transmission = cp.exp(1j*interaction*cp.asarray(layer, dtype=cp.float64))
                    wave = cp.fft.ifft2(spectrum*kinetic[si], norm="ortho")*transmission
                    spectrum = cp.fft.fft2(wave, norm="ortho")*kinetic[si]
                    if si+1 != hi or field is None:
                        if diagnostics is not None and si+1 in depth_indices:
                            # The diagnostic never becomes the propagation
                            # state and is not FFT-transformed back again.
                            edge_at(cp.fft.ifft2(spectrum, norm="ortho"), si+1, weight)
                if field is not None:
                    wave = cp.fft.ifft2(spectrum, norm="ortho")
                    wave = field(wave, distance/2, (dx, dy), cp)
                    spectrum = cp.fft.fft2(wave, norm="ortho")
                    edge_at(wave, hi, weight)
            power = cp.abs(spectrum)**2
            for di, mask in enumerate(masks):
                sums[di, begin:end] += weight*cp.sum(power[:, mask], axis=1, dtype=cp.float64)
            out_power[begin:end] += weight*cp.sum(power, axis=(-2, -1), dtype=cp.float64)
            if diffraction:
                if (readout_y, readout_x) != (ny, nx):
                    # Embed the actual complex exit field at unchanged cell
                    # area, then transform. Intensities at different reciprocal
                    # spacings must never simply be padded or interpolated.
                    extended = cp.zeros((end-begin, readout_y, readout_x), dtype=cp.complex128)
                    oy, ox = (readout_y-ny)//2, (readout_x-nx)//2
                    if field is None:
                        wave = cp.fft.ifft2(spectrum, norm="ortho")
                    extended[:, oy:oy+ny, ox:ox+nx] = wave
                    readout_power = abs(cp.fft.fft2(extended, norm="ortho"))**2
                else:
                    readout_power = power
                if diffraction_per_position:
                    diffraction_sum[begin:end] += weight*readout_power
                else:
                    diffraction_sum += weight*cp.sum(readout_power, axis=0, dtype=cp.float64)/n
        if progress_callback is not None:
            progress_callback(mi+1, mode_count, "CUDA resident mixed STEM components")
    observed, power = cp.asnumpy(sums), cp.asnumpy(out_power)
    loss = packet.fraction-power
    if not np.isfinite(observed).all() or np.min(loss) < -2e-11:
        raise ValueError("CUDA resident mixed STEM violates absolute current conservation")
    if diagnostics is not None:
        diagnostics.update(sampled_slice_boundaries=depth_indices.tolist(),
            sampled_depth_angstrom=[float(np.sum(slices[:i])) for i in depth_indices],
            maximum_unit_component_outer_2cell_fraction=cp.asnumpy(edge_max).tolist(),
            mean_weighted_outer_2cell_fraction_of_emitted=cp.asnumpy(edge_weighted).tolist(),
            scope="Depth/held-out diagnostics of a finite periodic grid, not the general isolated-carrier boundary certificate")
    return observed, np.maximum(loss, 0.), (None if diffraction_sum is None else cp.asnumpy(cp.fft.fftshift(diffraction_sum, axes=(-2, -1))))


def periodic_coefficients(cp, packet, potential, slices, fov_xy, tiles_xy,
                          intervals, *, interaction, bandwidth_fraction,
                          column_batch=24, cancelled=lambda: False,
                          progress_callback=None):
    """Exact finite-grid Bloch reduction, after literal periodicity proof."""
    tx, ty = tiles_xy
    nz, full_y, full_x = potential.shape
    if full_x % tx or full_y % ty:
        raise ValueError("The material grid is not commensurate with its crystal")
    ny, nx = full_y//ty, full_x//tx
    primitive = potential.primitive if isinstance(potential,CompactPeriodicPotential) else potential[:, :ny, :nx]
    # Compare literal tiles without constructing another full NZ*NY*NX copy.
    if isinstance(potential,CompactPeriodicPotential):
        literal_proof=potential.literal_periodicity_record(tiles_xy)
    else:
        for row in range(ty):
            for column in range(tx):
                if not np.array_equal(potential[:, row*ny:(row+1)*ny, column*nx:(column+1)*nx], primitive):
                    raise ValueError("Bloch reduction requires exact slice-by-slice periodicity")
        literal_proof={"proof":"Every dense tile compared literally with its primitive",
                       "tiles_xy":list(tiles_xy)}
    if slices.shape != (nz,) or np.any(slices <= 0):
        raise ValueError("Bloch reduction requires actual positive material slices")
    px, py = fov_xy[0]/tx, fov_xy[1]/ty
    bx = np.fft.fftfreq(nx, 1/nx).round().astype(np.int64)
    by = np.fft.fftfreq(ny, 1/ny).round().astype(np.int64)
    rx = (packet.gx+tx//2) % tx-tx//2
    ry = (packet.gy+ty//2) % ty-ty//2
    kx, ky = (packet.gx-rx)//tx, (packet.gy-ry)//ty
    mx, my = int(np.ptp(kx)), int(np.ptp(ky))
    shape = (2*my+1, 2*mx+1)
    harmonic = cp.zeros((len(intervals), np.prod(shape)), dtype=cp.complex128)
    density_modes = cp.asarray(packet.modes)*cp.sqrt(cp.asarray(packet.weights))[None, :]
    transmission = cp.exp(1j*interaction*cp.asarray(primitive, dtype=cp.float64))
    cutoff = bandwidth_fraction*min(.5*full_x/fov_xy[0], .5*full_y/fov_xy[1])
    sectors = sorted(set(zip(ry.tolist(), rx.tolist())))
    gain = 0.
    for number, (qy, qx) in enumerate(sectors):
        if cancelled():
            raise InterruptedError("Mixed STEM scan cancelled; no partial image published")
        index = np.flatnonzero((rx == qx) & (ry == qy))
        ix, iy = kx[index], ky[index]
        squared = (bx[None, :]/px+qx/fov_xy[0])**2+(by[:, None]/py+qy/fov_xy[1])**2
        admitted = squared <= cutoff**2
        if not np.all(admitted[iy % ny, ix % nx]):
            raise ValueError("The Bloch material passband clips the incident mixed state")
        angles = np.arcsin(np.minimum(1., packet.wavelength_angstrom*np.sqrt(squared)))*1000
        masks = [cp.asarray(((angles >= lo) & (angles < hi))[admitted]) for lo, hi in intervals.values()]
        keep = cp.asarray(admitted)
        # A sector is a full represented reciprocal rectangle too.
        half_cache = {}
        half = []
        for dz in slices:
            key = float(dz)
            if key not in half_cache:
                half_cache[key] = cp.asarray(exact_kinetic_phase(squared, packet.wavelength_angstrom, key/2)*admitted)
            half.append(half_cache[key])
        scatter = cp.empty((len(index), int(admitted.sum())), dtype=cp.complex128)
        for begin in range(0, len(index), column_batch):
            if cancelled():
                raise InterruptedError("Mixed STEM scan cancelled; no partial image published")
            end = min(len(index), begin+column_batch)
            spectrum = cp.zeros((end-begin, ny, nx), dtype=cp.complex128)
            spectrum[cp.arange(end-begin), cp.asarray(iy[begin:end] % ny), cp.asarray(ix[begin:end] % nx)] = 1
            for si in range(nz):
                wave = cp.fft.ifft2(spectrum*half[si], norm="ortho")*transmission[si]
                spectrum = cp.fft.fft2(wave, norm="ortho")*half[si]
            scatter[begin:end] = spectrum[:, keep]
        gain = max(gain, float(cp.max(cp.sum(abs(scatter)**2, axis=1)-1)))
        block = density_modes[cp.asarray(index)]
        rho = block.conj() @ block.T
        difference = cp.asarray(((iy[:, None]-iy[None, :]+my)*shape[1]+ix[:, None]-ix[None, :]+mx).ravel())
        for di, mask in enumerate(masks):
            receive = scatter[:, mask]
            if receive.shape[1]:
                terms = (rho*(receive.conj() @ receive.T)).ravel()
                harmonic[di] += cp.bincount(difference, weights=terms.real, minlength=np.prod(shape))
                harmonic[di] += 1j*cp.bincount(difference, weights=terms.imag, minlength=np.prod(shape))
        if progress_callback is not None:
            progress_callback(number+1, len(sectors), "CUDA Bloch material sectors")
    result = cp.asnumpy(harmonic).reshape((len(intervals), *shape))
    if gain > 2e-11 or not np.allclose(result, result[:, ::-1, ::-1].conj(), rtol=0, atol=2e-11):
        raise ValueError("Bloch flux or Hermitian scan coefficients failed validation")
    return result, {"period_angstrom_xy": [px, py], "bloch_sector_count": len(sectors),
                    "largest_unit_input_column_gain": gain, "complex_precision": "complex128",
                    "literal_periodicity":literal_proof}


def evaluate_periodic(harmonic, positions_angstrom, period_xy):
    ny, nx = harmonic.shape[-2:]
    x = np.exp(2j*np.pi*positions_angstrom[:, 0, None]*(np.arange(nx)-nx//2)/period_xy[0])
    y = np.exp(2j*np.pi*positions_angstrom[:, 1, None]*(np.arange(ny)-ny//2)/period_xy[1])
    result = np.einsum("py,dyx,px->dp", y, harmonic, x, optimize=True)
    if np.max(abs(result.imag)) > 2e-11 or result.real.min() < -2e-11:
        raise ValueError("Bloch detector probabilities are not real and positive")
    return np.maximum(result.real, 0.)
