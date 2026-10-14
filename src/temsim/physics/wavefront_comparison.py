"""Read-only, mode-matched transverse phase on the actual wave's native lattice.

The paired execution workflow must establish a shared executed source. Equal
mode labels alone cannot establish that provenance. No mixture phase, fitted
wavefront, normalization, or removal of focusing is performed here.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

from temsim.immutable_json import freeze_json, json_digest
from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE, wavelength_m
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_device import to_host
from temsim.physics.wave_execution import check_available_memory


def _freeze(array):
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class WavefrontComparison:
    actual_phase_rad: np.ndarray
    ideal_phase_rad: np.ndarray
    difference_rad: np.ndarray
    valid_mask: np.ndarray
    basis_m: np.ndarray
    origin_m: np.ndarray
    metadata: object

    def __post_init__(self):
        for name in ("actual_phase_rad", "ideal_phase_rad", "difference_rad",
                     "valid_mask", "basis_m", "origin_m"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))


def _cancel(cancelled):
    if cancelled():
        raise InterruptedError("Wavefront comparison cancelled")


def _selection(checkpoint, mode_id):
    """Inspect disk manifests before loading any selected large mode array."""
    from temsim.physics.wave_checkpoint_store import _StoredModes

    modes = checkpoint.beam.modes
    if isinstance(modes, _StoredModes):
        matches = [(index, row) for index, row in enumerate(modes.rows)
                   if row["mode_id"] == mode_id]
        if len(matches) != 1:
            raise ValueError("Wavefront comparison requires one matching mode_id in each checkpoint")
        index, row = matches[0]
        shape = tuple(row["arrays"]["amplitude"]["shape"])
        return index, shape, None
    matches = [(index, mode) for index, mode in enumerate(modes) if mode.mode_id == mode_id]
    if len(matches) != 1:
        raise ValueError("Wavefront comparison requires one matching mode_id in each checkpoint")
    index, mode = matches[0]
    return index, mode.plane.amplitude.shape, mode


def _shape_size(shape):
    if (len(shape) != 2 or any(type(n) is not int or n < 2 for n in shape)):
        raise ValueError("Wavefront comparison requires two-dimensional sampled wave lattices")
    return math.prod(shape)


def _selected_digest(checkpoint, mode, amplitude, cancelled):
    """Hash only the selected, already loaded mode, with bounded temporaries."""
    digest = sha256()
    for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        value = amplitude if name == "amplitude" else getattr(mode.plane, name)
        digest.update(name.encode())
        if value is None:
            digest.update(b"none")
            continue
        value = np.asarray(value)
        digest.update(str((value.dtype.str, value.shape)).encode())
        for first in range(0, len(value), 32):
            _cancel(cancelled)
            digest.update(memoryview(np.ascontiguousarray(value[first:first+32])).cast("B"))
    return json_digest({"payload": digest.hexdigest(), "mode_id": mode.mode_id,
        "energy_kev": mode.energy_kev, "reference_plane": mode.reference_plane,
        "weight": mode.weight_per_reference_electron,
        "scattering_history": mode.scattering_history,
        "axial_reference": mode.axial_reference, "record": checkpoint.record,
        "plane_z_mm": checkpoint.plane_z_mm, "reference_current_a": checkpoint.reference_current_a})


def _carrier(wave, xy, wavelength):
    delta = xy-wave.origin_m[:, None, None]
    q = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    return (2*np.pi/wavelength)*(
        .5*np.einsum("iyx,ij,jyx->yx", delta, q, delta)
        + np.einsum("i,iyx->yx", tilt, delta))


def _coefficients(amplitude, order, cancelled):
    _cancel(cancelled)
    real = spline_filter(amplitude.real, order=order, mode="constant")
    _cancel(cancelled)
    imag = spline_filter(amplitude.imag, order=order, mode="constant")
    return real, imag


def _peak(amplitude, rows, cancelled):
    peak = 0.
    for first in range(0, len(amplitude), rows):
        _cancel(cancelled)
        peak = max(peak, float(np.max(abs(amplitude[first:first+rows]))))
    return peak


def _interpolate(coefficients, coordinates, order):
    real, imag = coefficients
    return (map_coordinates(real, coordinates, order=order, mode="constant", cval=0., prefilter=False)
            + 1j*map_coordinates(imag, coordinates, order=order, mode="constant", cval=0., prefilter=False))


def compare_wavefronts(actual_checkpoint, ideal_checkpoint, mode_id, *,
                      maximum_working_bytes=512*1024**2, remove_piston=False,
                      relative_intensity_threshold=1e-6, interpolation_tolerance=1e-4,
                      cancelled=lambda: False):
    """Compare one matched mode at exactly the same physical observation Z.

    Results are wrapped transverse envelope-plus-carrier phase, in radians,
    on actual native Y/X samples. They exclude longitudinal action and time
    phase. Difference means actual minus ideal. The optional piston removes
    only the argument of their complex overlap on common reliable support.

    Relative intensity is tested against EACH original mode's peak. For an
    unlike affine grid, only the ideal complex envelope is interpolated;
    carriers are evaluated analytically at actual laboratory coordinates.
    Cubic/quintic disagreement and a three-cell source-edge guard exclude
    unreliable samples. No wave or physical mode weight is renormalized.
    """
    _cancel(cancelled)
    if type(maximum_working_bytes) is not int or maximum_working_bytes <= 0:
        raise ValueError("Wavefront working-memory budget must be a positive integer")
    if type(remove_piston) is not bool:
        raise ValueError("Piston removal must be an explicit boolean")
    if not isinstance(mode_id, str) or not mode_id:
        raise ValueError("Wavefront comparison requires a nonempty mode_id")
    for value, label, lower in ((relative_intensity_threshold, "relative intensity threshold", 0.),
                                (interpolation_tolerance, "interpolation tolerance", 0.)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not lower <= value < 1
                or (label == "interpolation tolerance" and value == 0)):
            raise ValueError(f"Wavefront {label} must be finite and within its valid fractional range")
    za, zi = actual_checkpoint.plane_z_mm, ideal_checkpoint.plane_z_mm
    if (isinstance(za, (bool, np.bool_)) or isinstance(zi, (bool, np.bool_))
            or not np.isfinite((za, zi)).all() or za != zi):
        raise ValueError("Wavefront checkpoints must have exactly the same physical observation Z")
    currents = (actual_checkpoint.reference_current_a, ideal_checkpoint.reference_current_a)
    if (any(isinstance(v, (bool, np.bool_)) for v in currents) or not np.isfinite(currents).all()
            or min(currents) < 0 or currents[0] != currents[1]):
        raise ValueError("Wavefront checkpoints must share the same tip reference current")
    if (actual_checkpoint.beam.reference_plane != TIP_REFERENCE
            or ideal_checkpoint.beam.reference_plane != TIP_REFERENCE):
        raise ValueError("Wavefront comparison requires the executed physical-tip reference")
    ai, ashape, actual = _selection(actual_checkpoint, mode_id)
    ii, ishape, ideal = _selection(ideal_checkpoint, mode_id)
    count, ideal_count = _shape_size(ashape), _shape_size(ishape)
    # Account for both input snapshots, immutable output copies and both real
    # and imaginary cubic/quintic coefficient arrays, before a disk read.
    known_same_grid = (actual is not None and ideal is not None and ashape == ishape
        and np.array_equal(actual.plane.basis_m, ideal.plane.basis_m)
        and np.array_equal(actual.plane.origin_m, ideal.plane.origin_m))
    base_bytes = 50*count + 32*(count+ideal_count) + (0 if known_same_grid else 32*ideal_count)
    row_bytes = 384*max(ashape[1], ishape[1])
    if maximum_working_bytes < base_bytes+row_bytes:
        raise MemoryError("Wavefront comparison exceeds its working-memory budget before loading arrays")
    rows = min(128, ashape[0], max(1, (maximum_working_bytes-base_bytes)//row_bytes))
    check_available_memory(base_bytes+rows*row_bytes)
    _cancel(cancelled)
    actual = actual_checkpoint.beam.modes[ai] if actual is None else actual
    _cancel(cancelled)
    ideal = ideal_checkpoint.beam.modes[ii] if ideal is None else ideal
    if (actual.mode_id != ideal.mode_id or actual.energy_kev != ideal.energy_kev
            or actual.reference_plane != ideal.reference_plane
            or actual.scattering_history != ideal.scattering_history):
        raise ValueError("Wavefront modes must match mode_id, energy, electron reference and scattering history")
    if not isinstance(actual.plane, PlaneWave) or not isinstance(ideal.plane, PlaneWave):
        raise ValueError("Wavefront comparison requires forward Cartesian PlaneWave modes")
    if actual.weight_per_reference_electron <= 0 or ideal.weight_per_reference_electron <= 0:
        raise ValueError("Wavefront comparison has no common valid support: a mode has zero weight")
    a, b = actual.plane, ideal.plane
    amplitude_a, amplitude_i = to_host(a.amplitude), to_host(b.amplitude)
    if amplitude_a.shape != ashape or amplitude_i.shape != ishape:
        raise ValueError("Selected wave arrays no longer match their declared geometry")
    same_grid = (ashape == ishape and np.array_equal(a.basis_m, b.basis_m)
                 and np.array_equal(a.origin_m, b.origin_m))
    _cancel(cancelled)
    # Maxima do not allocate full intensity arrays.
    peak_a = _peak(amplitude_a, rows, cancelled)
    peak_i = _peak(amplitude_i, rows, cancelled)
    if min(peak_a, peak_i) <= 0:
        raise ValueError("Wavefront comparison has no common valid support: an empty wave")
    coefficients = None if same_grid else (
        _coefficients(amplitude_i, 3, cancelled), _coefficients(amplitude_i, 5, cancelled))
    phase_a, phase_i, difference = [np.full(ashape, np.nan) for _ in range(3)]
    valid = np.zeros(ashape, dtype=bool)
    minimum_a, minimum_i = np.sqrt(relative_intensity_threshold)*np.array((peak_a, peak_i))
    wavelength = float(wavelength_m(actual.energy_kev*1000))
    overlap, overlap_scale = 0j, 0.
    covered_count = rejected_interpolation = 0
    maximum_interpolation_error = 0.
    inverse = np.linalg.inv(b.basis_m)
    for first in range(0, ashape[0], rows):
        _cancel(cancelled)
        last = min(ashape[0], first+rows)
        yy, xx = np.meshgrid(np.arange(first, last)-ashape[0]//2,
                             np.arange(ashape[1])-ashape[1]//2, indexing="ij")
        xy = a.origin_m[:, None, None]+np.einsum("ij,jyx->iyx", a.basis_m, np.stack((xx, yy)))
        av = amplitude_a[first:last]
        if same_grid:
            iv = amplitude_i[first:last]
            covered = np.ones(av.shape, dtype=bool)
            reliable = covered
        else:
            indices = np.einsum("ij,jyx->iyx", inverse, xy-b.origin_m[:, None, None])
            indices += np.array((ishape[1]//2, ishape[0]//2))[:, None, None]
            # Do not extrapolate from absent data or trust high-order spline
            # values at an edge whose continuation is not known.
            covered = np.all((indices >= 3) & (indices <= np.array(
                (ishape[1]-4, ishape[0]-4))[:, None, None]), axis=0)
            cubic = _interpolate(coefficients[0], indices[::-1], 3)
            _cancel(cancelled)
            iv = _interpolate(coefficients[1], indices[::-1], 5)
            error = abs(cubic-iv)/np.maximum(abs(iv), max(minimum_i, peak_i*np.finfo(float).eps))
            reliable = covered & np.isfinite(error) & (error <= interpolation_tolerance)
            rejected_interpolation += int(np.count_nonzero(covered & ~reliable))
            if np.any(covered):
                maximum_interpolation_error = max(maximum_interpolation_error, float(np.max(error[covered])))
        covered_count += int(np.count_nonzero(covered))
        keep = (reliable & (abs(av) > minimum_a) & (abs(iv) > minimum_i)
                & np.isfinite(av) & np.isfinite(iv))
        ca, ci = _carrier(a, xy, wavelength), _carrier(b, xy, wavelength)
        if not np.isfinite(ca).all() or not np.isfinite(ci).all():
            raise ValueError("Wavefront analytical carrier exceeds finite numerical range")
        cross = av*np.conjugate(iv)*np.exp(1j*(ca-ci))
        phase_a[first:last] = np.where(keep, np.angle(av*np.exp(1j*ca)), np.nan)
        phase_i[first:last] = np.where(keep, np.angle(iv*np.exp(1j*ci)), np.nan)
        difference[first:last] = np.where(keep, np.angle(cross), np.nan)
        valid[first:last] = keep
        overlap += complex(np.sum(cross[keep]))
        overlap_scale += float(np.sum(abs(cross[keep])))
    _cancel(cancelled)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise ValueError("Wavefront comparison has no common valid support after intensity, coverage and interpolation checks")
    relative_overlap = min(1., float(abs(overlap)/overlap_scale)) if overlap_scale else 0.
    alpha = 0.
    if remove_piston:
        if relative_overlap <= 1e-8:
            raise ValueError("Wavefront global piston is undefined: complex overlap is near zero")
        alpha = float(np.angle(overlap))
        for first in range(0, ashape[0], rows):
            _cancel(cancelled)
            last = min(ashape[0], first+rows)
            difference[first:last] = np.angle(np.exp(1j*(difference[first:last]-alpha)))
    metadata = dict(schema="wavefront-comparison-v1", plane_z_mm=float(za), mode_id=mode_id,
        energy_kev=actual.energy_kev, reference_plane=actual.reference_plane,
        reference_current_a=currents[0], actual_weight=actual.weight_per_reference_electron,
        ideal_weight=ideal.weight_per_reference_electron,
        actual_digest=_selected_digest(actual_checkpoint, actual, amplitude_a, cancelled),
        ideal_digest=_selected_digest(ideal_checkpoint, ideal, amplitude_i, cancelled),
        digest_scope="selected mode payload and checkpoint metadata; no unrelated modes loaded",
        source_provenance="Shared executed source must be verified by the paired execution workflow",
        phase_definition="wrapped transverse envelope plus analytical tilt/curvature carrier, radians; actual minus ideal",
        longitudinal_action_included=False, time_phase_included=False,
        piston_removed=remove_piston, piston_phase_rad=alpha, relative_complex_overlap=relative_overlap,
        piston_support="common valid pixels, complex shape overlap; no tilt or quadratic fit",
        relative_intensity_threshold=relative_intensity_threshold,
        intensity_threshold_reference="each original mode's peak intensity; both must exceed threshold",
        valid_pixel_count=valid_count, valid_fraction=valid_count/count,
        sampling="identical native lattice" if same_grid else "complex-envelope cubic/quintic on actual native affine lattice",
        interpolation_tolerance=interpolation_tolerance,
        maximum_interpolation_relative_error=maximum_interpolation_error,
        covered_pixel_count=covered_count, interpolation_rejected_pixel_count=rejected_interpolation,
        source_edge_guard_cells=0 if same_grid else 3,
        row_chunk_size=rows, estimated_working_bytes=base_bytes+rows*row_bytes,
        physical_weight_renormalized=False, array_order="yx", coordinate_frame="laboratory XY; metres",
        actual_axial_reference=actual.axial_reference, ideal_axial_reference=ideal.axial_reference)
    record = ideal_checkpoint.record
    if isinstance(record, Mapping) and "column_model" in record:
        metadata["ideal_column_model"] = record["column_model"]
    _cancel(cancelled)
    return WavefrontComparison(phase_a, phase_i, difference, valid, a.basis_m, a.origin_m, metadata)
