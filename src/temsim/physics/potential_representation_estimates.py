"""Uncertified, explicitly exploratory carrier representation estimates.

Changing a stored analytical carrier preserves its original grid nodes, but
does not in general preserve the finite Fourier interpolant between nodes.
The estimates here compare that interpolant on denser common SI grids. They
are empirical convergence diagnostics, never continuous-field error bounds.
They reserve L2 allowance alongside, without changing, the strict projection
ledger. There is no invented probability loss, phase fit, or normalization.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
from scipy.fft import next_fast_len

from temsim.cache_memory import retained_memory_inventory
from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.wave_flux import check_lossless_norm
from temsim.physics.wave_grid import (WaveGridBudgetError, WaveSamplingError,
    refine_plane_wave, _inplace_axis_fft)


_SCHEMA = "exploratory-representation-estimates-v1"
_LEGACY_RECEIPT = "common-si-carrier-representation-estimate-v1"
_RECEIPT = "common-si-carrier-representation-estimate-v2"
_SCOPE = "empirical finite Fourier representation estimate; not a certified global error bound"
_BINDING_KEYS = {"source_checkpoint_digest", "source_mode_id", "support_allocation_digest"}
_EPS = np.finfo(float).eps


class RepresentationBudgetError(WaveGridBudgetError):
    """An optional estimate exceeds the unchanged source L2 allowance."""


def _number(value, label, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
            or not np.isrealobj(value) or not math.isfinite(float(value))
            or float(value) < 0 or (positive and float(value) == 0)):
        raise ValueError(label+" must be finite and "+("positive" if positive else "nonnegative"))
    return float(value)


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Representation identities must be lowercase SHA-256 digests")
    return value


def _native_z(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
            or not np.isrealobj(value) or not math.isfinite(float(value))):
        raise ValueError("Representation native Z must be finite")
    return float(value)


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Representation prefix counts must be nonnegative integers")
    return value


def _support(value):
    return value if type(value) is NumericalSupportLedger else NumericalSupportLedger.from_dict(value)


def _binding(value):
    if value is None:
        return None
    if (not isinstance(value, Mapping) or set(value) != _BINDING_KEYS
            or any(type(item) is not str or not item or item.strip() != item for item in value.values())):
        raise ValueError("Representation history needs its exact source, mode and allocation binding")
    _digest(value["source_checkpoint_digest"])
    _digest(value["support_allocation_digest"])
    return freeze_json(value)


def _prefix(strict, count):
    if count > len(strict.events):
        raise ValueError("Representation history refers to a missing strict support prefix")
    return json_digest(NumericalSupportLedger(strict.maximum_l2, strict.maximum_probability,
                                            strict.events[:count]).to_dict())


def _shape(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(n) is not int or n < 2 for n in value)):
        raise ValueError("Representation probe shapes must be integer pairs")
    return tuple(value)


def _validate_receipt(receipt):
    required = {"schema", "certified", "error_scope", "normalization_applied", "phase_alignment_applied",
        "probability_loss_charged", "z_m", "mode_weight", "wavelength_m", "original_wave_digest",
        "recharted_wave_digest", "original_shape", "recharted_shape", "probes", "maximum_probe_l2",
        "probe_spread_l2", "roundoff_floor_l2", "estimated_l2_per_reference", "convergence_relative_limit",
        "input_probability", "output_probability", "norm_roundoff_per_reference"}
    mixed = isinstance(receipt, Mapping) and receipt.get("schema") == _RECEIPT
    if mixed:
        required |= {"absolute_tolerance_relative_amplitude", "absolute_convergence_tolerance_l2"}
    if (not isinstance(receipt, Mapping) or set(receipt) != required
            or receipt["schema"] not in (_LEGACY_RECEIPT, _RECEIPT)
            or receipt["error_scope"] != _SCOPE or receipt["certified"] is not False
            or receipt["normalization_applied"] is not False or receipt["phase_alignment_applied"] is not False
            or type(receipt["probability_loss_charged"]) not in (int, float)
            or receipt["probability_loss_charged"] != 0):
        raise ValueError("Invalid exploratory representation receipt")
    _native_z(receipt["z_m"])
    for key in ("mode_weight", "maximum_probe_l2", "probe_spread_l2", "roundoff_floor_l2",
                "estimated_l2_per_reference", "input_probability", "output_probability", "norm_roundoff_per_reference"):
        _number(receipt[key], key)
    _number(receipt["wavelength_m"], "Representation wavelength", positive=True)
    if receipt["convergence_relative_limit"] != .05:
        raise ValueError("Unknown empirical representation convergence criterion")
    _digest(receipt["original_wave_digest"])
    _digest(receipt["recharted_wave_digest"])
    old, new = _shape(receipt["original_shape"]), _shape(receipt["recharted_shape"])
    probes = receipt["probes"]
    if not isinstance(probes, (tuple, list)) or len(probes) != 3:
        raise ValueError("Representation estimates need two denser grids and a shifted probe")
    sizes, values = [], []
    for index, probe in enumerate(probes):
        if not isinstance(probe, Mapping) or set(probe) != {"shape", "shift_cells_xy", "weighted_raw_complex_l2"}:
            raise ValueError("Invalid representation quadrature probe")
        shape = _shape(probe["shape"])
        if any(n <= max(a, b) for n, a, b in zip(shape, old, new)):
            raise ValueError("Representation probes must be strictly denser than both source charts")
        if (not isinstance(probe["shift_cells_xy"], (list, tuple))
                or tuple(probe["shift_cells_xy"]) != ((.5, .5) if index == 2 else (0., 0.))):
            raise ValueError("Representation probe shifts do not match the declared rule")
        values.append(_number(probe["weighted_raw_complex_l2"], "Probe L2"))
        sizes.append(shape)
    if sizes[1] != sizes[2] or any(a >= b for a, b in zip(sizes[0], sizes[1])):
        raise ValueError("Representation probe grids must have two distinct increasing resolutions")
    maximum, spread = max(values), max(values)-min(values)
    if receipt["maximum_probe_l2"] != maximum or receipt["probe_spread_l2"] != spread:
        raise ValueError("Representation probe summaries disagree with their measurements")
    absolute = 0.
    if mixed:
        _number(receipt["absolute_convergence_tolerance_l2"], "Absolute representation tolerance")
        if receipt["absolute_tolerance_relative_amplitude"] != 1e-7:
            raise ValueError("Unknown absolute representation convergence criterion")
        absolute = 1e-7*math.sqrt(receipt["mode_weight"])*math.sqrt(receipt["input_probability"])
        if receipt["absolute_convergence_tolerance_l2"] != absolute:
            raise ValueError("Representation absolute tolerance changed its declared rule")
    estimate = math.fsum((maximum, spread, receipt["roundoff_floor_l2"], absolute))
    if receipt["estimated_l2_per_reference"] != estimate:
        raise ValueError("Representation estimate arithmetic is inconsistent")
    if spread > max(receipt["roundoff_floor_l2"], .05*maximum, absolute):
        raise ValueError("Representation receipt did not meet its declared local convergence criterion")
    check_lossless_norm(receipt["input_probability"], receipt["output_probability"],
                        context="Recorded carrier representation change")
    if receipt["norm_roundoff_per_reference"] != receipt["mode_weight"]*abs(
            receipt["output_probability"]-receipt["input_probability"]):
        raise ValueError("Representation norm roundoff does not match the recorded probabilities")
    floor = 256*_EPS*math.sqrt(receipt["mode_weight"])*math.sqrt(max(receipt["input_probability"], receipt["output_probability"]))
    if receipt["roundoff_floor_l2"] != floor:
        raise ValueError("Representation roundoff floor changed its declared rule")
    return freeze_json(receipt)


@dataclass(frozen=True, slots=True)
class RepresentationEstimateLedger:
    """Immutable estimates bound to one unchanged strict source allocation."""
    maximum_l2: float
    maximum_probability: float
    strict_anchor_count: int
    strict_anchor_digest: str
    source_binding: object = None
    events: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "maximum_l2", _number(self.maximum_l2, "Representation source L2 allowance"))
        object.__setattr__(self, "maximum_probability", _number(self.maximum_probability, "Representation source probability allowance"))
        _count(self.strict_anchor_count)
        _digest(self.strict_anchor_digest)
        object.__setattr__(self, "source_binding", _binding(self.source_binding))
        if not isinstance(self.events, (tuple, list)):
            raise ValueError("Representation events must be an ordered sequence")
        events = []
        previous_z, previous_count = -math.inf, self.strict_anchor_count
        for index, event in enumerate(self.events):
            if not isinstance(event, Mapping) or set(event) != {
                    "event_index", "strict_prefix_count", "strict_prefix_digest", "previous_estimates_digest", "receipt"}:
                raise ValueError("Invalid representation history event")
            if type(event["event_index"]) is not int or event["event_index"] != index:
                raise ValueError("Representation history event order changed")
            count = _count(event["strict_prefix_count"])
            if count < previous_count:
                raise ValueError("Representation strict prefix moved backwards")
            _digest(event["strict_prefix_digest"])
            receipt = _validate_receipt(event["receipt"])
            if receipt["z_m"] < previous_z:
                raise ValueError("Representation native plane moved backwards")
            expected = json_digest(self._payload(events))
            if event["previous_estimates_digest"] != expected:
                raise ValueError("Representation history prefix digest changed")
            events.append(freeze_json(event))
            previous_z, previous_count = receipt["z_m"], count
        object.__setattr__(self, "events", tuple(events))
        if self.used_l2_estimate > self.maximum_l2:
            raise ValueError("Representation history exceeds its original source allowance")

    def _payload(self, events):
        return {"schema": _SCHEMA, "certified": False, "error_scope": _SCOPE,
            "maximum_l2": self.maximum_l2, "maximum_probability": self.maximum_probability,
            "strict_anchor_count": self.strict_anchor_count, "strict_anchor_digest": self.strict_anchor_digest,
            "source_binding": thaw_json(self.source_binding), "events": [thaw_json(event) for event in events]}

    @classmethod
    def for_support(cls, strict_ledger, *, source_binding=None):
        strict = _support(strict_ledger)
        return cls(strict.maximum_l2, strict.maximum_probability, len(strict.events),
                   json_digest(strict.to_dict()), source_binding)

    @property
    def used_l2_estimate(self):
        return math.fsum(event["receipt"]["estimated_l2_per_reference"] for event in self.events)

    def to_dict(self):
        return self._payload(self.events)

    @classmethod
    def from_dict(cls, record):
        if (not isinstance(record, Mapping) or set(record) != {"schema", "certified", "error_scope",
                "maximum_l2", "maximum_probability", "strict_anchor_count", "strict_anchor_digest", "source_binding", "events"}
                or record["schema"] != _SCHEMA or record["certified"] is not False or record["error_scope"] != _SCOPE):
            raise ValueError("Invalid exploratory representation history")
        return cls(record["maximum_l2"], record["maximum_probability"], record["strict_anchor_count"],
                   record["strict_anchor_digest"], record["source_binding"], record["events"])

    def require_source_binding(self, binding):
        expected = _binding(binding)
        if self.source_binding != expected:
            raise ValueError("Representation history belongs to another source, mode or allocation")
        return self

    def require_prefix(self, previous_history):
        previous = (previous_history if type(previous_history) is RepresentationEstimateLedger
                    else self.from_dict(previous_history))
        if self._payload(()) != previous._payload(()) or self.events[:len(previous.events)] != previous.events:
            raise ValueError("Representation history does not preserve the accepted prefix")
        return self

    def append(self, receipt, strict_ledger):
        strict = _support(strict_ledger)
        remaining = require_combined_support_allowance(strict, self)
        receipt = _validate_receipt(receipt)
        if receipt["estimated_l2_per_reference"] > remaining:
            raise RepresentationBudgetError("Carrier representation estimate exceeds the remaining original L2 allowance")
        event = {"event_index": len(self.events), "strict_prefix_count": len(strict.events),
            "strict_prefix_digest": json_digest(strict.to_dict()), "previous_estimates_digest": json_digest(self.to_dict()),
            "receipt": thaw_json(receipt)}
        result = type(self)(self.maximum_l2, self.maximum_probability, self.strict_anchor_count,
                           self.strict_anchor_digest, self.source_binding, (*self.events, event))
        require_combined_support_allowance(strict, result)
        return result


def require_combined_support_allowance(strict_ledger, history):
    """Reserve estimates within the original L2 limit, without altering it."""
    strict = _support(strict_ledger)
    history = history if type(history) is RepresentationEstimateLedger else RepresentationEstimateLedger.from_dict(history)
    if (strict.maximum_l2 != history.maximum_l2 or strict.maximum_probability != history.maximum_probability
            or _prefix(strict, history.strict_anchor_count) != history.strict_anchor_digest):
        raise ValueError("Representation history changed the original support allocation or accepted prefix")
    for event in history.events:
        if _prefix(strict, event["strict_prefix_count"]) != event["strict_prefix_digest"]:
            raise ValueError("Representation event lost its accepted strict support prefix")
    used = math.fsum((strict.used_l2, history.used_l2_estimate))
    if used > strict.maximum_l2:
        raise RepresentationBudgetError("Strict projection costs plus representation estimates exceed the original L2 allowance")
    return max(0., strict.maximum_l2-used)


def _cancel(cancelled):
    if cancelled():
        raise InterruptedError("Carrier representation estimate cancelled")


def _wave_digest(wave, cancelled):
    # Rows are immutable CPU storage; hashing never creates another full field.
    digest = sha256()
    for start in range(0, wave.amplitude.shape[0], 64):
        _cancel(cancelled)
        digest.update(memoryview(wave.amplitude[start:start+64]).cast("B"))
    return json_digest({"amplitude_sha256": digest.hexdigest(), "shape": list(wave.amplitude.shape),
        "basis_m": wave.basis_m.tolist(), "origin_m": wave.origin_m.tolist(),
        "curvature_m1": None if wave.curvature_m1 is None else wave.curvature_m1.tolist(),
        "tilt_rad": None if wave.tilt_rad is None else wave.tilt_rad.tolist()})


def _probe_shapes(base, maximum):
    def fast(n):
        result = int(next_fast_len(n))
        while result % 2:
            result = int(next_fast_len(result+1))
        return result
    # Small charts can need several times their node density to resolve the
    # off-node difference. Use the available preparation ceiling, capped at
    # four times the input size; never make tiny inputs allocate a huge grid.
    scale = min(4., maximum/max(base))
    first = tuple(fast(max(n+2, math.ceil(n*max(1.1, .85*scale)))) for n in base)
    second = tuple(next((k for k in range(int(math.floor(n*scale))//2*2, a, -2)
                         if next_fast_len(k) == k), 0) for a, n in zip(first, base))
    if any(a <= n or b <= a for n, a, b in zip(base, first, second)) or max(second) > maximum:
        raise WaveSamplingError("Carrier estimate needs two strictly denser independent grids within the preparation cap")
    return first, second


def _shift_envelope(values, cancelled):
    """Evaluate the same periodic Fourier interpolant at half-cell offsets."""
    shifted = np.array(values, dtype=np.complex128, copy=True, order="C")
    for axis in (0, 1):
        _inplace_axis_fft(shifted, axis, cancelled=cancelled)
        phase = np.exp(1j*np.pi*np.fft.fftfreq(shifted.shape[axis]))
        shifted *= phase[:, None] if axis == 0 else phase[None, :]
        del phase
    for axis in (1, 0):
        _inplace_axis_fft(shifted, axis, inverse=True, cancelled=cancelled)
    return shifted


def _difference(first, second, basis, dq, dt, wavelength, shift, cancelled):
    """Tile the exact relative analytical phase; no fitted piston or tilt."""
    ny, nx = first.shape
    x = np.arange(nx, dtype=float)-nx//2+shift
    totals = []
    for start in range(0, ny, max(1, 65536//nx)):
        _cancel(cancelled)
        stop = min(ny, start+max(1, 65536//nx))
        y = (np.arange(start, stop, dtype=float)-ny//2+shift)[:, None]
        rx = basis[0, 0]*x[None, :]+basis[0, 1]*y
        ry = basis[1, 0]*x[None, :]+basis[1, 1]*y
        action = (.5*dq[0, 0]*rx*rx+.5*(dq[0, 1]+dq[1, 0])*rx*ry
                  +.5*dq[1, 1]*ry*ry+dt[0]*rx+dt[1]*ry)
        difference = second[start:stop]*np.exp((2j*np.pi/wavelength)*action)-first[start:stop]
        value = float(np.vdot(difference, difference).real)
        if not math.isfinite(value):
            raise WaveSamplingError("Carrier representation comparison exceeds finite numerical range")
        totals.append(value)
    total = math.fsum(totals)
    if not math.isfinite(total):
        raise WaveSamplingError("Carrier representation norm exceeds finite numerical range")
    return math.sqrt(max(0., total))


def estimate_carrier_representational_change(original, recharted, wavelength_m, *, z_m,
        mode_weight, numerics, retained_bytes=0, cancelled=lambda: False,
        absolute_quadrature_relative_tolerance=0.):
    """Estimate raw complex-field change using two denser grids and a shift.

    ``numerics`` is an explicit CPU preparation ceiling; it never modifies the
    propagation grid or host limit. The default v1 quadratures must agree to
    five percent of their measured maximum or the roundoff floor. An explicit
    ``absolute_quadrature_relative_tolerance=1e-7`` enables the v2 preview
    criterion. Its entire absolute tolerance is ADDED to the charged estimate,
    even when the relative criterion passes. No other tolerance is accepted.
    This is a locally checked estimate, not a continuous-field certificate.
    The caller must separately retain axial-reference/retarded phase unchanged.
    """
    _cancel(cancelled)
    numerics.validate()
    tolerance = _number(absolute_quadrature_relative_tolerance, "Absolute quadrature relative tolerance")
    if tolerance not in (0., 1e-7):
        raise ValueError("Representation quadrature tolerance must select the original or explicit preview rule")
    if numerics.compute_backend.lower() != "cpu":
        raise ValueError("Carrier representation estimates require explicit CPU preparation")
    if type(retained_bytes) is not int or retained_bytes < 0:
        raise ValueError("Representation retained bytes must be a nonnegative integer")
    wavelength = _number(wavelength_m, "Representation wavelength", positive=True)
    z, weight = _native_z(z_m), _number(mode_weight, "Representation mode weight")
    for wave in (original, recharted):
        if type(wave) is not PlaneWave or type(wave.amplitude) is not np.ndarray:
            raise ValueError("Carrier representation estimates need immutable CPU PlaneWave inputs")
    if not np.array_equal(original.origin_m, recharted.origin_m):
        raise ValueError("Carrier representation estimates require the identical physical origin")
    old_period = original.basis_m*np.asarray(original.amplitude.shape)[::-1]
    new_period = recharted.basis_m*np.asarray(recharted.amplitude.shape)[::-1]
    if np.linalg.norm(old_period-new_period) > 64*_EPS*np.linalg.norm(old_period):
        raise ValueError("Carrier representation estimates require the same physical Fourier period")
    check_lossless_norm(original.probability, recharted.probability, context="Carrier representation change")
    shapes = _probe_shapes(tuple(max(a, b) for a, b in zip(original.amplitude.shape,
                                                         recharted.amplitude.shape)), numerics.maximum_pixels)
    owners = retained_memory_inventory(original.amplitude, recharted.amplitude)
    outside = retained_bytes+sum(owners.values())
    # 256 B/cell covers both refined immutable fields, their private shifted
    # copies, tiled FFT/phase scratch and publication overlap. The additional
    # 8 MiB and axis reserve cover the refinement FFT implementation. Source
    # owners are counted once separately; no previous probe remains live.
    for shape in shapes:
        numerics.check_working_memory(shape, working_bytes=256*math.prod(shape)+8*1024**2+1024*max(shape),
                                      retained_bytes=outside)
    q = lambda wave: np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = lambda wave: np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    dq, dt = q(recharted)-q(original), tilt(recharted)-tilt(original)
    probes = []
    for index, shape in enumerate(shapes):
        _cancel(cancelled)
        first = refine_plane_wave(original, shape, numerics=numerics, retained_bytes=outside,
                                  cancelled=cancelled)
        second = refine_plane_wave(recharted, shape, numerics=numerics,
                                   retained_bytes=outside+first.amplitude.nbytes, cancelled=cancelled)
        error = math.sqrt(weight)*_difference(first.amplitude, second.amplitude, first.basis_m,
                                             dq, dt, wavelength, 0., cancelled)
        probes.append({"shape": list(shape), "shift_cells_xy": [0., 0.], "weighted_raw_complex_l2": error})
        if index == 1:
            a = _shift_envelope(first.amplitude, cancelled)
            b = _shift_envelope(second.amplitude, cancelled)
            error = math.sqrt(weight)*_difference(a, b, first.basis_m, dq, dt, wavelength, .5, cancelled)
            probes.append({"shape": list(shape), "shift_cells_xy": [.5, .5], "weighted_raw_complex_l2": error})
            del a, b
        del first, second
    values = [probe["weighted_raw_complex_l2"] for probe in probes]
    maximum, spread = max(values), max(values)-min(values)
    floor = 256*_EPS*math.sqrt(weight)*math.sqrt(max(original.probability, recharted.probability))
    absolute = tolerance*math.sqrt(weight)*math.sqrt(original.probability)
    if spread > max(floor, .05*maximum, absolute):
        raise WaveSamplingError("Carrier representation quadrature did not locally converge on the admitted grids")
    receipt = {"schema": _RECEIPT if tolerance else _LEGACY_RECEIPT, "certified": False, "error_scope": _SCOPE,
        "normalization_applied": False, "phase_alignment_applied": False, "probability_loss_charged": 0.,
        "z_m": z, "mode_weight": weight, "wavelength_m": wavelength,
        "original_wave_digest": _wave_digest(original, cancelled),
        "recharted_wave_digest": _wave_digest(recharted, cancelled),
        "original_shape": list(original.amplitude.shape), "recharted_shape": list(recharted.amplitude.shape),
        "probes": probes, "maximum_probe_l2": maximum, "probe_spread_l2": spread,
        "roundoff_floor_l2": floor, "estimated_l2_per_reference": math.fsum((maximum, spread, floor, absolute)),
        "convergence_relative_limit": .05, "input_probability": original.probability,
        "output_probability": recharted.probability,
        "norm_roundoff_per_reference": weight*abs(recharted.probability-original.probability)}
    if tolerance:
        receipt.update(absolute_tolerance_relative_amplitude=tolerance,
                       absolute_convergence_tolerance_l2=absolute)
    _cancel(cancelled)
    return thaw_json(_validate_receipt(receipt))
