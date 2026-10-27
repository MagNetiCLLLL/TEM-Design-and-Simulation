"""Strict, explicit numerical support projection for controlled potential waves.

This is a lossy numerical approximation, never an aperture or renormalization.
The receipt bounds the initial complex-field projection only. Different finite
FFT domains still require independent propagated complex-field convergence.
"""
from dataclasses import dataclass, replace
from collections.abc import Mapping
import math

import numpy as np

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.multiplane_wave import PlaneWave


_CROP_KIND = "numerical_support_projection"
_TAPER_KIND = "numerical_boundary_taper"
_SPECTRAL_KIND = "numerical_spectral_projection"
_CROP_SCOPE = "initial projection only; subsequent different-domain propagation needs independent convergence"
_TAPER_SCOPE = "initial same-grid boundary projection only; subsequent propagation needs independent convergence"
_PRESSURE_SCHEMA = "remaining-support-pressure-allocation-v1"
_SPATIAL_PRESSURE_SCHEMA = "remaining-spatial-support-pressure-allocation-v1"
_SPATIAL_PRESSURE_FAMILY = "centered_quintic_crop_remaining_share_v1"
SPATIAL_PRESSURE_SHARES = (.01, .02, .05, .1, .2, .5, 1.)
_BOUNDARY_TRANSITIONS = (2, 4, 8, 16)


def _ledger_payload(maximum_l2, maximum_probability, events):
    version = ("v6" if any(event["kind"] == _CROP_KIND and "pressure_allocation" in event for event in events) else
               "v5" if any(event["kind"] == _TAPER_KIND and event["transition_cells"] != 2 for event in events) else
               "v4" if any("pressure_allocation" in event for event in events) else
               "v3" if any(event["kind"] == _SPECTRAL_KIND for event in events) else
               "v2" if any(event["kind"] == _TAPER_KIND for event in events) else "v1")
    return {"schema": "controlled-numerical-support-ledger-"+version,
            "maximum_l2": maximum_l2, "maximum_probability": maximum_probability,
            "events": [thaw_json(event) for event in events]}


def _pressure_context(value, input_shape, *, spatial=False):
    """Validate reproducible resource targets without inventing a field gate."""
    required = {"schema", "stage", "failure_kind", "input_shape", "maximum_host_working_bytes",
        "maximum_device_working_bytes", "maximum_pixels", "retained_host_bytes",
        "operator_memory_policy", "requested_backend", "growth_guard_scale", "candidate_family"}
    if (not isinstance(value, Mapping) or not required <= set(value)
            or set(value)-required-({"source_failure_shape", "previous_shape"} if spatial else {"source_failure_shape"})
            or value["schema"] != "wave-resource-pressure-v1"
            or value["failure_kind"] not in ("WaveGridBudgetError", "WaveMemoryBudgetError")
            or value["candidate_family"] != (_SPATIAL_PRESSURE_FAMILY if spatial else "fixed_aspect_even_fft_friendly_v1")):
        raise ValueError("Invalid spectral pressure resource context")
    if (type(value["stage"]) is not str or not value["stage"]
            or value["stage"].strip() != value["stage"]):
        raise ValueError("Spectral pressure requires an explicit operator stage")
    for key in ("input_shape", "source_failure_shape", "previous_shape"):
        if key in value and (not isinstance(value[key], (tuple, list)) or len(value[key]) != 2
                or any(type(count) is not int or count <= 0 for count in value[key])):
            raise ValueError("Spectral pressure shapes must be positive integer pairs")
    if (not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2
            or any(type(count) is not int or count <= 0 for count in input_shape)
            or tuple(value["input_shape"]) != tuple(input_shape)):
        raise ValueError("Spectral pressure context differs from its input lattice")
    for key in ("maximum_host_working_bytes", "maximum_device_working_bytes", "maximum_pixels", "retained_host_bytes"):
        if type(value[key]) is not int or value[key] < (0 if key == "retained_host_bytes" else 1):
            raise ValueError("Spectral pressure resource limits must be explicit nonnegative integers")
    growth = _number(value["growth_guard_scale"], "Pressure sampling growth guard", True)
    if growth < 1.:
        raise ValueError("Spectral pressure sampling growth cannot be smaller than one")
    if spatial and growth != 1.:
        raise ValueError("Spatial pressure cannot inherit a different-domain sampling forecast")
    if value["operator_memory_policy"] not in (
            "carrier-residual-host-512-plus-64MiB/device-448",
            "carrier-validation-256-with-residual-admission"):
        raise ValueError("Unknown spectral pressure operator memory policy")
    from temsim.physics.wave_device import normalise_backend
    if type(value["requested_backend"]) is not str or normalise_backend(value["requested_backend"]) != value["requested_backend"]:
        raise ValueError("Spectral pressure backend must be an explicit normalized policy")
    return value


def _number(value, name, positive=False, signed=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isreal(value):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value) or (value < 0 and not signed) or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _interval(low, high, size):
    center = size//2
    radius = max(16, center-low+6, high+7-center)
    if 2*radius >= size or center-radius < 0 or center+radius > size:
        return 0, size
    return center-radius, center+radius


def compact_controlled_support(wave, *, wavelength_m, absolute_l2_budget, absolute_probability_budget):
    """Project only tails admitted by BOTH absolute budgets on the input field.

    Left and right tails are summed independently. No total-minus-retained
    subtraction or absolute epsilon allowance is used at any budget gate.
    Returns the unchanged input and None when no useful crop is admitted.
    """
    if type(wave) is not PlaneWave or not isinstance(wave.amplitude, np.ndarray):
        raise ValueError("Controlled support projection requires a CPU PlaneWave")
    wavelength = _number(wavelength_m, "Wavelength", True)
    l2_budget = _number(absolute_l2_budget, "Absolute support L2 budget")
    probability_budget = _number(absolute_probability_budget, "Absolute support probability budget")
    if l2_budget == 0 or probability_budget == 0 or wave.probability == 0:
        return wave, None
    with np.errstate(over="ignore", under="ignore"):
        probability = abs(wave.amplitude)**2
    if not np.isfinite(probability).all() or not math.isfinite(wave.probability):
        raise ValueError("Support projection requires finite cell and total probabilities")
    if np.any((wave.amplitude != 0.) & (probability < np.finfo(float).tiny)):
        return wave, None  # Squared subnormal tails cannot certify this receipt.
    target = min(probability_budget, l2_budget*l2_budget, .1*wave.probability)
    target *= 1.-32*np.finfo(float).eps  # Reserve relative rounding room, never increase a budget.
    if target == 0:
        return wave, None
    cores, intervals = [], []
    for axis, size in enumerate(wave.amplitude.shape):
        marginal = np.sum(probability, axis=1-axis)
        left, right = np.cumsum(marginal), np.cumsum(marginal[::-1])
        low = min(size-1, int(np.searchsorted(left, target/4., side="right")))
        removed_right = min(size-1, int(np.searchsorted(right, target/4., side="right")))
        high = size-removed_right-1
        cores.append((low, high))
        intervals.append(_interval(low, high, size))
    shape = tuple(stop-start for start, stop in intervals)
    if math.prod(shape) > .75*wave.amplitude.size:
        return wave, None
    (y0, y1), (x0, x1) = intervals
    windows = []
    for (start, stop), (low, high) in zip(intervals, cores):
        coordinate = np.arange(start, stop, dtype=float)
        t = np.minimum(np.maximum(np.maximum(low-coordinate, coordinate-high), 0.)/4., 1.)
        windows.append(1.-t**3*(10.+t*(-15.+6.*t)))
    window = windows[0][:, None]*windows[1][None, :]
    inside = probability[y0:y1, x0:x1]
    outside = math.fsum(float(np.sum(values)) for values in (
        probability[:y0], probability[y1:], probability[y0:y1, :x0], probability[y0:y1, x1:]))
    difference = 1.-window
    lost = math.fsum((outside, float(np.sum(inside*difference*(1.+window)))))
    squared_error = math.fsum((outside, float(np.sum(inside*difference*difference))))
    l2_error = math.sqrt(squared_error)
    if lost > probability_budget or l2_error > l2_budget:
        return wave, None  # Keep the entire input when the exact receipt does not fit.
    # Symmetric index intervals preserve the exact original coordinates and
    # carriers. No large phase is recomputed; unchanged core samples remain
    # bitwise unchanged even for large curvature or tiny antisymmetric noise.
    amplitude = wave.amplitude[y0:y1, x0:x1]*window
    result = replace(wave, amplitude=amplitude)
    return result, {"kind": "numerical_support_projection", "physical_absorption": False,
        "probability_repair": False, "absolute_probability_loss": lost,
        "absolute_l2_error": l2_error, "input_probability": wave.probability,
        "output_probability": result.probability, "from_shape": list(wave.amplitude.shape),
        "to_shape": list(shape), "crop_indices_yx": [[y0, y1], [x0, x1]],
        "core_indices_yx": [list(core) for core in cores], "origin_shift_m": [0., 0.],
        "carrier_constant_action_m": 0., "transition_cells": 4,
        "error_scope": "initial projection only; subsequent different-domain propagation needs independent convergence"}


@dataclass(frozen=True, slots=True)
class NumericalSupportLedger:
    """One controlled mode's cumulative errors per reference electron.

    Historical attempts spend at most one percent of the original budgets.
    New spectral resource-pressure receipts may use the recorded remaining
    allowances, without changing old events or renewing the source limits.
    Actual receipts, not their allocations, are accumulated. Unit-shape
    factorization is handled by supplying the current absolute mode weight.
    """
    maximum_l2: float
    maximum_probability: float
    events: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "maximum_l2", _number(self.maximum_l2, "Total support L2 budget"))
        object.__setattr__(self, "maximum_probability", _number(self.maximum_probability, "Total support probability budget"))
        if not isinstance(self.events, (tuple, list)) or any(not isinstance(event, Mapping) for event in self.events):
            raise ValueError("Support events must be a sequence of receipts")
        events = tuple(freeze_json(thaw_json(event)) for event in self.events)
        required = {"kind", "physical_absorption", "probability_repair", "absolute_probability_loss",
            "absolute_l2_error", "input_probability", "output_probability", "from_shape", "to_shape",
            "crop_indices_yx", "core_indices_yx", "origin_shift_m", "carrier_constant_action_m",
            "transition_cells", "error_scope", "event_index", "z_m", "mode_weight",
            "l2_per_reference", "probability_per_reference"}
        for index, event in enumerate(events):
            spectral = event.get("kind") == _SPECTRAL_KIND
            pressure = "pressure_allocation" in event
            event_required = ((required-{"crop_indices_yx", "core_indices_yx", "transition_cells"})
                | {"retained_frequency_intervals_yx", "guard_nyquist_fraction"}) if spectral else required
            if pressure:
                if not spectral and event.get("kind") != _CROP_KIND:
                    raise ValueError("Only spectral or spatial resource-pressure receipts may use remaining allocations")
                event_required = event_required | {"pressure_allocation"}
            if set(event) != event_required:
                raise ValueError("Support receipt has missing or unknown fields")
            if event.get("kind") not in (_CROP_KIND, _TAPER_KIND, _SPECTRAL_KIND) or event.get("physical_absorption") is not False:
                raise ValueError("Support ledger accepts numerical projection receipts only")
            if event["probability_repair"] is not False or type(event["event_index"]) is not int or event["event_index"] != index:
                raise ValueError("Support event identities must be consecutive and unnormalized")
            weight = _number(event["mode_weight"], "Receipt mode weight", True)
            _number(event["z_m"], "Receipt Z", signed=True)
            loss = _number(event["absolute_probability_loss"], "Receipt probability loss")
            error = _number(event["absolute_l2_error"], "Receipt L2 error")
            before = _number(event["input_probability"], "Receipt input probability", True)
            after = _number(event["output_probability"], "Receipt output probability")
            reference_error = _number(event["l2_per_reference"], "Reference L2 error")
            reference_loss = _number(event["probability_per_reference"], "Reference probability loss")
            rounding = 32*np.finfo(float).eps
            if (loss > before or after > before*(1.+rounding)
                    or abs((before-loss)-after) > rounding*before
                    or error*error > loss*(1.+rounding)
                    or loss > 2*math.sqrt(before)*error*(1.+rounding)):
                raise ValueError("Support receipt contains invalid probability accounting")
            if (reference_error != math.sqrt(weight)*error or reference_loss != weight*loss
                    or (error > 0 and reference_error == 0) or (loss > 0 and reference_loss == 0)):
                raise ValueError("Support receipt reference-weight conversion is inconsistent")
            if pressure:
                allocation = event["pressure_allocation"]
                fields = {"schema", "previous_ledger_digest", "remaining_l2_per_reference",
                          "remaining_probability_per_reference", "resource_context"}
                if not spectral:
                    fields = fields | {"allocation_fraction"}
                if (not isinstance(allocation, Mapping) or set(allocation) != fields
                        or allocation["schema"] != (_PRESSURE_SCHEMA if spectral else _SPATIAL_PRESSURE_SCHEMA)):
                    raise ValueError("Invalid remaining spectral pressure allocation")
                prefix = events[:index]
                previous_digest = json_digest(_ledger_payload(self.maximum_l2, self.maximum_probability, prefix))
                remaining_l2 = self.maximum_l2-math.fsum(float(row["l2_per_reference"]) for row in prefix)
                remaining_probability = self.maximum_probability-math.fsum(float(row["probability_per_reference"]) for row in prefix)
                if (allocation["previous_ledger_digest"] != previous_digest
                        or _number(allocation["remaining_l2_per_reference"], "Remaining pressure L2") != remaining_l2
                        or _number(allocation["remaining_probability_per_reference"], "Remaining pressure probability") != remaining_probability):
                    raise ValueError("Spectral pressure allocation changed or reset its preceding ledger")
                _pressure_context(allocation["resource_context"], event["from_shape"], spatial=not spectral)
                fraction = 1. if spectral else _number(allocation["allocation_fraction"], "Spatial pressure allocation share", True)
                if not spectral and fraction not in SPATIAL_PRESSURE_SHARES:
                    raise ValueError("Spatial pressure requires a declared bounded allocation share")
                if reference_error > fraction*remaining_l2 or reference_loss > fraction*remaining_probability:
                    raise ValueError("Spectral pressure receipt exceeds its remaining cumulative allocation")
            elif (event["l2_per_reference"] > self.maximum_l2/100.
                    or event["probability_per_reference"] > self.maximum_probability/100.):
                raise ValueError("Support receipt exceeds its individual allocation")
            for key in ("from_shape", "to_shape"):
                if not isinstance(event[key], tuple) or len(event[key]) != 2 or any(type(v) is not int or v < 1 for v in event[key]):
                    raise ValueError("Support receipt shapes must be positive integer pairs")
            if spectral:
                from temsim.physics.potential_wave_resampling import SPECTRAL_PROJECTION_SCOPE
                guard = _number(event["guard_nyquist_fraction"], "Spectral resampling guard", True)
                intervals = event["retained_frequency_intervals_yx"]
                if (guard >= .8 or event["error_scope"] != SPECTRAL_PROJECTION_SCOPE
                        or event["origin_shift_m"] != (0., 0.) or event["carrier_constant_action_m"] != 0.
                        or not isinstance(intervals, tuple) or len(intervals) != 2
                        or abs(error*error-loss) > rounding*max(loss, error*error)):
                    raise ValueError("Invalid same-period spectral projection receipt")
                for size, count, interval in zip(event["from_shape"], event["to_shape"], intervals):
                    if (count > size or not isinstance(interval, tuple) or len(interval) != 2
                            or any(type(v) is not int for v in interval)
                            or not -(size//2) <= interval[0] <= 0 <= interval[1] <= (size-1)//2
                            or (count == size and interval != (-(size//2), (size-1)//2))
                            or (count < size and (count < 32 or count % 2
                                or max(abs(v) for v in interval) >= guard*count/2))):
                        raise ValueError("Invalid retained spectral interval or sampling guard")
                if event["to_shape"] == event["from_shape"]:
                    raise ValueError("Spectral resampling must reduce the represented lattice")
                if pressure:
                    from temsim.physics.potential_wave_resampling import (
                        GUARD_NYQUIST_FRACTION, _even_fast_count,
                    )
                    source, target = event["from_shape"], event["to_shape"]
                    longest_axis = max(range(2), key=lambda axis: source[axis])
                    count = target[longest_axis]
                    expected = tuple(min(size, _even_fast_count(size*count/max(source))) for size in source)
                    complete_band = tuple((-(size//2), (size-1)//2) if kept == size else
                        (-(math.ceil(.5*guard*kept)-1), math.ceil(.5*guard*kept)-1)
                        for size, kept in zip(source, target))
                    if (guard != GUARD_NYQUIST_FRACTION or _even_fast_count(count) != count
                            or target != expected or math.prod(target) > .95*math.prod(source)
                            or intervals != complete_band):
                        raise ValueError("Pressure receipt must preserve its complete resource-target passband")
                continue
            if any(not isinstance(event[key], tuple) or len(event[key]) != 2
                   for key in ("crop_indices_yx", "core_indices_yx")):
                raise ValueError("Support receipt needs two crop/core intervals")
            for size, count, crop, core in zip(event["from_shape"], event["to_shape"],
                                              event["crop_indices_yx"], event["core_indices_yx"]):
                if (not isinstance(crop, tuple) or not isinstance(core, tuple)
                        or len(crop) != 2 or len(core) != 2 or any(type(v) is not int for v in (*crop, *core))
                        or not 0 <= crop[0] <= core[0] <= core[1] < crop[1] <= size
                        or crop[1]-crop[0] != count or crop[0]+count//2 != size//2):
                    raise ValueError("Support receipt crop geometry is inconsistent")
            taper = event["kind"] == _TAPER_KIND
            if (event["origin_shift_m"] != (0., 0.) or event["carrier_constant_action_m"] != 0.
                    or type(event["transition_cells"]) is not int
                    or (event["transition_cells"] not in _BOUNDARY_TRANSITIONS if taper else
                        event["transition_cells"] != 4)):
                raise ValueError("Controlled support receipts cannot rewrite phase carriers")
            width = event["transition_cells"]
            if taper and (event["from_shape"] != event["to_shape"] or any(
                    size < 2*width+1 or crop != (0, size) or core != (width, size-1-width)
                    for size, crop, core in zip(event["from_shape"], event["crop_indices_yx"], event["core_indices_yx"]))):
                raise ValueError("Boundary taper receipts must preserve the full grid and declared quintic window")
            if pressure and not spectral and (math.prod(event["to_shape"]) > .75*math.prod(event["from_shape"])
                    or any(crop != _interval(core[0], core[1], size)
                           for size, crop, core in zip(event["from_shape"], event["crop_indices_yx"], event["core_indices_yx"]))):
                raise ValueError("Spatial pressure must preserve its centered finite-window candidate family")
            if event["error_scope"] != (_TAPER_SCOPE if taper else _CROP_SCOPE):
                raise ValueError("Support receipt must retain its projection-only error scope")
        object.__setattr__(self, "events", events)
        if self.used_l2 > self.maximum_l2 or self.used_probability > self.maximum_probability:
            raise ValueError("Cumulative numerical support receipts exceed the declared total budget")

    @property
    def used_l2(self):
        return math.fsum(float(event["l2_per_reference"]) for event in self.events)

    @property
    def used_probability(self):
        return math.fsum(float(event["probability_per_reference"]) for event in self.events)

    def to_dict(self):
        return _ledger_payload(self.maximum_l2, self.maximum_probability, self.events)

    @classmethod
    def from_dict(cls, record):
        if (not isinstance(record, Mapping)
                or set(record) != {"schema", "maximum_l2", "maximum_probability", "events"}
                or record["schema"] not in ("controlled-numerical-support-ledger-v1", "controlled-numerical-support-ledger-v2",
                                            "controlled-numerical-support-ledger-v3", "controlled-numerical-support-ledger-v4",
                                            "controlled-numerical-support-ledger-v5", "controlled-numerical-support-ledger-v6")):
            raise ValueError("Invalid numerical support ledger schema")
        ledger = cls(record["maximum_l2"], record["maximum_probability"], record["events"])
        if record["schema"] != ledger.to_dict()["schema"]:
            raise ValueError("Support ledger schema does not describe its projection kinds")
        return ledger

    def compact(self, wave, *, wavelength_m, z_m, mode_weight=1.):
        weight = _number(mode_weight, "Support mode weight")
        z = _number(z_m, "Support projection Z", signed=True)
        if weight == 0:
            return wave, self
        l2 = min(self.maximum_l2/100., self.maximum_l2-self.used_l2)
        probability = min(self.maximum_probability/100., self.maximum_probability-self.used_probability)
        result, receipt = compact_controlled_support(wave, wavelength_m=wavelength_m,
            absolute_l2_budget=l2/math.sqrt(weight), absolute_probability_budget=probability/weight)
        if receipt is None:
            return wave, self
        if ((receipt["absolute_l2_error"] > 0 and math.sqrt(weight)*receipt["absolute_l2_error"] == 0)
                or (receipt["absolute_probability_loss"] > 0 and weight*receipt["absolute_probability_loss"] == 0)):
            return wave, self  # An underflowed reference cost cannot authorize a projection.
        event = {**receipt, "event_index": len(self.events), "z_m": z, "mode_weight": weight,
            "l2_per_reference": math.sqrt(weight)*receipt["absolute_l2_error"],
            "probability_per_reference": weight*receipt["absolute_probability_loss"]}
        return result, replace(self, events=(*self.events, event))

    def resample_spectrum(self, wave, *, z_m, mode_weight=1., numerics=None,
                          retained_bytes=0, cancelled=lambda: False):
        """Propose a weighted complex projection within the SAME source ledger.

        Spatial and spectral projections share the original cumulative limits.
        Publication remains the caller's transaction after physical admission.
        A segment boundary never reissues the original source allocation.
        """
        from temsim.physics.potential_wave_resampling import resample_controlled_spectrum
        from temsim.physics.wave_grid import WaveGridNumerics
        weight = _number(mode_weight, "Resampling mode weight")
        z = _number(z_m, "Resampling Z", signed=True)
        if weight == 0.:
            return wave, self
        l2 = min(self.maximum_l2/100., max(0., self.maximum_l2-self.used_l2))
        probability = min(self.maximum_probability/100., max(0., self.maximum_probability-self.used_probability))
        result, receipt = resample_controlled_spectrum(wave,
            absolute_l2_budget=l2/math.sqrt(weight), absolute_probability_budget=probability/weight,
            numerics=WaveGridNumerics() if numerics is None else numerics,
            retained_bytes=retained_bytes, cancelled=cancelled)
        if receipt is None:
            return wave, self
        reference_loss = weight*receipt["absolute_probability_loss"]
        reference_error = math.sqrt(weight)*receipt["absolute_l2_error"]
        if ((receipt["absolute_probability_loss"] > 0. and reference_loss == 0.)
                or (receipt["absolute_l2_error"] > 0. and reference_error == 0.)):
            return wave, self
        event = {**receipt, "event_index": len(self.events), "z_m": z, "mode_weight": weight,
                 "l2_per_reference": reference_error, "probability_per_reference": reference_loss}
        return result, replace(self, events=(*self.events, event))

    def compact_pressure(self, wave, *, wavelength_m, z_m, resource_context,
                         allocation_fraction, mode_weight=1., numerics=None,
                         retained_bytes=0, cancelled=lambda: False):
        """Propose one centered window from a fixed share of existing remaining errors.

        This does not update the source ledger or promise physical admission.
        The caller tries the declared shares from least to most destructive on
        the SAME original entrance and ledger. Only a complete operator success
        may publish the selected receipt. Historical compact() stays unchanged.
        """
        from temsim.physics.wave_grid import WaveGridNumerics
        if type(wave) is not PlaneWave or not isinstance(wave.amplitude, np.ndarray):
            raise ValueError("Spatial pressure projection requires a CPU PlaneWave")
        if type(retained_bytes) is not int or retained_bytes < 0:
            raise ValueError("Spatial pressure retained bytes must be a nonnegative integer")
        fraction = _number(allocation_fraction, "Spatial pressure allocation share", True)
        if fraction not in SPATIAL_PRESSURE_SHARES:
            raise ValueError("Spatial pressure requires a declared bounded allocation share")
        numeric = WaveGridNumerics() if numerics is None else numerics
        numeric.validate()
        context = _pressure_context(resource_context, wave.amplitude.shape, spatial=True)
        for key, expected in (("maximum_host_working_bytes", numeric.maximum_working_bytes),
                ("maximum_device_working_bytes", numeric.maximum_device_working_bytes),
                ("maximum_pixels", numeric.maximum_pixels), ("retained_host_bytes", retained_bytes)):
            if context[key] != expected:
                raise ValueError("Spatial pressure context differs from its actual numerical resource limits")
        weight = _number(mode_weight, "Spatial pressure mode weight")
        z = _number(z_m, "Spatial pressure Z", signed=True)
        remaining_l2 = self.maximum_l2-self.used_l2
        remaining_probability = self.maximum_probability-self.used_probability
        if cancelled():
            raise InterruptedError("Spatial pressure projection cancelled")
        if weight == 0. or remaining_l2 == 0. or remaining_probability == 0.:
            return wave, self
        numeric.check_working_memory(wave.amplitude.shape,
            working_bytes=256*wave.amplitude.size, retained_bytes=retained_bytes)
        result, receipt = compact_controlled_support(wave, wavelength_m=wavelength_m,
            absolute_l2_budget=fraction*remaining_l2/math.sqrt(weight),
            absolute_probability_budget=fraction*remaining_probability/weight)
        if cancelled():
            raise InterruptedError("Spatial pressure projection cancelled")
        if receipt is None:
            return wave, self
        reference_loss = weight*receipt["absolute_probability_loss"]
        reference_error = math.sqrt(weight)*receipt["absolute_l2_error"]
        if ((receipt["absolute_probability_loss"] > 0. and reference_loss == 0.)
                or (receipt["absolute_l2_error"] > 0. and reference_error == 0.)):
            return wave, self
        allocation = {"schema": _SPATIAL_PRESSURE_SCHEMA,
            "previous_ledger_digest": json_digest(self.to_dict()),
            "remaining_l2_per_reference": remaining_l2,
            "remaining_probability_per_reference": remaining_probability,
            "allocation_fraction": fraction, "resource_context": thaw_json(context)}
        event = {**receipt, "event_index": len(self.events), "z_m": z, "mode_weight": weight,
            "l2_per_reference": reference_error, "probability_per_reference": reference_loss,
            "pressure_allocation": allocation}
        return result, replace(self, events=(*self.events, event))

    def resample_spectrum_pressure(self, wave, *, z_m, admissible_shape, resource_context,
                                   mode_weight=1., numerics=None, retained_bytes=0,
                                   cancelled=lambda: False):
        """Target an affordable input lattice within the unchanged remaining budget.

        A typed resource failure and its operator-specific resource predicate
        belong to the caller. This method records that explicit target and the
        exact preceding ledger, never allocates a new source allowance. The
        returned proposal is still private until physical admission succeeds.
        """
        from temsim.physics.potential_wave_resampling import resample_controlled_spectrum
        from temsim.physics.wave_grid import WaveGridNumerics
        if type(wave) is not PlaneWave or not isinstance(wave.amplitude, np.ndarray):
            raise ValueError("Spectral pressure resampling requires a CPU PlaneWave")
        if not callable(admissible_shape):
            raise ValueError("Spectral pressure admission must be callable")
        if type(retained_bytes) is not int or retained_bytes < 0:
            raise ValueError("Spectral pressure retained bytes must be a nonnegative integer")
        numeric = WaveGridNumerics() if numerics is None else numerics
        numeric.validate()
        context = _pressure_context(resource_context, wave.amplitude.shape)
        for key, expected in (("maximum_host_working_bytes", numeric.maximum_working_bytes),
                ("maximum_device_working_bytes", numeric.maximum_device_working_bytes),
                ("maximum_pixels", numeric.maximum_pixels), ("retained_host_bytes", retained_bytes)):
            if context[key] != expected:
                raise ValueError("Spectral pressure context differs from its actual numerical resource limits")
        weight = _number(mode_weight, "Pressure resampling mode weight")
        z = _number(z_m, "Pressure resampling Z", signed=True)
        remaining_l2 = self.maximum_l2-self.used_l2
        remaining_probability = self.maximum_probability-self.used_probability
        if weight == 0. or remaining_l2 == 0. or remaining_probability == 0.:
            return wave, self
        result, receipt = resample_controlled_spectrum(wave,
            absolute_l2_budget=remaining_l2/math.sqrt(weight),
            absolute_probability_budget=remaining_probability/weight,
            numerics=numeric, retained_bytes=retained_bytes, cancelled=cancelled,
            admissible_shape=admissible_shape)
        if receipt is None:
            return wave, self
        reference_loss = weight*receipt["absolute_probability_loss"]
        reference_error = math.sqrt(weight)*receipt["absolute_l2_error"]
        if ((receipt["absolute_probability_loss"] > 0. and reference_loss == 0.)
                or (receipt["absolute_l2_error"] > 0. and reference_error == 0.)):
            return wave, self
        allocation = {"schema": _PRESSURE_SCHEMA, "previous_ledger_digest": json_digest(self.to_dict()),
            "remaining_l2_per_reference": remaining_l2,
            "remaining_probability_per_reference": remaining_probability,
            "resource_context": thaw_json(context)}
        event = {**receipt, "event_index": len(self.events), "z_m": z, "mode_weight": weight,
            "l2_per_reference": reference_error, "probability_per_reference": reference_loss,
            "pressure_allocation": allocation}
        return result, replace(self, events=(*self.events, event))

    def taper_boundary(self, wave, *, z_m, mode_weight=1., transition_cells=2):
        """Apply an explicit quintic edge window within existing budgets.

        This explicit operation keeps the entire finite lattice and carrier.
        It does not authorize boundary retries by itself. The caller must
        recheck the actual boundary and spectral admission gates. A rejected
        candidate returns the original wave and ledger, without normalization.
        The historical two-cell default is unchanged. Wider windows emit v5;
        preceding events retain their original schema semantics and digests.
        """
        if type(transition_cells) is not int or transition_cells not in _BOUNDARY_TRANSITIONS:
            raise ValueError("Boundary transition must be an explicit integer: 2, 4, 8 or 16")
        weight = _number(mode_weight, "Support mode weight")
        z = _number(z_m, "Support projection Z", signed=True)
        l2_cap = min(self.maximum_l2/100., self.maximum_l2-self.used_l2)
        probability_cap = min(self.maximum_probability/100., self.maximum_probability-self.used_probability)
        if weight == 0. or l2_cap == 0. or probability_cap == 0.:
            return wave, self
        if type(wave) is not PlaneWave or not isinstance(wave.amplitude, np.ndarray):
            raise ValueError("Controlled boundary taper requires a CPU PlaneWave")
        if min(wave.amplitude.shape) < 2*transition_cells+1 or wave.probability == 0.:
            return wave, self
        windows = []
        for size in wave.amplitude.shape:
            index = np.arange(size, dtype=float)
            t = np.minimum(np.minimum(index, size-1-index)/float(transition_cells), 1.)
            windows.append(t*t*t*(10.+t*(-15.+6.*t)))
        window = windows[0][:, None]*windows[1][None, :]
        changed = window != 1.
        amplitude = wave.amplitude.copy()
        np.multiply(wave.amplitude, window, out=amplitude, where=changed)
        with np.errstate(over="ignore", under="ignore"):
            probability = abs(wave.amplitude)**2
            difference = wave.amplitude-amplitude
            squared_difference = abs(difference)**2
            loss = probability*(1.-window)*(1.+window)
        if (not np.isfinite(probability).all() or not np.isfinite(squared_difference).all()
                or not math.isfinite(wave.probability)):
            raise ValueError("Boundary taper requires finite cell and total probabilities")
        if (np.any(changed & (wave.amplitude != 0.) & (probability < np.finfo(float).tiny))
                or np.any((difference != 0.) & (squared_difference < np.finfo(float).tiny))):
            return wave, self  # Unresolved squared tails cannot become a zero-cost receipt.
        lost = float(np.sum(loss))
        error = math.sqrt(float(np.sum(squared_difference)))
        reference_loss, reference_error = weight*lost, math.sqrt(weight)*error
        if (lost == 0. and error == 0.) or (lost > 0. and reference_loss == 0.) or (error > 0. and reference_error == 0.):
            return wave, self
        if (reference_error > l2_cap or reference_loss > probability_cap
                or math.fsum((self.used_l2, reference_error)) > self.maximum_l2
                or math.fsum((self.used_probability, reference_loss)) > self.maximum_probability):
            return wave, self
        result = replace(wave, amplitude=amplitude)
        event = {"kind": _TAPER_KIND, "physical_absorption": False, "probability_repair": False,
            "absolute_probability_loss": lost, "absolute_l2_error": error,
            "input_probability": wave.probability, "output_probability": result.probability,
            "from_shape": list(wave.amplitude.shape), "to_shape": list(wave.amplitude.shape),
            "crop_indices_yx": [[0, size] for size in wave.amplitude.shape],
            "core_indices_yx": [[transition_cells, size-1-transition_cells] for size in wave.amplitude.shape],
            "origin_shift_m": [0., 0.], "carrier_constant_action_m": 0., "transition_cells": transition_cells,
            "error_scope": _TAPER_SCOPE, "event_index": len(self.events), "z_m": z,
            "mode_weight": weight, "l2_per_reference": reference_error,
            "probability_per_reference": reference_loss}
        return result, replace(self, events=(*self.events, event))
