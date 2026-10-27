"""Private Fourier candidates between explicitly selected preview segments.

The caller supplies a completed, admitted CPU plane, its original immutable
support ledger, and an admission-only callback. This module never commits a
ledger, factorizes a mode, applies a physical operator, or invents a resource
failure. A caller may commit the returned proposal only after its complete
physical operator succeeds. Ordinary pressure recovery remains independent.
"""
from dataclasses import dataclass, replace
import math
from numbers import Real
import traceback

import numpy as np

from temsim.cache_memory import retained_memory_inventory
from temsim.immutable_json import freeze_json, json_digest
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.wave_flux import check_lossless_norm
from temsim.physics.wave_grid import WaveGridNumerics, WaveGridBudgetError, WaveSamplingError


@dataclass(frozen=True)
class SegmentResamplingProposal:
    """An admitted private candidate, or the identical original objects."""

    wave: PlaneWave
    ledger: NumericalSupportLedger
    diagnostics: object
    representation_estimates: object = None

    def __post_init__(self):
        if type(self.wave) is not PlaneWave or type(self.ledger) is not NumericalSupportLedger:
            raise ValueError("Segment proposals require a PlaneWave and its immutable support ledger")
        object.__setattr__(self, "diagnostics", freeze_json(self.diagnostics))
        if self.representation_estimates is not None:
            from temsim.physics.potential_representation_estimates import RepresentationEstimateLedger
            if type(self.representation_estimates) is not RepresentationEstimateLedger:
                raise ValueError("Phase proposals require an immutable representation estimate history")


class _CandidateSavingsDeclined(Exception):
    """An optional candidate is valid but no longer usefully smaller."""


def _cancel(cancelled):
    if cancelled():
        raise InterruptedError("Preview segment preparation cancelled")


def _outside_bytes(active, *retained):
    active_owners = retained_memory_inventory(active.amplitude)
    owners = retained_memory_inventory(*(wave.amplitude for wave in retained))
    return sum(size for key, size in owners.items() if key not in active_owners)


def _failure(error):
    """Keep primitive evidence and release failed callback workspaces."""
    result = {"error_type": type(error).__name__, "reason": str(error)}
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend((current.__cause__, current.__context__))
    return result


def _validate_admitted(source, admitted):
    """Check the inexpensive invariants of the admission-only contract.

    The trusted callback must perform only validation and unitary Fourier
    refinement; equality of norms alone does not prove equality of fields.
    """
    if type(admitted) is not PlaneWave or not isinstance(admitted.amplitude, np.ndarray):
        raise ValueError("Segment admission must return a CPU PlaneWave")
    if any(new < old for new, old in zip(admitted.amplitude.shape, source.amplitude.shape)):
        raise ValueError("Segment admission may only retain or losslessly refine its input lattice")
    for name in ("origin_m", "curvature_m1", "tilt_rad"):
        if not np.array_equal(getattr(admitted, name), getattr(source, name)):
            raise ValueError("Segment admission changed the physical origin or analytical phase carrier")
    before = source.basis_m*np.asarray(source.amplitude.shape)[::-1]
    after = admitted.basis_m*np.asarray(admitted.amplitude.shape)[::-1]
    if np.linalg.norm(after-before) > 64*np.finfo(float).eps*np.linalg.norm(before):
        raise ValueError("Segment admission changed the physical Fourier period")
    check_lossless_norm(source.probability, admitted.probability,
                        context="Preview segment admission-only refinement")


def prepare_segment_resampling(wave, ledger, *, z_m, mode_weight=1.,
        numerics=WaveGridNumerics(), retained_bytes=0, admit, cancelled=lambda: False,
        reserved_l2=0.):
    """Propose one useful segment-entrance reduction without spending history.

    ``admit(candidate, *, retained_bytes)`` returns the same represented CPU
    wave, optionally after lossless refinement. It must check the original
    boundary, alias, domain/coverage and resource gates, and may not propagate,
    filter, normalize, fit phase, or mutate a ledger. The supplied byte count
    includes external caller-owned buffers plus unique original/spectral
    buffers retained by this helper, excluding the active candidate.

    Only OccupiedCarrierBoundaryError enables the bounded 2/4/8/16-cell edge
    family. Each window starts from the SAME raw spectrum and spectrum ledger.
    WaveSamplingError and WaveGridBudgetError reject a numerical candidate;
    domain/invalid errors and cancellation propagate unchanged. A proposal
    must still save at least 5 percent of cells after admission, with neither
    axis larger than the original. The old one-percent cap applies to EACH
    newly proposed event; the original total and spent limits never change.
    ``reserved_l2`` reserves previously estimated representation error within
    that SAME total allowance; it does not alter or reinterpret old receipts.

    The caller owns provenance and physical-operator success. For a no-op,
    the result contains the identical original wave and ledger objects.
    """
    # Local import permits the carrier to call this helper without a cycle.
    from temsim.physics.potential_wave_carrier import OccupiedCarrierBoundaryError

    if type(wave) is not PlaneWave or not isinstance(wave.amplitude, np.ndarray):
        raise ValueError("Segment preparation requires an admitted CPU PlaneWave")
    if type(ledger) is not NumericalSupportLedger:
        raise ValueError("Segment preparation requires the original immutable support ledger")
    if type(numerics) is not WaveGridNumerics:
        raise ValueError("Segment preparation requires explicit WaveGridNumerics")
    numerics.validate()
    if (isinstance(reserved_l2, (bool, np.bool_)) or not isinstance(reserved_l2, Real)
            or not math.isfinite(reserved_l2) or reserved_l2 < 0):
        raise ValueError("Reserved representation L2 must be finite and nonnegative")
    if (type(retained_bytes) is not int or retained_bytes < 0
            or not callable(admit) or not callable(cancelled)):
        raise ValueError("Segment preparation requires nonnegative retained bytes and callable admission/cancellation")
    for name, value, signed in (("Z", z_m, True), ("mode weight", mode_weight, False)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or (not signed and value < 0)):
            raise ValueError(f"Segment {name} must be finite and {'numeric' if signed else 'nonnegative'}")
    _cancel(cancelled)
    attempts = []
    base = {"schema": "uncommitted-preview-segment-proposal-v1", "input_shape": wave.amplitude.shape,
        "z_m": float(z_m), "mode_weight": float(mode_weight), "minimum_saved_cell_fraction": .05,
        "original_ledger_digest": json_digest(ledger.to_dict()), "committed": False,
        "physical_operator_executed": False, "normalization_applied": False,
        "error_scope": "recorded numerical input projection only; no propagated complex-error or pointwise phase guarantee"}

    def unchanged(reason):
        return SegmentResamplingProposal(wave, ledger, {**base, "status": "unchanged",
            "reason": reason, "attempts": attempts})

    if reserved_l2:
        base["reserved_representation_l2_estimate"] = float(reserved_l2)
    if reserved_l2 and ledger.used_l2+reserved_l2 >= ledger.maximum_l2:
        return unchanged("No remaining combined support and representation allowance")

    try:
        raw, spectral_ledger = ledger.resample_spectrum(wave, z_m=z_m, mode_weight=mode_weight,
            numerics=numerics, retained_bytes=retained_bytes, cancelled=cancelled)
    except (WaveSamplingError, WaveGridBudgetError) as error:
        attempts.append({"kind": "spectral preparation", "status": "numerical rejection", **_failure(error)})
        _cancel(cancelled)
        return unchanged("Spectral preparation could not be admitted")
    _cancel(cancelled)
    if raw is wave or spectral_ledger is ledger:
        return unchanged("No smaller spectrum within the original one-percent event and remaining budgets")
    if spectral_ledger.used_l2+reserved_l2 > ledger.maximum_l2:
        return unchanged("Spectral candidate exceeds the combined support and representation allowance")
    widths = [0]
    for width in widths:
        _cancel(cancelled)
        candidate, pending = raw, spectral_ledger
        row = {"kind": "spectral" if width == 0 else "spectral with edge window", "transition_cells": width}
        try:
            if width:
                numerics.check_working_memory(raw.amplitude.shape,
                    working_bytes=256*raw.amplitude.size,
                    retained_bytes=retained_bytes+_outside_bytes(raw, wave))
                candidate, pending = spectral_ledger.taper_boundary(raw, z_m=z_m,
                    mode_weight=mode_weight, transition_cells=width)
                _cancel(cancelled)
                if candidate is raw or pending is spectral_ledger:
                    attempts.append({**row, "status": "no affordable edge window"})
                    continue
            if pending.used_l2+reserved_l2 > ledger.maximum_l2:
                attempts.append({**row, "status": "combined support and representation allowance exceeded"})
                continue
            outside = retained_bytes+_outside_bytes(candidate, wave, raw)
            row["admission_retained_bytes"] = outside
            admitted = admit(candidate, retained_bytes=outside)
            _cancel(cancelled)
            _validate_admitted(candidate, admitted)
            shape = admitted.amplitude.shape
            row["admitted_shape"] = shape
            if (any(new > old for new, old in zip(shape, wave.amplitude.shape))
                    or 100*admitted.amplitude.size > 95*wave.amplitude.size):
                attempts.append({**row, "status": "insufficient saving after admission"})
                del admitted
                continue
            attempts.append({**row, "status": "admitted private candidate"})
            return SegmentResamplingProposal(admitted, pending, {**base, "status": "candidate",
                "output_shape": shape, "selected_transition_cells": width, "attempts": attempts,
                "pending_ledger_digest": json_digest(pending.to_dict()),
                "additional_event_count": len(pending.events)-len(ledger.events),
                "weighted_complex_l2_cost": pending.used_l2-ledger.used_l2,
                "weighted_probability_cost": pending.used_probability-ledger.used_probability})
        except OccupiedCarrierBoundaryError as error:
            attempts.append({**row, "status": "occupied numerical edge", **_failure(error)})
            if width == 0:
                widths.extend((2, 4, 8, 16))
        except (WaveSamplingError, WaveGridBudgetError) as error:
            attempts.append({**row, "status": "numerical rejection", **_failure(error)})
        except _CandidateSavingsDeclined as error:
            attempts.append({**row, "status": "insufficient saving after admission", **_failure(error)})
        finally:
            # No rejected window, failed callback frame or diagnostic record
            # may keep an array alive while the next window is allocated.
            candidate, pending = raw, spectral_ledger
        _cancel(cancelled)
    return unchanged("No admitted segment candidate preserves a five-percent cell saving")


def prepare_phase_segment_resampling(wave, ledger, representation_estimates, *,
        wavelength_m, z_m, mode_weight=1., numerics=WaveGridNumerics(),
        retained_bytes=0, admit, cancelled=lambda: False):
    """Prepare an explicitly selected phase-chart and spectrum candidate.

    CPU preparation may use twice the propagation pixel ceiling, under the
    SAME host allocation limit. It changes the stored quadratic carrier with
    compensating envelope phase; it never removes physical phase. A separately
    labelled numerical representation estimate reserves part of the original
    cumulative L2 allowance. It is not a certified interpolation-error bound.
    Its explicit preview quadrature allowance is 1e-7 times the input weighted
    norm, charged in full as well as the measured maximum and probe spread.

    The caller's ``admit`` must use the ORIGINAL propagation numerics and the
    admission-only contract of :func:`prepare_segment_resampling`. Only a
    useful candidate that still fits that original ceiling is returned. Its
    three histories remain private until the complete next physical operator
    succeeds. No-op and numerical rejection return all original objects.
    """
    from temsim.physics.multiplane_wave import reselect_phase_carrier
    from temsim.physics.potential_representation_estimates import (
        RepresentationEstimateLedger, estimate_carrier_representational_change,
        require_combined_support_allowance,
    )

    if (type(wave) is not PlaneWave or not isinstance(wave.amplitude, np.ndarray)
            or type(ledger) is not NumericalSupportLedger
            or type(representation_estimates) is not RepresentationEstimateLedger
            or type(numerics) is not WaveGridNumerics):
        raise ValueError("Phase preparation requires an admitted CPU plane and its original two histories")
    numerics.validate()
    if (type(retained_bytes) is not int or retained_bytes < 0
            or not callable(admit) or not callable(cancelled)):
        raise ValueError("Phase preparation requires nonnegative retained bytes and callable admission/cancellation")
    for name, value, minimum in (("wavelength", wavelength_m, 0.), ("mode weight", mode_weight, 0.),
                                  ("Z", z_m, -math.inf)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or value < minimum
                or (name == "wavelength" and value == 0)):
            raise ValueError(f"Phase preparation {name} is invalid")
    _cancel(cancelled)
    # Existing history mismatch or overspending is an invalid entrance, not an
    # optional candidate failure that may silently fall back to the old wave.
    remaining = require_combined_support_allowance(ledger, representation_estimates)
    preparation = replace(numerics, maximum_pixels=min(65536, 2*numerics.maximum_pixels),
                          compute_backend="CPU")
    base = {"schema": "uncommitted-phase-prepared-segment-proposal-v1",
        "z_m": float(z_m), "mode_weight": float(mode_weight),
        "input_shape": wave.amplitude.shape, "propagation_maximum_pixels": numerics.maximum_pixels,
        "preparation_maximum_pixels": preparation.maximum_pixels,
        "preparation_backend": "CPU", "maximum_host_working_bytes": numerics.maximum_working_bytes,
        "absolute_quadrature_relative_tolerance": 1e-7,
        "original_ledger_digest": json_digest(ledger.to_dict()),
        "original_representation_estimate_digest": json_digest(representation_estimates.to_dict()),
        "committed": False, "physical_operator_executed": False,
        "normalization_applied": False, "certified_representation_error": False,
        "error_scope": "numerical representation estimate plus recorded input projections; no propagated or pointwise phase guarantee"}

    def unchanged(reason, **details):
        return SegmentResamplingProposal(wave, ledger,
            {**base, "status": "unchanged", "reason": reason, **details}, representation_estimates)

    if remaining <= 0 or mode_weight == 0 or ledger.used_probability >= ledger.maximum_probability:
        return unchanged("No remaining combined support and representation allowance")
    try:
        recharted, rows = reselect_phase_carrier(wave, wavelength_m,
            grid_numerics=preparation, retained_bytes=retained_bytes, cancelled=cancelled)
        _cancel(cancelled)
        if recharted is wave:
            return unchanged("No resolved useful phase-carrier change", phase_preparation=rows)
        receipt = estimate_carrier_representational_change(wave, recharted, wavelength_m,
            z_m=z_m, mode_weight=mode_weight, numerics=preparation,
            retained_bytes=retained_bytes, cancelled=cancelled,
            absolute_quadrature_relative_tolerance=1e-7)
        _cancel(cancelled)
        pending_estimates = representation_estimates.append(receipt, ledger)
        require_combined_support_allowance(ledger, pending_estimates)

        def admit_original_limit(candidate, *, retained_bytes):
            admitted = admit(candidate, retained_bytes=retained_bytes)
            _cancel(cancelled)
            _validate_admitted(candidate, admitted)
            shape = admitted.amplitude.shape
            if (max(shape) > numerics.maximum_pixels
                    or any(new > old for new, old in zip(shape, wave.amplitude.shape))
                    or 100*admitted.amplitude.size > 95*wave.amplitude.size):
                raise _CandidateSavingsDeclined("Phase candidate must save five percent against the original admitted plane")
            return admitted

        proposal = prepare_segment_resampling(recharted, ledger, z_m=z_m, mode_weight=mode_weight,
            numerics=preparation, retained_bytes=retained_bytes+_outside_bytes(recharted, wave),
            admit=admit_original_limit, cancelled=cancelled,
            reserved_l2=pending_estimates.used_l2_estimate)
        _cancel(cancelled)
        if proposal.wave is recharted:
            return unchanged("No useful admitted spectrum after phase preparation",
                phase_preparation=rows, segment_preparation=proposal.diagnostics)
        require_combined_support_allowance(proposal.ledger, pending_estimates)
        return SegmentResamplingProposal(proposal.wave, proposal.ledger,
            {**base, "status": "candidate", "output_shape": proposal.wave.amplitude.shape,
             "phase_preparation": rows, "segment_preparation": proposal.diagnostics,
             "pending_ledger_digest": json_digest(proposal.ledger.to_dict()),
             "pending_representation_estimate_digest": json_digest(pending_estimates.to_dict()),
             "additional_representation_l2_estimate": pending_estimates.used_l2_estimate-representation_estimates.used_l2_estimate,
             "weighted_complex_l2_cost": proposal.ledger.used_l2-ledger.used_l2,
             "weighted_probability_cost": proposal.ledger.used_probability-ledger.used_probability},
            pending_estimates)
    except (WaveSamplingError, WaveGridBudgetError) as error:
        evidence = _failure(error)
        _cancel(cancelled)
        return unchanged("Phase preparation rejected by numerical admission", numerical_failure=evidence)


__all__ = ("SegmentResamplingProposal", "prepare_segment_resampling", "prepare_phase_segment_resampling")
