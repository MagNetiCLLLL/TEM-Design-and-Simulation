"""Append-only CDF sampling of the existing classical cold-FEG source law.

Particle IDs select the original Halton dimensions, starting at index 18.
All source geometry, energy bounds and continuous-CDF calibration depend on
physical source parameters, never the requested particle count. Increasing N
adds CDF samples; it does not interpolate propagated rays or change old ones.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import math

import numpy as np

from .base import EmissionBundle


ENERGY_INTERPOLATION_SCHEMA = "fixed-young-logistic-energy-linear-table-v1"
MAXIMUM_ENERGY_INTERPOLATION_ERROR_EV = 1e-10
MAXIMUM_ENERGY_TABLE_INTERVALS = 1048576


def _readonly(values):
    array = np.asarray(values, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class FixedEnergyInterpolation:
    """Count-independent monotone shape-to-energy table, with eV axes."""
    calibration: object
    shape_ev: np.ndarray
    kinetic_ev: np.ndarray
    analytic_interpolation_error_bound_ev: float
    midpoint_maximum_error_ev: float

    def offsets(self, young_u, boersch_u):
        from .tip_joint_sampling import _shape
        cal = self.calibration
        shape = _shape(young_u, boersch_u, cal.target_std_ev,
                       cal.decay_ev, cal.boersch_ev, cal.half_range_ev)
        if not np.isfinite(shape).all() or np.any(shape < self.shape_ev[0]) or np.any(shape > self.shape_ev[-1]):
            raise ValueError("Source energy shape lies outside its fixed interpolation table")
        # The input is a new CDF sample, never an interpolated propagated ray.
        return np.interp(shape, self.shape_ev, self.kinetic_ev)-cal.mean_ev

    def to_record(self):
        roundoff = 64*np.finfo(float).eps*max(1., abs(self.calibration.lower_ev), abs(self.calibration.upper_ev))
        return {"schema": ENERGY_INTERPOLATION_SCHEMA,
            "method": "np.interp piecewise linear monotone shape-to-kinetic-energy map",
            "table_inputs": "Physical energy parameters only; independent of ray count, sampled extrema, seed, or transmitted particles",
            "shape_interval_ev": [float(self.shape_ev[0]), float(self.shape_ev[-1])],
            "kinetic_energy_bounds_ev": [self.calibration.lower_ev, self.calibration.upper_ev],
            "table_nodes": len(self.shape_ev),
            "table_sha256": sha256(self.shape_ev.tobytes()+self.kinetic_ev.tobytes()).hexdigest(),
            "maximum_requested_interpolation_error_ev": MAXIMUM_ENERGY_INTERPOLATION_ERROR_EV,
            "analytic_piecewise_linear_error_bound_ev": self.analytic_interpolation_error_bound_ev,
            "binary64_evaluation_roundoff_allowance_ev": roundoff,
            "analytic_bound_plus_roundoff_allowance_ev": self.analytic_interpolation_error_bound_ev+roundoff,
            "independent_midpoint_maximum_error_ev": self.midpoint_maximum_error_ev,
            "bound_scope": "Exact logistic second-derivative interpolation bound; separate binary64 evaluation roundoff is checked at table midpoints",
            "cdf_tail_policy": "Retain the existing fixed Young 1e-15 and logistic 1e-12 endpoint clipping, followed by fixed symmetric shape bounds; no sample-dependent tail fitting"}


@lru_cache(maxsize=16)
def _fixed_energy_interpolation(mean, minimum, fwhm, limit, decay, sigma):
    from scipy.special import expit
    from .tip_joint_sampling import calibrate_energy

    cal = calibrate_energy(mean, minimum, fwhm, limit, decay, sigma)
    if cal.target_std_ev == 0 or cal.shape_std_ev == 0:
        return FixedEnergyInterpolation(cal, _readonly([-limit, limit]),
            _readonly([mean, mean]), 0., 0.)
    # For logistic p, max |p(1-p)(1-2p)| = 1/(6 sqrt(3)). Linear
    # interpolation error is at most max|E''|*h^2/8 on every interval.
    second_derivative = ((cal.upper_ev-cal.lower_ev)*(cal.scale/cal.shape_std_ev)**2
                         / (6.*math.sqrt(3.)))
    roundoff = 64*np.finfo(float).eps*max(1., abs(cal.lower_ev), abs(cal.upper_ev))
    intervals = 2
    while second_derivative*(2.*limit/intervals)**2/8.+roundoff > MAXIMUM_ENERGY_INTERPOLATION_ERROR_EV:
        intervals *= 2
        if intervals > MAXIMUM_ENERGY_TABLE_INTERVALS:
            raise ValueError("The fixed source energy interpolation table exceeds its numerical node budget")
    shape = np.linspace(-limit, limit, intervals+1, dtype=np.float64)

    def kinetic(values):
        return cal.lower_ev+(cal.upper_ev-cal.lower_ev)*expit(
            cal.location+cal.scale*(values-cal.shape_mean_ev)/cal.shape_std_ev)

    energy = kinetic(shape)
    if not np.isfinite(energy).all() or np.any(np.diff(energy) < 0):
        raise ValueError("The fixed source energy interpolation table must be finite and monotone")
    midpoint = (shape[:-1]+shape[1:])*.5
    observed = float(np.max(np.abs(np.interp(midpoint, shape, energy)-kinetic(midpoint))))
    bound = second_derivative*float(np.max(np.diff(shape)))**2/8.
    if observed > bound+roundoff:
        raise ValueError("Fixed source energy table failed its independent midpoint error check")
    return FixedEnergyInterpolation(cal, _readonly(shape), _readonly(energy), float(bound), observed)


def fixed_energy_interpolation(emitter) -> FixedEnergyInterpolation:
    """Capture the same interpolation table for every N at fixed parameters."""
    from .tip_joint_sampling import energy_parameters
    return _fixed_energy_interpolation(*energy_parameters(emitter))


def halton_prefix_bundle(emitter, count: int) -> EmissionBundle:
    """Map an immutable prefix of six CDF dimensions to a source boundary.

    Local X/Y are metres, slopes are radians, and energies are eV offsets
    from the fixed launch mean. The ordinary curvature adapter retains the
    actual source surface and full direction. Source emission remains at its
    existing common epoch; sampling introduces no particle-dependent clock.
    Absolute quadrature probabilities are 1/N before physical interception.
    """
    from .emitter import HALTON_PREFIX_SAMPLING_SCHEMA, _halton_dimensions, _truncated_gaussian_disk
    from .tip_curvature import curve_bundle
    from .tip_joint_sampling import FWHM_TO_SIGMA

    plan = emitter.sampling.validate()
    if plan.schema != HALTON_PREFIX_SAMPLING_SCHEMA:
        raise ValueError("An explicit fixed Halton prefix sampling schema is required")
    plan.validate_count(count)
    coordinates = np.column_stack(_halton_dimensions(count, (2, 3, 5, 7, 11, 13)))
    spatial_sigma = emitter.virtual_source_fwhm_nm/FWHM_TO_SIGMA*1e-9
    x, y = _truncated_gaussian_disk(coordinates[:, 0], coordinates[:, 1], spatial_sigma, 3.*spatial_sigma)
    tx, ty = _truncated_gaussian_disk(coordinates[:, 2], coordinates[:, 3],
        emitter.angular_rms_mrad*1e-3, emitter.angular_cutoff_mrad*1e-3)
    energy = fixed_energy_interpolation(emitter).offsets(coordinates[:, 4], coordinates[:, 5])
    bundle = EmissionBundle(x, y, tx, ty, energy,
        np.full(count, 1./count, dtype=np.float64), np.arange(count, dtype=np.int64))
    return curve_bundle(bundle, emitter)


def halton_prefix_diagnostics(emitter) -> dict:
    """Describe the count-independent boundary map without emitting particles."""
    from .emitter import HALTON_PREFIX_SAMPLING_SCHEMA
    from .tip_joint_sampling import calibrate_energy, energy_parameters

    plan = emitter.sampling.validate()
    if plan.schema != HALTON_PREFIX_SAMPLING_SCHEMA:
        raise ValueError("An explicit fixed Halton prefix sampling schema is required")
    return {**asdict(plan), "dimensions": 6, "halton_bases": [2, 3, 5, 7, 11, 13],
        "halton_first_index": 18, "particle_id_to_halton_index": "ray_id + 18",
        "antithetic_pair_size": 1,
        "weights": "Original quadrature probability 1/N; no transmitted-current restoration",
        "refinement": "Append CDF samples using the same inverse marginal maps; no propagated-trajectory interpolation",
        "prefix_invariants": ["local position", "local direction", "energy", "source-surface position and normal", "common emission epoch"],
        "energy_calibration": asdict(calibrate_energy(*energy_parameters(emitter))),
        "energy_interpolation": fixed_energy_interpolation(emitter).to_record(),
        "energy_width_definition": "Configured FWHM is the existing RMS-equivalent width, 2.354820045 times continuous-law standard deviation",
        "energy_moment_scope": "The fixed continuous CDF law is calibrated once per physical parameter set; finite-N moments are observed and are never fitted or renormalized",
        "legacy_scope": "An absent sampling record retains legacy batch-dependent energy calibration; it is not relabelled as this schema"}


__all__ = ("FixedEnergyInterpolation", "fixed_energy_interpolation", "halton_prefix_bundle", "halton_prefix_diagnostics")
