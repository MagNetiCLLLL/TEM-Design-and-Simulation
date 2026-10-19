"""Nested classical tip quadrature with a count-independent energy law.

This is a numerical option for the existing truncated spatial/angular disks.
The energy shape retains the legacy Young plus logistic approximation (not a
new Gaussian Boersch law). Its positive-energy mapping is fitted to a fixed
CDF integral, never to the particles in the current emitted population.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, roots_legendre
from scipy.stats import qmc

from .base import EmissionBundle

ENERGY_SCHEMA = "legacy-continuous-young-logistic-cdf512-v1"
CDF_ORDER = 512
FWHM_TO_SIGMA = 2.354820045
LOGISTIC_STD = math.pi / math.sqrt(3.0)


@dataclass(frozen=True)
class EnergyCalibration:
    schema: str
    mean_ev: float
    lower_ev: float
    upper_ev: float
    target_std_ev: float
    decay_ev: float
    boersch_ev: float
    half_range_ev: float
    shape_mean_ev: float
    shape_std_ev: float
    location: float
    scale: float
    cdf_order: int
    integral_mean_residual_ev: float
    integral_std_residual_ev: float

    def offsets(self, young_u, boersch_u):
        shape = _shape(young_u, boersch_u, self.target_std_ev,
                       self.decay_ev, self.boersch_ev, self.half_range_ev)
        if self.target_std_ev == 0 or self.shape_std_ev == 0:
            return np.zeros_like(shape)
        standardized = (shape - self.shape_mean_ev) / self.shape_std_ev
        argument = self.location + self.scale * standardized
        kinetic = self.lower_ev + (self.upper_ev-self.lower_ev)*expit(argument)
        if not np.all(np.isfinite(kinetic)):
            raise ValueError("Source energy map produced non-finite kinetic energy")
        return kinetic - self.mean_ev


def _shape(u, v, target_std, decay, sigma, limit):
    # Analytic mean/variance of independent Young exponential and logistic
    # marginals are used before the historical symmetric half-range clipping.
    raw_std = math.hypot(decay, sigma)
    if target_std == 0 or raw_std == 0:
        return np.zeros(np.broadcast_shapes(np.shape(u), np.shape(v)))
    u = np.clip(np.asarray(u, dtype=float), 1e-15, 1.0)
    v = np.clip(np.asarray(v, dtype=float), 1e-12, 1.0-1e-12)
    shape = (decay*(np.log(u)+1.0)
             + sigma*(np.log(v)-np.log1p(-v))/LOGISTIC_STD)*target_std/raw_std
    return np.clip(shape, -limit, limit) if limit > 0 else shape


@lru_cache(maxsize=4)
def _cdf_rule(order):
    nodes, weights = roots_legendre(order)
    u = (nodes+1.0)/2.0
    w = weights/2.0
    return u, np.outer(w, w)


def energy_parameters(emitter):
    return tuple(float(getattr(emitter, name)) for name in (
        "emission_energy_ev", "minimum_kinetic_energy_ev", "energy_spread_fwhm_ev",
        "energy_half_range_ev", "young_decay_width_ev", "boersch_sigma_ev"))


@lru_cache(maxsize=64)
def calibrate_energy(mean, minimum, fwhm, limit, decay, sigma):
    """Fit the continuous legacy law using only physical source parameters.

    The finite integration rule is versioned and fixed independently of the
    ray budget and Sobol seed. Its residuals are integration-rule residuals,
    not a claim that every finite emitted population has exact moments.
    """
    values = (mean, minimum, fwhm, limit, decay, sigma)
    if not all(math.isfinite(x) for x in values) or min(minimum, fwhm, limit, decay, sigma) < 0:
        raise ValueError("Source energy parameters must be finite and non-negative")
    if mean <= minimum:
        raise ValueError("Mean source energy must exceed its non-negative minimum")
    if limit <= 0:
        raise ValueError("Sobol tip sampling requires a finite positive energy half range")
    std = fwhm/FWHM_TO_SIGMA
    lower = max(minimum, mean-limit)
    upper = mean+limit
    if std == 0 or decay == sigma == 0:
        return EnergyCalibration(ENERGY_SCHEMA, mean, lower, upper, 0., decay,
            sigma, limit, 0., 0., 0., 0., CDF_ORDER, 0., 0.)
    if std >= math.sqrt((mean-lower)*(upper-mean)):
        raise ValueError("Source energy spread requires a non-degenerate bounded distribution")
    u, weights = _cdf_rule(CDF_ORDER)
    shape = _shape(u[:, None], u[None, :], std, decay, sigma, limit)
    shape_mean = float(np.sum(weights*shape))
    shape_std = float(np.sqrt(np.sum(weights*(shape-shape_mean)**2)))
    if shape_std <= 0:
        raise ValueError("Source energy shape is degenerate")
    s = (shape-shape_mean)/shape_std
    span = upper-lower
    fraction = (mean-lower)/span

    def equations(parameters, jacobian=False):
        p = expit(parameters[0]+parameters[1]*s)
        delta = lower+span*p-mean
        if jacobian:
            derivative = span*p*(1-p)
            return np.array([[np.sum(weights*derivative)/std,
                              np.sum(weights*derivative*s)/std],
                             [2*np.sum(weights*delta*derivative)/std**2,
                              2*np.sum(weights*delta*derivative*s)/std**2]])
        return np.array([np.sum(weights*delta)/std,
                         np.sum(weights*delta**2)/std**2-1])

    fitted = least_squares(equations,
        (math.log(fraction/(1-fraction)), std/(span*fraction*(1-fraction))),
        jac=lambda p: equations(p, True), bounds=([-np.inf, 0.], [np.inf, 1024.]),
        xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=100)
    residual = equations(fitted.x)
    if not fitted.success or np.max(np.abs(residual)) > 1e-9:
        raise ValueError("Unable to calibrate the continuous source energy distribution")
    return EnergyCalibration(ENERGY_SCHEMA, mean, lower, upper, std, decay,
        sigma, limit, shape_mean, shape_std, float(fitted.x[0]), float(fitted.x[1]),
        CDF_ORDER, float(residual[0]*std), float((math.sqrt(1+residual[1])-1)*std))


def sampling_diagnostics(emitter):
    """JSON-safe description; this is not an emitted-batch convergence test."""
    table = emitter.cdf_quadrature
    if table is not None:
        table.validate()
        return {"schema": table.schema, "role": table.role, "table_digest": table.digest,
            "nodes": table.total, "construction": table.construction,
            "weights": "captured original CDF probabilities; never renormalized",
            "weight_sum": math.fsum(table.weights),
            "positive_weight_nodes": sum(w > 0 for w in table.weights),
            "energy_calibration": asdict(calibrate_energy(*energy_parameters(emitter)))}
    plan = emitter.sampling
    if plan is None:
        return {"schema": "legacy-joint-halton-batch-moments", "seed": None}
    plan.validate()
    return {**asdict(plan), "dimensions": 6, "antithetic_pair_size": 2,
            "pair_operation": "local x,y,tx,ty simultaneous inversion; same energy",
            "weights": "1/N; never renormalized after loss",
            "energy_calibration": asdict(calibrate_energy(*energy_parameters(emitter)))}


def phase_space_from_quantiles(emitter, quantiles):
    """Map six independent CDF coordinates to local x/y/slopes/energy.

    This pure physical-distribution map is separate from the selected joint
    quadrature. Energies returned here are offsets from the launch mean.
    """
    from .emitter import _truncated_gaussian_disk
    u = np.asarray(quantiles, dtype=float)
    if u.ndim != 2 or u.shape[1] != 6 or not np.all(np.isfinite(u)) or np.any((u < 0) | (u > 1)):
        raise ValueError("Source quantiles must be finite six-dimensional CDF coordinates in [0, 1]")
    spatial_sigma = emitter.virtual_source_fwhm_nm/FWHM_TO_SIGMA*1e-9
    x, y = _truncated_gaussian_disk(u[:, 0], u[:, 1], spatial_sigma, 3*spatial_sigma)
    tx, ty = _truncated_gaussian_disk(u[:, 2], u[:, 3],
        emitter.angular_rms_mrad*1e-3, emitter.angular_cutoff_mrad*1e-3)
    energy = calibrate_energy(*energy_parameters(emitter)).offsets(u[:, 4], u[:, 5])
    return x, y, tx, ty, energy


def joint_bundle(emitter, count):
    from .tip_curvature import curve_bundle
    plan = emitter.sampling.validate()
    plan.validate_count(count)
    u = qmc.Sobol(d=6, scramble=True, seed=plan.seed, bits=30).random_base2(
        int(count).bit_length()-2)
    x, y, tx, ty, energy = phase_space_from_quantiles(emitter, u)

    def paired(value):
        return np.column_stack((value, -value)).reshape(-1)

    bundle = EmissionBundle(paired(x), paired(y), paired(tx), paired(ty),
        np.repeat(energy, 2), np.full(count, 1.0/count), np.arange(count, dtype=np.int64))
    return curve_bundle(bundle, emitter)


def explicit_cdf_bundle(emitter):
    """Emit exactly the captured table through the ordinary physical tip map."""
    from .tip_curvature import curve_bundle
    table = emitter.cdf_quadrature.validate()
    x, y, tx, ty, energy = phase_space_from_quantiles(emitter, table.coordinates)
    sign = np.asarray(table.signs, dtype=np.int64)
    bundle = EmissionBundle(x*sign, y*sign, tx*sign, ty*sign, energy,
        np.asarray(table.weights, dtype=float), np.asarray(table.ray_ids, dtype=np.int64))
    return curve_bundle(bundle, emitter)
