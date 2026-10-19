"""Exact conditional CDF measure for offline source-integration validation.

This module changes only quadrature nodes. It calls the production fixed energy
calibration and leaves the physical source, clipping and energy map unchanged.
No physical gun/column propagation or optical acceptance model is used here.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.special import expit

from temsim.optics.electron_gun.tip_joint_sampling import (
    ENERGY_SCHEMA, LOGISTIC_STD, calibrate_energy, energy_parameters,
)

SCHEMA = "offline-conditional-energy-cdf-strata-v1"
YOUNG_MIN = 1e-15
BOERSCH_MIN = 1e-12
BOERSCH_MAX = 1.0-1e-12


def _readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class EnergyBandCuts:
    lower_cdf: np.ndarray
    upper_cdf: np.ndarray
    edges: np.ndarray
    interval_probability: np.ndarray
    conditional_minimum_ev: np.ndarray
    conditional_maximum_ev: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class EnergyStrata:
    base_points: np.ndarray
    base_probability: np.ndarray
    independent_base_index: np.ndarray
    stratum_index: np.ndarray
    interval_lower: np.ndarray
    interval_upper: np.ndarray
    cuts: EnergyBandCuts
    metadata: dict


def _calibration(emitter):
    if getattr(emitter, "surface_model", None) is not None or getattr(emitter, "coherence", None) is not None:
        raise ValueError("Energy strata require the classical fixed-CDF tip source")
    result = calibrate_energy(*energy_parameters(emitter))
    if result.schema != ENERGY_SCHEMA:
        raise ValueError("Unsupported fixed energy calibration schema")
    return result


def energy_band_cdf_intervals(emitter, young_u, band_ev):
    """Invert physical launch energy conditional on each original Young CDF.

    The cuts are P(E < band_lo | Young) and P(E <= band_hi | Young).
    Thus atoms caused by shape/quantile clipping belong to the inclusive band.
    All three intervals are retained in this record, including zero mass.
    This is algebraic inversion of the calibrated map, not a fitted aperture.
    """
    young = np.asarray(young_u, dtype=np.float64)
    band = np.asarray(band_ev, dtype=np.float64)
    if young.ndim != 1 or not len(young) or not np.isfinite(young).all() or np.any((young < 0)|(young > 1)):
        raise ValueError("Young coordinates must be a nonempty finite CDF vector")
    if band.shape != (2,) or not np.isfinite(band).all() or not 0 < band[0] <= band[1]:
        raise ValueError("The physical launch energy band must have positive finite ordered endpoints")
    cal = _calibration(emitter)
    # Add the mean exactly as the production emission -> launch path does.
    minimum = cal.mean_ev+cal.offsets(young, np.zeros_like(young))
    maximum = cal.mean_ev+cal.offsets(young, np.ones_like(young))
    raw_std = math.hypot(cal.decay_ev, cal.boersch_ev)
    constant = (cal.target_std_ev == 0 or cal.shape_std_ev == 0 or cal.scale == 0
                or cal.boersch_ev == 0 or raw_std == 0)
    if constant:
        lower = (minimum < band[0]).astype(np.float64)
        upper = (maximum <= band[1]).astype(np.float64)
    else:
        factor = cal.target_std_ev/raw_std
        a = cal.decay_ev*(np.log(np.clip(young, YOUNG_MIN, 1.))+1.)*factor
        b = cal.boersch_ev*factor/LOGISTIC_STD
        s_min = np.clip(a+b*(math.log(BOERSCH_MIN)-math.log1p(-BOERSCH_MIN)),
                        -cal.half_range_ev, cal.half_range_ev)
        s_max = np.clip(a+b*(math.log(BOERSCH_MAX)-math.log1p(-BOERSCH_MAX)),
                        -cal.half_range_ev, cal.half_range_ev)

        def inverse_cut(energy, inclusive):
            out = np.empty_like(young)
            below = energy < minimum if inclusive else energy <= minimum
            above = energy >= maximum if inclusive else energy > maximum
            out[below] = 0.
            out[above] = 1.
            active = ~(below|above)
            if active.any():
                # The equality cases use the actual clipped support shape,
                # preserving finite atoms without inverting a rounded tail E.
                shape = np.empty_like(young)
                at_min = active & (energy == minimum)
                at_max = active & (energy == maximum)
                shape[at_min], shape[at_max] = s_min[at_min], s_max[at_max]
                interior = active & ~(at_min|at_max)
                if interior.any():
                    if not cal.lower_ev < energy < cal.upper_ev:
                        raise ValueError("Energy threshold cannot be resolved inside the calibrated support")
                    log_odds = math.log(energy-cal.lower_ev)-math.log(cal.upper_ev-energy)
                    shape[interior] = cal.shape_mean_ev+cal.shape_std_ev*(log_odds-cal.location)/cal.scale
                shape[active] = np.clip(shape[active], s_min[active], s_max[active])
                logit_v = (shape[active]-a[active])/b
                # Avoid a reciprocal near one; the original CDF still remains
                # float64 and its energy-conditioned inverse can be ill-scaled.
                cdf = np.where(logit_v >= 0., 1.-expit(-logit_v), expit(logit_v))
                out[active] = np.clip(cdf, BOERSCH_MIN, BOERSCH_MAX)
            return out

        lower = inverse_cut(float(band[0]), False)
        upper = inverse_cut(float(band[1]), True)
    edges = np.column_stack((np.zeros_like(young), lower, upper, np.ones_like(young)))
    widths = np.diff(edges, axis=1)
    if not np.isfinite(edges).all() or np.any(widths < 0):
        raise ValueError("Conditional energy cuts did not form a finite full CDF partition")
    return EnergyBandCuts(*(_readonly(values) for values in
        (lower, upper, edges, widths, minimum, maximum)), metadata=dict(
            schema=SCHEMA, band_ev=[float(value) for value in band], energy_calibration=asdict(cal),
            cut_convention="lower=P(E<lo|Young); upper=P(E<=hi|Young); band includes clipping atoms",
            energy_definition="Physical launch kinetic energy = emission mean + emitted offset, in eV",
            probability_definition="Original uniform Boersch CDF interval lengths; no normalization",
            numerical_scope="Algebraic inverse of the captured calibration formulas, subject to floating roundoff; no optical convergence claim"))


def expand_energy_strata(emitter, original_base, band_ev):
    """Return positive-mass rows, ordered by original base then stratum.

    Input order: position radius CDF, position azimuth CDF, direction azimuth
    CDF, Young CDF, Boersch CDF. Only Boersch CDF is replaced by a+(b-a)*u.
    Returned base_probability already includes 1/M. Zero-mass rows are omitted
    from transport, but their exact zero masses remain in cuts. No surviving
    probability, node count or floating sum is renormalized.
    """
    base = np.asarray(original_base, dtype=np.float64)
    if (base.ndim != 2 or base.shape[1] != 5 or not len(base)
            or not np.isfinite(base).all() or np.any((base < 0)|(base > 1))):
        raise ValueError("Original bases must be a nonempty finite M-by-5 CDF array")
    cuts = energy_band_cdf_intervals(emitter, base[:, 3], band_ev)
    parent, stratum = np.nonzero(cuts.interval_probability > 0.)
    lo, hi = cuts.edges[parent, stratum], cuts.edges[parent, stratum+1]
    expanded = base[parent].copy()
    expanded[:, 4] = lo+(hi-lo)*base[parent, 4]
    probability = cuts.interval_probability[parent, stratum]/len(base)
    if not np.isfinite(expanded).all() or np.any((expanded < 0)|(expanded > 1)):
        raise ValueError("Mapped energy strata left the original CDF support")
    return EnergyStrata(_readonly(expanded), _readonly(probability),
        _readonly(parent, np.int64), _readonly(stratum, np.int64), _readonly(lo), _readonly(hi), cuts,
        metadata=dict(schema=SCHEMA, band_ev=cuts.metadata["band_ev"],
            independent_sobol_base_count=len(base), positive_energy_stratum_count=len(parent),
            omitted_zero_probability_strata=int(np.count_nonzero(cuts.interval_probability == 0.)),
            stratum_labels=["below inclusive band", "inside inclusive band", "above inclusive band"],
            weights="Each row carries its original Boersch CDF interval probability divided by M",
            azimuth_composition="Apply azimuth separately; each half-orbit gets base_probability/(K/2), then signs each get 1/2",
            prefix_scope="At fixed source parameters and band, base-major coordinates and labels have an exact M prefix; probabilities scale as 1/M",
            scope="Offline source quadrature helper only. Energy-band strata preserve the complete physical source measure; no aperture fitting, survivor normalization or physical transport validation."))
