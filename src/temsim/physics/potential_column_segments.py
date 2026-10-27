"""Field-aware checkpoints without truncating fields or renewing budgets.

A Gaussian lens has no finite field end. Its captured native entry and exit
are useful *nominal* checkpoints: all Gaussian tails and overlapping sources must
remain in the same snapshot on both sides. Compact native coils have actual
model support faces. A tilted face is not a common global-Z plane; its axial
centre is explicitly only a nominal checkpoint.

This module schedules storage/resampling opportunities, not optical actions.
It neither samples nor replaces a wave and has no authority to reset any
physical-loss, phase, support-error, timing, or mode-weight history.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from temsim.immutable_json import json_digest
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
from temsim.physics.dynamic_potential import DynamicPotentialSnapshot


@dataclass(frozen=True, slots=True)
class PotentialSegmentMarker:
    source_id: str
    z_m: float
    kind: str
    complete_global_z_face: bool

    def to_dict(self):
        return {"source_id": self.source_id, "z_m": self.z_m,
                "kind": self.kind,
                "complete_global_z_face": self.complete_global_z_face}


@dataclass(frozen=True, slots=True)
class PotentialColumnSegmentPlan:
    snapshot_digest: str
    start_z_m: float
    stop_z_m: float
    maximum_segment_m: float
    stops_m: tuple[float, ...]
    markers: tuple[PotentialSegmentMarker, ...]
    sources_without_axial_marker: tuple[str, ...]

    @property
    def digest(self):
        return json_digest(self.to_dict())

    def to_dict(self):
        return {"schema": "potential-column-segment-plan-v2",
                "snapshot_digest": self.snapshot_digest,
                "start_z_m": self.start_z_m, "stop_z_m": self.stop_z_m,
                "maximum_segment_m": self.maximum_segment_m,
                "stops_m": list(self.stops_m),
                "markers": [item.to_dict() for item in self.markers],
                "sources_without_axial_marker": list(self.sources_without_axial_marker),
                "field_policy": "retain the complete unchanged snapshot in every segment; no field removal",
                "history_policy": "continue original per-mode histories and weights; no budget renewal",
                "resampling_policy": "checkpoint opportunities only; complex-wave resampling requires separate admission"}


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real value in metres")
    return float(value)


def _registered_markers(source_id, registration, local_bounds_m, kind):
    rotation = registration.rotation_array
    origin_z = float(registration.origin_array_m[2])
    # Exact geometry comparison: a small nonzero tilt still creates an oblique
    # support face and must not acquire an exact-interface claim by tolerance.
    aligned = bool(np.array_equal(rotation[:2, 2], np.zeros(2))
                   and np.array_equal(rotation[2, :2], np.zeros(2)))
    endpoints = sorted(origin_z+float(value)*float(rotation[2, 2]) for value in local_bounds_m)
    if not np.isfinite(endpoints).all():
        raise ValueError("Field checkpoint registration exceeds the finite coordinate range")
    if kind == "gaussian":
        label = "nominal_gaussian" if aligned else "nominal_oblique_gaussian"
        return tuple(PotentialSegmentMarker(source_id, z, f"{label}_{side}", False)
                     for z, side in zip(endpoints, ("entry", "exit")))
    label = kind if aligned else f"nominal_oblique_{kind}"
    return tuple(PotentialSegmentMarker(source_id, z, f"{label}_{side}", aligned)
                 for z, side in zip(endpoints, ("entry", "exit")))


def _source_markers(row):
    from temsim.physics.analytic_potential import AnalyticMagneticPotential
    from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
    from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
    from temsim.physics.dynamic_potential import DynamicDipolePotential
    from temsim.physics.gun_magnetic_potential import GunMagneticPotential
    from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
    from temsim.physics.lens_field_provider import FrozenAnalyticField
    from temsim.physics.regular_axisymmetric_magnetic_potential import RegularAxisymmetricMagneticPotential

    provider = row.potential
    if type(provider) is GunMagneticPotential:
        # The native soft window is exactly zero beyond centre +/- half +/-
        # edge. Keep the same operation ordering as its domain predicate.
        return tuple(PotentialSegmentMarker(row.source_id, value, f"compact_support_{side}", True)
                     for center, half, edge, *_ in provider.terms
                     for value, side in ((center-half-edge, "entry"), (center+half+edge, "exit")))
    if type(provider) is DynamicDipolePotential:
        native, kind = provider.base.base.field, "compact_support"
    elif type(provider) is HighOrderRoundLensPotential:
        native = provider.native
        kind = "gaussian" if provider.axial_domain == "gaussian_all_z" else "validity_domain"
    elif type(provider) is AnalyticPotentialDomain:
        native = provider.base.field
        kind = "gaussian" if type(native) is FrozenAnalyticField else "compact_support"
    elif type(provider) is AnalyticMagneticPotential:
        native, kind = provider.field, "validity_domain"
    elif type(provider) in (AxisymmetricMagneticPotential, RegularAxisymmetricMagneticPotential):
        source = provider if type(provider) is AxisymmetricMagneticPotential else provider.source
        return _registered_markers(row.source_id, source.registration,
                                   (source.z_m[0], source.z_m[-1]), "validity_domain")
    else:
        # In particular, the FEG electric tail is not deleted or declared to
        # end where its tabulation ends. Its executor retains its own domain.
        return ()
    return _registered_markers(row.source_id, native.registration,
                               tuple(value*.001 for value in native.native_support_mm), kind)


def plan_potential_column_segments(snapshot, start_z_m, stop_z_m, *, maximum_segment_m):
    """Return deterministic metre checkpoints for a forward captured column.

    The supplied final float is retained exactly. Distinct representable
    marker planes are never epsilon-merged. Replanning a continuation may
    change checkpoint placement, so this is not a claim of bitwise equality
    between different integration partitions. The propagation operator still
    owns its usual step-size, convergence, domain, and history validation.
    """
    if type(snapshot) not in (ElectromagneticPotentialSnapshot, DynamicPotentialSnapshot):
        raise ValueError("Column segmentation requires an exact captured potential snapshot")
    start, stop, maximum = (_number(value, name) for value, name in (
        (start_z_m, "Segment start"), (stop_z_m, "Segment stop"),
        (maximum_segment_m, "Maximum segment length")))
    if stop < start or maximum <= 0 or not math.isfinite(stop-start):
        raise ValueError("Column segmentation requires a finite forward interval and positive maximum length")
    markers, missing = [], []
    for row in snapshot.contributions:
        source_markers = _source_markers(row)
        if not source_markers:
            missing.append(row.source_id)
        markers.extend(item for item in source_markers if start < item.z_m <= stop)
    markers.sort(key=lambda item: (item.z_m, item.source_id, item.kind))
    boundaries = sorted({item.z_m for item in markers} | ({stop} if stop > start else set()))
    stops, lower = [], start
    for upper in boundaries:
        ratio = (upper-lower)/maximum
        if not math.isfinite(ratio) or ratio > 1_000_000-len(stops):
            raise ValueError("Column segment schedule exceeds one million checkpoints")
        count = max(1, math.ceil(ratio))
        # Floating-point subtraction can exceed the maximum by one ULP even
        # when the mathematical equal subdivision meets it. Increasing count
        # avoids inventing tiny terminal intervals or relaxing the maximum.
        while True:
            candidate = tuple(lower+(upper-lower)*(index/count) for index in range(1, count))+(upper,)
            differences = np.diff((lower, *candidate))
            if np.any(differences <= 0.):
                raise ValueError("Column segment length cannot advance the represented coordinate")
            if float(np.max(differences)) <= maximum:
                break
            count += 1
            if count > 1_000_000-len(stops):
                raise ValueError("Column segment schedule exceeds one million checkpoints")
        stops.extend(candidate)
        lower = upper
    return PotentialColumnSegmentPlan(snapshot.numerical_digest, start, stop, maximum,
                                      tuple(stops), tuple(markers), tuple(sorted(missing)))


__all__ = ("PotentialSegmentMarker", "PotentialColumnSegmentPlan", "plan_potential_column_segments")
