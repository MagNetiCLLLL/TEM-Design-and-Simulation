"""Analytic straight paths on the unchanged homogenized rectangle pack.

Only the line integral changes representation: each original radial/fraction
Gauss sample retains its four closed straight sides and its prescribed NI.
This is not the original 24-node path rule and does not admit new query
domains, poses, material intersections or coherent operators.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import math
from numbers import Real
from pathlib import Path

import numpy as np
from scipy.constants import mu_0

from temsim.immutable_json import freeze_json
from .lens_field_provider import _identity_registration


SCHEMA = "analytic-rectangle-gauss-pack-v1"
MINIMUM_REFERENCE_PATH_ORDER = 24
MINIMUM_RADIUS_M = 1.e-6
MAXIMUM_ABSOLUTE_COORDINATE_M = 1.e4
MAXIMUM_SEGMENT_BORE_ASPECT = 16.
MINIMUM_ABSOLUTE_NONZERO_WEIGHT_TM = 1.e-100
MAXIMUM_ABSOLUTE_WEIGHT_TM = 1.e100
_FIELDS = frozenset((
    "family", "coil_index", "station_index", "group", "current_sign",
    "azimuth_deg", "center_z_mm", "radial_start_mm", "radial_end_mm",
    "outer_width_mm", "outer_height_mm", "window_width_mm", "window_height_mm",
))


def _readonly(values, shape):
    array = np.asarray(values)
    if array.dtype.kind not in "iuf" or array.shape != shape or not np.isfinite(array).all():
        raise ValueError("Analytic rectangle arrays require finite real SI coordinates and weights")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(shape)


def _geometry(starts, ends):
    """Continuous XY-segment distance, including tangential side interiors."""
    delta = ends-starts
    lengths = np.linalg.norm(delta, axis=1)
    planar = delta[:, :2]
    squared = np.einsum("ni,ni->n", planar, planar)
    parameter = np.zeros(len(starts), dtype=float)
    nonzero = squared != 0.
    parameter[nonzero] = np.clip(
        -np.einsum("ni,ni->n", starts[nonzero, :2], planar[nonzero])/squared[nonzero], 0., 1.)
    nearest = starts[:, :2]+parameter[:, None]*planar
    radius = np.linalg.norm(nearest, axis=1)
    return lengths, float(radius.min(initial=np.inf)), float(lengths.max(initial=0.))


@dataclass(frozen=True, slots=True)
class AnalyticRectanglePack:
    """Closed path endpoints in metres and scalar mu0*NI/(4pi) weights in T m.

    ``original_path_node_count`` describes nonzero-current nodes in the
    reference Gauss path representation. Zero-current sides are absent from
    both active representations; the caller retains all material/domain checks.
    """
    starts_m: np.ndarray
    ends_m: np.ndarray
    weights_t_m: np.ndarray
    minimum_radius_m: float
    maximum_length_m: float
    axis_zero: bool
    original_path_node_count: int
    pack_order: int
    reference_path_order: int

    def __post_init__(self):
        starts = np.asarray(self.starts_m)
        if starts.ndim != 2 or starts.shape[1:] != (3,):
            raise ValueError("Analytic rectangle endpoints must have shape (segments, 3)")
        count = len(starts)
        starts = _readonly(starts, (count, 3))
        ends = _readonly(self.ends_m, (count, 3))
        weights = _readonly(self.weights_t_m, (count,))
        if (isinstance(self.pack_order, (bool, np.bool_)) or not isinstance(self.pack_order, (int, np.integer))
                or not 2 <= self.pack_order <= 64 or isinstance(self.reference_path_order, (bool, np.bool_))
                or not isinstance(self.reference_path_order, (int, np.integer))
                or not MINIMUM_REFERENCE_PATH_ORDER <= self.reference_path_order <= 64):
            raise ValueError("Analytic rectangle pack/reference orders are invalid")
        if (isinstance(self.original_path_node_count, (bool, np.bool_))
                or not isinstance(self.original_path_node_count, (int, np.integer))
                or self.original_path_node_count != count*self.reference_path_order):
            raise ValueError("Analytic rectangle reference-node count is inconsistent")
        if not isinstance(self.axis_zero, (bool, np.bool_)):
            raise ValueError("Analytic rectangle axis identity must be explicit")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
               for value in (self.minimum_radius_m, self.maximum_length_m)):
            raise ValueError("Analytic rectangle geometry metadata requires real SI lengths")
        lengths, radius, maximum = _geometry(starts, ends)
        if (not np.isfinite(lengths).all() or np.any(lengths <= 0.)
                or self.minimum_radius_m != radius or self.maximum_length_m != maximum):
            raise ValueError("Analytic rectangle geometry metadata is inconsistent")
        if count and (radius < MINIMUM_RADIUS_M or maximum/(radius*.5) > MAXIMUM_SEGMENT_BORE_ASPECT):
            raise ValueError("Analytic rectangle path exceeds its numerical admission")
        object.__setattr__(self, "starts_m", starts)
        object.__setattr__(self, "ends_m", ends)
        object.__setattr__(self, "weights_t_m", weights)
        object.__setattr__(self, "axis_zero", bool(self.axis_zero))

    @property
    def segment_count(self):
        return len(self.starts_m)


@lru_cache(maxsize=1)
def analytic_rectangle_implementation_identity():
    return freeze_json(dict(schema=SCHEMA, implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        path="Analytic finite straight segments; original radial/fraction Gauss pack and NI",
        reference_path_order_minimum=MINIMUM_REFERENCE_PATH_ORDER,
        minimum_continuous_radius_m=MINIMUM_RADIUS_M,
        maximum_segment_to_half_bore_ratio=MAXIMUM_SEGMENT_BORE_ASPECT,
        absolute_coordinate_limit_m=MAXIMUM_ABSOLUTE_COORDINATE_M,
        nonzero_weight_range_t_m=[MINIMUM_ABSOLUTE_NONZERO_WEIGHT_TM, MAXIMUM_ABSOLUTE_WEIGHT_TM],
        unsupported="Retain the original Gauss path evaluator; no physical input or tolerance change"))


def pack_analytic_rectangle_potential(potential):
    """Return admitted immutable segments, or None for original-path execution.

    No field is queried here. In particular, this preparation does not bypass
    the original occupied-winding exclusion or any per-query accelerator guard.
    """
    from .transverse_coil_field import (
        MAXIMUM_QUADRATURE_NODES, _quadrature, _rotation,
        transverse_potential_implementation_is_supported,
    )
    from .radial_multipole_symmetry import complete_radial_multipole_axis_symmetry
    if (not transverse_potential_implementation_is_supported(potential)
            or not _identity_registration(potential.registration)
            or potential.path_order < MINIMUM_REFERENCE_PATH_ORDER
            or any(coil["family"] != "radial_rectangle" or set(coil) != _FIELDS for coil in potential.coils)):
        return None
    if potential.current_segments is not None:
        from temsim.ideal_transverse_segments import closed_layers
        layers=closed_layers(potential)
        if not (len(layers)==1 and layers[0] is potential):
            if len(layers)*len(potential.coils)*potential.pack_order**2*4*potential.path_order>MAXIMUM_QUADRATURE_NODES:
                return None
            packed=[pack_analytic_rectangle_potential(layer) for layer in layers]
            if any(row is None for row in packed):return None
            starts=np.concatenate([row.starts_m for row in packed])
            ends=np.concatenate([row.ends_m for row in packed])
            weights=np.concatenate([row.weights_t_m for row in packed])
            lengths,radius,maximum=_geometry(starts,ends)
            return AnalyticRectanglePack(starts,ends,weights,radius,maximum,
                all(row.axis_zero for row in packed),sum(row.original_path_node_count for row in packed),
                int(potential.pack_order),int(potential.path_order))
    requested = len(potential.coils)*potential.pack_order**2*4*potential.path_order
    if requested > MAXIMUM_QUADRATURE_NODES:
        return None
    starts, ends, weights = [], [], []
    fractions, fraction_weights = _quadrature(potential.pack_order, 0., 1.)
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for coil in potential.coils:
                current = potential.currents_a[0 if coil["group"] == "x" else 1]
                if current == 0.:
                    continue
                low, high = coil["radial_start_mm"]*.001, coil["radial_end_mm"]*.001
                if not np.isfinite((low, high)).all() or not 0. < low < high:
                    return None
                radial, radial_weights = _quadrature(potential.pack_order, low, high)
                rotation = _rotation(coil["azimuth_deg"])
                for radius, wr in zip(radial, radial_weights/(high-low)):
                    for fraction, wf in zip(fractions, fraction_weights):
                        # Match _rectangle/_unit_nodes/prototype arithmetic and
                        # ordering before replacing only the straight path rule.
                        width = (coil["window_width_mm"]+fraction*(coil["outer_width_mm"]-coil["window_width_mm"] ))*.001
                        height = (coil["window_height_mm"]+fraction*(coil["outer_height_mm"]-coil["window_height_mm"] ))*.001
                        center = coil["center_z_mm"]*.001
                        vertices = np.array([[radius, -width/2, center-height/2],
                            [radius, width/2, center-height/2], [radius, width/2, center+height/2],
                            [radius, -width/2, center+height/2]])@rotation.T
                        scalar = (potential.turn_count_per_coil*coil["current_sign"]*wr*wf*mu_0/(4*math.pi))*current
                        if (not math.isfinite(scalar) or scalar == 0.
                                or not MINIMUM_ABSOLUTE_NONZERO_WEIGHT_TM <= abs(scalar) <= MAXIMUM_ABSOLUTE_WEIGHT_TM):
                            return None
                        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
                            starts.append(start); ends.append(end); weights.append(scalar)
    except (ArithmeticError, ValueError):
        return None
    starts = np.asarray(starts, dtype=float).reshape(-1, 3)
    ends = np.asarray(ends, dtype=float).reshape(-1, 3)
    weights = np.asarray(weights, dtype=float)
    if (not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or np.max(abs(starts), initial=0.) > MAXIMUM_ABSOLUTE_COORDINATE_M
            or np.max(abs(ends), initial=0.) > MAXIMUM_ABSOLUTE_COORDINATE_M):
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lengths, radius, maximum = _geometry(starts, ends)
    if (not np.isfinite(lengths).all() or np.any(lengths <= 0.)
            or len(starts) and (radius < MINIMUM_RADIUS_M or maximum/(radius*.5) > MAXIMUM_SEGMENT_BORE_ASPECT)):
        return None
    return AnalyticRectanglePack(starts, ends, weights, radius, maximum,
        complete_radial_multipole_axis_symmetry(potential.coils, potential.provenance.get("multipole_order")),
        len(starts)*potential.path_order, int(potential.pack_order), int(potential.path_order))
