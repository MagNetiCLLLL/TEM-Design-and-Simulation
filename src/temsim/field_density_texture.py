"""CPU-only interpolation and screen-space hatching of section field magnitude.

No field solver or GUI dependency is used here. Callers own evaluation batch
limits and should capture ``section_grid_axes`` before applying physical masks.
The original axes prevent interpolation across a removed grid row or column.
"""
from __future__ import annotations

import numpy as np


_SECTION_AXES = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0), "UZ": (0, 2, 1)}

# Every denser pattern contains the preceding pattern. Screen coordinates and
# spacing use logical pixels, independently of the physical sample-grid density.
HATCH_THRESHOLDS = (1 / 8, 1 / 4, 1 / 2, 1., 2.)
HATCH_SPACINGS_PX = (128., 64., 32., 16., 8., 4.)


def _points_array(points_m):
    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Section points must have shape (N, 3)")
    return points


def _rotation(angle_deg):
    angle_deg = float(angle_deg)
    if not np.isfinite(angle_deg):
        raise ValueError("Longitudinal rotation angle must be finite")
    angle = np.deg2rad(angle_deg % 360.)
    return np.cos(angle), np.sin(angle)


def to_longitudinal(points, angle_deg):
    """Return U=X cos(theta)+Y sin(theta), V=-X sin(theta)+Y cos(theta), Z.

    The angle is in degrees. A nonzero V is preserved; this transform alone
    makes no assertion that the points belong to a section.
    """
    points = _points_array(points)
    cosine, sine = _rotation(angle_deg)
    local = points.copy()
    with np.errstate(invalid="ignore", over="ignore"):
        local[:, 0] = points[:, 0] * cosine + points[:, 1] * sine
        local[:, 1] = -points[:, 0] * sine + points[:, 1] * cosine
    return local


def from_longitudinal(points, angle_deg):
    """Rotate local UVZ into global XYZ, preserving the full local V offset."""
    points = _points_array(points)
    cosine, sine = _rotation(angle_deg)
    global_points = points.copy()
    with np.errstate(invalid="ignore", over="ignore"):
        global_points[:, 0] = points[:, 0] * cosine - points[:, 1] * sine
        global_points[:, 1] = points[:, 0] * sine + points[:, 1] * cosine
    return global_points


def _longitudinal_section_points(points, angle_deg):
    local = to_longitudinal(points, angle_deg)
    # Use each point's transverse scale: a distant sample must not make a
    # genuine near-axis V offset look like roundoff. Z cannot enlarge this
    # tolerance because rotation leaves Z unchanged.
    tolerance = 64 * np.finfo(float).eps * (np.abs(points[:, 0]) + np.abs(points[:, 1]))
    on_plane = np.isfinite(local[:, 1]) & (np.abs(local[:, 1]) <= tolerance)
    local[on_plane, 1] = 0.
    return local, tolerance


def _nearest_axis_indices(axis, coordinates):
    upper = np.clip(np.searchsorted(axis, coordinates), 0, len(axis) - 1)
    lower = np.maximum(upper - 1, 0)
    use_lower = np.abs(coordinates - axis[lower]) < np.abs(coordinates - axis[upper])
    return np.where(use_lower, lower, upper)


def _regular_axis(axis):
    if len(axis) < 2:
        return False
    steps = np.diff(axis)
    tolerance = 32 * np.finfo(float).eps * max(float(np.max(np.abs(axis))), 1e-300)
    return bool(np.allclose(steps, steps[0], rtol=1e-9, atol=tolerance))


def section_grid_axes(points_m, section, *, angle_deg=0.):
    """Return sorted physical coordinate axes for a complete regular section.

    The tuple contains coordinate arrays, ordered X/Y, X/Z, Y/Z, or U/Z. UZ
    points are supplied in global XYZ and must lie on the rotated V=0 plane;
    only residuals within 64 machine epsilons of each point's XY scale are
    normalized to zero. A 3D,
    nonplanar, incomplete, duplicated, or degenerate grid returns ``None``.
    Pass unfiltered points: surviving points alone cannot reveal a uniformly
    removed set of whole rows or columns.
    """
    axes = _SECTION_AXES.get(section)
    if axes is None:
        return None
    points = _points_array(points_m)
    if not len(points) or not np.isfinite(points).all():
        return None
    if section == "UZ":
        points, _tolerance = _longitudinal_section_points(points, angle_deg)
        if not np.all(points[:, 1] == 0.):
            return None
    first, second, normal = axes
    if not np.all(points[:, normal] == points[0, normal]):
        return None
    axis0, axis1 = np.unique(points[:, first]), np.unique(points[:, second])
    if not _regular_axis(axis0) or not _regular_axis(axis1):
        return None
    if len(points) != len(axis0) * len(axis1):
        return None
    indices = np.searchsorted(axis0, points[:, first]) * len(axis1)
    indices += np.searchsorted(axis1, points[:, second])
    if len(np.unique(indices)) != len(points):
        return None
    return axis0, axis1


class SectionMagnitudeGrid:
    """Bilinearly interpolate complete vectors, then return their magnitude.

    ``grid_axes_m`` is the pair of coordinate arrays captured before filtering.
    Missing samples remain unsupported; present samples with a false ``valid``
    flag or nonfinite vector remain supported but invalid. All four corners
    must be supported and valid, even at a cell edge with zero-weight corners.
    Unknown magnitudes are NaN, distinct from a valid zero field.

    Without metadata only a complete regular Cartesian grid is accepted.
    Explicit metadata may describe a nonuniform rectilinear grid. Evaluation
    outside the sampled rectangle or section plane is unsupported. A 3D or
    unidentifiable grid evaluates as unsupported everywhere.

    For UZ, input and query coordinates remain global XYZ; ``angle_deg``
    defines the V=0 plane. Grid metadata uses local U/Z axes. Coordinates are
    transformed for interpolation, while vector components remain global XYZ.
    """

    def __init__(self, points_m, vectors, valid, section, grid_axes_m=None, *, angle_deg=0.):
        points = _points_array(points_m)
        vectors = np.asarray(vectors, dtype=float)
        valid = np.asarray(valid, dtype=bool)
        if vectors.shape != points.shape or valid.shape != (len(points),):
            raise ValueError("Vectors and validity must match the section points")
        self._axes = _SECTION_AXES.get(section)
        self._longitudinal = section == "UZ"
        self._angle_deg = angle_deg
        self._grid_axes = None
        if self._axes is None or not len(points):
            return
        if not np.isfinite(points).all():
            raise ValueError("Section point coordinates must be finite")
        if self._longitudinal:
            points, coordinate_tolerance = _longitudinal_section_points(points, angle_deg)
            if not np.all(points[:, 1] == 0.):
                return
        first, second, normal = self._axes
        if not np.all(points[:, normal] == points[0, normal]):
            return
        if grid_axes_m is None:
            grid_axes_m = section_grid_axes(points, "XZ" if self._longitudinal else section)
            if grid_axes_m is None:
                return
        if len(grid_axes_m) != 2:
            raise ValueError("Grid metadata must contain two coordinate axes")
        axis0, axis1 = (np.asarray(axis, dtype=float).copy() for axis in grid_axes_m)
        for axis in (axis0, axis1):
            if (axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
                    or not np.all(np.diff(axis) > 0)):
                raise ValueError("Grid axes must be finite and strictly increasing")
        index0 = np.searchsorted(axis0, points[:, first])
        index1 = np.searchsorted(axis1, points[:, second])
        if self._longitudinal:
            index0 = _nearest_axis_indices(axis0, points[:, first])
            close = np.abs(axis0[index0] - points[:, first]) <= coordinate_tolerance
            if not np.all(close):
                raise ValueError("Section points must belong to the supplied grid axes")
            points[:, first] = axis0[index0]
        if (np.any(index0 >= len(axis0)) or np.any(index1 >= len(axis1))
                or not np.array_equal(axis0[index0], points[:, first])
                or not np.array_equal(axis1[index1], points[:, second])):
            raise ValueError("Section points must belong to the supplied grid axes")
        if len(np.unique(index0 * len(axis1) + index1)) != len(points):
            raise ValueError("Section grid points must be unique")
        self._grid_axes = (axis0, axis1)
        self._plane = float(points[0, normal])
        self._plane_tolerance = (0. if self._longitudinal else
                                 32 * np.finfo(float).eps * max(abs(self._plane), 1e-12))
        shape = (len(axis0), len(axis1))
        self._supported = np.zeros(shape, dtype=bool)
        self._valid = np.zeros(shape, dtype=bool)
        self._vectors = np.zeros((*shape, 3), dtype=float)
        good = valid & np.isfinite(vectors).all(axis=1)
        self._supported[index0, index1] = True
        self._valid[index0, index1] = good
        self._vectors[index0, index1] = np.where(good[:, None], vectors, 0.)

    def evaluate(self, points_m):
        """Return 1D ``(magnitudes, supported, valid)`` arrays for query XYZ."""
        points = _points_array(points_m)
        magnitudes = np.full(len(points), np.nan, dtype=float)
        supported = np.zeros(len(points), dtype=bool)
        valid = np.zeros(len(points), dtype=bool)
        if self._grid_axes is None or not len(points):
            return magnitudes, supported, valid
        first, second, normal = self._axes
        axis0, axis1 = self._grid_axes
        if self._longitudinal:
            points, coordinate_tolerance = _longitudinal_section_points(points, self._angle_deg)
            # A second inverse rotation may move an endpoint or a cell knot by
            # a few ulps. Snap only that bounded error, retaining real offsets.
            nearest = _nearest_axis_indices(axis0, points[:, first])
            close = np.abs(axis0[nearest] - points[:, first]) <= coordinate_tolerance
            points[close, first] = axis0[nearest[close]]
        inside = (np.isfinite(points).all(axis=1)
                  & (points[:, first] >= axis0[0]) & (points[:, first] <= axis0[-1])
                  & (points[:, second] >= axis1[0]) & (points[:, second] <= axis1[-1])
                  & (np.abs(points[:, normal] - self._plane) <= self._plane_tolerance))
        rows = np.flatnonzero(inside)
        if not len(rows):
            return magnitudes, supported, valid
        x, y = points[rows, first], points[rows, second]
        i = np.clip(np.searchsorted(axis0, x, side="right") - 1, 0, len(axis0) - 2)
        j = np.clip(np.searchsorted(axis1, y, side="right") - 1, 0, len(axis1) - 2)
        corners = ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))
        supported[rows] = np.logical_and.reduce([self._supported[a, b] for a, b in corners])
        valid[rows] = supported[rows] & np.logical_and.reduce(
            [self._valid[a, b] for a, b in corners])
        good = valid[rows]
        if np.any(good):
            rows, i, j, x, y = rows[good], i[good], j[good], x[good], y[good]
            tx = ((x - axis0[i]) / (axis0[i + 1] - axis0[i]))[:, None]
            ty = ((y - axis1[j]) / (axis1[j + 1] - axis1[j]))[:, None]
            # Difference form preserves a constant vector exactly. Otherwise
            # rounding at a hatch threshold can turn LOD changes into speckles.
            lower = self._vectors[i, j] + tx * (self._vectors[i + 1, j] - self._vectors[i, j])
            upper = self._vectors[i, j + 1] + tx * (
                self._vectors[i + 1, j + 1] - self._vectors[i, j + 1])
            interpolated = lower + ty * (upper - lower)
            magnitudes[rows] = np.hypot.reduce(interpolated, axis=1)
        return magnitudes, supported, valid


def hatch_mask(magnitude, reference, screen_x, screen_y):
    """Return a broadcast bool mask for deterministic double-diagonal hatching.

    Positive magnitude/reference selects spacings 128, 64, 32, 16, 8, 4 logical
    pixels, with transitions at 1/8, 1/4, 1/2, 1, and 2. Ratios >= 2 saturate
    at 4 pixels. Zero, negative, or nonfinite magnitude has no marks. Each
    diagonal uses a one-pixel phase interval; nested spacings guarantee that
    increasing magnitude never removes a mark at a fixed screen coordinate.
    Reference must be a finite positive scalar and must stay fixed when
    comparing views. The caller supplies absolute logical screen coordinates,
    so batching, sample counts, zoom, and image-buffer strides cannot rescale
    the pattern.
    """
    reference = float(reference)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("Hatch reference must be finite and positive")
    magnitude, screen_x, screen_y = np.broadcast_arrays(
        np.asarray(magnitude, dtype=float), np.asarray(screen_x, dtype=float),
        np.asarray(screen_y, dtype=float))
    usable = (np.isfinite(magnitude) & (magnitude > 0)
              & np.isfinite(screen_x) & np.isfinite(screen_y))
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = magnitude / reference
        levels = np.searchsorted(HATCH_THRESHOLDS, ratio, side="right")
        spacing = np.asarray(HATCH_SPACINGS_PX)[levels]
        hatch = (np.remainder(screen_x + screen_y, spacing) < 1.)
        hatch |= (np.remainder(screen_x - screen_y, spacing) < 1.)
    return usable & hatch
