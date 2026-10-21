"""Conservative, display-only rasterisation of a completed particle plane.

Arrays use image order ``[y, x]``. Values are absolute source fractions per
pixel, not probabilities renormalised to the viewport. Linear deposition onto
pixel centres preserves flux and first moments when its support is visible.
Optional smoothing is a finite, symmetric display kernel, not a detector PSF
or a coherent wave calculation. The halo is calculated before viewport crop;
neither cropped tails nor intercepted electrons are redistributed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.gui.beam_plane_data import BeamPlaneData, _frozen


@dataclass(frozen=True, slots=True)
class BeamSpotRaster:
    raw_fraction: np.ndarray
    display_fraction: np.ndarray
    x_edges_m: np.ndarray
    y_edges_m: np.ndarray
    total_source_fraction: float
    viewport_source_fraction: float
    centroid_m: tuple[float, float] | None
    ray_count: int
    effective_ray_count: float
    smoothing_sigma_px: float
    source_current_pa: float | None

    @property
    def pixel_area_m2(self) -> float:
        return float((self.x_edges_m[1] - self.x_edges_m[0])
                     * (self.y_edges_m[1] - self.y_edges_m[0]))

    @property
    def raw_visible_fraction(self) -> float:
        return float(np.sum(self.raw_fraction, dtype=np.float64))

    @property
    def display_visible_fraction(self) -> float:
        return float(np.sum(self.display_fraction, dtype=np.float64))

    @property
    def current_pa(self) -> float | None:
        if self.source_current_pa is None:
            return None
        return self.source_current_pa * self.total_source_fraction


def _parameters(bins, smoothing_sigma_px, projection_angle_deg):
    try:
        if (isinstance(bins, (bool, np.bool_)) or int(bins) != bins
                or not 2 <= int(bins) <= 1024):
            raise ValueError
        bins = int(bins)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Spot grid must be an integer from 2 to 1024.") from None
    try:
        if isinstance(smoothing_sigma_px, (bool, np.bool_)):
            raise ValueError
        sigma = float(smoothing_sigma_px)
        if not math.isfinite(sigma) or not 0.0 <= sigma <= 8.0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Display smoothing must be finite and between 0 and 8 pixels.") from None
    try:
        angle = float(projection_angle_deg)
        if not math.isfinite(angle):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Projection angle must be finite.") from None
    return bins, sigma, math.radians(angle % 360.0)


def _axis_edges(bounds, bins):
    try:
        low, high = (float(value) for value in bounds)
        if not math.isfinite(low) or not math.isfinite(high) or not low < high:
            raise ValueError
        width = (high - low) / bins
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError
        edges = np.linspace(low, high, bins + 1)
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Spot bounds must be finite, increasing and numerically resolved.") from None
    return edges, width


def _deposit(x, y, weights, x_edges, y_edges, dx, dy, halo):
    """Deposit only bounded, potentially visible supports onto a padded grid."""
    size = len(x_edges) - 1 + 2 * halo
    expanded = (x_edges[0] - (halo + 1) * dx, x_edges[-1] + (halo + 1) * dx,
                y_edges[0] - (halo + 1) * dy, y_edges[-1] + (halo + 1) * dy)
    if not all(math.isfinite(float(value)) for value in expanded):
        raise ValueError("Spot bounds leave no finite room for display support.")
    nearby = ((x >= expanded[0]) & (x <= expanded[1])
              & (y >= expanded[2]) & (y <= expanded[3]) & (weights > 0.0))
    # Filter distant particles before integer conversion. Even a tiny viewport
    # around the origin can safely inspect a much wider, off-axis beam.
    gx = (x[nearby] - x_edges[0]) / dx - 0.5
    gy = (y[nearby] - y_edges[0]) / dy - 0.5
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = gx - ix, gy - iy
    # Add the halo in integer space: smoothing must not alter floating-point
    # interpolation fractions or the underlying unsmoothed raster.
    ix, iy = ix + halo, iy + halo
    weights = weights[nearby]
    raster = np.zeros((size, size), dtype=np.float64)
    for ox, oy, coefficient in (
        (0, 0, (1.0 - fx) * (1.0 - fy)),
        (1, 0, fx * (1.0 - fy)),
        (0, 1, (1.0 - fx) * fy),
        (1, 1, fx * fy),
    ):
        px, py = ix + ox, iy + oy
        inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        # bincount adds repeated indices; advanced-index += would lose hits.
        raster += np.bincount(
            py[inside] * size + px[inside],
            weights=weights[inside] * coefficient[inside],
            minlength=size * size,
        ).reshape(size, size)
    return raster


def rasterize_beam_spot(
    data: BeamPlaneData,
    bins: int = 128,
    *,
    projection_angle_deg: float = 0.0,
    range_m=None,
    smoothing_sigma_px: float = 0.0,
) -> BeamSpotRaster:
    """Return an immutable particle spot without changing the physical sample.

    ``range_m`` is ``((xmin, xmax), (ymin, ymax))`` in projected metres. Without
    explicit bounds, symmetric optical-axis bounds include the complete finite
    display kernel. ``centroid_m`` always refers to all reaching particles,
    including those outside the viewport; it is not inferred from the image.
    ``viewport_source_fraction`` counts actual hit centres in the closed view.
    The raw/display image sums instead include their respective kernel crops.
    All fractions retain the original emitted-source denominator.
    """
    bins, sigma, angle = _parameters(bins, smoothing_sigma_px, projection_angle_deg)
    if not data.weights_valid:
        raise ValueError("Source probabilities are unavailable for this plane.")
    x, y, weights = (np.asarray(values, dtype=np.float64)
                     for values in (data.x_m, data.y_m, data.source_fraction))
    if (x.ndim != 1 or y.shape != x.shape or weights.shape != x.shape
            or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y))
            or np.any(~np.isfinite(weights)) or np.any(weights < 0.0)):
        raise ValueError("Spot positions and source probabilities must be finite, aligned and nonnegative.")
    total = math.fsum(float(value) for value in weights)
    if total > 1.0 + 1.0e-10:
        raise ValueError("Reaching source probabilities exceed one.")
    c, s = math.cos(angle), math.sin(angle)
    with np.errstate(over="ignore", invalid="ignore"):
        x, y = c * x + s * y, -s * x + c * y
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Projected coordinates are not numerically representable.")
    centroid = None
    effective_count = 0.0
    if total > 0.0:
        normalised = weights / total  # Statistics only; never image weights.
        centroid = (float(np.dot(x, normalised)), float(np.dot(y, normalised)))
        scaled = weights / float(np.max(weights))
        effective_count = float(np.sum(scaled) ** 2 / np.dot(scaled, scaled))
    radius = int(math.ceil(4.0 * sigma))
    if range_m is None:
        interior = 1.0 - (2 * radius + 1) / bins
        if interior <= 0.0:
            raise ValueError("Spot grid is too small for full smoothing support; increase the grid or supply explicit bounds.")
        halves = [np.nextafter(max(float(np.max(np.abs(axis))) / interior if axis.size else 0.0,
                                  1.0e-15), math.inf) for axis in (x, y)]
        bounds = tuple((-half, half) for half in halves)
    else:
        try:
            bounds = tuple(range_m)
        except TypeError:
            raise ValueError("Spot bounds need X and Y ranges.") from None
        if len(bounds) != 2:
            raise ValueError("Spot bounds need X and Y ranges.")
    x_edges, dx = _axis_edges(bounds[0], bins)
    y_edges, dy = _axis_edges(bounds[1], bins)
    if not math.isfinite(dx * dy) or dx * dy <= 0.0:
        raise ValueError("Spot pixel area is not numerically representable.")
    viewport = ((x >= x_edges[0]) & (x <= x_edges[-1])
                & (y >= y_edges[0]) & (y <= y_edges[-1]))
    viewport_fraction = math.fsum(float(value) for value in weights[viewport])
    halo = radius + 1
    padded = _deposit(x, y, weights, x_edges, y_edges, dx, dy, halo)
    raw = _frozen(padded[halo:-halo, halo:-halo])
    if sigma > 0.0:
        from scipy.ndimage import convolve1d

        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= np.sum(kernel)  # Unit kernel mass, never viewport normalisation.
        smoothed = convolve1d(padded, kernel, axis=0, mode="constant", cval=0.0)
        smoothed = convolve1d(smoothed, kernel, axis=1, mode="constant", cval=0.0)
        display = _frozen(smoothed[halo:-halo, halo:-halo])
    else:
        display = raw
    return BeamSpotRaster(
        raw, display, _frozen(x_edges), _frozen(y_edges), total,
        viewport_fraction, centroid, int(weights.size), effective_count,
        sigma, data.source_current_pa,
    )
