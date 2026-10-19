"""Physical Camera/screen images from retained first-interception records.

No rays are traced here. Pixels contain probability per emitted electron,
not probability density and not a count of numerical rays. Expected incident
electrons use the captured source current and an explicit exposure. Detector
PSF is the configured forward response; electronic noise/QE are not inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.component_keys import CAMERA, FLUORESCENT_SCREEN
from temsim.detector.particle_readout import measure_particle_detectors
from temsim.detector.compact_gaussian_response import compact_gaussian_response
from temsim.detector.point_spread import DetectorPointSpread, apply_point_spread


@dataclass(frozen=True)
class ReceiverImage:
    key: str
    name: str
    status: str
    detail: str
    ideal_probability: np.ndarray
    response_probability: np.ndarray | None
    x_mm: np.ndarray
    y_mm: np.ndarray
    exposure_s: float | None
    expected_electrons: np.ndarray | None
    current_pa: float | None
    metrics: dict

    def __post_init__(self):
        for name in ("ideal_probability", "response_probability", "x_mm", "y_mm",
                     "expected_electrons"):
            value = getattr(self, name)
            if value is not None:
                array = np.array(value, dtype=float, copy=True)
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "metrics", dict(self.metrics))


def available_receivers(state):
    """Installed imaging surfaces, including ones currently retracted.

    Selection never inserts a receiver or changes the active optical target.
    """
    return tuple(plane for plane in getattr(state, "recording_planes", ())
                 if str(plane.key) in {CAMERA, FLUORESCENT_SCREEN}
                 and bool(getattr(plane, "installed", True))
                 and bool(getattr(plane, "_layout_installed", True)))


_STATUS_DETAILS = {
    "NOT_INSERTED": "The selected receiver was retracted in this calculation.",
    "READOUT_DISABLED": "The selected receiver readout was disabled in this calculation.",
    "NOT_REACHED": "The retained calculation does not reach this receiver.",
    "NOT_CALCULATED": "No executed physical particle reception is available; calculate the particle path first.",
}


def _positions_at(branch, z_mm, count):
    z = np.asarray(branch.z, dtype=float)
    if (z.ndim != 1 or not z.size or not np.all(np.isfinite(z))
            or np.any(np.diff(z) <= 0.0)):
        raise ValueError("Receiver image requires ordered finite trajectory planes")
    if z_mm < z[0] - 1e-9 or z_mm > z[-1] + 1e-9:
        raise ValueError("Receiver image cannot extrapolate a retained trajectory")
    high = min(int(np.searchsorted(z, z_mm)), len(z) - 1)
    low = high if high == 0 or abs(float(z[high]) - z_mm) <= 1e-9 else high - 1
    fraction = 0.0 if high == low else (z_mm - z[low]) / (z[high] - z[low])
    positions = []
    for name in ("x", "y"):
        values = np.asarray(getattr(branch, name), dtype=float)
        if values.shape != (len(z), count):
            raise ValueError("Receiver coordinates do not match the particle trajectory")
        positions.append((values[low] + fraction * (values[high] - values[low])) * 1e3)
    return np.column_stack(positions)


def _display_extent(extent, physical):
    if extent is None:
        return physical
    values = np.asarray(extent, dtype=float)
    if (values.shape != (4,) or not np.isfinite(values).all()
            or values[0] >= values[1] or values[2] >= values[3]):
        raise ValueError("Receiver display extent must contain finite increasing X/Y bounds")
    clipped = (max(values[0], physical[0]), min(values[1], physical[1]),
               max(values[2], physical[2]), min(values[3], physical[3]))
    if clipped[0] >= clipped[1] or clipped[2] >= clipped[3]:
        raise ValueError("Receiver display extent does not overlap the physical receiver")
    return tuple(float(value) for value in clipped)


def _response_from_hits(positions, weights, x_edges, y_edges, point_spread, *, diagnostics=None):
    """Include response from outside the viewport without normalizing the crop.

    Compact axis-aligned Gaussian clouds use bounded sixth-order pixel
    integrals that retain their physical moments. Other small axis-aligned
    requests use direct Gaussian pixel integrals; remaining requests use a
    bounded convolution grid. Sub-PSF display pixels sample the smooth response
    density; they never narrow the configured physical PSF. Optional diagnostics
    report the compact expansion and its truncation bound, excluding roundoff.
    """
    if diagnostics is not None:
        diagnostics.clear()
    if not len(weights):
        return np.zeros((len(y_edges)-1, len(x_edges)-1)), "No physical hits", None
    compact = compact_gaussian_response(positions, weights, x_edges, y_edges, point_spread)
    if compact is not None:
        if diagnostics is not None:
            diagnostics.update(method="compact_gaussian_moments", **compact.metadata)
        return compact.probability, "Compact sixth-order Gaussian pixel integrals from physical hits", None
    angle = math.radians(point_spread.rotation_deg)
    c, s = math.cos(angle), math.sin(angle)
    a, b = point_spread.sigma_x_mm**2, point_spread.sigma_y_mm**2
    xx, yy, xy = c*c*a+s*s*b, s*s*a+c*c*b, c*s*(a-b)
    n = len(x_edges)-1
    if abs(xy) <= 1e-13*math.sqrt(xx*yy) and n*n*len(weights) <= 16*1024**2:
        from scipy.special import ndtr
        response = np.zeros((n, n), dtype=float)
        def mass(edges, centres, sigma):
            values = (edges[:, None]-centres[None, :])/sigma
            low, high = values[:-1], values[1:]
            return np.maximum(np.where(low > 0, ndtr(-low)-ndtr(-high),
                                       ndtr(high)-ndtr(low)), 0.)
        for start in range(0, len(weights), 128):
            stop = start+128
            mx = mass(x_edges, positions[start:stop, 0], math.sqrt(xx))
            my = mass(y_edges, positions[start:stop, 1], math.sqrt(yy))
            response += (my*weights[None, start:stop]) @ mx.T
        return response, "Gaussian pixel integrals from all physical hits", None
    dx, dy = x_edges[1]-x_edges[0], y_edges[1]-y_edges[0]
    sigma = max(point_spread.sigma_x_mm, point_spread.sigma_y_mm)
    # The physical response is smooth on the smallest PSF principal width.
    # Fine viewports use sigma/24; coarser viewports retain their pixel pitch.
    # This bounds work independently of a nanometre-sized view on an 80 mm screen.
    minimum_pitch = min(point_spread.sigma_x_mm, point_spread.sigma_y_mm)/24
    wx, wy = max(dx, minimum_pitch), max(dy, minimum_pitch)
    cx, cy = math.ceil((x_edges[-1]-x_edges[0])/wx), math.ceil((y_edges[-1]-y_edges[0])/wy)
    hx, hy = math.ceil(4*sigma/wx)+1, math.ceil(4*sigma/wy)+1
    nx, ny = cx+2*hx, cy+2*hy
    if nx*ny > 4*1024**2:
        raise ValueError("Detector PSF exceeds the bounded display grid; widen the view or inspect particle arrivals without Detector blur")
    extended_x = x_edges[0]+np.arange(-hx, cx+hx+1)*wx
    extended_y = y_edges[0]+np.arange(-hy, cy+hy+1)*wy
    # Linear deposition preserves sub-cell hit displacements and first
    # moments on the response grid, before convolution and interpolation.
    gx = (positions[:, 0]-extended_x[0])/wx-.5
    gy = (positions[:, 1]-extended_y[0])/wy-.5
    nearby = (gx >= -1) & (gx < nx) & (gy >= -1) & (gy < ny)
    gx, gy, nearby_weights = gx[nearby], gy[nearby], weights[nearby]
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = gx-ix, gy-iy
    padded = np.zeros((ny, nx))
    for ox, oy, fraction in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                              (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        cols, rows = ix+ox, iy+oy
        valid = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
        padded += np.bincount(rows[valid]*nx+cols[valid],
            weights=nearby_weights[valid]*fraction[valid], minlength=nx*ny).reshape(ny, nx)
    response = apply_point_spread(padded, point_spread,
                                 pixel_size_x_mm=wx, pixel_size_y_mm=wy)
    if wx == dx and wy == dy:
        return response[hy:hy+n, hx:hx+n], "Sampled Gaussian with four-sigma viewport halo", (wx, wy)
    from scipy.ndimage import map_coordinates
    ix = (.5*(x_edges[:-1]+x_edges[1:])-extended_x[0])/wx-.5
    iy = (.5*(y_edges[:-1]+y_edges[1:])-extended_y[0])/wy-.5
    grid_x, grid_y = np.meshgrid(ix, iy)
    sampled = map_coordinates(response, (grid_y, grid_x), order=1, mode="constant", cval=0.)
    return sampled*(dx*dy/(wx*wy)), "Bounded sampled Gaussian density; grid pitch is max(view pixel pitch, minimum PSF sigma/24) per axis", (wx, wy)


def receiver_image(result, key, *, pixels=512, exposure_s=1.0, extent_mm=None,
                   include_response=True):
    """Histogram first physical hits on one captured Camera or screen.

    The default grid covers the complete active-area bounding square in mm.
    An optional viewport is rebinned from those same physical hits. It never
    changes intercepted current or renormalizes cropped probability. Missing
    calculations have an explicit unavailable status,
    whereas a calculated but blocked/empty receiver has AVAILABLE zero pixels.
    ``include_response=False`` returns only arrivals; no detector-response or
    response-electron array is manufactured for that request.
    """
    if isinstance(pixels, bool) or int(pixels) != pixels or not 2 <= int(pixels) <= 4096:
        raise ValueError("Receiver image pixels must be an integer from 2 to 4096")
    n = int(pixels)
    if exposure_s is not None:
        exposure_s = float(exposure_s)
        if not math.isfinite(exposure_s) or exposure_s <= 0.0:
            raise ValueError("Receiver exposure must be positive and finite")
    state = getattr(result, "state_snapshot", None)
    plane = next((p for p in available_receivers(state) if str(p.key) == str(key)), None)
    if plane is None:
        raise ValueError(f"Unknown or uninstalled imaging receiver: {key}")
    width = float(plane.outer_width_mm)
    centre = np.array([getattr(plane, "centre_offset_x_mm", 0.0),
                       getattr(plane, "centre_offset_y_mm", 0.0)], dtype=float)
    z_mm = float(plane.z_mm)
    if (not math.isfinite(width) or width <= 0.0 or not np.all(np.isfinite(centre))
            or not math.isfinite(z_mm)):
        raise ValueError("Receiver dimensions and position must be finite with positive width")
    physical = (float(centre[0]-.5*width), float(centre[0]+.5*width),
                float(centre[1]-.5*width), float(centre[1]+.5*width))
    extent = _display_extent(extent_mm, physical)
    x_edges = np.linspace(extent[0], extent[1], n+1)
    y_edges = np.linspace(extent[2], extent[3], n+1)
    if np.any(np.diff(x_edges) <= 0.) or np.any(np.diff(y_edges) <= 0.):
        raise ValueError("Receiver display extent is below floating-point coordinate resolution")
    dx, dy = (extent[1]-extent[0])/n, (extent[3]-extent[2])/n
    x_mm, y_mm = 0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:])
    ideal = np.zeros((n, n), dtype=float)
    simulation = getattr(result, "simulation", None)
    captured_metrics = getattr(simulation, "metrics", {})
    metrics = {
        "model": "retained_physical_first_hits_v1",
        "probability_convention": "probability per emitted electron per pixel",
        "axis_convention": "array[y, x]; laboratory X/Y; pixel centres in mm",
        "receiver_key": str(plane.key),
        "receiver_z_mm": z_mm,
        "receiver_width_mm": width,
        "receiver_inserted": bool(getattr(plane, "inserted", False)),
        "capture_workflow": str(getattr(result, "workflow", "")),
        "capture_request_identity": dict(getattr(result, "signatures", {})).get("request", "UNRECORDED"),
        "capture_manifest_identity": getattr(getattr(result, "calculation_manifest", None), "digest", "UNRECORDED"),
        "capture_model_signature": str(getattr(result, "model_signature", "")),
        "section_target_z_mm": captured_metrics.get("section_target_z_mm"),
        "section_physics_scope": captured_metrics.get("section_physics_scope"),
        "projector_mode": str(getattr(state, "projector_mode", "")),
        "exposure_s": exposure_s,
        "physical_area_extent_mm": physical,
        "display_extent_mm": extent,
        "pixel_size_x_mm": dx,
        "pixel_size_y_mm": dy,
        "display_pixels": n,
        "display_scope": "viewport rebin of retained hits; current remains the full receiver current",
        "response_scope": ("incident-electron expectation with configured PSF; no electronic noise or QE model"
                           if include_response else "particle arrivals only; detector response not computed"),
    }

    def unavailable(status):
        empty = np.zeros_like(ideal)
        return ReceiverImage(str(plane.key), str(plane.name), status, _STATUS_DETAILS[status],
                             empty, empty if include_response else None,
                             x_mm, y_mm, exposure_s, None, None, metrics)

    if not bool(getattr(plane, "inserted", False)):
        return unavailable("NOT_INSERTED")
    if not bool(getattr(plane, "readout_enabled", True)):
        return unavailable("READOUT_DISABLED")
    if (simulation is None or str(getattr(result, "workflow", "")) == "rays"
            or bool(captured_metrics.get("optical_tuning", False))):
        return unavailable("NOT_CALCULATED")
    row = next(row for row in measure_particle_detectors(result, exposure_s=exposure_s)
               if row.key == str(plane.key))
    if row.status != "AVAILABLE":
        return unavailable(row.status)

    exit_state = getattr(result, "specimen_exit", None)
    outgoing = tuple(exit_state.branches) if exit_state is not None else tuple(simulation.branches.values())
    branches = (simulation.incident,) if z_mm <= float(state.sample.z_mm) else outgoing
    if not branches:
        from temsim.physics.beam_current import sample_illumination_absent
        if not sample_illumination_absent(simulation, state):
            return unavailable("NOT_CALCULATED")
    metrics["particle_source"] = "specimen_exit" if exit_state is not None else "simulation"
    hit_positions, hit_weights = [], []
    for branch in branches:
        z = np.asarray(branch.z, dtype=float)
        if not z.size or not float(z[0]) - 1e-9 <= z_mm <= float(z[-1]) + 1e-9:
            alive = np.asarray(branch.alive, dtype=bool)
            weights = np.asarray(branch.ray_weight, dtype=float)
            if np.any(alive & (weights > 0.0)) and float(branch.weight) > 0.0:
                return unavailable("NOT_REACHED")
            continue
        weights = np.asarray(branch.ray_weight, dtype=float)
        stops = np.asarray(branch.blocked_key, dtype=str)
        blocked = np.asarray(branch.blocked_z, dtype=float)
        # measure_particle_detectors has already validated the source weights.
        hit = (stops == str(plane.key)) & np.isfinite(blocked)
        if not np.any(hit):
            continue
        if np.any(np.abs(blocked[hit] - z_mm) > 1e-7):
            raise ValueError("Recorded receiver stop position differs from its captured physical plane")
        positions = _positions_at(branch, z_mm, len(weights))[hit]
        if not np.all(np.isfinite(positions)):
            raise ValueError("Physical receiver hits must have finite positions")
        accepted = np.asarray(plane.hit_mask(positions[:, 0], positions[:, 1]), dtype=bool)
        if not np.all(accepted):
            raise ValueError("Recorded receiver hits lie outside its captured active area")
        probability = 1.0 if branch is simulation.incident else float(branch.weight)
        hit_positions.append(positions)
        hit_weights.append(weights[hit]*probability)
    positions = np.concatenate(hit_positions) if hit_positions else np.empty((0, 2))
    weights = np.concatenate(hit_weights) if hit_weights else np.empty(0)
    intercepted = float(weights.sum())
    if not math.isclose(intercepted, float(row.fraction), rel_tol=2e-10, abs_tol=1e-12):
        raise ValueError("Receiver image does not conserve the recorded intercepted probability")
    ideal = np.histogram2d(positions[:, 1], positions[:, 0],
                          bins=(y_edges, x_edges), weights=weights)[0]
    point_spread = DetectorPointSpread.from_component(plane)
    response = ideal.copy() if include_response else None
    psf_method = "No detector blur" if include_response else "Not computed; particle arrivals requested"
    response_pitch = None
    response_diagnostics = {}
    if include_response and point_spread.enabled:
        response, psf_method, response_pitch = _response_from_hits(
            positions, weights, x_edges, y_edges, point_spread,
            diagnostics=response_diagnostics)
        # Without spreading all binned hits have already passed the exact
        # physical mask, including pixels straddling a curved screen edge.
        active = np.asarray(plane.hit_mask(x_mm[None, :], y_mm[:, None]), dtype=bool)
        response = np.where(active, response, 0.0)
    from temsim.physics.beam_current import effective_source_current_pa
    source_current_pa = float(effective_source_current_pa(state))
    if not math.isfinite(source_current_pa) or source_current_pa < 0.0:
        raise ValueError("Captured source current must be finite and non-negative")
    expected = (None if exposure_s is None or response is None else
                response * source_current_pa * 1e-12 / 1.602176634e-19 * exposure_s)
    occupied = positions[weights > 0]
    bounds = (None if not len(occupied) else
              (float(occupied[:, 0].min()), float(occupied[:, 0].max()),
               float(occupied[:, 1].min()), float(occupied[:, 1].max())))
    # Sampling diagnostics use the retained hit records and their physical
    # weights, never blurred pixels or an inferred Poisson count. Sparse bin
    # accumulation avoids another full display-sized diagnostic image.
    in_view = ((positions[:, 0] >= x_edges[0]) & (positions[:, 0] <= x_edges[-1])
               & (positions[:, 1] >= y_edges[0]) & (positions[:, 1] <= y_edges[-1]))
    positive = weights > 0.
    selected = in_view & positive
    ix = np.minimum(np.searchsorted(x_edges, positions[selected, 0], side="right")-1, n-1)
    iy = np.minimum(np.searchsorted(y_edges, positions[selected, 1], side="right")-1, n-1)
    bins, inverse = np.unique(iy*n+ix, return_inverse=True)
    bin_counts = np.bincount(inverse, minlength=len(bins))
    bin_weights = np.bincount(inverse, weights=weights[selected], minlength=len(bins))
    bin_squares = np.bincount(inverse, weights=weights[selected]**2, minlength=len(bins))
    effective = np.divide(bin_weights**2, bin_squares,
                          out=np.zeros_like(bin_weights, dtype=float), where=bin_squares > 0.)

    def summary(values):
        return {"minimum": float(np.min(values)) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "maximum": float(np.max(values)) if len(values) else None,
                "mean": float(np.mean(values)) if len(values) else None}

    weight_square_sum = float(np.sum(weights**2))
    view_weight_sum = float(np.sum(weights[in_view]))
    view_weight_square_sum = float(np.sum(weights[in_view]**2))
    metrics["sampling"] = dict(
        raw_hit_count=int(len(weights)), positive_weight_hit_count=int(np.count_nonzero(positive)),
        in_view_hit_count=int(np.count_nonzero(in_view)),
        in_view_positive_weight_hit_count=int(np.count_nonzero(selected)),
        occupied_bin_count=int(len(bins)), display_bin_count=int(n*n),
        hit_weight_sum=intercepted, hit_weight_square_sum=weight_square_sum,
        in_view_weight_sum=view_weight_sum, in_view_weight_square_sum=view_weight_square_sum,
        effective_hit_count=intercepted**2/weight_square_sum if weight_square_sum else 0.,
        in_view_effective_hit_count=view_weight_sum**2/view_weight_square_sum if view_weight_square_sum else 0.,
        positive_hits_per_occupied_bin=summary(bin_counts),
        effective_hits_per_occupied_bin=summary(effective),
        weight_square_sum_per_occupied_bin=summary(bin_squares),
        interpretation="Retained numerical hit records before detector blur. Weighted effective hits = (sum w)^2 / sum(w^2). Records may be deterministic or share source ancestors; these are sampling diagnostics, not independent electron counts, Poisson uncertainty, or a convergence proof.")
    metrics.update(intercepted_probability=intercepted,
                   displayed_probability=float(ideal.sum()),
                   first_hit_bounds_mm=bounds,
                   response_probability=None if response is None else float(response.sum()),
                   response_computed=bool(include_response),
                   response_grid_pixel_size_mm=response_pitch,
                   source_current_pa=source_current_pa,
                   intercepted_current_pa=float(row.current_pa),
                   point_spread_model=point_spread.model,
                   point_spread_status=point_spread.status,
                   point_spread_source=point_spread.source,
                   point_spread_sigma_x_mm=point_spread.sigma_x_mm,
                   point_spread_sigma_y_mm=point_spread.sigma_y_mm,
                   point_spread_evaluation=psf_method,
                   point_spread_numerics=response_diagnostics,
                   coherence="classical particle reception; coherent diffraction is not inferred")
    detail = ("No electrons reached this receiver in the retained calculation."
              if intercepted == 0 else "Actual first-interception particle distribution from the retained calculation.")
    return ReceiverImage(str(plane.key), str(plane.name), "AVAILABLE", detail, ideal, response,
                         x_mm, y_mm, exposure_s, expected, float(row.current_pa), metrics)
