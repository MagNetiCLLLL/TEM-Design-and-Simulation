"""Bounded, source-aware longitudinal sampling; no field evaluation or solves.

The returned nonuniform tensor grid retains explicit axes and finite activation
boundaries for strength hatching. Physical masks and provider validity remain
the caller's responsibility. All coordinates are metres.
"""
from __future__ import annotations

import numpy as np

from temsim.field_density_texture import SectionGridAxes, from_longitudinal, to_longitudinal


def source_support_planes(sources):
    """Capture actual axial activation boundaries, without evaluating fields.

    Magnetic diagnostics first apply a global Z bound and, for posed analytic
    providers, the native local Z interval as well. Preserve both. Native map
    domains and electric/material validity still come from the sampled flags;
    their enclosing boxes are never interpreted as evidence of valid zero.
    """
    planes = []
    for source in sources:
        # Retained potentials expose their exact support/domain faces directly.
        # An explicit empty set means no finite axial boundary (Gaussian tails),
        # not permission to substitute the display box as a physical cutoff.
        explicit_planes = getattr(source, "support_planes_m", None)
        if explicit_planes is not None:
            planes.extend(explicit_planes)
            continue
        native = getattr(source, "_magnetic", ())
        if not native or getattr(source, "_unknown_b", False) or getattr(source, "_unknown_e", False):
            bounds = np.asarray(source.bounds_m, dtype=float)
            planes.extend((0., 0., 1., z) for z in bounds[:, 2])
        for item in native:
            if getattr(item, "known_zero", False):
                continue
            planes.extend((0., 0., 1., z) for z in np.asarray(item.bounds_m)[:, 2])
            provider = getattr(item, "provider", None)
            native_support = getattr(provider, "native_field_support_mm", None)
            if getattr(item, "field_map", None) is not None or not callable(native_support):
                continue
            registration = getattr(provider, "registration", None)
            normal = (np.array([0., 0., 1.]) if registration is None
                      else np.asarray(registration.rotation_array)[:, 2])
            origin = np.zeros(3) if registration is None else registration.origin_array_m
            for z in np.asarray(native_support())*.001:
                planes.append((*normal, float(z + np.dot(normal, origin))))
    return np.unique(np.asarray(planes, dtype=float).reshape(-1, 4), axis=0)


def with_source_support(axes, sources):
    """Attach source boundaries to complete section axes, including XY views."""
    return None if axes is None else SectionGridAxes(axes, source_support_planes(sources))


def _section_interval(bounds, angle_deg):
    """Intersect a global enclosing box with V=0, returning U/Z bounds."""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        return None
    direction = np.array([np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))])
    low, high = -np.inf, np.inf
    for axis, component in enumerate(direction):
        if abs(component) < 1e-14:
            if not bounds[0, axis] <= 0. <= bounds[1, axis]:
                return None
        else:
            ends = bounds[:, axis] / component
            low, high = max(low, min(ends)), min(high, max(ends))
    if high <= low or bounds[1, 2] <= bounds[0, 2]:
        return None
    return np.array([low, bounds[0, 2]]), np.array([high, bounds[1, 2]])


def _axis(low, high, preferred, secondary, regular, limit):
    """Keep bounds and ordered anchors; fill remaining slots across the span."""
    chosen = [float(low), float(high)]
    tolerance = 64 * np.finfo(float).eps * max(abs(low), abs(high), high-low, 1e-300)
    for group in (preferred, secondary):
        candidates = np.unique([float(value) for value in group
                                if np.isfinite(value) and low <= value <= high])
        # If an unusually large assembly exceeds the budget, spread retained
        # anchors over the column instead of dropping its downstream end.
        available = limit-len(chosen)
        if len(candidates) > available > 0:
            candidates = candidates[np.unique(np.linspace(0, len(candidates)-1, available, dtype=int))]
        for value in candidates:
            if len(chosen) >= limit:
                break
            if min(abs(value-old) for old in chosen) > tolerance:
                chosen.append(float(value))
    candidates = np.unique(np.r_[regular, np.linspace(low, high, limit)])
    while len(chosen) < limit and len(candidates):
        distance = np.min(np.abs(candidates[:, None]-np.asarray(chosen)), axis=1)
        index = int(np.argmax(distance))
        if distance[index] <= tolerance:
            break
        chosen.append(float(candidates[index]))
        candidates = np.delete(candidates, index)
    return np.sort(chosen)


def longitudinal_field_grid(points_m, sources, *, angle_deg=0., component_centres=(), max_points=1600,
                            interpolation_sources=None):
    """Refine a visible U-Z window around captured source supports.

    Uses the complete source catalogue, independently of checked contributions.
    ``interpolation_sources`` optionally restricts discontinuity metadata to the
    displayed contributions; it never changes the common sampling locations.
    ``component_centres`` contains (key, global Z) for captured physical parts,
    including individual lenses in an inseparable joint magnetic circuit.
    At most 17 transverse rows leave room for thin axial features, with a hard
    1600-point cap for the full grid *before* physical-envelope masking.
    """
    points = np.asarray(points_m, dtype=float)
    if not len(points):
        return points.reshape(0, 3), None
    local = to_longitudinal(points, angle_deg)
    u_regular, z_regular = np.unique(local[:, 0]), np.unique(local[:, 2])
    if len(u_regular) < 2 or len(z_regular) < 2:
        return points, None
    budget = min(1600, int(max_points))
    if budget < 4:
        return np.empty((0, 3)), None
    low = np.array([u_regular[0], z_regular[0]])
    high = np.array([u_regular[-1], z_regular[-1]])
    u_centres, z_centres, u_near, z_near = [], [], [], []
    sampling_planes = source_support_planes(sources)
    support_planes = (sampling_planes if interpolation_sources is None
                      else source_support_planes(interpolation_sources))
    z_edges = []
    epsilon = 256*np.finfo(float).eps*max(np.max(abs(low)), np.max(abs(high)), np.max(high-low), 1e-300)
    # Distinct samples on either side retain a jump without a fictitious ramp.
    # The intervening tiny cell remains unsupported through plane metadata.
    direction = np.array([np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg)), 0.])
    for nx, ny, nz, offset in sampling_planes:
        if abs(nz) <= 1e-14:
            continue
        for u_anchor in (low[0], 0. if low[0] <= 0. <= high[0] else np.mean((low[0], high[0])), high[0]):
            z = (offset - np.dot((nx, ny, nz), direction)*u_anchor)/nz
            if low[1] <= z <= high[1]:
                z_edges.extend((z-epsilon, z+epsilon))
    component_keys = set()
    intervals = []
    for source in sources:
        # Unsupported sources also need positions: their samples must expose
        # unknown regions rather than letting a thin missing model disappear.
        component_keys.update(getattr(source, "component_keys", ()))
        native = getattr(source, "_magnetic", ())
        # A joint source may span multiple physical lenses. Keep their native
        # supports as well as the common source envelope.
        hints = getattr(source, "sampling_bounds_m", None)
        boxes = hints if hints is not None else (getattr(item, "bounds_m", None) for item in native or (source,))
        for bounds in boxes:
            if bounds is None:
                continue
            interval = _section_interval(bounds, angle_deg)
            if interval is None:
                continue
            start, end = interval
            if np.any(end < low) or np.any(start > high):
                continue
            centre = (start+end)*.5
            if low[1] <= centre[1] <= high[1]:
                z_centres.append(centre[1])
            z_near.extend((start[1]*.75+end[1]*.25, start[1]*.25+end[1]*.75))
            if low[0] <= centre[0] <= high[0]:
                u_centres.append(centre[0])
                intervals.append((end[0]-start[0], centre[0]))
    for key, z in component_centres:
        if key in component_keys and low[1] <= z <= high[1]:
            z_centres.append(float(z))
    # Capture the narrowest supported near-axis structure, including nonzero
    # multipole fields which vanish exactly on the axis. Never move a field.
    if intervals:
        width, centre = min(intervals)
        u_near.extend((centre-width*.25, centre+width*.25))
    if low[0] <= 0. <= high[0]:
        u_centres.insert(0, 0.)
    u_count = min(17, max(7, len(u_regular)), max(2, budget//2))
    # Small local views need no full-budget sampling. Allocate additional
    # columns only for source centres and their immediate neighbourhoods.
    z_count = min(budget//u_count, max(len(z_regular), len(set(z_centres+z_near+z_edges))+2))
    u = _axis(low[0], high[0], ([0.] if low[0] <= 0. <= high[0] else [])+u_near,
              u_centres, u_regular, u_count)
    z = _axis(low[1], high[1], z_centres+z_edges, z_near, z_regular, z_count)
    uu, zz = np.meshgrid(u, z, indexing="ij")
    result = from_longitudinal(np.column_stack((uu.ravel(), np.zeros(uu.size), zz.ravel())), angle_deg)
    return result, SectionGridAxes((u, z), support_planes)
