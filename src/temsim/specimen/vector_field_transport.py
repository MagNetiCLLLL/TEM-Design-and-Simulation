"""Specimen-local SI transport using the column's registered vector fields.

Positions exposed here are specimen-local nm; the Lorentz integrator operates
in local metres to avoid subtracting metre-scale column coordinates afterwards.
Static magnetic fields do no work. Ideal Optics instead uses straight local
flights and rejects unsupported overlap with a nonzero thin action. Collisions
remain the responsibility of the material transport kernel.
"""

from __future__ import annotations

from collections import OrderedDict
import math
import numpy as np

from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field
from temsim.physics.relativistic_lorentz import (
    ELECTRON, RelativisticPhaseSpace, boris_step,
    momentum_from_kinetic_energy_ev, velocity_from_momentum_m_per_s,
)
from temsim.simulation_modes import is_thin


class SpecimenFieldTransport:
    """One resolved field context per immutable calculation, never per collision.

    Create a new context after any state edit. Its small exact-position cache
    belongs only to this context; neither fields nor keys survive a calculation.
    """

    def __init__(self, state, *, field_cache_size: int = 8):
        from temsim.simulation_modes import is_unified
        if is_unified(state):
            raise ValueError("Finite-Field Optics unified particle fields are not yet coupled to finite-volume specimen collisions. "
                             "Use an explicitly labelled compatibility mode for this material calculation; "
                             "optical sections remain available without specimen interactions.")
        if (
            isinstance(field_cache_size, bool)
            or not isinstance(field_cache_size, (int, np.integer))
            or not 0 <= field_cache_size <= 8
        ):
            raise ValueError("Specimen field cache size must be an integer from 0 to 8")
        self._field_cache_size = int(field_cache_size)
        self._field_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._field_cache_hits = 0
        self._field_cache_misses = 0
        self.origin_m = np.array((0.0, 0.0, float(state.sample.z_mm) * 1e-3))
        self.thin_mode = is_thin(state)
        self.thin_actions = ()
        if getattr(self, "thin_mode", False):
            from temsim.physics.instrument_magnetic import active_column_events
            from temsim.physics.thin_optics import capture_all_thin_actions

            self.thin_actions = tuple(
                action for action in capture_all_thin_actions(state, active_column_events(state))
                if action.power_m1 != 0.0 or action.rotation_rad != 0.0
                or any(value != 0.0 for _i, _j, value in action.polynomial)
            )
            self.magnetic_field = None
            self.spatial_step_nm = 1e6
            return
        self.magnetic_field = capture_instrument_magnetic_field(state)
        # The specimen uses the same immutable magnetic graph as gun and
        # virtual electrons; only the coordinates here are specimen-local.
        scales = []
        for source in self.magnetic_field._sources:
            if source.known_zero:
                continue
            field_map = source.field_map
            if field_map is not None:
                scales.extend(float(np.min(np.diff(axis))) * 1e9 / 4
                              for axis in field_map.axes_m)
            else:
                scales.append(max((source.bounds_m[1, 2]-source.bounds_m[0, 2])*1e9/128, 1e-6))
        self.spatial_step_nm = min(scales, default=1e6)

    def _action_planes(self, action):
        """Physical posed plane and the global-Z plane that owns its action."""
        rotation = action.registration.rotation_array
        centre = (action.registration.origin_array_m
                  + rotation @ np.array((0.0, 0.0, action.local_z_m)))
        # Work near the specimen rather than adding sub-nm coordinates to a
        # metre-scale column position. The tolerance covers only roundoff in
        # the captured global origins, never a physical support thickness.
        relative_centre = centre - self.origin_m
        tolerance = 8 * np.finfo(float).eps * max(
            np.max(np.abs(centre)), np.max(np.abs(self.origin_m)), np.finfo(float).tiny)
        return ((relative_centre, rotation[:, 2], tolerance),
                (np.array((0.0, 0.0, action.z_mm * 1e-3)) - self.origin_m,
                 np.array((0.0, 0.0, 1.0)), tolerance))

    @staticmethod
    def _unsupported(action, context):
        raise ValueError(
            f"Ideal Optics cannot separate thin action {action.key!r} from {context}. "
            "Finite material flights and reference matching across a nonzero thin plane "
            "require inverse and scattered-ray action replay, which are not implemented. "
            "Select Finite-Field Optics for this geometry."
        )

    def validate_material_geometry(self, geometry):
        """Require actual material envelopes to be disjoint from thin actions.

        Axis-aligned boxes enclose the specimen and physical support. A tilted
        plane intersecting a box is conservatively unsupported even when a
        support opening might avoid it; no disjointness is inferred from Z alone.
        """
        if not getattr(self, "thin_mode", False) or not self.thin_actions:
            return
        boxes = []
        if geometry.sample_material is not None and geometry.sample_thickness_nm > 0.0:
            centre = np.array((*geometry.sample_centre_xy_nm, 0.0)) * 1e-9
            half = np.array((*geometry.sample_size_xy_nm, geometry.sample_thickness_nm)) * 0.5e-9
            boxes.append(("the finite specimen envelope", centre, half))
        if geometry.support_material is not None and geometry.support_bottom_nm > geometry.support_top_nm:
            radius = geometry.support_grid.outer_diameter_mm * 0.5e-3
            centre = np.array((*np.asarray(geometry.support_offset_xy_um) * 1e-6,
                               0.5 * (geometry.support_bottom_nm + geometry.support_top_nm) * 1e-9))
            half = np.array((radius, radius,
                             0.5 * (geometry.support_bottom_nm - geometry.support_top_nm) * 1e-9))
            boxes.append(("the physical support envelope (including unresolved openings)", centre, half))
        for action in self.thin_actions:
            for plane_centre, normal, tolerance in self._action_planes(action):
                for name, centre, half in boxes:
                    distance = float((centre - plane_centre) @ normal)
                    extent = float(half @ np.abs(normal))
                    if abs(distance) <= extent + tolerance:
                        self._unsupported(action, name)

    def _validate_thin_flight(self, start_m, end_m):
        for action in self.thin_actions:
            for centre, normal, tolerance in self._action_planes(action):
                distances = np.array(((start_m - centre) @ normal, (end_m - centre) @ normal))
                if np.min(distances) <= tolerance and np.max(distances) >= -tolerance:
                    self._unsupported(action, "a local specimen/reference-matching flight")

    def field_at_global_positions_t(self, local_positions_m):
        """Reuse identical scalar queries without rounding or interpolation.

        Boundary searches and Boris flights often query the same position in
        succession. Cache only three-vectors, keeping storage bounded even if
        callers request large batches. Zero capacity uses the original path.
        """
        if not getattr(self, "_field_cache_size", 0):
            return self._field_at_global_positions_t_uncached(local_positions_m)
        position = np.asarray(local_positions_m, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return self._field_at_global_positions_t_uncached(local_positions_m)
        # Exact native float64 bytes preserve nextafter differences and signed
        # zero. Shape is fixed at (3,); a batch never shares a scalar entry.
        key = position.tobytes()
        cached = self._field_cache.get(key)
        if cached is not None:
            self._field_cache.move_to_end(key)
            self._field_cache_hits += 1
            return cached.copy()
        self._field_cache_misses += 1
        value = self._field_at_global_positions_t_uncached(local_positions_m)
        if np.all(np.isfinite(value)):
            retained = value.copy()
            retained.setflags(write=False)
            self._field_cache[key] = retained
            if len(self._field_cache) > self._field_cache_size:
                self._field_cache.popitem(last=False)
        return value

    def field_cache_info(self) -> dict[str, int]:
        """Return cheap per-calculation scalar-query cache counters."""
        return {
            "capacity": self._field_cache_size,
            "entries": len(self._field_cache),
            "hits": self._field_cache_hits,
            "misses": self._field_cache_misses,
        }

    def _field_at_global_positions_t_uncached(self, local_positions_m):
        """Original vector-field evaluation, also used as a test reference."""
        if getattr(self, "thin_mode", False):
            positions = np.asarray(local_positions_m, dtype=float)
            if positions.ndim < 1 or positions.shape[-1] != 3 or not np.isfinite(positions).all():
                raise ValueError("Specimen positions must be finite three-vectors")
            return np.zeros_like(positions)
        positions = np.asarray(local_positions_m, dtype=float) + self.origin_m
        return self.magnetic_field.field_at_global_positions_t(positions)

    def advance(self, position_nm, direction, path_length_nm, *, energy_ev,
                return_elapsed_time=False):
        """Execute one local flight, optionally returning its signed time.

        The finite-field clock sums Boris steps; the thin clock uses the exact
        straight length and relativistic speed. Negative lengths are reference
        matching operations, not additional physical travel.
        """
        position = np.asarray(position_nm, dtype=float) * 1e-9
        momentum = momentum_from_kinetic_energy_ev(energy_ev, direction)
        speed = float(np.linalg.norm(velocity_from_momentum_m_per_s(momentum)))
        length = float(path_length_nm)
        if not math.isfinite(length):
            raise ValueError("Magnetic flight length must be finite")
        phase = RelativisticPhaseSpace(position, momentum)
        if getattr(self, "thin_mode", False):
            unit = momentum / np.linalg.norm(momentum)
            endpoint_nm = np.asarray(position_nm, dtype=float) + length * unit
            if not np.isfinite(endpoint_nm).all():
                raise ValueError("Thin specimen flight exceeded finite coordinates")
            if length != 0.0:
                self._validate_thin_flight(position, endpoint_nm * 1e-9)
            result = (endpoint_nm, unit)
            elapsed = length * 1e-9 / speed
            return (*result, elapsed) if return_elapsed_time else result
        remaining = abs(length)
        sign = math.copysign(1.0, length)
        for _ in range(100000):
            if remaining <= 1e-12:
                result = (phase.position_m * 1e9,
                          phase.momentum_kg_m_per_s / np.linalg.norm(phase.momentum_kg_m_per_s))
                return (*result, float(phase.time_s)) if return_elapsed_time else result
            field = float(np.linalg.norm(self.field_at_global_positions_t(phase.position_m)))
            curvature = abs(ELECTRON.charge_c) * field / np.linalg.norm(momentum) * 1e-9
            step = min(remaining, self.spatial_step_nm, 1e-3 / max(curvature, 1e-30))
            phase = boris_step(phase, sign * step * 1e-9 / speed, self)
            remaining -= step
        raise ValueError("Specimen field transport exceeded its step budget")

    def to_plane(self, position_nm, direction, target_z_nm, *, energy_ev,
                 return_elapsed_time=False):
        """Match Z by shooting; report only the accepted flight's signed time."""
        position = np.asarray(position_nm, dtype=float)
        unit = np.asarray(direction, dtype=float)
        unit = unit / np.linalg.norm(unit)
        if abs(unit[2]) < 1e-10:
            raise ValueError("Cannot match a reference plane with a grazing ray")
        length = (float(target_z_nm) - position[2]) / unit[2]
        for _ in range(16):
            endpoint, final, elapsed = self.advance(
                position, unit, length, energy_ev=energy_ev, return_elapsed_time=True)
            residual = endpoint[2] - target_z_nm
            if abs(residual) < 1e-7:
                return (endpoint, final, elapsed) if return_elapsed_time else (endpoint, final)
            if abs(final[2]) < 1e-10:
                break
            length -= residual / final[2]
        raise ValueError("Magnetic trajectory did not converge to the reference plane")

    def plane_polyline(self, position_nm, direction, target_z_nm, *, energy_ev, point_count=9):
        planes = np.linspace(float(position_nm[2]), float(target_z_nm), max(point_count, 2))
        values = [self.to_plane(position_nm, direction, z, energy_ev=energy_ev) for z in planes]
        return np.asarray([value[0] for value in values]), values[-1][1]

    def boundary_distance(self, geometry, position, direction, region, *, maximum_distance_nm, energy_ev):
        """Intersect short magnetic arcs with the authoritative material geometry.

        Each chord's sagitta is limited to one quarter of the geometry epsilon.
        A tangent intercept only bounds the trial step: the magnetic arc and
        its chord are still evaluated before accepting a material crossing.
        No tangent intercept does NOT imply that a curved ray misses matter.
        """
        travelled = 0.0
        point, unit = np.asarray(position), np.asarray(direction)
        bounds = geometry.material_z_bounds_nm
        if bounds is None:
            return None
        if getattr(self, "thin_mode", False):
            # A boundary query is not a flight. Intersect the actual straight
            # ray directly, so a trial chord beyond the next material face
            # cannot spuriously claim an unexecuted downstream thin action.
            return geometry.next_region_boundary_distance_nm(
                point, unit, region, maximum_distance_nm=maximum_distance_nm)
        momentum = float(np.linalg.norm(momentum_from_kinetic_energy_ev(energy_ev, unit)))
        for _ in range(100000):
            if region is None and (
                (point[2] > bounds[1] and unit[2] > 0)
                or (point[2] < bounds[0] and unit[2] < 0)
            ):
                # Beyond the finite specimen envelope, the downstream column
                # owns the trajectory, including any later magnetic turning.
                return None
            remaining = maximum_distance_nm - travelled
            if remaining <= 1e-12:
                return None
            field = float(np.linalg.norm(np.cross(unit, self.field_at_global_positions_t(point * 1e-9))))
            curvature = abs(ELECTRON.charge_c) * field / momentum * 1e-9
            step = min(remaining, self.spatial_step_nm,
                       math.sqrt(2 * geometry.epsilon_nm / max(curvature, 1e-30)))
            tangent_hit = geometry.next_region_boundary_distance_nm(
                point, unit, region, maximum_distance_nm=step,
            )
            if tangent_hit is not None:
                # Do not integrate micrometres just to find a boundary a few
                # picometres away. A small overshoot avoids asymptotic steps
                # on approach; the original spatial/sagitta limits still win.
                step = min(step, tangent_hit + max(
                    8.0 * geometry.epsilon_nm, 0.01 * tangent_hit,
                ))
            endpoint, final = self.advance(point, unit, step, energy_ev=energy_ev)
            chord = endpoint - point
            norm = float(np.linalg.norm(chord))
            if norm <= 0:
                raise ValueError("Magnetic boundary search made no progress")
            hit = geometry.next_region_boundary_distance_nm(
                point, chord / norm, region, maximum_distance_nm=norm,
            )
            if hit is not None:
                return travelled + step * hit / norm
            travelled += step
            point, unit = endpoint, final
        raise ValueError("Magnetic boundary search exceeded its step budget")
