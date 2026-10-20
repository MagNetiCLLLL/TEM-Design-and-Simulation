"""Software projection of sampled physical E, B and electron-force vectors.

This widget only projects supplied SI arrays. It can plan bounded viewport
sampling, but never evaluates a field itself.
Independent coordinate scales are applied to positions and vector directions
consistently, and disclosed in the plot. Out-of-plane signs remain visible in
section views; the hover readout always reports all three physical components.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QToolTip, QWidget


class FieldVectorCanvas(QWidget):
    """Bounded vector display; cyan B, amber E, and violet electron force."""

    projection_angle_requested = Signal(float)
    component_selected = Signal(str)
    component_context_requested = Signal(str, object)
    view_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 220)
        self.setMouseTracking(True)
        self._sample = None
        self._bounds_m = np.array([[-.001, -.001, 0.], [.001, .001, .01]])
        self._view_bounds_m = self._bounds_m.copy()
        self._physical_meshes = ()
        self._model_visible = False
        self._model_opacity = .3
        self._fit_models = False
        self._section_cache = None
        self._section = "UZ"
        self._sample_plane_m = 0.
        self._mode = "overlay"
        self._render_style = "arrows"
        self._observation_regions = None
        self._observation_component = None
        self._density_grid_cache = None
        self._density_image_cache = None
        self._density_notice = ""
        self._zoom = 1.
        self._pan = QPointF()
        self._projection_angle_deg = 0.
        self._selected_component_keys = frozenset()
        self._drag = None
        self._press_position = None
        self._drag_button = Qt.MouseButton.NoButton
        self._dragged = False
        self._screen_points = np.empty((0, 2))
        self._visible_sample_indices = np.empty(0, dtype=int)
        self._source_regions = ()
        self.references = {"B": None, "E": None, "F": None}
        self._message = "Calculate the column, then select sources."
        self.setToolTip("Left drag a longitudinal section or 3D volume: rotate the shared observation angle. "
                        "Right or middle drag: pan. Wheel: display zoom. "
                        "Click a physical model to select; right-click it for navigation. "
                        "Double click: reset view. Hover a sample for physical XYZ components. "
                        "Navigation immediately reprojects cached samples. The page may refine a visible "
                        "section after navigation stops. The longitudinal U-Z section rotates with "
                        "Ray Diagram and Physical Layout's rotating section, with Z always to the right.")

    def sizeHint(self):
        return QSize(680, 420)

    @property
    def sample(self):
        return self._sample

    @property
    def projection_angle_deg(self):
        return self._projection_angle_deg

    def set_projection_angle(self, angle):
        """Project cached XYZ using the shared U=X cos(angle)+Y sin(angle)."""
        angle = float(angle)
        if not math.isfinite(angle):
            raise ValueError("Projection angle must be finite")
        angle %= 360.
        if angle != self._projection_angle_deg:
            self._projection_angle_deg = angle
            self._view_updated()

    def set_selected_component(self, key):
        self.set_selected_components((key,) if key else ())

    def set_selected_components(self, keys):
        """Highlight actual material keys; the page resolves physical children."""
        self._selected_component_keys = frozenset(str(key) for key in keys if key)
        self.update()

    @property
    def legend_text(self):
        fields = self._fields()
        if self._sample is None:
            return "B [T] | E [V/m] | electron force [N]"
        labels = []
        for label, unit, values, valid, _colour in fields:
            good = valid & np.isfinite(values).all(axis=1)
            peak = float(np.max(np.linalg.norm(values[good], axis=1))) if np.any(good) else None
            reference = self.references[label]
            kind = "hatch" if self._render_style == "density" else "arrow"
            reference_text = f"; {kind} ref {reference:.5g}" if reference is not None else "; reference pending"
            labels.append((f"{label} [{unit}]: max {peak:.5g}" if peak is not None
                           else f"{label} [{unit}]: unavailable") + reference_text)
        text = " | ".join(labels)
        if self._render_style == "density":
            text += " | Full XYZ magnitude; denser = stronger; fixed ref; saturation at 2x ref"
        return text

    def clear(self, message="No current field samples."):
        self._sample = None
        self._density_grid_cache = None
        self._density_image_cache = None
        self._physical_meshes = ()
        self._section_cache = None
        self._screen_points = np.empty((0, 2))
        self._visible_sample_indices = np.empty(0, dtype=int)
        self._message = str(message)
        self.update()

    def set_sample(self, sample, bounds_m, section):
        bounds = np.asarray(bounds_m, dtype=float)
        if section == "UZ":
            self._sample_plane_m = 0.
        elif section != "3D":
            normal = {"XY": 2, "XZ": 1, "YZ": 0}[section]
            if len(sample.points_m):
                midpoint = float(np.mean(bounds[:, normal]))
                coordinates = sample.points_m[:, normal]
                tolerance = 64*np.finfo(float).eps*max(float(np.max(np.abs(bounds[:, normal]))), 1e-15)
                # Bounds remain the public section declaration for legacy
                # point clouds. Use exact sample coordinates only when they
                # describe that same plane within floating-point roundoff.
                self._sample_plane_m = (float(coordinates[0])
                                        if np.all(coordinates == coordinates[0])
                                        and abs(float(coordinates[0])-midpoint) <= tolerance
                                        else midpoint)
            elif section != self._section or not np.array_equal(bounds, self._bounds_m):
                self._sample_plane_m = float(np.mean(bounds[:, normal]))
        self._sample = sample
        self._density_grid_cache = None
        self._density_image_cache = None
        self._bounds_m = bounds.copy()
        # Meshes belong to the same accepted snapshot/selection as the sample;
        # publication must explicitly supply them again after a replacement.
        self._physical_meshes = ()
        self._section_cache = None
        self._view_bounds_m = self._bounds_m.copy()
        self._section = str(section)
        self.use_current_peaks(only_unset=True)
        self._screen_points = np.empty((0, 2))
        self._visible_sample_indices = np.empty(0, dtype=int)
        self.update()

    def _view_updated(self):
        """Invalidate picking immediately, even before Qt schedules a repaint."""
        self._screen_points = np.empty((0, 2))
        self._visible_sample_indices = np.empty(0, dtype=int)
        self._density_image_cache = None
        self.update()
        self.view_changed.emit()

    def viewport_sample_grid(self, count=15, max_points=1600):
        """Return a bounded global-XYZ grid in the visible physical section.

        The affine projection is inverted on the existing section, then clipped
        to the original physical sampling bounds. Camera movement never changes
        those bounds or the section's normal coordinate. Snap only the sampling
        limits to a small fraction of a screen pixel so an LRU cache can reuse
        essentially identical views; the exact returned coordinates form its key.
        A 3D volume or an edge-on section has no unambiguous 2D inverse.
        """
        if self._sample is None or self._section == "3D":
            return None
        if self._section == "UZ":
            return self._longitudinal_viewport_grid(count, max_points)
        budget = min(1600, int(max_points))
        if budget < 4:
            return None
        axes = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}
        first, second, normal = axes[self._section]
        active = [first, second]
        plot = self._plot_rect()
        centre, span, basis, size, _origin = self._projection(plot)
        plane_basis = basis[:2, active]
        if abs(float(np.linalg.det(plane_basis))) < 1e-8:
            return None
        matrix = size[:, None] * plane_basis / span[active]
        plane_centre = self._section_centre()
        anchor = self._project_positions(plane_centre[None, :], plot)[0, :2]
        corners = np.array([[plot.left(), plot.top()], [plot.right(), plot.top()],
                            [plot.right(), plot.bottom()], [plot.left(), plot.bottom()]])
        coordinates = np.linalg.solve(matrix, (corners-anchor).T).T + plane_centre[active]
        domain_low, domain_high = self._bounds_m[:, active]
        low = np.maximum(coordinates.min(axis=0), domain_low)
        high = np.minimum(coordinates.max(axis=0), domain_high)
        domain_span = domain_high-domain_low
        if np.any(high <= low) or np.any(domain_span <= 0.):
            # An invertible plane with no visible physical domain is a valid
            # empty view, distinct from an undefined edge-on inverse. Publish
            # it so the page cannot retain stale local samples after a pan.
            signature = (self._section, "empty", float(plane_centre[normal]),
                         tuple(self._bounds_m.ravel()))
            return np.empty((0, 3), dtype=float), signature
        # Power-of-two quantization makes nearby pan/zoom requests share both
        # a stable cache key and identical coordinates. Clamp after snapping:
        # rounding must never extend a sample outside the physical envelope.
        fraction = (high-low)/domain_span
        initial_pixels = np.linalg.norm(matrix, axis=0) * (high-low)
        subdivisions = np.maximum(4096., initial_pixels*4.)
        quantization = domain_span * np.exp2(np.floor(np.log2(fraction/subdivisions)))
        low = np.maximum(domain_low, domain_low + np.floor((low-domain_low)/quantization)*quantization)
        high = np.minimum(domain_high, domain_low + np.ceil((high-domain_low)/quantization)*quantization)
        pixel_extent = np.linalg.norm(matrix, axis=0) * (high-low)
        spacing = 32. * 15. / max(4, int(count))
        counts = np.clip(np.floor(pixel_extent/spacing).astype(int)+1, 2, 40)
        while int(np.prod(counts)) > budget:
            axis = int(np.argmax(counts))
            counts[axis] -= 1
        vectors = [np.linspace(low[index], high[index], int(counts[index])) for index in range(2)]
        grid = np.meshgrid(*vectors, indexing="ij")
        points = np.tile(plane_centre, (int(np.prod(counts)), 1))
        points[:, first], points[:, second] = grid[0].ravel(), grid[1].ravel()
        points[:, normal] = plane_centre[normal]
        signature = (self._section, float(plane_centre[normal]), tuple(low), tuple(high), tuple(map(int, counts)))
        return points, signature

    def _longitudinal_viewport_grid(self, count, max_points):
        """Bounded U-Z grid on the actual rotated V=0 plane, not a projected XZ slice."""
        budget = min(1600, int(max_points))
        if budget < 4:
            return None
        plot = self._plot_rect()
        inverse = self._section_inverse(plot)
        corners = np.array([[plot.left(), plot.top()], [plot.right(), plot.top()],
                            [plot.right(), plot.bottom()], [plot.left(), plot.bottom()]])
        from temsim.field_density_texture import to_longitudinal, from_longitudinal
        visible = to_longitudinal(self._section_positions(corners, inverse), self._projection_angle_deg)
        centre = np.mean(self._bounds_m, axis=0)
        centre_u = float(centre @ self._basis()[1])
        half_u = max(np.ptp(self._bounds_m[:, 0]), np.ptp(self._bounds_m[:, 1]))*.5
        domain_low = np.array([centre_u-half_u, self._bounds_m[0, 2]])
        domain_high = np.array([centre_u+half_u, self._bounds_m[1, 2]])
        low = np.maximum(visible[:, [0, 2]].min(axis=0), domain_low)
        high = np.minimum(visible[:, [0, 2]].max(axis=0), domain_high)
        domain_span = domain_high-domain_low
        if np.any(high <= low) or np.any(domain_span <= 0):
            return np.empty((0, 3)), ("UZ", self._projection_angle_deg, "empty",
                                     tuple(self._bounds_m.ravel()))
        # Columns of the inverse plane basis correspond to U and Z.
        matrix = np.linalg.inv(inverse[1])
        pixel_scale = np.linalg.norm(matrix, axis=0)
        subdivisions = np.maximum(4096., pixel_scale*(high-low)*4.)
        quantization = domain_span*np.exp2(np.floor(np.log2((high-low)/domain_span/subdivisions)))
        low = np.maximum(domain_low, domain_low+np.floor((low-domain_low)/quantization)*quantization)
        high = np.minimum(domain_high, domain_low+np.ceil((high-domain_low)/quantization)*quantization)
        spacing = 32.*15./max(4, int(count))
        counts = np.clip(np.floor(pixel_scale*(high-low)/spacing).astype(int)+1, 2, 40)
        while int(np.prod(counts)) > budget:
            counts[int(np.argmax(counts))] -= 1
        uu, zz = np.meshgrid(*(np.linspace(low[i], high[i], int(counts[i])) for i in range(2)),
                             indexing="ij")
        local = np.column_stack((uu.ravel(), np.zeros(uu.size), zz.ravel()))
        points = from_longitudinal(local, self._projection_angle_deg)
        signature = ("UZ", self._projection_angle_deg, tuple(low), tuple(high), tuple(map(int, counts)))
        return points, signature

    def use_current_peaks(self, *, only_unset=False):
        if self._sample is None:
            return
        for key, values, valid in (("B", self._sample.b_t, self._sample.magnetic_valid),
                                   ("E", self._sample.e_v_per_m, self._sample.electric_valid),
                                   ("F", self._sample.force_n, self._sample.force_valid)):
            if only_unset and self.references[key] is not None:
                continue
            good = valid & np.isfinite(values).all(axis=1)
            peak = float(np.linalg.norm(values[good], axis=1).max()) if np.any(good) else 0.
            if peak > 0:
                self.references[key] = peak
        self.update()

    def set_reference(self, key, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Field strength reference must be finite and positive")
        self.references[key] = value
        self.update()

    def _display_vectors(self, values, valid, basis, span, size, max_length, reference):
        """Affine directions and fixed physical strength, capped at 4× ref."""
        finite = np.nan_to_num(values)
        vectors = (finite / span) @ basis.T
        directions = vectors[:, :2] * size
        lengths = np.linalg.norm(directions, axis=1)
        unit = np.divide(directions, lengths[:, None], out=np.zeros_like(directions), where=lengths[:, None] > 0)
        norms = np.linalg.norm(finite, axis=1)
        strength = np.minimum(norms / reference, 4.) if reference else np.zeros_like(norms)
        transformed_norms = np.linalg.norm(vectors, axis=1)
        fraction = np.divide(np.linalg.norm(vectors[:, :2], axis=1), transformed_norms,
                             out=np.zeros_like(norms), where=transformed_norms > 0)
        result = unit * (strength * fraction * max_length)[:, None]
        result[~valid] = 0
        return result

    def set_mode(self, mode):
        self._mode = str(mode)
        self._visible_sample_indices = np.empty(0, dtype=int)
        self.update()

    def set_render_style(self, style):
        """Change only presentation; no field-provider or trajectory work."""
        if style not in ("arrows", "density"):
            raise ValueError("Field display style must be arrows or density")
        self._render_style = style
        self._visible_sample_indices = np.empty(0, dtype=int)
        self.update()

    def set_observation_regions(self, regions, component_key=None):
        """Captured geometry supplies a CPU-only mask, never field values."""
        if regions is not self._observation_regions or component_key != self._observation_component:
            self._density_image_cache = None
        self._observation_regions = regions
        self._observation_component = component_key
        self.update()

    def _section_centre(self):
        centre = np.mean(self._bounds_m, axis=0)
        if self._section == "UZ":
            u = self._basis()[1]
            centre = u*float(centre @ u) + np.array([0., 0., centre[2]])
        elif self._section != "3D":
            normal = {"XY": 2, "XZ": 1, "YZ": 0}[self._section]
            # Preserve the exact sampled plane, also through empty local views.
            # Averaging centre +/- radius can lose a small transverse offset.
            centre[normal] = self._sample_plane_m
        return centre

    def _section_inverse(self, plot):
        """Map screen XY back to the existing physical plane, including rotation."""
        if self._section == "3D":
            return None
        if self._section == "UZ":
            _centre, span, basis, size, _origin = self._projection(plot)
            plane_axes = np.column_stack((basis[1], np.array([0., 0., 1.])))
            matrix = size[:, None]*((basis[:2]/span) @ plane_axes)
            centre = self._section_centre()
            anchor = self._project_positions(centre[None], plot)[0, :2]
            return plane_axes, np.linalg.inv(matrix), centre, anchor
        active = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[self._section]
        _centre, span, basis, size, _origin = self._projection(plot)
        plane_basis = basis[:2, active]
        if abs(float(np.linalg.det(plane_basis))) < 1e-8:
            return None
        matrix = size[:, None] * plane_basis / span[list(active)]
        plane_centre = self._section_centre()
        anchor = self._project_positions(plane_centre[None, :], plot)[0, :2]
        return active, np.linalg.inv(matrix), plane_centre, anchor

    @staticmethod
    def _section_positions(screen, inverse):
        active, matrix, centre, anchor = inverse
        points = np.tile(centre, (len(screen), 1))
        if np.ndim(active) == 2:
            points += ((screen-anchor) @ matrix.T) @ active.T
        else:
            points[:, active] += (screen-anchor) @ matrix.T
        return points

    def _magnitude_grid(self):
        if self._sample is None or len(self._fields()) != 1 or self._section == "3D":
            return None
        angle = getattr(self._sample, "section_angle_deg", self._projection_angle_deg)
        key = (id(self._sample), self._mode, self._section, angle)
        if self._density_grid_cache is None or self._density_grid_cache[0] != key:
            from temsim.field_density_texture import SectionMagnitudeGrid
            _label, _unit, values, valid, _colour = self._fields()[0]
            grid = SectionMagnitudeGrid(
                self._sample.points_m, values, valid, self._section,
                grid_axes_m=getattr(self._sample, "grid_axes_m", None), angle_deg=angle)
            self._density_grid_cache = (key, grid)
        return self._density_grid_cache[1]

    def _observation_mask(self, points):
        regions = self._observation_regions
        if regions is None:
            return np.ones(len(points), dtype=bool)
        from temsim.field_display_regions import FieldDisplayRegions
        if not isinstance(regions, FieldDisplayRegions):
            return regions.mask(points, self._observation_component)
        # Avoid transforming each texel through every component in the column.
        # Cull whole regions by this batch's AABB, then test only unmatched
        # points inside that region's enclosing box against its true shape.
        result = np.zeros(len(points), dtype=bool)
        if not len(points):
            return result
        low, high = points.min(axis=0), points.max(axis=0)
        for region in regions.regions_for_component(self._observation_component):
            bounds = region.bounds_m
            padding = 64*np.finfo(float).eps*np.maximum(np.max(np.abs(bounds), axis=0),
                                                       np.ptp(bounds, axis=0))
            bound_low, bound_high = bounds[0]-padding, bounds[1]+padding
            if np.any(bound_high < low) or np.any(bound_low > high):
                continue
            candidates = np.flatnonzero(~result & np.all(points >= bound_low, axis=1)
                                        & np.all(points <= bound_high, axis=1))
            if len(candidates):
                result[candidates] = region.contains(points[candidates])
            if np.all(result):
                break
        return result

    def _density_image(self, plot):
        """One bounded raster cache; no Qt item per hatch and no GPU context.

        Interpolate cached vectors in batches, then take their full XYZ norm.
        A second geometric mask clips every texel to the actual observation
        envelope, including curved or tilted boundaries between sample points.
        """
        self._density_notice = ""
        if self._section == "3D":
            self._density_notice = "Strength hatching requires a 2D section."
            return None
        if len(self._fields()) != 1:
            self._density_notice = "Choose B, E or force: each needs its own strength scale."
            return None
        inverse = self._section_inverse(plot)
        if inverse is None:
            self._density_notice = "Section viewed edge-on; rotate the Ray Diagram view to see its area."
            return None
        label, _unit, _values, _valid, colour = self._fields()[0]
        reference = self.references[label]
        key = (id(self._sample), self._mode, self._section, reference,
               tuple(self._view_bounds_m.ravel()), self._projection_angle_deg,
               self._zoom, self._pan.x(), self._pan.y(), plot.width(), plot.height(),
               id(self._observation_regions), self._observation_component)
        if self._density_image_cache is not None and self._density_image_cache[0] == key:
            self._density_notice = self._density_image_cache[2]
            return self._density_image_cache[1]
        from temsim.field_density_texture import hatch_mask
        # At most 2 MiB retained RGBA, regardless of monitor size or DPI.
        # Only 8192 texels' interpolation/mask temporaries are live at a time.
        scale = min(1., 1024./max(plot.width(), plot.height()),
                    math.sqrt(524288./(plot.width()*plot.height())))
        width = max(1, int(plot.width()*scale))
        height = max(1, int(plot.height()*scale))
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        step_x, step_y = plot.width()/width, plot.height()/height
        columns = plot.left()+(np.arange(width)+.5)*step_x
        grid = self._magnitude_grid()
        any_support = False
        rows_per_batch = max(1, 8192//width)
        for start in range(0, height, rows_per_batch):
            stop = min(height, start+rows_per_batch)
            rows = plot.top()+(np.arange(start, stop)+.5)*step_y
            xx, yy = np.meshgrid(columns, rows)
            screen = np.column_stack((xx.ravel(), yy.ravel()))
            points = self._section_positions(screen, inverse)
            magnitude, supported, valid = grid.evaluate(points)
            if self._observation_regions is not None:
                candidates = np.flatnonzero(supported)
                if len(candidates):
                    supported[candidates] &= self._observation_mask(points[candidates])
            any_support |= bool(np.any(supported))
            available = supported & valid
            target = rgba[start:stop].reshape(-1, 4)
            # A quiet neutral base identifies sampled zero-field areas; unknown
            # cells have grey dots, distinct from the coloured strength mesh.
            target[available] = (130, 150, 170, 18)
            unknown = supported & ~valid
            dots = ((np.floor(screen[:, 0]).astype(int) % 12 < 2)
                    & (np.floor(screen[:, 1]).astype(int) % 12 < 2))
            target[unknown] = (140, 140, 140, 16)
            target[unknown & dots] = (160, 160, 160, 160)
            if reference is not None:
                # Four area samples soften aliasing when a very large widget
                # must share the same bounded texture budget.
                coverage = np.zeros(len(points), dtype=np.float32)
                offsets = ((0., 0.),) if scale == 1. else ((-.25, -.25), (.25, -.25), (-.25, .25), (.25, .25))
                for dx, dy in offsets:
                    coverage += hatch_mask(magnitude, reference, screen[:, 0]+dx*step_x,
                                           screen[:, 1]+dy*step_y)
                marked = available & (coverage > 0)
                target[marked, :3] = (colour.red(), colour.green(), colour.blue())
                target[marked, 3] = np.asarray(220*coverage[marked]/len(offsets), dtype=np.uint8)
        if not any_support:
            self._density_notice = "No complete sampled cells here; refine the section or reduce its extent."
        image = QImage(rgba.data, width, height, width*4, QImage.Format.Format_RGBA8888).copy()
        self._density_image_cache = (key, image, self._density_notice)
        return image

    def _draw_density_key(self, painter, foreground):
        """Small swatches explain quantised density without consuming plot space."""
        from temsim.field_density_texture import hatch_mask
        fields = self._fields()
        if len(fields) != 1:
            return
        colour = fields[0][-1]
        labels = ((0., "0"), (.0625, ">0"), (.125, "1/8"), (.25, "1/4"),
                  (.5, "1/2"), (1., "1"), (2., "2+"))
        width = min(82., (self.width()-24.)/len(labels))
        for index, (value, label) in enumerate(labels):
            x = 8.+index*width
            rect = QRectF(x, self.height()-23., 28., 16.)
            yy, xx = np.mgrid[:16, :28]
            mask = hatch_mask(np.full(xx.shape, value), 1., xx, yy)
            pixels = np.zeros((16, 28, 4), dtype=np.uint8)
            pixels[mask] = (colour.red(), colour.green(), colour.blue(), 220)
            image = QImage(pixels.data, 28, 16, 28*4, QImage.Format.Format_RGBA8888)
            painter.drawImage(rect, image)
            painter.setPen(QPen(foreground, .5))
            painter.drawRect(rect)
            painter.drawText(QRectF(x+31, rect.top(), width-31, 16),
                             Qt.AlignmentFlag.AlignVCenter, label)
        if self.width() > 650:
            painter.drawText(QRectF(8.+len(labels)*width, self.height()-23., 220, 16),
                             Qt.AlignmentFlag.AlignVCenter, "× fixed reference")

    def _density_hover_text(self, position):
        grid = self._magnitude_grid()
        inverse = self._section_inverse(self._plot_rect())
        if grid is None or inverse is None:
            return None
        point = self._section_positions(np.array([[position.x(), position.y()]]), inverse)
        if (self._observation_regions is not None
                and not self._observation_mask(point)[0]):
            return None
        magnitude, supported, valid = grid.evaluate(point)
        if not supported[0]:
            return None
        xyz = point[0]*1000
        text = f"X={xyz[0]:.6g}, Y={xyz[1]:.6g}, Z={xyz[2]:.6g} mm\n"
        label, unit, _values, _valid, _colour = self._fields()[0]
        if valid[0]:
            text += f"|{label}| ≈ {magnitude[0]:.6g} {unit}\nFull XYZ magnitude of interpolated cached vectors."
        else:
            text += f"{label}: unavailable / outside model validity (not zero)."
        return text

    def set_source_regions(self, regions):
        self._source_regions = tuple((str(label), np.asarray(bounds, dtype=float).copy()) for label, bounds in regions)
        self.update()

    def set_physical_meshes(self, meshes):
        """Set existing placed material meshes, with global XYZ in metres.

        TriangleMesh normally uses millimetres. The field-scene adapter makes
        this explicit conversion before calling this display-only API.
        """
        accepted = []
        for mesh in meshes:
            vertices = np.asarray(mesh.vertices, dtype=float)
            faces = np.asarray(mesh.faces)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError("Physical mesh vertices must be finite global XYZ metres")
            if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
                raise ValueError("Physical mesh faces must be integer triangle indices")
            if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError("Physical mesh face index is outside its vertices")
            if len(vertices) and len(faces) and not mesh.wireframe:
                accepted.append(mesh)
        self._physical_meshes = tuple(accepted)
        self._section_cache = None
        self._update_view_bounds()
        self.update()

    def set_model_visible(self, visible):
        self._model_visible = bool(visible)
        self.update()

    def set_model_opacity(self, opacity):
        opacity = float(opacity)
        if not np.isfinite(opacity) or not 0. <= opacity <= 1.:
            raise ValueError("Physical model opacity must be between zero and one")
        self._model_opacity = opacity
        self.update()

    def _update_view_bounds(self):
        self._view_bounds_m = self._bounds_m.copy()
        if self._fit_models and self._physical_meshes:
            minima = [self._bounds_m[0]] + [mesh.vertices.min(axis=0) for mesh in self._physical_meshes]
            maxima = [self._bounds_m[1]] + [mesh.vertices.max(axis=0) for mesh in self._physical_meshes]
            self._view_bounds_m = np.array([np.min(minima, axis=0), np.max(maxima, axis=0)])

    def fit_physical_models(self):
        """Fit the union of model and sample extents; never alter the sample."""
        self._fit_models = True
        self._zoom, self._pan = 1., QPointF()
        self._update_view_bounds()
        if self._section == "3D":
            # The independently scaled bounding box has half-width .5 on
            # every axis; include its rotated corners with a 10% margin.
            self._zoom = .9 / (.72 * np.abs(self._basis()[:2]).sum(axis=1).max())
        else:
            self._zoom = .9
        self._view_updated()

    def reset_view(self):
        self._zoom, self._pan = 1., QPointF()
        self._fit_models = False
        self._update_view_bounds()
        self._view_updated()

    def _plot_rect(self):
        return QRectF(64., 30., max(1., self.width() - 88.), max(1., self.height() - 111.))

    def _projection(self, plot=None):
        """One affine transform shared by material, samples and directions."""
        plot = self._plot_rect() if plot is None else plot
        centre = np.mean(self._view_bounds_m, axis=0)
        span = np.maximum(np.ptp(self._view_bounds_m, axis=0), 1e-15)
        if self._section != "XY":
            # A common transverse scale is essential: independently scaling X
            # and Y before rotation would change the physical observation angle.
            span[:2] = max(span[0], span[1])
        size = np.array([plot.width(), -plot.height()]) * self._zoom
        if self._section == "3D":
            size *= .72
        origin = np.array([plot.center().x() + self._pan.x(), plot.center().y() + self._pan.y()])
        return centre, span, self._basis(), size, origin

    def _project_positions(self, positions, plot=None):
        """Return screen XY plus affine camera depth for global XYZ metres."""
        centre, span, basis, size, origin = self._projection(plot)
        projected = ((np.asarray(positions, dtype=float) - centre) / span) @ basis.T
        projected[..., :2] = projected[..., :2] * size + origin
        return projected

    def _physical_sections(self):
        """Actual mesh cuts at the sampled plane, returned as global XYZ m."""
        if self._section == "3D":
            return ()
        if self._section == "UZ":
            key = ("UZ", self._projection_angle_deg)
            if self._section_cache is not None and self._section_cache[0] == key:
                return self._section_cache[1]
            from temsim.assembly_section import mesh_section
            from temsim.field_density_texture import from_longitudinal
            cuts = []
            for mesh in self._physical_meshes:
                section = mesh_section(mesh, self._projection_angle_deg, tolerance_mm=1e-11)
                if len(section.segments_mm):
                    zu = section.segments_mm.reshape(-1, 2)
                    local = np.column_stack((zu[:, 1], np.zeros(len(zu)), zu[:, 0]))
                    segments = from_longitudinal(local, self._projection_angle_deg).reshape(-1, 2, 3)
                    cuts.append((mesh, segments))
            self._section_cache = key, tuple(cuts)
            return self._section_cache[1]
        first, second, normal = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}[self._section]
        plane = float(self._section_centre()[normal])
        key = (self._section, plane)
        if self._section_cache is not None and self._section_cache[0] == key:
            return self._section_cache[1]
        from temsim.assembly_section import mesh_section
        cuts = []
        for mesh in self._physical_meshes:
            # mesh_section is unit-neutral: permute the selected plane into
            # local Y=0, retaining metres and using a metre tolerance (10 pm).
            vertices = np.asarray(mesh.vertices)[:, [first, normal, second]].copy()
            vertices[:, 1] -= plane
            section = mesh_section(replace(mesh, vertices=vertices), 0., tolerance_mm=1e-11)
            segments = np.empty(section.segments_mm.shape[:-1] + (3,))
            segments[..., normal] = plane
            segments[..., first] = section.segments_mm[..., 1]
            segments[..., second] = section.segments_mm[..., 0]
            if len(segments):
                cuts.append((mesh, segments))
        output = tuple(cuts)
        self._section_cache = key, output
        return output

    @property
    def model_overlay_visible(self):
        return bool(self._model_visible and self._model_opacity > 0. and self._physical_meshes)

    def _model_colour(self, mesh, *, group_surface=False):
        rgba = np.asarray((*tuple(mesh.color)[:3], mesh.color[3] if len(mesh.color) > 3 else 1.), dtype=float)
        opacity = self._model_opacity
        if mesh.key in self._selected_component_keys:
            rgba[:3] = .55 * rgba[:3] + .45 * np.array([1., .77, .33])
            # Selected surfaces retain their material alpha in the temporary
            # layer. The whole selected assembly receives the opacity once.
            opacity = 1. if group_surface else min(opacity, .35)
        rgba[3] *= opacity
        return np.clip(rgba, 0., 1.)

    def _draw_physical_models(self, painter, plot):
        if not self.model_overlay_visible:
            return
        self._draw_model_pass(painter, plot, selected=False)
        if not any(mesh.key in self._selected_component_keys and self._model_colour(mesh)[3] > 0.
                   for mesh in self._physical_meshes):
            return
        # One viewport-sized software layer bounds memory independently of the
        # number of selected children. Front/back faces and adjacent windings
        # cannot accumulate beyond the assembly's 35% inspection opacity.
        viewport = plot.toAlignedRect().intersected(self.rect())
        if viewport.isEmpty():
            return
        ratio = float(self.devicePixelRatioF())
        layer = QImage(math.ceil(viewport.width()*ratio), math.ceil(viewport.height()*ratio),
                       QImage.Format.Format_ARGB32_Premultiplied)
        layer.setDevicePixelRatio(ratio)
        layer.fill(Qt.GlobalColor.transparent)
        layer_painter = QPainter(layer)
        try:
            layer_painter.setRenderHints(painter.renderHints())
            layer_painter.translate(-viewport.x(), -viewport.y())
            layer_painter.setClipRect(plot)
            self._draw_model_pass(layer_painter, plot, selected=True)
        finally:
            layer_painter.end()
        painter.save()
        painter.setOpacity(painter.opacity()*min(self._model_opacity, .35))
        painter.drawImage(QPointF(viewport.topLeft()), layer)
        painter.restore()

    def _draw_model_pass(self, painter, plot, *, selected):
        meshes = tuple(mesh for mesh in self._physical_meshes
                       if (mesh.key in self._selected_component_keys) == selected)
        if not meshes:
            return
        if self._section != "3D":
            for mesh, segments in self._physical_sections():
                if (mesh.key in self._selected_component_keys) != selected:
                    continue
                colour = QColor.fromRgbF(*self._model_colour(mesh, group_surface=selected))
                painter.setPen(QPen(colour, 2.2 if selected else 1.7))
                for segment in self._project_positions(segments, plot):
                    painter.drawLine(QPointF(*segment[0, :2]), QPointF(*segment[1, :2]))
            return
        # Depth-sort each pass's real material triangles. Selected assemblies
        # form a translucent inspection overlay; vectors remain foreground.
        triangles, colours = [], []
        centre, span, basis, _size, _origin = self._projection(plot)
        for mesh in meshes:
            projected = self._project_positions(mesh.vertices, plot)[mesh.faces]
            triangles.append(projected)
            colour = self._model_colour(mesh, group_surface=selected)
            world = ((mesh.vertices - centre) / span) @ basis.T
            face_vertices = world[mesh.faces]
            normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
            norm = np.linalg.norm(normals, axis=1)
            shade = .62 + .38 * np.divide(np.abs(normals[:, 2]), norm, out=np.zeros_like(norm), where=norm > 0)
            rgba = np.tile(colour, (len(projected), 1))
            rgba[:, :3] *= shade[:, None]
            colours.append(np.clip(rgba, 0., 1.))
        triangles, colours = np.concatenate(triangles), np.concatenate(colours)
        order = np.argsort(triangles[:, :, 2].mean(axis=1), kind="stable")
        painter.setPen(Qt.PenStyle.NoPen)
        for index in order:
            triangle = triangles[index, :, :2]
            # Rectangle culling only; no material faces are decimated.
            if (triangle[:, 0].max() < plot.left() or triangle[:, 0].min() > plot.right()
                    or triangle[:, 1].max() < plot.top() or triangle[:, 1].min() > plot.bottom()):
                continue
            painter.setBrush(QColor.fromRgbF(*colours[index]))
            painter.drawPolygon(QPolygonF([QPointF(*point) for point in triangle]))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _fields(self):
        if self._sample is None:
            return ()
        s = self._sample
        fields = {
            "B": ("B", "T", s.b_t, s.magnetic_valid, QColor("#36a9df")),
            "E": ("E", "V/m", s.e_v_per_m, s.electric_valid, QColor("#df9b30")),
            "force": ("F", "N", s.force_n, s.force_valid, QColor("#ac73e6")),
        }
        return tuple(fields[key] for key in (("B", "E") if self._mode == "overlay" else (self._mode,)))

    def _basis(self):
        if self._section == "XY":
            return np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        angle = math.radians(self._projection_angle_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        # Proper right-handed camera (Z,U,V), with screen Y inverted later.
        return np.array([[0., 0., 1.], [cosine, sine, 0.], [-sine, cosine, 0.]])

    def _longitudinal_box_cut(self, bounds):
        """Intersect an enclosing box with V=0; keep Z/U order for its outline."""
        corners = np.array([[bounds[x, 0], bounds[y, 1], bounds[z, 2]]
                            for x in (0, 1) for y in (0, 1) for z in (0, 1)])
        distances = corners @ self._basis()[2]
        tolerance = 64*np.finfo(float).eps*max(float(np.max(np.abs(corners[:, :2]))), 1e-12)
        on_plane = np.abs(distances) <= tolerance
        points = list(corners[on_plane])
        for start in range(8):
            for bit in (1, 2, 4):
                end = start ^ bit
                if start < end and not on_plane[start] and not on_plane[end] and distances[start]*distances[end] < 0:
                    fraction = distances[start]/(distances[start]-distances[end])
                    points.append(corners[start]+fraction*(corners[end]-corners[start]))
        if not points:
            return np.empty((0, 3))
        points = np.unique(points, axis=0)
        coordinates = points @ self._basis()[:2].T
        centred = coordinates-coordinates.mean(axis=0)
        return points[np.argsort(np.arctan2(centred[:, 1], centred[:, 0]))]

    def _glyph_indices(self, projected, plot=None, *, cell_px=24., max_glyphs=1000):
        """Choose visible representatives without modifying cached SI samples.

        Opposite out-of-plane signs receive distinct representatives in the
        same screen cell, including when 3D samples project onto each other.
        Unavailable values keep their own marker class. Only glyph density is
        reduced; sample validity, values and physical references stay intact.
        """
        plot = self._plot_rect() if plot is None else plot
        projected = np.asarray(projected)
        points = projected[:, :2]
        inside = (np.isfinite(points).all(axis=1)
                  & (points[:, 0] >= plot.left()) & (points[:, 0] <= plot.right())
                  & (points[:, 1] >= plot.top()) & (points[:, 1] <= plot.bottom()))
        candidates = np.flatnonzero(inside)
        if not len(candidates) or max_glyphs < 1:
            return np.empty(0, dtype=int)
        basis = self._basis()
        classes = []
        for _label, _unit, values, valid, _colour in self._fields():
            available = np.asarray(valid, dtype=bool) & np.isfinite(values).all(axis=1)
            normal = np.asarray(values) @ basis[2]
            classes.append(np.where(available, np.where(normal > 0, 1, np.where(normal < 0, -1, 0)), 2))
        classes = np.column_stack(classes)
        spacing = max(1., float(cell_px))
        origin = np.array([plot.left(), plot.top()])
        for _attempt in range(12):
            cells = np.floor((points[candidates]-origin)/spacing).astype(int)
            centres = origin + (cells+.5)*spacing
            distances = np.sum((points[candidates]-centres)**2, axis=1)
            depth = projected[candidates, 2] if projected.shape[1] > 2 else np.zeros(len(candidates))
            order = np.lexsort((candidates, -depth, distances))
            chosen = {}
            for offset in order:
                index = int(candidates[offset])
                key = (*cells[offset], *classes[index])
                if key not in chosen:
                    chosen[key] = index
            if len(chosen) <= max_glyphs:
                return np.array(sorted(chosen.values()), dtype=int)
            spacing *= max(1.1, math.sqrt(len(chosen)/max_glyphs))
        # A hard limit also bounds unusual imported data with many overlaps.
        return np.array(sorted(chosen.values())[:max_glyphs], dtype=int)

    def pick_component_at(self, position):
        """Pick actual visible geometry, never a field-support bounding box."""
        if self._sample is None or not self.model_overlay_visible or not self._plot_rect().contains(position):
            return None
        cursor = np.array([position.x(), position.y()])
        if self._section != "3D":
            hits = []
            for mesh, segments in self._physical_sections():
                if self._model_colour(mesh)[3] <= 0.:
                    continue
                projected = self._project_positions(segments)
                vectors = projected[:, 1, :2] - projected[:, 0, :2]
                squared = np.sum(vectors*vectors, axis=1)
                amount = np.clip(np.divide(np.sum((cursor-projected[:, 0, :2])*vectors, axis=1), squared,
                                          out=np.zeros(len(squared)), where=squared > 0), 0., 1.)
                closest = projected[:, 0] + amount[:, None] * (projected[:, 1]-projected[:, 0])
                distances = np.sum((closest[:, :2]-cursor)**2, axis=1)
                index = int(np.argmin(distances))
                if distances[index] <= 25.:
                    hits.append((float(distances[index]), -float(closest[index, 2]), mesh.key))
            return min(hits)[2] if hits else None
        hits = []
        for mesh in self._physical_meshes:
            if self._model_colour(mesh)[3] <= 0.:
                continue
            triangles = self._project_positions(mesh.vertices)[mesh.faces]
            a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
            edge1, edge2, relative = b[:, :2]-a[:, :2], c[:, :2]-a[:, :2], cursor-a[:, :2]
            determinant = edge1[:, 0]*edge2[:, 1]-edge1[:, 1]*edge2[:, 0]
            regular = np.abs(determinant) > 1e-10
            u = np.divide(relative[:, 0]*edge2[:, 1]-relative[:, 1]*edge2[:, 0], determinant,
                          out=np.zeros(len(a)), where=regular)
            v = np.divide(edge1[:, 0]*relative[:, 1]-edge1[:, 1]*relative[:, 0], determinant,
                          out=np.zeros(len(a)), where=regular)
            inside = regular & (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1.+1e-9)
            if np.any(inside):
                depth = a[:, 2] + u*(b[:, 2]-a[:, 2]) + v*(c[:, 2]-a[:, 2])
                hits.append((-float(depth[inside].max()), mesh.key))
        return min(hits)[1] if hits else None

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.base())
        foreground = palette.text().color()
        painter.setPen(foreground)
        plot = self._plot_rect()
        painter.drawRect(plot)
        if self._sample is None:
            painter.drawText(plot.adjusted(10, 10, -10, -10), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._message)
            painter.end()
            return
        centre, span, basis, size, origin = self._projection(plot)
        projected = self._project_positions(self._sample.points_m, plot)
        points = projected[:, :2]
        self._screen_points = points
        glyph_indices = self._glyph_indices(projected, plot) if self._render_style == "arrows" else ()
        drawn_indices = set()
        painter.save()
        painter.setClipRect(plot)
        grid_pen = QPen(foreground)
        grid_pen.setColor(QColor(foreground.red(), foreground.green(), foreground.blue(), 35))
        painter.setPen(grid_pen)
        for fraction in (.25, .5, .75):
            painter.drawLine(QPointF(plot.left() + plot.width() * fraction, plot.top()), QPointF(plot.left() + plot.width() * fraction, plot.bottom()))
            painter.drawLine(QPointF(plot.left(), plot.top() + plot.height() * fraction), QPointF(plot.right(), plot.top() + plot.height() * fraction))
        if self._render_style == "density":
            image = self._density_image(plot)
            if image is not None:
                painter.drawImage(plot, image)
            if self._density_notice:
                painter.setPen(foreground)
                painter.drawText(plot.adjusted(8, 8, -8, -8),
                                 Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._density_notice)
        region_pen = QPen(QColor(foreground.red(), foreground.green(), foreground.blue(), 90), .8, Qt.PenStyle.DashLine)
        painter.setPen(region_pen)
        for label, bounds in self._source_regions:
            if self._section == "UZ":
                corners = self._longitudinal_box_cut(bounds)
                if not len(corners):
                    continue
                edges = tuple((i, (i+1) % len(corners)) for i in range(len(corners)))
            elif self._section != "3D":
                first, second, normal = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}[self._section]
                plane = self._section_centre()[normal]
                if not bounds[0, normal] <= plane <= bounds[1, normal]:
                    continue
                corners = np.tile(centre, (4, 1))
                corners[:, normal] = plane
                corners[:, first] = bounds[[0, 1, 1, 0], first]
                corners[:, second] = bounds[[0, 0, 1, 1], second]
                edges = ((0, 1), (1, 2), (2, 3), (3, 0))
            else:
                corners = np.array([[bounds[x, 0], bounds[y, 1], bounds[z, 2]]
                                    for x in (0, 1) for y in (0, 1) for z in (0, 1)])
                edges = tuple((index, index ^ bit) for index in range(8) for bit in (1, 2, 4) if index < (index ^ bit))
            outline = (((corners-centre)/span) @ basis.T)[:, :2] * size + origin
            for first, second in edges:
                painter.drawLine(QPointF(*outline[first]), QPointF(*outline[second]))
            # Only a few labels are drawn; the complete selected catalogue and
            # numeric enclosing boxes remain available in model details.
            if len(self._source_regions) <= 5:
                painter.drawText(QPointF(*outline[0]) + QPointF(3, -3), f"{label}: model bounds")
        self._draw_physical_models(painter, plot)
        # A fixed reference has a fixed pixel length regardless of sample
        # density, so refining a viewport cannot imitate a weaker field.
        max_length = 20.
        arrow_fields = self._fields() if self._render_style == "arrows" else ()
        for field_index, (_label, _unit, values, valid, colour) in enumerate(arrow_fields):
            values = np.asarray(values)
            valid = np.asarray(valid, dtype=bool) & np.isfinite(values).all(axis=1)
            norms = np.linalg.norm(values[valid], axis=1)
            peak = float(norms.max()) if len(norms) else 0.
            reference = self.references[_label]
            directions = self._display_vectors(values, valid, basis, span, size, max_length, reference)
            painter.setPen(QPen(colour, 1.15))
            for index in glyph_indices:
                point = points[index]
                if not valid[index]:
                    delta = 2.2
                    offset = (field_index - .5) * 3.5 if self._mode == "overlay" else 0.
                    x, y = point + offset
                    painter.setPen(QPen(QColor(colour.red(), colour.green(), colour.blue(), 110), .9))
                    painter.drawRect(QRectF(x-delta, y-delta, 2*delta, 2*delta))
                    drawn_indices.add(int(index))
                    painter.setPen(QPen(colour, 1.15))
                    continue
                vector = directions[index]
                length = np.linalg.norm(vector)
                if length > .8:
                    end = point + vector
                    painter.drawLine(QPointF(*point), QPointF(*end))
                    unit = vector / length
                    side = np.array([-unit[1], unit[0]])
                    head = min(4., length * .4)
                    painter.drawLine(QPointF(*end), QPointF(*(end - head * unit + .45 * head * side)))
                    painter.drawLine(QPointF(*end), QPointF(*(end - head * unit - .45 * head * side)))
                    drawn_indices.add(int(index))
                if peak > 0:
                    normal = float(values[index] @ basis[2])
                    if abs(normal) > 0 and reference:
                        radius = min(8., max(.5, 3. * abs(normal) / reference))
                        spot = QPointF(*point)
                        painter.drawEllipse(spot, radius, radius)
                        drawn_indices.add(int(index))
                        if normal > 0:
                            painter.drawPoint(spot)
                        else:
                            painter.drawLine(spot + QPointF(-radius*.6, -radius*.6), spot + QPointF(radius*.6, radius*.6))
                            painter.drawLine(spot + QPointF(-radius*.6, radius*.6), spot + QPointF(radius*.6, -radius*.6))
                elif length <= .8:
                    painter.drawPoint(QPointF(*point))
                    drawn_indices.add(int(index))
        self._visible_sample_indices = np.array(sorted(drawn_indices), dtype=int)
        painter.restore()
        painter.setPen(foreground)
        longitudinal = self._section != "XY"
        centres = (centre[2], float(centre @ basis[1])) if longitudinal else (centre[0], centre[1])
        spans = (span[2], span[0]) if longitudinal else (span[0], span[1])
        for frac in (0., .5, 1.):
            # Ticks describe projected physical coordinates, including pan/zoom.
            value = centres[0] + ((plot.left()+frac*plot.width()-origin[0])/size[0]) * spans[0]
            painter.drawText(QRectF(plot.left()+frac*plot.width()-45, plot.bottom()+4, 90, 18), Qt.AlignmentFlag.AlignCenter, f"{value*1000:.5g}")
            value = centres[1] + ((plot.bottom()-frac*plot.height()-origin[1])/size[1]) * spans[1]
            painter.drawText(QRectF(1, plot.bottom()-frac*plot.height()-9, 57, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{value*1000:.5g}")
        axes = "Z [mm] / vertical U [mm] · independent display scales" if longitudinal else "X [mm] / vertical Y [mm] · axes independently scaled"
        painter.drawText(QRectF(plot.left(), plot.bottom()+21, plot.width(), 20), Qt.AlignmentFlag.AlignCenter, axes)
        painter.drawText(QRectF(8, 3, self.width()-16, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.projection_caption)
        normal_axis = "V" if longitudinal else "Z"
        footer = (f"Hollow square: unavailable; dot: +{normal_axis}; cross: -{normal_axis}; "
                  "arrows capped at 4x reference")
        if self._render_style == "density":
            footer = "Denser = stronger; grey dots = unavailable; empty = unsampled / outside. Scale: strength / ref"
        painter.drawText(QRectF(8, self.height()-41, self.width()-16, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         footer)
        if self._render_style == "density":
            self._draw_density_key(painter, foreground)
        elif self.model_overlay_visible:
            qualifier = "Translucent material surfaces" if self._section == "3D" else "Material boundaries at the sampled section"
            painter.drawText(QRectF(8, self.height()-22, self.width()-16, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             f"{qualifier}; field arrows drawn in front for visibility")
        painter.end()

    @property
    def projection_caption(self):
        if self._section == "UZ":
            return (f"Longitudinal U-Z section · shared angle {self._projection_angle_deg:g}° "
                    "· Z right, U up · V = 0")
        if self._section == "XY":
            return "Sampled XY section · axial XY view (sampling plane unchanged)"
        sampled = "3D volume" if self._section == "3D" else f"{self._section} section (plane unchanged)"
        return f"Sampled {sampled} · Ray projection {self._projection_angle_deg:g}°: Z right, U up"

    def mousePressEvent(self, event: QMouseEvent):
        if (event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton)
                and self._plot_rect().contains(event.position())):
            self._drag = event.position()
            self._press_position = event.position()
            self._drag_button = event.button()
            self._dragged = False
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._drag is None or event.button() != self._drag_button:
            super().mouseReleaseEvent(event)
            return
        if not self._dragged and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            key = self.pick_component_at(event.position())
            if key is not None:
                self.set_selected_component(key)
                self.component_selected.emit(key)
                if event.button() == Qt.MouseButton.RightButton:
                    self.component_context_requested.emit(key, event.globalPosition().toPoint())
        self._drag = None
        self._press_position = None
        self._drag_button = Qt.MouseButton.NoButton
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag is not None:
            displacement = event.position()-self._press_position
            self._dragged |= abs(displacement.x())+abs(displacement.y()) >= 4.
            if not self._dragged:
                event.accept()
                return
            delta = event.position() - self._drag
            self._drag = event.position()
            if self._section in ("UZ", "3D") and self._drag_button == Qt.MouseButton.LeftButton:
                previous = self._projection_angle_deg
                self.set_projection_angle(self._projection_angle_deg+delta.x()*.5)
                if self._projection_angle_deg != previous:
                    self.projection_angle_requested.emit(self._projection_angle_deg)
            else:
                self._pan += delta
                self._view_updated()
            event.accept()
            return
        if (self._render_style == "density" and self._sample is not None
                and self._plot_rect().contains(event.position())):
            text = self._density_hover_text(event.position())
            if text:
                QToolTip.showText(event.globalPosition().toPoint(), text, self)
                return
        if (self._sample is not None and len(self._visible_sample_indices)
                and self._plot_rect().contains(event.position())):
            point = np.array([event.position().x(), event.position().y()])
            indices = self._visible_sample_indices
            distance = np.sum((self._screen_points[indices] - point)**2, axis=1)
            nearest = int(np.argmin(distance))
            index = int(indices[nearest])
            if distance[nearest] < 144:
                xyz = self._sample.points_m[index] * 1000
                lines = [f"X={xyz[0]:.6g}, Y={xyz[1]:.6g}, Z={xyz[2]:.6g} mm"]
                for label, unit, values, valid, _colour in self._fields():
                    value = values[index]
                    lines.append(f"{label} XYZ = ({value[0]:.6g}, {value[1]:.6g}, {value[2]:.6g}) {unit}" if valid[index] else f"{label}: outside model support / unmodelled")
                QToolTip.showText(event.globalPosition().toPoint(), "\n".join(lines), self)
                return
        QToolTip.hideText()

    def wheelEvent(self, event: QWheelEvent):
        factor = math.exp(event.angleDelta().y() / 120. * .15)
        previous = self._zoom
        self._zoom = min(50., max(.05, self._zoom * factor))
        ratio = self._zoom/previous
        # Preserve the physical point beneath the cursor. Do not anchor to
        # a label outside the plot when a wheel event arrives there.
        plot = self._plot_rect()
        cursor = event.position() if plot.contains(event.position()) else plot.center()
        offset = cursor-plot.center()-self._pan
        self._pan += (1.-ratio)*offset
        self._view_updated()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._view_updated()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)
