"""Software projection of sampled physical E, B and electron-force vectors.

This widget only projects supplied SI arrays. Navigation never samples a field.
Independent coordinate scales are applied to positions and vector directions
consistently, and disclosed in the plot. Out-of-plane signs remain visible in
section views; the hover readout always reports all three physical components.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QToolTip, QWidget


class FieldVectorCanvas(QWidget):
    """Bounded vector display; cyan B, amber E, and violet electron force."""

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
        self._section = "XY"
        self._mode = "overlay"
        self._zoom = 1.
        self._pan = QPointF()
        self._yaw = .55
        self._pitch = .45
        self._drag = None
        self._screen_points = np.empty((0, 2))
        self._source_regions = ()
        self.references = {"B": None, "E": None, "F": None}
        self._message = "Calculate the column, then select sources."
        self.setToolTip("Drag to pan; drag a 3D volume to rotate. Wheel: display zoom. "
                        "Double click: reset view. Hover a sample for physical XYZ components. "
                        "Camera navigation never resamples or changes the optical model.")

    def sizeHint(self):
        return QSize(680, 420)

    @property
    def sample(self):
        return self._sample

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
            reference_text = f"; arrow ref {reference:.5g}" if reference is not None else "; reference pending"
            labels.append((f"{label} [{unit}]: max {peak:.5g}" if peak is not None
                           else f"{label} [{unit}]: unavailable") + reference_text)
        return " | ".join(labels)

    def clear(self, message="No current field samples."):
        self._sample = None
        self._physical_meshes = ()
        self._section_cache = None
        self._screen_points = np.empty((0, 2))
        self._message = str(message)
        self.update()

    def set_sample(self, sample, bounds_m, section):
        self._sample = sample
        self._bounds_m = np.asarray(bounds_m, dtype=float).copy()
        # Meshes belong to the same accepted snapshot/selection as the sample;
        # publication must explicitly supply them again after a replacement.
        self._physical_meshes = ()
        self._section_cache = None
        self._view_bounds_m = self._bounds_m.copy()
        self._section = str(section)
        self.use_current_peaks(only_unset=True)
        self._screen_points = np.empty((0, 2))
        self.update()

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
            raise ValueError("Arrow reference must be finite and positive")
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
        self.update()

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
        self.update()

    def reset_view(self):
        self._zoom, self._pan = 1., QPointF()
        self._yaw, self._pitch = .55, .45
        self._fit_models = False
        self._update_view_bounds()
        self.update()

    def _plot_rect(self):
        return QRectF(64., 30., max(1., self.width() - 88.), max(1., self.height() - 111.))

    def _projection(self, plot=None):
        """One affine transform shared by material, samples and directions."""
        plot = self._plot_rect() if plot is None else plot
        centre = np.mean(self._view_bounds_m, axis=0)
        span = np.maximum(np.ptp(self._view_bounds_m, axis=0), 1e-15)
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
        first, second, normal = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}[self._section]
        plane = float(np.mean(self._bounds_m, axis=0)[normal])
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

    def _draw_physical_models(self, painter, plot):
        if not self.model_overlay_visible:
            return
        if self._section != "3D":
            for mesh, segments in self._physical_sections():
                colour = QColor.fromRgbF(*tuple(mesh.color)[:3])
                alpha = mesh.color[3] if len(mesh.color) > 3 else 1.
                colour.setAlphaF(min(1., max(0., alpha * self._model_opacity)))
                painter.setPen(QPen(colour, 1.7))
                for segment in self._project_positions(segments, plot):
                    painter.drawLine(QPointF(*segment[0, :2]), QPointF(*segment[1, :2]))
            return
        # Sort every supplied triangle globally. No sampling or synthetic
        # envelopes replace the material surfaces; vectors remain foreground.
        triangles, colours = [], []
        centre, span, basis, _size, _origin = self._projection(plot)
        for mesh in self._physical_meshes:
            projected = self._project_positions(mesh.vertices, plot)[mesh.faces]
            triangles.append(projected)
            colour = np.asarray((*tuple(mesh.color)[:3], mesh.color[3] if len(mesh.color) > 3 else 1.))
            world = ((mesh.vertices - centre) / span) @ basis.T
            face_vertices = world[mesh.faces]
            normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
            norm = np.linalg.norm(normals, axis=1)
            shade = .62 + .38 * np.divide(np.abs(normals[:, 2]), norm, out=np.zeros_like(norm), where=norm > 0)
            rgba = np.tile(colour, (len(projected), 1))
            rgba[:, :3] *= shade[:, None]
            rgba[:, 3] *= self._model_opacity
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
        if self._section == "XZ":
            return np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
        if self._section == "YZ":
            return np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        cy, sy, cp, sp = math.cos(self._yaw), math.sin(self._yaw), math.cos(self._pitch), math.sin(self._pitch)
        return np.array([[cy, -sy, 0.], [sy * sp, cy * sp, cp], [-sy * cp, -cy * cp, sp]])

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
        points = self._project_positions(self._sample.points_m, plot)[:, :2]
        self._screen_points = points
        painter.save()
        painter.setClipRect(plot)
        grid_pen = QPen(foreground)
        grid_pen.setColor(QColor(foreground.red(), foreground.green(), foreground.blue(), 35))
        painter.setPen(grid_pen)
        for fraction in (.25, .5, .75):
            painter.drawLine(QPointF(plot.left() + plot.width() * fraction, plot.top()), QPointF(plot.left() + plot.width() * fraction, plot.bottom()))
            painter.drawLine(QPointF(plot.left(), plot.top() + plot.height() * fraction), QPointF(plot.right(), plot.top() + plot.height() * fraction))
        region_pen = QPen(QColor(foreground.red(), foreground.green(), foreground.blue(), 90), .8, Qt.PenStyle.DashLine)
        painter.setPen(region_pen)
        for label, bounds in self._source_regions:
            if self._section != "3D":
                first, second, normal = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}[self._section]
                plane = np.mean(self._bounds_m, axis=0)[normal]
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
        max_length = min(30., max(8., min(plot.width(), plot.height()) / max(8., len(points) ** .5) * .7))
        for field_index, (_label, _unit, values, valid, colour) in enumerate(self._fields()):
            values = np.asarray(values)
            valid = np.asarray(valid, dtype=bool) & np.isfinite(values).all(axis=1)
            norms = np.linalg.norm(values[valid], axis=1)
            peak = float(norms.max()) if len(norms) else 0.
            reference = self.references[_label]
            directions = self._display_vectors(values, valid, basis, span, size, max_length, reference)
            painter.setPen(QPen(colour, 1.15))
            for index, point in enumerate(points):
                if not plot.adjusted(-10, -10, 10, 10).contains(QPointF(*point)):
                    continue
                if not valid[index]:
                    delta = 2.2
                    offset = (field_index - .5) * 3.5 if self._mode == "overlay" else 0.
                    x, y = point + offset
                    painter.setPen(QPen(QColor(colour.red(), colour.green(), colour.blue(), 110), .9))
                    painter.drawRect(QRectF(x-delta, y-delta, 2*delta, 2*delta))
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
                if self._section != "3D" and peak > 0:
                    normal = float(values[index] @ basis[2])
                    if abs(normal) > 0 and reference:
                        radius = min(8., max(.5, 3. * abs(normal) / reference))
                        spot = QPointF(*point)
                        painter.drawEllipse(spot, radius, radius)
                        if normal > 0:
                            painter.drawPoint(spot)
                        else:
                            painter.drawLine(spot + QPointF(-radius*.6, -radius*.6), spot + QPointF(radius*.6, radius*.6))
                            painter.drawLine(spot + QPointF(-radius*.6, radius*.6), spot + QPointF(radius*.6, -radius*.6))
                elif length <= .8:
                    painter.drawPoint(QPointF(*point))
        painter.restore()
        painter.setPen(foreground)
        if self._section != "3D":
            ax1, ax2 = ({"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)})[self._section]
            for frac in (0., .5, 1.):
                # Ticks follow display pan/zoom, not sampling coordinates.
                value = centre[ax1] + ((plot.left()+frac*plot.width()-origin[0])/size[0]) * span[ax1]
                painter.drawText(QRectF(plot.left()+frac*plot.width()-45, plot.bottom()+4, 90, 18), Qt.AlignmentFlag.AlignCenter, f"{value*1000:.5g}")
                value = centre[ax2] + ((plot.bottom()-frac*plot.height()-origin[1])/size[1]) * span[ax2]
                painter.drawText(QRectF(1, plot.bottom()-frac*plot.height()-9, 57, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{value*1000:.5g}")
            painter.drawText(QRectF(plot.left(), plot.bottom()+21, plot.width(), 20), Qt.AlignmentFlag.AlignCenter, f"{'XYZ'[ax1]} [mm]    /    vertical {'XYZ'[ax2]} [mm]")
        else:
            painter.drawText(QRectF(plot.left(), plot.bottom()+5, plot.width(), 20), Qt.AlignmentFlag.AlignCenter, "3D volume: drag to rotate; physical XYZ coordinates on hover")
        painter.drawText(QRectF(8, 3, self.width()-16, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{self._section} · physical axes independently scaled")
        normal_axis = {"XY": "+Z", "XZ": "+Y", "YZ": "+X"}.get(self._section, "normal")
        footer = "Hollow square: unavailable; arrows capped at 4x reference"
        if self._section != "3D":
            footer = (f"Hollow square: unavailable; circled dot: {normal_axis}; "
                      f"circled cross: -{normal_axis[-1]}; arrows capped at 4x reference")
        painter.drawText(QRectF(8, self.height()-41, self.width()-16, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         footer)
        if self.model_overlay_visible:
            qualifier = "Translucent material surfaces" if self._section == "3D" else "Material boundaries at the sampled section"
            painter.drawText(QRectF(8, self.height()-22, self.width()-16, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             f"{qualifier}; field arrows drawn in front for visibility")
        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag = None
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag is not None:
            delta = event.position() - self._drag
            self._drag = event.position()
            if self._section == "3D":
                self._yaw += delta.x() * .009
                self._pitch = min(1.5, max(-1.5, self._pitch + delta.y() * .009))
            else:
                self._pan += delta
            self.update()
            return
        if self._sample is not None and len(self._screen_points):
            point = np.array([event.position().x(), event.position().y()])
            distance = np.sum((self._screen_points - point)**2, axis=1)
            index = int(np.argmin(distance))
            if distance[index] < 144:
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
        self._zoom = min(50., max(.05, self._zoom * factor))
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.reset_view()
        event.accept()
