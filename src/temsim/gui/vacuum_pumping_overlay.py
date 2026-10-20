"""Pump services drawn in pixels, connected to the axial map's real Z ports."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QGraphicsObject


ROLE_COLOURS = {"high_vacuum": "#38bdf8", "roughing": "#fbbf24",
                "backing": "#c4b5fd", "transfer": "#6ee7b7"}
ROLE_NAMES = {"high_vacuum": "High vacuum", "roughing": "Roughing",
              "backing": "Backing", "transfer": "Transfer"}
PUMP_NAMES = {"igp": "IGP · ion getter", "turbo": "Turbomolecular",
              "scroll": "Dry scroll", "rotary": "Rotary vane",
              "cryo": "Cryopump", "diffusion": "Diffusion pump", "other": "Vacuum pump"}


def _font(size=8.5, bold=False):
    font = QFont()
    font.setPointSizeF(size)
    font.setBold(bold)
    return font


class VacuumPumpingOverlay(QGraphicsObject):
    """One persistent item; selection never deletes a mouse-event recipient."""

    def __init__(self, diagram):
        super().__init__(diagram.getViewBox())
        self.diagram = diagram
        self.layout = {}
        self.anchors = {}
        self.current, self.current_line = "", ""
        self.node_rects, self.line_paths, self.region_ports = {}, {}, {}
        self.line_labels = {}
        self._node_data, self._line_data = {}, {}
        self._valves = {}
        self._hover_endpoint, self._hover_line = "", ""
        self._rect, self._service_rect = QRectF(), QRectF()
        self._offscreen = 0
        self.setZValue(50)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setToolTip("Pump icons and pipe routing are schematic; their ports use the vacuum map's real Z. "
                        "Pressures remain prescribed inputs.")

    def set_layout(self, layout, anchors=None):
        self.layout = deepcopy(layout or {})
        self.anchors = dict(anchors or {})

    def set_selection(self, endpoint="", line_key=""):
        self.current, self.current_line = endpoint, line_key
        self.update()

    def required_height(self, width):
        pumps = self.layout.get("pumps", ())
        upper = sum(p["kind"] not in {"scroll", "rotary"} for p in pumps)+len(self.layout.get("auxiliaries", ()))
        lower = len(pumps)-sum(p["kind"] not in {"scroll", "rotary"} for p in pumps)
        columns = max(1, int((width-20)/122))
        rows = max(1, math.ceil(upper/columns))+math.ceil(lower/columns)
        return 30+rows*74+14

    def boundingRect(self):
        return self._rect

    def shape(self):
        path = QPainterPath()
        for rectangle in self.node_rects.values():
            path.addRect(rectangle)
        for rectangle, _ in self.line_labels.values():
            path.addRect(rectangle)
        for point in self.region_ports.values():
            if self._rect.contains(point):
                path.addEllipse(point, 6, 6)
        stroker = QPainterPathStroker()
        stroker.setWidth(9)
        for line in self.line_paths.values():
            path.addPath(stroker.createStroke(line))
        return path

    def rebuild(self, service_top):
        """Update pixel geometry without changing either physical axis."""
        vb = self.diagram.getViewBox()
        self.prepareGeometryChange()
        self._rect = QRectF(vb.boundingRect())
        self._service_rect = QRectF(0, service_top, self._rect.width(), max(0, self._rect.height()-service_top))
        self.node_rects, self.line_paths, self.region_ports = {}, {}, {}
        self.line_labels, self._valves = {}, {}
        self._node_data, self._line_data = {}, {}
        rows = [row for row in self.diagram.rows
                if row.radius_mm is None and row.end_medium is None and not row.key.startswith("transition:")]
        for row in rows:
            # Keep the true coordinate even outside the viewport. Never clamp
            # an unseen region to a fake port at the plot edge.
            scene = vb.mapViewToScene(QPointF((row.start_z_mm+row.end_z_mm)/2, 1.0))
            self.region_ports[f"region:{row.key}"] = self.mapFromScene(scene)
        visible_ports = {key for key, point in self.region_ports.items() if 0 <= point.x() <= self._rect.width()}
        self._offscreen = len(self.region_ports)-len(visible_ports)
        pumps = list(self.layout.get("pumps", ()))
        auxiliaries = list(self.layout.get("auxiliaries", ()))
        lines = list(self.layout.get("lines", ()))
        self._line_data = {line["key"]: line for line in lines}
        region_order = {f"region:{row.key}": index for index, row in enumerate(sorted(rows, key=lambda r: r.start_z_mm))}
        adjacency = defaultdict(list)
        for line in lines:
            adjacency[line["source"]].append(line["target"])
            adjacency[line["target"]].append(line["source"])

        def location(endpoint):
            values = [region_order[other] for other in adjacency[endpoint] if other in region_order]
            return sum(values)/len(values) if values else len(region_order)

        upper, lower = [], []
        for pump in pumps:
            endpoint = f"pump:{pump['key']}"
            self._node_data[endpoint] = (pump["name"], PUMP_NAMES.get(pump["kind"], pump["kind"]),
                                         "Configured pump", pump["name"])
            (lower if pump["kind"] in {"scroll", "rotary"} else upper).append((location(endpoint), endpoint))
        for auxiliary in auxiliaries:
            endpoint = f"aux:{auxiliary['key']}"
            z = self.anchors.get(auxiliary["anchor"])
            label = "Location unavailable"
            order = location(endpoint)
            if isinstance(z, (int, float)) and math.isfinite(z) and rows:
                closest = min(rows, key=lambda row: (max(row.start_z_mm-z, z-row.end_z_mm, 0),
                                                      row.end_z_mm-row.start_z_mm))
                order = region_order[f"region:{closest.key}"]+.15
                label = f"At {auxiliary['anchor']}"
            self._node_data[endpoint] = (auxiliary["name"], "Off-axis chamber", label,
                                         auxiliary["name"]+f"\nSaved location: {auxiliary['anchor']}\nNo solved pressure")
            upper.append((order, endpoint))
        columns = max(1, int((self._rect.width()-20)/122))
        cursor = service_top+32
        for tier in (sorted(upper), sorted(lower)):
            for begin in range(0, len(tier), columns):
                group = tier[begin:begin+columns]
                spacing = (self._rect.width()-20)/max(1, len(group))
                card_width = min(132, max(90, spacing-10))
                for index, (_order, endpoint) in enumerate(group):
                    x = 10+spacing*(index+.5)-card_width/2
                    self.node_rects[endpoint] = QRectF(x, cursor, card_width, 54)
                cursor += 74
        used_labels = []
        node_boxes = list(self.node_rects.values())
        group_counts = defaultdict(int)
        for number, line in enumerate(lines):
            source, target = line["source"], line["target"]
            if any(endpoint.startswith("region:") and endpoint not in visible_ports for endpoint in (source, target)):
                continue
            if any(endpoint not in self.region_ports and endpoint not in self.node_rects for endpoint in (source, target)):
                continue  # A pending split/merge redraw may precede the new graph.

            def port(endpoint, other):
                if endpoint in self.region_ports:
                    return self.region_ports[endpoint]
                rectangle = self.node_rects[endpoint]
                if other in self.region_ports:
                    return QPointF(rectangle.center().x(), rectangle.top())
                other_rect = self.node_rects[other]
                y = rectangle.top() if other_rect.top() < rectangle.top() else rectangle.bottom()
                return QPointF(rectangle.center().x(), y)

            start, end = port(source, target), port(target, source)
            group = (round(min(start.y(), end.y())), round(max(start.y(), end.y())))
            lane = group_counts[group]
            group_counts[group] += 1
            if source in self.node_rects and target in self.node_rects and self.node_rects[source].top() == self.node_rects[target].top():
                bus = start.y()+9+lane*4
            else:
                bus = (start.y()+end.y())/2+(lane % 4-1.5)*5
            points = self._route(start, end, bus, node_boxes, number)
            path = QPainterPath(start)
            for point in points[1:]:
                path.lineTo(point)
            self.line_paths[line["key"]] = path
            if line.get("valve"):
                segments = sorted(zip(points, points[1:]), key=lambda pair:
                                  abs(pair[1].x()-pair[0].x())+abs(pair[1].y()-pair[0].y()), reverse=True)
                label = "OPEN" if line["open"] else "CLOSED"
                width = QFontMetricsF(_font(7.5)).horizontalAdvance(label)+8
                choices = []
                for a, b in segments:
                    for fraction in (.5, .25, .75):
                        centre = a+(b-a)*fraction
                        for delta in (QPointF(width/2+9, 0), QPointF(-width/2-9, 0),
                                      QPointF(0, -13), QPointF(0, 13)):
                            text = centre+delta
                            rect = QRectF(text.x()-width/2, text.y()-8, width, 16)
                            overlaps = sum(rect.intersects(box) for box in node_boxes+used_labels)
                            edge = 0 if self._rect.contains(rect) else 100
                            choices.append((overlaps*1000+edge, centre, rect))
                if choices:
                    _score, centre, rectangle = min(choices, key=lambda item: item[0])
                    self._valves[line["key"]] = centre
                    self.line_labels[line["key"]] = (rectangle, label)
                    used_labels.append(rectangle)
        self.update()

    @staticmethod
    def _route(start, end, bus, boxes, number):
        """Orthogonal service routing; column endpoints remain unmodified."""
        candidates = [
            [start, QPointF(start.x(), bus), QPointF(end.x(), bus), end],
        ]
        # A detour beside a card keeps arbitrarily long auxiliary/backing pipes
        # out of pump bodies. Pipe length has no physical simulation meaning.
        for box in boxes:
            for x in (box.left()-7-number % 3*3, box.right()+7+number % 3*3):
                candidates.append([start, QPointF(start.x(), start.y()+7),
                                   QPointF(x, start.y()+7), QPointF(x, end.y()-7),
                                   QPointF(end.x(), end.y()-7), end])

        def intersections(points):
            score = 0
            for a, b in zip(points, points[1:]):
                for rectangle in boxes:
                    if a.x() == b.x():
                        hit = (rectangle.left() < a.x() < rectangle.right() and
                               max(min(a.y(), b.y()), rectangle.top()) < min(max(a.y(), b.y()), rectangle.bottom()))
                    else:
                        hit = (rectangle.top() < a.y() < rectangle.bottom() and
                               max(min(a.x(), b.x()), rectangle.left()) < min(max(a.x(), b.x()), rectangle.right()))
                    score += bool(hit)*100000
                score += abs(a.x()-b.x())+abs(a.y()-b.y())
            return score
        points = min(candidates, key=intersections)
        return [point for index, point in enumerate(points) if index == 0 or point != points[index-1]]

    def _active_lines(self):
        endpoint = self._hover_endpoint or ("" if self._hover_line else self.current)
        line_key = self._hover_line or ("" if endpoint else self.current_line)
        return {key for key, line in self._line_data.items()
                if key == line_key or endpoint in (line["source"], line["target"])}

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(self._rect)
        painter.fillRect(self._service_rect, QColor("#080e1c"))
        painter.setPen(QPen(QColor("#334155"), 1))
        painter.drawLine(self._service_rect.topLeft(), self._service_rect.topRight())
        painter.setFont(_font(8))
        painter.setPen(QColor("#94a3b8"))
        caption = "Pumps / off-axis services · icon positions schematic; ports follow real Z"
        if self._offscreen:
            caption += f" · {self._offscreen} ports outside Z view"
        painter.drawText(self._service_rect.adjusted(7, 3, -7, 0), Qt.AlignmentFlag.AlignTop, caption)
        active = self._active_lines()
        for key, path in self.line_paths.items():
            line = self._line_data[key]
            opened = not line.get("valve") or line["open"]
            colour = QColor("#fbbf24" if key in active else ROLE_COLOURS.get(line["role"], "#94a3b8"))
            if not opened and key not in active:
                colour.setAlpha(160)
            pen = QPen(colour, 2.5 if key in active else 1.4)
            if not opened:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            if key in self._valves:
                centre = self._valves[key]
                painter.setBrush(QColor("#080e1c"))
                painter.drawEllipse(centre, 4, 4)
                if not opened:
                    painter.setPen(QPen(QColor("#fb7185"), 1.5))
                    painter.drawLine(centre-QPointF(4, 4), centre+QPointF(4, 4))
                rectangle, label = self.line_labels[key]
                painter.fillRect(rectangle, QColor("#080e1c"))
                painter.setPen(colour)
                painter.setFont(_font(7.5))
                painter.drawText(rectangle, Qt.AlignmentFlag.AlignCenter, label)
        active_nodes = {self._hover_endpoint or self.current}
        for key in active:
            active_nodes.update((self._line_data[key]["source"], self._line_data[key]["target"]))
        for key, rectangle in self.node_rects.items():
            selected = key in active_nodes
            painter.setPen(QPen(QColor("#fbbf24" if selected else "#8397bd"), 2 if selected else 1))
            painter.setBrush(QColor("#27384b" if selected else "#1b2639"))
            painter.drawRoundedRect(rectangle, 4, 4)
            name, kind, footer, _tooltip = self._node_data[key]
            for offset, text, size, colour, bold in (
                (5, name, 8.5, "#f8fafc", True),
                (23, kind, 8, "#cbd5e1", False),
                (39, footer, 7.5, "#94a3b8", False),
            ):
                painter.setFont(_font(size, bold))
                painter.setPen(QColor(colour))
                text = QFontMetricsF(painter.font()).elidedText(text, Qt.TextElideMode.ElideRight, rectangle.width()-12)
                painter.drawText(rectangle.adjusted(6, offset, -6, 0), Qt.AlignmentFlag.AlignTop, text)
        painter.setPen(QPen(QColor("#67e8f9"), 1))
        painter.setBrush(QColor("#0e7490"))
        for key, point in self.region_ports.items():
            if self._rect.contains(point):
                painter.drawEllipse(point, 3, 3)

    def hit_at(self, point):
        for key, rectangle in self.node_rects.items():
            if rectangle.contains(point):
                return key, ""
        for key, port in self.region_ports.items():
            if self._rect.contains(port) and (port-point).manhattanLength() < 8:
                return key, ""
        for key, (rectangle, _label) in self.line_labels.items():
            if rectangle.contains(point):
                return "", key
        stroker = QPainterPathStroker()
        stroker.setWidth(8)
        for key, path in reversed(tuple(self.line_paths.items())):
            if stroker.createStroke(path).contains(point):
                return "", key
        return "", ""

    def mousePressEvent(self, event):
        endpoint, line_key = self.hit_at(event.pos())
        if event.button() != Qt.MouseButton.LeftButton or not (endpoint or line_key):
            event.ignore()
            return
        event.accept()
        self.set_selection(endpoint, line_key)
        if endpoint.startswith("region:"):
            self.diagram.selected.emit(endpoint.split(":", 1)[1])
        elif endpoint:
            self.diagram.pumping_selected.emit(endpoint)
        else:
            self.diagram.pumping_line_selected.emit(line_key)

    def hoverMoveEvent(self, event):
        self._hover_endpoint, self._hover_line = self.hit_at(event.pos())
        if self._hover_endpoint in self._node_data:
            self.setToolTip(self._node_data[self._hover_endpoint][3])
        elif self._hover_line in self._line_data:
            line = self._line_data[self._hover_line]
            self.setToolTip(f"{line['key']} · {ROLE_NAMES.get(line['role'], line['role'])}\n"
                            f"{line['source']} → {line['target']}\n"
                            f"{line['valve'] or 'Unvalved pipe'} · {'open' if line['open'] else 'closed'}\n"
                            "Display only; pressure is prescribed.")
        elif self._hover_endpoint.startswith("region:"):
            self.setToolTip(self._hover_endpoint+"\nActual axial midpoint of this pressure region.")
        self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover_endpoint = self._hover_line = ""
        self.update()
        super().hoverLeaveEvent(event)
