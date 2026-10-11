"""Captured mechanical positions and separately tilted optical references."""

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from temsim.lens_pose import (
    ComponentDatums, PlaneDatum, PHYSICAL_POSE_FIELDS, component_datums_mm,
)


def display_datums(part, result):
    """Resolve real assemblies strictly; accept old unposed diagnostic fixtures."""
    if hasattr(part, "module_key"):
        assembly = getattr(result, "assembly", None)
        if assembly is None:
            assembly = getattr(result, "state_snapshot", None)
        return component_datums_mm(part, assembly)
    data = part.data
    if (any(name in data for name in PHYSICAL_POSE_FIELDS)
            or data.get("parent_key") or data.get("physical_host_key")
            or data.get("model_3d", {}).get("transform")):
        raise ValueError("Placed component markers require their captured assembly")
    centre = float(part.center_z_mm)
    local = float(data.get("local_center_z_mm", centre))
    normal = (0., 0., 1.)
    reference = None
    interactions = ()
    if not data.get("mechanical_only", False):
        virtual = "virtual_reference_local_z_mm" in data
        value = data.get("virtual_reference_local_z_mm", data.get("optical_reference_local_z_mm"))
        if value is not None:
            reference = PlaneDatum((0., 0., centre - local + float(value)), normal,
                                   "virtual_reference" if virtual else "optical_reference", virtual)
        interactions = tuple(PlaneDatum((0., 0., centre - local + float(z)), normal,
                                       "interaction_plane")
                             for z in data.get("interaction_centers_local_z_mm", ()))
    return ComponentDatums((0., 0., centre), reference, interactions, "hardware", False,
                           ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))


def datum_tooltip(name, datums):
    def xyz(point):
        return "(" + ", ".join(f"{v:.9g}" for v in point) + ")"
    lines = [str(name), f"Mechanical centre X/Y/Z = {xyz(datums.mechanical_center_mm)} mm",
             "Declared assembly datum; not the edited solid's bounding-box centre."]
    ref = datums.optical_reference
    if ref is not None:
        title = "Virtual optical reference" if ref.is_virtual else "Optical reference"
        lines += [f"{title} X/Y/Z = {xyz(ref.point_mm)} mm",
                  f"Plane normal X/Y/Z = {xyz(ref.normal)}"]
        if np.allclose(datums.mechanical_center_mm, ref.point_mm, atol=1e-9, rtol=0):
            lines.append("Mechanical centre and optical reference coincide.")
        lines.append("Reference of the finite optical model; not an inferred focal or conjugate plane.")
    else:
        lines.append("No separate optical reference plane is declared for this part.")
    if datums.representation_role == "control_channel":
        lines.append("Control channel reference; no extra material body is implied.")
    if datums.is_virtual:
        lines.append("Virtual assembly datum; no material surface is implied.")
    lines.append("Captured result: optical-axis tilt does not rotate the mechanical envelope. "
                 "Position offsets remain shared; optical strength is unchanged.")
    return "\n".join(lines)


def update_datum_projection(view, item):
    """Project a stored 3D point and the central section of its reference plane."""
    pair = getattr(item, "_datum_pair_points_mm", None)
    if pair is not None:
        points = np.asarray(pair, dtype=float)
        item.setData(points[:, 2], view._project_transverse(points[:, 0], points[:, 1]))
        return
    point = getattr(item, "datum_point_mm", None)
    if point is None:
        return
    u = float(view._project_transverse(point[0], point[1]))
    if isinstance(item, pg.ScatterPlotItem):
        item.setData(x=[point[2]], y=[u])
        return
    if not isinstance(item, pg.InfiniteLine) or not hasattr(item, "_datum_pen"):
        return
    item.setPos((point[2], u))
    normal = item.datum_normal
    defined = True
    if normal is None:
        item.setAngle(90)
    else:
        nu = float(view._project_transverse(normal[0], normal[1]))
        nz = normal[2]
        defined = math.hypot(nu, nz) > 1e-12
        item.setAngle(math.degrees(math.atan2(nz, -nu)) % 180 if defined else 90)
    # An edge-on viewing direction can put the entire longitudinal section in
    # the plane. Show its datum point, not an invented unique line in that case.
    item.setPen(item._datum_pen if defined else pg.mkPen(None))
    tooltip = item._datum_tooltip
    if normal is None:
        tooltip += "\nDotted Z guide through the mechanical centre; not an optical plane."
    elif defined:
        tooltip += ("\nLine: central section through this reference point in the current view."
                    "\nTransverse display magnification means its drawn angle is not a physical tilt angle.")
    else:
        tooltip += "\nThis view has no unique plane-section line; the reference point remains shown."
    item.setToolTip(tooltip)
    item.label.setToolTip(tooltip)
    anchor = getattr(item, "datum_anchor", None)
    if anchor is not None:
        anchor.setData(x=[point[2]], y=[u])
        anchor.setToolTip(tooltip)


def add_datum_line(view, part, point, normal, role, label, tooltip, *,
                   colour="#22d3ee", width=1.3, label_position=.8,
                   target=None, priority=True):
    line = pg.InfiniteLine(
        pos=(point[2], 0.), angle=90,
        pen=pg.mkPen(colour, width=width,
                     style=Qt.PenStyle.DotLine if normal is None else Qt.PenStyle.DashLine),
        label=label,
        labelOpts={"position": label_position, "color": colour, "rotateAxis": (1, 0)},
    )
    line.component_key = str(part.key)
    line.datum_role = role
    line.datum_point_mm = tuple(point)
    line.datum_normal = None if normal is None else tuple(normal)
    line._datum_pen = line.pen
    line._datum_tooltip = tooltip
    line.setZValue(11 if role == "interaction_plane" else 6)
    view._register_ray_label(line.label)
    view.plot.addItem(line)
    # Keep the actual transverse location visible even when a Z guide spans
    # the plot. The static layer owns this point together with its line.
    anchor = pg.ScatterPlotItem(symbol="+", size=7, pen=pg.mkPen(colour, width=1.2))
    anchor.setZValue(12)
    view.plot.addItem(anchor)
    line.datum_anchor = anchor
    (view.component_marker_items if target is None else target).append(line)
    view._component_labels.append((line.label, point[2], priority))
    update_datum_projection(view, line)
    return line
