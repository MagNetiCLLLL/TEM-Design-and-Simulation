"""Analytic construction shared by CAD export and native display.

All coordinates are module-local millimetres. A SolidSpec describes material,
never a field or an optical calibration. STEP uses analytic B-rep; the GUI
tessellates these same parameters without importing OpenCascade.
"""
from dataclasses import dataclass, field


MATERIAL_COLORS = {
    "copper_winding": (.83, .43, .20, 1.),
    "soft_magnetic": (.36, .49, .64, 1.),
    "insulator": (.87, .83, .68, 1.),
    "nonmagnetic_housing": (.61, .65, .69, 1.),
    "electrode": (.71, .73, .77, 1.),
    "tungsten": (.50, .54, .60, 1.),
    "silicon": (.28, .50, .75, 1.),
    "scintillator_unspecified": (.51, .78, .47, 1.),
    "optical_coupler_unspecified": (.42, .69, .74, 1.),
}


@dataclass(frozen=True)
class SolidSpec:
    key: str
    region: str
    material: str
    shape: dict
    evidence: tuple[str, ...] = ()
    status: str = "engineering_reconstruction"
    note: str = "Internal dimensions are reconstruction parameters, not OEM measurements."
    parameters: dict = field(default_factory=dict)

    @property
    def color(self):
        if "copper" in self.material or "winding" in self.material:
            return MATERIAL_COLORS["copper_winding"]
        if "soft_magnetic" in self.material:
            return MATERIAL_COLORS["soft_magnetic"]
        return MATERIAL_COLORS.get(self.material, (.64, .70, .75, 1.))


def revolve(section):
    return {"kind": "revolve", "section_zr_mm": [list(p) for p in section]}


def annulus(start, end, inner, outer):
    return revolve([(start, inner), (start, outer), (end, outer), (end, inner)])


def box(center, size):
    return {"kind": "box", "center_mm": list(center), "size_mm": list(size)}


def sector(start, end, inner, outer, angle_deg, span_deg):
    return {"kind": "annular_sector", "start_mm": start, "end_mm": end,
            "inner_mm": inner, "outer_mm": outer,
            "angle_deg": angle_deg, "span_deg": span_deg}


def difference(base, *tools):
    return {"kind": "difference", "base": base, "tools": list(tools)}


def union(*shapes):
    return {"kind": "union", "shapes": list(shapes)}


def transformed(shape, *, rotation=None, offset=(0., 0., 0.)):
    return {"kind": "transform", "shape": shape,
            "rotation": rotation if rotation is not None else [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            "offset_mm": list(offset)}
