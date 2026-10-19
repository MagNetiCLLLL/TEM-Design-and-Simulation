"""Mechanical dimensions for the documented six-pin probe corrector family.

Lengths are module-local millimetres, independent of optical field FWHM.
The construction is a source-backed topology with engineering dimensions,
not an OEM drawing, winding prescription or a magnetic-field calibration.
"""
from __future__ import annotations

import math
from numbers import Real

PREFIX = "mechanical_hexapole_"
PROBE_HOSTS = {"probe_hp1_hexapole", "probe_hp2_hexapole"}


def is_probe_hexapole(part):
    from temsim.component_representation import non_material_role
    return (str(part.get("geometry_template_key", part.get("key", ""))) in PROBE_HOSTS
            and not non_material_role(part))


def _number(part, key, default=None):
    value = part.get(key, default)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{part.get('key', 'hexapole')}.{key} must be finite")
    return float(value)


def probe_hexapole_dimensions(part):
    """Resolve saved physical dimensions; older files use declared reconstruction defaults."""
    start, end = (_number(part, k) for k in ("local_start_z_mm", "local_end_z_mm"))
    bore = next((k for k in ("mechanical_inner_diameter_mm", "mechanical_bore_diameter_mm",
        "mechanical_clear_bore_diameter_mm", "bore_diameter_mm", "clear_bore_diameter_mm") if k in part), None)
    if bore is None:
        raise ValueError(f"{part['key']}: hexapole material bore must be declared")
    inner = _number(part, bore)/2
    outer = _number(part, "mechanical_outer_diameter_mm", part.get("outer_diameter_mm"))/2
    if not end > start or not 0 < inner < outer:
        raise ValueError(f"{part['key']}: hexapole requires positive length and an open annulus")
    if "mechanical_clear_bore_diameter_mm" in part:
        clear = _number(part, "mechanical_clear_bore_diameter_mm")
        if not 0 < clear <= 2*inner:
            raise ValueError(f"{part['key']}: hexapole material must preserve the clear bore")
    length, width = end-start, outer-inner
    available = min(2*(inner+.42*width)*math.tan(math.pi/6), .6*width)
    defaults = dict(pole_length_mm=.8*length, tip_width_mm=.16*available,
        root_width_mm=.32*available, taper_length_mm=.35*width,
        stem_height_mm=.40*length, yoke_thickness_mm=.22*width,
        coil_radial_inset_mm=.42*width, coil_radial_length_mm=.27*width,
        coil_width_mm=.80*available, coil_height_mm=.70*length,
        coil_window_width_mm=.45*available, coil_window_height_mm=.48*length)
    d = {k: _number(part, PREFIX+k, default) for k, default in defaults.items()}
    if any(v <= 0 for v in d.values()):
        raise ValueError(f"{part['key']}: mechanical hexapole dimensions must be positive")
    if not d['stem_height_mm'] <= d['pole_length_mm'] <= length+1e-9:
        raise ValueError(f"{part['key']}: mechanical hexapole pole length must enclose the stem and fit the envelope")
    if not d['stem_height_mm'] < d['coil_window_height_mm'] < d['coil_height_mm'] <= length+1e-9:
        raise ValueError(f"{part['key']}: hexapole coil window must clear the stem and fit the axial envelope")
    if not d['tip_width_mm'] < d['root_width_mm'] < d['coil_window_width_mm'] < d['coil_width_mm']:
        raise ValueError(f"{part['key']}: tapered hexapole pole and coil window widths require positive clearances")
    yoke_inner = outer-d['yoke_thickness_mm']
    coil_start = inner+d['coil_radial_inset_mm']
    coil_end = coil_start+d['coil_radial_length_mm']
    if not inner < inner+d['taper_length_mm'] < coil_start < coil_end < yoke_inner < outer:
        raise ValueError(f"{part['key']}: hexapole taper, winding and yoke must fit without radial overlap")
    if math.hypot(coil_end, d['coil_width_mm']/2) >= yoke_inner:
        raise ValueError(f"{part['key']}: hexapole winding corner intersects the return yoke")
    # Each pin and its coil must stay strictly inside its own 60-degree cell.
    for radius, full_width in ((inner, d['tip_width_mm']),
            (inner+d['taper_length_mm'], d['root_width_mm']), (coil_start, d['coil_width_mm'])):
        if full_width/2 >= radius*math.tan(math.pi/6):
            raise ValueError(f"{part['key']}: neighbouring hexapole poles or windings overlap")
    return dict(start_mm=start, end_mm=end, inner_radius_mm=inner, outer_radius_mm=outer,
                **d)


def validate_probe_hexapole_geometry(part):
    if is_probe_hexapole(part):
        probe_hexapole_dimensions(part)
