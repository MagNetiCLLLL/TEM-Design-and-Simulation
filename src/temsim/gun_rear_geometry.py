"""Tip-anchored, axisymmetric rear FEG hardware and explicit electrical ties.

The FEI patent supplies the EEE topology and approximate tip protrusion, not
the undisclosed dimensions or its normal-emission wiring. The latter remain
declared simulator design inputs. Insulator polarization is not solved.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np

SOURCE = "https://patents.google.com/patent/US8736170B1/en"
ROLES = {"emitter_enclosure_electrode", "cathode_holder", "insulator"}
REFERENCES = {"tip", "extractor", "gun_lens", "ground"}


def rear_part_dimensions(part):
    role = part.get("rear_gun_role")
    if role is None:
        return None
    if role not in ROLES:
        raise ValueError("Unknown rear gun role")
    if part.get("model_3d") or any(part.get(k, 0.) != 0. for k in
            ("offset_x_mm", "offset_y_mm", "offset_z_mm", "rotation_x_mrad", "rotation_y_mrad")):
        raise ValueError("Rear gun hardware requires its shared coaxial annular geometry; independent CAD/pose overrides are unsupported")
    names = ("local_start_z_mm", "local_end_z_mm", "bore_diameter_mm", "outer_diameter_mm")
    values = tuple(part[name] for name in names)
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in values):
        raise ValueError("Rear gun dimensions must be finite real numbers")
    start, stop, bore, outer = values
    if not start < stop < 0 or not 0 < bore < outer:
        raise ValueError("Rear gun material requires start < end < tip Z=0 and 0 < bore < outer diameter")
    if role != "insulator":
        reference, bias = part.get("rear_voltage_reference"), part.get("rear_bias_kv")
        if reference not in REFERENCES or isinstance(bias, bool) or not isinstance(bias, Real) or not math.isfinite(bias):
            raise ValueError("Rear gun electrode requires an explicit voltage reference and finite bias in kV")
        if role == "cathode_holder" and (reference != "tip" or bias != 0.):
            raise ValueError("The contacting cathode holder must be electrically tied to the tip")
    return float(start), float(stop), float(bore)/2, float(outer)/2


def validate_rear_gun_document(document):
    """A rear physical extent must never replace the source reference port."""
    if document.get("module", {}).get("type") != "gun":
        return
    if float(document["ports"]["entrance"]["local_z_mm"]) != 0.:
        raise ValueError("The gun entrance reference must stay at tip Z=0; rear hardware uses negative Z")
    parts = document.get("parts", ())
    tip = next((part for part in parts if part["key"] == "feg_tip"), None)
    if tip is not None and (float(tip["local_end_z_mm"]) != 0. or float(tip["optical_reference_local_z_mm"]) != 0.):
        raise ValueError("The FEG tip apex and source reference must remain at Z=0")
    for part in parts:
        dimensions = rear_part_dimensions(part)
        if dimensions is None:
            continue
        if tip is None or dimensions[0] < float(tip["local_start_z_mm"]):
            raise ValueError("The tip shank/back-domain must reach every rear gun component")
        from temsim.optics.electron_gun.tip_assembly import model_from_part
        geometry = model_from_part(tip).geometry
        start, stop, inner, _ = dimensions
        radius = float(geometry.radius_m(start*1e-3))*1000.
        if part["rear_gun_role"] in {"emitter_enclosure_electrode", "insulator"} and radius >= inner:
            raise ValueError("Rear enclosure and insulator bores must clear the tip shank")
        if part["rear_gun_role"] == "cathode_holder" and float(geometry.radius_m(stop*1e-3))*1000. <= inner:
            raise ValueError("Cathode holder must contact the tip shank")


def rear_gun_solids(part):
    dimensions = rear_part_dimensions(part)
    if dimensions is None or part.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing":
        return None
    from temsim.cad_geometry import SolidSpec, annulus
    material = "insulator" if part["rear_gun_role"] == "insulator" else "electrode"
    return (SolidSpec(part["key"], "body", material, annulus(*dimensions), (SOURCE,), "configured_geometry",
        "Axisymmetric rear gun engineering adaptation with tip at Z=0. Undisclosed dimensions and wiring "
        "are explicit model assumptions. Curved-tip conductors enter electrostatics; insulator polarization is omitted.",
        parameters=dict(zip(("start_z_mm", "end_z_mm", "inner_radius_mm", "outer_radius_mm"), dimensions))),)


def capture_rear_parts(assembly):
    rows = []
    for part in assembly.parts:
        dimensions = rear_part_dimensions(part.data)
        if dimensions is None:
            continue
        shift = part.start_z_mm-dimensions[0]
        if not math.isclose(shift, 0., abs_tol=1e-12):
            raise ValueError("Rear gun coordinates must be relative to the unchanged tip origin")
        start, stop, inner, outer = dimensions
        rows.append({"key": part.key, "role": part.data["rear_gun_role"],
                     "start_mm": start, "stop_mm": stop, "inner_mm": inner, "outer_mm": outer,
                     "voltage_reference": part.data.get("rear_voltage_reference"),
                     "bias_kv": part.data.get("rear_bias_kv")})
    return tuple(rows)


def rear_electrode_request(gun, geometry, *, continuous_cathode=False):
    """Resolve exposed conductor boundaries for the selected cathode geometry."""
    from temsim.physics.gun_field_environment import ensure_gun_field_environment
    if getattr(gun, "type_key", None) == "cold_feg":
        ensure_gun_field_environment(gun)
    rows = getattr(gun, "_gun_rear_parts", ())
    ht, ext = gun.accelerator.high_tension_kv*1000., gun.extractor.voltage_kv*1000.
    lens = gun.electrostatic_lens.potential_rise_from_tip_v(ext/1000., ht/1000.)
    base = {"tip": 0., "extractor": ext, "gun_lens": lens, "ground": ht}
    rings, connected, embedded = [], [], []
    for row in rows:
        if row["role"] == "insulator":
            continue
        voltage = base[row["voltage_reference"]]+row["bias_kv"]*1000.
        if geometry is not None:
            start, stop, inner, outer = (row[k]*1e-3 for k in ("start_mm", "stop_mm", "inner_mm", "outer_mm"))
            if start < -geometry.shank_length_um*1e-6-1e-14:
                raise ValueError("Rear gun conductor lies outside the selected curved-tip back domain")
            # The legacy near-flat cap can envelop the entire rear assembly.
            # Same-potential metal wholly inside the cap adds no vacuum boundary.
            # Distinct biases and partially exposed EEE overlaps still fail.
            if continuous_cathode and voltage == 0. and float(geometry.radius_m(stop)) >= outer:
                embedded.append(row["key"])
                if row["role"] == "cathode_holder":
                    connected.append(row["key"])
                continue
            if row["role"] == "emitter_enclosure_electrode" and float(geometry.radius_m(start)) >= inner:
                raise ValueError("EEE bore must clear the physical tip shank")
            if row["role"] == "cathode_holder":
                if voltage != 0. or float(geometry.radius_m(stop)) <= inner:
                    raise ValueError("Cathode holder must contact the tip shank at the same potential")
                connected.append(row["key"])
            rings.append({"key": row["key"], "start_m": start, "stop_m": stop,
                          "inner_m": inner, "outer_m": outer, "potential_rise_v": voltage})
    if geometry is None:
        status = "omitted_by_infinite_planar_cathode"
    elif embedded and not rings:
        status = "embedded_in_continuous_cathode"
    else:
        status = "active_curved_conductors"
    return {"tip_origin_z_mm": 0., "parts": [dict(row) for row in rows], "rings": rings,
            "tip_connected_keys": connected, "embedded_tip_electrode_keys": embedded,
            "field_status": status,
            "insulators": "geometry_and_interception_only_no_dielectric_polarization"}


def annulus_contact_fractions(start_mm, end_mm, row):
    """First contact with a finite annulus, including its two planar faces.

    Synthetic straight chords use millimetres. A path entirely outside the
    outer radius is not absorbed by an invented infinite rear plate.
    """
    p, q = np.asarray(start_mm, float), np.asarray(end_mm, float)
    d = q-p
    z0, z1, ri, ro = (row[k] for k in ("start_mm", "stop_mm", "inner_mm", "outer_mm"))
    times = [np.zeros(len(p))]
    for face in (z0, z1):
        times.append(np.divide(face-p[:, 2], d[:, 2], out=np.full(len(p), np.inf), where=d[:, 2] != 0.))
    a = np.sum(d[:, :2]**2, axis=1)
    b = 2*np.sum(p[:, :2]*d[:, :2], axis=1)
    for radius in (ri, ro):
        c = np.sum(p[:, :2]**2, axis=1)-radius**2
        disc = b*b-4*a*c
        valid = (a > 0.) & (disc >= 0.)
        root = np.sqrt(np.maximum(0., disc))
        # Stable quadratic roots, including grazing and initially-on-wall rays.
        numerator = -.5*(b+np.copysign(root, b))
        times.append(np.divide(numerator, a, out=np.full(len(p), np.inf), where=valid))
        times.append(np.divide(c, numerator, out=np.full(len(p), np.inf), where=valid & (numerator != 0.)))
    t = np.asarray(times).T
    valid = np.isfinite(t) & (t >= 0.) & (t <= 1.)
    points = p[:, None, :]+np.where(valid, t, 0.)[:, :, None]*d[:, None, :]
    r = np.linalg.norm(points[:, :, :2], axis=2)
    tol = 64*np.finfo(float).eps*max(abs(z0), abs(z1), ro)
    valid &= (points[:, :, 2] >= z0-tol) & (points[:, :, 2] <= z1+tol) & (r >= ri-tol) & (r <= ro+tol)
    return np.min(np.where(valid, t, np.inf), axis=1)
