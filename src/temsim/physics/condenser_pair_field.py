"""Joint nonlinear condenser poles and returns with open-bore field recovery (SI).

The imposed windings are tracked analytically. Only the induced field recovered
from the converged magnetization is added to those windings during transport.
"""

import hashlib
import json

import numpy as np
from numba import njit

from temsim.magnetic_materials import BHCurve, MU0
from temsim.part_materials import is_magnetostatic_body, material_catalog
from temsim.physics.axisymmetric_magnetostatics import _part_mask
from temsim.physics.lens_field_provider import lens_geometry_binding
from temsim.physics.nonlinear_magnetostatics import prepared_mesh, solve_nonlinear


def installed_pair(state):
    """Resolve the installed parts once even when a cartridge is shared."""
    parts = {}
    for key in ("condenser_lens_1", "condenser_lens_2"):
        geometry = json.loads(lens_geometry_binding(state, key).canonical_geometry_json)
        for part in geometry["lens_assembly"]["parts"]:
            if is_magnetostatic_body(part["data"]):
                parts[part["key"]] = part
    curves = []
    grades = {}
    armco = next(row["bh_material"] for row in material_catalog()
                 if row["material_key"] == "armco_pure_iron")
    for name, part in parts.items():
        data = part["data"]
        assignment = data.get("material_regions", {}).get("body", {}).get("bh_material")
        if assignment is None and data.get("mechanical_profile") != "magnetic_lens_yoke":
            raise ValueError(f"No B-H assignment for {name}")
        curve = armco if assignment is None else assignment
        if curve not in curves:
            curves.append(curve)
        grades[name] = dict(index=curves.index(curve), assumed=assignment is None)
    if len(parts) != 4:
        raise ValueError("Review the changed condenser circuit inventory before using this study")
    return parts, curves, grades


def _axis(edges, maximum_step):
    edges = np.unique(np.round(edges, 10))
    pieces = [np.linspace(a, b, max(1, int(np.ceil((b-a)/maximum_step((a+b)/2))))+1)[:-1]
              for a, b in zip(edges[:-1], edges[1:])]
    return np.r_[np.concatenate(pieces), edges[-1]] * 1e-3


@njit(cache=True)
def induced_axis(at, radius, position, weighted_mr, weighted_mz):
    """Azimuthally integrated vacuum dipole field of an axisymmetric magnetization."""
    result = np.zeros(len(at))
    for j, z in enumerate(at):
        value = 0.0
        for i in range(len(radius)):
            r, d = radius[i], z - position[i]
            d2 = r*r + d*d
            value += ((2*d*d-r*r)*weighted_mz[i] - 3*r*d*weighted_mr[i]) / (d2*d2*np.sqrt(d2))
        result[j] = MU0 / 2 * value
    return result


def solve_pair(state, windings, currents, *, pitch_mm=2.0, domain_mm=550.0):
    """Solve both signed condenser drives together in the incident C3 winding field."""
    if len(currents) != len(windings) or not len(windings):
        raise ValueError("Each participating winding requires its own signed series current")
    parts, curves, grades = installed_pair(state)
    rad = [0, 2, 6, 9.6, 20, 30, 81, 85, 100, 140, 225, domain_mm]
    axial = [450-domain_mm, 260, 360, 430, 450, 475, 490, 521, 540, 550, 560,
             582.8, 620, 750, 775, 820, 1000, 1030, 1100, 1140+domain_mm]
    for part in parts.values():
        data = part["data"]
        axial.extend([part["start_z_mm"], part["end_z_mm"]])
        rad.extend(v/2 for k, v in data.items() if k in {
            "mechanical_inner_diameter_mm", "mechanical_bore_diameter_mm",
            "mechanical_outer_diameter_mm", "mechanical_tip_diameter_mm"})
    for w in windings:
        rad.extend([1000*w.inner_radius_m, 1000*w.outer_radius_m])
        axial.extend([1000*(w.center_z_m-w.length_m/2), 1000*(w.center_z_m+w.length_m/2)])
    r = _axis(rad, lambda x: pitch_mm if x <= 85 else max(pitch_mm*2, x*.13))
    z = _axis(axial, lambda x: pitch_mm if 430<x<775 else
              (3*pitch_mm if 775<=x<1030 else max(7, .15*min(abs(x-450), abs(x-1000)))))

    def material(radius, position):
        labels = np.full(radius.shape, -1)
        for name, part in parts.items():
            mask = _part_mask(part, radius, position)
            if np.any(mask & (labels >= 0)):
                raise ValueError("Magnetic condenser regions overlap")
            labels[mask] = grades[name]["index"]
        return labels

    def density(radius, position):
        current = np.zeros(radius.shape)
        for w, amps in zip(windings, currents, strict=True):
            inside = ((radius >= w.inner_radius_m) & (radius <= w.outer_radius_m) &
                      (abs(position-w.center_z_m) <= w.length_m/2))
            current[inside] += w.ampere_turns * amps / (w.length_m*(w.outer_radius_m-w.inner_radius_m))
        return current

    solution = solve_nonlinear(r, z, material, density, curves, relative_tolerance=2e-7)
    mesh = prepared_mesh(tuple(r), tuple(z))
    labels = material(*mesh.centres.T)
    sel = labels >= 0
    tri = mesh.triangles[sel]
    B = np.einsum("tqci,ti->tqc", mesh.curl[sel], solution.a_phi_tm.ravel()[tri])
    norm = np.linalg.norm(B, axis=2)
    M = np.zeros_like(B)
    for i, row in enumerate(curves):
        mask = labels[sel] == i
        H, _ = BHCurve(row).evaluate(norm[mask])
        nu = np.divide(H, norm[mask], out=np.zeros_like(H), where=norm[mask]>0)
        M[mask] = B[mask] * (1/MU0 - nu)[:, :, None]
    rw = mesh.weights[sel]
    R, Z = np.meshgrid(r, z, indexing="ij")
    radial, position = (arr.ravel()[tri] @ mesh.shapes.T for arr in (R, Z))
    knots = np.linspace(-0.004, 1.604, 3217)
    induced = induced_axis(knots, radial.ravel(), position.ravel(),
                           (rw*M[:, :, 0]).ravel(), (rw*M[:, :, 1]).ravel())
    identity = hashlib.sha256(json.dumps(dict(parts=parts, grades=grades, curves=curves),
                                           sort_keys=True).encode()).hexdigest()
    return dict(z=knots, induced_T=induced, fem_z=z, fem_bore_T=solution.bz_t[0],
                relative_residual=solution.relative_residual, peak_material_T=solution.peak_material_t,
                iterations=solution.iterations, nodes=mesh.size, pitch_mm=pitch_mm,
                domain_mm=domain_mm, currents=np.array(currents), geometry_id=identity,
                parts_json=json.dumps({name: dict(z_mm=[p["start_z_mm"],p["end_z_mm"]],
                                      **grades[name]) for name,p in parts.items()}),
                curve_digests=np.array([m["source_sha256"] for m in curves]))
