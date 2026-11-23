"""Independent axisymmetric A-phi check of each installed round return circuit."""

import json

import numpy as np
from matplotlib.path import Path
from threadpoolctl import threadpool_limits

from circuit_audit import ADL, GLOBAL_OFFSET_MM, ROUND, TL12, WORK, audit_inputs
from temsim.corrector_field_cli import _save
from temsim.physics.annular_coil_potential import AnnularCoilPotential
from temsim.physics.nonlinear_magnetostatics import prepared_mesh, solve_linear, solve_nonlinear


def subdivide(knots, fine, near, far):
    axis = []
    knots = np.unique(np.round(knots, 10))
    for lo, hi in zip(knots, knots[1:]):
        distance = abs(0.5 * (lo + hi))
        step = fine if distance <= near else min(far, fine + 0.08 * (distance - near))
        axis.extend(np.linspace(lo, hi, int(np.ceil((hi - lo) / step)) + 1)[:-1])
    return np.r_[axis, knots[-1]]


def downstream_sections():
    from temsim.cad_catalog import part_solids
    from temsim.module_manifest import read_document
    from temsim.part_materials import material_for_region
    from temsim.paths import INSTRUMENT_CONFIG_ROOT

    doc = read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    parts = {p["key"]: p for p in doc["parts"]}
    shapes = {}
    for key in (
        "mini_condenser_upper_pole",
        "mini_condenser_lower_pole",
        "mini_condenser_yoke",
        "objective_upper_pole",
        "objective_lower_pole",
        "objective_lens_yoke",
        "objective_pole_sleeves",
        "objective_return_plates",
    ):
        for solid in part_solids(parts[key], parts):
            assign = material_for_region(parts[key], solid.region)
            if assign is None and solid.material != "soft_magnetic":
                raise ValueError(f"Unspecified magnetic state of {key}/{solid.region}")
            if assign and assign["material_key"] != "armco_pure_iron":
                raise ValueError(f"Neighbor material is no longer the specified ARMCO grade on {key}")
            shapes[f"{key}/{solid.region}"] = np.asarray(solid.shape["section_zr_mm"])
    return shapes


def solve_section(recipe, lens, amps, spacing, padding, *, vacuum=False, neighbors=None):
    row = recipe["sources"][lens]
    ((lo, hi),) = row["intervals"]
    z0 = (lo + hi) / 2
    a, b = row["inner"], row["outer"]
    shapes = {
        s["name"]: np.array(s["shape"]["section_zr_mm"]) - [z0, 0]
        for s in recipe["solids"]
        if s["name"].startswith(lens + "_")
    }
    assert len(shapes) == 7
    mats = {s["material"]: s["curve"] for s in recipe["solids"] if s["name"] in shapes}
    assert len(mats) == 1 and None not in mats.values()
    grid_shapes = dict(shapes)
    if neighbors is not None:
        external = {k: v - [z0, 0] for k, v in downstream_sections().items()}
        grid_shapes.update(external)
        if neighbors:
            shapes.update(external)
    vertices = np.concatenate(list(grid_shapes.values()))
    zmax = max(abs(vertices[:, 0])) + padding
    rmax = vertices[:, 1].max() + padding
    r = (
        subdivide([0, a, b, *vertices[:, 1], 1.2, 3, 7, 10, 20, 40, 65, 85, 110, 145, rmax], spacing, 40, 9)
        * 0.001
    )
    z_half = subdivide(
        [0, hi - z0, *abs(vertices[:, 0]), 16, 21.2, 30, 46.2, 60, 90, 125, zmax], spacing, 30, 9
    )
    z = np.r_[-z_half[:0:-1], z_half] * 0.001
    mesh = prepared_mesh(tuple(r), tuple(z))
    rz = mesh.centres[:, ::-1] * 1000
    masks = {name: Path(coords).contains_points(rz) for name, coords in shapes.items()}
    occupied = np.sum(list(masks.values()), axis=0)
    if occupied.max() != 1 or not all(m.any() for m in masks.values()):
        raise ValueError("Overlapping or unresolved magnetic sections")
    in_coil = (abs(rz[:, 0]) < (hi - lo) / 2) & (rz[:, 1] > a) & (rz[:, 1] < b)
    assert not np.any(in_coil & (occupied != 0))
    area = (b - a) * (hi - lo) * 1e-6
    vertex_r = r[mesh.triangles // len(z)]
    triangle_area = mesh.weights.sum(axis=1) / vertex_r.mean(axis=1)
    integrated_area = triangle_area[in_coil].sum()
    np.testing.assert_allclose(integrated_area, area, rtol=2e-10)
    ni = amps * row["turns"]
    density = in_coil * (ni / integrated_area)
    if vacuum:
        field = solve_linear(r, z, lambda rr, zz: 1.0, lambda rr, zz: density)
    else:
        field = solve_nonlinear(
            r,
            z,
            lambda rr, zz: occupied - 1,
            lambda rr, zz: density,
            list(mats.values()),
            relative_tolerance=2e-7,
            max_iterations=150,
        )
    exact = AnnularCoilPotential(lens, a * 0.001, b * 0.001, (hi - lo) * 0.001, ni, quadrature_order=12)
    bare = exact.field_at_global_positions_t(np.column_stack((0 * z, 0 * z, z)))[:, 2]
    # The material checks use the element curl, not a nodal derivative across a material jump.
    flux = np.linalg.norm(
        np.einsum("tqci,ti->tqc", mesh.curl, field.a_phi_tm.ravel()[mesh.triangles]), axis=2
    )
    peaks = {k: float(flux[m].max()) for k, m in masks.items()}
    name = f"{lens}-{amps:g}-{spacing:g}-{padding:g}" + ("-vacuum" if vacuum else "")
    if neighbors is not None:
        name += "-neighbors" if neighbors else "-neighbor-control"
    _save(
        WORK / "sections" / (name + ".npz"),
        r_mm=r * 1000,
        z_mm=z * 1000 + z0 + GLOBAL_OFFSET_MM,
        bz_t=field.bz_t[0],
        vacuum_t=bare,
        poloidal_flux_wb=2 * np.pi * r[:, None] * field.a_phi_tm,
        section_zr_mm=json.dumps({k: (v + [z0 + GLOBAL_OFFSET_MM, 0]).tolist() for k, v in shapes.items()}),
        receipt=json.dumps(
            dict(
                lens=lens,
                amps=amps,
                turns=row["turns"],
                ampere_turns=ni,
                central_spacing_mm=spacing,
                padding_mm=padding,
                nodes=mesh.size,
                triangles=len(mesh.triangles),
                residual=field.relative_residual,
                iterations=field.iterations,
                peaks_t=peaks,
                current_area_m2=area,
                z_center_mm=z0 + GLOBAL_OFFSET_MM,
                passive_neighbors=neighbors,
            )
        ),
    )
    print(
        name,
        "peak field T",
        float(max(abs(field.bz_t[0]))),
        "residual",
        field.relative_residual,
        flush=True,
    )


if __name__ == "__main__":
    recipe, _, _ = audit_inputs()
    with threadpool_limits(limits=2):
        for lens, amps, grid, padding, vacuum in [
            (TL12, 10.0, 1.15, 170.0, False),
            (TL12, 10.0, 0.58, 170.0, False),
            (TL12, 10.0, 0.58, 270.0, False),
            (TL12, 10.0, 0.58, 270.0, True),
            (TL12, 1.0, 0.82, 170.0, False),
            (TL12, 30.0, 0.82, 170.0, False),
            (ADL, 10.0, 0.82, 170.0, False),
            (ROUND[1], -145.0, 0.82, 170.0, False),
            (ROUND[1], -143.55, 0.82, 170.0, False),
            (ROUND[1], -10.0, 0.82, 170.0, False),
            (ROUND[2], 330.0, 0.82, 170.0, False),
            (ROUND[2], 333.3, 0.82, 170.0, False),
            (ROUND[2], 10.0, 0.82, 170.0, False),
        ]:
            solve_section(recipe, lens, amps, grid, padding, vacuum=vacuum)
        for neighbors in (False, True):
            solve_section(recipe, TL12, 10.0, 0.82, 170.0, neighbors=neighbors)
