"""Magnetostatic FEM for the two CESCOR cartridges and their transfer doublet.

The impressed field is the Biot-Savart field of closed windings. The reduced
scalar potential has a continuous nodal P1 basis on a conforming tetrahedral
mesh. All magnetic bodies share one nonlinear solve. Coordinates in the CAD
are module-local mm; the numerical equations and field samples use SI.
"""

from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
from scipy.constants import mu_0
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve

from temsim.cad_catalog import part_solids
from temsim.part_materials import material_catalog, material_for_region
from temsim.physics.annular_coil_potential import AnnularCoilPotential

HP = ("probe_hp2_hexapole", "probe_hp1_hexapole")
TL = ("probe_tl22_lens", "probe_tl21_lens")


def _occ_solid(cad, shape):
    kind = shape["kind"]
    if kind == "revolve":
        coords = []
        for z, r in shape["section_zr_mm"]:
            if not coords or coords[-1] != (r, 0.0, z):
                coords.append((r, 0.0, z))
        if coords[-1] == coords[0]:
            coords.pop()
        pts = [cad.addPoint(*p) for p in coords]
        wire = cad.addWire([cad.addLine(a, b) for a, b in zip(pts, pts[1:] + pts[:1])])
        surf = cad.addPlaneSurface([wire])
        entities = cad.revolve([(2, surf)], 0, 0, 0, 0, 0, 1, 2 * np.pi)
        return next(tag for dim, tag in entities if dim == 3)
    if kind == "box":
        size = np.asarray(shape["size_mm"])
        origin = np.asarray(shape["center_mm"]) - size / 2
        return cad.addBox(*origin, *size)
    if kind == "transform":
        rotation = np.array(shape["rotation"])
        if not np.allclose(rotation[2], [0, 0, 1]) or not np.allclose(rotation[:, 2], [0, 0, 1]):
            raise ValueError("Corrector FEM supports native coaxial cartridges and axial winding rotations")
        tag = _occ_solid(cad, shape["shape"])
        cad.rotate([(3, tag)], 0, 0, 0, 0, 0, 1, np.arctan2(rotation[1, 0], rotation[0, 0]))
        cad.translate([(3, tag)], *shape["offset_mm"])
        return tag
    if kind == "difference":
        base = _occ_solid(cad, shape["base"])
        tools = [(3, _occ_solid(cad, t)) for t in shape["tools"]]
        result, _ = cad.cut([(3, base)], tools)
    elif kind == "union":
        base, *tools = [(3, _occ_solid(cad, t)) for t in shape["shapes"]]
        result, _ = cad.fuse([base], tools)
    else:
        raise ValueError(f"Unsupported corrector material CAD primitive {kind!r}")
    (volume,) = [tag for dim, tag in result if dim == 3]
    return volume


def circuit(document):
    """Capture the material selection of every real region and the fixed coil geometry."""
    from temsim.multipole_winding import multipole_winding_definition
    from temsim.cad_round_winding import round_winding_envelope
    from temsim.lens_pose import effective_part_transform_mm, mechanical_part_transform_mm

    parts = {row["key"]: row for row in document["parts"]}
    fallback = next(row for row in material_catalog() if row["material_key"] == "armco_pure_iron")
    solids, sources = [], {}
    for key in (
        *HP,
        *[
            f"{t}_{s}"
            for t in TL
            for s in (
                "upper_pole",
                "lower_pole",
                "pole_sleeves",
                "return_plates",
                "yoke",
                "excitation_coil",
            )
        ],
    ):
        row = parts[key]
        model = row.get("model_3d", {})
        transform = model.get("transform", {})
        placements = (mechanical_part_transform_mm(row, parts), effective_part_transform_mm(row, parts))
        if (
            model.get("base", {}).get("kind", "existing") != "existing"
            or model.get("features")
            or not np.array_equal(transform.get("scale_xy", [1, 1]), [1, 1])
            or np.any(transform.get("offset_mm", [0, 0, 0]))
            or np.any(transform.get("rotation_deg", [0, 0, 0]))
            or any(
                not np.allclose(rotation, np.eye(3), atol=1e-14, rtol=0)
                or not np.allclose(translation, 0.0, atol=1e-12)
                for rotation, translation in placements
            )
        ):
            raise ValueError(
                f"{key}: this solve requires centered installed geometry and native material shapes"
            )
        if key.endswith("_excitation_coil"):
            e = round_winding_envelope(row, parts)
            sources[row["parent_key"]] = dict(
                kind="round",
                turns=row["winding_turn_count"],
                intervals=e["intervals_mm"],
                inner=e["inner_diameter_mm"] / 2,
                outer=e["outer_diameter_mm"] / 2,
            )
            continue
        if key in HP:
            definition = multipole_winding_definition(row, parts)
            sources[key] = dict(
                kind="hexapole",
                turns=row["hexapole_winding_turn_count_per_coil"],
                coils=definition["coils"],
            )
        for solid in part_solids(row, parts):
            if solid.material != "soft_magnetic":
                continue
            assign = material_for_region(row, solid.region) or fallback
            solids.append(
                dict(
                    name=f"{key}/{solid.region}",
                    shape=solid.shape,
                    material=assign["material_key"],
                    curve=assign.get("bh_material"),
                )
            )
    return dict(solids=solids, sources=sources)


def mesh_circuit(recipe, *, spacing_mm=5.0, padding_mm=170.0):
    try:
        import gmsh
    except ImportError as error:
        raise RuntimeError(
            "Install the corrector-fem optional dependency to mesh a probe corrector"
        ) from error
    if not (0 < spacing_mm <= 12 and np.isfinite(padding_mm) and padding_mm >= 85):
        raise ValueError(
            "Use a material mesh size in (0, 12] mm and finite exterior padding of at least 85 mm"
        )
    if gmsh.isInitialized():
        raise RuntimeError("Run the corrector mesher in its own process")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 4)
        gmsh.model.add("probe_corrector")
        cad = gmsh.model.occ
        bodies = [_occ_solid(cad, entry["shape"]) for entry in recipe["solids"]]
        boxes = np.array([cad.getBoundingBox(3, tag) for tag in bodies])
        low, high = boxes[:, :3].min(axis=0) - padding_mm, boxes[:, 3:].max(axis=0) + padding_mm
        outer = cad.addBox(*low, *(high - low))
        _, mapping = cad.fragment([(3, outer)], [(3, tag) for tag in bodies])
        cad.synchronize()
        material = {}
        for entry, out in zip(recipe["solids"], mapping[1:]):
            (tag,) = [t for dim, t in out if dim == 3]
            if tag in material:
                raise ValueError(f"Overlapping material volumes at {entry['name']}")
            material[tag] = entry["name"]
            points = gmsh.model.getBoundary([(3, tag)], recursive=True, oriented=False)
            gmsh.model.mesh.setSize(points, spacing_mm * (0.55 if "pole_pin_" in entry["name"] else 1))
        gmsh.option.setNumber("Mesh.MeshSizeMax", min(45.0, padding_mm / 2))
        gmsh.option.setNumber("Mesh.MinimumCircleNodes", 72)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 42)
        # Resolve the connected beam vacuum, pole noses and fringes even
        # where CAD supplies no vertices on the electron axis.
        zlo, zhi = boxes[:, 2].min() - 15.0, boxes[:, 5].max() + 15.0
        core = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            core, "F", f"{spacing_mm/7}+0.36*Sqrt(x*x+y*y)+0.30*(Max(0,{zlo}-z)+Max(0,z-{zhi}))"
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(core)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.model.mesh.generate(3)
        ids, coord, _ = gmsh.model.mesh.getNodes()
        xyz = coord.reshape(-1, 3)
        order = np.argsort(ids)
        cells, regions = [], []
        for _, tag in gmsh.model.getEntities(3):
            types, _, node_tags = gmsh.model.mesh.getElements(3, tag)
            if list(types) != [4]:
                raise ValueError("Expected a positive tetrahedral mesh in every material volume")
            conn = order[np.searchsorted(ids[order], node_tags[0])].reshape(-1, 4)
            cells.append(conn)
            regions.extend([material.get(tag, "air")] * len(conn))
        tet = np.concatenate(cells)
        used, tet = np.unique(tet, return_inverse=True)
        xyz = xyz[used]
        boundary = np.flatnonzero(np.any((abs(xyz - low) < 1e-6) | (abs(xyz - high) < 1e-6), axis=1))
        if not len(boundary) or set(regions) != set(material.values()) | {"air"}:
            raise ValueError("Incomplete magnetic mesh or missing far boundary")
        return xyz * 0.001, tet.reshape(-1, 4).astype(np.int32), np.array(regions), boundary
    finally:
        gmsh.finalize()


def impressed_h(recipe, points, currents):
    """Field strength in A/m; currents are signed amperes per closed physical winding."""
    from temsim.physics.transverse_coil_field import _unit_nodes

    result = np.zeros_like(points, dtype=float)
    for key, coil in recipe["sources"].items():
        amps = currents[key]
        if not np.isfinite(amps):
            raise ValueError("All four signed physical coil currents must be finite")
        if not amps:
            continue
        if coil["kind"] == "round":
            ((a, b),) = coil["intervals"]
            source = AnnularCoilPotential(
                key,
                coil["inner"] * 0.001,
                coil["outer"] * 0.001,
                (b - a) * 0.001,
                amps * coil["turns"],
                (a + b) * 0.0005,
                quadrature_order=6,
            )
            result += source.field_at_global_positions_t(points) / mu_0
        else:
            nodes, elements, *_ = _unit_nodes(
                json.dumps(coil["coils"], sort_keys=True), coil["turns"], 2, 8
            )
            for start in range(0, len(points), 48):
                r = points[start : start + 48, None, :] - nodes
                result[start : start + 48] += (
                    amps
                    / mu_0
                    * np.einsum("ijk,ij->ik", np.cross(elements, r), np.sum(r * r, axis=-1) ** -1.5)
                )
    return result


def _linear_step(matrix, rhs, backend):
    if backend == "cpu":
        return spsolve(matrix.tocsc(), rhs, permc_spec="MMD_AT_PLUS_A")
    if backend != "cuda":
        raise ValueError("Unknown corrector FEM linear solver")
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    from nvmath.sparse.advanced import DirectSolver, DirectSolverMatrixType, ExecutionCUDA

    device = csr_matrix(matrix.sorted_indices())
    options = {"blocking": True, "sparse_system_type": DirectSolverMatrixType.GENERAL}
    with DirectSolver(device, cp.asarray(rhs), options=options, execution=ExecutionCUDA()) as factor:
        factor.plan()
        factor.factorization_config.pivot_eps = 0.0
        info = factor.factorize()
        if info.info:
            raise ValueError(f"Corrector FEM sparse solve failed ({info.info})")
        return cp.asnumpy(factor.solve())


def _conforming_nodes(xyz, tet, boundary):
    """Eliminate zero-measure edge splits while retaining continuous P1 traces."""
    from itertools import combinations

    x = xyz[tet]
    edges = x[:, 1:] - x[:, :1]
    length = np.max(abs(edges), axis=(1, 2))
    good = abs(np.linalg.det(edges)) > 1e-11 * length**3
    if np.all(good):
        return good, None
    ties = {}
    for cell in tet[~good]:
        for a, b in combinations(cell, 2):
            line = xyz[b] - xyz[a]
            den = line @ line
            if not den:
                continue
            for c in set(cell) - {a, b}:
                t = (xyz[c] - xyz[a]) @ line / den
                if 0 < t < 1 and np.linalg.norm(xyz[c] - xyz[a] - t * line) < 1e-9 * np.sqrt(den):
                    link = (b, a, 1 - t) if a > b else (a, b, t)
                    if c in ties and (ties[c][:2] != link[:2] or abs(ties[c][2] - link[2]) > 1e-10):
                        raise ValueError("Inconsistent constraints on a degenerate mesh edge; remesh")
                    ties[c] = link
                    break
            else:
                continue
            break
        else:
            raise ValueError("Degenerate tetrahedron without a conforming edge relation; remesh")
    free = np.ones(len(xyz), bool)
    free[boundary] = False
    free[list(ties)] = False
    p = eye(len(xyz), format="lil")
    for c, (a, b, t) in ties.items():
        if a in ties or b in ties or c in boundary:
            raise ValueError("Unsupported chained or boundary edge split; remesh")
        p.rows[c], p.data[c] = [a, b], [1 - t, t]
    return good, p.tocsr()[:, free]


def solve(
    recipe,
    mesh,
    currents,
    *,
    tolerance=2e-7,
    max_iterations=100,
    backend="cpu",
    progress=None,
    air_solution=None,
    potential_seed_a=None,
    potential_checkpoint=None,
):
    """Return SI dipole moments of all material tetrahedra and convergence evidence."""
    from threadpoolctl import threadpool_limits

    xyz, tet, labels, boundary = mesh
    if set(currents) != set(recipe["sources"]):
        raise ValueError("Specify HP2, TL22, TL21 and HP1 coil currents separately")
    if not all(np.isfinite(i) for i in currents.values()):
        raise ValueError("Winding currents must be finite signed amperes")
    good, prolong = _conforming_nodes(xyz, tet, boundary)
    tet, labels = tet[good], labels[good]
    corners = xyz[tet]
    edges = corners[:, 1:] - corners[:, :1]
    volume = abs(np.linalg.det(edges)) / 6
    grad3 = np.linalg.inv(edges).transpose(0, 2, 1)
    grad = np.concatenate((-grad3.sum(axis=1, keepdims=True), grad3), axis=1)
    active = labels != "air"
    ids = np.flatnonzero(active)
    centres = corners[active].mean(axis=1)
    h0 = impressed_h(recipe, centres, currents)
    curves, domains = {}, {}
    for row in recipe["solids"]:
        if row["curve"]:
            curves[row["material"]] = row["curve"]
            domains.setdefault(row["material"], []).append(row["name"])
    masks = {k: np.isin(labels[active], names) for k, names in domains.items()}
    free = np.ones(len(xyz), bool)
    free[boundary] = False
    u = np.zeros(len(xyz))
    ii, jj = np.broadcast_arrays(tet[:, :, None], tet[:, None, :])
    dot = np.einsum("eik,ejk->eij", grad, grad)

    def system(u, matrix=False):
        gu = np.einsum("ei,eij->ej", u[tet], grad)
        energy = 0.5 * np.sum(volume * np.einsum("ij,ij->i", gu, gu))
        h = h0 - gu[active]
        norm = np.linalg.norm(h, axis=1)
        sec, tangent = np.ones(len(h)), np.ones(len(h))
        for key, m in masks.items():
            curve = curves[key]
            hv, bv = np.asarray(curve["h_a_per_m"]), np.asarray(curve["b_t"]) / mu_0
            ix = np.clip(np.searchsorted(hv, norm[m], side="right") - 1, 0, len(hv) - 2)
            slope = np.diff(bv)[ix] / np.diff(hv)[ix]
            delta_h = norm[m] - hv[ix]
            induction = bv[ix] + slope * delta_h
            # Integral of B(H)/mu0 - H. Its derivative is exactly M(H).
            primitive = np.r_[0.0, np.cumsum(0.5 * (bv[:-1] + bv[1:]) * np.diff(hv))]
            energy += np.sum(
                volume[ids[m]] * (primitive[ix] + 0.5 * (induction + bv[ix]) * delta_h - 0.5 * norm[m] ** 2)
            )
            sec[m] = np.divide(induction, norm[m], out=slope.copy(), where=norm[m] > 0)
            tangent[m] = slope
        mag = (sec[:, None] - 1) * h
        gu[active] -= mag
        residual = np.bincount(
            tet.ravel(), (volume[:, None] * np.einsum("eij,ej->ei", grad, gu)).ravel(), minlength=len(u)
        )
        if not matrix:
            return restrict(residual), mag, norm, sec, energy
        local = dot.copy()
        hn = h / np.maximum(norm[:, None], 1e-40)
        proj = np.einsum("eij,ej->ei", grad[active], hn)
        local[ids] = (
            sec[:, None, None] * local[ids]
            + (tangent - sec)[:, None, None] * proj[:, :, None] * proj[:, None, :]
        )
        stiffness = coo_matrix(
            ((volume[:, None, None] * local).ravel(), (ii.ravel(), jj.ravel())), shape=(len(u), len(u))
        ).tocsr()
        reduced = stiffness[free][:, free] if prolong is None else (prolong.T @ stiffness @ prolong).tocsr()
        return restrict(residual), reduced, energy

    def restrict(vec):
        return vec[free] if prolong is None else prolong.T @ vec

    def increment(vec):
        if prolong is not None:
            return prolong @ vec
        full = np.zeros_like(u)
        full[free] = vec
        return full

    with threadpool_limits(limits=2):
        scale = max(np.linalg.norm(system(u)[0]), 1e-24)
        independent_count = np.count_nonzero(free) if prolong is None else prolong.shape[1]
        unknowns = np.zeros(independent_count)
        # Start from the exact low-field linear solution. Starting from H=H0
        # in high-permeability bodies can put nearly every cell into saturation.
        if potential_seed_a is None:
            target_h = h0.copy()
            h0[:] = 0
            _, linear, _ = system(u, True)
            lowfield_m = np.zeros_like(h0)
            for key, m in masks.items():
                c = curves[key]
                chi = c["b_t"][1] / (mu_0 * c["h_a_per_m"][1]) - 1.0
                lowfield_m[m] = chi * target_h[m]
            rhs = np.bincount(
                tet[active].ravel(),
                (volume[ids, None] * np.einsum("eij,ej->ei", grad[active], lowfield_m)).ravel(),
                minlength=len(u),
            )
            h0[:] = target_h
            if restrict(rhs).any():
                unknowns = _linear_step(linear, restrict(rhs), backend)
                u = increment(unknowns)
        else:
            # Store free degrees of freedom. Reconstruct constrained edge nodes and the
            # exterior Dirichlet boundary exactly as for a cold solve.
            seed = np.asarray(potential_seed_a, dtype=float)
            if seed.shape != unknowns.shape or not np.isfinite(seed).all():
                raise ValueError("Invalid magnetic scalar potential continuation seed")
            unknowns = seed.copy()
            u = increment(unknowns)
        for iteration in range(max_iterations + 1):
            residual, mat, energy = system(u, True)
            value = np.linalg.norm(residual) / scale
            if progress:
                progress(iteration, value)
            if value < tolerance:
                break
            update = _linear_step(mat, -residual, backend)
            descent = float(np.dot(residual, update))
            for step in 0.5 ** np.arange(18):
                trial = u + step * increment(update)
                trial_r, *_, trial_energy = system(trial)
                if trial_energy <= energy + 1e-4 * step * descent or (
                    abs(trial_energy - energy) <= 2e-13 * max(abs(energy), 1e-30)
                    and np.linalg.norm(trial_r) < np.linalg.norm(residual)
                ):
                    u = trial
                    unknowns += step * update
                    break
            else:
                raise ValueError(f"Corrector FEM Newton line search stalled at {value:g}")
        else:
            raise ValueError("Corrector FEM did not converge")
    _, mag, norm, sec, _ = system(u)
    peaks = {}
    for key, m in masks.items():
        peak = float(norm[m].max())
        peaks[key] = dict(h_a_per_m=peak, b_t=float((mu_0 * sec[m] * norm[m]).max()))
        if peak > curves[key]["h_a_per_m"][-1] * (1 + 1e-9):
            raise ValueError(f"{key}: {peak:g} A/m exceeds the sourced B-H curve")
    receipt = dict(
        currents_A=currents,
        turns={k: v["turns"] for k, v in recipe["sources"].items()},
        material_peaks=peaks,
        residual=float(value),
        iterations=iteration,
        nodes=len(xyz),
        tetrahedra=len(tet),
        zero_measure_edge_splits=int(np.count_nonzero(~good)),
        volumes_mm3={str(k): float(volume[labels == k].sum() * 1e9) for k in np.unique(labels)},
        circuit_sha256=sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
    )
    if air_solution is not None:
        keep = ~active
        nodes, conn = np.unique(tet[keep], return_inverse=True)
        air_solution.update(
            air_xyz_m=xyz[nodes], air_tet=conn.reshape(-1, 4).astype(np.int32), air_u_A=u[nodes]
        )
    if potential_checkpoint is not None:
        potential_checkpoint["free_potential_a"] = unknowns.copy()
    return corners[active], mag * volume[active, None], receipt
