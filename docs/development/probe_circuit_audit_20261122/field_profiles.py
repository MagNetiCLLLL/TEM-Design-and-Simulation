"""Measure axial DC fields and individual excitations on the same assembled 3-D mesh."""

import argparse
import json

import numpy as np
from scipy.constants import mu_0

from circuit_audit import ADL, GLOBAL_OFFSET_MM, LOWER_MM, TL12, UPPER_MM, WORK, audit_inputs
from temsim.corrector_field_cli import _save, cached_mesh
from temsim.physics.corrector_vacuum_field import AirPotential, _coil_b, harmonic_table, vacuum_field
from temsim.physics.probe_corrector_fem import _conforming_nodes, solve


def load_solution(path):
    with np.load(path) as f:
        return dict(f)


def difference(after, before, step):
    for k in ("recipe", "corners_m", "air_xyz_m", "air_tet"):
        np.testing.assert_array_equal(after[k], before[k])
    ia, ib = (json.loads(str(x["report"]))["currents_A"] for x in (after, before))
    return {
        **after,
        "moments_Am2": (after["moments_Am2"] - before["moments_Am2"]) / step,
        "air_u_A": (after["air_u_A"] - before["air_u_A"]) / step,
        "report": json.dumps(dict(currents_A={k: (ia[k] - ib[k]) / step for k in ia})),
    }


def complex_sextupole(field, xy, radius):
    mid = len(xy) // 2
    j = int(round(radius / (xy[1] - xy[0])))
    np.testing.assert_allclose(xy[mid + j], radius, atol=1e-14)
    v = (
        field[:, mid, mid + j, :2]
        + field[:, mid, mid - j, :2]
        - field[:, mid + j, mid, :2]
        - field[:, mid - j, mid, :2]
    ) / (4 * radius**2)
    return v[:, 0] + 1j * v[:, 1]


def sample_delta(solution, target, receipt):
    z, xy, field = harmonic_table(solution, 0.0005, grid_half=6, angles=36)
    z = z * 1000 + GLOBAL_OFFSET_MM
    # Both ends are within the native air-potential region for these meshes.
    # The solver's exterior volume-moment continuation is a separate diagnostic.
    keep = (z >= 995) & (z <= 1460)
    mid = len(xy) // 2
    check_z = np.array(
        [
            1088.2,
            1144.50,
            1166.35,
            1249.95,
            1300.85,
            1317.2,
            1359.3,
            1366.3,
            1373.3,
            1379.0,
            UPPER_MM,
            LOWER_MM,
        ]
    )
    pts = np.column_stack((0 * check_z, 0 * check_z, (check_z - GLOBAL_OFFSET_MM) * 0.001))
    independent = vacuum_field(pts, solution, quadrature=4)[:, 2]
    native = (_coil_b(pts, solution) - mu_0 * AirPotential(solution).evaluate(pts, gradient=True))[:, 2]
    _save(
        target,
        z_mm=z[keep],
        bz_t=field[keep, mid, mid, 2],
        s3_t_m2=complex_sextupole(field[keep], xy, 0.0012),
        s3_inner_t_m2=complex_sextupole(field[keep], xy, 0.0006),
        check_z_mm=check_z,
        material_volume_check_bz_t=independent,
        native_air_bz_t=native,
        receipt=json.dumps(receipt),
    )


def induction_by_region(solution, mesh, recipe):
    xyz, tet, labels, far = mesh
    good, _ = _conforming_nodes(xyz, tet, far)
    keep = good & (labels != "air")
    corners = solution["corners_m"]
    np.testing.assert_array_equal(corners, xyz[tet[keep]])
    volume = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6
    polarization = mu_0 * np.linalg.norm(solution["moments_Am2"], axis=1) / volume
    data, grade_peak = {}, {}
    for row in recipe["solids"]:
        mat = row["material"]
        c = row["curve"]
        j = np.asarray(c["b_t"]) - mu_0 * np.asarray(c["h_a_per_m"])
        # This inversion is valid only for a strictly increasing J(H).
        assert (np.diff(j) > 0).all()
        sel = labels[keep] == row["name"]
        assert polarization[sel].max() <= j[-1] + 1e-10
        b = np.interp(polarization[sel], j, c["b_t"])
        order = np.argsort(b)
        cumulative = np.cumsum(volume[sel][order]) / volume[sel].sum()
        data[row["name"]] = dict(
            material=mat,
            max_T=float(b.max()),
            volume_p95_T=float(np.interp(0.95, cumulative, b[order])),
            volume_fraction_over_1p8_T=float(volume[sel][b > 1.8].sum() / volume[sel].sum()),
        )
        grade_peak[mat] = max(grade_peak.get(mat, 0), b.max())
    actual = json.loads(str(solution["report"]))["material_peaks"]
    for k, v in grade_peak.items():
        np.testing.assert_allclose(v, actual[k]["b_t"], rtol=1e-9)
    return data


def run(spacing, padding, all_channels, *, suffix="", nonlinear_solve=solve, channels=None):
    recipe, bias, _ = audit_inputs()
    mid, mesh = cached_mesh(recipe, spacing, padding, WORK / "meshes")
    folder = WORK / f"profiles-{spacing:g}-{padding:g}{suffix}"

    def get(case):
        s = load_solution(WORK / f"{spacing:g}-{padding:g}-{case}{suffix}.npz")
        if json.loads(str(s["report"]))["mesh_id"] != mid:
            raise ValueError("Incompatible mesh")
        return s

    both = get("both")
    report = json.loads(str(both["report"]))

    def save(key, after, before=None, step=1.0):
        ra = json.loads(str(after["report"]))
        rb = json.loads(str(before["report"])) if before is not None else None
        de = dict(
            key=key,
            current_step_A=step,
            after=ra["currents_A"],
            before=rb["currents_A"] if rb else None,
            residuals=[ra["residual"]] + ([rb["residual"]] if rb else []),
            mesh_id=mid,
        )
        path = folder / (key + ".npz")
        if path.is_file():
            with np.load(path) as old:
                if json.loads(str(old["receipt"])) == de:
                    return
        sample_delta(difference(after, before, step) if before is not None else after, path, de)
        print("Sampled", key, flush=True)

    if channels is None:
        adl = get("adl")
        save("total", both)
        save(TL12, both, adl, 10.0)
        save(ADL, adl, get("bias"), 10.0)
        del adl
        result = dict(
            mesh_id=mid,
            mesh_mm=spacing,
            padding_mm=padding,
            reference=report,
            regional_induction=induction_by_region(both, mesh, recipe),
        )
        (folder / "induction.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        bias = {key: bias[key] for key in channels}
    if all_channels:
        for key, amps in bias.items():
            path = folder / (key + ".npz")
            step = 0.01 * abs(amps)
            currents = {**report["currents_A"], key: amps + step}
            if path.is_file():
                with np.load(path) as f:
                    info = json.loads(str(f["receipt"]))
                    if (
                        info["mesh_id"] == mid
                        and info["after"] == currents
                        and info["before"] == report["currents_A"]
                    ):
                        continue
            new = {}
            corners, moments, receipt = nonlinear_solve(
                recipe,
                mesh,
                currents,
                backend="cuda",
                air_solution=new,
                potential_seed_a=both["free_potential_a"],
                tolerance=2e-9,
                max_iterations=150,
                progress=lambda i, r: print(f"{key} Newton {i}: {r:.4g}", flush=True),
            )
            new.update(
                corners_m=corners, moments_Am2=moments, recipe=both["recipe"], report=json.dumps(receipt)
            )
            save(key, new, both, step)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh-mm", type=float, default=5.0)
    p.add_argument("--padding-mm", type=float, default=170.0)
    p.add_argument("--all-channels", action="store_true")
    a = p.parse_args()
    run(a.mesh_mm, a.padding_mm, a.all_channels)
