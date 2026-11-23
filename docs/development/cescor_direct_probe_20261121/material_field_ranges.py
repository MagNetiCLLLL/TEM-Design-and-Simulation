"""Measure individual small-signal material-field profiles at the final corrector bias."""

from hashlib import sha256
import json

import numpy as np

from relay_magnetics import BASE, WORK
from relay_transfer import HERE, START
from temsim.corrector_field_cli import _save
from temsim.physics.corrector_vacuum_field import harmonic_table
from temsim.physics.probe_corrector_fem import solve


CACHE = WORK / "field-ranges"


def coefficient(field, xy, kind, radius_mm=1.2):
    mid = len(xy)//2
    if kind == "round":
        return field[:, mid, mid, 2].copy()
    j = int(round(radius_mm*.001/(xy[1]-xy[0])))
    np.testing.assert_allclose(xy[mid+j], radius_mm*.001, atol=1e-14)
    # Opposite cardinal samples cancel dipole and linear terms. Their difference
    # measures both phases of the leading sextupole coefficient in T/m^2.
    delta = (field[:, mid, mid+j, :2] + field[:, mid, mid-j, :2]
             - field[:, mid+j, mid, :2] - field[:, mid-j, mid, :2])/(4*xy[mid+j]**2)
    return delta[:, 0] + 1j*delta[:, 1]


def run():
    evidence = HERE / "evidence"
    record = json.loads((evidence/"normal-objective-probe.json").read_text())
    path = evidence/"normal-final-corrector-solution.npz"
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != record["field_archive_sha256"][path.name]:
        raise ValueError("Final corrector solution changed")
    with np.load(path) as f:
        base = dict(f)
    receipt = json.loads(str(base["report"]))
    recipe = json.loads(str(base["recipe"]))
    currents = receipt["currents_A"]
    if currents != record["after"]["corrector_A"]:
        raise ValueError("Final circuit bias changed")
    with np.load(BASE/"meshes"/(receipt["mesh_id"]+".npz")) as mesh_file:
        mesh = mesh_file["xyz"], mesh_file["tet"], mesh_file["names"][mesh_file["index"]], mesh_file["far"]
    with np.load(WORK/"normal-core-seed.npz") as seed:
        if seed["mesh_id"].item() != receipt["mesh_id"]:
            raise ValueError("Wrong continuation mesh")
        starting = seed["free_potential_a"]
    for key, amps in currents.items():
        kind = recipe["sources"][key]["kind"]
        increment = abs(amps)*.01
        target = CACHE/(key+".npz")
        if target.exists():
            with np.load(target) as prior:
                if prior["source_sha256"].item() == digest and prior["increment_A"] == increment:
                    continue
        slopes, inner, residuals = [], [], []
        for step in (-increment, increment):
            changed = dict(currents)
            changed[key] += step
            delta = {}
            print("Solving", key, "at", changed[key], "A", flush=True)
            corners, moments, report = solve(recipe, mesh, changed, backend="cuda", tolerance=2e-8,
                                             air_solution=delta, potential_seed_a=starting)
            for k, a in (("air_xyz_m", delta["air_xyz_m"]), ("air_tet", delta["air_tet"]), ("corners_m", corners)):
                np.testing.assert_array_equal(a, base[k])
            # Table reconstruction is linear in impressed current, magnetic
            # moments and the solved air potential. Subtract AFTER nonlinear solves.
            impulse = {k: 1. if k == key else 0. for k in currents}
            delta.update(corners_m=corners, moments_Am2=(moments-base["moments_Am2"])/step,
                         air_u_A=(delta["air_u_A"]-base["air_u_A"])/step,
                         recipe=base["recipe"], report=json.dumps(dict(currents_A=impulse)))
            z, xy, field = harmonic_table(delta, .0005, grid_half=6)
            slopes.append(coefficient(field, xy, kind))
            inner.append(coefficient(field, xy, kind, .6))
            residuals.append(report["residual"])
        _save(target, z_mm=(START+z)*1000, plus=slopes[1], minus=slopes[0],
              inner=(inner[1]+inner[0])/2, residuals=residuals, source_sha256=digest,
              kind=kind, bias_A=amps, increment_A=increment)
        print("Saved", key, flush=True)


if __name__ == "__main__":
    run()
