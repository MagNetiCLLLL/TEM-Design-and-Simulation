"""Nonlinear relay winding sweeps with manufacturer data for the same installed alloys."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.constants import mu_0

from adjust_a2 import OUT, WORK as BASE, refine_grid
from temsim.corrector_field_cli import _save
from temsim.physics.probe_corrector_fem import solve


WORK = OUT.parent / "cescor-relay-20261121"


def iron_points(path, curve):
    """Read intrinsic polarization from the vector trace of Figure 8, printed p. 7."""
    import pymupdf

    if sha256(path.read_bytes()).hexdigest() != curve["source_sha256"]:
        raise ValueError("Iron manufacturer document differs from the library reference")
    with pymupdf.open(path) as doc:
        drawings = doc[8].get_drawings()
    line = drawings[0]["items"]
    assert len(line) == 6 and all(c[0] == "c" for c in line)
    t = np.linspace(0, 1, 2001)[:, None]
    samples = np.vstack(
        [
            (1 - t) ** 3 * np.array(p0)
            + 3 * (1 - t) ** 2 * t * np.array(p1)
            + 3 * (1 - t) * t**2 * np.array(p2)
            + t**3 * np.array(p3)
            for _, p0, p1, p2, p3 in line
        ]
    )
    xt = np.sort([drawings[i]["rect"].x0 for i in (2, 3, 4, 5)])
    slope, offset = np.polyfit(np.log10([500, 5000, 50000, 500000]), xt, 1)
    y15, y22 = drawings[6]["rect"].y0, drawings[13]["rect"].y0
    h = np.array(
        [25000.0, 30000.0, 40000.0, 50000.0, 65000.0, 85000.0, 120000.0, 180000.0, 300000.0, 500000.0]
    )
    y = np.interp(slope * np.log10(h) + offset, samples[:, 0], samples[:, 1])
    j = 1.5 + 0.7 * (y - y15) / (y22 - y15)
    return h, j + mu_0 * h


def material_recipe(original, shape):
    """Keep all native low-field knots and all assigned material and geometry identities."""
    result = deepcopy(original)
    tables = {}
    for solid in result["solids"]:
        c = solid["curve"]
        if c is None:
            continue
        key = c["key"]
        if key not in tables:
            if key == "armco_pure_iron":
                hv, bv = iron_points(WORK / "iron.pdf", c)
                source = "Cleveland-Cliffs, Figure 8, intrinsic J(H) + mu0 H. First additional point 25 kA/m."
            elif key == "permenorm_5000_h2":
                if sha256((WORK / "nife.pdf").read_bytes()).hexdigest() != c["source_sha256"]:
                    raise ValueError("Ni-Fe manufacturer document differs from the library reference")
                # VAC reports B=1.60 T at H=40 kA/m. It does not publish intermediate
                # solid-material measurements in this gap. Bracket the effect of the interpolation in J.
                hv = np.array([40000.0])
                bv = np.array([1.60])
                if shape == "log-J":
                    h0, b0 = c["h_a_per_m"][-1], c["b_t"][-1]
                    hv = np.r_[1500, 2500, 4000, 7000, 12000, 22000, hv]
                    j = np.interp(np.log(hv), np.log([h0, 40000.0]), [b0 - mu_0 * h0, 1.60 - mu_0 * 40000.0])
                    bv = j + mu_0 * hv
                source = f"VAC solid datasheet p. 2, 1.60 T at 40 kA/m; {shape} interpolation to that measured endpoint."
            else:
                raise ValueError(key)
            c["h_a_per_m"] += hv.tolist()
            c["b_t"] += bv.tolist()
            assert np.all(np.diff(c["b_t"]) > 0) and np.all(np.diff(c["h_a_per_m"]) > 0)
            c["source_block"] += " " + source
            c["scope"] += " High-field extension for the relay study: " + source
            tables[key] = c
        solid["curve"] = tables[key]
    return result


def run(a):
    with np.load(OUT / "a2-cubic-both-solution.npz") as f:
        base = json.loads(str(f["report"]))
        original = json.loads(str(f["recipe"]))
    recipe = material_recipe(original, a.interpolation)
    if a.mesh is not None:
        with np.load(OUT / (a.mesh + "-solution.npz")) as f:
            other = json.loads(str(f["report"]))
        for key in ("mesh_id", "material_size_mm", "exterior_padding_mm"):
            base[key] = other[key]
    with np.load(BASE / "meshes" / (base["mesh_id"] + ".npz")) as f:
        mesh = f["xyz"], f["tet"], f["names"][f["index"]], f["far"]
    currents = base["currents_A"].copy()
    currents.update(probe_tl21_lens=a.tl21, probe_tl22_lens=a.tl22)
    for name, value in (
        ("probe_hp1_hexapole", a.hp1),
        ("probe_hp2_hexapole", a.hp2),
        ("probe_hpol_hexapole_x", a.hpol_x),
        ("probe_hpol_hexapole_y", a.hpol_y),
    ):
        if value is not None:
            currents[name] = value
    seed = None
    if a.seed:
        with np.load(WORK / (a.seed + "-seed.npz")) as f:
            assert str(f["mesh_id"]) == base["mesh_id"]
            seed = f["free_potential_a"] * a.scale
        if a.seed2:
            with np.load(WORK / (a.seed2 + "-seed.npz")) as f:
                assert str(f["mesh_id"]) == base["mesh_id"]
                seed = (1 - a.blend) * seed + a.blend * f["free_potential_a"]
    air, checkpoint = {}, {}
    corners, moments, report = solve(
        recipe,
        mesh,
        currents,
        backend="cuda",
        air_solution=air,
        potential_seed_a=seed,
        potential_checkpoint=checkpoint,
        max_iterations=240,
        progress=lambda i, r: print(f"Newton {i}: {r:.4g}", flush=True),
    )
    for key in (
        "mesh_id",
        "material_size_mm",
        "exterior_padding_mm",
        "source_file",
        "coordinate_frame",
        "base_circuit_sha256",
    ):
        report[key] = base[key]
    report["high_field_interpolation"] = a.interpolation
    _save(WORK / (a.name + "-seed.npz"), mesh_id=base["mesh_id"], **checkpoint)
    solution = WORK / (a.name + "-solution.npz")
    _save(
        solution,
        corners_m=corners,
        moments_Am2=moments,
        recipe=json.dumps(recipe, sort_keys=True),
        report=json.dumps(report, sort_keys=True),
        **air,
    )
    print("Sampling nonlinear solution", report["material_peaks"], flush=True)
    refine_grid(solution, WORK / (a.name + "-field.npz"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("tl21", type=float)
    p.add_argument("tl22", type=float)
    p.add_argument("--hp1", type=float, help="Signed HP1 current in amperes")
    p.add_argument("--hp2", type=float, help="Signed HP2 current in amperes")
    p.add_argument("--hpol-x", type=float, help="Signed HPol X current in amperes")
    p.add_argument("--hpol-y", type=float, help="Signed HPol Y current in amperes")
    p.add_argument("--seed")
    p.add_argument("--seed2")
    p.add_argument("--blend", type=float, default=0.0)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--mesh", help="Archived independent mesh tag")
    p.add_argument("--interpolation", choices=("linear-H", "log-J"), default="linear-H")
    run(p.parse_args())
