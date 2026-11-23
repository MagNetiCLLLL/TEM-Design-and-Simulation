"""Check the first source image with two independent transverse perturbations."""

import argparse
import numpy as np

from gun_focus import BASELINE_KV, HERE, angular_trace, prepare, write_json


def crossings(z, column):
    """Locate near-antiparallel vector reversals without counting beam rotation."""
    roots = []
    for i, (a, b) in enumerate(zip(column[:-1], column[1:])):
        product = np.linalg.norm(a)*np.linalg.norm(b)
        if not product or np.dot(a, b) > -.8*product:
            continue
        delta = b-a
        fraction = -np.dot(a, delta)/np.dot(delta, delta)
        residual = np.linalg.norm(a + fraction*delta)
        roots.append(dict(z_mm=float(z[i]+fraction*(z[i+1]-z[i])),
                          bracket_mm=z[i:i+2].tolist(),
                          opposite_ray_residual_nm_at_1urad=float(residual*2000)))
    return roots


def verify(kv, mesh, step_factor):
    gun, _ = prepare()
    if mesh is not None:
        gun._gun_field_cells_per_bore = mesh
    gun.trace_step_mm *= step_factor
    # Exact plane crossings; include the complete foil, the acceleration tube,
    # and the entrance fringe of the installed C1 lens at the gun handoff.
    z = np.unique(np.r_[.0001, .001, .005, .01, .025, .05, .1, .2, .3, .5, .75,
        np.arange(1, 30.01, .25), np.arange(22.5, 25.51, .025),
        23.9, 23.99, 23.995, 23.999, 24., 24.001, 24.005, 24.01, 24.1,
        np.arange(31, 100.1, 1.), np.arange(102, 450.01, 2.), 445.].round(8))
    amplitudes = [.5, 1., 2.]
    matrices, identity, _ = angular_trace(gun, kv, z, amplitudes)
    mesh = gun.base_electric_field.request["numerics"]["cells_per_bore"]
    label = "before" if kv == BASELINE_KV else "after"
    prefix = f"{label}-mesh-{mesh}-step-{step_factor:g}"
    np.savez_compressed(HERE / f"{prefix}.npz", z_mm=z, B_m_per_rad=matrices,
                        amplitudes_urad=amplitudes, voltage_kV=kv, field_sha256=identity)
    rows = []
    for a, b in zip(amplitudes, matrices):
        roots = [crossings(z, b[:, :, j]) for j in range(2)]
        at_foil = b[np.flatnonzero(z==24.).item()]
        rows.append(dict(perturbation_urad=a, roots_x=roots[0], roots_y=roots[1],
                         foil_separation_nm=(np.linalg.norm(at_foil, axis=0)*2e3*a).tolist()))
    print(prefix, kv, [(r["perturbation_urad"], [v["z_mm"] for v in r["roots_x"]],
                         r["foil_separation_nm"]) for r in rows], flush=True)
    write_json(HERE / f"{prefix}.json", dict(lens_kV=kv, field_sha256=identity,
        cells_per_bore=mesh, grid_shape=gun.base_electric_field.report["grid_shape"],
        trace_step_mm=gun.trace_step_mm, step_factor=step_factor,
        emission_energy_eV=gun.emitter.emission_energy_ev, angular_checks=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kv", type=float)
    parser.add_argument("--mesh", type=int)
    parser.add_argument("--step-factor", type=float, default=1.)
    options = parser.parse_args()
    verify(options.kv, options.mesh, options.step_factor)
