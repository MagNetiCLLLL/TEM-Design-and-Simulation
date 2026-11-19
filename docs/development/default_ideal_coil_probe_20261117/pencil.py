"""Source-free nonlinear sensitivity of the two installed hexapoles and Objective lens.

The calculation uses the physical on-axis coil fields, leading transverse HP
coefficients, round-lens off-axis terms through third order and mechanical
relativistic slopes. Pupil pencils carry zero current. Full Lorentz transport
of a new original source is required before accepting any electrical recipe.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.interpolate import make_interp_spline
from scipy.optimize import least_squares

from measure import desktop_state
from design import FACTOR
from temsim.ideal_coil_segments import set_current_segments
from temsim.physics.cuda_coil_potential import prepare_cuda_coil_potential
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
from temsim.physics.multipole_coil_field import capture_multipole_coil_fields
from temsim.physics.potential_capture import capture_round_lens_potential


def hex_coefficients(state, field, z):
    snapshot = ElectromagneticPotentialSnapshot(
        capture_physical_scene(state),
        (
            PotentialContribution(
                field.component_key, (field.component_key,), field.potential, "magnetic"
            ),
        ),
    )
    sampler = prepare_cuda_coil_potential(snapshot, execution_backend="Require GPU")
    theta = np.arange(12) * 2 * np.pi / 12
    radii = np.array([25e-6, 50e-6])
    xyz = np.empty((len(z), 2, len(theta), 3))
    xyz[..., 0] = radii[:, None] * np.cos(theta)
    xyz[..., 1] = radii[:, None] * np.sin(theta)
    xyz[..., 2] = z[:, None, None]
    b = np.concatenate(
        [sampler.sample_fields(chunk)["B_t"] for chunk in np.array_split(xyz.reshape(-1, 3), 32)]
    ).reshape(xyz.shape)
    normal = (b[..., 1] * np.cos(2 * theta) + b[..., 0] * np.sin(2 * theta)).mean(axis=2) / radii**2
    skew = (b[..., 1] * np.sin(2 * theta) - b[..., 0] * np.cos(2 * theta)).mean(axis=2) / radii**2
    return np.stack([(4 * c[:, 0] - c[:, 1]) / 3 for c in (normal, skew)], axis=1)


def coefficients(recipe, mesh_mm):
    _, s = desktop_state(1024)
    s.ac_deflector.scan_enabled = False
    s.descan_deflector.scan_enabled = False
    for owner, groups in recipe["recipes"].items():
        set_current_segments(s, owner, groups)
    start = s.electron_gun.exit_plane_z_mm * 0.001
    end = s.sample.upper_surface_z_mm * 0.001
    z = np.linspace(start, end, int(np.ceil((end - start) / 0.0002)) + 1)
    mid = np.linspace(start, end, 2 * int(np.ceil((end - start) / (mesh_mm * 0.001))) + 1)
    h = 2 * (mid[1] - mid[0])
    field = np.zeros(len(z))
    objective = None
    for lens in s.lenses:
        if not lens.enabled:
            continue
        potential = capture_round_lens_potential(s, lens.key).potential
        value = np.zeros(len(z))
        for c in potential.coils:
            if any(c.registration.origin_global_m) or not np.array_equal(
                c.registration.rotation_array, np.eye(3)
            ):
                raise ValueError("Analytical pencil basis requires axial round coils")
            value += np.asarray(c.axial_derivatives(z, 0)[0])
        field += value
        if lens.key == "objective_lens":
            objective = value
    if objective is None:
        raise ValueError("Missing objective field")
    fields = []
    for v in (field, objective):
        spline = make_interp_spline(z, v, k=7)
        fields.extend(spline(mid, nu=n) for n in range(4))
    hp_names = []
    linear = []
    grids = np.linspace(start, end, int(np.ceil((end - start) / 0.0005)) + 1)
    for hp in capture_multipole_coil_fields(s):
        if not any(hp.potential.currents_a):
            continue
        if hp.component_key not in ("probe_hp1_hexapole", "probe_hp2_hexapole"):
            raise ValueError("Unexpected excited transverse multipole")
        hp_names.append(hp.component_key)
        curve = hex_coefficients(s, hp, grids)
        spline = make_interp_spline(grids, curve, k=5)
        for v in (spline(mid), spline(mid, nu=1)):
            fields.extend(v.T)
        linear.append(
            float(next(a.strength_m3 for a in s.corrector_elements if a.key == hp.component_key))
        )
    if len(hp_names) != 2:
        raise ValueError("Two active corrector hexapoles are required")
    return np.ascontiguousarray(np.array(fields).T), h, hp_names, linear


@njit(cache=True)
def force(v, f):
    x, y, tx, ty = v
    rr = x * x + y * y
    b0, b1, b2, b3, hn, hs, dn, ds = f
    radial = -b1 / 2 + rr * b3 / 16
    bx = x * radial + 2 * hn * x * y - hs * (x * x - y * y)
    by = y * radial + hn * (x * x - y * y) + 2 * hs * x * y
    bz = b0 - rr * b2 / 4 + dn * y * (x * x - y * y / 3) - ds * x * (x * x / 3 - y * y)
    scale = -2 * FACTOR * np.sqrt(1 + tx * tx + ty * ty)
    return np.array(
        [
            tx,
            ty,
            scale * (ty * bz - (1 + tx * tx) * by + tx * ty * bx),
            scale * ((1 + ty * ty) * bx - tx * bz - tx * ty * by),
        ]
    )


@njit(cache=True)
def propagate(initial, field, h):
    out = initial.copy()
    for j in range(len(initial)):
        v = initial[j].copy()
        for i in range(0, len(field) - 1, 2):
            a = force(v, field[i])
            b = force(v + h * a / 2, field[i + 1])
            c = force(v + h * b / 2, field[i + 1])
            d = force(v + h * c, field[i + 2])
            v += h * (a + 2 * b + 2 * c + d) / 6
        out[j] = v
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mesh-mm", type=float, default=0.15)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    r = json.loads(args.recipe.read_text())
    f, h, names, strengths = coefficients(r, args.mesh_mm)
    np.savez_compressed(args.output / "operator.npz", fields=f, h=h)
    # Four rings and eight azimuths identify defocus and angular harmonics.
    theta = np.arange(8) * 2 * np.pi / 8
    xy = (
        r["gun_r95_m"]
        * np.array([0.2, 0.5, 0.8, 1.0])[:, None, None]
        * np.array([np.cos(theta), np.sin(theta)]).T
    ).reshape(-1, 2)
    # The radial incident derivative is measured on the source used for initial design.
    source = np.column_stack([xy, 0.9020248818935516 * xy])

    def run(x):
        field = np.column_stack(
            (f[:, :4] + x[2] * f[:, 4:8] / 1000, x[0] * f[:, 8:12] + x[1] * f[:, 12:16])
        )
        return propagate(source, np.ascontiguousarray(field), h)

    def residual(x):
        return (run(x)[:, :2] * 1e8).ravel()

    start = np.array([1.0, 1.0, 0.0])
    initial = run(start)
    print("Executed initial zero-current pencil", np.max(np.hypot(*initial[:, :2].T)) * 1e9, flush=True)
    # Round focus and both actual hexapole commands are independent variables.
    selections = []
    for seed in (start, np.array([0.0, 0.0, 0.0]), np.array([-1.0, 1.0, 0.0])):
        fit = least_squares(
            residual,
            seed,
            bounds=([-100.0, -100.0, -8.0], [100.0, 100.0, 8.0]),
            max_nfev=45,
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        selections.append((np.linalg.norm(fit.fun), fit.x))
        print("Pencil optimization", selections[-1], flush=True)
    _, best = min(selections, key=lambda row: row[0])
    final = run(best)
    trial = dict(r)
    trial["diagnostic_parent"] = str(args.recipe)
    trial["status"] = "Zero-current Taylor-field candidate; requires new emitted-source validation"
    trial["hexapole_strength_m3"] = {key: float(k * v) for key, k, v in zip(names, best, strengths)}
    objective_scale = 1.0 + best[2] / 1000.0
    for winding in trial["recipes"]["objective_lens"]["windings"].values():
        for cell in winding:
            cell["current_ratio"] *= objective_scale
    for i, owner in enumerate(trial["owner_per_group"]):
        if owner == "objective_lens":
            trial["current_a"][i] *= objective_scale
    trial["pencil"] = dict(
        hexapole_order=names,
        controls=best.tolist(),
        mesh_mm=args.mesh_mm,
        original_max_radius_nm=float(np.max(np.hypot(*initial[:, :2].T)) * 1e9),
        fitted_max_radius_nm=float(np.max(np.hypot(*final[:, :2].T)) * 1e9),
        fitted_r95_ring_angle_mrad=(np.arctan(np.hypot(*final[-8:, 2:].T)) * 1e3).tolist(),
    )
    (args.output / "trial-currents.json").write_text(
        json.dumps(trial, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(trial["pencil"], flush=True)


if __name__ == "__main__":
    main()
