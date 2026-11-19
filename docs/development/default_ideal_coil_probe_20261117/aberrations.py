"""Use both installed auxiliary hexapoles to fit the nonlinear zero-current pupil."""

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.interpolate import make_interp_spline
from scipy.linalg import null_space
from scipy.optimize import least_squares

from design import FACTOR
from measure import desktop_state
from pencil import hex_coefficients
from temsim.physics.multipole_coil_field import capture_multipole_coil_fields


@njit(cache=True, inline="always")
def force(x, y, tx, ty, f, rigidity=1.0):
    r2 = x * x + y * y
    radial = -f[1] * 0.5 + r2 * f[3] / 16
    bx = x * radial + 2 * f[4] * x * y - f[5] * (x * x - y * y)
    by = y * radial + f[4] * (x * x - y * y) + 2 * f[5] * x * y
    bz = f[0] - r2 * f[2] * 0.25 + f[6] * y * (x * x - y * y / 3) - f[7] * x * (x * x / 3 - y * y)
    magnetic = -2 * FACTOR * rigidity * np.sqrt(1 + tx * tx + ty * ty)
    return (
        tx,
        ty,
        magnetic * (ty * bz - (1 + tx * tx) * by + tx * ty * bx),
        magnetic * ((1 + ty * ty) * bx - tx * bz - tx * ty * by),
    )


@njit(cache=True)
def trace_spectral(source, field, h, rigidity):
    result = np.empty_like(source)
    for j in range(len(source)):
        x, y, tx, ty = source[j]
        for i in range(0, len(field) - 1, 2):
            a, b, c, d = force(x, y, tx, ty, field[i], rigidity[j])
            p, q, r, s = force(
                x + h * a / 2, y + h * b / 2, tx + h * c / 2, ty + h * d / 2, field[i + 1], rigidity[j]
            )
            u, v, w, t = force(
                x + h * p / 2, y + h * q / 2, tx + h * r / 2, ty + h * s / 2, field[i + 1], rigidity[j]
            )
            k, l, m, n = force(x + h * u, y + h * v, tx + h * w, ty + h * t, field[i + 2], rigidity[j])
            x += h * (a + 2 * p + 2 * u + k) / 6
            y += h * (b + 2 * q + 2 * v + l) / 6
            tx += h * (c + 2 * r + 2 * w + m) / 6
            ty += h * (d + 2 * s + 2 * t + n) / 6
        result[j] = np.array([x, y, tx, ty])
    return result


@njit(cache=True)
def trace(source, field, h):
    return trace_spectral(source, field, h, np.ones(len(source)))


def auxiliary(fields, h, output):
    _, state = desktop_state(1024)
    state.ac_deflector.scan_enabled = False
    state.descan_deflector.scan_enabled = False
    start = state.electron_gun.exit_plane_z_mm * 0.001
    stop = state.sample.upper_surface_z_mm * 0.001
    mid = np.linspace(start, stop, len(fields))
    z = np.linspace(start, stop, int(np.ceil((stop - start) / 0.0005)) + 1)
    channels = [("probe_hp2_hexapole", "strength_m3"), ("probe_hp1_hexapole", "strength_m3")]
    components = {c.key: c for c in state.corrector_elements}
    values = [getattr(components[key], attr) for key, attr in channels]
    extra = []
    for key in ("probe_hpc_hexapole", "probe_hpol_hexapole"):
        owner = components[key]
        for attr in ("strength_m3", "skew_strength_m3"):
            if getattr(owner, attr) != 0.0:
                raise ValueError("Auxiliary basis expects the original parked zero command")
            value = 1e6
            setattr(owner, attr, value)
            (hp,) = capture_multipole_coil_fields(state, component_keys=(key,))
            curve = hex_coefficients(state, hp, z)
            spline = make_interp_spline(z, curve, k=5)
            extra.extend((*spline(mid).T, *spline(mid, nu=1).T))
            setattr(owner, attr, 0.0)
            channels.append((key, attr))
            values.append(value)
    augmented = np.column_stack((fields, np.array(extra).T))
    bounds = np.array(
        [components[key].maximum_strength_m3 / abs(v) for (key, attr), v in zip(channels, values)]
    )
    np.savez_compressed(
        output, fields=augmented, h=h, names=np.array(channels), values=values, bounds=bounds
    )
    return augmented, channels, np.array(values), bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("operator", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    recipe = json.loads(args.recipe.read_text())
    with np.load(args.operator) as data:
        field = data["fields"]
        h = float(data["h"])
    cache = args.output / "six-channel-operator.npz"
    if cache.exists():
        with np.load(cache) as stored:
            field, channels, values, bounds = (
                stored[k] for k in ("fields", "names", "values", "bounds")
            )
    else:
        field, channels, values, bounds = auxiliary(field, h, cache)
    size = recipe["gun_r95_m"]
    angles = np.arange(8) * np.pi / 4
    radii = np.array([0.15, 0.4, 0.75, 1.0])

    def pupil(angles, radii):
        xy = (size * radii[:, None, None] * np.column_stack((np.cos(angles), np.sin(angles)))).reshape(
            -1, 2
        )
        return np.column_stack((xy, 0.9020248818935516 * xy))

    inputs = pupil(angles, radii)

    def run(x, initial=inputs):
        combined = field[:, 8:].reshape(len(field), 6, 4).transpose(0, 2, 1) @ x[:6]
        columns = np.column_stack((field[:, :4] + x[6] * field[:, 4:8] / 1000, combined))
        return trace(initial, np.ascontiguousarray(columns), h)

    def resid(x):
        return (1e9 * run(x)[:, :2]).ravel()

    zero = np.zeros(7)

    def harmonic(x, n):
        y = run(x)[-len(angles) :, :2]
        return np.mean((y[:, 0] + 1j * y[:, 1]) * np.exp(-1j * n * angles))

    eye = np.eye(7)
    quadr = np.array([(harmonic(v, -2) - harmonic(-v, -2)) / 2 for v in eye[:6]])
    scale = np.maximum(abs(quadr), max(abs(quadr)) * 0.001)
    space = null_space(np.stack((quadr.real / scale, quadr.imag / scale))) / scale[:, None]
    space /= np.max(abs(space), axis=0)

    # Identify a quadratically cancelling direction with an opposing cubic
    # coefficient using distinct radii to separate its linear defocus term.
    def cubic(x):
        yy = run(x)[:, :2].reshape(len(radii), len(angles), 2)
        mode = np.mean((yy[:, :, 0] + 1j * yy[:, :, 1]) * np.exp(-1j * angles), axis=1)
        return np.linalg.lstsq(np.column_stack((radii, radii**3, radii**5)), mode, rcond=None)[0][1]

    reference = cubic(zero)
    rotation = np.conj(reference) / abs(reference)

    def projected(v):
        trial = np.r_[space @ v, 0.0]
        return float(np.real(((cubic(trial) + cubic(-trial)) / 2 - reference) * rotation))

    k = space.shape[1]
    v = np.eye(k) * 2
    mat = np.zeros((k, k))
    for i in range(k):
        mat[i, i] = projected(v[i]) / 4
        for j in range(i):
            mat[i, j] = mat[j, i] = (projected(v[i] + v[j]) - projected(v[i]) - projected(v[j])) / 8
    eigen, basis = np.linalg.eigh(mat)
    seeds = [np.r_[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    for i, ev in enumerate(eigen):
        if ev < 0:
            seed = space @ basis[:, i] * np.sqrt(abs(reference) / (-ev))
            seed /= max(1.0, np.max(abs(seed) / (bounds * 0.9)))
            seeds.extend([np.r_[sign * seed, 0.0] for sign in (-1.0, 1.0)])
    upper = np.r_[bounds, 8.0]
    print("Baseline cubic m", reference, "null-space curvature", eigen, "seeds", len(seeds), flush=True)
    best = np.inf
    for seed in seeds:
        fit = least_squares(
            resid,
            seed,
            bounds=(-upper, upper),
            max_nfev=120,
            x_scale="jac",
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        norm = np.linalg.norm(fit.fun)
        print("Nonlinear pupil", norm, fit.x, flush=True)
        if norm >= best:
            continue
        best = norm
        test = pupil(np.arange(24) * 2 * np.pi / 24, np.linspace(0.08, 1.15, 17))
        check = run(fit.x, test)
        trial = json.loads(args.recipe.read_text())
        commands = {}
        for (key, attr), command in zip(channels, fit.x[:6] * values):
            commands.setdefault(str(key), {})[str(attr)] = float(command)
        trial["multipole_commands"] = commands
        factor = 1 + fit.x[6] / 1000
        for w in trial["recipes"]["objective_lens"]["windings"].values():
            for c in w:
                c["current_ratio"] *= factor
        for i, key in enumerate(trial["owner_per_group"]):
            if key == "objective_lens":
                trial["current_a"][i] *= factor
        report = dict(
            controls=fit.x.tolist(),
            objective_scale=factor,
            optimization_rms_nm=norm / np.sqrt(len(inputs)),
            maximum_holdout_radius_nm=float(np.max(np.hypot(*check[:, :2].T)) * 1e9),
            rings=np.linspace(0.08, 1.15, 17).tolist(),
            holdout_rms_per_ring_nm=np.sqrt(
                np.mean(np.hypot(*check[:, :2].T).reshape(17, 24) ** 2, axis=1)
            )
            .__mul__(1e9)
            .tolist(),
        )
        trial["zero_current_aberration_fit"] = report
        trial["status"] = "Nonlinear source-free design; physical full-current qualification pending"
        (args.output / "trial-currents.json").write_text(
            json.dumps(trial, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("Holdout RMS nm per ring", report["holdout_rms_per_ring_nm"], flush=True)


if __name__ == "__main__":
    main()
