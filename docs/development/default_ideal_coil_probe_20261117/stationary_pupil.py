"""Static vacuum winding-field operator for bounded, zero-current optical design.

Round fields use the captured loop quadrature through axial derivative eleven.
The sextupole scalar-potential expansion includes its longitudinal fringe and
first transverse fringe correction. This is independently checked against full
native Lorentz transport before interpreting a physical working point.
"""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange
from scipy.constants import c, e, m_e
from scipy.interpolate import make_interp_spline

from design import FACTOR
from measure import desktop_state
from pencil import hex_coefficients
from temsim.ideal_coil_segments import set_current_segments
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.physics.multipole_coil_field import capture_multipole_coil_fields

CHANNELS = [
    ("probe_hp2_hexapole", "strength_m3"),
    ("probe_hp1_hexapole", "strength_m3"),
    ("probe_hpc_hexapole", "strength_m3"),
    ("probe_hpc_hexapole", "skew_strength_m3"),
    ("probe_hpol_hexapole", "strength_m3"),
    ("probe_hpol_hexapole", "skew_strength_m3"),
]
UNIT = 1e6
REST_EV = m_e * c * c / e
REFERENCE_PC_EV = np.sqrt(300000.0 * (300000.0 + 2 * REST_EV))


def current_vector(recipe):
    return np.array(recipe["current_a"], dtype=float)


def hex_vector(recipe):
    return np.array([recipe["multipole_commands"][k][a] / UNIT for k, a in CHANNELS])


def capture(recipe, output, spacing_mm=0.25):
    _, state = desktop_state(1024)
    for key, groups in recipe["recipes"].items():
        set_current_segments(state, key, groups)
    start = state.electron_gun.exit_plane_z_mm * 0.001
    surface = state.sample.upper_surface_z_mm * 0.001
    z = np.linspace(start, surface + 0.002, int(np.ceil((surface + 0.002 - start) * 1000 / spacing_mm)) + 1)
    round_fields, owners = [], []
    current = current_vector(recipe)
    for lens in state.lenses:
        if not lens.enabled:
            continue
        coils = capture_round_lens_potential(state, lens.key).potential.coils
        for coil in coils:
            if any(coil.registration.origin_global_m) or not np.array_equal(
                coil.registration.rotation_array, np.eye(3)
            ):
                raise ValueError("Pupil design requires centred round windings")
            i = len(owners)
            if recipe["owner_per_group"][i] != lens.key or current[i] == 0:
                raise ValueError("Round winding group order or nonzero reference current changed")
            group_turns = coil.ampere_turns / current[i]
            if not np.isclose(group_turns, round(group_turns), rtol=0, atol=1e-7) or group_turns < 1:
                raise ValueError("Captured coil strength disagrees with integer-turn recipe")
            owners.append(lens.key)
            round_fields.append(coil.axial_derivatives(z, 11).T / current[i])
    if owners != recipe["owner_per_group"]:
        raise ValueError("Electrical recipe and captured coils differ")
    components = {item.key: item for item in state.corrector_elements}
    hex_fields = []
    for key, attr in CHANNELS:
        owner = components[key]
        saved = {
            name: getattr(owner, name) for name in ("strength_m3", "skew_strength_m3") if hasattr(owner, name)
        }
        for name in saved:
            setattr(owner, name, UNIT if name == attr else 0.0)
        (hp,) = capture_multipole_coil_fields(state, component_keys=(key,))
        # Larger reference drive improves the small-radius numerical extraction.
        curve = hex_coefficients(state, hp, z)
        spline = make_interp_spline(z, curve, k=5)
        hex_fields.append(np.column_stack([spline(z, nu=n) for n in range(4)]))
        for name, value in saved.items():
            setattr(owner, name, value)
    np.savez_compressed(
        output,
        z=z,
        surface=surface,
        owners=owners,
        round=np.array(round_fields),
        hex=np.array(hex_fields),
        channels=np.array(CHANNELS),
        reference=current,
        geometry=json.dumps(
            {
                k: [
                    [r["interval_index"], r["turn_count"], r["axial_fraction"], r["radial_fraction"]]
                    for w in v["windings"].values()
                    for r in w
                ]
                for k, v in recipe["recipes"].items()
            },
            sort_keys=True,
        ),
    )


class Column:
    def __init__(self, archive, recipe, nodes):
        with np.load(archive) as fields:
            z = fields["z"]
            if list(fields["owners"]) != recipe["owner_per_group"] or fields["channels"].tolist() != [
                list(c) for c in CHANNELS
            ]:
                raise ValueError("Winding operator incompatible with recipe")
            geometry = {
                k: [
                    [r["interval_index"], r["turn_count"], r["axial_fraction"], r["radial_fraction"]]
                    for w in v["windings"].values()
                    for r in w
                ]
                for k, v in recipe["recipes"].items()
            }
            if geometry != json.loads(str(fields["geometry"])):
                raise ValueError("Only unchanged complete-turn groups may share the winding basis")
            if nodes[0] < z[0] or nodes[-1] > z[-1]:
                raise ValueError("Optical design mesh outside captured integration interval")
            self.round = np.ascontiguousarray(make_interp_spline(z, fields["round"], k=5, axis=1)(nodes))
            self.hex = np.ascontiguousarray(make_interp_spline(z, fields["hex"], k=5, axis=1)(nodes))

    def field(self, currents, commands):
        return np.ascontiguousarray(
            np.column_stack(
                (
                    (currents @ self.round.reshape(len(currents), -1)).reshape(-1, 12),
                    (commands @ self.hex.reshape(len(commands), -1)).reshape(-1, 8),
                )
            )
        )


def half_nodes(z):
    result = np.empty(2 * len(z) - 1)
    result[::2] = z
    result[1::2] = (z[:-1] + z[1:]) / 2
    return result


@njit(cache=True, inline="always")
def magnetic(x, y, f):
    s = x * x + y * y
    bz = f[0] + s * (
        -f[2] / 4 + s * (f[4] / 64 + s * (-f[6] / 2304 + s * (f[8] / 147456 - s * f[10] / 14745600)))
    )
    radial = -0.5 * f[1] + s * (
        f[3] / 16 + s * (-f[5] / 384 + s * (f[7] / 18432 + s * (-f[9] / 1474560 + s * f[11] / 176947200)))
    )
    # The scalar magnetic potential for m=3 is P3(x,y)[H(z)-r²H''(z)/16+...].
    n, k = f[12] - s * f[16] / 16, f[13] - s * f[17] / 16
    p, q = y * (x * x - y * y / 3), -x * (x * x / 3 - y * y)
    radial -= (p * f[16] + q * f[17]) / 8
    bx = x * radial + 2 * n * x * y - k * (x * x - y * y)
    by = y * radial + n * (x * x - y * y) + 2 * k * x * y
    bz += p * (f[14] - s * f[18] / 16) + q * (f[15] - s * f[19] / 16)
    return bx, by, bz


def electric_mesh(snapshot, nodes):
    if snapshot.metadata["basis"] != "axisymmetric-r2-z-bilinear-v1":
        raise ValueError("Stationary verification requires the captured bilinear electric basis")
    a = snapshot.arrays
    z = a["z"]
    if nodes[0] < z[0] or nodes[-1] > z[-1]:
        raise ValueError("Electric mesh exceeds the solved domain")
    j = np.clip(np.searchsorted(z, nodes, side="right") - 1, 0, len(z) - 2)
    dz = z[j + 1] - z[j]
    return a["r"], a["voltage"], j, (nodes - z[j]) / dz, dz


@njit(cache=True, inline="always")
def electric_at(x, y, node, electric):
    r, voltage, indices, fractions, steps = electric
    radius = np.hypot(x, y)
    if radius > r[-1]:
        # A per-ray exception disables Numba's outer parallel loop. Nonfinite
        # samples propagate to the verifier, which rejects the entire run.
        return np.nan, np.nan, np.nan, np.nan
    i = min(max(np.searchsorted(r, radius, side="right") - 1, 0), len(r) - 2)
    j, v, dz = indices[node], fractions[node], steps[node]
    ds = r[i + 1] ** 2 - r[i] ** 2
    u = (radius**2 - r[i] ** 2) / ds
    a, b, c0, d = voltage[i, j], voltage[i + 1, j], voltage[i, j + 1], voltage[i + 1, j + 1]
    bottom, top = a + u * (b - a), c0 + u * (d - c0)
    slope = ((1 - v) * (b - a) + v * (d - c0)) / ds
    return bottom + v * (top - bottom), -2 * x * slope, -2 * y * slope, -(top - bottom) / dz


@njit(cache=True, inline="always")
def rhs(x, y, tx, ty, f, rigidity, inv_speed, electric, node, entrance_energy, phi0):
    bx, by, bz = magnetic(x, y, f)
    norm = np.sqrt(1 + tx * tx + ty * ty)
    etx, ety = 0.0, 0.0
    if electric is not None:
        phi, ex, ey, ez = electric_at(x, y, node, electric)
        kinetic = entrance_energy + (phi - phi0)
        pc = np.sqrt(kinetic * (kinetic + 2 * REST_EV))
        rigidity = REFERENCE_PC_EV / pc
        inv_speed = (kinetic + REST_EV) / (c * pc)
        escale = -(kinetic + REST_EV) * norm**2 / (pc * pc)
        etx, ety = escale * (ex - tx * ez), escale * (ey - ty * ez)
    a = -2 * FACTOR * rigidity * norm
    return (
        tx,
        ty,
        a * (ty * bz - (1 + tx * tx) * by + tx * ty * bx) + etx,
        a * ((1 + ty * ty) * bx - tx * bz - tx * ty * by) + ety,
        norm * inv_speed,
    )


@njit(parallel=True)
def _transport(
    initial, fields, steps, rigidity, inv_speed, save, observe_envelope, electric=None, energy=None
):
    result = np.empty((len(save), len(initial), 5))
    extent = np.zeros((get_num_threads(), len(steps) if observe_envelope else 0))
    for j in prange(len(initial)):
        x, y, tx, ty, clock = initial[j]
        phi0, k0 = 0.0, 0.0
        if electric is not None:
            phi0 = electric_at(x, y, 0, electric)[0]
            k0 = energy[j]
        row = 0
        worker = get_thread_id()
        for i in range(len(steps) + 1):
            if row < len(save) and i == save[row]:
                result[row, j] = np.array((x, y, tx, ty, clock))
                row += 1
            if i == len(steps):
                break
            h = steps[i]
            a = rhs(x, y, tx, ty, fields[2 * i], rigidity[j], inv_speed[j], electric, 2 * i, k0, phi0)
            b = rhs(
                x + h * a[0] / 2,
                y + h * a[1] / 2,
                tx + h * a[2] / 2,
                ty + h * a[3] / 2,
                fields[2 * i + 1],
                rigidity[j],
                inv_speed[j],
                electric,
                2 * i + 1,
                k0,
                phi0,
            )
            d = rhs(
                x + h * b[0] / 2,
                y + h * b[1] / 2,
                tx + h * b[2] / 2,
                ty + h * b[3] / 2,
                fields[2 * i + 1],
                rigidity[j],
                inv_speed[j],
                electric,
                2 * i + 1,
                k0,
                phi0,
            )
            g = rhs(
                x + h * d[0],
                y + h * d[1],
                tx + h * d[2],
                ty + h * d[3],
                fields[2 * i + 2],
                rigidity[j],
                inv_speed[j],
                electric,
                2 * i + 2,
                k0,
                phi0,
            )
            if observe_envelope:
                extent[worker, i] = max(
                    extent[worker, i],
                    np.hypot(x, y),
                    np.hypot(x + h * a[0] / 2, y + h * a[1] / 2),
                    np.hypot(x + h * b[0] / 2, y + h * b[1] / 2),
                    np.hypot(x + h * d[0], y + h * d[1]),
                )
            x += h / 6 * (a[0] + 2 * b[0] + 2 * d[0] + g[0])
            y += h / 6 * (a[1] + 2 * b[1] + 2 * d[1] + g[1])
            tx += h / 6 * (a[2] + 2 * b[2] + 2 * d[2] + g[2])
            ty += h / 6 * (a[3] + 2 * b[3] + 2 * d[3] + g[3])
            clock += h / 6 * (a[4] + 2 * b[4] + 2 * d[4] + g[4])
            if observe_envelope:
                extent[worker, i] = max(extent[worker, i], np.hypot(x, y))
    return result, extent


def transport(initial, fields, steps, rigidity, inv_speed, save):
    return _transport(initial, fields, steps, rigidity, inv_speed, save, False)[0]


def kinematics(energy_ev):
    rest = m_e * c * c / e
    p = np.sqrt(energy_ev * (energy_ev + 2 * rest))
    return np.sqrt(300000.0 * (300000.0 + 2 * rest)) / p, (energy_ev + rest) / (c * p)


def read_gun(directory):
    path = directory / "gun.npz"
    meta = json.loads((directory / "gun.json").read_text())
    if sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Gun checkpoint hash mismatch")
    with np.load(path) as g:
        if (
            not np.array_equal(g["ray_id"], g["emission_ray_id"])
            or len(np.unique(g["ray_id"])) != len(g["ray_id"])
            or int(np.sum(g["weight"] > 0)) != meta["population"]["positive_current_count"]
            or meta["population"]["positive_current_count"] < 1000
        ):
            raise ValueError("Gun checkpoint lacks sufficient original positive-current particles")
        active = g["alive"] & (g["weight"] > 0)
        phase = np.column_stack([g[k][active] for k in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s")])
        return (
            phase,
            g["weight"][active],
            g["ray_id"][active],
            g["energy_offset_ev"][active],
            float(g["emitted_current_a"]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacing-mm", type=float, default=0.25)
    args = parser.parse_args()
    capture(json.loads(args.recipe.read_text()), args.output, args.spacing_mm)
