"""Matched column and conditional, authenticated gun-exit distribution."""

import json
import sys

import numpy as np
from scipy.interpolate import make_interp_spline

from fields import CACHE, HERE, ROOT

OLD = HERE.parent / "default_ideal_coil_probe_20261117"
sys.path.insert(0, str(OLD))
from stationary_pupil import CHANNELS, FACTOR, current_vector, half_nodes, hex_vector, read_gun
from temsim.physics.beam_statistics import _weighted_quantile, transverse_beam_statistics

POINT = OLD / "working-point"
BASIS = ROOT / "outputs/default-ideal-coil-probe-20261117/release-basis.npz"
GROUPS = ("condenser_lens_2", "condenser_lens_3", "mini_condenser", "objective_lens")
SURFACE = 1.5991999975000002
START = 0.45


def metrics(phase, weights, current):
    stats = transverse_beam_statistics(*phase[:, :4].T, weights=weights)
    core = np.hypot(phase[:, 0], phase[:, 1]) <= 50e-12
    radius = np.hypot(phase[:, 0] - stats.mean_x_m, phase[:, 1] - stats.mean_y_m)
    return dict(
        D95_nm=2e9 * stats.radius_95_m,
        D90_nm=2e9 * _weighted_quantile(radius, weights, 0.9),
        rms_radius_nm=1e9 * stats.radius_rms_m,
        alpha95_mrad=stats.convergence_95_mrad,
        core_count=int(core.sum()),
        core_50pm_nA=1e9 * current * weights[core].sum(),
        transmitted_uA=1e6 * current * weights.sum(),
    )


def prepare(cache=CACHE, basis_path=None):
    from measure import desktop_state
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    from temsim.physics.potential_geometry import capture_potential_geometry
    from temsim.physics.unified_field_runtime import get_unified_field_runtime
    from temsim.physics.unified_particle_contacts import compile_contact_geometry

    cache.mkdir(parents=True, exist_ok=True)
    state = desktop_state(32768)[1]
    assert abs(state.sample.upper_surface_z_mm * 0.001 - SURFACE) < 1e-12
    geometry = compile_contact_geometry(capture_potential_geometry(state, START, SURFACE + 0.0004))
    assert geometry.native.all() and not np.any(geometry.origins_mm[:, :2])
    keys = np.array([p[0] for p in geometry.rows])
    assert np.isclose(geometry.radii_mm[keys == "condenser_aperture_2"].min(), 0.05)
    positions = {q.key: q.z_mm for q in state.lenses}
    np.savez(
        cache / "mechanical.npz",
        lo=geometry.axial_lower_m,
        hi=geometry.axial_upper_m,
        r=geometry.radii_mm * 0.001,
        names=keys,
        intervals=np.array([positions[k] for k in (GROUPS[0], GROUPS[1], "adapter_lens", GROUPS[2])]) * 0.001,
    )
    (potential,) = (
        part.potential
        for part in get_unified_field_runtime(state).snapshot.contributions
        if type(part.potential) is ElectricPotentialSnapshot
    )
    np.savez_compressed(cache / "electric.npz", **potential.arrays, digest=potential.digest)
    recipe = json.loads((POINT / "currents.json").read_text())
    basis = basis_path if basis_path is not None else BASIS
    if not basis.exists():
        from stationary_pupil import capture

        if basis_path is None:
            basis = cache / "release-basis.npz"
        if not basis.exists():
            capture(recipe, basis, spacing_mm=0.125)
    with np.load(basis) as f:
        owners = f["owners"]
        assert owners.tolist() == recipe["owner_per_group"]
        assert f["channels"].tolist() == [list(c) for c in CHANNELS]
        assert json.loads(str(f["geometry"])) == {
            k: [
                [u["interval_index"], u["turn_count"], u["axial_fraction"], u["radial_fraction"]]
                for w in v["windings"].values()
                for u in w
            ]
            for k, v in recipe["recipes"].items()
        }
        z, rd, hx = f["z"], f["round"], f["hex"]
    currents = current_vector(recipe)
    groups = [~np.isin(owners, GROUPS)] + [owners == k for k in GROUPS]
    combined = np.stack([np.einsum("i,ijk->jk", currents[g], rd[g]) for g in groups])
    transfer_names = [
        "adapter_lens",
        "probe_tl22_lens",
        "probe_tl21_lens",
        "probe_tl12_lens",
        "condenser_lens_1",
    ]
    transfers = np.stack(
        [np.einsum("i,ijk->jk", currents[owners == name], rd[owners == name]) for name in transfer_names]
    )
    np.savez_compressed(
        cache / "columns.npz",
        z=z,
        groups=combined,
        relay=transfers[0],
        transfers=transfers,
        transfer_names=transfer_names,
        hex=np.einsum("i,ijk->jk", hex_vector(recipe), hx),
    )


class Mesh:
    def __init__(self, step_mm, observe, cache=CACHE):
        self.cache = cache
        with np.load(cache / "mechanical.npz") as m:
            lo, hi, rad, self.keys, self.intervals = (m[k] for k in ("lo", "hi", "r", "names", "intervals"))
        observe = np.unique(np.asarray(observe))
        stop = max(observe.max(), SURFACE)
        faces = np.r_[lo, hi]
        core = np.linspace(START, SURFACE, int(np.ceil((SURFACE - START) * 1000 / step_mm)) + 1)
        focal_nodes = SURFACE + np.r_[np.arange(-300.0, 301.0, 5.0) * 1e-6, np.arange(-0.5, 0.51, 0.1) * 1e-6]
        z = np.unique(
            np.r_[
                core,
                focal_nodes[focal_nodes <= stop],
                np.arange(SURFACE + step_mm * 0.001, stop, step_mm * 0.001),
                stop,
                faces[(faces > START) & (faces < stop)],
                observe,
            ]
        )
        assert z[0] == START
        self.z, self.nodes, self.observe = z, half_nodes(z), observe
        self.saved = np.searchsorted(z, observe)
        middle = (z[:-1] + z[1:]) / 2
        self.radius = np.full(len(middle), np.inf)
        self.owner = np.full(len(middle), -1)
        for index, (left, right, r) in enumerate(zip(lo, hi, rad)):
            hits = (middle >= left) & (middle <= right) & (r < self.radius)
            self.radius[hits], self.owner[hits] = r, index
        with np.load(cache / "columns.npz") as f:
            self.group = make_interp_spline(f["z"], f["groups"], k=5, axis=1)(self.nodes)
            self.hex = make_interp_spline(f["z"], f["hex"], k=5)(self.nodes)
            self.relay = make_interp_spline(f["z"], f["relay"], k=5)(self.nodes)
            self.transfers = make_interp_spline(f["z"], f["transfers"], k=5, axis=1)(self.nodes)
        with np.load(cache / "electric.npz") as e:
            z_e = e["z"]
            self.r_e, self.v_e = e["r"], e["voltage"]
            idx = np.clip(np.searchsorted(z_e, self.nodes, side="right") - 1, 0, len(z_e) - 2)
            dz = np.diff(z_e)[idx]
            self.electric = self.r_e, self.v_e, idx, (self.nodes - z_e[idx]) / dz, dz

    def base(self, case, ni, level, padding):
        if case == "parked":
            return self.group[0] + self.group[4]
        sign = np.sign(ni)
        with np.load(self.cache / f"{case}-{abs(ni):.12g}-{level}-p{padding:g}.npz") as f:
            z, radial, potential = f["z"], f["r"], f["a_phi"]
        # Use the complete beam bore. A_phi is continuous up to its material boundary.
        radius = 0.00288
        inside = (radial > 0) & (radial <= radius * (1 + 1e-14))
        vand = np.polynomial.polynomial.polyvander((radial[inside] / radius) ** 2, 3)
        co, _, rank, _ = np.linalg.lstsq(vand, potential[inside] / radial[inside, None], rcond=None)
        assert rank == 4
        assert z[0] <= self.nodes[0] and z[-1] >= self.nodes[-1]
        sp = make_interp_spline(z, co.T, k=3)
        field = self.group[0].copy()
        factors = sign * np.array([2.0, -16.0, 384.0, -18432.0]) / radius ** np.arange(0, 8, 2)
        field[:, 0:8:2] += sp(self.nodes) * factors
        field[:, 1:8:2] += sp(self.nodes, nu=1) * factors
        return field

    def field(self, base, controls):
        field = base.copy()
        for i, v in enumerate(controls[:3]):
            field += v * self.group[i + 1]
        for factor, transfer in zip(controls[3:], self.transfers):
            field += (factor - 1.0) * transfer
        return np.ascontiguousarray(np.column_stack((field, self.hex)))


if __name__ == "__main__":
    prepare()
