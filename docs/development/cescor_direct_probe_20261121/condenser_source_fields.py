"""Rebuild the incident round-winding basis and the gun field at the selected FEM solution."""

from dataclasses import replace
from functools import cached_property
import gzip
import hashlib
import json

import numpy as np
from numba import njit
from scipy.constants import mu_0
from scipy.interpolate import CubicSpline, make_interp_spline

from column_current import ACTIVE, BASIS, HERE, OWNER, POINT, WORK, desktop_state
from joint_pivot_tuning import OUT as JOINT
from temsim.ideal_coil_segments import set_current_segments
from temsim.optics.electron_gun.emitter import EmissionSampling
from temsim.physics.corrector_vacuum_field import vacuum_field
from temsim.physics.incremental_particles import capture_emission_phase
from temsim.physics.instrument_electric import capture_instrument_electric_field
from temsim.physics.potential_capture import capture_round_lens_potential


EV = HERE / "evidence"
OUT = HERE.parents[2] / "outputs/condenser-source-audit-20261122"
FINAL = json.loads((EV / "joint-objective-focus.json").read_text())
RECEIPT = json.loads((EV / "joint-three-plane-result.json").read_text())
FEM = JOINT / f"tl12-{FINAL['TL12_A']:.10g}A-field-24.npz"
MAT = EV / "joint-tl12-material-solution.npz"
OBJ = EV / "joint-objective-material-field.npz"
LENSES = ("condenser_lens_1", "condenser_lens_2", "condenser_lens_3")
PROJECTION = ("diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_file(path, digest):
    if sha(path) != digest:
        raise ValueError(f"Field or input receipt mismatch: {path}")


def prepared_source():
    state = desktop_state(32768)[1]
    gun = state.electron_gun
    nodes = json.loads(gzip.decompress((POINT / "inputs.json.gz").read_bytes()))["graph"]["nodes"]
    (src,) = [n["attributes"] for n in nodes if n.get("type", "").endswith(":ColdFieldEmitter")]
    gun.emitter.sampling = EmissionSampling(**nodes[src["_emission_sampling"]["ref"]]["attributes"])
    gun.emitter.emission_current_na = 15000.0
    gun.c1_aperture.diameter_um = 1000.0
    emit = gun.emit()
    phase = capture_emission_phase(gun, emit)
    check_file(POINT / "gun.npz", json.loads((POINT / "gun.json").read_text())["sha256"])
    with np.load(POINT / "gun.npz") as g:
        for name in ("ray_id", "position_m", "direction", "normal"):
            np.testing.assert_array_equal(phase[name], g["emission_" + name])
        np.testing.assert_array_equal(phase["weight"], g["weight"])
    return state, emit


def closed_form(coil, z):
    """Biot-Savart integrated over a uniform rectangular winding, SI per ampere."""

    def primitive(d):
        a, b = coil.inner_radius_m, coil.outer_radius_m
        return d * np.log((b + np.hypot(d, b)) / (a + np.hypot(d, a)))

    d = np.asarray(z) - coil.center_z_m
    L = coil.length_m
    return (
        mu_0
        * coil.ampere_turns
        / (2 * L * (coil.outer_radius_m - coil.inner_radius_m))
        * (primitive(d + L / 2) - primitive(d - L / 2))
    )


class Fields:
    def __init__(self):
        OUT.mkdir(exist_ok=True, parents=True)
        self.state, self.emission = prepared_source()
        check_file(
            EV / "joint-objective-focus.json", RECEIPT["evidence_sha256"]["joint-objective-focus.json"]
        )
        recipe = json.loads((POINT / "currents.json").read_text())
        self.recipe = recipe
        original = np.asarray(recipe["current_a"])
        for name, definition in recipe["recipes"].items():
            set_current_segments(self.state, name, definition)
        self.units = []
        for owner in dict.fromkeys(OWNER):
            coils = capture_round_lens_potential(self.state, str(owner)).potential.coils
            for coil, reference in zip(coils, original[OWNER == owner], strict=True):
                turns = coil.ampere_turns / reference
                np.testing.assert_allclose(turns, round(turns), rtol=0, atol=1e-8)
                np.testing.assert_array_equal(coil.registration.rotation_array, np.eye(3))
                np.testing.assert_array_equal(coil.registration.origin_global_m, np.zeros(3))
                self.units.append(replace(coil, ampere_turns=float(round(turns))))
        self.before = original.copy()
        self.before[ACTIVE] = FINAL["fixed_vacuum_currents_A"]
        # Both corrector transfer pairs, ADL and the objective are owned by their FEM solutions.
        self.replaced = ~np.isin(OWNER, (*LENSES, "mini_condenser", *PROJECTION))
        self.before[self.replaced] = 0.0
        for file in (MAT, OBJ):
            check_file(file, RECEIPT["evidence_sha256"][file.name])
        check_file(FEM, RECEIPT["tracking_field_sha256"])
        with np.load(OBJ) as f:
            r, z, a = f["r"], f["z"], f["a_phi"]
            radius = 0.00288
            inside = (r > 0) & (r <= radius * (1 + 1e-14))
            vand = np.polynomial.polynomial.polyvander((r[inside] / radius) ** 2, 3)
            coeff = np.linalg.lstsq(vand, a[inside] / r[inside, None], rcond=None)[0]
            self.obj = make_interp_spline(z, coeff.T, k=3)
            self.factors = (
                np.sign(FINAL["objective_At"])
                * np.array([2.0, -16.0, 384.0, -18432.0])
                / radius ** np.arange(0, 8, 2)
            )
        with np.load(FEM) as f:
            mid = len(f["xy_m"]) // 2
            self.zjoint, self.bjoint = f["z_local_m"] + 0.45, f["B_zyx_T"][:, mid, mid]

    @cached_property
    def electric(self):
        return capture_instrument_electric_field(self.state)

    @cached_property
    def physical_key(self):
        from temsim.physics.closed_gun_field import closed_field_request
        from temsim.physics.ray_identity import emission_reference
        from temsim.optics.electron_gun import tracing
        from temsim.vacuum import bind_gun_environment
        from temsim.physics.instrument_electric import (
            configure_instrument_electric_domain,
            instrument_electric_end_mm,
        )

        gun = self.state.electron_gun
        bind_gun_environment(self.state)
        configure_instrument_electric_domain(gun, instrument_electric_end_mm(self.state))
        body = dict(
            units=[c.to_metadata() for c in self.units],
            joint=sha(MAT),
            objective=sha(OBJ),
            table=sha(FEM),
            electrostatic=self.electric.numerical_identity,
            gun_electrodes=closed_field_request(gun),
            gun_apertures=[
                [p.z_mm, p.radius_mm, p.offset_x_mm, p.offset_y_mm, p.enabled]
                for p in (gun.dpa_aperture, gun.c1_aperture)
            ],
            gun_step=gun.trace_step_mm,
            rate=gun.emitted_current_a,
            gun_solver=sha(type(HERE)(tracing.__file__)),
            field_evaluator=sha(HERE / "condenser_source_fields.py"),
            source_solver=sha(HERE / "condenser_source_gun.py"),
        )
        h = hashlib.sha256(json.dumps(body, sort_keys=True).encode())
        for arr in (
            *emission_reference(self.emission, None).values(),
            self.emission.energy_offset_ev,
            self.emission.weight,
        ):
            if isinstance(arr, np.ndarray):
                h.update(np.ascontiguousarray(arr))
        return h.hexdigest()

    def gun_key(self, q, n):
        h = hashlib.sha256(self.physical_key.encode())
        h.update(np.asarray(q, np.float64))
        h.update(str(n).encode())
        return h.hexdigest()

    def current(self, i1, i2):
        q = self.before.copy()
        q[:4], q[4:8] = i1, i2
        return q

    def round_axis(self, z):
        return np.asarray([closed_form(coil, z) for coil in self.units])

    def objective(self, z):
        field = np.zeros((len(z), 12))
        field[:, 0:8:2] = self.obj(z) * self.factors
        field[:, 1:8:2] = self.obj(z, nu=1) * self.factors
        return field

    @cached_property
    def far_corrector(self):
        # Use the same converged winding and volume magnetization outside the archived bore table.
        z = np.linspace(-0.001, 0.453, 228)
        with np.load(MAT) as f:
            solution = {k: f[k] for k in ("corners_m", "moments_Am2", "recipe", "report")}
        delta = 3e-5
        xy = np.array([[0, 0], [delta, 0], [-delta, 0], [0, delta], [0, -delta]])
        points = np.zeros((len(z), len(xy), 3))
        points[:, :, :2] = xy
        points[:, :, 2] = z[:, None] - 0.45
        b = vacuum_field(points.reshape(-1, 3), solution, quadrature=1).reshape(points.shape)
        return CubicSpline(
            z,
            np.stack(
                [b[:, 0], (b[:, 1] - b[:, 2]) / (2 * delta), (b[:, 3] - b[:, 4]) / (2 * delta)], axis=1
            ),
        )

    def gun_field(self, q):
        return GunMagnet(self, q)


class GunMagnet:
    """Smooth upstream field from the active round coils, objective and exterior joint solution."""

    def __init__(self, fields, current):
        z = np.linspace(-0.002, 0.455, 1829)
        axial = make_interp_spline(z, current @ fields.round_axis(z) + fields.objective(z)[:, 0], k=5)
        values = np.column_stack(
            [axial(z, nu=i) for i in range(5)] + [fields.far_corrector(z).reshape(len(z), 9)]
        )
        self.dz = z[1] - z[0]
        self.coeff = np.ascontiguousarray(CubicSpline(z, values).c.transpose(1, 2, 0))

    def field_at_global_positions_t(self, points):
        return evaluate_gun(
            np.asarray(points, dtype=float).reshape(-1, 3), self.coeff, self.dz
        ).reshape(np.shape(points))


@njit(cache=True)
def evaluate_gun(points, coeff, dz):
    field = np.empty_like(points)
    u = np.empty(14)
    for j in range(len(points)):
        x, y, z = points[j]
        t = z + 0.002
        i = int(np.floor(t / dz))
        if i < 0 or i >= len(coeff) or x * x + y * y > 50e-6**2:
            raise ValueError("Gun query outside the verified magnetic domain")
        t -= i * dz
        for k in range(14):
            a, b, c, d = coeff[i, k]
            u[k] = ((a * t + b) * t + c) * t + d
        s = x * x + y * y
        radial = -0.5 * u[1] + s * u[3] / 16
        field[j, 0] = x * radial + u[5] + x * u[8] + y * u[11]
        field[j, 1] = y * radial + u[6] + x * u[9] + y * u[12]
        field[j, 2] = u[0] - s * u[2] / 4 + s * s * u[4] / 64 + u[7] + x * u[10] + y * u[13]
    return field
