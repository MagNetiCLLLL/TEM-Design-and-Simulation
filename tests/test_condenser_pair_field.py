from types import SimpleNamespace

import numpy as np
from scipy.special import roots_legendre

from temsim.magnetic_materials import MU0
from temsim.physics.condenser_pair_field import induced_axis, installed_pair, solve_pair


def test_joint_circuit_only_contains_installed_magnetic_parts():
    from temsim.optics.column import default_state

    parts, curves, assignments = installed_pair(default_state())
    poles = {"condenser_lens_1_lower_pole", "condenser_lens_2_upper_pole"}
    yokes = {"condenser_lens_1_yoke", "condenser_lens_2_yoke"}
    assert set(parts) == poles | yokes
    assert {key for key, row in assignments.items() if row["assumed"]} == yokes
    assert all(curves[assignments[key]["index"]]["source_sha256"] for key in parts)
    assert (
        parts["condenser_lens_2_upper_pole"]["start_z_mm"] - parts["condenser_lens_1_lower_pole"]["end_z_mm"]
        == 20
    )


def test_open_bore_recovery_of_a_magnetized_annular_cylinder():
    """Independent end-face solution tests sign, radial weight and absolute scale."""
    a, b, half_length, magnetization = 0.011, 0.029, 0.028, 3.3e5
    nodes, weight = roots_legendre(48)
    r = (a + b) / 2 + (b - a) / 2 * nodes
    z = half_length * nodes
    rr, zz = np.meshgrid(r, z, indexing="ij")
    moment = rr * np.outer(weight * (b - a) / 2, weight * half_length) * magnetization
    at = np.linspace(-0.14, 0.12, 75)

    def cylinder(radius):
        first, last = at + half_length, at - half_length
        return MU0 * magnetization / 2 * (first / np.hypot(first, radius) - last / np.hypot(last, radius))

    actual = induced_axis(at, rr.ravel(), zz.ravel(), np.zeros(rr.size), moment.ravel())
    np.testing.assert_allclose(actual, cylinder(b) - cylinder(a), rtol=1e-10, atol=1e-13)


def test_transverse_magnetization_has_odd_on_axis_field():
    """Ring of radial dipoles with integral M_r r dr dz = 1 A m^2/radian."""
    r = 0.025
    at = np.array([-r, 0.0, r])
    field = induced_axis(at, np.array([r]), np.array([0.0]), np.ones(1), np.zeros(1))
    expected = -1.5 * MU0 * r * at / (r * r + at * at) ** 2.5
    np.testing.assert_allclose(field, expected, rtol=1e-13, atol=0)


def test_zero_drive_yields_no_residual_field_from_previous_operating_points():
    from temsim.optics.column import default_state

    windings = [
        SimpleNamespace(
            center_z_m=z, length_m=length, inner_radius_m=0.048, outer_radius_m=0.078, ampere_turns=turns
        )
        for z, length, turns in ((0.5, 0.09, 300), (0.65, 0.162, 540), (0.915, 0.153, 510))
    ]
    result = solve_pair(default_state(), windings, [0.0, 0.0, 0.0], pitch_mm=5.0)
    np.testing.assert_array_equal(result["induced_T"], 0.0)
    np.testing.assert_array_equal(result["fem_bore_T"], 0.0)
    assert result["peak_material_T"] == 0.0
