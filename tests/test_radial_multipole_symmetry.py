"""Pure-field structural identities; no finite source or wave calculation."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.radial_multipole_symmetry import complete_radial_multipole_axis_symmetry
from temsim.physics.transverse_coil_field import TransverseCoilPotential


def _coils(order=3, groups=1):
    return tuple(dict(
        family="radial_rectangle", coil_index=group*2*order+index+1,
        station_index=0, group="xy"[group], current_sign=(-1)**index,
        azimuth_deg=30.+group*90/order+index*180/order, center_z_mm=4.,
        radial_start_mm=19.9, radial_end_mm=20.1,
        outer_width_mm=12.2, outer_height_mm=14.2,
        window_width_mm=12., window_height_mm=14.,
    ) for group in range(groups) for index in range(2*order))


def _potential(order=3, groups=1, **changes):
    return TransverseCoilPotential(
        "radial_multipole", _coils(order, groups), 2, (134526., -80899.),
        pack_order=4, path_order=8, provenance={"multipole_order": order}, **changes,
    )


@pytest.fixture(autouse=True)
def no_source_or_wave(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Radial symmetry checks must not emit or transport a source")
    monkeypatch.setattr("temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit", forbidden)
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", forbidden)


@pytest.mark.parametrize("order", (2, 3))
@pytest.mark.parametrize("groups", (1, 2))
def test_complete_groups_prove_identity_independently_of_current_sign_and_size(order, groups):
    source = _potential(order, groups)
    assert complete_radial_multipole_axis_symmetry(source.coils, order)
    points = np.array([[0., 0., -.4], [0., 0., .004], [0., 0., .37]])
    for currents in ((134526., -80899.), (-2., 7.), (0., 1.)):
        potential = replace(source, currents_a=currents)
        field, jet = potential.sample(points), potential.sample_jet(points)
        np.testing.assert_array_equal(field["A_t_m"], np.zeros_like(points))
        np.testing.assert_array_equal(field["B_t"], np.zeros_like(points))
        np.testing.assert_array_equal(jet["A_t_m"], np.zeros_like(points))


@pytest.mark.parametrize("order", (2, 3))
def test_axis_identity_follows_a_translated_and_tilted_source_local_axis(order):
    rotation = ((0., 0., 1.), (1., 0., 0.), (0., 1., 0.))
    registration = CoordinateRegistration((.125, -.25, .5), rotation)
    source = _potential(order, 2, registration=registration)
    local = np.array([[0., 0., -.375], [0., 0., .25]])
    global_points = local @ np.asarray(rotation).T + registration.origin_array_m
    np.testing.assert_array_equal(registration.positions_global_to_local_m(global_points), local)
    for value in source.sample(global_points).values():
        np.testing.assert_array_equal(value, np.zeros_like(global_points))


@pytest.mark.parametrize("order", (2, 3))
def test_axis_potential_derivatives_are_retained_with_real_multipole_response(order):
    source = _potential(order, 2)
    points = np.array([[0., 0., .004], [0., 0., .01]])
    actual = source.sample_jet(points)
    # Remove only the proof label in an independent raw integral reference.
    # No geometry, currents, quadrature or numerical tolerances are changed.
    raw = replace(source, provenance={}).sample_jet(points)
    np.testing.assert_array_equal(actual["jac_A_t"], raw["jac_A_t"])
    np.testing.assert_array_equal(actual["hess_A_t_per_m"], raw["hess_A_t_per_m"])
    assert np.linalg.norm(actual["hess_A_t_per_m"]) > 0


@pytest.mark.parametrize("order", (2, 3))
def test_off_axis_values_are_not_projected_or_thresholded(order):
    source = _potential(order, 2)
    points = np.array([[1e-20, 0., .004], [.001, -.002, .01]])
    raw = replace(source, provenance={}).sample(points)
    assert np.any(raw["A_t_m"][0] != 0.)
    for name, actual in source.sample(points).items():
        np.testing.assert_array_equal(actual, raw[name])


@pytest.mark.parametrize("kind", ("missing", "dimension", "angle", "sign", "station", "unknown"))
def test_incomplete_or_nonidentical_geometry_cannot_activate_identity(kind):
    source = _potential(3, 2)
    coils = [dict(row) for row in source.coils]
    if kind == "missing":
        coils.pop()
    elif kind == "dimension":
        coils[-1]["radial_end_mm"] += .001
    elif kind == "angle":
        coils[-1]["azimuth_deg"] = np.nextafter(coils[-1]["azimuth_deg"], np.inf)
    elif kind == "sign":
        coils[-1]["current_sign"] *= -1
    elif kind == "station":
        coils[-1]["station_index"] += 1
    else:
        coils[-1]["unrecognised_geometry"] = 1.
    assert not complete_radial_multipole_axis_symmetry(tuple(coils), 3)
    # Station/unknown metadata need not be admissible to the physical source;
    # the structural check still declines rather than inferring symmetry.
    if kind not in ("station", "unknown"):
        broken = replace(source, coils=tuple(coils))
        points = np.array([[0., 0., .004]])
        for name, value in broken.sample(points).items():
            np.testing.assert_array_equal(value, replace(broken, provenance={}).sample(points)[name])


@pytest.mark.parametrize("order", (None, True, 1, 4, 3.))
def test_multipole_label_without_an_admitted_order_is_not_a_proof(order):
    assert not complete_radial_multipole_axis_symmetry(_coils(), order)
