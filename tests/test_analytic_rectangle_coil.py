"""CPU-only pack and finite-path mathematics; no source, transport or wave."""
from dataclasses import FrozenInstanceError, replace
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import mu_0

from temsim.physics.analytic_rectangle_coil import (
    AnalyticRectanglePack, analytic_rectangle_implementation_identity,
    pack_analytic_rectangle_potential,
)
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.transverse_coil_field import TransverseCoilPotential


@pytest.fixture(autouse=True)
def no_physical_execution(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Analytic rectangle tests must not emit, trace or propagate a source")
    monkeypatch.setattr("temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit", forbidden)
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", forbidden)


def rectangle(**changes):
    return dict(dict(family="radial_rectangle", coil_index=1, station_index=0,
        group="x", current_sign=1, azimuth_deg=0., center_z_mm=0.,
        radial_start_mm=19.9, radial_end_mm=20.1, outer_width_mm=24.2,
        outer_height_mm=24.2, window_width_mm=24., window_height_mm=24.), **changes)


def potential(*, coils=None, **changes):
    if coils is None:
        coils = (rectangle(), rectangle(coil_index=2, group="y", current_sign=-1, azimuth_deg=90.))
    return TransverseCoilPotential(**dict(dict(source_id="mathematical-rectangle",
        coils=coils, turn_count_per_coil=7, currents_a=(1.7, -.3)), **changes))


def analytic_field(pack, points):
    """Independent CPU reduction of exact finite-segment path integrals."""
    result = []
    displacement = pack.ends_m-pack.starts_m
    length = np.linalg.norm(displacement, axis=1)
    for point in points:
        r1, r2 = point-pack.starts_m, point-pack.ends_m
        n1, n2 = np.linalg.norm(r1, axis=1), np.linalg.norm(r2, axis=1)
        aa = displacement*(pack.weights_t_m*np.log1p(2*length/(n1+n2-length))/length)[:, None]
        bb = np.cross(displacement, r1)*(pack.weights_t_m*(n1+n2)/
            (n1*n2*(n1*n2+np.einsum("ni,ni->n", r1, r2))))[:, None]
        result.append([math.fsum(aa[:, axis]) for axis in range(3)]
                      +[math.fsum(bb[:, axis]) for axis in range(3)])
    return np.asarray(result)


def independent_gauss_field(pack, points, order=96):
    """Independent high-order integration along the same saved endpoints."""
    t, w = np.polynomial.legendre.leggauss(order)
    t, w = (t+1)/2, w/2
    displacement = pack.ends_m-pack.starts_m
    nodes = (pack.starts_m[:, None]+t[None, :, None]*displacement[:, None]).reshape(-1, 3)
    dl = (displacement[:, None]*w[None, :, None]*pack.weights_t_m[:, None, None]).reshape(-1, 3)
    result = []
    for point in points:
        r = point-nodes
        norm2 = np.einsum("ni,ni->n", r, r)
        inv = 1/np.sqrt(norm2)
        aa, bb = dl*inv[:, None], np.cross(dl, r)*(inv/norm2)[:, None]
        result.append([math.fsum(aa[:, axis]) for axis in range(3)]
                      +[math.fsum(bb[:, axis]) for axis in range(3)])
    return np.asarray(result)


def test_default_pack_count_orders_and_reference_line_nodes_are_retained():
    from temsim.immutable_json import thaw_json
    from temsim.physics.transverse_coil_field import _quadrature, _unit_nodes
    value = potential()
    packed = pack_analytic_rectangle_potential(value)
    assert packed is not None
    assert packed.segment_count == 2*12*12*4
    assert packed.original_path_node_count == packed.segment_count*24
    assert (packed.pack_order, packed.reference_path_order) == (12, 24)
    t, w = _quadrature(24, 0., 1.)
    delta = packed.ends_m-packed.starts_m
    reconstructed = (packed.starts_m[:, None]+t[None, :, None]*delta[:, None]).reshape(-1, 3)
    weighted = (w[None, :, None]*delta[:, None]*packed.weights_t_m[:, None, None]).reshape(-1, 3)
    coils_json = json.dumps(thaw_json(value.coils), sort_keys=True, separators=(",", ":"), allow_nan=False)
    native_nodes, native_dl, groups = _unit_nodes(coils_json, 7, 12, 24)
    native_weighted = native_dl*np.asarray(value.currents_a)[groups, None]
    active = np.any(native_weighted != 0., axis=1)
    np.testing.assert_array_equal(reconstructed, native_nodes[active])
    np.testing.assert_allclose(weighted, native_weighted[active], rtol=5.e-16, atol=1.e-27)


def test_closed_paths_keep_each_group_signed_NI_and_SI_coordinates():
    value = potential(coils=(rectangle(center_z_mm=900.), rectangle(coil_index=2,
        group="y", current_sign=-1, azimuth_deg=90., center_z_mm=900.)))
    packed = pack_analytic_rectangle_potential(value)
    for first in range(0, packed.segment_count, 4):
        np.testing.assert_array_equal(packed.ends_m[first:first+4], np.roll(packed.starts_m[first:first+4], -1, axis=0))
        np.testing.assert_array_equal(packed.weights_t_m[first:first+4], np.repeat(packed.weights_t_m[first], 4))
    count_per_coil = 12*12*4
    for index, (current, sign) in enumerate(zip(value.currents_a, (1, -1))):
        actual = math.fsum(packed.weights_t_m[index*count_per_coil:(index+1)*count_per_coil:4])
        np.testing.assert_allclose(actual, 7*current*sign*mu_0/(4*math.pi), rtol=3.e-15, atol=0.)
    np.testing.assert_allclose(packed.starts_m[:4, 2].mean(), .9, rtol=0., atol=2.e-16)


def test_minimum_radius_uses_segment_interior_instead_of_endpoints():
    from temsim.physics.transverse_coil_field import _quadrature
    value = potential(coils=(rectangle(),))
    packed = pack_analytic_rectangle_potential(value)
    radius, _ = _quadrature(12, 19.9*.001, 20.1*.001)
    np.testing.assert_allclose(packed.minimum_radius_m, radius.min(), rtol=0., atol=2.e-18)
    assert packed.minimum_radius_m < np.hypot(packed.starts_m[:, 0], packed.starts_m[:, 1]).min()
    assert packed.maximum_length_m > .024


def test_pack_is_frozen_and_array_storage_cannot_be_reenabled():
    packed = pack_analytic_rectangle_potential(potential())
    with pytest.raises(FrozenInstanceError):
        packed.maximum_length_m = 1.
    for array in (packed.starts_m, packed.ends_m, packed.weights_t_m):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_all_zero_current_has_empty_active_pack_and_no_fictitious_nodes():
    packed = pack_analytic_rectangle_potential(potential(currents_a=(0., 0.)))
    assert packed is not None and packed.segment_count == packed.original_path_node_count == 0
    assert packed.starts_m.shape == packed.ends_m.shape == (0, 3)
    assert packed.weights_t_m.shape == (0,)
    assert math.isinf(packed.minimum_radius_m) and packed.maximum_length_m == 0.


@pytest.mark.parametrize("path_order", [2, 8, 23])
def test_explicit_low_reference_path_order_keeps_original_evaluator(path_order):
    assert pack_analytic_rectangle_potential(potential(path_order=path_order)) is None


@pytest.mark.parametrize("path_order", [24, 32, 64])
def test_supported_high_reference_order_keeps_pack_geometry_and_NI(path_order):
    first = pack_analytic_rectangle_potential(potential())
    other = pack_analytic_rectangle_potential(potential(path_order=path_order))
    np.testing.assert_array_equal(other.starts_m, first.starts_m)
    np.testing.assert_array_equal(other.ends_m, first.ends_m)
    np.testing.assert_array_equal(other.weights_t_m, first.weights_t_m)
    assert other.original_path_node_count == other.segment_count*path_order


def test_saddle_pose_unknown_descriptor_and_overridden_law_decline(monkeypatch):
    saddle = dict(family="saddle", coil_index=1, station_index=0, group="x", current_sign=1,
        azimuth_deg=0., center_z_mm=0., start_z_mm=-5., end_z_mm=5., inner_radius_mm=10.,
        outer_radius_mm=12., span_deg=60., band_mm=1., window_half_angle_rad=.2)
    assert pack_analytic_rectangle_potential(potential(coils=(saddle,))) is None
    assert pack_analytic_rectangle_potential(potential(registration=CoordinateRegistration((1.e-6, 0., 0.)))) is None
    assert pack_analytic_rectangle_potential(potential(coils=(rectangle(future_descriptor=1.),))) is None
    assert pack_analytic_rectangle_potential(SimpleNamespace()) is None
    import temsim.physics.transverse_coil_field as module
    monkeypatch.setattr(module, "_rotation", lambda _angle: np.eye(3))
    assert pack_analytic_rectangle_potential(potential()) is None


@pytest.mark.parametrize("coil", [
    rectangle(outer_height_mm=400., window_height_mm=390.),
    rectangle(radial_start_mm=.0001, radial_end_mm=.0002),
    rectangle(center_z_mm=1.e8),
    rectangle(center_z_mm=1.e6, outer_height_mm=2.e-15, window_height_mm=1.e-15),
])
def test_aspect_radius_global_range_and_collapsed_float_sides_decline(coil):
    assert pack_analytic_rectangle_potential(potential(coils=(coil,))) is None


@pytest.mark.parametrize("current", [1.e200, 1.e-200])
def test_extreme_finite_weight_preserves_original_numerical_fallback(current):
    assert pack_analytic_rectangle_potential(potential(currents_a=(current, 0.))) is None


def test_original_quadrature_allocation_budget_is_not_bypassed():
    coils = tuple(rectangle(coil_index=index+1, azimuth_deg=index*60.) for index in range(6))
    assert pack_analytic_rectangle_potential(potential(coils=coils, pack_order=64, path_order=64)) is None


def test_complete_hexapole_axis_identity_requires_the_existing_strict_proof():
    coils = tuple(rectangle(coil_index=i+1, azimuth_deg=30.+i*60., current_sign=(-1)**i) for i in range(6))
    value = potential(coils=coils, currents_a=(-1.e5, 0.), turn_count_per_coil=2,
                      provenance={"multipole_order": 3})
    packed = pack_analytic_rectangle_potential(value)
    assert packed.axis_zero and packed.segment_count == 3456 and packed.original_path_node_count == 82944
    assert not pack_analytic_rectangle_potential(replace(value, provenance={})).axis_zero
    broken = (dict(coils[0], current_sign=-1), *coils[1:])
    assert not pack_analytic_rectangle_potential(replace(value, coils=broken)).axis_zero


def test_finite_path_formula_matches_independent_96_point_line_integrals():
    packed = pack_analytic_rectangle_potential(potential(pack_order=2))
    points = np.array([[.001, -.002, .003], [-.003, .002, -.001], [.002, -.001, .3]])
    analytic = analytic_field(packed, points)
    integrated = independent_gauss_field(packed, points)
    np.testing.assert_allclose(analytic[:, :3], integrated[:, :3], rtol=2.e-12, atol=1.e-20)
    np.testing.assert_allclose(analytic[:, 3:], integrated[:, 3:], rtol=2.e-12, atol=1.e-17)
    assert np.linalg.norm(analytic[-1, 3:]) > 0.


def test_implementation_identity_declares_changed_path_and_fixed_pack():
    identity = analytic_rectangle_implementation_identity()
    assert identity["schema"] == "analytic-rectangle-gauss-pack-v1"
    assert len(identity["implementation_sha256"]) == 64
    assert "Analytic finite straight segments" in identity["path"]
    assert identity["reference_path_order_minimum"] == 24
    with pytest.raises(TypeError):
        identity["schema"] = "changed"
