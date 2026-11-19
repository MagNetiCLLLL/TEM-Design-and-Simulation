"""Source-free exact-node field and Gun composition checks; no wave trials."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from temsim.immutable_json import thaw_json
from temsim.magnetic_field_scene import _Source
from temsim.physics.compiled_copper_instrument import pack_copper_instrument_sources, copper_instrument_batch
from temsim.physics.compiled_transverse_coil import (
    compiled_transverse_field, pack_transverse_source, transverse_b_batch,
)
from temsim.physics.frozen_copper_coil_field import is_all_z_copper_field
from temsim.physics.instrument_magnetic import InstrumentMagneticField
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.transverse_coil_field import TransverseCoilPotential, FrozenTransverseCoilField, _unit_nodes


@pytest.fixture(autouse=True)
def no_physical_source_or_wave(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Pure field/Gun operator tests must not emit or trace a physical source")
    monkeypatch.setattr("temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit", forbidden)
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", forbidden)


def rectangle(**changes):
    return dict(dict(family="radial_rectangle", coil_index=1, station_index=0,
        group="x", current_sign=1, azimuth_deg=0., center_z_mm=0.,
        radial_start_mm=19.9, radial_end_mm=20.1, outer_width_mm=24.2,
        outer_height_mm=24.2, window_width_mm=24., window_height_mm=24.), **changes)


def winding(*, family="rectangle", currents=(1.7, -.3), **changes):
    if family == "rectangle":
        coils = (rectangle(), rectangle(coil_index=2, group="y", azimuth_deg=90.))
    else:
        one = dict(family="saddle", coil_index=1, station_index=0, group="x",
            current_sign=1, azimuth_deg=0., center_z_mm=0., start_z_mm=-5.,
            end_z_mm=5., inner_radius_mm=10., outer_radius_mm=12.,
            span_deg=60., band_mm=1., window_half_angle_rad=.2)
        coils = (one, dict(one, coil_index=2, group="y", azimuth_deg=90.))
    potential = TransverseCoilPotential(**dict(dict(source_id="synthetic-winding", coils=coils,
        turn_count_per_coil=7, currents_a=currents), **changes))
    return FrozenTransverseCoilField(potential, potential.source_id, 3.e-22)


def source(field):
    return _Source(field.key, field, field.bounds_m, field.radial_support_m)


def native_field(provider, points):
    """Independent source-order oracle bypasses the new frozen B adapter."""
    total = np.zeros_like(points)
    for row in provider._sources:
        if row.known_zero:
            continue
        active = (np.ones(len(points), bool) if is_all_z_copper_field(row.provider)
                  else ((points[:, 2] >= row.bounds_m[0, 2]) & (points[:, 2] <= row.bounds_m[1, 2])))
        if active.any():
            field = row.provider
            value = (field.potential.field_at_global_positions_t(points[active])
                     if type(field) is FrozenTransverseCoilField
                     else field.field_at_global_positions_t(points[active]))
            total[active] += value
    return total


def test_pack_keeps_every_nonzero_original_node_order_and_default_quadrature():
    field = winding(); potential = field.potential
    nodes, weighted, geometry, zero = pack_transverse_source(field)
    raw = json.dumps(thaw_json(potential.coils), sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected_nodes, dl, groups = _unit_nodes(raw, 7, 12, 24)
    expected_weights = dl*np.asarray(potential.currents_a)[groups, None]
    active = np.any(expected_weights != 0., axis=1)
    np.testing.assert_array_equal(nodes, expected_nodes[active])
    np.testing.assert_array_equal(weighted, expected_weights[active])
    assert len(nodes) == 2*12*12*4*24 and not zero
    assert pack_transverse_source(field)[0] is nodes
    for array in (nodes, weighted, geometry):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("family", ["rectangle", "saddle"])
def test_compiled_field_matches_reference_at_scalar_batched_and_full_tail_points(family):
    field = winding(family=family)
    points = np.array([[0., 0., 0.], [1.e-5, -2.e-5, .001], [-2.e-5, 3.e-5, .04],
                       [1.e-8, -2.e-8, 1.6], [1.e-8, 2.e-8, -1.]])
    actual = compiled_transverse_field(field, points)
    assert actual is not None
    expected = field.potential.field_at_global_positions_t(points)
    np.testing.assert_allclose(actual, expected, rtol=5.e-14, atol=1.e-21)
    np.testing.assert_allclose(compiled_transverse_field(field, points[1]), expected[1], rtol=5.e-14, atol=1.e-21)
    assert np.linalg.norm(actual[-1]) > 0.
    np.testing.assert_array_equal(compiled_transverse_field(field, np.empty((0, 3))), np.empty((0, 3)))


def test_zero_currents_and_exact_complete_hexapole_axis_keep_cancellation():
    field = winding(currents=(0., 0.))
    points = np.array([[0., 0., 0.], [1.e-8, 0., 2.]])
    np.testing.assert_array_equal(compiled_transverse_field(field, points), np.zeros_like(points))
    coils = tuple(rectangle(coil_index=i+1, azimuth_deg=30.+i*60., current_sign=(-1)**i) for i in range(6))
    potential = TransverseCoilPotential("complete-hexapole", coils, 2, (-1.e5, 0.),
                                       provenance={"multipole_order": 3})
    field = FrozenTransverseCoilField(potential, potential.source_id, 3.e-22)
    points = np.array([[0., 0., 0.], [0., 0., .04], [0., 0., 1.6],
                       [1.e-8, -2.e-8, 0.], [1.e-5, 2.e-5, .001]])
    expected = potential.field_at_global_positions_t(points)
    actual = compiled_transverse_field(field, points)
    np.testing.assert_array_equal(actual[:3], np.zeros((3, 3)))
    # Node cancellation is measured with an absolute roundoff allowance.
    np.testing.assert_allclose(actual, expected, rtol=5.e-12, atol=2.e-17)
    assert np.linalg.norm(actual[-1]) > 0.


def test_pose_and_overridden_law_preserve_the_reference_fallback(monkeypatch):
    field = winding(registration=CoordinateRegistration((.001, .002, .003)))
    assert pack_transverse_source(field) is None
    points = np.array([[0., 0., .001]])
    np.testing.assert_array_equal(field.field_at_global_positions_t(points), field.potential.field_at_global_positions_t(points))
    field = winding()
    with monkeypatch.context() as patch:
        patch.setattr(TransverseCoilPotential, "sample", lambda self, p: {"B_t": np.ones_like(p)})
        assert pack_transverse_source(field) is None
    with monkeypatch.context() as patch:
        patch.setattr(FrozenTransverseCoilField, "field_at_global_positions_t", lambda self, p: np.ones_like(p))
        assert pack_transverse_source(field) is None
    with monkeypatch.context() as patch:
        import temsim.physics.transverse_coil_field as module
        patch.setattr(module, "_inside_pack", lambda p, c: np.ones(p.shape[:-1], bool))
        assert pack_transverse_source(field) is None


def test_optional_compiler_unavailable_retains_native_evaluator(monkeypatch):
    import temsim.physics.compiled_transverse_coil as module
    field = winding(); points = np.array([[0., 0., .04]])
    monkeypatch.setattr(module, "njit", None)
    assert pack_transverse_source(field) is None
    np.testing.assert_array_equal(field.field_at_global_positions_t(points), field.potential.sample(points)["B_t"])


@pytest.mark.parametrize("point", [[np.nan, 0., 0.], [np.inf, 0., 0.], [0., 0., 1.e101]])
def test_nonfinite_and_extreme_queries_decline_without_clipping(point):
    field = winding(); points = np.array([point])
    assert compiled_transverse_field(field, points) is None
    assert not transverse_b_batch(points, pack_transverse_source(field))[0]


@pytest.mark.parametrize("point", [[.020, .01205, 0.], [.0199, .01205, 0.], [.020, .012, 0.]])
def test_winding_material_and_boundary_use_native_exclusion(point):
    field = winding(); points = np.array([point])
    assert compiled_transverse_field(field, points) is None
    with pytest.raises(ValueError, match="occupied winding pack"):
        field.field_at_global_positions_t(points)


def test_rounded_boundary_guard_declines_a_native_valid_vacuum_point():
    field = winding()
    points = np.array([[.0199-1.e-16, .01205, 0.]])
    assert compiled_transverse_field(field, points) is None
    assert np.isfinite(field.potential.sample(points)["B_t"]).all()


def test_combined_copper_native_and_transverse_preserve_source_order_and_tails():
    from test_compiled_copper_gun import _copper, _native
    for rows in ((_copper(), source(winding()), _native()),
                 (source(winding()), _native(), _copper())):
        magnetic = InstrumentMagneticField(rows, None, None)
        packed = pack_copper_instrument_sources(rows)
        assert packed is not None
        points = np.array([[1.e-8, -2.e-8, -.01], [0., 0., .00055], [1.e-8, 2.e-8, 1.6]])
        admitted, actual = copper_instrument_batch(points, packed)
        assert admitted
        np.testing.assert_allclose(actual, native_field(magnetic, points), rtol=5.e-14, atol=1.e-20)
        assert all(not value.flags.writeable for value in packed)


def test_transverse_only_combination_is_packable_and_pose_declines():
    rows = (source(winding()),)
    packed = pack_copper_instrument_sources(rows)
    assert packed is not None
    points = np.array([[0., 0., .04]])
    admitted, actual = copper_instrument_batch(points, packed)
    assert admitted
    np.testing.assert_allclose(actual, rows[0].provider.potential.sample(points)["B_t"], rtol=5.e-14, atol=1.e-21)
    posed = source(winding(registration=CoordinateRegistration((1.e-6, 0., 0.))))
    assert pack_copper_instrument_sources((posed,)) is None


def test_installed_current_graph_is_compiled_without_any_source_execution():
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field
    from temsim.simulation_modes import switch_mode
    state = default_state(); catalog = AssemblyCatalog(); selection = catalog.default_selection()
    catalog.apply(state, selection); switch_mode(state, "unified")
    apply_operating_mode_pair(state, "nano_probe", "diffraction", column_name=selection.column, recording_name=selection.recording)
    state.round_lens_field_model = "copper_coil"
    magnetic = capture_instrument_magnetic_field(state)
    packed = pack_copper_instrument_sources(magnetic._sources)
    assert packed is not None and len(packed[10]) == 387072
    points = np.array([[1.e-8, -2.e-8, .001], [0., 0., .04], [1.e-8, 2.e-8, 1.5992]])
    admitted, actual = copper_instrument_batch(points, packed)
    assert admitted
    np.testing.assert_allclose(actual, native_field(magnetic, points), rtol=5.e-14, atol=1.e-20)


def test_gun_dg_composes_transverse_current_sources_with_unchanged_tolerances(monkeypatch):
    from test_compiled_copper_gun import _copper, _native, _phase, _electric
    from temsim.physics import grounded_particle_step as compiled
    from temsim.physics.static_energy_lorentz import static_energy_step
    rows = (_copper(), source(winding()), _native())
    magnetic = InstrumentMagneticField(rows, None, None)
    phase, electric, dt = _phase(), _electric(), 1.e-14
    actual = compiled.try_step(phase, dt, magnetic, electric, 1.e-13, 64)
    assert actual is not None
    with monkeypatch.context() as patch:
        patch.setattr(compiled, "try_step", lambda *_a, **_k: None)
        patch.setattr(FrozenTransverseCoilField, "field_at_global_positions_t",
                      lambda self, p: self.potential.field_at_global_positions_t(p))
        reference = static_energy_step(phase, dt, magnetic, electric, tolerance=1.e-13)
    np.testing.assert_allclose(actual.position_m, reference.position_m, rtol=2.e-14, atol=1.e-22)
    np.testing.assert_allclose(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s, rtol=2.e-12, atol=1.e-36)
    assert actual.time_s == reference.time_s == dt
