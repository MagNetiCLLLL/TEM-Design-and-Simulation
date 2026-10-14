"""First-order reference retains executed focusing, source modes and stops."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.column_wave import _column_transports, _prepare_column, _propagate_column
from temsim.physics.ideal_column_reference import (
    REFERENCE_MODEL_VERSION, column_plan_for_model, column_plan_record,
)
from temsim.physics.intrinsic_aberrations import FrozenIntrinsicAberrationKick


@pytest.fixture(scope="module")
def captured_plans():
    from temsim.optics.column import default_state
    state = default_state()
    state.lens_intrinsic_aberrations = {"probe_tl22_lens": {
        "c1_mm": .002, "a1_mm": .003, "a1_azimuth_deg": 17., "c5_mm": .4}}
    physical = _prepare_column(state, 1000., 1370., 1.)
    reference = _prepare_column(state, 1000., 1370., 1., column_model="first_order_reference")
    return state, physical, reference


def test_native_reference_keeps_physical_grid_hardware_and_focus(captured_plans):
    state, physical, reference = captured_plans
    actual, ideal = physical[0], reference[0]
    assert np.any(actual.cs_kick_m3)
    assert np.any(actual.midpoint_hex_normal_m3) or np.any(actual.midpoint_hex_skew_m3)
    assert np.any(actual.midpoint_magnetic_t)
    for name in ("z_mm", "step_m", "magnetic_t", "midpoint_magnetic_t", "sx_m2", "sy_m2",
                 "sxy_m2", "midpoint_sx_m2", "midpoint_sy_m2", "midpoint_sxy_m2",
                 "kick_x_rad", "kick_y_rad", "dipole_bx_t", "dipole_by_t", "thin_power_m1",
                 "thin_rotation_rad", "save_index", "checkpoint_index"):
        np.testing.assert_array_equal(getattr(ideal, name), getattr(actual, name))
    np.testing.assert_array_equal(reference[1], physical[1])
    assert reference[2] == physical[2] and reference[2]  # Actual condenser aperture remains.
    assert reference[3] == physical[3]
    assert ideal.electric_field_identity == actual.electric_field_identity
    for name in ("cs_kick_m3", "hex_normal_m3", "hex_skew_m3",
                 "midpoint_hex_normal_m3", "midpoint_hex_skew_m3"):
        assert not np.any(getattr(ideal, name))
        assert not getattr(ideal, name).flags.writeable
    assert ideal.hexapole_exit_nodes == () and ideal.posed_spherical_kicks == ()
    coefficients = dict(ideal.intrinsic_aberration_kicks[0].coefficients)
    assert coefficients["c1_mm"] == .002
    assert all(value == 0. for key, value in coefficients.items() if key != "c1_mm")
    assert dict(actual.intrinsic_aberration_kicks[0].coefficients)["a1_mm"] == .003
    assert state.lens_intrinsic_aberrations["probe_tl22_lens"]["a1_mm"] == .003
    assert ideal.signature != actual.signature
    assert REFERENCE_MODEL_VERSION in ideal.solver_signature
    assert column_plan_for_model(ideal, "first_order_reference") is ideal
    assert column_plan_for_model(actual, "physical") is actual
    assert "intrinsic C1 defocus" in column_plan_record(ideal)["preserved"]


def test_reference_preserves_each_energy_linear_transport(captured_plans):
    _, physical, reference = captured_plans
    by_energy = []
    for energy in (280., 320.):
        actual_paths = tuple(_column_transports(physical[0], energy))
        ideal_paths = tuple(_column_transports(reference[0], energy))
        for actual, ideal in zip(actual_paths, ideal_paths):
            np.testing.assert_array_equal(ideal.matrix, actual.matrix)
            np.testing.assert_array_equal(ideal.offset, actual.offset)
            assert ideal.reference_phase_rad == actual.reference_phase_rad
        by_energy.append(ideal_paths)
    assert any(not np.array_equal(a.matrix, b.matrix) for a, b in zip(*by_energy))


def test_reference_refuses_real_posed_lens_without_affecting_physical_route():
    from test_tilted_column_wave import _objective
    state = _objective(angle_mrad=1.)
    z = state.lenses[0].z_mm
    assert _prepare_column(state, z-.1, z+.1, .1)[0].mapped_fields
    with pytest.raises(ValueError, match="mapped or posed magnetic fields"):
        _prepare_column(state, z-.1, z+.1, .1, column_model="first_order_reference")


@pytest.mark.parametrize("span", ("incident", "bcor"))
def test_combined_probe_corrector_bcor_reference_is_available_on_real_geometry(span):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    state, catalog = default_state(), AssemblyCatalog()
    selection = replace(catalog.default_selection(), column="C3 + Probe Corrector + B-COR")
    catalog.apply(state, selection)
    if span == "incident":
        start, stop = state.electron_gun.exit_plane_z_mm, state.sample.z_mm-.1
    else:
        start = state.sample.z_mm+.1
        stop = max(lens.z_mm for lens in state.lenses if lens.key.startswith("image_"))+20.
    actual = _prepare_column(state, start, stop, 1.)
    ideal = _prepare_column(state, start, stop, 1., column_model="first_order_reference")
    assert not actual[0].mapped_fields and not ideal[0].mapped_fields
    assert np.any(actual[0].midpoint_hex_normal_m3) or np.any(actual[0].midpoint_hex_skew_m3)
    assert np.any(ideal[0].midpoint_magnetic_t)
    assert not np.any(ideal[0].midpoint_hex_normal_m3)
    assert not np.any(ideal[0].midpoint_hex_skew_m3)
    np.testing.assert_array_equal(ideal[0].z_mm, actual[0].z_mm)
    assert state.image_corrector_design == "bcor"


@pytest.mark.parametrize("segmented", (False, True))
def test_reference_executes_c1_and_preserves_mode_weights_and_energy(tmp_path, segmented):
    """An independent quadratic C1 law is retained after removing Cs/HP/A1."""
    from test_tip_wave_pipeline import _quiet_prepared_column
    from test_wave_detector_readout import checkpoint
    from temsim.physics.column_wave import _propagate_column_segmented
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.wave_grid import WaveGridNumerics
    from temsim.physics.wave_reference import AxialWaveReference
    source = checkpoint(two=True)
    source = replace(source, beam=replace(source.beam, modes=tuple(
        replace(mode, energy_kev=energy, axial_reference=AxialWaveReference(1e-8, 1e-22))
        for mode, energy in zip(source.beam.modes, (280., 320.)))))
    start, stop = source.plane_z_mm, source.plane_z_mm+.01
    base = _quiet_prepared_column(None, start, stop, .02)
    base[0].magnetic_t = np.zeros(2)
    flawed = SimpleNamespace(**{**vars(base[0]),
        "cs_kick_m3": np.array((0., 1e6)),
        "midpoint_hex_normal_m3": np.array((1e7,)),
        "intrinsic_aberration_kicks": (FrozenIntrinsicAberrationKick(
            "declared-lens", stop*1e-3, .01, (("c1_mm", .001), ("a1_mm", .002), ("c5_mm", .1))),)})
    prepared = (flawed, *base[1:])
    numerics = WaveGridNumerics(compute_backend="CPU")
    vacuum = _propagate_column(SimpleNamespace(), source, stop, _prepared=base, grid_numerics=numerics)
    if segmented:
        output, _ = _propagate_column_segmented(SimpleNamespace(), source, stop,
            _prepared=prepared, grid_numerics=numerics, segment_steps=1,
            store=ExecutedWaveStore(tmp_path, "reference-c1", 1 << 26),
            column_model="first_order_reference")
    else:
        output = _propagate_column(SimpleNamespace(), source, stop, _prepared=prepared,
            grid_numerics=numerics, column_model="first_order_reference")
    for original, before_c1, after in zip(source.beam.modes, vacuum.beam.modes, output.beam.modes):
        assert after.mode_id == original.mode_id
        assert after.energy_kev == original.energy_kev
        assert after.weight_per_reference_electron == pytest.approx(original.weight_per_reference_electron)
        assert after.axial_reference == before_c1.axial_reference
        np.testing.assert_allclose(after.plane.curvature_m1-before_c1.plane.curvature_m1,
                                   -np.eye(2)*(.001*1e-3/.01**2), atol=1e-14)
        np.testing.assert_allclose(after.plane.amplitude, before_c1.plane.amplitude, atol=1e-14)
    assert output.record["column_model"] == "first_order_reference"
    assert output.record["model_version"] == REFERENCE_MODEL_VERSION


@pytest.mark.parametrize("field", ("electric", "dipole"))
def test_radial_route_preserves_input_for_electric_or_finite_dipole(monkeypatch, field):
    from temsim.physics.radial_column_wave import _round_column_prefix
    plan = SimpleNamespace(electric_field=object() if field == "electric" else None,
        dipole_bx_t=np.array((.1 if field == "dipole" else 0.,)), dipole_by_t=np.zeros(1))
    monkeypatch.setattr("temsim.physics.column_wave._prepare_column", lambda *a: (plan, (), {}, ()))
    source = SimpleNamespace(plane_z_mm=1.)
    z, modes, record = _round_column_prefix(SimpleNamespace(), source, 2.)
    assert z == source.plane_z_mm and modes == ()
    assert "two-dimensional column operator" in record["reason"]


def test_radial_reference_uses_transformed_plan_and_same_energy_modes(monkeypatch):
    from scipy.constants import e
    from temsim.physics import radial_column_wave as radial
    from temsim.physics.tip_gun_wave import _momentum_velocity
    from test_tip_wave_pipeline import _quiet_prepared_column
    prepared = _quiet_prepared_column(None, 0., 1., 1.)
    prepared[0].midpoint_magnetic_t = np.array((.2,))
    prepared[0].cs_kick_m3 = np.array((0., 1e12))
    seen = []
    def prepare(*args, column_model="physical"):
        seen.append(column_model)
        return (column_plan_for_model(prepared[0], column_model), *prepared[1:])
    monkeypatch.setattr("temsim.physics.column_wave._prepare_column", prepare)
    monkeypatch.setattr(radial, "radial_cs", lambda *a: pytest.fail("Reference must not execute Cs"))
    radius = np.geomspace(1e-12, 1e-4, 32768, endpoint=False)
    sigma = 1e-6
    wave = radial.RadialWave(radius, np.exp(-radius**2/(2*sigma*sigma))/(np.sqrt(np.pi)*sigma))
    monkeypatch.setattr(radial, "from_executed_gun", lambda *a: wave)
    modes = tuple(SimpleNamespace(mode_id=f"energy-{energy}", energy_kev=energy,
        weight_per_reference_electron=weight) for energy, weight in ((280., .4), (320., .6)))
    source = SimpleNamespace(plane_z_mm=0., digest="isolated-radial-reference",
        beam=SimpleNamespace(modes=modes), record={"radial_output_modes":
            tuple({"mode_id": mode.mode_id} for mode in modes)})
    z, output, receipt = radial._round_column_prefix(SimpleNamespace(), source, 1.,
        column_model="first_order_reference")
    assert seen == ["first_order_reference"] and z == 1.
    for (mode, propagated), original in zip(output, modes):
        assert mode is original
        momentum = float(_momentum_velocity(mode.energy_kev*1000)[0])
        g = -e*.2/(2*momentum)
        assert propagated.curvature_m1 == pytest.approx(-g*np.tan(g*.001), rel=1e-12)
        assert propagated.probability == pytest.approx(wave.probability, rel=2e-12)
    assert receipt["column_model"] == "first_order_reference"
