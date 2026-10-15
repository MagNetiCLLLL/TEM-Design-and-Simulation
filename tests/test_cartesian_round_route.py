"""Routing/admission tests using a declared mathematical fixture, not a gun."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
from temsim.physics.cartesian_radial_handoff import CartesianRadialNumerics, cartesian_to_radial
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.radial_cartesian_handoff import RadialColumnNumerics
from temsim.physics.round_column_checkpoint import execute_round_prefix
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_grid import WaveGridNumerics, WaveMemoryBudgetError
from temsim.physics.wave_reference import AxialWaveReference


def fixture(two=True):
    pitch, sigma = 4e-7, 1.5e-6
    axis = (np.arange(64)-32)*pitch
    radius2 = (axis[:, None]**2+axis[None, :]**2)/sigma**2
    amplitude = ((1-.12*radius2)+.025j*radius2**2)*np.exp(-radius2/2+.73j)*pitch/sigma/np.sqrt(np.pi*.8038)
    plane = PlaneWave(amplitude, np.eye(2)*pitch, np.zeros(2))
    mode = WaveMode(plane, .3, TIP_REFERENCE, "executed-fixture-E0", 300.,
                    AxialWaveReference(1e-9, 1e-25), ({"event": "retained-fixture-history"},))
    modes = (mode, replace(mode, mode_id="executed-fixture-E1", energy_kev=299., weight_per_reference_electron=.2)) if two else (mode,)
    return TipGunCheckpoint(BeamState(modes, TIP_REFERENCE), 450., 1e-9,
        {"schema": "mathematical-executed-fixture-not-a-source"})


def state():
    return SimpleNamespace(electron_gun=SimpleNamespace(exit_plane_z_mm=450.))


def numerics():
    return RadialColumnNumerics(initial_samples=4096, initial_cartesian_pixels=32)


def test_all_modes_qualify_before_round_execution_and_keep_provenance(monkeypatch):
    import temsim.physics.round_column_checkpoint as adapter
    checkpoint = fixture()
    called = []
    def transport(_, incoming, stop, **kwargs):
        called.append(kwargs)
        assert [row["mode_id"] for row in kwargs["cartesian_admission"]] == [m.mode_id for m in incoming.beam.modes]
        assert kwargs["maximum_working_bytes"] == 256*1024**2
        assert kwargs["retained_bytes"] >= sum(m.plane.amplitude.nbytes for m in incoming.beam.modes)
        values = tuple((mode, cartesian_to_radial(mode.plane, wavelength_m=2e-12,
            samples=4096, numerics=kwargs["cartesian_numerics"])[0]) for mode in incoming.beam.modes)
        return 451., values, {"cartesian_admission": kwargs["cartesian_admission"]}
    monkeypatch.setattr(adapter, "round_column_prefix", transport)
    result = execute_round_prefix(state(), checkpoint, 451., numerics=numerics(),
        grid_numerics=WaveGridNumerics(maximum_working_bytes=256*1024**2), allow_cartesian=True)
    assert len(called) == 1
    assert result.record["upstream_digest"] == checkpoint.digest
    assert result.record["upstream"] == checkpoint.record
    for old, new in zip(checkpoint.beam.modes, result.beam.modes):
        assert new.mode_id == old.mode_id
        assert new.energy_kev == old.energy_kev
        assert new.scattering_history == old.scattering_history
        assert new.weight_per_reference_electron == pytest.approx(old.weight_per_reference_electron*old.plane.probability, rel=1e-8)
        assert new.axial_reference.flight_time_s > old.axial_reference.flight_time_s
        # Absolute complex phase remains observable after the adapter factors
        # unit shape and weight; the source is never re-normalized to unity.
        x, y = new.plane.coordinates_m()
        r2 = (x*x+y*y)/(1.5e-6)**2
        expected = np.sqrt(old.weight_per_reference_electron)*((1-.12*r2)+.025j*r2**2)*np.exp(-r2/2+.73j)/(1.5e-6*np.sqrt(np.pi*.8038))
        assert np.linalg.norm(new.weighted_density_amplitude()-expected)*new.plane.basis_m[0, 0] < 1e-6


def test_one_rejected_mode_preserves_entire_original_mixture(monkeypatch):
    import temsim.physics.round_column_checkpoint as adapter
    checkpoint = fixture()
    a, b = checkpoint.beam.modes
    b = replace(b, plane=replace(b.plane, tilt_rad=np.array((1e-4, 0.))))
    checkpoint = replace(checkpoint, beam=replace(checkpoint.beam, modes=(a, b)))
    monkeypatch.setattr(adapter, "round_column_prefix", lambda *a, **kw: pytest.fail("No operator before full admission"))
    assert execute_round_prefix(state(), checkpoint, 451., numerics=numerics(), allow_cartesian=True) is checkpoint


def test_cartesian_route_is_only_available_at_real_gun_exit(monkeypatch):
    import temsim.physics.round_column_checkpoint as adapter
    checkpoint = replace(fixture(), plane_z_mm=451.)
    monkeypatch.setattr(adapter, "round_column_prefix", lambda *a, **kw: pytest.fail("No downstream substitution"))
    assert execute_round_prefix(state(), checkpoint, 452., numerics=numerics(), allow_cartesian=True) is checkpoint


def test_memory_or_cancellation_is_not_misreported_as_symmetry_decline():
    checkpoint = fixture()
    with pytest.raises(WaveMemoryBudgetError):
        execute_round_prefix(state(), checkpoint, 451., numerics=numerics(), allow_cartesian=True,
            grid_numerics=WaveGridNumerics(maximum_working_bytes=1024))
    with pytest.raises(InterruptedError):
        execute_round_prefix(state(), checkpoint, 451., numerics=numerics(), allow_cartesian=True,
            cancelled=lambda: True)


def test_actual_round_executor_loads_checked_cartesian_field_with_budget(monkeypatch):
    from temsim.physics import radial_column_wave as radial
    from temsim.physics.canonical_action import CanonicalPath
    source = fixture()
    plan = SimpleNamespace(z_mm=np.array((450., 451.)), step_m=np.array((.001,)),
        mapped_fields=(), signature="identity-round-fixture", midpoint_magnetic_t=np.zeros(1),
        cs_kick_m3=np.zeros(2), kick_x_rad=np.zeros(2), kick_y_rad=np.zeros(2))
    for name in ("midpoint_sx_m2", "midpoint_sy_m2", "midpoint_sxy_m2", "midpoint_hex_normal_m3", "midpoint_hex_skew_m3"):
        setattr(plan, name, np.zeros(1))
    monkeypatch.setattr("temsim.physics.column_wave._prepare_column", lambda *a, **kw: (plan, np.full(2, np.inf), {}, ()))
    monkeypatch.setattr("temsim.physics.column_wave._column_transports", lambda *a, **kw: (CanonicalPath(.001),))
    conversion = CartesianRadialNumerics(maximum_working_bytes=256*1024**2)
    admissions = tuple({"mode_id": mode.mode_id} for mode in source.beam.modes)
    stop, output, record = radial._round_column_prefix(state(), source, 451., initial_samples=4096,
        cartesian_numerics=conversion, cartesian_admission=admissions,
        maximum_working_bytes=256*1024**2, retained_bytes=65536)
    assert stop == 451.
    assert record["cartesian_admission"] == admissions
    assert record["maximum_working_bytes"] == 256*1024**2
    for (mode, value), old in zip(output, source.beam.modes):
        expected, _ = cartesian_to_radial(old.plane, wavelength_m=2e-12, samples=4096, numerics=conversion)
        assert mode is old
        # FFTLog's 1/r density magnifies roundoff at the tiny-radius endpoint;
        # compare the complete complex field in its physical radial measure.
        error = np.sqrt(2*np.pi*value.dln*np.sum(abs(value.radius_m*(value.amplitude-expected.amplitude))**2))
        assert error < 1e-8
        assert value.probability == pytest.approx(expected.probability, rel=1e-10)


@pytest.mark.parametrize("kind", ["custom", "slit", "shifted"])
def test_nonround_aperture_is_left_for_two_dimensional_operator(monkeypatch, kind):
    from temsim.physics import radial_column_wave as radial
    source = fixture()
    plan = SimpleNamespace(z_mm=np.array((450., 451.)), step_m=np.array((.001,)),
        mapped_fields=(), signature="mask-fixture", midpoint_magnetic_t=np.zeros(1),
        cs_kick_m3=np.zeros(2), kick_x_rad=np.zeros(2), kick_y_rad=np.zeros(2))
    for name in ("midpoint_sx_m2", "midpoint_sy_m2", "midpoint_sxy_m2", "midpoint_hex_normal_m3", "midpoint_hex_skew_m3"):
        setattr(plan, name, np.zeros(1))
    aperture = SimpleNamespace(key="must-retain-mask", radius_mm=1., offset_x_mm=0., offset_y_mm=0.)
    if kind == "custom":
        aperture.transmission_mask = lambda x, y: abs(x) < .1
    elif kind == "slit":
        aperture.kind = "energy_selection_slit"
    else:
        aperture.offset_x_mm = .1
    monkeypatch.setattr("temsim.physics.column_wave._prepare_column", lambda *a, **kw: (plan, np.full(2, np.inf), {1: (aperture,)}, ()))
    monkeypatch.setattr(radial, "RadialExecution", lambda *a, **kw: pytest.fail("No radial substitute for this mask"))
    stop, output, record = radial._round_column_prefix(state(), source, 451.)
    assert stop == 450. and output == ()
    assert "non-axisymmetric" in record["reason"]


def test_nonzero_angular_payload_is_not_reconstructed_as_zero_order(monkeypatch):
    import temsim.physics.round_column_checkpoint as adapter
    source = fixture(False)
    source = replace(source, record={"radial_output_modes": ({"mode_id": source.beam.modes[0].mode_id,
        "angular_order": 1},)})
    monkeypatch.setattr(adapter, "round_column_prefix", lambda *a, **kw: pytest.fail("Cannot discard angular phase"))
    assert execute_round_prefix(state(), source, 451.) is source
