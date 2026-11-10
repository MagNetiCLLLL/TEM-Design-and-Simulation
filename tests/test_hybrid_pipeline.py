"""Mocked boundaries and small array checks, not physical finite-source runs."""
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from temsim.physics.hybrid_pipeline import HybridGrid, HybridRequest, HybridResult, _grid_schedule, _material, _downstream
from temsim.physics.hybrid_exit import sample_exit
from temsim.physics.hybrid_semiclassical import SemiclassicalNumerics
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState
from test_hybrid_semiclassical import gaussian_mode
from test_potential_ray import uniform_snapshot


def checkpoint(mode=None):
    mode = gaussian_mode() if mode is None else mode
    return TipGunCheckpoint(BeamState((mode,), mode.reference_plane), 1., 1e-9, {"schema": "array-test"})


def test_hybrid_binds_gun_environment_before_immutable_capture(monkeypatch):
    """Preparation only: a later real gun trace must not add vacuum inputs."""
    import sys
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.vacuum import bind_gun_environment
    from temsim.physics.hybrid_pipeline import simulate_hybrid
    state = default_state()
    state.simulation_mode = "unified"
    before = capture_instrument_snapshot(state).digest
    monkeypatch.setattr("temsim.physics.compute_backend.require_coherent_cuda", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "cupy", np)
    def capture(working, **kwargs):
        assert hasattr(working.electron_gun, "_vacuum_regions")
        identity = capture_instrument_snapshot(working).digest
        bind_gun_environment(working)
        assert capture_instrument_snapshot(working).digest == identity
        raise InterruptedError("Mocked capture after stable source preparation")
    monkeypatch.setattr("temsim.physics.unified_field_runtime.capture_unified_field_runtime", capture)
    request = HybridRequest("camera", HybridGrid(state.sample.z_mm-state.sample.thickness_nm*.5e-6, 32, 100.))
    with pytest.raises(InterruptedError, match="stable source preparation"):
        simulate_hybrid(state, request)
    assert capture_instrument_snapshot(state).digest == before


def test_dynamic_scan_is_rejected_before_emission_without_freezing(monkeypatch):
    """Mocked admission: an active drive must never become a static image."""
    import sys
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.hybrid_pipeline import simulate_hybrid
    state = default_state()
    state.simulation_mode = "unified"
    before = capture_instrument_snapshot(state).digest
    monkeypatch.setattr("temsim.physics.compute_backend.require_coherent_cuda", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "cupy", np)
    dynamic = SimpleNamespace(dynamic_coils=(
        SimpleNamespace(drives=(SimpleNamespace(key="ac_deflector"),)),
        SimpleNamespace(drives=(SimpleNamespace(key="descan_deflector"),))))
    monkeypatch.setattr("temsim.physics.unified_field_runtime.capture_unified_field_runtime",
        lambda *a, **k: SimpleNamespace(snapshot=dynamic))
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit",
        lambda *a, **k: pytest.fail("An unsupported dynamic image started source execution"))
    request = HybridRequest("camera", HybridGrid(state.sample.z_mm-state.sample.thickness_nm*.5e-6, 32, 100.))
    with pytest.raises(ValueError, match="active drives: ac_deflector, descan_deflector"):
        simulate_hybrid(state, request)
    assert capture_instrument_snapshot(state).digest == before


def test_finite_plate_faces_and_thickness_are_resolved():
    from temsim.physics.potential_geometry import PotentialGeometrySnapshot, PotentialBoundaryHost
    from temsim.physics.posed_wave_aperture import PosedWaveAperture
    from temsim.physics.lens_field_provider import CoordinateRegistration
    plate = PosedWaveAperture("test-plate", 1.002, 1.006, 20e-6, CoordinateRegistration())
    geometry = PotentialGeometrySnapshot(.001, .00101, (PotentialBoundaryHost("test-plate", ("test-plate",), (plate,)),))
    request = HybridRequest("camera", HybridGrid(1.01, 48, 48.))
    schedule = _grid_schedule(checkpoint(), request, geometry)
    z = np.array([value for value, _ in schedule])
    assert 1.002 in z and 1.006 in z and z[-1] == 1.01
    active = z[(z >= 1.002) & (z <= 1.006)]
    assert len(active) >= 9
    assert np.max(np.diff(active)) <= .004/8+1e-15
    with pytest.raises(MemoryError):
        _grid_schedule(checkpoint(), replace(request, maximum_geometry_steps=2), geometry)


def test_material_budget_is_bound_at_entrance_and_parent_retained(tmp_path, monkeypatch):
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.electric_wave_reference import ElectricWaveHistory
    initial, snapshot = checkpoint(), uniform_snapshot()
    monkeypatch.setattr("temsim.specimen.source.specimen_is_vacuum", lambda sample: False)
    def executed(state, entry, fields, **kwargs):
        ledger = KineticPhaseLedger.from_dict(entry.record["kinetic_phase_ledger"])
        assert ledger.last_z_m == entry.plane_z_mm*.001
        assert ledger.binding["snapshot_digest"] == fields.numerical_digest
        assert ledger.used_phase_rad == 0. and len(ledger.events) == 0
        assert entry.record["upstream_digest"] == initial.digest
        assert ElectricWaveHistory.from_dict(entry.record["electric_wave_history"]).last_z_m == ledger.last_z_m
        mode = replace(entry.beam.modes[0], mode_id="mode:0/loss", weight_per_reference_electron=.17)
        return replace(entry, beam=BeamState((mode,), entry.beam.reference_plane), plane_z_mm=1.00001,
                       record={"declared_mock_material": True})
    monkeypatch.setattr("temsim.physics.potential_specimen_transport.propagate_potential_specimen", executed)
    request = HybridRequest("camera", HybridGrid(1., 48, 48.))
    request = replace(request, wave=replace(request.wave,
        execution=replace(request.wave.execution, cache_directory=str(tmp_path))))
    runtime = SimpleNamespace(snapshot=snapshot, geometry=None, digest="mock-runtime")
    result = _material(SimpleNamespace(sample=object()), initial, runtime, request,
                       cancelled=lambda: False, progress_callback=None)
    assert result.beam.total_weight == .17
    assert result.record["sample_feedback_on_external_fields"] is False


def test_fully_absorbed_current_stays_zero_through_handoff():
    mode = replace(gaussian_mode(), weight_per_reference_electron=0.)
    initial = checkpoint(mode)
    particles, spectra = sample_exit(initial, uniform_snapshot(), 32, SemiclassicalNumerics(), xp=np)
    assert len(particles.weight) == 0 and not spectra
    result = _downstream(None, None, particles, 2., None, start_z_mm=1.)
    assert result["remaining_weight"] == result["balance_residual"] == 0.


def test_export_retains_complex_wave_modes_and_loss_accounting(tmp_path):
    from temsim.hybrid_cli import save_result
    initial = checkpoint()
    particles, spectra = sample_exit(initial, uniform_snapshot(), 32,
        SemiclassicalNumerics(frame_stride=12, window_sigma_pixels=12.), xp=np)
    result = HybridResult(initial, initial, particles, spectra,
        {"remaining_weight": .1, "detected_or_absorbed_weight_by_key": {"camera": .3}},
        {"status": "ARRAY_TEST_ONLY"}, {"ray_id": np.arange(3)})
    destination = tmp_path/"result"
    save_result(result, destination)
    report = json.loads((destination/"report.json").read_text())
    row = report["wave_checkpoints"]["incident"]["modes"][0]
    with np.load(destination/"hybrid_arrays.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays[row["arrays"]["amplitude"]], initial.beam.modes[0].plane.amplitude)
    assert report["downstream_accounting"]["detected_or_absorbed_weight_by_key"] == {"camera": .3}
    with pytest.raises(FileExistsError):
        save_result(result, destination)
