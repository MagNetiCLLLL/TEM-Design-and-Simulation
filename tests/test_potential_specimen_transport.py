"""Actual atomistic multislice and conditional histories with potential transport."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
from temsim.physics.potential_specimen_transport import propagate_potential_specimen, PotentialMaterialTransport
from temsim.physics.potential_specimen_transport import propagate_potential_specimen_to_detector
from temsim.physics.potential_wave_observation import observe_potential_checkpoint
from temsim.physics.tip_gun_wave import TipGunCheckpoint, TIP_REFERENCE, _momentum_velocity
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_execution import InelasticWaveNumerics
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_grid import WaveGridNumerics
from temsim.physics.wave_reference import AxialWaveReference
from test_segmented_tip_wave import quiet_state
from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import encode_instrument


def _fixture():
    state = quiet_state()
    sample = state.sample
    sample.specimen_mode = "atomic"
    sample.cif_path = str(Path(__file__).parent/"fixtures"/"cif"/"Si.cif")
    sample.inserted = True
    sample.thickness_nm = .4
    sample.wave_slice_thickness_angstrom = 2.
    sample.wave_grid_pixels = 256
    sample.wave_field_of_view_angstrom = 30.
    sample.wave_frozen_phonon_enabled = False
    axis = (np.arange(256)-128)*1e-11
    x, y = np.meshgrid(axis, axis)
    amplitude = np.exp(-(x*x+y*y)/(2*(.2e-9)**2)).astype(complex)*np.exp(.37j)
    amplitude /= np.linalg.norm(amplitude)
    mode = WaveMode(PlaneWave(amplitude, np.eye(2)*1e-11, np.zeros(2),
        np.eye(2)*20., np.array((1e-6, -2e-6))), .4, TIP_REFERENCE, "actual-Si-entrance",
        300., AxialWaveReference(1e-8, 1e-22))
    entrance = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), sample.z_mm-.2e-6,
        1e-9, {"source_scope": "controlled complex entrance, actual Si finite material"})
    return state, entrance


def _snapshot(state, strength=.4, *, dynamic=False):
    centre = state.sample.z_mm
    momentum = _momentum_velocity(300000.)[0]
    native = FrozenPosedMultipole("material-test-dipole", "material-test-dipole", CoordinateRegistration(),
        ((1, 0, -strength),), "uniform", centre*.001, .001, (centre-1., centre+1.),
        .003, momentum, dynamic=dynamic)
    potential = AnalyticPotentialDomain(AnalyticMagneticPotential(native))
    return ElectromagneticPotentialSnapshot(capture_physical_scene(state),
        (PotentialContribution("controlled-dipole", ("controlled-dipole",), potential, "magnetic"),))


def _certain_loss(_state):
    """Deterministic 100 eV test instrument; real atomic potential is unchanged."""
    return SimpleNamespace(absorbed_probability=0., mean_inelastic_events=1., channels=(
        SimpleNamespace(key="real_zero_loss", probability=0., energy_loss_ev=0.,
            characteristic_angle_mrad=0., mean_events=0., approximation="deterministic test instrument"),
        SimpleNamespace(key="real_plasmon", probability=1., energy_loss_ev=100.,
            characteristic_angle_mrad=0., mean_events=1., approximation="deterministic test instrument")))


def _run(state, entrance, snapshot, path, **kwargs):
    return propagate_potential_specimen(state, entrance, snapshot,
        store=ExecutedWaveStore(path, "controlled-potential-material", 1<<29),
        inelastic_numerics=InelasticWaveNumerics(trajectories_per_mode=1),
        grid_numerics=WaveGridNumerics(maximum_pixels=4096),
        maximum_step_mm=.001, **kwargs)


@pytest.mark.parametrize("with_geometry", [False, True])
def test_migrated_pressure_prefix_survives_actual_material_branches_and_downstream(tmp_path, monkeypatch, with_geometry):
    from hashlib import sha256
    from temsim.immutable_json import thaw_json
    from temsim.physics.potential_wave_policy import (
        transition_support_policy, validate_support_policy_transition, potential_support_policy,
        potential_wave_operator_policy,
    )
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.potential_support_lineage import common_support_summary
    from temsim.physics.unified_wave_support import prepare_unified_support, stamp_unified_support
    from temsim.physics.potential_geometry import capture_potential_geometry
    from test_potential_wave_policy import _legacy_transition_fixture
    from test_potential_weighted_resampling import _pressure_context
    state, entrance = _fixture()
    snapshot = _snapshot(state)
    mode = entrance.beam.modes[0]
    source_digest = entrance.digest
    record = prepare_unified_support({}, mode, enabled=True, l2_limit=.02,
        probability_limit=1e-6, source_digest=source_digest, allow_allocate=True)
    payload = {"policy": "weighted_segment_and_retry", "trigger": "typed_resource_failure_only"}
    record.update(snapshot_digest=snapshot.numerical_digest, source_checkpoint_digest=source_digest,
        source_mode_id=mode.mode_id, source_mode_weight_per_reference=mode.weight_per_reference_electron,
        unified_wave_identity="old-executor", potential_operator_policy=potential_wave_operator_policy({}),
        potential_support_policy={"schema": "potential-support-policy-v2", **payload, "digest": json_digest(payload)})
    _, transition = _legacy_transition_fixture()
    transition["native_plane_m"] = entrance.plane_z_mm*.001
    transition["source_reference"].update(source_checkpoint_digest=source_digest, mode_id=mode.mode_id,
        mode_weight=mode.weight_per_reference_electron, checkpoint_digest=entrance.digest,
        array_descriptors={name: {"file": name+".npy", "sha256": sha256(getattr(mode.plane, name).tobytes()).hexdigest(),
            "shape": list(getattr(mode.plane, name).shape), "nbytes": getattr(mode.plane, name).nbytes}
            for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad")})
    record = thaw_json(transition_support_policy(record, **transition))
    original_allocation = record["unified_support_allocation"]
    ledger = NumericalSupportLedger.from_dict(record["support_ledger"])
    projected, ledger = ledger.resample_spectrum_pressure(mode.plane,
        z_m=entrance.plane_z_mm*.001, mode_weight=mode.weight_per_reference_electron,
        admissible_shape=lambda shape: max(shape) <= 240, resource_context=_pressure_context(mode.plane))
    assert ledger.to_dict()["schema"] == "controlled-numerical-support-ledger-v4" and len(ledger.events) == 1
    norm = projected.probability
    mode = replace(mode, plane=replace(projected, amplitude=projected.amplitude/np.sqrt(norm)),
                   weight_per_reference_electron=mode.weight_per_reference_electron*norm)
    record["support_ledger"] = ledger.to_dict()
    record = stamp_unified_support(record)
    entrance = replace(entrance, beam=BeamState((mode,), TIP_REFERENCE), record=record)
    validate_support_policy_transition(entrance.record)
    monkeypatch.setattr("temsim.specimen.inelastic.real_inelastic_distribution", _certain_loss)
    stop_mm = state.sample.z_mm+.2e-6+1e-6
    geometry = capture_potential_geometry(state, entrance.plane_z_mm*.001, stop_mm*.001)
    material = _run(state, entrance, snapshot, tmp_path,
                    geometry=geometry if with_geometry else None)
    assert material.beam.modes[0].energy_kev < mode.energy_kev
    assert material.record.get("support_ledger") is None
    summary = common_support_summary(material.record)
    assert summary["accounting_multiplicity"] == 1 and not summary["remaining_budget_reissued"]
    assert summary["probability_removed_per_reference"] == ledger.used_probability
    row = material.record["modes"][0]
    context = row["column_transport_context"]
    assert context["support_policy_transition"] == entrance.record["support_policy_transition"]
    assert context["unified_support_allocation"] == original_allocation
    validate_support_policy_transition(context)
    branch = replace(material, record=context)
    continued, = observe_potential_checkpoint(snapshot, branch, (stop_mm*.001,), maximum_step_m=1e-9,
        grid_numerics=WaveGridNumerics(maximum_pixels=4096))
    validate_support_policy_transition(continued.record)
    assert continued.record["support_ledger"] is None
    assert common_support_summary(continued.record) == summary
    assert continued.record["unified_support_ledger_digest"] == json_digest(ledger.to_dict())
    preserved = prepare_unified_support(continued.record, continued.beam.modes[0], enabled=True,
        l2_limit=.02, probability_limit=1e-6, source_digest=source_digest, allow_allocate=False)
    assert preserved["unified_support_allocation"] == original_allocation
    with pytest.raises(ValueError, match="projection is disabled"):
        potential_support_policy({"support_projection_policy": "weighted_segment_and_retry"}, continued.record)


def test_geometry_callback_counts_all_modes_once_and_retains_the_aggregate_when_resumed(monkeypatch):
    from temsim.physics.potential_geometry import PotentialBoundaryHost, PotentialGeometrySnapshot
    from temsim.physics.posed_wave_aperture import PosedWaveAperture
    import temsim.physics.potential_geometry as boundary

    axis = (np.arange(64)-32)*2e-8
    x, y = np.meshgrid(axis, axis)
    amplitude = np.exp(-(x*x+y*y)/(4*(8e-8)**2)).astype(complex)*np.exp(.37j)
    amplitude /= np.linalg.norm(amplitude)
    plane = PlaneWave(amplitude, np.eye(2)*2e-8, np.zeros(2))
    modes = (WaveMode(plane, .27, TIP_REFERENCE, "central", 300.),
             WaveMode(replace(plane, origin_m=np.array([3e-6, 0.])), .13, TIP_REFERENCE, "displaced", 299.9))
    entrance = TipGunCheckpoint(BeamState(modes, TIP_REFERENCE), 1000., 1e-9, {})
    snapshot = ElectromagneticPotentialSnapshot(capture_physical_scene(SimpleNamespace(lenses=[])), ())
    plates = (
        PosedWaveAperture("outer", 1000.000009, 1000.000011, .0015, CoordinateRegistration()),
        # Read the partial mask at the finite plate's exact entrance face.
        # Propagation after this new edge needs separate grid qualification.
        PosedWaveAperture("inner", 1000.00004, 1000.000042, .0002, CoordinateRegistration()),
    )
    geometry = PotentialGeometrySnapshot(1., 1.00000004,
        tuple(PotentialBoundaryHost(plate.key, (plate.key,), (plate,)) for plate in plates))
    transport = PotentialMaterialTransport(snapshot, (1000., 1000.00004), "controlled two-mode boundary",
                                          {"kinetic_order": 2}, geometry)
    calls = []
    actual = boundary.observe_potential_geometry_checkpoint
    def counted(*args, **kwargs):
        calls.append(tuple(mode.mode_id for mode in args[2].beam.modes))
        return actual(*args, **kwargs)
    monkeypatch.setattr(boundary, "observe_potential_geometry_checkpoint", counted)
    def execute(checkpoint, stop):
        return transport(SimpleNamespace(), checkpoint, stop, maximum_step_mm=4e-6,
            grid_numerics=WaveGridNumerics(maximum_pixels=512), cancelled=lambda: False)

    middle = execute(entrance, 1000.00002)
    first = middle.record["potential_geometry"]
    assert calls == [("central", "displaced")]
    assert middle.beam.modes[0].weight_per_reference_electron == pytest.approx(.27, rel=0., abs=2e-14)
    assert middle.beam.modes[1].weight_per_reference_electron == 0.
    assert first["entry_weight_per_reference"] == pytest.approx(.4)
    assert first["physical_loss_per_reference"] == pytest.approx(.13, rel=0., abs=2e-14)
    assert first["output_weight_per_reference"] == pytest.approx(.27, rel=0., abs=2e-14)

    continued = execute(middle, 1000.00004)
    whole = execute(entrance, 1000.00004)
    assert calls == [("central", "displaced")]*3
    for result in (continued, whole):
        total = result.record["potential_geometry"]
        per_mode = {row["mode_id"]: row["potential_execution"]["geometry_mode"] for row in result.record["modes"]}
        assert 0. < per_mode["central"]["physical_loss_per_reference"] < .27
        assert per_mode["displaced"]["physical_loss_per_reference"] == pytest.approx(.13, rel=0., abs=2e-14)
        assert total["physical_loss_per_reference"] == pytest.approx(
            sum(row["physical_loss_per_reference"] for row in per_mode.values()), rel=0., abs=2e-14)
        assert total["entry_weight_per_reference"] == pytest.approx(.4)
        assert total["output_weight_per_reference"] == result.beam.total_weight
        assert total["entry_weight_per_reference"]-total["physical_loss_per_reference"] == pytest.approx(
            result.beam.total_weight, rel=0., abs=2e-13)
        assert all(all(loss["mode_id"] == key for loss in row["physical_losses"]) for key, row in per_mode.items())
        assert [mode.energy_kev for mode in result.beam.modes] == [300., 299.9]
    assert continued.beam.total_weight == pytest.approx(whole.beam.total_weight, rel=0., abs=2e-12)
    for split, direct in zip(continued.beam.modes, whole.beam.modes, strict=True):
        np.testing.assert_allclose(split.plane.amplitude, direct.plane.amplitude, rtol=0., atol=2e-11)


@pytest.mark.parametrize("order", [4, 6, 8])
def test_actual_si_slices_execute_nonzero_magnetic_field_at_each_new_energy(tmp_path, monkeypatch, order):
    state, entrance = _fixture()
    snapshot = _snapshot(state)
    before = json_digest(encode_instrument(state))
    import temsim.physics.potential_specimen_transport as bridge
    actual = bridge.observe_potential_mode
    energies = []
    def traced(snapshot, mode, *args, **kwargs):
        energies.append(mode.energy_kev)
        return actual(snapshot, mode, *args, **kwargs)
    monkeypatch.setattr(bridge, "observe_potential_mode", traced)
    monkeypatch.setattr("temsim.specimen.inelastic.real_inelastic_distribution", _certain_loss)
    def legacy(*args, **kwargs):
        pytest.fail("The material bridge must not execute legacy column optics or an empirical pupil")
    monkeypatch.setattr("temsim.physics.column_wave._propagate_column", legacy)
    monkeypatch.setattr("temsim.physics.specimen_wave_transport._propagate_column", legacy)
    monkeypatch.setattr("temsim.physics.wave_imaging.explicit_pupil_spectrum", legacy)
    operator_options = {} if order == 4 else {"kinetic_order": order, "scalar_splitting": True,
                                             "higher_order_scalar_splitting": True}
    output = _run(state, entrance, snapshot, tmp_path, operator_options=operator_options)
    assert energies == pytest.approx([300., 299.9], rel=0, abs=1e-10)
    mode, = output.beam.modes
    assert mode.energy_kev == pytest.approx(299.8)
    assert len(mode.scattering_history) == 2
    assert all(event["energy_loss_ev"] == 100. for event in mode.scattering_history)
    assert output.plane_z_mm == state.sample.z_mm+.2e-6
    assert output.record["snapshot_digest"] == snapshot.numerical_digest
    assert output.record["legacy_column_executed"] is False
    assert output.record["empirical_aberrations_applied"] is False
    assert output.record["potential"]["coherent_material_grid"]["executed_potential_pixels"] >= 128
    row = output.record["modes"][0]
    assert abs(row["probability_residual"]) < 1e-13
    assert mode.weight_per_reference_electron <= .4+1e-12
    for index, step in enumerate(row["steps"]):
        column, = step["column"]
        assert column["potential_execution"]["snapshot_digest"] == snapshot.numerical_digest
        assert column["potential_execution"]["potential_operator_policy"]["policy"]["kinetic_order"] == order
        assert column["energy_kev"] == pytest.approx(300.-.1*index)
        assert step["phase_before"]["method"] != "no surviving wave"
    p0, v0 = _momentum_velocity(300000.)
    p1, v1 = _momentum_velocity(299900.)
    initial = entrance.beam.modes[0].axial_reference
    assert mode.axial_reference.flight_time_s-initial.flight_time_s == pytest.approx(.2e-9/v0+.2e-9/v1, rel=3e-6, abs=1e-23)
    assert mode.axial_reference.longitudinal_action_j_s-initial.longitudinal_action_j_s == pytest.approx(.2e-9*(p0+p1), rel=3e-6, abs=1e-38)
    assert json_digest(encode_instrument(state)) == before
    # A committed material checkpoint reuses the actual conditional histories,
    # with no extra physical field or energy-loss actions.
    cached = _run(state, entrance, snapshot, tmp_path, operator_options=operator_options)
    assert cached.digest == output.digest
    assert len(energies) == 2


def test_material_transport_budget_is_shared_and_snapshot_change_invalidates_identity():
    state, entrance = _fixture()
    start, stop = entrance.plane_z_mm, state.sample.z_mm+.2e-6
    transport = PotentialMaterialTransport(_snapshot(state), (start, stop), "input")
    middle = (start+stop)/2
    a, b = transport.options_for(start, middle), transport.options_for(middle, stop)
    assert a["exponential_error"]+b["exponential_error"] == pytest.approx(1e-10, rel=1e-15)
    assert a["kinetic_remainder_phase_limit"]+b["kinetic_remainder_phase_limit"] == pytest.approx(.01, rel=1e-15)
    changed = PotentialMaterialTransport(_snapshot(state, .41), (start, stop), "input")
    assert changed.digest != transport.digest
    with pytest.raises(ValueError, match="interval"):
        transport.options_for(start-1., stop)
    with pytest.raises(TypeError):
        transport.operator_options["exponential_error"] = 1.


def test_actual_material_exit_continues_through_same_geometry_to_absolute_detector(tmp_path, monkeypatch):
    from temsim.physics.potential_geometry import capture_potential_geometry
    from temsim.detector.wave_readout import WaveReadoutOptions
    from test_wave_detector_readout import detector
    state, entrance = _fixture()
    snapshot = _snapshot(state)
    sensor = detector()
    sensor.z_mm = state.sample.z_mm+.2e-6+1e-6  # 1 nm after the exit, a distinct physical plane.
    sensor.outer_width_mm = .001
    geometry = capture_potential_geometry(state, entrance.plane_z_mm*.001, sensor.z_mm*.001)
    monkeypatch.setattr("temsim.specimen.inelastic.real_inelastic_distribution", _certain_loss)
    def legacy(*args, **kwargs):
        pytest.fail("The potential workflow must never append the legacy column/pupil")
    monkeypatch.setattr("temsim.physics.column_wave._propagate_column", legacy)
    monkeypatch.setattr("temsim.physics.specimen_wave_transport._propagate_column", legacy)
    monkeypatch.setattr("temsim.physics.wave_imaging.explicit_pupil_spectrum", legacy)
    result = propagate_potential_specimen_to_detector(state, entrance, snapshot, geometry, sensor,
        store=ExecutedWaveStore(tmp_path, "actual-potential-material-detector", 1<<29),
        maximum_step_mm=.001, inelastic_numerics=InelasticWaveNumerics(trajectories_per_mode=1),
        grid_numerics=WaveGridNumerics(maximum_pixels=4096),
        maximum_readout_bytes=2*1024**3,
        readout_options=WaveReadoutOptions(phase=True, complex_amplitude=True))
    assert result.specimen.plane_z_mm < result.detector_checkpoint.plane_z_mm == sensor.z_mm
    incoming, outgoing = result.specimen.beam.modes[0], result.detector_checkpoint.beam.modes[0]
    assert outgoing.energy_kev == incoming.energy_kev == pytest.approx(299.8)
    assert outgoing.scattering_history == incoming.scattering_history
    assert outgoing.axial_reference.flight_time_s > incoming.axial_reference.flight_time_s
    assert outgoing.axial_reference.longitudinal_action_j_s > incoming.axial_reference.longitudinal_action_j_s
    assert result.detector_checkpoint.record["geometry_digest"] == geometry.digest
    history = result.detector_checkpoint.record["modes"][0]["execution"]["potential_geometry"]
    assert history["entry_z_m"] == entrance.plane_z_mm*.001
    assert history["last_z_m"] == sensor.z_mm*.001
    assert history["accepted_steps"] >= 3
    assert result.readout.detected_probability.sum() == pytest.approx(outgoing.weight_per_reference_electron, abs=2e-12)
    assert result.readout.modes[0].energy_kev == outgoing.energy_kev
    assert "UNDEFINED" in result.readout.record["mixed_total_phase"]
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    # Detector phase is an explicitly labelled point readout of the exact
    # carrier, not a Fourier expansion or an unwrapped reconstruction.
    xy = outgoing.plane.coordinates_m()-outgoing.plane.origin_m[:, None, None]
    q, t = outgoing.plane.curvature_m1, outgoing.plane.tilt_rad
    phase = (.5*np.einsum("iyx,ij,jyx->yx", xy, q, xy)+np.einsum("i,iyx->yx", t, xy))
    expected = outgoing.plane.amplitude*np.exp(2j*np.pi*phase/float(wavelength_m(outgoing.energy_kev*1000)))*np.sqrt(outgoing.weight_per_reference_electron)
    assert "wrapped point samples" in result.readout.record["phase_sampling"]
    np.testing.assert_allclose(result.readout.modes[0].complex_cell_amplitude, expected, atol=2e-13)


@pytest.mark.parametrize("kinetic_budget", (False, True))
def test_material_branching_keeps_one_shared_support_prefix_through_detector(tmp_path, monkeypatch, kinetic_budget):
    from test_inelastic_wave import material_fixture
    from test_wave_detector_readout import detector
    from temsim.physics.potential_geometry import capture_potential_geometry
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.potential_support_lineage import common_support_summary
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    state, _ = _fixture()
    _, entrance, material = material_fixture()
    original = entrance.beam.modes[0]
    wide = np.pad(original.plane.amplitude, ((32, 32), (32, 32)))
    wide[64, 120] = 1e-11
    plane, ledger = NumericalSupportLedger(1e-5, 1e-10).compact(
        replace(original.plane, amplitude=wide), wavelength_m=float(wavelength_m(300000.)),
        z_m=entrance.plane_z_mm*.001, mode_weight=original.weight_per_reference_electron)
    assert ledger.used_probability > 0 and ledger.events
    norm = plane.probability
    mode = replace(original, plane=replace(plane, amplitude=plane.amplitude/np.sqrt(norm)),
                   weight_per_reference_electron=original.weight_per_reference_electron*norm)
    snapshot = _snapshot(state)
    entrance = replace(entrance, beam=BeamState((mode,), TIP_REFERENCE), record={
        "fixture": "manufactured smooth finite elastic material plus actual conditional instrument",
        "snapshot_digest": snapshot.numerical_digest, "support_ledger": ledger.to_dict()})
    if kinetic_budget:
        from temsim.physics.potential_phase_budget import KineticPhaseLedger
        state.sample.size_x_nm = state.sample.size_y_nm = 100.
        snapshot = _snapshot(state)
        upstream = replace(entrance, plane_z_mm=entrance.plane_z_mm-1e-6,
            record={"snapshot_digest": snapshot.numerical_digest})
        entrance, = observe_potential_checkpoint(snapshot, upstream,
            (entrance.plane_z_mm*.001,), maximum_step_m=1e-9,
            kinetic_phase_ledger=KineticPhaseLedger(.01),
            grid_numerics=WaveGridNumerics(maximum_pixels=512))
        entrance = replace(entrance, record={**entrance.record, "support_ledger": ledger.to_dict()})
    monkeypatch.setattr("temsim.physics.specimen_wave_transport._prepare_material_grid", lambda *args, **kwargs: material)
    monkeypatch.setattr("temsim.specimen.inelastic.real_inelastic_distribution", _certain_loss)
    sensor = detector()
    sensor.z_mm = state.sample.z_mm+.2e-6+1e-6
    geometry = capture_potential_geometry(state, entrance.plane_z_mm*.001, sensor.z_mm*.001)
    result = propagate_potential_specimen_to_detector(state, entrance, snapshot, geometry, sensor,
        store=ExecutedWaveStore(tmp_path, "shared-prefix-material", 1<<28),
        maximum_step_mm=.001, inelastic_numerics=InelasticWaveNumerics(trajectories_per_mode=3),
        grid_numerics=WaveGridNumerics(maximum_pixels=512))
    assert len(result.detector_checkpoint.beam.modes) == 3
    assert all(mode.energy_kev == pytest.approx(299.8) for mode in result.detector_checkpoint.beam.modes)
    summaries = [common_support_summary(obj.record) for obj in
                 (result.specimen, result.detector_checkpoint, result.readout)]
    assert summaries[0] == summaries[1] == summaries[2]
    assert summaries[0]["accounting_multiplicity"] == 1
    assert summaries[0]["probability_removed_per_reference"] == ledger.used_probability
    assert not summaries[0]["remaining_budget_reissued"]
    assert result.specimen.record.get("support_ledger") is None
    if kinetic_budget:
        from temsim.physics.potential_material_phase_budget import material_kinetic_summary
        for item in (result.detector_checkpoint, result.readout):
            assert item.record["common_kinetic_prefix"] == result.specimen.record["common_kinetic_prefix"]
        input_cost = KineticPhaseLedger.from_dict(entrance.record["kinetic_phase_ledger"]).used_phase_rad
        for row in result.detector_checkpoint.record["modes"]:
            summary = material_kinetic_summary(row["execution"])
            assert input_cost < summary["history_used_phase_rad"] <= .01
            assert summary["maximum_phase_rad"] == .01
            assert result.readout.record["kinetic_phase_histories"][row["mode_id"]] == summary
    assert all(row["column_transport_context"]["support_prefix_digest"] == summaries[0]["prefix_digest"]
               for row in result.specimen.record["modes"])
    expected_guard = "separate per-mode execution contexts" if kinetic_budget else "projection is disabled"
    with pytest.raises(ValueError, match=expected_guard):
        observe_potential_checkpoint(snapshot, result.detector_checkpoint,
            ((sensor.z_mm+1e-6)*.001,), support_ledger=NumericalSupportLedger(1., 1.))
    if kinetic_budget:
        context = result.detector_checkpoint.record["modes"][0]["execution"]
        phase = KineticPhaseLedger.from_dict(context["kinetic_phase_ledger"])
        branch = replace(result.detector_checkpoint, plane_z_mm=phase.last_z_m*1000.,
            beam=BeamState((result.detector_checkpoint.beam.modes[0],), TIP_REFERENCE), record=context)
        with pytest.raises(ValueError, match="projection is disabled"):
            observe_potential_checkpoint(snapshot, branch,
                ((sensor.z_mm+1e-6)*.001,), support_ledger=NumericalSupportLedger(1., 1.))


@pytest.mark.parametrize("kinetic_budget", (False, True))
def test_zero_loss_branch_uses_real_material_attenuation_with_same_potential(tmp_path, monkeypatch, kinetic_budget):
    from test_inelastic_wave import material_fixture
    from temsim.specimen.inelastic import real_inelastic_distribution
    state, _ = _fixture()
    _, entrance, material = material_fixture()
    # Imported CIFs intentionally have no invented IMFP. Supply explicit
    # declared material inputs and execute the real distribution builder.
    state.sample.real_inelastic_enabled = True
    state.sample.real_plasmon_mean_free_path_nm = 100.
    state.sample.real_plasmon_energy_ev = 16.
    monkeypatch.setattr("temsim.physics.specimen_wave_transport._prepare_material_grid", lambda *args, **kwargs: material)
    distribution = real_inelastic_distribution(state)
    assert 0 < distribution.total_inelastic_mean_free_path_nm < np.inf
    snapshot = _snapshot(state)
    if kinetic_budget:
        from temsim.physics.potential_phase_budget import KineticPhaseLedger
        upstream = replace(entrance, plane_z_mm=entrance.plane_z_mm-1e-6)
        entrance, = observe_potential_checkpoint(snapshot, upstream, (entrance.plane_z_mm*.001,),
            maximum_step_m=1e-9, kinetic_phase_ledger=KineticPhaseLedger(.01),
            grid_numerics=WaveGridNumerics(maximum_pixels=512))
    output = propagate_potential_specimen(state, entrance, snapshot,
        store=ExecutedWaveStore(tmp_path, "real-material-attenuation", 1<<28),
        inelastic_numerics=InelasticWaveNumerics(method="zero_loss"),
        grid_numerics=WaveGridNumerics(maximum_pixels=512))
    mode, = output.beam.modes
    row, = output.record["modes"]
    assert mode.energy_kev == entrance.beam.modes[0].energy_kev
    assert 0 < mode.weight_per_reference_electron < entrance.beam.total_weight
    assert row["inelastic_material"]["zero_loss_removal_rate_per_nm"] > 0
    assert sum(row["probability_balance"]["first_events"].values()) > 0
    assert row["probability_balance"]["accounted_weight"] == pytest.approx(entrance.beam.total_weight, abs=1e-13)
    assert all(step["column"][0]["potential_execution"]["energy_ev"] == 300000. for step in row["slices"])
    assert output.record["legacy_column_executed"] is False
    if kinetic_budget:
        from temsim.physics.potential_material_phase_budget import material_kinetic_summary
        summary = material_kinetic_summary(row["column_transport_context"])
        initial = KineticPhaseLedger.from_dict(entrance.record["kinetic_phase_ledger"])
        assert initial.used_phase_rad < summary["history_used_phase_rad"] <= .01
        assert not row["column_transport_context"]["material_kinetic_history"]["completed_energy_segments"]


@pytest.mark.parametrize("case", ("dynamic", "support", "tilt", "entrance"))
def test_unadmitted_material_field_geometry_rejects_before_atom_preparation(tmp_path, monkeypatch, case):
    state, entrance = _fixture()
    snapshot = _snapshot(state, dynamic=case == "dynamic")
    if case == "support":
        state.sample.eds_support_material_key = "copper"
    if case == "tilt":
        from temsim.specimen.geometry import quaternion_from_euler_xyz_deg
        state.sample.specimen_orientation_quaternion_wxyz = quaternion_from_euler_xyz_deg((1., 0., 0.))
    if case == "entrance":
        entrance = replace(entrance, plane_z_mm=np.nextafter(entrance.plane_z_mm, np.inf))
    monkeypatch.setattr("temsim.physics.specimen_wave_transport._prepare_material_grid",
        lambda *args, **kwargs: pytest.fail("Unadmitted case reached potential preparation"))
    with pytest.raises(ValueError):
        _run(state, entrance, snapshot, tmp_path)


def test_material_cannot_silently_discard_a_malformed_cumulative_kinetic_ledger(tmp_path, monkeypatch):
    state, entrance = _fixture()
    snapshot = _snapshot(state)
    # Invalid accounting cannot become an empty budget at a material boundary.
    entrance = replace(entrance, record={**entrance.record, "kinetic_phase_ledger": {"active": True}})
    monkeypatch.setattr("temsim.physics.specimen_wave_transport._prepare_material_grid",
        lambda *args, **kwargs: pytest.fail("Unsupported kinetic history reached material preparation"))
    monkeypatch.setattr("temsim.physics.potential_specimen_transport.observe_potential_mode",
        lambda *args, **kwargs: pytest.fail("Unsupported kinetic history reached propagation"))
    with pytest.raises(ValueError, match="kinetic phase ledger"):
        _run(state, entrance, snapshot, tmp_path)
    transport = PotentialMaterialTransport(snapshot, (entrance.plane_z_mm, state.sample.z_mm+.2e-6), "guard")
    with pytest.raises(ValueError, match="kinetic phase ledger"):
        transport(state, entrance, state.sample.z_mm, maximum_step_mm=.001,
            grid_numerics=WaveGridNumerics(), cancelled=lambda: False)
    with pytest.raises(ValueError, match="kinetic phase ledger"):
        PotentialMaterialTransport.context(entrance.record)
    assert PotentialMaterialTransport.context({"kinetic_phase_ledger": None}) == {"kinetic_phase_ledger": None}


def test_actual_si_energy_histories_preserve_the_same_cumulative_kinetic_prefix(tmp_path, monkeypatch):
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.potential_material_phase_budget import validate_material_kinetic_record, material_kinetic_summary
    state, entrance = _fixture()
    # This qualification uses a uniform Si slab over the entire numerical
    # window. A partially illuminated material/vacuum boundary is rejected by
    # the separate state-dependent conditioning test.
    state.sample.size_x_nm = state.sample.size_y_nm = 100.
    snapshot = _snapshot(state)
    # Execute a real nonzero-field prefix; do not invent a receipt at the face.
    upstream = replace(entrance, plane_z_mm=entrance.plane_z_mm-1e-6)
    entrance, = observe_potential_checkpoint(snapshot, upstream, (entrance.plane_z_mm*.001,),
        maximum_step_m=1e-9, kinetic_phase_ledger=KineticPhaseLedger(.01),
        grid_numerics=WaveGridNumerics(maximum_pixels=4096))
    original = KineticPhaseLedger.from_dict(entrance.record["kinetic_phase_ledger"])
    assert original.events and original.used_phase_rad > 0.
    monkeypatch.setattr("temsim.specimen.inelastic.real_inelastic_distribution", _certain_loss)
    store = ExecutedWaveStore(tmp_path, "kinetic-si-history", 1<<29)
    def run():
        return propagate_potential_specimen(state, entrance, snapshot, store=store,
            inelastic_numerics=InelasticWaveNumerics(trajectories_per_mode=2),
            grid_numerics=WaveGridNumerics(maximum_pixels=4096), maximum_step_mm=.001)
    result = run()
    assert len(result.beam.modes) == 2
    from temsim.immutable_json import thaw_json
    assert thaw_json(result.record["common_kinetic_prefix"]["ledger"]) == original.to_dict()
    prefixes = []
    for mode, row in zip(result.beam.modes, result.record["modes"], strict=True):
        context = row["column_transport_context"]
        history = validate_material_kinetic_record(context)
        prefixes.append(history["prefix"])
        assert mode.energy_kev == pytest.approx(299.8)
        segments = history["completed_energy_segments"]
        assert len(segments) == 2
        assert [segment["ledger"]["binding"]["energy_ev"] for segment in segments] == [300000., 299900.]
        first = KineticPhaseLedger.from_dict(segments[0]["ledger"])
        assert first.events[:len(original.events)] == original.events
        assert first.used_phase_rad > original.used_phase_rad
        active = KineticPhaseLedger.from_dict(context["kinetic_phase_ledger"])
        assert active.binding["energy_ev"] == pytest.approx(299800.) and not active.events
        summary = material_kinetic_summary(context)
        assert original.used_phase_rad < summary["history_used_phase_rad"] <= .01
        assert active.maximum_phase_rad == summary["remaining_phase_rad"]
    assert prefixes[0] == prefixes[1] == result.record["common_kinetic_prefix"]
    restored = run()
    assert restored.record["kinetic_phase_histories"] == result.record["kinetic_phase_histories"]
    assert restored.record["common_kinetic_prefix"] == result.record["common_kinetic_prefix"]


def test_actual_si_slices_rebase_static_electric_energy_only_at_real_collisions(tmp_path, monkeypatch):
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    from temsim.physics.planar_gun_field import PlanarGunField
    from temsim.physics.electric_wave_reference import ElectricWaveHistory
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.potential_material_phase_budget import material_kinetic_summary, validate_material_kinetic_record
    from temsim.physics.potential_geometry import capture_potential_geometry
    from temsim.detector.wave_readout import WaveReadoutOptions
    from test_wave_detector_readout import detector
    state, entrance = _fixture()
    state.sample.size_x_nm = state.sample.size_y_nm = 100.
    start = entrance.plane_z_mm*.001
    stop = (state.sample.z_mm+.2e-6)*.001
    z = np.array([start-1e-9, start, stop, stop+1e-9])
    r = np.array([0., .001])
    # A weak nonzero static E acts throughout BOTH real Si slices. Native
    # node voltages, not a rounded global-Z subtraction, define its gain.
    voltage = np.broadcast_to(np.array([-.001, 0., .0004, .0014]), (2, 4)).copy()
    electric = ElectricPotentialSnapshot.from_provider(PlanarGunField({}, r, z, voltage))
    magnetic = _snapshot(state)
    snapshot = ElectromagneticPotentialSnapshot(magnetic.scene,
        (*magnetic.contributions, PotentialContribution("material-E", ("material-E",), electric, "electric")))
    # Keep the controlled atomic-resolution entrance prefix within its
    # declared full-lattice kinetic budget, as in the magnetic qualification.
    upstream = replace(entrance, plane_z_mm=(start-1e-9)*1000.)
    entrance, = observe_potential_checkpoint(snapshot, upstream, (start,),
        maximum_step_m=1e-9, kinetic_phase_ledger=KineticPhaseLedger(.01),
        grid_numerics=WaveGridNumerics(maximum_pixels=4096))
    initial = entrance.beam.modes[0]
    assert initial.energy_kev > 300.
    initial_reference = initial.axial_reference
    original_electric = ElectricWaveHistory.from_dict(entrance.record["electric_wave_history"])
    monkeypatch.setattr("temsim.specimen.inelastic.real_inelastic_distribution", _certain_loss)
    def forbidden(*_args, **_kwargs):
        pytest.fail("Electric material transport cannot call legacy column propagation")
    monkeypatch.setattr("temsim.physics.specimen_wave_transport._propagate_column", forbidden)
    monkeypatch.setattr("temsim.physics.column_wave._propagate_column", forbidden)
    monkeypatch.setattr("temsim.physics.wave_imaging.explicit_pupil_spectrum", forbidden)
    store = ExecutedWaveStore(tmp_path, "actual-electric-Si", 1<<29)
    sensor = detector()
    sensor.z_mm = state.sample.z_mm+.2e-6+.1e-6
    sensor.outer_width_mm = .001
    geometry = capture_potential_geometry(state, start, sensor.z_mm*.001)
    def execute():
        return propagate_potential_specimen_to_detector(state, entrance, snapshot, geometry, sensor, store=store,
            inelastic_numerics=InelasticWaveNumerics(trajectories_per_mode=2),
            grid_numerics=WaveGridNumerics(maximum_pixels=4096), maximum_step_mm=.001,
            readout_options=WaveReadoutOptions(phase=True, complex_amplitude=True),
            maximum_readout_bytes=2*1024**3)
    result = execute()
    output = result.specimen
    assert len(output.beam.modes) == 2
    for mode, row in zip(output.beam.modes, output.record["modes"], strict=True):
        assert mode.energy_kev == pytest.approx(initial.energy_kev-.2+.0004*.001, rel=0., abs=2e-12)
        assert len(mode.scattering_history) == 2
        assert mode.axial_reference.flight_time_s > initial_reference.flight_time_s
        assert mode.axial_reference.longitudinal_action_j_s > initial_reference.longitudinal_action_j_s
        context = row["column_transport_context"]
        history = validate_material_kinetic_record(context)
        assert len(history["completed_energy_segments"]) == 2
        for segment, collision in zip(history["completed_energy_segments"], mode.scattering_history, strict=True):
            assert segment["collision"] == collision
            assert collision["input_energy_kev"]*1000. != segment["ledger"]["binding"]["energy_ev"]
        active = ElectricWaveHistory.from_dict(context["electric_wave_history"])
        assert active.reference.entrance_energy_ev == mode.energy_kev*1000.
        assert active.last_z_m == stop
        summary = material_kinetic_summary(context)["electric_forward"]
        assert summary["maximum_error"] == original_electric.maximum_forward_error
        assert original_electric.used_forward_error < summary["history_error"] <= summary["maximum_error"]
    for initial_mode, mode, evidence, observed in zip(output.beam.modes,
            result.detector_checkpoint.beam.modes, result.detector_checkpoint.record["modes"],
            result.readout.modes, strict=True):
        context = evidence["execution"]
        active = ElectricWaveHistory.from_dict(context["electric_wave_history"])
        assert mode.energy_kev > initial_mode.energy_kev
        assert mode.energy_kev == active.reference.axis_energy(snapshot, sensor.z_mm*.001)*.001
        assert observed.energy_kev == evidence["energy_kev"] == mode.energy_kev
        assert mode.scattering_history == initial_mode.scattering_history
        assert mode.axial_reference.flight_time_s > initial_mode.axial_reference.flight_time_s
        assert mode.axial_reference.longitudinal_action_j_s > initial_mode.axial_reference.longitudinal_action_j_s
        assert len(validate_material_kinetic_record(context)["completed_energy_segments"]) == 2
        assert result.readout.record["kinetic_phase_histories"][mode.mode_id] == material_kinetic_summary(context)
    assert result.readout.detected_probability.sum() == pytest.approx(result.detector_checkpoint.beam.total_weight, abs=2e-12)
    cached = execute().specimen
    assert cached.record["common_kinetic_prefix"] == output.record["common_kinetic_prefix"]
    assert cached.record["kinetic_phase_histories"] == output.record["kinetic_phase_histories"]
