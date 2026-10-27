"""Derived scan drives must be the actual fields shared by rays and waves."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal

from temsim.calculation_cache import state_model_signature
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.physics import scan_preparation as preparation


@pytest.fixture(autouse=True)
def empty_preparation_cache():
    preparation._PREPARED.clear()
    yield
    preparation._PREPARED.clear()


def _state():
    state = default_state()
    state.acceleration_backend = "CPU"
    state.acceleration_enabled = False
    state.step_mm = 0.2
    state.descan_deflector.scan_enabled = False
    return state


def _resolved_test_drives(state):
    # Deliberately cross-coupled matrices catch a lost off-diagonal entry;
    # setters also change scalar compatibility readbacks.
    state.ac_deflector.set_pure_shift_coupling(((-1.2, .1), (-.2, -1.3)))
    state.ac_deflector._pure_shift_angle_target = "canonical"
    state.ac_deflector.set_scan_command_matrix_mrad(((.001, .0002), (-.0003, .002)))
    state.descan_deflector.set_image_plane_coupling(((-.7, .3), (.2, -.8)),
        target_key="test_target", target_z_mm=1800.)
    state.descan_deflector.set_scan_command_matrix_mrad(((-.001, -.0002), (.0003, -.002)))
    state.descan_deflector.scan_pixels_x = state.ac_deflector.scan_pixels_x


def test_real_automatic_preparation_is_private_and_reused(monkeypatch):
    import temsim.physics.scan_geometry as geometry
    live = _state()
    before = capture_instrument_snapshot(live)
    working = before.restore()
    real = geometry.calibrate_scan_system
    calls = []

    def calculate(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(geometry, "calibrate_scan_system", calculate)
    preparation.prepare_scan_drives(working, before.digest, observation_stop_z_mm=working.sample.z_mm)
    assert calls == [1]
    assert working.ac_deflector._scan_scale_calibrated
    assert not np.array_equal(working.ac_deflector.scan_command_matrix_mrad,
                              live.ac_deflector.scan_command_matrix_mrad)
    identity = preparation.scan_drive_identity(working)
    second = before.restore()
    preparation.prepare_scan_drives(second, before.digest, observation_stop_z_mm=second.sample.z_mm)
    assert calls == [1]
    assert preparation.scan_drive_identity(second) == identity
    assert capture_instrument_snapshot(live).digest == before.digest
    assert second.electron_gun.to_dict() == live.electron_gun.to_dict()


def test_held_record_is_restored_and_scaled_without_overwriting_live_controls(monkeypatch):
    from temsim.physics.scan_calibration import restore_held
    import temsim.physics.scan_geometry as geometry
    live = _state()
    ac = live.ac_deflector
    record = dict(version=1, ac_ratio=[[-1., 0.], [0., -1.]],
        descan_ratio=[[-.5, 0.], [0., -.5]], command_mrad=[[.001, .0002], [.0003, .002]],
        fov_nm=[ac.scan_field_of_view_x_nm, ac.scan_field_of_view_y_nm],
        target_z_mm=1800., target_key="test", descan_calibrated=False,
        reference=ac.scan_reference)
    ac.calibration_mode = "held"
    ac.calibration_record_json = json.dumps(record)
    ac.scan_pixel_size_nm *= 2
    snapshot = capture_instrument_snapshot(live)
    working = snapshot.restore()
    # Exercise the real persistent-record restoration without spending time
    # on its independent geometric residual measurement.
    monkeypatch.setattr(geometry, "calibrate_scan_system", lambda state, **_: restore_held(state))
    preparation.prepare_scan_drives(working, snapshot.digest, observation_stop_z_mm=working.sample.z_mm)
    np.testing.assert_array_equal(working.ac_deflector.scan_command_matrix_mrad,
                                  np.array(record["command_mrad"])*2)
    assert working.ac_deflector.calibration_record_json == ac.calibration_record_json
    assert working.ac_deflector.calibration_mode == "held"
    assert capture_instrument_snapshot(live).digest == snapshot.digest


def test_upstream_observation_does_not_require_downstream_held_record():
    state = _state()
    state.ac_deflector.calibration_mode = "held"
    snapshot = capture_instrument_snapshot(state)
    preparation.prepare_scan_drives(state, snapshot.digest,
        observation_stop_z_mm=state.ac_deflector.upper_z_mm-state.ac_deflector.effective_thickness_mm)
    assert capture_instrument_snapshot(state).digest == snapshot.digest
    with pytest.raises(ValueError, match="No held scan calibration"):
        preparation.prepare_scan_drives(state, snapshot.digest,
            observation_stop_z_mm=state.ac_deflector.upper_z_mm-1.)


@pytest.mark.parametrize("unused", ("disabled", "upstream"))
@pytest.mark.parametrize("record", ("", "{not-json}"))
def test_unused_held_record_can_be_identified_without_certifying_execution(unused, record):
    from temsim.physics.scan_calibration import restore_held
    state = _state()
    ac = state.ac_deflector
    ac.calibration_mode = "held"
    ac.calibration_record_json = record
    ac.scan_enabled = unused != "disabled"
    stop = (ac.upper_z_mm-ac.effective_thickness_mm
            if unused == "upstream" else state.sample.z_mm)
    preparation.prepare_scan_drives(state, "unused-held", observation_stop_z_mm=stop)
    identity = preparation.scan_drive_identity(state)
    assert identity == preparation.scan_drive_identity(state)
    # Invalid records remain distinct captured inputs, not a shared empty
    # canonical calibration, and are still rejected if execution consumes one.
    ac.calibration_record_json = record + " "
    assert preparation.scan_drive_identity(state) != identity
    with pytest.raises(ValueError):
        restore_held(state)


def test_posed_coil_leading_support_prepares_before_unposed_boundary(monkeypatch):
    import temsim.physics.scan_geometry as geometry
    from temsim.physics.instrument_magnetic import column_dipole_fields
    state = _state()
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, "rotation_y_mrad": 200.})
        if part.key == "ac_deflector" else part for part in state._resolved_assembly.parts))
    unposed = state.ac_deflector.upper_z_mm-state.ac_deflector.effective_thickness_mm*.5
    leading = min(coil.field_support_mm[0] for coil in column_dipole_fields(state)
                  if "ac_deflector" in coil.drive_keys)
    assert leading < unposed
    stop = (leading+unposed)*.5
    calls = []
    monkeypatch.setattr(geometry, "calibrate_scan_system",
        lambda working, **options: calls.append(options["observation_stop_z_mm"]))
    preparation.prepare_scan_drives(state, "posed-request", observation_stop_z_mm=stop)
    assert calls == [stop]


def test_prepare_cache_keeps_dynamic_request_identity(monkeypatch):
    import temsim.physics.scan_geometry as geometry
    calls = []
    monkeypatch.setattr(geometry, "calibrate_scan_system", lambda state, **_: (calls.append(1), _resolved_test_drives(state)))
    state = _state()
    for time in (0., .2):
        state.simulation_time_s = time
        snapshot = capture_instrument_snapshot(state)
        preparation.prepare_scan_drives(snapshot.restore(), snapshot.digest)
    assert calls == [1, 1]


@pytest.mark.parametrize("entry", ("direct", "session"))
def test_wave_entry_prepares_before_column_field_capture(monkeypatch, entry):
    from temsim.physics import compute_backend
    monkeypatch.setattr(compute_backend, "coherent_cuda_capability",
        lambda: compute_backend.BackendCapability(True, "Stops before column execution"))
    import temsim.physics.tip_wave_pipeline as pipeline
    import temsim.physics.scan_geometry as geometry
    state = _state()
    before = capture_instrument_snapshot(state)
    calls = []
    monkeypatch.setattr(geometry, "calibrate_scan_system", lambda working, **_: (calls.append(1), _resolved_test_drives(working)))

    class ReachedPreparedColumn(Exception):
        pass

    def prepared_column(working, *_args):
        assert working.ac_deflector._scan_scale_calibrated
        assert working.ac_deflector.scan_command_matrix_mrad[0][1] == .0002
        raise ReachedPreparedColumn()

    monkeypatch.setattr(pipeline, "_prepare_column", prepared_column)
    request = replace(pipeline.TipWaveRequest(), stop="plane", observation_z_mm=state.sample.z_mm)
    session = pipeline.TipWaveObservationSession(state, request)
    for index in range(2):
        with pytest.raises(ReachedPreparedColumn):
            if entry == "direct":
                pipeline.simulate_tip_wave(state, request, use_cache=False)
            else:
                session.observe(state.sample.z_mm-index*.01)
    assert calls == [1]
    assert capture_instrument_snapshot(state).digest == before.digest


def test_scan_dwells_keep_one_prepared_request_and_distinct_emission_times(monkeypatch):
    from temsim.physics import compute_backend
    monkeypatch.setattr(compute_backend, "coherent_cuda_capability",
        lambda: compute_backend.BackendCapability(True, "Substituted scan execution"))
    import temsim.physics.tip_wave_scan as scanning
    import temsim.physics.tip_wave_pipeline as pipeline
    import temsim.physics.scan_geometry as geometry
    calls, identities, times = [], [], []
    state = _state()
    state.ac_deflector.scan_pixels_x = state.ac_deflector.scan_lines = 2
    monkeypatch.setattr(geometry, "calibrate_scan_system", lambda working, **_: (calls.append(1), _resolved_test_drives(working)))

    def execute(working, request, snapshot, identity, **kwargs):
        preparation.prepare_scan_drives(working, identity, observation_stop_z_mm=working.sample.z_mm,
                                       session_cache=kwargs["scan_preparation_cache"])
        identities.append(identity)
        times.append(request.tip_time_s)
        # Other background work may evict every global entry during a scan;
        # this captured scan must retain one fixed set of resolved drives.
        with preparation._PREPARED_LOCK:
            preparation._PREPARED.clear()
        return SimpleNamespace()

    monkeypatch.setattr(pipeline, "_execute_tip_wave", execute)
    rows = list(scanning.simulate_tip_scan(state, use_cache=False))
    assert len(rows) == 4 and len(set(times)) == 4
    assert len(set(identities)) == 1 and calls == [1]


class _Calculations(QObject):
    result_ready = Signal(str, object, float)
    failed = Signal(str, str)
    progress_changed = Signal(str, int, int, str)
    progress_detail = Signal(object)

    def __init__(self):
        super().__init__()
        self.submissions = []
        self.pool = SimpleNamespace(coordinator=SimpleNamespace(register_retained=lambda *_: None))

    def submit_background(self, *args, **kwargs):
        self.submissions.append((args, kwargs))

    def invalidate_pending(self, **kwargs):
        pass


@pytest.mark.parametrize("value", [0., -0.25, 0.125, np.nextafter(0., 1.)])
def test_emission_epoch_is_finite_signed_and_keeps_exact_float_identity(value):
    from temsim.physics.emission_timing import resolve_emission_time_s, require_matching_emission_time
    state = SimpleNamespace(simulation_time_s=value)
    assert resolve_emission_time_s(state) == value
    assert require_matching_emission_time(state, SimpleNamespace(tip_time_s=None)) == value
    assert require_matching_emission_time(state, SimpleNamespace(tip_time_s=value)) == value
    with pytest.raises(ValueError, match="same tip emission time"):
        require_matching_emission_time(state, SimpleNamespace(tip_time_s=np.nextafter(value, np.inf)))


@pytest.mark.parametrize("value", [True, np.bool_(False), "0", None, np.nan, np.inf, -np.inf])
def test_emission_epoch_rejects_invalid_instrument_time(value):
    from temsim.physics.emission_timing import resolve_emission_time_s
    with pytest.raises(ValueError, match="finite real"):
        resolve_emission_time_s(SimpleNamespace(simulation_time_s=value))
    assert resolve_emission_time_s(SimpleNamespace()) == 0.


def test_pair_rejects_other_epoch_before_capture_cancellation_or_dispatch(qapp, monkeypatch):
    from temsim.gui import paired_beam_controller as pairing
    state = SimpleNamespace(simulation_time_s=.125)
    calculations = _Calculations()
    controller = pairing.PairedBeamController(lambda: state, calculations=calculations)
    controller.token = "retained-pair"
    retained = object()
    controller.context = retained
    monkeypatch.setattr(pairing, "capture_instrument_snapshot", lambda *_:
                        pytest.fail("A mismatched time must fail before capturing another pair"))
    monkeypatch.setattr(calculations, "invalidate_pending", lambda **_:
                        pytest.fail("An invalid new request must not cancel a valid old pair"))

    with pytest.raises(ValueError, match="same tip emission time"):
        controller.start(state, SimpleNamespace(tip_time_s=np.nextafter(.125, 1.)), {}, 9, .5)

    assert controller.token == "retained-pair"
    assert controller.context is retained
    assert calculations.submissions == []


def test_dynamic_pair_checks_executed_epoch_receipts_not_only_ui_values():
    from temsim.gui.paired_beam_controller import BeamPairContext
    state = _state()
    state.simulation_time_s = -.125
    snapshot = capture_instrument_snapshot(state)
    record = {"dynamic_snapshot_digest": "d"*64, "emission_time_s": -.125}
    result = SimpleNamespace(state_snapshot=state, simulation=SimpleNamespace(metrics=dict(record)))
    context = BeamPairContext("pair", snapshot.physical_digest, result, False,
        state.sample.z_mm, snapshot.digest, state_model_signature(state),
        state.electron_gun.emitter.ray_count, state.step_mm,
        preparation.scan_drive_identity(state), emission_time_s=-.125)
    wave = SimpleNamespace(instrument_digest=snapshot.digest,
        resolved_scan_identity=context.executed_scan_identity,
        request=SimpleNamespace(tip_time_s=-.125), checkpoint=SimpleNamespace(record=dict(record)))
    context.verify_particle()
    context.verify_wave(wave)
    for bad in (None, .125, np.nextafter(-.125, 0.)):
        wave.checkpoint.record = {**record, "emission_time_s": bad}
        with pytest.raises(ValueError, match="emission time"):
            context.verify_wave(wave)
        result.simulation.metrics = {**record, "emission_time_s": bad}
        with pytest.raises(ValueError, match="emission time"):
            context.verify_particle()
    result.simulation.metrics = dict(record)
    wave.checkpoint.record = {"dynamic_snapshot_digest": "d"*64}
    with pytest.raises(ValueError, match="lost.*emission time"):
        context.verify_wave(wave)
    wave.checkpoint.record = dict(record)
    wave.request = SimpleNamespace(tip_time_s=.125)
    with pytest.raises(ValueError, match="request emission time"):
        context.verify_wave(wave)


def test_pair_reads_only_known_recording_wrappers_and_checks_the_actual_retained_beam():
    from temsim.gui.paired_beam_controller import BeamPairContext
    from temsim.detector.wave_readout import _apply_recording_stop
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_flux import BeamState, WaveMode
    from temsim.immutable_json import thaw_json
    state = _state()
    snapshot = capture_instrument_snapshot(state)
    identity = ("a"*64, "b"*64, "c"*64)
    record = dict(zip(("unified_runtime_digest", "unified_field_digest", "unified_geometry_digest"), identity))
    record.update(emission_time_s=.125, dynamic_snapshot_digest="d"*64, unified_executed_plane_z_m=.5)
    plane = PlaneWave(np.full((16, 16), 1/16, complex), np.eye(2)*1e-6, np.zeros(2))
    from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
    mode = WaveMode(plane, .37, TIP_REFERENCE, "mode", 300.)
    original = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), 500., 1e-9, record)
    detector = SimpleNamespace(key="receiver", geometry="square", outer_width_mm=.004,
                               inner_diameter_mm=0., readout_enabled=False)
    transmitted = _apply_recording_stop(original, detector)
    detector.outer_width_mm = .006
    transmitted = _apply_recording_stop(transmitted, detector)
    context = BeamPairContext("pair", snapshot.physical_digest, None, False,
        state.sample.z_mm, snapshot.digest, state_model_signature(state),
        state.electron_gun.emitter.ray_count, state.step_mm,
        preparation.scan_drive_identity(state), identity, .125)
    wave = SimpleNamespace(instrument_digest=snapshot.digest,
        resolved_scan_identity=context.executed_scan_identity,
        request=SimpleNamespace(tip_time_s=.125), checkpoint=transmitted)
    context.verify_wave(wave)
    assert 0 < transmitted.beam.total_weight < original.beam.total_weight
    for mutation in ("unknown", "weight", "mode", "plane", "time", "field"):
        bad = thaw_json(transmitted.record)
        bad_plane = transmitted.plane_z_mm
        if mutation == "unknown":
            bad["schema"] = "untrusted-upstream-wrapper"
        elif mutation == "weight":
            bad["modes"][0]["transmitted_weight"] *= .5
        elif mutation == "mode":
            bad["modes"].append(dict(bad["modes"][0]))
        elif mutation == "plane":
            bad_plane = np.nextafter(bad_plane, np.inf)
        elif mutation == "time":
            bad["upstream"]["upstream"]["emission_time_s"] = .25
        else:
            bad["upstream"]["upstream"]["unified_field_digest"] = "e"*64
        wave.checkpoint = replace(transmitted, record=bad, plane_z_mm=bad_plane)
        with pytest.raises(ValueError):
            context.verify_wave(wave)
    # A valid unit-shape rounding residual must not invalidate an untouched
    # transmitted beam or be silently clamped to zero by identity validation.
    rounded = replace(original, beam=BeamState((replace(mode, plane=replace(
        plane, amplitude=plane.amplitude*np.sqrt(1+1e-14))),), TIP_REFERENCE))
    detector.centre_offset_x_mm = 10.
    missed = _apply_recording_stop(rounded, detector)
    residual = missed.record["modes"][0]["received_weight"]
    assert residual < 0.
    wave.checkpoint = missed
    context.verify_wave(wave)
    assert wave.checkpoint.record["modes"][0]["received_weight"] == residual
    bad = thaw_json(missed.record)
    bad["modes"][0]["received_weight"] = -.01
    wave.checkpoint = replace(missed, record=bad)
    with pytest.raises(ValueError, match="weights are invalid"):
        context.verify_wave(wave)


def test_unified_scan_captures_each_real_dwell_epoch_and_does_not_relabel_an_old_runtime(monkeypatch):
    import temsim.physics.tip_wave_scan as scanning
    from temsim.physics import compute_backend
    from temsim.simulation_modes import switch_mode
    from temsim.physics.emission_timing import require_matching_emission_time
    state = _state()
    switch_mode(state, "unified")
    state.simulation_time_s = -.1
    state.ac_deflector.scan_pixels_x = state.ac_deflector.scan_lines = 2
    before = capture_instrument_snapshot(state)
    seen = []
    def calculate(working, request, **options):
        assert request.wave_grid.compute_backend == "Require GPU"
        assert require_matching_emission_time(working, request) == request.tip_time_s
        captured = capture_instrument_snapshot(working)
        seen.append((captured.digest, request.tip_time_s))
        return SimpleNamespace(instrument_digest=captured.digest, request=request,
            detector=SimpleNamespace(record={"detector": "screen", "response_weight": .25}))
    monkeypatch.setattr(scanning, "simulate_tip_wave", calculate)
    monkeypatch.setattr(compute_backend, "coherent_cuda_capability", lambda:
        compute_backend.BackendCapability(True, "Substituted per-dwell dispatch"))
    request = scanning.TipWaveRequest()
    request = replace(request, wave_grid=replace(request.wave_grid, compute_backend="Require GPU"))
    samples = list(scanning.simulate_tip_scan(state, request, use_cache=False))
    assert len(samples) == len({identity for identity, _ in seen}) == 4
    assert len({time for _, time in seen}) == 4
    assert all(sample.scan_instrument_digest == before.digest for sample in samples)
    assert capture_instrument_snapshot(state).digest == before.digest
    accumulator = scanning.ScanResponseAccumulator()
    for sample in samples:
        accumulator.add(sample)
    np.testing.assert_array_equal(accumulator.arrays()["response_per_tip_electron"], .25)
    for key, value in (("dwell_instrument_digest", "f"*64),
                       ("tip_time_s", samples[0].tip_time_s+.1)):
        with pytest.raises(ValueError, match="captured"):
            scanning.ScanResponseAccumulator().add(replace(samples[0], **{key: value}))
    changed = scanning.ScanResponseAccumulator()
    changed.add(samples[0])
    with pytest.raises(ValueError, match="Instrument inputs changed"):
        changed.add(replace(samples[1], scan_instrument_digest="f"*64))


@pytest.mark.parametrize("override_numerics", [False, True])
def test_pair_hands_actual_particle_drives_to_wave_and_guards_them(qapp, monkeypatch, override_numerics):
    from temsim.gui import paired_beam_controller as pairing
    import temsim.physics.scan_geometry as geometry
    state = _state()
    before = capture_instrument_snapshot(state)
    controller = pairing.PairedBeamController(lambda: state, calculations=_Calculations())
    monkeypatch.setattr(pairing, "specimen_interactions_active", lambda _: False)
    monkeypatch.setattr(geometry, "calibrate_scan_system", lambda *_a, **_k: pytest.fail("GUI handoff must not recalibrate"))
    ready, errors = [], []
    controller.ready.connect(lambda *args: ready.append(args))
    controller.failed.connect(errors.append)
    count = 9 if override_numerics else state.electron_gun.emitter.ray_count
    step = 1. if override_numerics else state.step_mm
    request = object()
    controller.start(state, request, {}, count, step)
    executed = before.restore()
    executed.electron_gun.emitter.ray_count = count
    executed.step_mm = step
    _resolved_test_drives(executed)
    assert state_model_signature(state) == state_model_signature(executed)
    controller._particle_ready("High accuracy", SimpleNamespace(workflow="rays", state_snapshot=executed), .01)
    assert not errors and len(ready) == 1
    prepared, _, _, context, _ = ready[0]
    assert ready[0][1] is request
    assert prepared.electron_gun.emitter.ray_count == count
    assert prepared.step_mm == step
    assert preparation.scan_drive_identity(prepared) == preparation.scan_drive_identity(executed)
    assert capture_instrument_snapshot(prepared).digest == context.instrument_identity
    assert controller.is_current()
    assert capture_instrument_snapshot(state).digest == before.digest
    # Wave preparation recognises this exact resolved execution and does not
    # silently solve again at another particle/wave numerical step.
    preparation.prepare_scan_drives(prepared, context.instrument_identity)
    context.verify_particle()
    wave = SimpleNamespace(instrument_digest=context.instrument_identity,
                           resolved_scan_identity=context.executed_scan_identity)
    context.verify_wave(wave)
    wave.resolved_scan_identity = "different-executed-drive"
    with pytest.raises(ValueError, match="wave scan drives"):
        context.verify_wave(wave)
    wave.resolved_scan_identity = preparation.scan_drive_identity(prepared,
        emission_time_s=getattr(prepared, "simulation_time_s", 0.)+.1)
    with pytest.raises(ValueError, match="wave scan drives"):
        context.verify_wave(wave)
    # Old retained results remain readable, but cannot be claimed as a newly
    # verified matched execution without actual drive provenance.
    with pytest.raises(ValueError, match="wave scan drives"):
        context.verify_wave(SimpleNamespace(instrument_digest=context.instrument_identity))
    executed.ac_deflector.set_scan_command_matrix_mrad(((.002, .0002), (-.0003, .002)))
    assert state_model_signature(state) == state_model_signature(executed)
    with pytest.raises(ValueError, match="scan drives"):
        context.verify_particle()
    state.mini_condenser.percent += 1
    assert not controller.is_current()


def test_scan_drive_identity_detects_physical_host_and_timing_changes():
    state = _state()
    original = preparation.scan_drive_identity(state)
    state.ac_deflector.scan_frame_period_s *= 2
    assert preparation.scan_drive_identity(state) != original
    original = preparation.scan_drive_identity(state)
    state.ac_deflector.effective_thickness_mm /= 2
    assert preparation.scan_drive_identity(state) != original
    original = preparation.scan_drive_identity(state)
    state.simulation_time_s = getattr(state, "simulation_time_s", 0.)+.1
    assert preparation.scan_drive_identity(state) != original


def test_observed_backend_does_not_change_inputs_but_requested_backend_policy_does():
    from temsim.physics.core import _record_active_backend
    state = _state()
    original = capture_instrument_snapshot(state)
    _record_active_backend(state, "CPU", "Actual bounded diagnostic")
    assert "Actual bounded diagnostic" in state.active_backend
    assert capture_instrument_snapshot(state).digest == original.digest
    from temsim.physics.compute_backend import BACKEND_CUDA
    _record_active_backend(state, BACKEND_CUDA)
    assert state.active_backend == f"{BACKEND_CUDA} (+ CPU auxiliaries)"
    assert capture_instrument_snapshot(state).digest == original.digest
    restored = capture_instrument_snapshot(state).restore()
    assert restored.acceleration_backend == state.acceleration_backend
    assert restored.acceleration_enabled == state.acceleration_enabled
    for attribute, value in (("acceleration_backend", "Require GPU"), ("acceleration_enabled", True)):
        changed = original.restore()
        setattr(changed, attribute, value)
        assert capture_instrument_snapshot(changed).digest != original.digest


def test_unified_pair_requires_executed_field_and_geometry_identity():
    """Synthetic receipts test the publication guard, not field accuracy."""
    from temsim.gui.paired_beam_controller import BeamPairContext
    from temsim.simulation_modes import switch_mode
    state = _state()
    switch_mode(state, "unified")
    snapshot = capture_instrument_snapshot(state)
    identity = ("a" * 64, "b" * 64, "c" * 64)
    keys = ("unified_runtime_digest", "unified_field_digest", "unified_geometry_digest")
    metrics = dict(zip(keys, identity))
    result = SimpleNamespace(state_snapshot=state, simulation=SimpleNamespace(metrics=metrics))
    context = BeamPairContext("pair", snapshot.physical_digest, result, False,
        state.sample.z_mm, snapshot.digest, state_model_signature(state),
        state.electron_gun.emitter.ray_count, state.step_mm,
        preparation.scan_drive_identity(state), identity)
    context.verify_particle()
    wave = SimpleNamespace(instrument_digest=snapshot.digest,
        resolved_scan_identity=context.executed_scan_identity,
        checkpoint=SimpleNamespace(record=dict(metrics)))
    context.verify_wave(wave)
    for key in keys:
        wave.checkpoint.record = {**metrics, key: "d" * 64}
        with pytest.raises(ValueError, match="potentials or geometry"):
            context.verify_wave(wave)
    wave.checkpoint.record = {}
    with pytest.raises(ValueError, match="potentials or geometry"):
        context.verify_wave(wave)
    metrics.pop("unified_geometry_digest")
    with pytest.raises(ValueError, match="lacks executed"):
        context.verify_particle()


def test_unified_pair_uses_executed_particle_numerics_for_one_actual_runtime(qapp, monkeypatch):
    """Real captures, substituted particle publication; no propagation claim."""
    from temsim.gui import paired_beam_controller as pairing
    from temsim.simulation_modes import switch_mode
    from temsim.physics.unified_field_runtime import get_unified_field_runtime
    state = _state()
    switch_mode(state, "unified")
    state.ac_deflector.scan_enabled = state.descan_deflector.scan_enabled = False
    controller = pairing.PairedBeamController(lambda: state, calculations=_Calculations())
    monkeypatch.setattr(pairing, "specimen_interactions_active", lambda _: False)
    ready, errors = [], []
    controller.ready.connect(lambda *args: ready.append(args))
    controller.failed.connect(errors.append)
    request = object()
    controller.start(state, request, {}, 9, 1.)
    executed = capture_instrument_snapshot(state).restore()
    executed.electron_gun.emitter.ray_count, executed.step_mm = 9, 1.
    runtime = get_unified_field_runtime(executed)
    metrics = dict(zip(("unified_runtime_digest", "unified_field_digest", "unified_geometry_digest"),
        (runtime.digest, runtime.snapshot.numerical_digest, runtime.geometry.digest)))
    result = SimpleNamespace(workflow="rays", state_snapshot=executed,
        simulation=SimpleNamespace(metrics=metrics))
    controller._particle_ready("High accuracy", result, .01)
    assert not errors and len(ready) == 1
    prepared, wave_request, _, context, _ = ready[0]
    assert wave_request is request
    assert get_unified_field_runtime(prepared) is runtime
    context.verify_particle()


@pytest.mark.parametrize("execution", ("matched", "mismatch", "legacy"))
def test_coherent_worker_only_compares_matching_executed_scan_drives(qapp, monkeypatch, execution):
    """Real Qt worker/guard; its numerical propagation is explicitly a stub."""
    from threading import Event
    import temsim.gui.coherent_beam as gui
    import temsim.gui.beam_plane_data as plane_data
    import temsim.physics.beam_comparison as comparison
    from temsim.gui.paired_beam_controller import BeamPairContext
    from temsim.physics.tip_wave_pipeline import TipWaveRequest, TipWaveResult
    state = _state()
    _resolved_test_drives(state)
    snapshot = capture_instrument_snapshot(state)
    drive_identity = preparation.scan_drive_identity(state)
    context = BeamPairContext("pair", snapshot.physical_digest,
        SimpleNamespace(state_snapshot=state), False, state.sample.z_mm,
        snapshot.digest, state_model_signature(state), state.electron_gun.emitter.ray_count,
        state.step_mm, drive_identity)
    request = TipWaveRequest(stop="plane", observation_z_mm=float(state.sample.z_mm))
    values = {} if execution == "legacy" else {
        "resolved_scan_identity": drive_identity if execution == "matched" else "other-drive"}
    result = TipWaveResult(SimpleNamespace(plane_z_mm=state.sample.z_mm), None,
                           request, False, snapshot.digest, **values)
    monkeypatch.setattr(gui, "simulate_tip_wave", lambda *_a, **_k: result)
    monkeypatch.setattr(gui, "_intensity_preview", lambda *_a, **_k:
        gui._Preview(np.ones((2, 2)), np.zeros((2, 2)), 1., 1, 32))
    monkeypatch.setattr(gui, "_plane_magnetic_gauge", lambda *_: None)
    sampled = []
    monkeypatch.setattr(plane_data, "sample_beam_plane", lambda *_:
        (sampled.append(1), SimpleNamespace())[1])
    monkeypatch.setattr(comparison, "compare_beam_planes", lambda *_: {"guard_fixture": True})
    worker = gui._WaveWorker(1, 1, state, request, Event(), pair_context=context)
    solved, failures = [], []
    worker.signals.solved.connect(lambda *args: solved.append(args))
    worker.signals.failed.connect(lambda *args: failures.append(args))
    worker.run()
    assert not failures and len(solved) == 1
    preview = solved[0][-1]
    if execution == "matched":
        assert sampled == [1] and preview.comparison == {"guard_fixture": True}
        assert preview.comparison_error is None
    else:
        assert not sampled and preview.comparison is None
        assert "wave scan drives" in preview.comparison_error


def test_concurrent_preparation_cache_eviction_keeps_private_outputs(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import temsim.physics.scan_geometry as geometry
    monkeypatch.setattr(geometry, "calibrate_scan_system", lambda state, **_: _resolved_test_drives(state))
    states = [_state() for _ in range(4)]
    barrier = Barrier(len(states))

    def work(index):
        state = states[index]
        barrier.wait(timeout=10)
        for iteration in range(40):
            key = f"request-{index}-{iteration}"
            preparation.prepare_scan_drives(state, key)
            preparation.prepare_scan_drives(state, key)
            assert state.ac_deflector.scan_command_matrix_mrad[0][1] == .0002
        return preparation.scan_drive_identity(state)

    with ThreadPoolExecutor(max_workers=len(states)) as pool:
        identities = list(pool.map(work, range(len(states))))
    assert len(set(identities)) == 1
    assert len(preparation._PREPARED) <= preparation._MAXIMUM_PREPARED


def test_calibration_model_versions_preparation_and_executed_drive_identity(monkeypatch):
    import temsim.physics.scan_geometry as geometry
    calls = []
    monkeypatch.setattr(geometry, "calibrate_scan_system",
        lambda state, **_: (calls.append(1), _resolved_test_drives(state)))
    state = _state()
    cache = {}
    preparation.prepare_scan_drives(state, "same-input", session_cache=cache)
    identity = preparation.scan_drive_identity(state)
    target = _state()
    preparation.copy_resolved_scan_drives(state, target)
    assert target.ac_deflector._pure_shift_angle_target == "canonical"
    preparation.prepare_scan_drives(target, "same-input", session_cache=cache)
    assert calls == [1]
    monkeypatch.setattr(preparation, "SCAN_CALIBRATION_MODEL", "different-algorithm")
    assert preparation.scan_drive_identity(target) != identity
    preparation.prepare_scan_drives(target, "same-input", session_cache=cache)
    assert calls == [1, 1]
