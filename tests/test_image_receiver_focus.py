"""Image projection follows the physical receiver without changing illumination."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.validate_nanoprobe_condenser import _configured_state
from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import encode_instrument, decode_instrument
from temsim.operating_modes import apply_projector_mode, apply_operating_mode_pair, mode_by_key
from temsim.optics.direct_alignment import canonical_source_basis
from temsim.physics.first_order import trace_transverse_transfer


def _state(receiver="flu_screen", objective=68.9801):
    state, _, _, _ = _configured_state("C3 + Probe Corrector", "normal")
    state.simulation_mode = "custom"
    state.acceleration_enabled = True
    state.acceleration_backend = "Numba CPU"
    state.step_mm = .05
    state.objective_lens.percent = objective
    state.fluorescent_screen.inserted = receiver == "flu_screen"
    state.camera.inserted = receiver == "camera"
    for detector in state.stem_detectors:
        detector.inserted = detector.readout_enabled = False
    state.ac_deflector.calibration_mode = "held"
    state.descan_deflector.scan_enabled = False
    return state


def _nonprojector(state):
    copied = decode_instrument(encode_instrument(state))
    copied.projector_mode = "diffraction"
    for lens in copied.lenses:
        if lens.key in mode_by_key("imaging").devices:
            lens.percent = 0.
    return json_digest(encode_instrument(copied))


@pytest.mark.parametrize("receiver,objective", [
    ("flu_screen", 68.9801), ("camera", 68.9801), ("flu_screen", 54.3),
])
def test_image_focus_uses_sample_conjugacy_at_receiver_and_preserves_incident_graph(receiver, objective):
    state = _state(receiver, objective)
    before = _nonprojector(state)
    result = apply_projector_mode(state, "imaging")
    assert _nonprojector(state) == before
    focus = result.image_projection
    assert focus.receiver_key == receiver
    assert focus.receiver_inserted
    assert focus.image_residual_m_per_rad <= 2.e-5
    assert focus.magnification == pytest.approx(focus.target_magnification, rel=1.e-3)
    # canonical_transfers fixes its full-field path at .025 mm. Request the
    # finer production Jacobian explicitly so this is a real refinement.
    transfer = trace_transverse_transfer(state, state.sample.z_mm,
                                        focus.receiver_z_mm, maximum_step_mm=.0125)
    matrix = transfer.matrix @ canonical_source_basis(state, state.sample.z_mm)
    residual = np.linalg.norm(matrix[:2, 2:], ord=2)
    assert residual <= 2.e-5
    assert np.sqrt(abs(np.linalg.det(matrix[:2, :2]))) == pytest.approx(focus.target_magnification, rel=1.e-3)
    lenses = {lens.key: lens for lens in state.lenses}
    for key in ("projector_lens_1", "projector_lens_2"):
        assert lenses[key].percent == mode_by_key("imaging").devices[key]["percent"]
    assert "First-order sample image" in result.summary
    assert "higher-order aberrations are not corrected" in result.summary


def test_complete_tem_pair_focuses_inserted_screen_and_reapplication_can_target_camera():
    state = _state()
    result = apply_operating_mode_pair(state, "micro_probe", "imaging")
    assert state.fluorescent_screen.inserted
    assert result.image_projection.receiver_key == "flu_screen"
    assert result.image_projection.image_residual_m_per_rad <= 2.e-5
    state.fluorescent_screen.inserted = False
    state.camera.inserted = True
    before = _nonprojector(state)
    result = apply_projector_mode(state, "imaging")
    assert result.image_projection.receiver_key == "camera"
    assert _nonprojector(state) == before
    assert result.image_projection.image_residual_m_per_rad <= 2.e-5


def test_retracted_receiver_is_explicit_optical_reference_not_exposure():
    state = _state("none")
    result = apply_projector_mode(state, "imaging")
    assert result.image_projection.receiver_key == "camera"
    assert not result.image_projection.receiver_inserted
    assert not state.camera.inserted and not state.fluorescent_screen.inserted
    assert "optical reference (retracted)" in result.summary


@pytest.mark.parametrize("pair", [False, True])
def test_failed_image_refocus_restores_complete_preset_transaction(monkeypatch, pair):
    state = _state()
    before = encode_instrument(state)
    def fail(candidate):
        next(lens for lens in candidate.lenses if lens.key == "diffraction_lens").percent = 1.
        raise ValueError("injected independent focus validation failure")
    monkeypatch.setattr("temsim.optics.projector_reference.focus_image_projector", fail)
    with pytest.raises(ValueError, match="independent focus validation"):
        if pair:
            apply_operating_mode_pair(state, "micro_probe", "imaging")
        else:
            apply_projector_mode(state, "imaging")
    assert encode_instrument(state) == before


def test_cancelled_image_focus_keeps_raw_currents_and_mode():
    state = _state()
    before = deepcopy(state.to_dict())
    state._tuning_cancelled = lambda: True
    with pytest.raises(RuntimeError, match="Superseded"):
        apply_projector_mode(state, "imaging")
    assert state.to_dict() == before


def test_diffraction_switch_retains_stored_strengths_and_unqualified_status():
    state = _state()
    before = _nonprojector(state)
    result = apply_projector_mode(state, "diffraction")
    assert _nonprojector(state) == before
    assert result.image_projection is None
    assert "unqualified" in result.summary
    assert {lens.key: lens.percent for lens in state.lenses if lens.key in result.changed_devices} == {
        key: value["percent"] for key, value in mode_by_key("diffraction").devices.items()}


def test_fixed_image_controls_have_fourth_order_screen_transfer_refinement():
    """A stable paired kick does not prove absolute Image positions converged.

    Hold the physical projector currents while refining the actual continuous
    field grid. Compare B differences, not its residual after separately
    refocusing each grid (which would hide the integration error).
    """
    state = _state()
    apply_projector_mode(state, "imaging")
    controls = {lens.key: lens.percent for lens in state.lenses}
    matrices = []
    for step in (.05, .025, .0125):
        state.step_mm = step
        transfer = trace_transverse_transfer(
            state, state.sample.z_mm, state.fluorescent_screen.z_mm,
            maximum_step_mm=step)
        matrices.append(transfer.matrix @ canonical_source_basis(state, state.sample.z_mm))
    changes = [np.linalg.norm((left-right)[:2, 2:], ord=2)
               for left, right in zip(matrices, matrices[1:])]
    # Fourth-order RK gives approximately 16 here; allow finite event-grid
    # end intervals, but reject no improvement or second-order degradation.
    assert changes[1] > 1.e-13
    assert 8. < changes[0]/changes[1] < 32.
    assert {lens.key: lens.percent for lens in state.lenses} == controls
