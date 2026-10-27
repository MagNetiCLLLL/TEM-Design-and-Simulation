"""Local raw-phase pilots of real C3 + Probe native fields with a controlled mode.

These are finite static windows, not a physical-gun simulation or an accepted
end-to-end TEM/STEM column. No optical working point is retuned. The identical
controlled entrance is propagated under each explicitly changed snapshot.
"""
from functools import lru_cache
import json

import numpy as np
import pytest

from temsim.physics.potential_wave_observation import observe_potential_checkpoint, PotentialAperturePlane
from temsim.physics.tip_gun_wave import TipGunCheckpoint, TIP_REFERENCE
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_reference import AxialWaveReference
from temsim.physics.wavefront_comparison import compare_wavefronts
from test_potential_probe_pilot import _state, _capture
from test_potential_wave_carrier import _wave


CASES = {
    "HP2_thickness": ("probe_hp2_hexapole", "effective_length_mm", 1.2, 0., 1.142, 1.147),
    "A1_coarse_DPH2": ("probe_dph2_deflector", "kick_x_mrad", 1., .001, 1.142, 1.147),
    "A2_coarse_HP2": ("probe_hp2_hexapole", "strength_m3", 1.001, 0., 1.142, 1.147),
    "A1_fine_stigmator": ("condenser_stigmator", "strength_x_percent", 1., .01, 1.451, 1.452),
    "A2_fine_HPC": ("probe_hpc_hexapole", "strength_m3", 1., 100., 1.200, 1.201),
}


def _entrance(z_m):
    plane = _wave(n=64, pitch=5e-8, width=1.5e-7)
    mode = WaveMode(plane, .25, TIP_REFERENCE, "controlled:phase:0", 300., AxialWaveReference(0., 0.))
    return TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), z_m*1000., 1e-12,
        {"source_scope": "controlled Gaussian complex mode; not physical gun emission",
         "field_scope": "local static native-potential pilot"})


@lru_cache(maxsize=48)
def _run(mode, case, changed, step):
    # Build the same real preset independently. The test does not restore a
    # historical implementation-bound instrument graph during parallel work.
    state = _state(mode)
    key, attribute, factor, increment, start, stop = CASES[case]
    if changed:
        component = next(item for item in (*state.corrector_elements, *state.stigmators, *state.deflectors)
                         if item.key == key)
        setattr(component, attribute, getattr(component, attribute)*factor+increment)
    snapshot, _, _ = _capture(state, 3)
    apertures = tuple(PotentialAperturePlane(item.key, item.z_mm*.001, item.radius_mm*.001,
        (item.offset_x_mm*.001, item.offset_y_mm*.001)) for item in state.apertures
        if item.enabled and getattr(item, "installed", True) and getattr(item, "inserted", True)
        and start <= item.z_mm*.001 <= stop)
    outputs = observe_potential_checkpoint(snapshot, _entrance(start), ((start+stop)/2, stop),
        maximum_step_m=step, apertures=apertures)
    return snapshot, outputs


def _phase_rms(actual, reference):
    result = compare_wavefronts(actual, reference, "controlled:phase:0", remove_piston=False)
    values = result.difference_rad[result.valid_mask]
    assert len(values) > 100
    return float(np.sqrt(np.mean(values**2))), result


@pytest.mark.parametrize("mode", ["nano_probe", "nano_probe_large_angle"])
@pytest.mark.parametrize("case", tuple(CASES))
def test_actual_native_thickness_and_coarse_fine_drives_change_raw_phase_with_step_convergence(
        mode, case, record_testsuite_property):
    responses, original_outputs, changed_outputs = [], [], []
    for step in (.00025, .000125):
        original_snapshot, original = _run(mode, case, False, step)
        changed_snapshot, changed = _run(mode, case, True, step)
        assert len(original_snapshot.contributions) == len(changed_snapshot.contributions) == 29
        assert changed_snapshot.numerical_digest != original_snapshot.numerical_digest
        original_outputs.append(original[-1])
        changed_outputs.append(changed[-1])
        response, comparison = _phase_rms(changed[-1], original[-1])
        responses.append(response)
        assert comparison.metadata["piston_removed"] is False
        for output in (*original, *changed):
            assert output.beam.total_weight == pytest.approx(.25, abs=2e-12)
            assert output.record["production_admission"] is False
            assert output.beam.modes[0].axial_reference.longitudinal_action_j_s > 0
    original_error, _ = _phase_rms(original_outputs[0], original_outputs[1])
    changed_error, _ = _phase_rms(changed_outputs[0], changed_outputs[1])
    assert responses[-1] > 1e-10
    assert abs(responses[0]-responses[1]) < .05*responses[-1]+1e-10
    assert max(original_error, changed_error) < 1e-3
    evidence = {"mode": mode, "perturbation": case, "parameters": CASES[case],
        "scope": __doc__, "snapshot_digest": changed_snapshot.numerical_digest,
        "phase_rms_response_rad": responses, "unmodified_phase_step_error_rad": original_error,
        "modified_phase_step_error_rad": changed_error,
        "intensity_or_phase_renormalization": False, "piston_tilt_defocus_removed": False,
        "relative_response_gate": .05, "absolute_phase_step_gate_rad": 1e-3,
        "coherent_mode_count": 1, "production_admission": False}
    record_testsuite_property(mode+"_"+case, json.dumps(evidence, sort_keys=True))


def test_C3_actual_mode_strength_changes_phase_for_the_same_controlled_entrance(record_testsuite_property):
    outputs = []
    for mode in ("nano_probe", "nano_probe_large_angle"):
        state = _state(mode)
        snapshot, _, _ = _capture(state, 3)
        result, = observe_potential_checkpoint(snapshot, _entrance(.914), (.916,), maximum_step_m=.000125)
        outputs.append(result)
    response, comparison = _phase_rms(outputs[1], outputs[0])
    assert response > 1e-5
    assert comparison.metadata["piston_removed"] is False
    record_testsuite_property("C3_local_mode_phase", json.dumps({"interval_m": [.914, .916],
        "raw_phase_rms_difference_rad": response, "source_scope": "identical controlled entrance",
        "production_admission": False}))


@pytest.mark.parametrize("mode", ["nano_probe", "nano_probe_large_angle"])
@pytest.mark.parametrize("compiled_backend", ["native_compiled", "native_plane"])
def test_actual_29_source_compiled_sampling_preserves_local_raw_phase(mode, compiled_backend, record_testsuite_property):
    snapshot, _, _ = _capture(_state(mode), 3)
    outputs = []
    for backend in ("snapshot", compiled_backend):
        result, = observe_potential_checkpoint(snapshot, _entrance(1.142), (1.147,),
            maximum_step_m=.00025, sampling_backend=backend,
            scalar_splitting=True, reselect_carrier=True)
        outputs.append(result)
    phase_error, comparison = _phase_rms(outputs[1], outputs[0])
    assert phase_error < 1e-8
    assert comparison.metadata["piston_removed"] is False
    assert outputs[1].beam.total_weight == pytest.approx(outputs[0].beam.total_weight, abs=2e-12)
    record_testsuite_property(mode+"_local_"+compiled_backend+"_phase", json.dumps({
        "phase_rms_difference_rad": phase_error, "source_count": len(snapshot.contributions),
        "snapshot_digest": snapshot.numerical_digest, "interval_m": [1.142, 1.147],
        "scope": "controlled local 29-source field-sampling parity; not a full-column convergence result",
        "production_admission": False}))
