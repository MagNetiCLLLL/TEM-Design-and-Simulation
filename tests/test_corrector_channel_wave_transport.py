"""Physical distributed channels reach the Cartesian wave and actual phase view.

Bounded single-element fixtures exercise real field capture and propagation;
they are not an instrument calibration or a gun-to-specimen qualification.
"""
from dataclasses import replace

import numpy as np
import pytest

from temsim.optics.electron_gun.tip_coherence import wavelength_m
from temsim.physics.column_wave import _propagate_column
from temsim.physics.wave_grid import WaveGridNumerics
from temsim.physics.wave_plane_observables import mode_phase_samples
from temsim.physics.wavefront_comparison import compare_wavefronts
from test_hexapole_thickness_response import isolated_hexapole_state
from test_wave_detector_readout import checkpoint


def _source(tilt_rad=None):
    source = replace(checkpoint(), plane_z_mm=.01)
    mode = source.beam.modes[0]
    # Centering permits an independent reflection-symmetry check. Preserve
    # the nonzero defocus carrier so readout cannot substitute envelope phase.
    mode = replace(mode, plane=replace(mode.plane,
        tilt_rad=np.zeros(2) if tilt_rad is None else np.asarray(tilt_rad)))
    return replace(source, beam=replace(source.beam, modes=(mode,)))


def _execute(state, source, step=.02, *, stop_z_mm=4.):
    state.step_mm = state.history_step_mm = step
    return _propagate_column(
        state, source, stop_z_mm, maximum_step_mm=step,
        grid_numerics=WaveGridNumerics(
            compute_backend="CPU", maximum_pixels=256, maximum_working_bytes=1 << 28,
            maximum_support_loss_fraction=0., maximum_spectral_loss_fraction=0.,
        ),
    )


def _full_field(result):
    mode = result.beam.modes[0]
    return mode.plane.full_amplitude(wavelength_m(mode.energy_kev*1000.))


def _assert_actual_phase(result, source, *, stop_z_mm=4.):
    mode = result.beam.modes[0]
    amplitude = _full_field(result)
    mask = abs(amplitude)**2 > 1e-5*np.max(abs(amplitude)**2)
    phase = mode_phase_samples(mode, maximum_working_bytes=1 << 24)
    np.testing.assert_allclose(np.exp(1j*phase[mask]), amplitude[mask]/abs(amplitude[mask]), atol=2e-12)
    # Focusing is still part of actual phase, not silently fitted away.
    assert np.linalg.norm(np.exp(1j*phase[mask])-
                          np.exp(1j*np.angle(mode.plane.amplitude[mask]))) > .1
    assert result.plane_z_mm == stop_z_mm
    assert result.record["upstream_digest"] == source.digest
    assert mode.mode_id == source.beam.modes[0].mode_id
    assert mode.energy_kev == pytest.approx(300., abs=1e-12)
    assert result.beam.total_weight == pytest.approx(source.beam.total_weight, rel=1e-10)
    return phase, mask


@pytest.mark.parametrize("orientation,reflection_axis", ((0., 1), (np.pi/6, 0)), ids=("normal", "skew"))
def test_signed_real_hexapole_field_changes_executed_phase_and_intensity(orientation, reflection_axis):
    source = _source()
    saved = source.beam.modes[0].plane.full_amplitude(wavelength_m(300000.)).copy()
    results = [_execute(isolated_hexapole_state(.2, sign*1e6, orientation_rad=orientation), source)
               for sign in (1., -1.)]
    for result in results:
        _assert_actual_phase(result, source)
    positive, negative = map(_full_field, results)
    assert np.linalg.norm(positive-negative) > .1
    assert np.linalg.norm(abs(positive)**2-abs(negative)**2) > 1e-7
    # Changing the sign of a normal (skew) harmonic reflects a centered round
    # beam in X (Y). This checks real propagation, not a local phase formula.
    indices = (-np.arange(positive.shape[reflection_axis])) % positive.shape[reflection_axis]
    reflected = np.take(positive, indices, axis=reflection_axis)
    assert np.linalg.norm(negative-reflected) < 1e-6  # negligible periodic boundary tails
    np.testing.assert_array_equal(source.beam.modes[0].plane.full_amplitude(wavelength_m(300000.)), saved)


def test_changing_finite_width_reaches_actual_phase_and_probability_distribution():
    source = _source()
    results = [_execute(isolated_hexapole_state(width, 1e6), source) for width in (.1, .2)]
    phases = [_assert_actual_phase(result, source) for result in results]
    short, long = map(_full_field, results)
    mask = phases[0][1] & phases[1][1]
    assert np.linalg.norm(np.exp(1j*phases[0][0][mask])-np.exp(1j*phases[1][0][mask])) > .1
    assert np.linalg.norm(short-long) > .03
    assert np.linalg.norm(abs(short)**2-abs(long)**2) > 5e-8


def test_distributed_cartesian_field_is_convergent_under_axial_step_halving():
    source = _source()
    values = [_full_field(_execute(isolated_hexapole_state(.1, 1e6), source, step))
              for step in (.04, .02, .01)]
    coarse_delta = np.linalg.norm(values[0]-values[1])
    fine_delta = np.linalg.norm(values[1]-values[2])
    # The split field/propagation operator is second order in axial spacing.
    # All three calculations use the same resolved lattice without cropping.
    assert 1e-12 < fine_delta < .35*coarse_delta
    assert fine_delta < 1e-9


def test_measured_candidate_applies_to_real_state_and_changes_propagated_phase():
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.column import default_state
    from temsim.optics.corrector_channels import (
        apply_channel_change, calibrate_channel, prepare_channel_change,
    )

    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection("FEG", "C3 + Image Corrector", "No Energy Filter"))
    # Preserve real installed component classes, complete State serialization
    # and snapshot identities, while isolating one short physical field. The
    # moved sample/reference make this a bounded component test, not a full
    # instrument configuration or a validated corrector working point.
    for collection in (state.lenses, state.stigmators, state.corrector_elements, state.deflectors):
        for component in collection:
            component.enabled = False
    hexapole = next(c for c in state.corrector_elements if c.key == "image_hpol_hexapole")
    hexapole.enabled = True
    hexapole.strength_m3 = 1e6
    hexapole.orientation_rad = 0.
    hexapole.effective_length_mm = .2
    output = next(c for c in state.corrector_elements if c.key == "image_sad_plane")
    output.enabled = True
    output.z_mm = hexapole.z_mm+2.
    state.sample.z_mm = hexapole.z_mm-2.
    state.apertures = []
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.step_mm = state.history_step_mm = .02
    state.image_aberrations = {"c3_mm": 123., "a2_mm": .4}
    source = replace(_source((1e-8, -2e-8)), plane_z_mm=state.sample.z_mm)
    before = _execute(state, source, stop_z_mm=output.z_mm)
    baseline_digest = capture_instrument_snapshot(state).physical_digest
    calibration = calibrate_channel(state, "image", "a2_fine", semiangle_mrad=5., maximum_step_mm=.02)
    delta = calibration.jacobian @ (calibration.steps*np.array((-.25, .125)))
    candidate = prepare_channel_change(state, calibration, delta)
    assert capture_instrument_snapshot(state).physical_digest == baseline_digest
    assert {item["key"] for item in candidate.control_changes} == {hexapole.key}
    assert {item["field"] for item in candidate.control_changes} == {"strength_m3", "orientation_rad"}
    assert candidate.diagnostics["independent_candidate_measured"]
    assert not candidate.diagnostics["compensation_phase_applied"]

    apply_channel_change(state, candidate)
    assert capture_instrument_snapshot(state).physical_digest == candidate.resulting_state_digest
    assert state.image_aberrations == {"c3_mm": 123., "a2_mm": .4}
    after = _execute(state, source, stop_z_mm=output.z_mm)
    for result in (before, after):
        _assert_actual_phase(result, source, stop_z_mm=output.z_mm)
        assert np.linalg.norm(result.beam.modes[0].plane.tilt_rad) > 1e-8
    a, b = _full_field(before), _full_field(after)
    # Physical channel application reaches the propagated Cartesian field and
    # its probability, without injecting the diagnostic coefficient values.
    assert np.linalg.norm(a-b) > 1e-5
    assert np.linalg.norm(abs(a)**2-abs(b)**2) > 1e-11
    comparison = compare_wavefronts(
        after, before, source.beam.modes[0].mode_id,
        maximum_working_bytes=1 << 24, relative_intensity_threshold=1e-5,
    )
    # Compare the actual phases at identical physical coordinates, including
    # their tilt/defocus carriers. No phase piston or tilt is fitted away.
    assert not comparison.metadata["piston_removed"]
    assert np.linalg.norm(comparison.difference_rad[comparison.valid_mask]) > 1e-4
