"""Focused ledger/control regressions, not column-physics qualification.

The support operator is an exactly solvable scalar contraction here. Actual
affine cropping, phase preservation and its measured error have separate tests
in test_wave_support.py. These checks exercise production column orchestration,
normalisation, streamed checkpoints and cumulative admission decisions.
"""
from dataclasses import replace
from types import SimpleNamespace
import math

import numpy as np
import pytest

from temsim.physics import column_wave as column
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError, refine_plane_wave
from temsim.optics.electron_gun.tip_coherence import wavelength_m
from test_column_wave_transport import (
    _quadratic_fixture, _resolved_non_gaussian_wave, _sample_full_field_at_points,
)
from test_wave_detector_readout import checkpoint


def _controlled_column(monkeypatch, *, transmission=1.):
    from temsim.physics import wave_support
    from test_tip_wave_pipeline import _quiet_prepared_column
    from test_wave_detector_readout import checkpoint

    source = checkpoint(two=True)
    state = object()  # All physical paths are supplied by the analytic fixture.
    prepared = _quiet_prepared_column(state, 2000., 2002., 1.)
    prepared[0].cs_kick_m3[1:] = 2.5
    calls = {"attempts": 0, "compactions": 0, "applied_norms": []}
    survival = .99

    def phase(wave, wavelength, *, spectral_tail_fraction, sampling_record, **strengths):
        assert strengths == {"spherical_m3": 2.5}
        calls["attempts"] += 1
        if calls["attempts"] % 2:
            raise WaveSamplingError("Controlled unresolved Cs sampling", 2.)
        calls["applied_norms"].append(wave.probability)
        return replace(wave, amplitude=wave.amplitude*np.exp(.13j))

    def compact(wave, *, maximum_loss_fraction, wavelength_m):
        calls["compactions"] += 1
        before = wave.probability
        return replace(wave, amplitude=wave.amplitude*math.sqrt(survival)), {
            "relative_l2_error": 1.-math.sqrt(survival),
            "absolute_probability_loss": before*(1.-survival),
        }

    def clip(wave, radius, stops, z, weight, **kwargs):
        if z == 2001.:
            return replace(wave, amplitude=wave.amplitude*math.sqrt(transmission)), []
        return wave, []

    monkeypatch.setattr(column, "apply_multipole_phase", phase)
    monkeypatch.setattr(wave_support, "compact_wave_support", compact)
    monkeypatch.setattr(column, "reselect_phase_carrier", lambda wave, *a, **kw: (wave, []))
    monkeypatch.setattr(column, "propagate_plane_wave", lambda wave, *a, **kw: wave)
    monkeypatch.setattr(column, "_clip", clip)
    monkeypatch.setattr(column, "numerical_thread_budget", lambda requested=None: 1)
    return state, source, prepared, calls, survival


def test_latest_mode_ledger_survives_wrappers_without_mutating_checkpoint():
    previous = {"reference_mode_weight": .6, "accumulated_relative_l2_bound": .004}
    newest = {**previous, "accumulated_relative_l2_bound": .007}
    record = {"upstream": {"modes": [
        {"mode_id": "target", "support_approximation": newest}],
        "upstream": {"support_error_by_mode": {"target": previous}}}}
    found = column._previous_support_error(record, "target")
    assert found == newest
    found["accumulated_relative_l2_bound"] = 0.
    assert newest["accumulated_relative_l2_bound"] == .007
    assert column._previous_support_error(record, "absent") is None


def test_sampling_evidence_survives_segments_and_counts_only_applied_operators(monkeypatch, tmp_path):
    state, source, prepared, calls, survival = _controlled_column(monkeypatch)
    actual_phase = column.apply_multipole_phase
    def phase(wave, wavelength, *, spectral_tail_fraction, sampling_record, **strengths):
        assert spectral_tail_fraction == 1e-8
        sampling_record.update(ignored_probability_fraction_yx=[.9, .9],
            combined_bandwidth_rad_per_step_yx=[4., 4.])
        result = actual_phase(wave, wavelength, spectral_tail_fraction=spectral_tail_fraction,
            sampling_record=sampling_record, **strengths)
        sampling_record.update(ignored_probability_fraction_yx=[2e-9, 3e-9],
            combined_bandwidth_rad_per_step_yx=[1.2, 1.3])
        return result
    monkeypatch.setattr(column, "apply_multipole_phase", phase)
    numerics = WaveGridNumerics(multipole_spectral_tail_fraction=1e-8,
        maximum_support_loss_fraction=.1, maximum_accumulated_support_l2=.1)
    args = dict(store=ExecutedWaveStore(tmp_path/"executed", "sampling-ledger-fixture", 1 << 26),
        segment_steps=1, maximum_step_mm=1., grid_numerics=numerics, _prepared=prepared)
    output, hit = column._propagate_column_segmented(state, source, 2002., **args)
    assert not hit
    for mode in output.beam.modes:
        evidence = column._previous_multipole_sampling(output.record, mode.mode_id)
        assert evidence["accepted_operations"] == 2
        assert evidence["maximum_ignored_spectral_probability_fraction_yx"] == [2e-9, 3e-9]
        assert evidence["maximum_combined_bandwidth_rad_per_step_yx"] == [1.2, 1.3]
        # Returned evidence is independently owned, including nested arrays.
        evidence["maximum_ignored_spectral_probability_fraction_yx"][0] = 1.
        assert column._previous_multipole_sampling(output.record, mode.mode_id)[
            "maximum_ignored_spectral_probability_fraction_yx"][0] == 2e-9
    restored, hit = column._propagate_column_segmented(state, source, 2002., **args)
    assert hit and restored.digest == output.digest
    assert calls["attempts"] == 8


@pytest.mark.parametrize("loss_budget,l2_budget,accepted", [
    (0., .1, 0), (.1, .0075, 1), (.1, .1, 2),
])
def test_streamed_ledger_is_cumulative_and_declined_crop_still_applies_cs(
        monkeypatch, tmp_path, loss_budget, l2_budget, accepted):
    state, source, prepared, calls, survival = _controlled_column(monkeypatch)
    numerics = WaveGridNumerics(compute_backend="CPU",
        maximum_support_loss_fraction=loss_budget,
        maximum_accumulated_support_l2=l2_budget)
    store = ExecutedWaveStore(tmp_path/"executed", "support-ledger-fixture", 1 << 26)
    args = dict(store=store, segment_steps=1, maximum_step_mm=1.,
                grid_numerics=numerics, _prepared=prepared)
    result, hit = column._propagate_column_segmented(state, source, 2002., **args)
    assert not hit
    error = 1.-math.sqrt(survival)
    expected_l2 = sum(error*survival**(j/2.) for j in range(accepted))
    for initial, actual in zip(source.beam.modes, result.beam.modes):
        ledger = column._previous_support_error(result.record, actual.mode_id)
        assert ledger["reference_mode_weight"] == initial.weight_per_reference_electron
        assert ledger["accumulated_relative_l2_bound"] == pytest.approx(expected_l2)
        assert ledger["probability_loss_per_tip_electron"] == pytest.approx(
            initial.weight_per_reference_electron*(1.-survival**accepted), abs=2e-16)
        assert actual.weight_per_reference_electron == pytest.approx(
            initial.weight_per_reference_electron*survival**accepted)
        # Both physical Cs events executed even when a support crop was denied.
        # Denial now takes the strict sampling refinement; compare the same
        # physical Fourier interpolant on its legitimately finer lattice.
        from temsim.physics.wave_grid import refine_plane_wave
        expected = refine_plane_wave(initial.plane, actual.plane.amplitude.shape)
        np.testing.assert_allclose(actual.plane.amplitude,
                                   expected.amplitude*np.exp(.26j), atol=1e-16)
    assert calls["attempts"] == 8  # Two attempts at each of two events/modes.
    assert calls["compactions"] == (0 if loss_budget == 0 else 4)
    count = calls["attempts"]
    restored, hit = column._propagate_column_segmented(state, source, 2002., **args)
    assert hit and calls["attempts"] == count
    assert restored.digest == result.digest
    for mode in restored.beam.modes:
        assert column._previous_support_error(restored.record, mode.mode_id)[
            "accumulated_relative_l2_bound"] == pytest.approx(expected_l2)


@pytest.mark.parametrize("segmented", [False, True])
def test_cumulative_l2_scales_with_surviving_flux_before_or_after_normalisation(
        monkeypatch, tmp_path, segmented):
    state, source, prepared, calls, survival = _controlled_column(monkeypatch, transmission=.25)
    numerics = WaveGridNumerics(compute_backend="CPU",
        maximum_support_loss_fraction=.1, maximum_accumulated_support_l2=.1)
    args = dict(maximum_step_mm=1., grid_numerics=numerics, _prepared=prepared)
    if segmented:
        result, _ = column._propagate_column_segmented(state, source, 2002.,
            store=ExecutedWaveStore(tmp_path/"executed", "flux-ledger-fixture", 1 << 26),
            segment_steps=1, **args)
    else:
        result = column._propagate_column(state, source, 2002., **args)
    error = 1.-math.sqrt(survival)
    expected_l2 = error*(1.+math.sqrt(survival*.25))
    for initial, actual in zip(source.beam.modes, result.beam.modes):
        ledger = column._previous_support_error(result.record, actual.mode_id)
        assert ledger["reference_mode_weight"] == initial.weight_per_reference_electron
        assert ledger["accumulated_relative_l2_bound"] == pytest.approx(expected_l2)
        assert ledger["relative_intensity_l1_bound"] == pytest.approx(2*expected_l2+expected_l2**2)
        assert ledger["probability_loss_per_tip_electron"] == pytest.approx(
            initial.weight_per_reference_electron*(1.-survival)*(1.+survival*.25), abs=2e-16)
        assert actual.weight_per_reference_electron == pytest.approx(
            initial.weight_per_reference_electron*survival**2*.25)


def test_material_branches_and_slices_inherit_support_ledger_and_separate_losses(monkeypatch):
    from types import SimpleNamespace
    from temsim.physics import specimen_wave_transport as specimen
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_flux import BeamState
    from test_wave_detector_readout import checkpoint

    source = checkpoint()
    mode_id = source.beam.modes[0].mode_id
    parent = {"reference_mode_weight": .8, "accumulated_relative_l2_bound": .02,
              "probability_loss_per_tip_electron": .0004}
    source = replace(source, record={"support_error_by_mode": {mode_id: parent}})
    state = SimpleNamespace(sample=SimpleNamespace(z_mm=2000.000001,
                            thickness_nm=2., wave_bandwidth_fraction=1.))
    x = (np.arange(64)-32)*1e-6
    scene = SimpleNamespace(sample_contains_xy=lambda x, y: np.ones_like(x, dtype=bool))
    prepared = SimpleNamespace(potential_configurations_v_angstrom=(
        np.zeros((2, 64, 64)), np.zeros((2, 64, 64))), metrics={"fixture": True})
    material = scene, prepared, x, x, np.array((10., 10.))
    monkeypatch.setattr(specimen, "_regrid_refined_mode", lambda mode, *a: (mode, {}))
    monkeypatch.setattr(specimen, "_material_phase", lambda mode, *a, **kw: (mode, {}))
    monkeypatch.setattr(specimen, "_bandlimit_mode", lambda mode, *a, **kw: (mode, 0.))
    monkeypatch.setattr(specimen, "_attenuate_zero_loss", lambda mode, *a:
                        (mode, {"events": [], "removed_weight": 0.}))
    distribution = SimpleNamespace(total_inelastic_mean_free_path_nm=math.inf,
        absorption_mean_free_path_nm=math.inf, material_key="fixture", model="fixture",
        beam_energy_kev=300., warnings=(), channels=())
    monkeypatch.setattr("temsim.specimen.inelastic.real_inelastic_distribution", lambda state: distribution)
    incoming = []

    def transport(state, local, stop, **kwargs):
        mode = local.beam.modes[0]
        ledger = column._previous_support_error(local.record, mode.mode_id)
        incoming.append((mode.mode_id, dict(ledger)))
        ledger["accumulated_relative_l2_bound"] += .001
        ledger["probability_loss_per_tip_electron"] += .00005
        output = replace(mode, weight_per_reference_electron=mode.weight_per_reference_electron-.00007)
        return TipGunCheckpoint(BeamState((output,), local.beam.reference_plane), stop,
            local.reference_current_a, {"modes": [{"mode_id": mode.mode_id,
                                                   "support_approximation": ledger}]})

    monkeypatch.setattr(specimen, "_propagate_column", transport)
    result = specimen._propagate_specimen_attempt(state, source, maximum_step_mm=.5,
        maximum_checkpoint_bytes=1 << 26, grid_numerics=WaveGridNumerics(specimen_phase_method="sampled"),
        tip_time_s=None, cancelled=lambda: False, progress_callback=None,
        material=material, refinement_factor=1, refinement_history=[])
    assert len(result.beam.modes) == 2 and len(incoming) == 4
    for config in range(2):
        rows = incoming[2*config:2*config+2]
        assert all(row[0] == mode_id+f"/phonon:{config}" for row in rows)
        assert [row[1]["reference_mode_weight"] for row in rows] == [.4, .4]
        assert [row[1]["accumulated_relative_l2_bound"] for row in rows] == pytest.approx([.02, .021])
        assert [row[1]["probability_loss_per_tip_electron"] for row in rows] == pytest.approx([.0002, .00025])
        actual = result.beam.modes[config]
        ledger = column._previous_support_error(result.record, actual.mode_id)
        assert ledger["reference_mode_weight"] == .4
        assert ledger["accumulated_relative_l2_bound"] == pytest.approx(.022)
        assert ledger["probability_loss_per_tip_electron"] == pytest.approx(.0003)
        row = result.record["modes"][config]
        assert row["numerical_support_loss"] == pytest.approx(.0001)
        assert all(s["numerical_support_loss"] == pytest.approx(.00005) for s in row["slices"])
        assert row["probability_balance"]["column_absorbed"] == pytest.approx(.00004)
        assert row["probability_balance"]["accounted_weight"] == pytest.approx(.3)
    assert parent["accumulated_relative_l2_bound"] == .02
    assert parent["probability_loss_per_tip_electron"] == .0004


def _carrier_source():
    original = checkpoint()
    mode = replace(original.beam.modes[0], plane=_resolved_non_gaussian_wave())
    return replace(original, beam=replace(original.beam, modes=(mode,)))


@pytest.mark.parametrize('event,expected_checks', [('linear', 1), ('cs', 2), ('mask', 2)])
def test_carrier_checks_follow_non_linear_changes_and_match_every_step_reference(monkeypatch, event, expected_checks):
    prepared = _quadratic_fixture(6)
    plan = prepared[0]
    plan.midpoint_sx_m2[:] = 3.
    plan.midpoint_sy_m2[:] = 5.
    plan.midpoint_sxy_m2[:] = .4
    if event == 'cs':
        plan.cs_kick_m3[3] = 1e10
    elif event == 'mask':
        prepared[2][3] = (SimpleNamespace(key='test-half-plane', radius_mm=1.,
            transmission_mask=lambda x, y: x < .0002),)
    source = _carrier_source()
    reselect = column.reselect_phase_carrier
    transport = column.propagate_plane_wave
    selections = []

    def observed(wave, *args, **kwargs):
        selections.append(wave.amplitude.shape)
        return reselect(wave, *args, **kwargs)

    monkeypatch.setattr(column, 'reselect_phase_carrier', observed)
    numerics = WaveGridNumerics(compute_backend='CPU', maximum_pixels=2048)
    args = dict(_prepared=prepared, _combine_linear=False, grid_numerics=numerics)
    actual = column._propagate_column(object(), source, plan.z_mm[-1], **args)
    assert len(selections) == expected_checks

    def every_step(wave, matrix, offset, wavelength, **kwargs):
        selected, _ = reselect(wave, wavelength, grid_numerics=numerics)
        return transport(selected, matrix, offset, wavelength, **kwargs)

    monkeypatch.setattr(column, 'propagate_plane_wave', every_step)
    reference = column._propagate_column(object(), source, plan.z_mm[-1], **args)
    a, b = actual.beam.modes[0], reference.beam.modes[0]
    assert a.weight_per_reference_electron == pytest.approx(b.weight_per_reference_electron, abs=2e-10)
    yy, xx = np.meshgrid(np.linspace(-1.5e-6, 1.5e-6, 9), np.linspace(-1.5e-6, 1.5e-6, 11))
    points = np.stack((xx.ravel(), yy.ravel()))
    wavelength = float(wavelength_m(a.energy_kev*1000))
    full_a = _sample_full_field_at_points(a.plane, wavelength, points)
    full_b = _sample_full_field_at_points(b.plane, wavelength, points)
    # Absolute complex comparison, including scalar/lift phase. No fitted
    # global phase, translated beam or replacement intensity distribution.
    relative_error = np.linalg.norm(full_a-full_b)/np.linalg.norm(full_b)
    if event != 'mask':
        assert relative_error < 2e-8
    else:
        # A hard edge on 64² cells is nonsmooth and not band-limited. Gauge
        # resampling is therefore not a roundoff-equivalence test here. Bound
        # the complete-field discrepancy and check it decreases after one
        # independent 4x spatial refinement, without moving the physical mask.
        # The discontinuity needs this independently chosen 256² input mesh;
        # retain the existing 1.5 ppm fine-field gate across FFT grid policies.
        assert relative_error < 1e-5  # Ten ppm complex-field gate for this coarse hard edge.
        mode = source.beam.modes[0]
        refined_mode = replace(mode, plane=refine_plane_wave(mode.plane,(256,256),numerics=numerics))
        refined_source = replace(source, beam=replace(source.beam,modes=(refined_mode,)))
        monkeypatch.setattr(column, 'propagate_plane_wave', transport)
        finer = column._propagate_column(object(),refined_source,plan.z_mm[-1],**args)
        monkeypatch.setattr(column, 'propagate_plane_wave', every_step)
        finer_reference = column._propagate_column(object(),refined_source,plan.z_mm[-1],**args)
        fa = _sample_full_field_at_points(finer.beam.modes[0].plane,wavelength,points)
        fb = _sample_full_field_at_points(finer_reference.beam.modes[0].plane,wavelength,points)
        fine_error = np.linalg.norm(fa-fb)/np.linalg.norm(fb)
        assert fine_error < min(1.5e-6,.75*relative_error)


def test_unchanged_linear_carrier_retries_typed_sampling_failure(monkeypatch):
    prepared = _quadratic_fixture(3)
    original = column.propagate_plane_wave
    reselect = column.reselect_phase_carrier
    calls, selections = [], []

    def observed(wave, *args, **kwargs):
        selections.append(wave.amplitude.shape)
        return reselect(wave, *args, **kwargs)

    def fail_once(wave, *args, **kwargs):
        calls.append(wave.amplitude.shape)
        if len(calls) == 2:
            raise WaveSamplingError('Controlled alternate-carrier request', 2.)
        return original(wave, *args, **kwargs)

    monkeypatch.setattr(column, 'reselect_phase_carrier', observed)
    monkeypatch.setattr(column, 'propagate_plane_wave', fail_once)
    result = column._propagate_column(object(), _carrier_source(), prepared[0].z_mm[-1],
        _prepared=prepared, _combine_linear=False,
        grid_numerics=WaveGridNumerics(compute_backend='CPU'))
    assert len(selections) == 2  # Initial selection, then the failed chart.
    assert len(calls) == 4  # All three physical maps execute, with one retry.
    assert result.beam.total_weight == pytest.approx(_carrier_source().beam.total_weight, abs=1e-12)


def test_optional_carrier_does_not_expand_an_already_resolved_physical_grid(monkeypatch):
    prepared = _quadratic_fixture(3)
    requests = []

    def needs_expansion(wave, wavelength, *, grid_numerics, **kwargs):
        requests.append(grid_numerics.automatic_refinement)
        raise WaveSamplingError('Controlled optional chirp needs expansion',2.)

    monkeypatch.setattr(column,'reselect_phase_carrier',needs_expansion)
    source = _carrier_source()
    result = column._propagate_column(object(),source,prepared[0].z_mm[-1],
        _prepared=prepared,_combine_linear=False,
        grid_numerics=WaveGridNumerics(compute_backend='CPU'))
    assert requests == [False]
    assert result.beam.modes[0].plane.amplitude.shape == source.beam.modes[0].plane.amplitude.shape
    assert result.beam.total_weight == pytest.approx(source.beam.total_weight,abs=1e-12)


def _hexapole_compaction_fixture(monkeypatch, *, known_exit=True):
    """Real cubic/quartic phases; identity linear maps isolate scheduling.

    The observation spy does not project the field, so the resulting complete
    complex amplitude also has a closed-form independent polynomial reference.
    This checks operation ownership across checkpoints, not column calibration.
    """
    from temsim.physics import wave_support
    from temsim.physics.multipole_wave import apply_multipole_phase
    from temsim.physics.wave_reference import AxialWaveReference

    prepared = _quadratic_fixture(3 if known_exit else 2)
    plan = prepared[0]
    plan.midpoint_hex_normal_m3[:2] = (2e7, -1e7)
    plan.midpoint_hex_skew_m3[:2] = (1e7, 2e7)
    if known_exit:
        plan.cs_kick_m3[-1] = 1e9
    source = checkpoint()
    initial = replace(_resolved_non_gaussian_wave(), origin_m=np.zeros(2),
                      curvature_m1=np.zeros((2, 2)), tilt_rad=np.zeros(2))
    initial = refine_plane_wave(initial, (512, 512), numerics=WaveGridNumerics(compute_backend='CPU'))
    mode = replace(source.beam.modes[0], plane=initial,
                   axial_reference=AxialWaveReference(1e-9, 1e-22))
    source = replace(source, beam=replace(source.beam, modes=(mode,)))
    operations, compactions = [], []

    def observed_phase(wave, wavelength, **strengths):
        result = apply_multipole_phase(wave, wavelength, **strengths)
        operations.append(strengths)
        return result

    def observed_projection(wave, *, maximum_loss_fraction):
        compactions.append(len(operations))
        return wave, None

    monkeypatch.setattr(column, 'apply_multipole_phase', observed_phase)
    monkeypatch.setattr(wave_support, 'compact_wave_spectrum', observed_projection)
    monkeypatch.setattr(column, 'reselect_phase_carrier', lambda wave, *a, **kw: (wave, []))
    monkeypatch.setattr(column, 'propagate_plane_wave', lambda wave, *a, **kw: wave)
    monkeypatch.setattr(column, 'numerical_thread_budget', lambda requested=None: 1)
    return source, prepared, operations, compactions


@pytest.mark.parametrize('segment_steps', (None, 1, 2))
def test_hexapole_grid_is_retained_across_segments_until_true_exit_and_cs_is_unchanged(
        monkeypatch, tmp_path, segment_steps):
    source, prepared, operations, compactions = _hexapole_compaction_fixture(monkeypatch)
    plan = prepared[0]
    numerics = WaveGridNumerics(compute_backend='CPU')
    if segment_steps is None:
        result = column._propagate_column(object(), source, plan.z_mm[-1],
            _prepared=prepared, grid_numerics=numerics)
    else:
        store = ExecutedWaveStore(tmp_path/'hp', 'hexapole-scheduling-fixture', 1 << 28)
        result, hit = column._propagate_column_segmented(object(), source, plan.z_mm[-1],
            _prepared=prepared, grid_numerics=numerics, store=store, segment_steps=segment_steps)
        assert not hit
    # Four real HP half-steps, followed by exactly one existing discrete Cs.
    assert len(operations) == 5
    assert sum(bool(row.get('spherical_m3')) for row in operations) == 1
    # No projection at the internal checkpoint or half-step. One at the true
    # HP exit, and one after the unchanged discrete-Cs path.
    assert compactions == [4, 5]
    original = source.beam.modes[0]
    x, y = original.plane.coordinates_m()
    hn = float(plan.step_m @ plan.midpoint_hex_normal_m3)
    hs = float(plan.step_m @ plan.midpoint_hex_skew_m3)
    cs = float(plan.cs_kick_m3.sum())
    phase = 2*np.pi/float(wavelength_m(original.energy_kev*1000))*(
        -hn*(x**3/3-x*y*y)-hs*(x*x*y-y**3/3)-cs*(x*x+y*y)**2/4)
    expected = original.plane.amplitude*np.exp(1j*phase)
    actual = result.beam.modes[0]
    np.testing.assert_allclose(actual.plane.amplitude, expected, rtol=2e-12, atol=2e-14)
    assert actual.weight_per_reference_electron == pytest.approx(original.weight_per_reference_electron, abs=2e-13)
    assert actual.axial_reference.flight_time_s > original.axial_reference.flight_time_s
    assert actual.axial_reference.longitudinal_action_j_s > original.axial_reference.longitudinal_action_j_s


def test_observation_endpoint_inside_hexapole_does_not_claim_physical_exit(monkeypatch):
    source, prepared, operations, compactions = _hexapole_compaction_fixture(monkeypatch, known_exit=False)
    column._propagate_column(object(), source, prepared[0].z_mm[-1], _prepared=prepared,
        grid_numerics=WaveGridNumerics(compute_backend='CPU'))
    assert len(operations) == 4
    assert compactions == []


def test_nested_plan_slices_keep_the_original_hexapole_exit():
    prepared = _quadratic_fixture(6)
    prepared[0].midpoint_hex_normal_m3[:4] = 1.
    prefix = column._slice_prepared(prepared, 0, 4)
    middle = column._slice_prepared(prefix, 1, 3)
    end = column._slice_prepared(prefix, 3, 4)
    assert column._hexapole_exit_nodes(prefix[0]) == (4,)
    assert column._hexapole_exit_nodes(middle[0]) == ()
    assert column._hexapole_exit_nodes(end[0]) == (1,)


@pytest.mark.parametrize("stop,expected", ((4., ()), (5., (5.,)), (6., (5.,))))
@pytest.mark.parametrize("vector_only", (False, True))
def test_prepared_hexapole_exit_uses_physical_union_despite_cancelled_fields(monkeypatch, stop, expected, vector_only):
    from dataclasses import dataclass
    from temsim.physics.core import hexapole_field_components

    @dataclass(frozen=True)
    class FixturePlan:
        z_mm: np.ndarray
        midpoint_hex_normal_m3: np.ndarray
        midpoint_hex_skew_m3: np.ndarray
        signature: str = "opposing-hexapole-fixture"
        mapped_fields: tuple = ()
        thin_power_m1: tuple = ()
        thin_rotation_rad: tuple = ()
        hexapole_exit_nodes: tuple | None = None

    def component(key, lower, upper, strength):
        return SimpleNamespace(key=key, enabled=True, strength_m3=strength,
            field_support_mm=lambda cutoff: (lower, upper),
            hexapole_strength_m3=lambda z: strength*np.ones_like(z))

    state = SimpleNamespace(step_mm=1., corrector_elements=(
        component("a", -1., 3., 1.), component("b", 1., 5., -1.)))
    if vector_only:
        # Core also accepts custom vector providers without a scalar strength
        # or key. They must still prevent an exit while their support overlaps.
        custom = state.corrector_elements[1]
        del custom.hexapole_strength_m3, custom.strength_m3, custom.key
        custom.hexapole_strength_components_m3 = lambda z: (-np.ones_like(z), np.zeros_like(z))
    def prepared(state, start, stop, events, *, save_z_mm, maximum_step_mm):
        z = np.unique(np.r_[np.arange(start, stop, maximum_step_mm), stop, save_z_mm])
        normal, skew = hexapole_field_components((z[:-1]+z[1:])/2, state)
        return FixturePlan(z, normal, skew)

    monkeypatch.setattr(column, "build_propagation_plan", prepared)
    monkeypatch.setattr(column, "_wave_apertures", lambda *a: ())
    monkeypatch.setattr(column, "_wave_bores", lambda *a: ((), ()))
    monkeypatch.setattr(column, "_component_events", lambda *a: ((), ()))
    monkeypatch.setattr("temsim.physics.wave_field_admission.require_supported_column_wave_fields", lambda *a: None)
    plan = column._prepare_column(state, 0., stop, 1.)[0]
    assert tuple(plan.z_mm[list(column._hexapole_exit_nodes(plan))]) == expected
    # The cancellation between the still-present fields is not an exit.
    assert plan.midpoint_hex_normal_m3[1] == 0.
    assert plan.signature != "opposing-hexapole-fixture"


def _independent_mode_fixture():
    from temsim.physics.wave_flux import BeamState
    source = checkpoint()
    original = source.beam.modes[0]
    modes = tuple(replace(original, mode_id=f"independent:{i}", weight_per_reference_electron=weight,
        plane=replace(original.plane, amplitude=np.roll(original.plane.amplitude, 3*i, axis=1)))
        for i, weight in enumerate((.7, .01, .02, .04)))
    return replace(source, beam=BeamState(modes, source.beam.reference_plane))


def test_independent_mode_omission_uses_total_budget_and_keeps_absolute_weights():
    from temsim.physics.wave_mode_support import select_incoherent_modes, incoherent_mode_approximation
    source = _independent_mode_fixture()
    digest = source.digest
    selected = select_incoherent_modes(source, maximum_loss_probability=.035)
    evidence = incoherent_mode_approximation(selected.record)
    assert [m.mode_id for m in selected.beam.modes] == ["independent:0", "independent:3"]
    assert all(a is b for a, b in zip(selected.beam.modes, (source.beam.modes[0], source.beam.modes[3])))
    assert source.digest == digest and selected.reference_current_a == source.reference_current_a
    assert selected.plane_z_mm == source.plane_z_mm
    assert evidence["omitted_probability_per_tip_electron"] == pytest.approx(.03)
    assert .03 <= evidence["intensity_l1_error_bound_per_tip_electron"] <= .035
    assert not evidence["renormalised"]
    # A per-mode threshold would also remove .02 under .025; the total guard
    # keeps it because .01+.02 exceeds that caller's one global allowance.
    tighter = select_incoherent_modes(source, maximum_loss_probability=.025)
    assert [m.mode_id for m in tighter.beam.modes] == ["independent:0", "independent:2", "independent:3"]
    strict = select_incoherent_modes(source, maximum_loss_probability=0.)
    assert strict.beam.modes == source.beam.modes
    low = replace(source, beam=replace(source.beam, modes=tuple(
        replace(m, weight_per_reference_electron=1e-5) for m in source.beam.modes)))
    assert len(select_incoherent_modes(low, maximum_loss_probability=.1).beam.modes) == 1


@pytest.mark.parametrize("budget", [-1., .101, float("nan"), float("inf"), True])
def test_independent_mode_budget_rejects_invalid_values(budget):
    from temsim.physics.wave_mode_support import select_incoherent_modes
    with pytest.raises(ValueError, match="loss budget"):
        select_incoherent_modes(_independent_mode_fixture(), maximum_loss_probability=budget)


def test_independent_mode_intensity_bound_after_lossless_and_absorbing_channels():
    from temsim.physics.wave_mode_support import select_incoherent_modes, incoherent_mode_approximation
    source = _independent_mode_fixture()
    selected = select_incoherent_modes(source, maximum_loss_probability=.035)
    evidence = incoherent_mode_approximation(selected.record)
    for absorbing in (False, True):
        def density(modes):
            total = np.zeros(source.beam.modes[0].plane.amplitude.shape)
            for mode in modes:
                field = np.fft.fft2(mode.plane.amplitude, norm="ortho")
                if absorbing:
                    field[:, ::2] = 0.
                field = np.fft.ifft2(field, norm="ortho")
                total += mode.weight_per_reference_electron*abs(field)**2
            return total
        full, kept = density(source.beam.modes), density(selected.beam.modes)
        exact_omission = density((source.beam.modes[1], source.beam.modes[2]))
        np.testing.assert_allclose(full-kept, exact_omission, rtol=1e-12, atol=2e-18)
        error = float(np.sum(abs(full-kept)))
        assert error <= evidence["intensity_l1_error_bound_per_tip_electron"]+1e-15
        if not absorbing:
            assert error == pytest.approx(.03, abs=1e-15)


def test_independent_mode_budget_is_spent_once_through_segment_and_branch_records():
    from temsim.physics.wave_mode_support import select_incoherent_modes, incoherent_mode_approximation
    selected = select_incoherent_modes(_independent_mode_fixture(), maximum_loss_probability=.035)
    record = selected.record
    for schema in ("executed-column-segment-v1", "executed-specimen-wave-v1", "executed-recording-stop-v1"):
        record = {"schema": schema, "upstream": record}
    branch = replace(selected, beam=replace(selected.beam, modes=tuple(
        replace(m, mode_id=m.mode_id+"/phonon:0", weight_per_reference_electron=m.weight_per_reference_electron*.001)
        for m in selected.beam.modes)), record=record)
    assert select_incoherent_modes(branch, maximum_loss_probability=.035) is branch
    assert incoherent_mode_approximation(branch.record) == incoherent_mode_approximation(selected.record)
    with pytest.raises(ValueError, match="restart from the complete gun"):
        select_incoherent_modes(branch, maximum_loss_probability=0.)


def test_pipeline_independent_mode_selection_cache_and_strict_budget_reuse_gun(monkeypatch, tmp_path):
    from temsim.physics.compute_backend import coherent_cuda_capability
    capability = coherent_cuda_capability()
    if not capability.available:
        pytest.skip(f"CuPy CUDA unavailable: {capability.detail}")
    from collections import OrderedDict
    from temsim.physics import tip_wave_pipeline as pipeline
    from temsim.physics.wave_mode_support import incoherent_mode_approximation
    from test_tip_wave_pipeline import _quiet_observation_state, _quiet_prepared_column, _observation_request
    state = _quiet_observation_state()
    start = state.electron_gun.exit_plane_z_mm
    source = replace(_independent_mode_fixture(), plane_z_mm=start)
    calls = []

    def gun(*args, **kwargs):
        calls.append("gun")
        return source

    monkeypatch.setattr(pipeline, "_STAGES", OrderedDict())
    monkeypatch.setattr(pipeline, "_OBSERVATION_PREFIXES", OrderedDict())
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", gun)
    monkeypatch.setattr(pipeline, "_prepare_column", _quiet_prepared_column)
    monkeypatch.setattr(column, "_prepare_column", _quiet_prepared_column)
    request = _observation_request(tmp_path, start+.3, segmented=True,
        wave_grid=WaveGridNumerics(compute_backend="Require GPU", maximum_incoherent_mode_loss_probability=.035))
    first = pipeline.simulate_tip_wave(state, request)
    second = pipeline.simulate_tip_wave(state, request)
    assert len(first.checkpoint.beam.modes) == len(second.checkpoint.beam.modes) == 2
    assert second.propagation_cache_hit and calls == ["gun"]
    assert first.checkpoint.digest == second.checkpoint.digest
    assert incoherent_mode_approximation(second.checkpoint.record)["omitted_probability_per_tip_electron"] == pytest.approx(.03)
    strict = pipeline.simulate_tip_wave(state, replace(request,
        wave_grid=replace(request.wave_grid, maximum_incoherent_mode_loss_probability=0.)))
    assert len(strict.checkpoint.beam.modes) == 4 and calls == ["gun"]
    assert incoherent_mode_approximation(strict.checkpoint.record)["omitted_probability_per_tip_electron"] == 0.
    assert source.beam.total_weight == pytest.approx(.77)
