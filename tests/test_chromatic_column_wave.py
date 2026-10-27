"""Actual Objective/C2 wave routing of declared additional chromatic screens.

The input is a synthetic executed-mode fixture, not a full physical gun run.
Independent thin-action formula checks live in test_shared_chromatic_screens.
"""
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE, wavelength_m
from temsim.physics.column_wave import (
    _prepare_column, _propagate_column, _propagate_column_segmented, _slice_prepared,
)
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_grid import WaveGridNumerics
from temsim.physics.wave_plane_observables import mode_phase_samples
from temsim.physics.wave_reference import AxialWaveReference


LENSES = ("objective_lens", "condenser_lens_2")


def isolated_lens(key):
    original = default_state()
    lens = next(item for item in original.lenses if item.key == key)
    state = SimpleNamespace(lenses=[lens], objective_lens=original.objective_lens,
        condenser_system=original.condenser_system, _resolved_assembly=original._resolved_assembly,
        stigmators=[], corrector_elements=[], deflectors=[], apertures=[], recording_planes=[],
        simulation_mode="ideal", beam_voltage_kv=300., step_mm=.02, history_step_mm=.02,
        acceleration_enabled=True, acceleration_backend="CPU", projector_mode="diffraction",
        equivalent_image_lenses_enabled=False, sample=original.sample,
        chromatic_aberration_enabled=True, lens_intrinsic_aberrations={})
    axis = (np.arange(64)-32)*5e-9
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    amplitude = np.exp(-((xx-3e-9)**2+(yy+6e-9)**2)/(2*(20e-9)**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    plane = PlaneWave(amplitude, np.eye(2)*5e-9, np.array((45e-9, -30e-9)),
        np.zeros((2, 2)), np.array((2e-6, -1e-6)))
    modes = tuple(WaveMode(plane, weight, TIP_REFERENCE, f"synthetic-energy:{energy}", energy,
                          axial_reference=AxialWaveReference(2e-8, 1e-22))
                  for energy, weight in ((299.7, .18), (300., .27), (300.3, .11)))
    checkpoint = TipGunCheckpoint(BeamState(modes, TIP_REFERENCE), float(lens.z_mm)-.1,
        1e-9, {"fixture": "isolated real lens and synthetic mixed-energy incident modes"})
    return state, lens, checkpoint


def numerics(backend="CPU"):
    return WaveGridNumerics(compute_backend=backend, maximum_pixels=256,
        maximum_working_bytes=512*1024**2, maximum_device_working_bytes=512*1024**2)


def full_field(mode):
    return mode.plane.full_amplitude(float(wavelength_m(mode.energy_kev*1000)))


def chromatic_records(checkpoint, *, history=False):
    result, record = [], checkpoint.record
    while isinstance(record, Mapping):
        for mode in record.get("modes", ()):
            for action in mode.get("posed_spherical_actions", ()):
                if action.get("model") == "additional-chromatic-canonical-screen-v1":
                    result.append((mode["mode_id"], action))
        if not history:
            break
        record = record.get("upstream")
    return result


def assert_modes_equal(actual, expected):
    assert len(actual.beam.modes) == len(expected.beam.modes)
    for left, right in zip(actual.beam.modes, expected.beam.modes):
        assert left.mode_id == right.mode_id
        assert left.energy_kev == pytest.approx(right.energy_kev, abs=1e-11)
        assert left.weight_per_reference_electron == pytest.approx(right.weight_per_reference_electron, rel=3e-13)
        for attribute in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
            # A near-zero off-diagonal curvature can differ by roundoff from
            # the much larger diagonal terms during CPU/CUDA matrix products.
            # Scale that absolute allowance by the entire matrix, rather than
            # assigning an arbitrary physical tolerance to a zero entry. The
            # full complex-field comparison below remains the phase contract.
            absolute = (16*np.finfo(float).eps*max(1., np.linalg.norm(right.plane.curvature_m1, ord=2))
                        if attribute == "curvature_m1" else 1e-18)
            np.testing.assert_allclose(getattr(left.plane, attribute), getattr(right.plane, attribute),
                rtol=2e-10, atol=absolute)
        np.testing.assert_allclose(full_field(left), full_field(right), rtol=2e-9, atol=2e-12)
        xy = left.plane.coordinates_m()
        centroid = np.sum(xy*abs(left.plane.amplitude)[None]**2, axis=(1, 2))
        assert np.linalg.norm(centroid) > 1e-8  # Physical offsets were not recentered.


@pytest.mark.parametrize("key", LENSES)
def test_real_lens_cc_updates_plan_complex_wave_and_phase_plate_for_each_energy(key):
    state, lens, source = isolated_lens(key)
    outputs, signatures = {}, set()
    for cc in (0., 2., 20.):
        lens.cc_mm = cc
        prepared = _prepare_column(state, source.plane_z_mm, lens.z_mm, .02)
        signatures.add(prepared[0].signature)
        outputs[cc] = _propagate_column(state, source, lens.z_mm, maximum_step_mm=.02,
            grid_numerics=numerics(), _prepared=prepared)
        for before, after in zip(source.beam.modes, outputs[cc].beam.modes):
            assert after.weight_per_reference_electron == pytest.approx(before.weight_per_reference_electron, rel=3e-13)
        assert len(chromatic_records(outputs[cc])) == (0 if cc == 0 else 3)
    assert len(signatures) == 3
    for index in (0, 2):
        baseline = outputs[0.].beam.modes[index]
        lower, upper = (outputs[cc].beam.modes[index] for cc in (2., 20.))
        # Observe directly after the screen: identical physical lattices make
        # phase differences genuine, rather than a changed plotting grid.
        np.testing.assert_allclose(lower.plane.basis_m, baseline.plane.basis_m, rtol=2e-13)
        difference2 = np.linalg.norm(full_field(lower)-full_field(baseline))
        difference20 = np.linalg.norm(full_field(upper)-full_field(baseline))
        assert difference2 > 1e-8
        assert difference20 > 5*difference2
        valid = abs(baseline.plane.amplitude) > np.max(abs(baseline.plane.amplitude))*.01
        phase_delta = np.angle(np.exp(1j*(mode_phase_samples(upper)-mode_phase_samples(baseline))))
        assert np.max(abs(phase_delta[valid])) > 1e-7
    # A monoenergetic nominal mode must not acquire fictitious Cc from
    # differing mass constants or an inverse-momentum roundoff residue.
    np.testing.assert_allclose(full_field(outputs[20.].beam.modes[1]),
        full_field(outputs[0.].beam.modes[1]), rtol=2e-11, atol=2e-13)
    nominal = [row for mode_id, row in chromatic_records(outputs[20.]) if mode_id.endswith(":300.0")]
    assert nominal[0]["relative_energy_offset"] == 0.
    state.chromatic_aberration_enabled = False
    prepared = _prepare_column(state, source.plane_z_mm, lens.z_mm, .02)
    assert not prepared[0].intrinsic_aberration_kicks
    disabled = _propagate_column(state, source, lens.z_mm, maximum_step_mm=.02,
        grid_numerics=numerics(), _prepared=prepared)
    assert not chromatic_records(disabled)
    assert_modes_equal(disabled, outputs[0.])


@pytest.mark.parametrize("key", LENSES)
def test_cc_at_checkpoint_and_disk_segments_is_executed_once(key, tmp_path):
    state, lens, source = isolated_lens(key)
    lens.cc_mm = 20.
    stop = float(lens.z_mm)+.1
    prepared = _prepare_column(state, source.plane_z_mm, stop, .02)
    expected = _propagate_column(state, source, stop, maximum_step_mm=.02,
        grid_numerics=numerics(), _prepared=prepared)
    node, = np.flatnonzero(prepared[0].z_mm == lens.z_mm)
    left = _slice_prepared(prepared, 0, int(node))
    first = _propagate_column(state, source, lens.z_mm, maximum_step_mm=.02,
        grid_numerics=numerics(), _prepared=left)
    # A captured whole-plan slice and a freshly requested observation at the
    # same physical lens plane must have identical event ownership, including
    # mm -> m -> mm roundoff in the stored lens registration.
    independently_prepared_first = _propagate_column(state, source, lens.z_mm,
        maximum_step_mm=.02, grid_numerics=numerics())
    assert len(chromatic_records(first)) == 3
    assert_modes_equal(first, independently_prepared_first)
    right = _slice_prepared(prepared, int(node), len(prepared[0].step_m))
    resumed = _propagate_column(state, first, stop, maximum_step_mm=.02,
        grid_numerics=numerics(), _prepared=right)
    assert not chromatic_records(resumed)
    assert len(chromatic_records(resumed, history=True)) == 3
    assert_modes_equal(resumed, expected)
    independently_prepared_resume = _propagate_column(state, independently_prepared_first, stop,
        maximum_step_mm=.02, grid_numerics=numerics())
    assert not chromatic_records(independently_prepared_resume)
    assert_modes_equal(independently_prepared_resume, expected)
    store = ExecutedWaveStore(tmp_path, f"actual-lens-{key}-cc", 64*1024**2)
    segmented, hit = _propagate_column_segmented(state, source, stop, store=store,
        segment_steps=int(node), maximum_step_mm=.02, grid_numerics=numerics(), _prepared=prepared)
    assert not hit
    assert len(chromatic_records(segmented, history=True)) == 3
    assert_modes_equal(segmented, expected)
    cached, hit = _propagate_column_segmented(state, source, stop, store=store,
        segment_steps=int(node), maximum_step_mm=.02, grid_numerics=numerics(), _prepared=prepared)
    assert hit
    assert len(chromatic_records(cached, history=True)) == 3
    assert_modes_equal(cached, expected)


@pytest.mark.parametrize("key", LENSES)
def test_real_lens_cc_mixed_modes_cuda_matches_cpu_complex_phase_and_weight(key):
    from test_wave_device import _cuda
    _cuda()
    state, lens, source = isolated_lens(key)
    lens.cc_mm = 20.
    stop = float(lens.z_mm)+.1
    prepared = _prepare_column(state, source.plane_z_mm, stop, .02)
    expected = _propagate_column(state, source, stop, maximum_step_mm=.02,
        grid_numerics=numerics(), _prepared=prepared)
    actual = _propagate_column(state, source, stop, maximum_step_mm=.02,
        grid_numerics=numerics("Require GPU"), _prepared=prepared)
    assert all(row["compute_backend"] == "cupy" for row in actual.record["modes"])
    assert len(chromatic_records(actual)) == 3
    assert_modes_equal(actual, expected)
