"""Scalar phase/absorption slice arithmetic; no atom or coherent-field solve."""
from dataclasses import replace
from types import SimpleNamespace
import math

import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_checkpoint import EMITTED_ELECTRON_REFERENCE, WaveCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_grid import WaveGridNumerics


@pytest.fixture
def scalar_material(monkeypatch):
    from temsim.physics import specimen_wave_transport as specimen
    from temsim.physics import wave_imaging
    from temsim.specimen import inelastic
    from temsim.specimen.scene import SpecimenScene

    original_dz = np.array([2., 3., 5.])  # Angstrom, intentionally unequal.
    orientation = (math.sqrt(.5), 0., math.sqrt(.5), 0.)
    crystal_origin = (1.25, -2.5, 3.75)
    sample = SimpleNamespace(z_mm=1605., thickness_nm=1., wave_bandwidth_fraction=1.,
        specimen_orientation_quaternion_wxyz=orientation, crystal_origin_angstrom=crystal_origin)
    state = SimpleNamespace(sample=sample)
    upper = sample.z_mm - sample.thickness_nm * .5e-6
    bottom = sample.z_mm + sample.thickness_nm * .5e-6
    plane = PlaneWave(np.ones((2, 2), complex) / 2, np.eye(2) * 1e-10, np.zeros(2))
    mode = WaveMode(plane, .4, EMITTED_ELECTRON_REFERENCE, "scalar-array", 300.)
    entrance = WaveCheckpoint(BeamState((mode,), EMITTED_ELECTRON_REFERENCE), upper, 1e-9,
        {"fixture": "Scalar array mathematics only"})
    scene = SimpleNamespace(is_vacuum=False,
        sample_contains_xy=lambda x, y: np.ones_like(x, dtype=bool))
    axis = np.array([-.5, .5]) * 1e-10
    phase_calls, thickness_calls, preparation = [], [], []
    prepared = SimpleNamespace(metrics={"crystal_origin_angstrom": crystal_origin,
        "orientation_quaternion_wxyz": orientation})

    monkeypatch.setattr(SpecimenScene, "from_state", staticmethod(lambda *_: scene))
    monkeypatch.setattr(specimen, "_material_wave_axes", lambda x, y, _numerics: (x, y))
    monkeypatch.setattr(specimen, "_regrid_refined_mode", lambda mode, *_: (mode, {"mocked": True}))
    monkeypatch.setattr(specimen, "_bandlimit_mode", lambda mode, *_a, **_k: (mode, 0.))
    monkeypatch.setattr(wave_imaging, "interaction_constant_rad_per_v_angstrom", lambda _: .17)
    distribution = SimpleNamespace(total_inelastic_mean_free_path_nm=np.inf,
        absorption_mean_free_path_nm=5., material_key="scalar", model="analytic mock",
        beam_energy_kev=300., warnings=(), channels=())
    monkeypatch.setattr(inelastic, "real_inelastic_distribution", lambda _: distribution)

    def phase(mode, projected, x, y, sigma, fraction, **kwargs):
        assert np.all(projected == projected.flat[0])
        value = float(projected.flat[0]) * sigma * fraction
        phase_calls.append(value)
        return replace(mode, plane=replace(mode.plane,
            amplitude=mode.plane.amplitude * np.exp(1j * value))), {"scalar_phase_rad": value}

    def absorb(mode, inside, thickness_nm, actual_distribution):
        assert inside.all() and actual_distribution is distribution
        thickness_calls.append(thickness_nm)
        retained = mode.weight_per_reference_electron * math.exp(-thickness_nm / 5.)
        removed = mode.weight_per_reference_electron - retained
        return replace(mode, weight_per_reference_electron=retained), {
            "removed_weight": removed, "events": [{"kind": "effective_absorption", "weight": removed}]}

    monkeypatch.setattr(specimen, "_material_phase", phase)
    monkeypatch.setattr(specimen, "_attenuate_zero_loss", absorb)

    class Transport:
        digest = "mocked-scalar-transport"

        @staticmethod
        def context(record, factor=1.):
            return {}

        def __call__(self, original_state, checkpoint, next_z, **kwargs):
            assert original_state is state
            return replace(checkpoint, plane_z_mm=next_z,
                record={**checkpoint.record, "modes": [{"scalar_transport": True}]})

    def calculate(target, layout):
        # A uniform one-volt material has projected potential equal to its
        # Angstrom thickness. Both supported potential layouts must agree.
        potential = (original_dz[:, None, None] * np.ones((3, 2, 2)) if layout == "slices"
                     else np.ones((2, 2)) * original_dz.sum())
        prepared.potential_configurations_v_angstrom = (potential,)
        material = (scene, prepared, axis, axis, original_dz)

        def prepare(original_state, checkpoint, execute, **kwargs):
            preparation.append((original_state, checkpoint, original_state.sample.z_mm,
                original_state.sample.thickness_nm, original_state.sample.crystal_origin_angstrom,
                original_state.sample.specimen_orientation_quaternion_wxyz))
            return execute(material, 1, [])

        monkeypatch.setattr(specimen, "_with_material_refinement", prepare)
        return specimen._propagate_specimen(state, entrance, observation_z_mm=target,
            grid_numerics=WaveGridNumerics(compute_backend="CPU"), column_propagator=Transport())

    return SimpleNamespace(calculate=calculate, state=state, entrance=entrance, original_dz=original_dz,
        upper=upper, bottom=bottom, phase_calls=phase_calls, thickness_calls=thickness_calls,
        preparation=preparation, orientation=orientation, crystal_origin=crystal_origin)


@pytest.mark.parametrize("layout", ["slices", "projected"])
@pytest.mark.parametrize("depth_angstrom", [1., 7., 10.])
def test_partial_material_phase_absorption_and_unchanged_crystal_coordinates(
        scalar_material, layout, depth_angstrom):
    fixture = scalar_material
    target = (fixture.bottom if depth_angstrom == 10.
              else fixture.upper + depth_angstrom * 1e-7)
    sample_before = vars(fixture.state.sample).copy()
    output = fixture.calculate(target, layout)
    row, = output.record["modes"]
    slices = row["slices"]
    executed_dz = np.array([step["executed_slice_thickness_angstrom"] for step in slices])
    full_count = {1.: 0, 7.: 2, 10.: 3}[depth_angstrom]
    # Exact original values detect cancellation from subtracting large lab Z.
    np.testing.assert_array_equal(executed_dz[:full_count], fixture.original_dz[:full_count])
    assert len(slices) == {1.: 1, 7.: 3, 10.: 3}[depth_angstrom]
    if depth_angstrom != 10.:
        assert 0 < slices[-1]["original_slice_fraction"] < 1
    np.testing.assert_allclose(fixture.thickness_calls, executed_dz * .1, rtol=0, atol=0)
    expected_phase = .17 * math.fsum(executed_dz)
    assert math.fsum(fixture.phase_calls) == pytest.approx(expected_phase, abs=2e-15)
    np.testing.assert_allclose(output.beam.modes[0].plane.amplitude,
        fixture.entrance.beam.modes[0].plane.amplitude * np.exp(1j * expected_phase), atol=3e-16)
    expected_weight = .4 * math.exp(-math.fsum(executed_dz) * .1 / 5.)
    assert output.beam.total_weight == pytest.approx(expected_weight, abs=1e-16)
    # Only a partial final slice needs subtraction of represented lab planes.
    # Its depth error is bounded by their float64 coordinate resolution.
    assert math.fsum(executed_dz) == pytest.approx(depth_angstrom,
        abs=4 * abs(np.spacing(fixture.upper)) * 1e7)
    assert output.plane_z_mm == target
    assert output.record["physical_specimen_upper_z_mm"] == fixture.upper
    assert output.record["physical_specimen_lower_z_mm"] == fixture.bottom
    assert output.record["potential"]["crystal_origin_angstrom"] == fixture.crystal_origin
    assert vars(fixture.state.sample) == sample_before
    assert fixture.preparation[0][2:] == (1605., 1., fixture.crystal_origin, fixture.orientation)
    assert row["probability_balance"]["accounted_weight"] == pytest.approx(.4, abs=1e-16)


def test_upper_face_returns_input_without_preparing_or_attenuating_material(scalar_material):
    fixture = scalar_material
    assert fixture.calculate(fixture.upper, "slices") is fixture.entrance
    assert not fixture.preparation and not fixture.phase_calls and not fixture.thickness_calls
