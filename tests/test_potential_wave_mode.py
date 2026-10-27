"""Mode/phase-reference persistence for the bounded potential wave benchmark."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, e, m_e

from temsim.immutable_json import thaw_json
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
from temsim.physics.potential_wave import propagate_plane_wave_potential, propagate_wave_mode_potential
from temsim.physics.tip_gun_wave import TIP_REFERENCE, TipGunCheckpoint
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_reference import AxialWaveReference


def _fixture(reference=True):
    xy = (np.arange(41)-20)*1e-9
    xx, yy = np.meshgrid(xy, xy)
    amplitude = np.exp(-(xx*xx+yy*yy)/(2*(3e-9)**2)+.31j)
    amplitude /= np.sqrt(np.sum(abs(amplitude)**2))
    plane = PlaneWave(amplitude, np.eye(2)*1e-9, np.zeros(2), np.eye(2)*100., np.array((1e-5, -2e-5)))
    # Synthetic tip-referenced mode for persistence only, not an emitted gun.
    mode = WaveMode(plane, .27, TIP_REFERENCE, "synthetic:mode:0", 300.,
                    AxialWaveReference(2e-12, 3e-25) if reference else None)
    snapshot = ElectromagneticPotentialSnapshot(capture_physical_scene(SimpleNamespace()), ())
    return snapshot, mode, xy


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("reference", [False, True])
def test_mode_factorization_preserves_absolute_complex_field_and_reference(blocked, reference):
    snapshot, mode, x = _fixture(reference)
    mask = np.zeros(mode.plane.amplitude.shape, bool) if blocked else np.broadcast_to(x > 0, mode.plane.amplitude.shape)
    raw, raw_record = propagate_plane_wave_potential(snapshot, mode.plane, (0., 1e-6), energy_ev=300000., aperture_masks={1: mask})
    output, record = propagate_wave_mode_potential(snapshot, mode, (0., 1e-6), aperture_masks={1: mask})
    np.testing.assert_allclose(output.plane.amplitude*np.sqrt(output.weight_per_reference_electron),
                               raw.amplitude*np.sqrt(mode.weight_per_reference_electron), rtol=3e-15, atol=1e-20)
    assert output.weight_per_reference_electron == pytest.approx(.27*raw.probability, abs=1e-15)
    assert output.mode_id == mode.mode_id and output.reference_plane == mode.reference_plane
    assert output.energy_kev == mode.energy_kev
    assert record["longitudinal_carrier_in_amplitude"] is False
    assert record["production_admission"] is False
    if reference:
        kinetic = 300000*e
        momentum = np.sqrt(kinetic*(kinetic+2*m_e*c*c))/c
        speed = momentum*c*c/(kinetic+m_e*c*c)
        assert output.axial_reference.longitudinal_action_j_s == pytest.approx(3e-25+momentum*1e-6, rel=1e-15)
        assert output.axial_reference.flight_time_s == pytest.approx(2e-12+1e-6/speed, rel=1e-15)
    else:
        assert output.axial_reference is None
    assert mode.plane.curvature_m1 is not None and mode.plane.tilt_rad is not None
    assert raw_record["normalization_applied"] is False


def test_executed_checkpoint_resume_keeps_weight_complex_phase_and_axial_action(tmp_path):
    snapshot, mode, _ = _fixture()
    first, record = propagate_wave_mode_potential(snapshot, mode, (0., .5e-6))
    checkpoint = TipGunCheckpoint(BeamState((first,), first.reference_plane), .5e-3, 1e-12, thaw_json(record))
    store = ExecutedWaveStore(tmp_path, snapshot.numerical_digest, 20*1024**2)
    key = store.key("potential-benchmark", "middle")
    store.put(key, checkpoint)
    restored = store.get(key)
    loaded = restored.beam.modes[0]
    np.testing.assert_array_equal(loaded.plane.amplitude, first.plane.amplitude)
    assert loaded.axial_reference == first.axial_reference
    assert loaded.weight_per_reference_electron == first.weight_per_reference_electron
    final, _ = propagate_wave_mode_potential(snapshot, loaded, (.5e-6, 1e-6))
    direct, _ = propagate_wave_mode_potential(snapshot, mode, (0., .5e-6, 1e-6))
    np.testing.assert_allclose(final.plane.amplitude*np.sqrt(final.weight_per_reference_electron),
                               direct.plane.amplitude*np.sqrt(direct.weight_per_reference_electron), rtol=3e-11, atol=3e-13)
    np.testing.assert_allclose(
        (final.axial_reference.flight_time_s, final.axial_reference.longitudinal_action_j_s),
        (direct.axial_reference.flight_time_s, direct.axial_reference.longitudinal_action_j_s),
        rtol=3*np.finfo(float).eps, atol=0.)
    changed = ElectromagneticPotentialSnapshot(capture_physical_scene(SimpleNamespace(beam_voltage_kv=301.)), ())
    other = ExecutedWaveStore(tmp_path, changed.numerical_digest, 20*1024**2)
    assert other.key("potential-benchmark", "middle") != key
    with pytest.raises(ValueError, match="identity/dependency"):
        other.get(key)


def test_mode_bridge_requires_declared_mode_not_arbitrary_object():
    snapshot, mode, _ = _fixture()
    with pytest.raises(ValueError, match="WaveMode"):
        propagate_wave_mode_potential(snapshot, mode.plane, (0., 1e-6))
