"""Local material ROI keeps physical pitch and incoming reciprocal support."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_checkpoint import WaveCheckpoint, EMITTED_ELECTRON_REFERENCE
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_grid import WaveGridNumerics


@dataclass
class Prepared:
    x_angstrom: object
    y_angstrom: object
    potential_configurations_v_angstrom: tuple
    slice_thicknesses_angstrom: object
    metrics: dict


@pytest.mark.parametrize("basis", [np.eye(2)*4e-9/512,
    np.array([[1., -.6], [.6, 1.]])*4e-9/512,
    np.array([[1., .5], [0., .7]])*4e-9/512,
    np.eye(2)*4e-9/1024])
def test_local_roi_does_not_reuse_large_fov_pixel_count_or_coarsen_incoming_lattice(monkeypatch, basis):
    from temsim.physics import specimen_wave_transport as specimen
    from temsim.physics import wave_imaging
    from temsim.specimen import presets
    from temsim.specimen.scene import SpecimenScene
    sample = SimpleNamespace(z_mm=1600., thickness_nm=.4, wave_multislice_enabled=True,
        real_high_angle_tail_enabled=False, wave_grid_pixels=2048,
        wave_field_of_view_angstrom=640., wave_frozen_phonon_enabled=False,
        wave_slice_thickness_angstrom=2.)
    state = SimpleNamespace(sample=sample)
    scene = SimpleNamespace(structure_available=True, wave_template_key="controlled")
    monkeypatch.setattr(SpecimenScene, "from_state", lambda state: scene)
    monkeypatch.setattr(presets, "load_specimen_preset", lambda key: SimpleNamespace(pixels=2048, field_of_view_angstrom=640.))
    executed = []
    def prepare(numerical, preset, **kwargs):
        size = numerical.sample.wave_grid_pixels
        fov = kwargs["field_of_view_angstrom_override"]
        axis = (np.arange(size)-size//2)*fov/size
        executed.append((size, fov))
        return Prepared(axis, axis, (np.zeros((size, size)),), np.array([4.]), {})
    monkeypatch.setattr(wave_imaging, "prepare_specimen_potentials", prepare)
    amplitude = np.ones((512, 512), complex)/512
    mode = WaveMode(PlaneWave(amplitude, basis, np.array([2e-9, -1e-9])),
        .5, EMITTED_ELECTRON_REFERENCE, "controlled", 300.)
    checkpoint = WaveCheckpoint(BeamState((mode,), EMITTED_ELECTRON_REFERENCE),
        sample.z_mm-sample.thickness_nm*.5e-6, 1e-5, {})
    _, prepared, x, y, dz = specimen._prepare_material_grid(state, checkpoint,
        grid_numerics=WaveGridNumerics(maximum_pixels=8192))
    record = prepared.metrics["coherent_material_grid"]
    incoming_density = np.max(np.sum(abs(np.linalg.inv(basis).T), axis=1))
    coarse_pixels = record["executed_wave_pixels"]
    coarse_pitch = record["field_of_view_m"]/coarse_pixels
    assert 1./coarse_pitch >= incoming_density*(1.-1e-14)
    assert coarse_pitch <= 640e-10/2048
    assert len(x) == len(y) == executed[0][0]
    assert sample.wave_grid_pixels == 2048 and sample.wave_field_of_view_angstrom == 640.
    if np.array_equal(basis, np.eye(2)*4e-9/512):
        assert record["executed_wave_pixels"] == 528
        assert record["executed_potential_pixels"] == 1056
