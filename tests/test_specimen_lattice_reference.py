"""Entrance-fixed CIF lattice phase; static atom geometry, never propagation."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from temsim.immutable_json import thaw_json
from temsim.instrument_snapshot import decode_instrument, encode_instrument
from temsim.optics.model import Sample, State
from temsim.optics.column import default_state
from temsim.specimen.atomistic import build_cif_equilibrium_atoms
from temsim.specimen.cif_io import read_cif_atoms
from temsim.specimen.geometry import quaternion_from_zone_axes, quaternion_to_matrix
from temsim.specimen.lattice_reference import lattice_phase_for_sample, lattice_translation_z_angstrom
from temsim.specimen.scene import SpecimenScene


CIF = Path(__file__).resolve().parents[1]/"configs/specimens/Si.cif"
SITE_TOLERANCE_ANGSTROM = 5e-8


@pytest.fixture
def sample():
    unit = read_cif_atoms(CIF)
    q = quaternion_from_zone_axes(unit.cell.array, (1, 1, 0), (-1, 1, 0))
    return Sample(z_mm=1599.2, thickness_nm=.4, inserted=True, specimen_mode="atomic",
        cif_path=str(CIF), size_x_nm=10., size_y_nm=10., envelope_shape="disk",
        specimen_orientation_quaternion_wxyz=q)


def atoms(sample, *, shift=None, fov=40.):
    return build_cif_equilibrium_atoms(sample.cif_path,
        thickness_angstrom=sample.thickness_nm*10, field_of_view_angstrom=fov,
        rotation_matrix=quaternion_to_matrix(sample.specimen_orientation_quaternion_wxyz),
        specimen_size_xy_angstrom=(100., 100.), specimen_envelope_shape="disk",
        lattice_translation_z_angstrom=lattice_translation_z_angstrom(sample) if shift is None else shift)[0]


def matching_sites(left, right):
    assert len(left) == len(right)
    error = max(cKDTree(left).query(right)[0].max(initial=0.),
                cKDTree(right).query(left)[0].max(initial=0.))
    assert error <= SITE_TOLERANCE_ANGSTROM
    return error


def test_repeated_thickness_edits_preserve_original_si110_entrance_atoms(sample):
    original_hash = sha256(CIF.read_bytes()).hexdigest()
    old = atoms(sample).positions.copy()
    upper = sample.upper_surface_reference_z_mm
    for thickness in (10., 2., .8, .2, 5., .4):
        sample.thickness_nm = thickness
        assert sample.lattice_reference_thickness_nm == .4
        assert abs((sample.z_mm-thickness*.5e-6)-upper) <= 4*abs(np.spacing(upper))
        depth = min(.4, thickness)*10
        prefix = atoms(sample).positions
        prefix = prefix[prefix[:, 2] <= depth+SITE_TOLERANCE_ANGSTROM]
        expected = old[old[:, 2] <= depth+SITE_TOLERANCE_ANGSTROM]
        matching_sites(expected, prefix)
    np.testing.assert_array_equal(atoms(sample).positions, old)
    assert sha256(CIF.read_bytes()).hexdigest() == original_hash


def test_save_restore_preserves_lattice_reference_and_future_thickness_behavior(sample, tmp_path):
    old = atoms(sample).positions.copy()
    sample.thickness_nm = 10.
    state = default_state()
    state.sample = sample
    graph = encode_instrument(state)
    path = tmp_path/"working-inputs.json"
    path.write_text(json.dumps(thaw_json(graph)), encoding="utf-8")
    restored = decode_instrument(json.loads(path.read_text())).sample
    assert restored.lattice_reference_thickness_nm == .4
    assert restored.upper_surface_reference_z_mm == sample.upper_surface_reference_z_mm
    assert restored.z_mm == sample.z_mm and restored.thickness_nm == 10.
    restored.thickness_nm = 1.2
    prefix = atoms(restored).positions
    matching_sites(old, prefix[prefix[:, 2] <= 4.+SITE_TOLERANCE_ANGSTROM])


def test_reference_equal_to_current_thickness_has_bitwise_default_center_sites(sample):
    sample.thickness_nm = 10.
    # Adopting an already-prepared 10 nm CIF must not apply another correction.
    sample.lattice_reference_thickness_nm = 10.
    explicit = atoms(sample).positions
    default = build_cif_equilibrium_atoms(sample.cif_path, thickness_angstrom=100.,
        field_of_view_angstrom=40., rotation_matrix=quaternion_to_matrix(sample.specimen_orientation_quaternion_wxyz),
        specimen_size_xy_angstrom=(100., 100.), specimen_envelope_shape="disk")[0].positions
    np.testing.assert_array_equal(explicit, default)
    assert explicit.tobytes() == default.tobytes()
    legacy = SimpleNamespace(thickness_nm=10., lattice_reference_thickness_nm=None)
    before = vars(legacy).copy()
    assert lattice_translation_z_angstrom(legacy) == 0.
    assert vars(legacy) == before


def test_scene_and_finite_material_reference_share_the_same_phase(sample, monkeypatch):
    from temsim.physics.mixed_stem_material import finite_cif_potential
    import temsim.specimen.atomistic as module
    sample.thickness_nm = 10.
    scene = SpecimenScene.from_state(SimpleNamespace(sample=sample))
    assert scene.lattice_reference_thickness_nm == .4
    assert lattice_translation_z_angstrom(scene) == -48.
    found = []
    class StopBeforePotential(Exception):
        pass
    def capture(*args, **kwargs):
        found.append(kwargs["lattice_translation_z_angstrom"])
        raise StopBeforePotential
    monkeypatch.setattr(module, "build_cif_equilibrium_atoms", capture)
    with pytest.raises(StopBeforePotential):
        finite_cif_potential(scene, (40., 40.), (64, 64), 8,
            maximum_working_bytes=1024**3,
            crystal=(np.array([5., 5., 6.1]), np.zeros((1, 3)), np.array([14]), {}),
            energy_kev=300., xp=np)
    assert found == [-48.]


def test_projected_site_enumeration_translates_replication_bounds_and_cache_key(monkeypatch):
    from temsim.specimen import projected_scattering as module
    monkeypatch.setattr(module, "_unit_cell", lambda _: (np.diag((10., 10., 1.)),
        np.array([[0., 0., .25]]), np.array([14]), np.array([1.])))
    geometry = (1., 1., .4, 0., 0., "rectangle", (1., 0., 0., 0.))
    roi = (-.5, -.5, .5, .5)
    assert module._projected_sites(b"lattice-phase-array-fixture", geometry+(0.,), roi, (0., 0.))[-1] == 0
    # Ten full cells away: translating only after enumeration would miss it.
    assert module._projected_sites(b"lattice-phase-array-fixture", geometry+(-10.1,), roi, (0., 0.))[-1] == 1
    assert module._projected_sites(b"lattice-phase-array-fixture", geometry+(0.,), roi, (0., 0.))[-1] == 0


def test_preview_reduced_window_keeps_actual_lattice_shift(sample, monkeypatch):
    from temsim.specimen.geometry import build_sample_geometry_snapshot
    import temsim.specimen.atomistic as module
    sample.thickness_nm = 10.
    original = module.build_cif_equilibrium_atoms
    shifts = []
    def capture(*args, **kwargs):
        shifts.append((kwargs["thickness_angstrom"], kwargs["lattice_translation_z_angstrom"]))
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "build_cif_equilibrium_atoms", capture)
    preview = build_sample_geometry_snapshot(sample, maximum_display_atoms=100)
    assert preview.atom_display_capped
    assert preview.lattice_translation_z_nm == -4.8
    assert shifts[-1][0] < 100.
    assert shifts[-1][1] == -48.


def test_prepared_potential_cache_identity_tracks_lattice_reference(sample):
    from temsim.physics.wave_imaging import _prepared_specimen_identity
    from temsim.specimen.presets import load_specimen_preset
    sample.thickness_nm = 10.
    state = default_state()
    state.sample = sample
    preset = load_specimen_preset("si_110")
    arguments = dict(field_of_view_angstrom_override=None, calculation_roi_centre_nm=(0., 0.),
                     calculation_roi_bounds_nm=None)
    first = _prepared_specimen_identity(state, preset, **arguments)
    state.sample.lattice_reference_thickness_nm = 10.
    second = _prepared_specimen_identity(state, preset, **arguments)
    assert first != second


def test_already_translated_10nm_cif_is_not_compensated_twice(sample, tmp_path):
    unit = read_cif_atoms(CIF)
    rotation = quaternion_to_matrix(sample.specimen_orientation_quaternion_wxyz)
    unit.translate(rotation.T @ np.array((0., 0., -48.)))
    unit.wrap(eps=1e-12)
    lengths = unit.cell.lengths()
    lines = ["data_shifted_fixture", "_symmetry_space_group_name_H-M 'P 1'",
        *[f"_cell_length_{name} {value:.17g}" for name, value in zip("abc", lengths)],
        "_cell_angle_alpha 90", "_cell_angle_beta 90", "_cell_angle_gamma 90",
        "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
        "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_occupancy"]
    for i, row in enumerate(unit.get_scaled_positions(wrap=True)):
        lines.append(f"Si{i} Si "+" ".join(f"{v:.17g}" for v in row)+" 1")
    path = tmp_path/"shifted-input.cif"
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    original = atoms(sample).positions
    thick = Sample(z_mm=sample.upper_surface_z_mm+5e-6, thickness_nm=10.,
        specimen_mode="atomic", inserted=True, cif_path=str(path),
        specimen_orientation_quaternion_wxyz=sample.specimen_orientation_quaternion_wxyz)
    assert thick.lattice_reference_thickness_nm == 10.
    adopted = atoms(thick).positions
    zero_offset = atoms(thick, shift=0.).positions
    assert adopted.tobytes() == zero_offset.tobytes()
    object.__setattr__(thick, "lattice_reference_thickness_nm", None)
    legacy = atoms(thick).positions
    assert legacy.tobytes() == adopted.tobytes()
    matching_sites(original, adopted[adopted[:, 2] <= 4.+SITE_TOLERANCE_ANGSTROM])


def test_periodic_atom_layers_share_reference_and_equal_none_is_bitwise(sample, monkeypatch):
    import abtem
    from temsim.physics.mixed_stem_scan import (
        _legacy_abtem_periodic_potential as _periodic_potential,
        _rectangular_crystal,
    )
    captures = []
    class Potential:
        def __init__(self, atoms, *, gpts, slice_thickness, **kwargs):
            captures.append(atoms.positions.copy())
            self.gpts, self.dz = gpts, slice_thickness
            self.integrator = self
        def get_integral_table(self, *args):
            return SimpleNamespace(radial_gpts=np.array([0., .1]))
        @property
        def sampling(self):
            return (1., 1.)
        def build(self, *, first_slice, last_slice, **kwargs):
            count = last_slice-first_slice
            return SimpleNamespace(array=np.zeros((count, *self.gpts), dtype=np.float32),
                                   slice_thickness=np.full(count, self.dz))
    monkeypatch.setattr(abtem, "Potential", Potential)
    thin_scene = SpecimenScene.from_state(SimpleNamespace(sample=sample))
    crystal = _rectangular_crystal(thin_scene)
    tiles = (2, 2)
    fov = tuple(crystal[0][:2]*tiles)
    _, _, old_record = _periodic_potential(crystal, thin_scene, fov, tiles, 120, 8)
    thin_sites = captures[-1].copy()
    thin_sites[:, 2] -= old_record["finite_axial_padding_angstrom"]
    sample.thickness_nm = 10.
    thick_scene = SpecimenScene.from_state(SimpleNamespace(sample=sample))
    _, _, new_record = _periodic_potential(crystal, thick_scene, fov, tiles, 120, 8)
    thick_sites = captures[-1].copy()
    thick_sites[:, 2] -= new_record["finite_axial_padding_angstrom"]
    matching_sites(thin_sites, thick_sites[thick_sites[:, 2] < 4.])
    legacy_scene = replace(thick_scene, lattice_reference_thickness_nm=None)
    _periodic_potential(crystal, legacy_scene, fov, tiles, 120, 8)
    legacy_sites = captures[-1].copy()
    adopted_scene = replace(thick_scene, lattice_reference_thickness_nm=10.)
    _periodic_potential(crystal, adopted_scene, fov, tiles, 120, 8)
    assert legacy_sites.tobytes() == captures[-1].tobytes()
