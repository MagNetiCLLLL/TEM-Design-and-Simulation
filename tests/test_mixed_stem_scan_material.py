"""Explicit legacy material baseline with a fake builder, no physical wave."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.mixed_stem_material import (
    _legacy_abtem_finite_cif_potential as finite_cif_potential,
)


@pytest.fixture
def fake_material(monkeypatch):
    from temsim.specimen import atomistic, geometry
    from temsim.physics import host_memory_budget
    calls = []

    class Atoms:
        positions = np.array([[20., 16., 0.], [21., 17., 99.]]).copy()
        pbc = None

        def set_cell(self, value, *, scale_atoms):
            self.cell = value
            assert scale_atoms is False

        def get_chemical_symbols(self):
            return ["Si", "Si"]

        def __len__(self):
            return 2

    atoms = Atoms()

    def sites(path, **kwargs):
        calls.append(("sites", path, kwargs))
        return atoms, (5., 5., 5.)

    class Potential:
        def __init__(self, actual_atoms, **kwargs):
            assert actual_atoms is atoms
            calls.append(("potential", kwargs))
            self.shape = kwargs["gpts"]
            self.dz = kwargs["slice_thickness"]
            self.sampling = tuple(a/b for a, b in zip(atoms.cell[:2], self.shape))
            self.integrator = SimpleNamespace(get_integral_table=lambda *_:
                SimpleNamespace(radial_gpts=np.array([0., 1.])))

        def build(self, *, first_slice, last_slice, lazy):
            assert lazy is False
            calls.append(("build", first_slice, last_slice))
            x, y = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing="ij")
            values = np.arange(first_slice, last_slice)[:, None, None]*10000+x[None]*100+y[None]
            return SimpleNamespace(array=values, slice_thickness=np.full(last_slice-first_slice, self.dz))

    monkeypatch.setitem(sys.modules, "abtem", SimpleNamespace(Potential=Potential,
        config=SimpleNamespace(set=lambda _: None)))
    monkeypatch.setattr(atomistic, "build_cif_equilibrium_atoms", sites)
    monkeypatch.setattr(geometry, "quaternion_to_matrix", lambda _: np.eye(3))
    monkeypatch.setattr(host_memory_budget, "check_host_allocation", lambda *_a, **_k: None)
    scene = SimpleNamespace(source_kind="cif", cif_path="external-structure.cif", thickness_nm=10.,
        orientation_quaternion_wxyz=(1., 0., 0., 0.), size_xy_nm=(3., 4.),
        envelope_shape="disk", centre_xy_nm=(1.2, -.3))
    return scene, atoms, calls


def test_finite_builder_preserves_physical_site_selection_and_crops_only_vacuum_roi(fake_material):
    scene, atoms, calls = fake_material
    original_positions = atoms.positions.copy()
    array, slices, record = finite_cif_potential(scene, (40., 24.), (8, 10), 17,
        maximum_working_bytes=1_000_000)
    site_call = next(call for call in calls if call[0] == "sites")
    assert site_call[1] == scene.cif_path
    kwargs = site_call[2]
    assert kwargs["thickness_angstrom"] == 100.
    assert kwargs["specimen_size_xy_angstrom"] == (30., 40.)
    assert kwargs["specimen_envelope_shape"] == "disk"
    assert kwargs["specimen_centre_xy_angstrom"] == (12., -3.)
    np.testing.assert_array_equal(kwargs["rotation_matrix"], np.eye(3))
    assert atoms.pbc is False
    np.testing.assert_array_equal(atoms.positions[:, :2], original_positions[:, :2])
    np.testing.assert_allclose(atoms.positions[:, 2]-original_positions[:, 2], 4*100/17, atol=1e-14)
    assert next(call for call in calls if call[0] == "potential")[1]["periodic"] is False
    assert [call[2]-call[1] for call in calls if call[0] == "build"] == [8, 8, 1]
    z, y, x = np.meshgrid(np.arange(4, 21), np.arange(7, 15), np.arange(5, 15), indexing="ij")
    np.testing.assert_array_equal(array, z*10000+x*100+y)
    np.testing.assert_allclose(slices, np.full(17, 100/17), rtol=0, atol=0)
    assert record["physical_atomic_site_selection_before_potential"] is True
    assert record["potential_pixel_envelope_mask_applied"] is False
    assert record["grid_yx"] == [8, 10]


def test_finite_builder_refuses_budget_before_allocating_atoms(fake_material):
    scene, _, calls = fake_material
    with pytest.raises(MemoryError, match="host-memory budget"):
        finite_cif_potential(scene, (40., 24.), (8, 10), 17, maximum_working_bytes=100)
    assert calls == []
