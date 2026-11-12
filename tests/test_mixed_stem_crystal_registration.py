"""Original CIF registration certificates; no beam or propagation is evaluated."""
from types import SimpleNamespace

from ase import Atoms
import numpy as np

from temsim.physics.mixed_stem_scan import _rectangular_crystal


def test_crystal_certificate_recovers_common_rounding_from_original_sites(monkeypatch):
    from temsim.specimen import cif_io
    original = np.array([[.5, 1., 2*.123456789123]])
    atoms = Atoms("Si", positions=original, cell=np.eye(3)*2, pbc=True)
    monkeypatch.setattr(cif_io, "read_cif_atoms", lambda _: atoms)
    scene = SimpleNamespace(is_vacuum=False, source_kind="cif", cif_path="math-only.cif",
                            orientation_quaternion_wxyz=(1., 0., 0., 0.))
    repeat, stored, numbers, proof = _rectangular_crystal(scene)
    assert numbers.tolist() == [14]
    # Keep the historical rounded basis itself unchanged. Recovery is a
    # separately recorded correction derived from the actual unrounded CIF.
    np.testing.assert_array_equal(stored, [[.5, 1., 2*.123456789]])
    assert abs(stored[0, 2]-original[0, 2]) > 1e-11
    common = np.asarray(proof["basis_common_offset_angstrom_xyz"])
    error_bound = np.asarray(proof["basis_rounding_error_bound_angstrom_xyz"])
    corrected = np.remainder(stored+common, repeat)
    difference = (original-corrected+repeat/2) % repeat-repeat/2
    assert np.all(abs(difference) <= error_bound)
    assert error_bound.max() < 1e-12
    assert proof["basis_registration_raw_site_count"] == 729
    assert np.all(error_bound >= proof["basis_registration_floating_floor_angstrom"])
