"""Analytical coherent Gram diagnostics, not microscope qualification."""
import copy

import numpy as np
import pytest
from scipy.spatial import cKDTree

from scripts import experimental_gaussian_gram as gram


def test_independent_complex_gaussian_pair_and_polynomial_quadrature():
    evidence = gram.self_validation()
    assert evidence['pair_formula']['independent_uniform_quadrature_absolute_error'] < 1e-12
    assert evidence['cross_sum']['self_cross_vs_quadrature_error'] < 1e-12
    assert evidence['cross_sum']['self_split_norm_error'] < 1e-12
    assert evidence['spatial_moments']['independent_quadrature_max_absolute_error'] < 1e-12


@pytest.mark.parametrize('key', ['mode_id', 'mode_index', 'energy_kev', 'mode_weight',
                               'wavelength_m', 'output_plane_z_mm', 'executed_input_mode_identity',
                               'state_file_sha256'])
def test_convergence_rejects_changed_source_mode_plane_or_physical_state(key):
    original = {'mode_id':'source:E0:HG0,0', 'mode_index':0, 'energy_kev':300.,
        'mode_weight':.1, 'wavelength_m':2e-12, 'output_plane_z_mm':1599.2,
        'executed_input_mode_identity':'actual-array-identity', 'state_file_sha256':'captured-state'}
    changed = copy.deepcopy(original)
    changed[key] = 'different' if isinstance(changed[key], str) else changed[key]+1
    with pytest.raises(ValueError):
        gram.require_matched_modes(original, changed)


def test_dense_neighbor_query_rejects_before_allocating_unbounded_pair_lists():
    points = np.zeros((21, 4))
    with pytest.raises(MemoryError, match='neighbor list'):
        list(gram._pair_batches(cKDTree(points), points, 1., maximum_pairs=20))


def test_sparse_batches_retain_every_pair_once():
    points = np.c_[np.arange(25), np.zeros((25, 3))]
    pairs = [(int(i), int(j)) for ii,jj in gram._pair_batches(cKDTree(points),points,1.1,maximum_pairs=10)
             for i,j in zip(ii,jj)]
    expected = [(i,j) for i in range(25) for j in range(25) if abs(i-j) <= 1]
    assert sorted(pairs) == expected


def test_nonfinite_or_unordered_pair_radii_are_rejected():
    for values in ((), (0.,), (1., 1.), (2., 1.), (np.nan,), (np.inf,)):
        with pytest.raises(ValueError):
            gram._validate_radii(values)
