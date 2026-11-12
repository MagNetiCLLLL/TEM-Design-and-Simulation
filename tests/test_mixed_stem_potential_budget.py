"""Mocked qualification bookkeeping, without a physical source or propagation."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import mixed_stem_scan as scan
from temsim.physics import mixed_stem_material


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), -1., True])
def test_invalid_potential_bound_cannot_enter_qualification(bound):
    with pytest.raises(ValueError, match="potential error bound"):
        scan._potential_error_fraction({"numerical_potential_error": {
            "detector_absolute_error_bound_per_incident_weight": bound}}, .5)


@pytest.mark.parametrize("finite_bound,qualified", [(.004, True), (.4, False)])
def test_known_potential_error_can_reject_identical_mock_diffraction(
        monkeypatch, finite_bound, qualified):
    # Identical mocked observations isolate the acceptance bookkeeping. These
    # uniform arrays are not a wave calculation or a finite-source trial.
    fraction = .5
    positions = np.zeros((5, 2))
    packet = SimpleNamespace(fraction=fraction, gx=np.array([0]), weights=np.array([fraction]))
    checkpoint = SimpleNamespace(beam=SimpleNamespace(modes=(SimpleNamespace(
        plane=SimpleNamespace(basis_m=np.eye(2)*1e-10)),)))
    request = scan.MixedStemScanRequest()
    monkeypatch.setattr(scan, "_admit_memory", lambda *_a, **_k: {})
    monkeypatch.setattr(scan, "_heldout_actual_positions", lambda _: (np.arange(5), 0, 0))
    monkeypatch.setattr(scan, "_periodic_potential", lambda *_a, **_k:
        (np.zeros((1, 4, 4)), np.ones(1), {}))
    monkeypatch.setattr(scan, "_field_model", lambda *_a, **_k:
        (None, {"detector_absolute_error_bound_per_incident_weight": 0.}))
    monkeypatch.setattr(scan, "_packet", lambda *_a, **_k: (packet, {}))
    monkeypatch.setattr(scan, "_detectors", lambda *_a, **_k:
        {"bf": np.zeros(5), "adf": np.zeros(5)})
    monkeypatch.setattr(mixed_stem_material, "finite_cif_potential", lambda *_a, **_k:
        (np.zeros((1, 8, 8)), np.ones(1), {"numerical_potential_error": {
            "detector_absolute_error_bound_per_incident_weight": finite_bound}}))

    def observe(_cp, _packet, potential, _slices, _fov, xy, _intervals, **kwargs):
        shape = kwargs.get("diffraction_grid_shape", potential.shape[-2:])
        pattern = np.full(shape, fraction/np.prod(shape))
        if kwargs.get("diffraction_per_position", False):
            pattern = np.broadcast_to(pattern, (len(xy), *shape))
        return np.zeros((2, len(xy))), np.zeros(len(xy)), pattern

    monkeypatch.setattr(scan, "resident_observe", observe)
    base = {"fov": (2., 2.), "plan": {"grid_yx": [4, 4], "slices": 1},
        "requested_pixels": 4, "raw_packet": packet, "field_absolute": 0.,
        "potential_absolute": .002}
    refined = {**base, "potential_absolute": .003}
    domain, extra_bound, _, record = scan._qualify_domain_and_diffraction(
        np, object(), object(), SimpleNamespace(mean_energy_ev=300000.), checkpoint, request, positions,
        base, refined, (np.ones(3),), (2, 2), 1., {}, lambda: False, None)
    comparison = record["raw_diffraction_comparison"]
    assert domain == {"bf": 0., "adf": 0.}
    assert comparison["grid_slice_l1_fraction"] == 0.
    assert comparison["domain_l1_fraction"] == 0.
    assert comparison["qualified"] is qualified
    # Base/refined errors are charged once by the scan, while this return adds
    # only the independently enlarged finite reference to its BF/ADF budget.
    assert extra_bound == pytest.approx(finite_bound*fraction)
    assert comparison["potential_remainder_absolute_fraction_bound"] == pytest.approx(
        .002+.003+finite_bound*fraction)
    assert comparison["combined_relative_l1"] == pytest.approx(
        (.002+.003+finite_bound*fraction)/fraction)
