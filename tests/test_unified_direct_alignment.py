"""Unqualified canonical/fit routes cannot silently replace unified fields."""
from types import SimpleNamespace

import pytest

import temsim.optics.direct_alignment as alignment


@pytest.mark.parametrize("entry", [
    lambda state: alignment.canonical_source_basis(state, 1000.),
    lambda state: alignment.canonical_transfers(state, 1000., (1001.,)),
    lambda state: alignment.canonical_transfers(state, 1000., (1000.,)),
    lambda state: alignment._LiveFirstOrderModel(state, 1000., 1001., ("c2",), step_mm=.1),
    lambda state: alignment._EquivalentImageFirstOrderModel(state, 1000., 1001.),
    lambda state: alignment._ProjectorMeasurementModel(state,
        SimpleNamespace(key=alignment.IMAGE_MAGNIFICATION), step_mm=.1),
    lambda state: alignment._solve_direct_alignment(state, alignment.IMAGE_MAGNIFICATION, 100.),
    lambda state: alignment.apply_direct_alignment(state, alignment.IMAGE_MAGNIFICATION, 100.),
])
def test_unified_rejects_unqualified_calibration_before_legacy_fields_or_fit(monkeypatch, entry):
    state = SimpleNamespace(simulation_mode="unified", equivalent_image_lenses_enabled=True,
                            projector_mode="Image")
    before = dict(vars(state))
    def forbidden(*_args, **_kwargs):
        pytest.fail("Unified diagnostics must not consume legacy fitted optics")
    for name in ("fields", "equivalent_image_calibrations", "_rk4_transfer_matrices",
                 "_rk4_axisymmetric_larmor_matrices", "_get_vector"):
        monkeypatch.setattr(alignment, name, forbidden)
    with pytest.raises(ValueError, match="not yet qualified.*unified Finite-Field"):
        entry(state)
    assert vars(state) == before


@pytest.mark.parametrize("mode,expected", [("ideal", "equivalent"), ("ideal_thin", "live")])
def test_historical_and_thin_image_route_selection_is_preserved(monkeypatch, mode, expected):
    # This test checks admission/routing only; independent numerical suites
    # qualify the existing finite-field and thin diagnostic implementations.
    state = SimpleNamespace(simulation_mode=mode)
    selected = []
    monkeypatch.setattr("temsim.instrument_snapshot.capture_instrument_snapshot",
                        lambda value: SimpleNamespace(restore=lambda: value))
    monkeypatch.setattr(alignment, "tem_projection_reference_plane", lambda _state: SimpleNamespace(z_mm=2000.))
    monkeypatch.setattr(alignment, "equivalent_image_lenses_enabled", lambda _state: True)
    state.sample = SimpleNamespace(z_mm=1000.)
    def factory(label):
        def model(*_args, **_kwargs):
            selected.append(label)
            return SimpleNamespace(upper=(100.,) * 5)
        return model
    monkeypatch.setattr(alignment, "_EquivalentImageFirstOrderModel", factory("equivalent"))
    monkeypatch.setattr(alignment, "_LiveFirstOrderModel", factory("live"))
    alignment._ProjectorMeasurementModel(state,
        SimpleNamespace(key=alignment.IMAGE_MAGNIFICATION), step_mm=.1)
    assert selected == [expected]
