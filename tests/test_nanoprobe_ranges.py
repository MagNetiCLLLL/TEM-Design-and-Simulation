"""Range identity and transactional controls; physical presets are audited separately."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.operating_modes import (
    apply_operating_mode_pair, condenser_mode_key, direct_alignment_by_key,
    is_nanoprobe_mode, load_operating_mode_catalog, mode_by_key,
)
from temsim.optics.column import default_state
from temsim.optics.model import State


@pytest.mark.parametrize("attributes,expected", [
    ({"illumination_mode": "STEM"}, "nano_probe"),
    ({"illumination_mode": "STEM", "nanoprobe_range": "large_angle"}, "nano_probe_large_angle"),
    ({"illumination_mode": "TEM", "nanoprobe_range": "large_angle"}, "micro_probe"),
    ({"probe_mode": "nano_probe", "nanoprobe_range": "large_angle"}, "nano_probe_large_angle"),
    ({"probe_mode": "nano_probe_large_angle"}, "nano_probe_large_angle"),
    ({"probe_mode": "micro_probe", "illumination_mode": "STEM"}, "micro_probe"),
    ({"probe_mode": "invalid", "illumination_mode": "STEM"}, "nano_probe"),
    ({}, ""),
])
def test_condenser_mode_helper_preserves_observer_compatibility(attributes, expected):
    assert condenser_mode_key(SimpleNamespace(**attributes)) == expected
    assert is_nanoprobe_mode(expected) == expected.startswith("nano_probe")


def test_range_switch_keeps_cm_magnitude_and_microprobe_retains_selected_range():
    state = default_state()
    state.mini_condenser.signed_excitation_percent = -51.25
    for key, expected_range, polarity in (
        ("nano_probe_large_angle", "large_angle", -1),
        ("micro_probe", "large_angle", 1),
        ("nano_probe_large_angle", "large_angle", -1),
        ("nano_probe", "normal", -1),
    ):
        result = apply_operating_mode_pair(state, key, "diffraction")
        assert result.condenser.key == condenser_mode_key(state) == key
        assert state.nanoprobe_range == expected_range
        assert state.mini_condenser.signed_excitation_percent == polarity * 51.25
        assert state.mini_condenser.enabled
        c3 = next(lens for lens in state.lenses if lens.key == "condenser_lens_3")
        assert c3.percent == mode_by_key(key).devices[c3.key]["percent"]


@pytest.mark.parametrize("installed", [False, True])
def test_failed_range_switch_restores_controls_and_selected_range(monkeypatch, installed):
    import temsim.operating_modes as modes
    state = default_state()
    state.nanopulser.installed = installed
    state.nanoprobe_range = "normal"
    before = state.to_dict()
    apply_values = modes._apply_values

    def fail_after_condenser(candidate, mode):
        if mode.family == "projector":
            assert candidate.nanoprobe_range == "large_angle"
            raise ValueError("synthetic projector failure")
        return apply_values(candidate, mode)

    monkeypatch.setattr(modes, "_apply_values", fail_after_condenser)
    with pytest.raises(ValueError, match="synthetic projector"):
        apply_operating_mode_pair(state, "nano_probe_large_angle", "diffraction")
    assert state.to_dict() == before


@pytest.mark.parametrize("illumination", ["STEM", "TEM"])
def test_range_and_custom_controls_round_trip_without_applying_presets(monkeypatch, illumination):
    import temsim.operating_modes as modes
    state = default_state()
    state.illumination_mode = illumination
    state.nanoprobe_range = "large_angle"
    state.lenses[0].percent = 33.125
    state.objective_lens.percent = 67.25
    before = state.to_dict()
    monkeypatch.setattr(modes, "apply_operating_mode_pair", lambda *_a, **_k: pytest.fail("No presets on restore"))
    for restored in (State.from_dict(before), capture_instrument_snapshot(state).restore()):
        assert restored.nanoprobe_range == "large_angle"
        assert restored.illumination_mode == illumination
        actual = restored.to_dict()
        for key in ("lenses", "corrector_elements", "electron_gun", "stigmators"):
            assert actual[key] == before[key]


def test_previous_state_defaults_to_normal_without_changing_optics():
    state = default_state()
    state.lenses[0].percent = 33.125
    current = state.to_dict()
    previous = deepcopy(current)
    previous["schema_version"] = 79
    del previous["nanoprobe_range"]
    restored = State.from_dict(previous).to_dict()
    assert restored["nanoprobe_range"] == "normal"
    assert previous["schema_version"] == 79 and "nanoprobe_range" not in previous
    for key in ("lenses", "corrector_elements", "electron_gun", "stigmators"):
        assert restored[key] == current[key]


@pytest.mark.parametrize("invalid", ["large", "", None, True])
def test_invalid_range_is_rejected_in_state_and_profile(invalid):
    from temsim.profile_io import apply_profile_values
    state = default_state()
    payload = state.to_dict()
    payload["nanoprobe_range"] = invalid
    with pytest.raises(ValueError, match="Nanoprobe range"):
        State.from_dict(payload)
    with pytest.raises(ValueError):
        apply_profile_values(state, {"simulation": {"nanoprobe_range": invalid}})
    assert state.nanoprobe_range == "normal"


def test_profile_range_round_trip_and_legacy_default_preserve_live_values(tmp_path):
    import tomllib
    import tomli_w
    from temsim.profile_io import apply_profile_values, read_profile, save_profile
    state = default_state()
    state.nanoprobe_range = "large_angle"
    state.lenses[0].percent = 33.125
    path = tmp_path / "range.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    _, values = read_profile(path)
    restored = default_state()
    apply_profile_values(restored, values)
    assert restored.nanoprobe_range == "large_angle"
    assert restored.lenses[0].percent == state.lenses[0].percent
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    del document["devices"]["simulation"]["nanoprobe_range"]
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    _, legacy = read_profile(path)
    apply_profile_values(restored, legacy)
    assert restored.nanoprobe_range == "normal"
    assert restored.lenses[0].percent == state.lenses[0].percent
    restored.nanoprobe_range = "large_angle"
    apply_profile_values(restored, {"simulation": {"step_mm": 1.}})
    assert restored.nanoprobe_range == "large_angle"


def test_large_angle_blanker_recalibration_uses_nanoprobe_alignment(monkeypatch):
    import temsim.optics.condenser_recalibration as calibration
    state = default_state()
    state.nanoprobe_range = "large_angle"
    mode = mode_by_key("nano_probe_large_angle")
    seen = []

    def calibrated(candidate, selected, definition):
        seen.append((candidate.nanoprobe_range, selected.key, definition.key))
        return selected

    monkeypatch.setattr(calibration, "_recalibrate_transmitted_condenser", calibrated)
    monkeypatch.setattr(calibration, "_CACHE", {})
    calibration.recalibrate_nanopulser_condenser(state, mode, load_operating_mode_catalog())
    assert seen == [("large_angle", "nano_probe_large_angle", "nanoprobe_convergence")]


def test_large_angle_blanker_refines_focus_and_reports_waist(monkeypatch):
    import temsim.optics.condenser_recalibration as calibration
    import temsim.optics.direct_alignment as alignment
    state = default_state()
    state.nanoprobe_range = "large_angle"
    mode = mode_by_key("nano_probe_large_angle")
    definition = direct_alignment_by_key("nanoprobe_convergence")
    seen = []

    def refine(candidate, requested, target, initial, step):
        seen.append((requested.key, target))
        return initial, None

    monkeypatch.setattr(calibration, "_calibration_key", lambda *_a: "synthetic-large")
    monkeypatch.setattr(calibration, "_CACHE", {})
    monkeypatch.setattr(alignment, "_refine_nanoprobe_production_focus", refine)
    monkeypatch.setattr(alignment, "apply_direct_alignment", lambda *_a, **_k: SimpleNamespace(
        success=True, strengths={}, convergence_95_mrad=30., illumination_diameter_95_um=.001,
        constraint_value=1e-7, validation_step_mm=.05))
    result = calibration._recalibrate_transmitted_condenser(state, mode, definition)
    assert seen == [("nanoprobe_convergence", definition.default_value)]
    assert result.targets["achieved_waist_offset_nm"] == pytest.approx(.1)
    assert "achieved_wavefront_curvature_per_m" not in result.targets


def test_large_angle_assembly_target_uses_probe_focus_and_preserves_mode():
    from temsim.optics.assembly_illumination import load_targets, seed_illumination
    state = default_state()
    target = load_targets()["nano_probe_large_angle"]
    assert target.key == "nano_probe_large_angle" and target.diameter95_um is None
    statistics = SimpleNamespace(surviving_rays=32, convergence_95_mrad=target.alpha95_mrad,
                                 waist_offset_m=0.)
    np.testing.assert_array_equal(target.residual(statistics), np.zeros(2))
    seen = []
    measurement = SimpleNamespace(statistics=statistics,
        accepts=lambda *a, **kw: seen.append((a, kw)) or True)
    assert target.accepts(measurement) and len(seen) == 1
    seed_illumination(state, target.key)
    assert condenser_mode_key(state) == target.key
    assert state.mini_condenser.polarity == -1


def test_large_angle_does_not_reuse_normal_full_column_topology_reference():
    from temsim.topology_evidence import topology_reference
    state = default_state()
    state.nanoprobe_range = "large_angle"
    report = topology_reference(capture_instrument_snapshot(state))
    assert report["status"] == "NO_SCOPED_REFERENCE"
    assert report["qualification"] == "NOT_ESTABLISHED"
    assert "reference" not in report
    state.illumination_mode = "TEM"
    assert topology_reference(capture_instrument_snapshot(state))["status"] == "TARGET_ONLY"


@pytest.mark.parametrize("key", ["column_transport", "beam_centre_direction", "condenser_twofold_shape"])
def test_shared_alignments_support_both_nanoprobe_ranges(key):
    assert {"nano_probe", "nano_probe_large_angle"}.issubset(direct_alignment_by_key(key).active_mode_keys)
