"""Explicit N/I equivalence, immutable field inputs and atomic CAD drafts.

These are component/operator checks. No finite source is emitted or transported.
"""
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from temsim.coil_reference_metadata import OWNER_CURRENT_FIELD, round_coil_operating_point
from temsim.physics.potential_capture import capture_round_lens_potential


@pytest.fixture(autouse=True)
def no_source_transport(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Winding definition checks must not emit or transport particles")
    monkeypatch.setattr("temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit", forbidden)
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", forbidden)


def _coil(key="wire", turns=100, diameter=.25):
    return dict(key=key, winding_turn_count=turns, winding_wire_diameter_mm=diameter)


def test_explicit_series_windings_share_current_and_conserve_signed_ampere_turns():
    owner = dict(key="round", **{OWNER_CURRENT_FIELD: 2.})
    result = round_coil_operating_point(owner, [_coil("upper", 75), _coil("lower", 25)],
                                        percent=25., polarity=-1)
    assert result["reference_current_a"] == 2.
    assert result["current_a"] == -.5
    assert result["reference_ampere_turns"] == 200.
    assert result["ampere_turns"] == -50.
    assert result["total_turn_count"] == 100
    assert result["winding_reference_ampere_turns"] == {"upper": 150., "lower": 50.}


@pytest.mark.parametrize("current", [None, -1., float("nan"), float("inf"), True, "1"])
def test_invalid_prescribed_current_never_becomes_a_legacy_reference(current):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        round_coil_operating_point({"key": "round", OWNER_CURRENT_FIELD: current,
                                   "air_core_reference_ampere_turns": 3000.}, [_coil()])


def test_zero_current_and_disabled_lens_have_zero_ni_without_losing_turns():
    owner = {OWNER_CURRENT_FIELD: 0.}
    result = round_coil_operating_point(owner, [_coil()])
    assert result["ampere_turns"] == result["current_a"] == 0.
    owner[OWNER_CURRENT_FIELD] = 3.
    off = round_coil_operating_point(owner, [_coil()], enabled=False)
    assert off["total_turn_count"] == 100
    assert off["current_a"] == off["ampere_turns"] == 0.


def test_unrepresentable_turn_product_is_rejected_as_an_input_error():
    with pytest.raises(ValueError, match="finite numerical range"):
        round_coil_operating_point({OWNER_CURRENT_FIELD: 1.}, [_coil(turns=10**400)])


def test_legacy_ni_is_not_converted_into_a_guessed_winding():
    owner = dict(key="round", air_core_reference_ampere_turns=3000.)
    before = deepcopy(owner)
    result = round_coil_operating_point(owner, [], percent=37.125, polarity=-1)
    assert result["ampere_turns"] == 3000.*37.125*.01*-1
    assert result["current_a"] is result["total_turn_count"] is None
    assert owner == before


@pytest.fixture(scope="module")
def installed():
    from temsim.optics.column import default_state
    state = default_state()
    state.simulation_mode = "unified"
    state.round_lens_field_model = "copper_coil"
    return state


def _state(installed):
    state = copy(installed)
    state.lenses = deepcopy(installed.lenses)
    state._runtime_copper_coil_provider_cache = {}
    state._runtime_lens_field_provider_cache = {}
    # These equivalence tests deliberately start from an old NI-only input,
    # independently of the newly explicit catalog winding defaults.
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(part,data={k:v for k,v in part.data.items() if k != OWNER_CURRENT_FIELD})
        if part.key == "intermediate_lens" else part for part in state._resolved_assembly.parts))
    return state


def _edit(state, key, **fields):
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, **fields}) if part.key == key else part
        for part in state._resolved_assembly.parts))


def _define(state, *, turns=100, current=30., diameter=.25, key="intermediate_lens"):
    coil = next(part for part in state._resolved_assembly.parts
                if part.data.get("mechanical_profile") == "magnetic_excitation_coil"
                and part.data.get("field_source_key") == key)
    _edit(state, coil.key, winding_turn_count=turns, winding_interval_turn_counts=[turns],
          winding_wire_diameter_mm=diameter)
    _edit(state, key, **{OWNER_CURRENT_FIELD: current})
    return coil.key


def test_same_envelope_and_ni_preserve_field_while_wire_size_is_cad_only(installed):
    state = _state(installed)
    key = "intermediate_lens"
    legacy = capture_round_lens_potential(state, key).potential
    coil_key = _define(state)
    explicit = capture_round_lens_potential(state, key).potential
    center = explicit.coils[0].center_z_m
    points = np.array([[0., 0., center], [1e-4, -2e-4, center+.01], [0., 0., center+.2]])
    np.testing.assert_allclose(explicit.sample(points)["B_t"], legacy.sample(points)["B_t"],
                               rtol=4e-14, atol=1e-20)
    assert [(c.inner_radius_m, c.outer_radius_m, c.length_m, c.center_z_m) for c in explicit.coils] == [
        (c.inner_radius_m, c.outer_radius_m, c.length_m, c.center_z_m) for c in legacy.coils]
    before = explicit.sample(points)["B_t"]
    _edit(state, coil_key, winding_wire_diameter_mm=.125)
    thinner = capture_round_lens_potential(state, key).potential
    np.testing.assert_array_equal(thinner.sample(points)["B_t"], before)
    assert thinner.digest != explicit.digest  # Captured provenance remains distinct.
    _edit(state, coil_key, winding_turn_count=200, winding_interval_turn_counts=[200])
    _edit(state, key, **{OWNER_CURRENT_FIELD: 15.})
    equal_ni = capture_round_lens_potential(state, key).potential
    np.testing.assert_array_equal(equal_ni.sample(points)["B_t"], before)
    assert equal_ni.provenance["reference_status"].startswith("user_defined")
    assert equal_ni.provenance["legacy_reference_record"]["active"] is False


def test_explicit_current_overrides_legacy_ni_and_changes_upstream_cache_identity(installed):
    from temsim.round_lens_models import round_lens_cache_identity
    from temsim.calculation_cache import calculation_signatures
    state = _state(installed)
    key = "intermediate_lens"
    _define(state)
    original = capture_round_lens_potential(state, key).potential
    cache = round_lens_cache_identity(state)
    signatures = calculation_signatures(state)
    points = [[0., 0., original.coils[0].center_z_m]]
    _edit(state, key, air_core_reference_ampere_turns=999999.)
    inactive = capture_round_lens_potential(state, key).potential
    np.testing.assert_array_equal(inactive.sample(points)["B_t"], original.sample(points)["B_t"])
    _edit(state, key, **{OWNER_CURRENT_FIELD: 60.})
    doubled = capture_round_lens_potential(state, key).potential
    np.testing.assert_allclose(doubled.sample(points)["B_t"], 2*original.sample(points)["B_t"], rtol=4e-14)
    assert round_lens_cache_identity(state) != cache
    assert calculation_signatures(state)["incident"] != signatures["incident"]
    restored = type(doubled).from_dict(doubled.to_dict())
    np.testing.assert_array_equal(restored.sample(points)["B_t"], doubled.sample(points)["B_t"])


def test_area_only_wire_input_preserves_field_and_records_exact_metal_area(installed):
    from temsim.round_lens_models import round_lens_cache_identity
    state = _state(installed)
    coil_key = _define(state)
    before = capture_round_lens_potential(state, "intermediate_lens").potential
    identity = round_lens_cache_identity(state)
    def set_area(area):
        state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
            replace(part,data={**{k:v for k,v in part.data.items() if k != "winding_wire_diameter_mm"},
                               "winding_wire_cross_section_mm2":area}) if part.key==coil_key else part
            for part in state._resolved_assembly.parts))
    set_area(.1)
    area_defined = capture_round_lens_potential(state, "intermediate_lens").potential
    assert round_lens_cache_identity(state) != identity
    assert area_defined.digest != before.digest
    assert [c.to_metadata() for c in area_defined.coils] == [c.to_metadata() for c in before.coils]
    witness = area_defined.provenance["winding_current_definition"]["windings"][0]
    assert witness["wire_cross_section_mm2"] == .1
    assert witness["conductor_cross_section_mm2"] == 10.
    assert witness["wire_diameter_mm"] == pytest.approx(np.sqrt(.4/np.pi))
    center = area_defined.coils[0].center_z_m
    points = [[0.,0.,center], [1e-4,-2e-4,center+.01]]
    np.testing.assert_array_equal(area_defined.sample(points)["B_t"],before.sample(points)["B_t"])
    set_area(.2)
    larger = capture_round_lens_potential(state, "intermediate_lens").potential
    np.testing.assert_array_equal(larger.sample(points)["B_t"],before.sample(points)["B_t"])
    assert larger.provenance["winding_current_definition"]["windings"][0]["wire_cross_section_mm2"] == .2


def test_new_winding_current_inputs_are_not_ignored_as_legacy_metadata():
    from temsim.input_design_compatibility import additive_coil_parts_metadata_only
    original = [dict(key="round", mechanical_profile="magnetic_lens_assembly"),
                dict(key="wire", mechanical_profile="magnetic_excitation_coil", field_source_key="round")]
    changed = deepcopy(original)
    changed[0][OWNER_CURRENT_FIELD] = 1.
    changed[1].update(winding_turn_count=100, winding_wire_diameter_mm=.25)
    assert not additive_coil_parts_metadata_only(original, changed)


def test_atomic_winding_draft_rejects_overcapacity_and_roundtrips_explicit_inputs(tmp_path):
    from temsim.part_model_document import PartModelDocument
    from temsim.module_manifest import read_document
    path = Path(__file__).resolve().parents[1]/"configs/instruments/column/C3_ProbeCorrector_ImageCorrector.toml"
    original_bytes = path.read_bytes()
    draft = PartModelDocument(path)
    key, owner = "condenser_lens_2_excitation_coil", "condenser_lens_2"
    baseline = deepcopy(draft.document)
    history = deepcopy(draft._history)
    with pytest.raises(ValueError):
        draft.configure_winding(key, winding_turn_count=1000000,
            winding_interval_turn_counts=[1000000],
            winding_wire_diameter_mm=100., ideal_coil_current_a_at_100_percent=30.)
    assert draft.document == baseline and draft._history == history
    draft.configure_winding(key, winding_turn_count=100,
        winding_interval_turn_counts=[100],
        winding_wire_diameter_mm=.5, ideal_coil_current_a_at_100_percent=30.)
    assert len(draft._history) == len(history)+1
    assert draft.part(key)["winding_turn_count"] == 100
    assert draft.part(owner)[OWNER_CURRENT_FIELD] == 30.
    draft.undo()
    assert draft.document == baseline
    draft.redo()
    output = tmp_path/"explicit-round-winding.toml"
    draft.save_copy(output)
    rows = {part["key"]: part for part in read_document(output)["parts"]}
    assert rows[key]["winding_turn_count"] == 100 and type(rows[key]["winding_turn_count"]) is int
    assert rows[key]["winding_wire_diameter_mm"] == .5
    assert rows[owner][OWNER_CURRENT_FIELD] == 30.
    assert path.read_bytes() == original_bytes
