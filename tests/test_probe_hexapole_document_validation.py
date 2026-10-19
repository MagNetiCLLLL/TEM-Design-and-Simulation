"""Probe hexapole material dimensions use the normal transactional editor."""

from copy import deepcopy
import tomllib

import pytest
import tomli_w

from temsim import module_manifest
from temsim.part_model_3d import part_dimension_specs
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


KEY = "probe_hp2_hexapole"
POLE_LENGTH = "mechanical_hexapole_pole_length_mm"


@pytest.fixture
def draft(tmp_path):
    from temsim.shared_tip import copy_catalog_inputs

    root = tmp_path / "instruments"
    relative = "column/C3_ProbeCorrector.toml"
    copy_catalog_inputs(INSTRUMENT_CONFIG_ROOT, root, (relative,))
    source = PartModelDocument(root / relative)
    # Materialize a standalone file so the checks also cover old saved files
    # gaining an explicit physical dimension, with no dependency rewrites.
    document = tomllib.loads(source.independent_copy_text())
    part = next(row for row in document["parts"] if row["key"] == KEY)
    part.setdefault(POLE_LENGTH, .8 * (part["local_end_z_mm"] - part["local_start_z_mm"]))
    path = tmp_path / "probe_material.toml"
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    return PartModelDocument(path)


@pytest.mark.parametrize("value", [0., -1., 1000., float("nan"), float("inf"), True])
def test_module_load_rejects_invalid_physical_pole_lengths_without_mutation(draft, value):
    candidate = deepcopy(draft.document)
    part = next(row for row in candidate["parts"] if row["key"] == KEY)
    part[POLE_LENGTH] = value
    before = tomli_w.dumps(candidate)
    with pytest.raises(ValueError):
        module_manifest.validate_document(candidate)
    assert tomli_w.dumps(candidate) == before

    invalid_file = draft.path.with_name("invalid_probe.toml")
    invalid_file.write_text(before, encoding="utf-8")
    original_bytes = invalid_file.read_bytes()
    with pytest.raises(ValueError):
        PartModelDocument(invalid_file)
    assert invalid_file.read_bytes() == original_bytes


@pytest.mark.parametrize("value", [0., -1., 1000., float("nan"), float("inf"), True])
def test_rejected_material_edit_preserves_draft_source_and_undo_state(draft, value):
    original = deepcopy(draft.document)
    original_bytes = draft.path.read_bytes()
    history = deepcopy(draft._history)
    index = draft._history_index
    with pytest.raises(ValueError):
        draft.set_dimension(("parts", KEY, POLE_LENGTH), value)
    assert draft.document == original
    assert draft.path.read_bytes() == original_bytes
    assert draft._history == history and draft._history_index == index
    assert not draft.dirty


def test_material_length_is_editable_saved_and_independent_of_effective_field(draft, tmp_path):
    dimensions = {item.path[-1]: item for item in part_dimension_specs(draft.document, KEY)}
    assert dimensions[POLE_LENGTH].editable
    assert dimensions[POLE_LENGTH].unit == "mm"
    original = deepcopy(draft.part(KEY))
    original_bytes = draft.path.read_bytes()
    changed_length = original[POLE_LENGTH] * .99
    draft.set_dimension(("parts", KEY, POLE_LENGTH), changed_length)
    expected = {**original, POLE_LENGTH: changed_length}
    assert draft.part(KEY) == expected
    assert draft.path.read_bytes() == original_bytes
    draft.undo()
    assert draft.part(KEY) == original
    draft.redo()
    assert draft.part(KEY) == expected
    saved = tmp_path / "edited_probe.toml"
    draft.save_copy(saved)
    assert PartModelDocument(saved).part(KEY) == expected


def test_envelope_shrink_does_not_silently_rescale_saved_pole_length(draft):
    original = deepcopy(draft.document)
    source = draft.path.read_bytes()
    with pytest.raises(ValueError):
        draft.set_dimension(("parts", KEY, "length_mm"), draft.part(KEY)[POLE_LENGTH] / 2.)
    assert draft.document == original
    assert draft.path.read_bytes() == source
    assert not draft.dirty


def test_mechanical_copy_retains_probe_validation_through_template_identity(draft):
    draft.copy_component_from(draft.document, KEY, "copied_probe_hp", 670., include_children=False)
    copied = draft.part("copied_probe_hp")
    assert copied["geometry_template_key"] == KEY
    original = deepcopy(draft.document)
    with pytest.raises(ValueError):
        draft.set_dimension(("parts", copied["key"], POLE_LENGTH), 1000.)
    assert draft.document == original
