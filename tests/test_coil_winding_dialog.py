"""Explicit winding UI and atomic drafts; no source, field or mesh calculation."""
from copy import deepcopy
from types import SimpleNamespace
import math

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLineEdit

from temsim.gui.coil_winding_dialog import CURRENT_FIELD, WindingConfigurationDialog
from temsim.coil_winding import WIRE_AREA_FIELD, INTERVAL_TURNS_FIELD
from temsim.gui.part_model_editor import PartModelEditorPage
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


def records():
    owner = dict(key="lens", name="Test lens", mechanical_profile="magnetic_lens_assembly",
                 air_core_reference_ampere_turns=3000.)
    coil = dict(key="coil", name="Test winding", parent_key="lens", field_source_key="lens",
                mechanical_profile="magnetic_excitation_coil", material_class="insulated_copper_winding",
                local_start_z_mm=10., local_end_z_mm=18.,
                mechanical_inner_diameter_mm=4., mechanical_outer_diameter_mm=8.)
    return coil, owner, {"lens": owner, "coil": coil}


def fill(dialog, *, turns="12", wire="0.4", current="2.5"):
    dialog.turn_count.setText(turns)
    area = "" if wire == "" else format(math.pi*float(wire)**2/4, ".17g")
    dialog.wire_cross_section.setText(area)
    dialog.current_at_100_percent.setText(current)


def make_dialog(qtbot, **kwargs):
    coil, owner, parts = records()
    dialog = WindingConfigurationDialog(coil, owner, parts, **kwargs)
    qtbot.addWidget(dialog)
    return dialog, parts


def test_unknown_legacy_winding_has_no_guessed_turn_wire_or_current_defaults(qtbot):
    dialog, parts = make_dialog(qtbot)
    before = deepcopy(parts)
    assert all(control.text() == "" for control in
               (dialog.turn_count, dialog.wire_cross_section, dialog.current_at_100_percent))
    assert not dialog.apply_button.isEnabled()
    assert "ID 4 mm" in dialog.envelope_label.text()
    assert "10 to 18 mm" in dialog.envelope_label.text()
    assert "No historical turn count" in dialog.legacy_label.text()
    assert dialog.envelope_label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    assert dialog.packing.isReadOnly()
    assert parts == before


def test_packing_and_signed_operating_current_use_original_envelope_without_mutation(qtbot):
    dialog, parts = make_dialog(qtbot, runtime_values=dict(percent=40., polarity=-1, enabled=True))
    before = deepcopy(parts)
    fill(dialog)
    assert dialog.apply_button.isEnabled()
    assert dialog.values() == dict(winding_turn_count=12, winding_wire_cross_section_mm2=math.pi*.4**2/4,
                                  winding_wire_diameter_mm=None, ideal_coil_current_a_at_100_percent=2.5)
    assert "signed conductor current -1 A" in dialog.operating_label.text()
    assert "30 A·turn at 100%; -12 A·turn" in dialog.operating_label.text()
    assert "Conductor turns: 12" in dialog.packing.toPlainText()
    assert "Remaining insulation volume:" in dialog.packing.toPlainText()
    assert dialog._geometry.field_envelope == dialog.envelope
    assert parts == before


def test_missing_polarity_and_disconnected_runtime_are_explicit(qtbot):
    dialog, _ = make_dialog(qtbot, runtime_values=dict(percent=40.))
    fill(dialog)
    assert "current magnitude 1 A; polarity readback is unavailable" in dialog.operating_label.text()
    dialog.set_runtime_values({})
    assert "Actual current readback unavailable" in dialog.operating_label.text()
    dialog.set_runtime_values(dict(percent=101., polarity=1))
    assert "outside 0 to 100%" in dialog.operating_label.text()


def test_disabled_excitation_and_explicit_zero_reference_current_are_supported(qtbot):
    dialog, _ = make_dialog(qtbot, runtime_values=dict(percent=40., polarity=-1, enabled=False))
    fill(dialog)
    assert "Excitation disabled; signed conductor current 0 A" in dialog.operating_label.text()
    dialog.set_runtime_values(dict(percent=40., polarity=-1, enabled=True))
    fill(dialog, current="0")
    assert dialog.apply_button.isEnabled()
    assert dialog.values()[CURRENT_FIELD] == 0.


def test_fixed_turns_and_large_current_are_valid_without_inventing_wire_diameter(qtbot):
    dialog, parts = make_dialog(qtbot, runtime_values=dict(percent=40., polarity=1, enabled=True))
    before = deepcopy(parts)
    fill(dialog, turns="1172", wire="", current="1575.233")
    assert dialog.apply_button.isEnabled()
    assert dialog.values()["winding_turn_count"] == 1172
    assert dialog.values()["winding_wire_diameter_mm"] is None
    assert dialog._geometry is None
    assert "metal cross-section is unknown; envelope CAD is retained" in dialog.packing.toPlainText()
    assert "metal area and insulation fraction are unavailable" in dialog.packing.toPlainText()
    assert "Remaining insulation volume:" not in dialog.packing.toPlainText()
    dialog.current_at_100_percent.setText("10000000")
    assert dialog.apply_button.isEnabled()
    assert dialog.turn_count.text() == "1172"
    assert parts == before


@pytest.mark.parametrize("current", [1575.2330285262728, 1.2345678901234567e100])
def test_saved_current_prefill_roundtrips_exactly_without_an_edit(qtbot, current):
    coil, owner, parts = records()
    coil["winding_turn_count"] = 12
    owner[CURRENT_FIELD] = current
    dialog = WindingConfigurationDialog(coil, owner, parts)
    qtbot.addWidget(dialog)
    assert dialog.apply_button.isEnabled()
    assert float(dialog.current_at_100_percent.text()) == current
    assert dialog.values()[CURRENT_FIELD] == current


def test_user_edits_only_metal_area_and_diameter_is_calculated_read_only(qtbot):
    dialog, _ = make_dialog(qtbot)
    fill(dialog,turns='4')
    assert isinstance(dialog.wire_cross_section,QLineEdit)
    assert not isinstance(dialog.wire_diameter,QLineEdit)
    dialog.wire_cross_section.setText("1")
    assert dialog.apply_button.isEnabled()
    assert dialog.values()[WIRE_AREA_FIELD] == 1.
    assert dialog.values()["winding_wire_diameter_mm"] is None
    assert "1.12837917 mm (calculated)" == dialog.wire_diameter.text()


def test_area_and_turn_edits_repack_insulation_inside_the_unchanged_envelope(qtbot):
    dialog, _ = make_dialog(qtbot)
    fill(dialog)
    original = dialog._geometry.layout.to_metadata()
    envelope = deepcopy(dialog.envelope)
    dialog.wire_cross_section.setText("0.5")
    assert dialog.apply_button.isEnabled()
    larger = dialog._geometry.layout.to_metadata()
    assert larger['conductor_volume_mm3'] > original['conductor_volume_mm3']
    assert larger['insulation_volume_mm3'] < original['insulation_volume_mm3']
    assert dialog._geometry.field_envelope == envelope
    dialog.turn_count.setText("20")
    assert dialog.apply_button.isEnabled()
    more = dialog._geometry.layout.to_metadata()
    assert more['conductor_volume_mm3'] > larger['conductor_volume_mm3']
    assert more['insulation_volume_mm3'] < larger['insulation_volume_mm3']
    assert dialog._geometry.field_envelope == envelope
    assert "Remaining insulation volume:" in dialog.packing.toPlainText()


def test_unchanged_legacy_diameter_is_not_reencoded_by_the_area_readback(qtbot):
    coil, owner, parts = records()
    diameter=.12345678901234567
    coil.update(winding_turn_count=12,winding_wire_diameter_mm=diameter)
    owner[CURRENT_FIELD]=2.5
    dialog=WindingConfigurationDialog(coil,owner,parts)
    qtbot.addWidget(dialog)
    assert dialog.apply_button.isEnabled()
    assert dialog.values()['winding_wire_diameter_mm']==diameter
    assert WIRE_AREA_FIELD not in dialog.values()


def test_area_only_prefill_preserves_its_exact_saved_value(qtbot):
    coil, owner, parts = records()
    area=.12345678901234567
    coil.update(winding_turn_count=12,**{WIRE_AREA_FIELD:area})
    owner[CURRENT_FIELD]=2.5
    dialog=WindingConfigurationDialog(coil,owner,parts)
    qtbot.addWidget(dialog)
    assert dialog.apply_button.isEnabled()
    assert dialog.values()[WIRE_AREA_FIELD]==area
    assert 'winding_wire_diameter_mm' not in dialog.values()


def test_single_interval_total_edit_updates_its_identical_count_and_insulation(qtbot):
    coil,owner,parts=records()
    coil.update(winding_turn_count=12,winding_interval_turn_counts=[12],
                **{WIRE_AREA_FIELD:.1})
    owner[CURRENT_FIELD]=2.5
    dialog=WindingConfigurationDialog(coil,owner,parts)
    qtbot.addWidget(dialog)
    before=dialog._geometry.layout.area.insulation_volume_mm3
    assert dialog.interval_turn_counts.isHidden()
    dialog.turn_count.setText('20')
    assert dialog.apply_button.isEnabled()
    assert dialog.values()[INTERVAL_TURNS_FIELD]==[20]
    assert dialog._geometry.layout.area.insulation_volume_mm3 < before


@pytest.mark.parametrize("module,coil_key", [
    ("column/C2.toml", "condenser_lens_1_excitation_coil"),
    ("column/C3_ProbeCorrector.toml", "probe_tl21_lens_excitation_coil"),
    ("column/C3_BCOR.toml", "image_tl31_lens_excitation_coil"),
    ("project_and_recording_system/EnergyFilter.toml", "projector_lens_2_excitation_coil"),
])
def test_catalog_round_lens_families_share_winding_action_and_current_inputs(qtbot, module, coil_key):
    document = PartModelDocument(INSTRUMENT_CONFIG_ROOT / module)
    before = deepcopy(document.document)
    context = SimpleNamespace(session=document, _selected_key=coil_key)
    part, owner, parts = PartModelEditorPage._winding_context(context)
    dialog = WindingConfigurationDialog(part, owner, parts,
        runtime_values=dict(percent=40., polarity=-1, enabled=True))
    qtbot.addWidget(dialog)
    dialog.turn_count.setText(str(part.get("winding_turn_count", 12)))
    dialog.wire_cross_section.setText("")
    dialog.current_at_100_percent.setText("10000000")
    assert dialog.apply_button.isEnabled()
    assert dialog.values()[CURRENT_FIELD] == 10000000.
    assert "signed conductor current -4000000 A" in dialog.operating_label.text()
    assert document.document == before


def test_split_turns_remain_fixed_when_current_changes_and_total_must_match(qtbot):
    coil, owner, parts = records()
    coil.update(material_intervals_mm=[[10., 13.], [15., 18.]],
                winding_turn_count=1172, winding_interval_turn_counts=[586, 586])
    dialog = WindingConfigurationDialog(coil, owner, parts)
    qtbot.addWidget(dialog)
    dialog.current_at_100_percent.setText("1575.233")
    assert dialog.apply_button.isEnabled()
    assert dialog.interval_turn_counts.text() == "586, 586"
    assert dialog.values()["winding_interval_turn_counts"] == [586, 586]
    assert "Fixed turns per axial interval: 586, 586" in dialog.packing.toPlainText()
    returned = dialog.values()
    returned["winding_interval_turn_counts"][0] = 1
    assert dialog.values()["winding_interval_turn_counts"] == [586, 586]
    dialog.current_at_100_percent.setText("1000000")
    assert dialog.turn_count.text() == "1172"
    assert dialog.interval_turn_counts.text() == "586, 586"
    dialog.turn_count.setText("1173")
    assert not dialog.apply_button.isEnabled()
    assert "must sum" in dialog.validation.text()
    assert dialog.interval_turn_counts.text() == "586, 586"


@pytest.mark.parametrize("counts", ["586", "586, 586, 1", "586, -586", "586, 585", "586.0, 586", "586,"])
def test_invalid_interval_turns_do_not_enable_apply(qtbot, counts):
    coil, owner, parts = records()
    coil["material_intervals_mm"] = [[10., 13.], [15., 18.]]
    dialog = WindingConfigurationDialog(coil, owner, parts)
    qtbot.addWidget(dialog)
    fill(dialog, turns="1172", wire="", current="2.5")
    dialog.interval_turn_counts.setText(counts)
    assert not dialog.apply_button.isEnabled()


@pytest.mark.parametrize("changes", [dict(turns="0"), dict(turns="1.5"), dict(turns="-1"),
    dict(wire="0"), dict(wire="4"), dict(wire="nan"), dict(current="-1"),
    dict(current="inf"), dict(current="1e308")])
def test_invalid_or_overfull_inputs_do_not_enable_apply_or_mutate_inputs(qtbot, changes):
    dialog, parts = make_dialog(qtbot)
    before = deepcopy(parts)
    fill(dialog, **changes)
    assert not dialog.apply_button.isEnabled()
    assert parts == before
    with pytest.raises(ValueError):
        dialog.values()


def test_cancel_does_not_call_commit(qtbot):
    calls = []
    dialog, _ = make_dialog(qtbot, commit=calls.append)
    fill(dialog)
    dialog.show()
    qtbot.mouseClick(dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel), Qt.MouseButton.LeftButton)
    assert not calls
    assert dialog.result() == QDialog.DialogCode.Rejected


def test_commit_failure_preserves_dialog_and_success_is_one_explicit_call(qtbot):
    calls = []

    def commit(values):
        calls.append(values)
        if len(calls) == 1:
            raise ValueError("Whole-document validation rejected this draft")

    dialog, _ = make_dialog(qtbot, commit=commit)
    fill(dialog)
    dialog.show()
    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    assert dialog.isVisible()
    assert "Whole-document validation rejected" in dialog.validation.text()
    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    assert len(calls) == 2 and calls[0] == calls[1] == dialog.values()
    assert dialog.result() == QDialog.DialogCode.Accepted


MODULE = "project_and_recording_system/EnergyFilter.toml"
COIL = "intermediate_lens_excitation_coil"
OWNER = "intermediate_lens"


@pytest.fixture
def page(qtbot, tmp_path, monkeypatch):
    # Native file loading and document transactions are real. CAD rendering,
    # intersection jobs and physical calculations are outside this UI test.
    monkeypatch.setattr(PartModelEditorPage, "_render", lambda *args, **kwargs: True)
    monkeypatch.setattr(PartModelEditorPage, "_queue_overlap_check", lambda *args, **kwargs: None)
    from temsim.physics import simulation
    monkeypatch.setattr(simulation, "run", lambda *args, **kwargs: pytest.fail("No physical transport in a winding UI test"))
    path = tmp_path / "model.toml"
    PartModelDocument(INSTRUMENT_CONFIG_ROOT / MODULE).save_copy(path)
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    page.set_project_context(tmp_path, SimpleNamespace(selected_module_paths=(("recording", "model.toml"),)),
        None, runtime_values={OWNER: dict(percent=40., polarity=-1, enabled=True)})
    assert page.open_path(path, selected_key=COIL)
    return page


def test_round_winding_action_stages_owner_and_coil_as_one_undo_without_saving(page, qtbot, monkeypatch):
    before = deepcopy(page.session.document)
    original_bytes = page.session.path.read_bytes()
    saved_signals = []
    page.saved.connect(saved_signals.append)
    assert page.configure_winding_button.isEnabled()
    calls = []
    configure = page.session.configure_winding

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return configure(*args, **kwargs)

    monkeypatch.setattr(page.session, "configure_winding", record)
    dialog = page.configure_winding()
    assert dialog is page._winding_dialog
    fill(dialog, turns="12", wire="0.2", current="2.5")
    dialog.interval_turn_counts.setText("12")
    assert "signed conductor current -1 A" in dialog.operating_label.text()
    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    assert len(calls) == 1 and calls[0][0] == (COIL,)
    assert page._winding_dialog is None
    assert page.session.part(COIL)["winding_turn_count"] == 12
    assert page.session.part(COIL)[WIRE_AREA_FIELD] == math.pi*.2**2/4
    assert "winding_wire_diameter_mm" not in page.session.part(COIL)
    assert page.session.part(OWNER)[CURRENT_FIELD] == 2.5
    assert page.session.dirty and page.session.path.read_bytes() == original_bytes
    assert not saved_signals
    page.undo()
    assert page.session.document == before
    page.redo()
    assert page.session.part(COIL)["winding_turn_count"] == 12
    assert page.session.part(OWNER)[CURRENT_FIELD] == 2.5


def test_dialog_rejects_stale_draft_and_runtime_updates_do_not_mutate_it(page, qtbot):
    original_current = page.session.part(OWNER)[CURRENT_FIELD]
    dialog = page.configure_winding()
    fill(dialog, wire="0.2")
    dialog.interval_turn_counts.setText("12")
    page.set_runtime_values({OWNER: dict(percent=20., polarity=1, enabled=True)})
    assert "signed conductor current 0.5 A" in dialog.operating_label.text()
    page.session.set_dimension(("parts", COIL, "mechanical_inner_diameter_mm"), 70.)
    current = deepcopy(page.session.document)
    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    assert "draft changed" in dialog.validation.text()
    assert page.session.document == current
    assert page.session.part(OWNER)[CURRENT_FIELD] == original_current


def test_round_winding_action_can_clear_known_wire_size_and_undo_as_one_change(page, qtbot):
    page.session.configure_winding(COIL, winding_turn_count=12, winding_wire_diameter_mm=.2,
        winding_interval_turn_counts=[12], ideal_coil_current_a_at_100_percent=2.5)
    before = deepcopy(page.session.document)
    dialog = page.configure_winding()
    dialog.wire_cross_section.clear()
    dialog.current_at_100_percent.setText("100000")
    assert dialog.apply_button.isEnabled()
    qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    assert "winding_wire_diameter_mm" not in page.session.part(COIL)
    assert page.session.part(COIL)["winding_interval_turn_counts"] == [12]
    assert page.session.part(OWNER)[CURRENT_FIELD] == 100000.
    page.undo()
    assert page.session.document == before


def test_action_is_hidden_for_non_winding_and_blocked_by_invalid_editor_input(page):
    page.select_part(OWNER)
    assert page.configure_winding_button.isHidden()
    assert page.configure_winding() is None
    page.select_part(COIL)
    page._invalid_inputs = {("parts", COIL, "length_mm"): "invalid"}
    page._update_buttons()
    assert not page.configure_winding_button.isEnabled()
    assert page.configure_winding() is None
