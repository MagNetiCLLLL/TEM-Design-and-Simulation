"""Legacy CAD-only HP input compatibility and current winding-editor routing."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QLineEdit

from temsim.cad_probe_winding import AREA_FIELD, FIELDS, NOTE_FIELD, STATUS_FIELD, TURN_FIELD
from temsim.gui.probe_winding_dialog import ProbeWindingConfigurationDialog
from temsim.gui.multipole_winding_dialog import MultipoleWindingConfigurationDialog
from temsim.gui.part_model_editor import PartModelEditorPage
from temsim.parameter_impact import describe_parameter_impact
from temsim.parameter_semantics import describe_parameter
from temsim.part_model_3d import part_dimension_specs
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT

KEY = 'probe_hp2_hexapole'
PATH = INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml'


@pytest.fixture(autouse=True)
def no_physical_run(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('No particle or coherent calculation in a radial-winding UI test')
    monkeypatch.setattr('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit', forbidden)
    monkeypatch.setattr('temsim.optics.electron_gun.source.trace_source_to_exit', forbidden)


@pytest.fixture
def document(tmp_path):
    path = tmp_path/'probe.toml'
    source=PartModelDocument(PATH)
    # These cases retain the pre-electrical CAD-only contract explicitly.
    # Current defaults have separate multipole-current tests.
    from temsim.multipole_winding import FIELDS as MULTIPOLE_FIELDS
    from temsim.multipole_current import FIELDS as CURRENT_FIELDS
    candidate=deepcopy(source.document)
    part=next(p for p in candidate['parts'] if p['key']==KEY)
    metadata=deepcopy(part.get('parameter_metadata',{}))
    for field in MULTIPOLE_FIELDS|CURRENT_FIELDS|{AREA_FIELD}:
        part.pop(field,None);metadata.pop(field,None)
    part['parameter_metadata']=metadata
    source._commit_component_operation(candidate)
    source.save_copy(path)
    return PartModelDocument(path)


def configure(document, turns=4, area=.01):
    document.configure_probe_winding(KEY, turn_count_per_coil=turns, wire_cross_section_mm2=area)


def test_atomic_definition_preserves_native_inputs_and_roundtrips(document):
    before = deepcopy(document.document)
    configure(document)
    current = document.part(KEY)
    assert current[TURN_FIELD] == 4 and current[AREA_FIELD] == .01
    assert current[STATUS_FIELD] == 'engineering_reconstruction_not_oem'
    assert 'current' in current[NOTE_FIELD]
    for old in before['parts']:
        actual = document.part(old['key'])
        assert {k:v for k,v in actual.items() if k not in FIELDS | {'parameter_metadata'}} == {
            k:v for k,v in old.items() if k not in FIELDS | {'parameter_metadata'}}
    configured = deepcopy(document.document)
    history = len(document._history)
    configure(document)
    assert document.document == configured and len(document._history) == history
    document.undo(); assert document.document == before
    document.redo(); assert document.document == configured
    document.save()
    restored = PartModelDocument(document.path)
    assert restored.part(KEY)[TURN_FIELD] == 4 and restored.part(KEY)[AREA_FIELD] == .01
    document.configure_probe_winding(KEY, turn_count_per_coil=None, wire_cross_section_mm2=None)
    document.save()
    restored = PartModelDocument(document.path)
    assert not FIELDS.intersection(restored.part(KEY))


@pytest.mark.parametrize(('turns','area'),[(0,.01),(-1,.01),(True,.01),(1.5,.01),(4,0.),
    (4,float('nan')),(4,100.),(1000000,.01),(None,.01)])
def test_invalid_inputs_preserve_draft_and_undo(document, turns, area):
    before=deepcopy(document.document);history=deepcopy(document._history)
    with pytest.raises((ValueError,TypeError)):
        configure(document,turns,area)
    assert document.document == before and document._history == history


def test_dimensions_expose_per_coil_count_area_and_calculated_diameter(document):
    configure(document)
    specs={s.path[-1]:s for s in part_dimension_specs(document.document,KEY)}
    assert specs[TURN_FIELD].editable and specs[TURN_FIELD].unit == 'turn'
    assert specs[AREA_FIELD].editable and specs[AREA_FIELD].unit == 'mm²'
    diameter=specs['hexapole_winding_wire_diameter_mm']
    assert not diameter.editable and diameter.path[0] == 'derived'
    assert 'Computed from' in diameter.meaning.source_note
    before=deepcopy(document.part(KEY))
    document.set_dimension(('parts',KEY,TURN_FIELD),5.)
    document.set_dimension(('parts',KEY,AREA_FIELD),.02)
    assert document.part(KEY)[TURN_FIELD] == 5 and document.part(KEY)[AREA_FIELD] == .02
    for field in ('effective_length_mm','local_start_z_mm','local_end_z_mm','mechanical_inner_diameter_mm'):
        assert document.part(KEY)[field] == before[field]


@pytest.mark.parametrize('field', [TURN_FIELD, AREA_FIELD])
def test_input_explanation_does_not_imply_current_or_new_field(document,field):
    configure(document)
    part=document.part(KEY);path=('parts',KEY,field)
    meaning=describe_parameter(part,path)
    impact=describe_parameter_impact(part,path,simulation_mode='unified')
    assert meaning.category == 'physical' and meaning.source_kind == 'user_defined'
    assert 'radial' in meaning.description and 'OEM' in meaning.source_note
    assert impact.active is False and 'magnetic_field' not in impact.effects


def test_unknown_dialog_never_guesses_inputs_and_clear_is_explicit(document, qtbot):
    part=deepcopy(document.part(KEY))
    for field in FIELDS:
        part.pop(field,None)
    dialog=ProbeWindingConfigurationDialog(part)
    qtbot.addWidget(dialog)
    assert dialog.turn_count.text()==dialog.wire_cross_section.text()==''
    assert dialog.values()==dict(turn_count_per_coil=None,wire_cross_section_mm2=None)
    assert isinstance(dialog.wire_cross_section,QLineEdit) and isinstance(dialog.wire_diameter,QLabel)
    assert 'bulk' in dialog.packing.toPlainText()
    assert dialog.apply_button.isEnabled()
    dialog.turn_count.setText('4')
    assert dialog.apply_button.isEnabled()
    assert dialog.values()==dict(turn_count_per_coil=4,wire_cross_section_mm2=None)
    assert 'Bulk copper' in dialog.packing.toPlainText()
    dialog.wire_cross_section.setText('.01')
    assert dialog.apply_button.isEnabled()
    assert 'calculated' in dialog.wire_diameter.text()
    assert 'Total conductor turns: 24' in dialog.packing.toPlainText()
    assert dialog.values()==dict(turn_count_per_coil=4,wire_cross_section_mm2=.01)
    assert not FIELDS.intersection(part)
    dialog.wire_cross_section.setText('100')
    assert not dialog.apply_button.isEnabled()


def test_known_turns_roundtrip_without_inventing_area_or_current(document, qtbot):
    before=deepcopy(document.document)
    original=deepcopy(document.part(KEY).get('parameter_metadata',{}))
    default_turns=document.part(KEY)[TURN_FIELD]
    dialog=ProbeWindingConfigurationDialog(document.part(KEY))
    qtbot.addWidget(dialog)
    assert dialog.turn_count.text()==str(default_turns)
    assert 'initial engineering estimate' in dialog.turn_count.toolTip()
    assert dialog.wire_cross_section.text()==''
    assert dialog.values()==dict(turn_count_per_coil=default_turns,wire_cross_section_mm2=None)
    assert 'unspecified' in dialog.packing.toPlainText()
    configure(document,default_turns,None)
    assert document.document==before
    assert document.part(KEY)['parameter_metadata']==original
    document.set_dimension(('parts',KEY,TURN_FIELD),float(default_turns+1))
    assert document.part(KEY)[TURN_FIELD]==default_turns+1
    assert AREA_FIELD not in document.part(KEY)
    document.save()
    restored=PartModelDocument(document.path)
    assert restored.part(KEY)[TURN_FIELD]==default_turns+1
    assert AREA_FIELD not in restored.part(KEY)
    assert not any('current' in k for k in restored.part(KEY))
    document.undo()
    assert document.document==before


def test_metal_area_can_be_removed_while_retaining_known_turns(document):
    configure(document)
    configured=deepcopy(document.document)
    configure(document,4,None)
    assert document.part(KEY)[TURN_FIELD]==4
    assert AREA_FIELD not in document.part(KEY)
    assert AREA_FIELD not in document.part(KEY)['parameter_metadata']
    document.save()
    assert AREA_FIELD not in PartModelDocument(document.path).part(KEY)
    document.undo()
    assert document.document==configured


def test_adding_metal_area_preserves_unchanged_estimated_turn_provenance(document):
    original=deepcopy(document.part(KEY)['parameter_metadata'][TURN_FIELD])
    configure(document,document.part(KEY)[TURN_FIELD],.01)
    assert document.part(KEY)['parameter_metadata'][TURN_FIELD]==original
    assert document.part(KEY)['parameter_metadata'][AREA_FIELD]['source_kind']=='user_defined'


def test_area_without_turns_is_rejected_in_dialog(document,qtbot):
    dialog=ProbeWindingConfigurationDialog(document.part(KEY))
    qtbot.addWidget(dialog)
    dialog.turn_count.clear()
    dialog.wire_cross_section.setText('.01')
    assert not dialog.apply_button.isEnabled()
    assert 'requires' in dialog.validation.text()


@pytest.fixture
def page(document,qtbot,monkeypatch):
    monkeypatch.setattr(PartModelEditorPage,'_render',lambda *args,**kwargs:True)
    monkeypatch.setattr(PartModelEditorPage,'_queue_overlap_check',lambda *args,**kwargs:None)
    result=PartModelEditorPage();qtbot.addWidget(result)
    result.set_project_context(document.path.parent,
        SimpleNamespace(selected_module_paths=(('column',document.path.name),)),None)
    assert result.open_path(document.path,selected_key=KEY)
    return result


def test_actual_editor_routes_hp_to_area_dialog_and_stages_once(page,qtbot):
    assert page.configure_winding_button.isVisibleTo(page)
    before=deepcopy(page.session.document);original=page.session.path.read_bytes()
    dialog=page.configure_winding()
    assert isinstance(dialog,MultipoleWindingConfigurationDialog)
    dialog.turn_count.setText('4');dialog.wire_cross_section.setText('.01')
    qtbot.mouseClick(dialog.apply_button,Qt.MouseButton.LeftButton)
    assert page.session.part(KEY)[TURN_FIELD]==4
    assert page.session.path.read_bytes()==original
    page.session.undo();assert page.session.document==before


def test_stale_dialog_does_not_commit_to_changed_draft(page,qtbot):
    dialog=page.configure_winding()
    dialog.turn_count.setText('4');dialog.wire_cross_section.setText('.01')
    page.session.set_dimension(('parts',KEY,'mechanical_hexapole_tip_width_mm'),2.4)
    prior=deepcopy(page.session.document)
    qtbot.mouseClick(dialog.apply_button,Qt.MouseButton.LeftButton)
    assert page.session.document==prior
    assert 'draft changed' in dialog.validation.text()


def test_actual_dimensions_table_edits_per_coil_turns_and_metal_area(page):
    configure(page.session)
    page._load_parameters()
    def item(field):
        return next(page.dimensions.item(row,1) for row in range(page.dimensions.rowCount())
            if page.dimensions.item(row,1).data(Qt.ItemDataRole.UserRole)[-1] == field)
    assert item(TURN_FIELD).flags() & Qt.ItemFlag.ItemIsEditable
    assert item(AREA_FIELD).flags() & Qt.ItemFlag.ItemIsEditable
    assert not item('hexapole_winding_wire_diameter_mm').flags() & Qt.ItemFlag.ItemIsEditable
    item(TURN_FIELD).setText('5')
    item(AREA_FIELD).setText('0.02')
    assert page.session.part(KEY)[TURN_FIELD] == 5
    assert page.session.part(KEY)[AREA_FIELD] == .02


def test_supported_control_channel_exposes_winding_only_editor_without_new_host(page):
    page.select_part('probe_hpol_hexapole',emit=False)
    context=page._winding_context()
    assert context is not None and context[0]['key']=='probe_hpol_hexapole'
    assert context[0].get('layout_role')=='control_channel'
    assert page.configure_winding_button.isVisibleTo(page)
