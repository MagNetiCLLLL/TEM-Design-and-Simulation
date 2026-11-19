"""Transverse winding/current edits are atomic, explicit, and undoable."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QLineEdit

from temsim.gui.part_model_editor import PartModelEditorPage
from temsim.gui.transverse_winding_dialog import TransverseWindingConfigurationDialog
from temsim.parameter_impact import describe_parameter_impact
from temsim.parameter_semantics import describe_parameter
from temsim.part_model_3d import part_dimension_specs
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.transverse_current import (
    BIAS_FIELD, CALIBRATION_KIND, FIELDS as CURRENT_FIELDS, KIND_FIELD, MATRIX_FIELD,
    NOTE_FIELD as CURRENT_NOTE, REFERENCE_ENERGY_FIELD, STATUS_FIELD as CURRENT_STATUS,
)
from temsim.transverse_winding import (
    AREA_FIELD, FIELDS, INSULATION_FIELD, NOTE_FIELD, STATUS_FIELD, TURN_FIELD,
    transverse_winding_definition,
)


KEY='probe_dph2_coils'
OWNER='probe_dph2_deflector'
PATH=INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml'


@pytest.fixture(autouse=True)
def no_source_or_wave(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Input/CAD transactions must not emit or transport a source')
    monkeypatch.setattr('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden)
    monkeypatch.setattr('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden)


@pytest.fixture
def document(tmp_path):
    path=tmp_path/'transverse.toml'
    PartModelDocument(PATH).save_copy(path)
    result=PartModelDocument(path)
    # A controlled source-free fixture, independent of installed calibration.
    result.configure_transverse_winding(KEY,turn_count_per_coil=None)
    result.save()
    return PartModelDocument(path)


def configure(document,turns=4,area=.01,insulation=.005,**kwargs):
    return document.configure_transverse_winding(KEY,turn_count_per_coil=turns,
        wire_cross_section_mm2=area,insulation_thickness_mm=insulation,**kwargs)


def install_electrical_fixture(document):
    configure(document)
    candidate=deepcopy(document.document)
    part=next(p for p in candidate['parts'] if p['key']==KEY)
    part.update({MATRIX_FIELD:[[.15,-.01],[.02,.18]],BIAS_FIELD:[0.,0.],
        REFERENCE_ENERGY_FIELD:300000.,KIND_FIELD:CALIBRATION_KIND,
        CURRENT_STATUS:'test_reference',CURRENT_NOTE:'Synthetic field-free electrical-input fixture; not a calibration.'})
    document._commit_component_operation(candidate)


def test_atomic_inputs_save_reload_undo_clear_and_noop(document):
    before=deepcopy(document.document)
    configure(document)
    configured=deepcopy(document.document)
    part=document.part(KEY)
    assert part[TURN_FIELD]==4 and part[AREA_FIELD]==.01
    assert part[INSULATION_FIELD]==.005
    assert part[STATUS_FIELD]=='engineering_reconstruction_not_oem'
    assert MATRIX_FIELD not in part
    assert transverse_winding_definition(part)['detailed_cad']
    history=len(document._history)
    configure(document)
    assert document.document==configured and len(document._history)==history
    document.undo();assert document.document==before
    document.redo();assert document.document==configured
    document.save()
    assert PartModelDocument(document.path).part(KEY)==part
    configure(document,None,None,None)
    document.save()
    assert not (FIELDS|CURRENT_FIELDS).intersection(PartModelDocument(document.path).part(KEY))


def test_known_turns_without_wire_size_remain_editable(document):
    configure(document,6,None,None)
    d=transverse_winding_definition(document.part(KEY))
    assert not d['detailed_cad'] and d['turn_count_per_coil']==6
    document.set_dimension(('parts',KEY,TURN_FIELD),7.)
    assert document.part(KEY)[TURN_FIELD]==7 and AREA_FIELD not in document.part(KEY)
    document.save()
    assert PartModelDocument(document.path).part(KEY)[TURN_FIELD]==7


@pytest.mark.parametrize(('turns','area','insulation'),[
    (True,.01,.005),(0,.01,.005),(1.5,.01,.005),(4,-1.,.005),
    (4,float('nan'),.005),(4,.01,-.1),(4,.01,float('inf')),
    (4,10.,.005),(None,.01,None),(None,None,.005),(1000000,.01,.005),
])
def test_invalid_or_impossible_layout_preserves_draft_and_history(document,turns,area,insulation):
    before=deepcopy(document.document)
    history=deepcopy(document._history)
    with pytest.raises((TypeError,ValueError)):
        configure(document,turns,area,insulation)
    assert document.document==before and document._history==history


def test_bias_edit_does_not_change_geometry_matrix_or_native_controls(document):
    install_electrical_fixture(document)
    before=deepcopy(document.document)
    configure(document,current_bias_a=[-.25,.4])
    part=document.part(KEY)
    assert part[BIAS_FIELD]==[-.25,.4]
    assert part[MATRIX_FIELD]==[[.15,-.01],[.02,.18]]
    assert part[CURRENT_STATUS]=='manual_current_edit'
    assert document.part(OWNER)==next(p for p in before['parts'] if p['key']==OWNER)
    document.save()
    restored=PartModelDocument(document.path)
    assert restored.part(KEY)[BIAS_FIELD]==[-.25,.4]
    restored.configure_transverse_winding(KEY,turn_count_per_coil=None)
    restored.save()
    assert not CURRENT_FIELDS.intersection(PartModelDocument(restored.path).part(KEY))
    document.undo();assert document.document==before


def test_unchanged_optional_zero_bias_preserves_absence_and_calibration(document):
    install_electrical_fixture(document)
    candidate=deepcopy(document.document)
    part=next(p for p in candidate['parts'] if p['key']==KEY)
    part.pop(BIAS_FIELD)
    document._commit_component_operation(candidate)
    before=deepcopy(document.document)
    history=len(document._history)
    configure(document,current_bias_a=[0.,0.])
    assert document.document==before and len(document._history)==history
    assert BIAS_FIELD not in document.part(KEY)


def test_turn_edit_marks_old_calibration_stale_without_refitting(document):
    install_electrical_fixture(document)
    before=deepcopy(document.part(KEY))
    document.set_dimension(('parts',KEY,TURN_FIELD),5.)
    after=document.part(KEY)
    assert after[CURRENT_STATUS]=='stale_geometry_or_turn_edit'
    assert after[MATRIX_FIELD]==before[MATRIX_FIELD] and after[BIAS_FIELD]==before[BIAS_FIELD]
    assert after[AREA_FIELD]==before[AREA_FIELD]
    assert 'no automatic current fit' in after[CURRENT_NOTE]


def test_geometry_edit_marks_old_fit_stale_but_area_only_does_not(document):
    install_electrical_fixture(document)
    matrix=deepcopy(document.part(KEY)[MATRIX_FIELD])
    document.set_dimension(('parts',KEY,AREA_FIELD),.012)
    assert document.part(KEY)[CURRENT_STATUS]=='test_reference'
    document.set_dimension(('parts',KEY,'winding_band_width_mm'),1.1)
    assert document.part(KEY)[CURRENT_STATUS]=='stale_geometry_or_turn_edit'
    assert document.part(KEY)[MATRIX_FIELD]==matrix


@pytest.mark.parametrize('nested',[False,True])
def test_separate_owner_reference_edits_stale_material_fit_but_commands_do_not(document,nested):
    from temsim.part_model_document import _invalidate_transverse_geometry_calibrations

    install_electrical_fixture(document)
    before=deepcopy(document.document)
    owner=next(p for p in before['parts'] if p['key']==OWNER)
    reference=owner.setdefault('model',{}).setdefault('params',{}) if nested else owner
    reference['effective_thickness_mm']=4.
    candidate=deepcopy(before)
    changed_owner=next(p for p in candidate['parts'] if p['key']==OWNER)
    changed_owner['kick_x_mrad']=.2
    changed_owner.setdefault('model',{}).setdefault('params',{})['kick_y_mrad']=-.1
    _invalidate_transverse_geometry_calibrations(before,candidate)
    host=next(p for p in candidate['parts'] if p['key']==KEY)
    assert host[CURRENT_STATUS]=='test_reference'
    changed_reference=changed_owner['model']['params'] if nested else changed_owner
    changed_reference['effective_thickness_mm']=3.5
    _invalidate_transverse_geometry_calibrations(before,candidate)
    assert host[CURRENT_STATUS]=='stale_geometry_or_turn_edit'
    original_host=next(p for p in before['parts'] if p['key']==KEY)
    assert host[MATRIX_FIELD]==original_host[MATRIX_FIELD]
    assert host[BIAS_FIELD]==original_host[BIAS_FIELD]


@pytest.mark.parametrize('bias',[[0.],[0.,float('nan')],[True,0.],'0, 0'])
def test_bad_bias_keeps_preceding_draft(document,bias):
    install_electrical_fixture(document)
    before=deepcopy(document.document)
    with pytest.raises((ValueError,TypeError)):
        configure(document,current_bias_a=bias)
    assert document.document==before


def test_no_current_is_guessed_for_unconfigured_host(document):
    before=deepcopy(document.document)
    with pytest.raises(ValueError,match='saved calibrated'):
        configure(document,current_bias_a=[0.,0.])
    assert document.document==before
    with pytest.raises(ValueError,match='Clearing'):
        configure(document,None,None,None,current_bias_a=[0.,0.])
    assert document.document==before


def test_nonmaterial_alias_and_replaced_cad_cannot_be_configured(document):
    with pytest.raises(ValueError,match='native physical'):
        document.configure_transverse_winding(OWNER,turn_count_per_coil=4,
            wire_cross_section_mm2=.01,insulation_thickness_mm=.005)
    candidate=deepcopy(document.part(KEY))
    candidate['model_3d']={'base':{'kind':'box'}}
    document._commit_part(KEY,candidate)
    with pytest.raises(ValueError,match='native physical'):
        configure(document)


def test_dimensions_and_explanations_distinguish_cad_and_current(document):
    install_electrical_fixture(document)
    part=document.part(KEY)
    specs=part_dimension_specs(document.document,KEY)
    by_field={spec.path[2]:spec for spec in specs}
    assert by_field[TURN_FIELD].editable and by_field[TURN_FIELD].unit=='turn'
    assert by_field[AREA_FIELD].editable and by_field[AREA_FIELD].unit=='mm²'
    assert by_field[BIAS_FIELD].editable and by_field[BIAS_FIELD].unit=='A'
    assert not by_field[MATRIX_FIELD].editable
    assert not by_field['transverse_winding_wire_diameter_mm'].editable
    meaning=describe_parameter(part,('parts',KEY,BIAS_FIELD,0))
    assert 'percent' in meaning.description and 'amperes' in meaning.description
    ideal=dict(round_lens_field_model='copper_coil')
    impact=describe_parameter_impact(part,('parts',KEY,BIAS_FIELD,0),
        simulation_mode='unified',round_lens_settings=ideal)
    assert impact.active and 'magnetic_field' in impact.effects
    area=describe_parameter_impact(part,('parts',KEY,AREA_FIELD),
        simulation_mode='unified',round_lens_settings=ideal)
    assert area.active is False and 'magnetic_field' not in area.effects
    native=describe_parameter_impact(part,('parts',KEY,BIAS_FIELD,0),simulation_mode='unified')
    assert native.active is False
    with pytest.raises(ValueError,match='read-only'):
        document.set_dimension(('parts',KEY,MATRIX_FIELD,0,0),1.)


def test_dialog_preserves_unknown_current_and_derived_diameter(document,qtbot):
    part=deepcopy(document.part(KEY))
    dialog=TransverseWindingConfigurationDialog(part,document.part(OWNER),
        {p['key']:p for p in document.document['parts']})
    qtbot.addWidget(dialog)
    assert not any(edit.isEnabled() for edit in dialog.current_bias)
    assert 'undefined' in dialog.reference.toPlainText()
    dialog.turn_count.setText('4')
    assert dialog.apply_button.isEnabled()
    dialog.wire_cross_section.setText('.01')
    dialog.insulation_thickness.setText('.005')
    assert dialog.values()==dict(turn_count_per_coil=4,wire_cross_section_mm2=.01,insulation_thickness_mm=.005)
    assert isinstance(dialog.wire_diameter,QLabel)
    assert 'calculated' in dialog.wire_diameter.text()
    assert part==document.part(KEY)
    dialog.wire_cross_section.setText('10')
    assert not dialog.apply_button.isEnabled()


def test_dialog_signed_bias_and_command_readback(document,qtbot):
    install_electrical_fixture(document)
    dialog=TransverseWindingConfigurationDialog(document.part(KEY),document.part(OWNER),
        {p['key']:p for p in document.document['parts']},runtime_values=dict(kick_x_mrad=1.,kick_y_mrad=2.))
    qtbot.addWidget(dialog)
    assert dialog.reference.isReadOnly()
    assert 'A/mrad' in dialog.reference.toPlainText()
    assert '0.13, 0.38' in dialog.reference.toPlainText()
    dialog.current_bias[0].setText('-.25')
    assert dialog.values()['current_bias_a']==[-.25,0.]
    assert '-0.12, 0.38' in dialog.reference.toPlainText()
    dialog.current_bias[1].setText('nan')
    assert not dialog.apply_button.isEnabled()


@pytest.mark.parametrize(('key','count','unit','runtime'),[
    ('condenser_stigmator',2,'A/percent',dict(strength_x_percent=50.,strength_y_percent=-25.)),
    ('beam_deflector',4,'A/mrad',dict(upper_x_mrad=1.,upper_y_mrad=2.,lower_x_mrad=3.,lower_y_mrad=4.)),
])
def test_stigmator_and_multistation_dialog_keep_full_matrix_channel_order(document,qtbot,key,count,unit,runtime):
    part=deepcopy(document.part(key))
    for field in FIELDS|CURRENT_FIELDS:
        part.pop(field,None)
    part.update({TURN_FIELD:4,AREA_FIELD:.01,INSULATION_FIELD:.005,
        STATUS_FIELD:'engineering_reconstruction_not_oem',NOTE_FIELD:'Test-only physical winding inputs; no real calibration.',
        MATRIX_FIELD:[[.01*(1+i+j) for j in range(count)] for i in range(count)],
        BIAS_FIELD:[0.]*count,REFERENCE_ENERGY_FIELD:300000.,KIND_FIELD:CALIBRATION_KIND,
        CURRENT_STATUS:'test_reference',CURRENT_NOTE:'Synthetic full-matrix GUI fixture, not a field calibration.'})
    dialog=TransverseWindingConfigurationDialog(part,part,{key:part},runtime_values=runtime)
    qtbot.addWidget(dialog)
    assert dialog.channel_count==count and len(dialog.current_bias)==count
    assert unit in dialog.reference.toPlainText()
    if count==4:
        assert dialog.channel_labels==['Station 1 X','Station 1 Y','Station 2 X','Station 2 Y']
        assert '0.3, 0.4, 0.5, 0.6' in dialog.reference.toPlainText()
    else:
        assert dialog.channel_labels==['Group X','Group Y']
        assert '0, 0.25' in dialog.reference.toPlainText()
    assert dialog.values()['turn_count_per_coil']==4


@pytest.mark.parametrize(('key','count','unit','runtime','expected'),[
    ('feg_deflector',4,'A/mT',dict(upper_field_x_mt=1.,upper_field_y_mt=2.,lower_field_x_mt=3.,lower_field_y_mt=4.),
        '0.3, 0.4, 0.5, 0.6'),
    ('feg_stigmator',2,'A/(T/m)',dict(gradient_t_per_m=2.,rotation_deg=45.),'0.04, 0.06'),
])
def test_gun_dialog_preserves_magnetic_field_command_units(qtbot,key,count,unit,runtime,expected):
    gun=PartModelDocument(INSTRUMENT_CONFIG_ROOT/'gun/FEG.toml')
    part=deepcopy(gun.part(key))
    part.update({TURN_FIELD:4,AREA_FIELD:.01,INSULATION_FIELD:.005,
        STATUS_FIELD:'engineering_reconstruction_not_oem',NOTE_FIELD:'Explicit test-only winding.',
        MATRIX_FIELD:[[.01*(1+i+j) for j in range(count)] for i in range(count)],
        BIAS_FIELD:[0.]*count,REFERENCE_ENERGY_FIELD:300000.,KIND_FIELD:CALIBRATION_KIND,
        CURRENT_STATUS:'test_reference',CURRENT_NOTE:'Synthetic current-input example; no field fit.'})
    dialog=TransverseWindingConfigurationDialog(part,part,{key:part},runtime_values=runtime)
    qtbot.addWidget(dialog)
    assert dialog.apply_button.isEnabled()
    assert unit in dialog.reference.toPlainText()
    assert expected in dialog.reference.toPlainText()
    if key.endswith('stigmator'):
        assert dialog.channel_labels==['Group X','Group Y']
        assert dialog.command_labels==['Normal gradient','Skew gradient']


def test_saved_electrical_recipe_is_inactive_during_cad_replacement(document):
    install_electrical_fixture(document)
    current=deepcopy(document.part(KEY)[MATRIX_FIELD])
    part=deepcopy(document.part(KEY))
    part['model_3d']={'base':{'kind':'box'}}
    document._commit_part(KEY,part)
    assert document.part(KEY)[MATRIX_FIELD]==current
    restored=deepcopy(document.part(KEY))
    restored['model_3d']={'base':{'kind':'existing'}}
    document._commit_part(KEY,restored)
    assert document.part(KEY)[MATRIX_FIELD]==current
    assert document.part(KEY)[CURRENT_STATUS]=='stale_geometry_or_turn_edit'


@pytest.fixture
def page(document,qtbot,monkeypatch):
    monkeypatch.setattr(PartModelEditorPage,'_render',lambda *args,**kwargs:True)
    monkeypatch.setattr(PartModelEditorPage,'_queue_overlap_check',lambda *args,**kwargs:None)
    result=PartModelEditorPage()
    qtbot.addWidget(result)
    result.set_project_context(document.path.parent,
        SimpleNamespace(selected_module_paths=(('column',document.path.name),)),None)
    assert result.open_path(document.path,selected_key=KEY)
    return result


def test_actual_editor_routes_physical_insert_and_stages_once(page,qtbot):
    context=page._winding_context()
    assert context[0]['key']==KEY and context[1]['key']==OWNER
    original=page.session.path.read_bytes()
    before=deepcopy(page.session.document)
    dialog=page.configure_winding()
    assert isinstance(dialog,TransverseWindingConfigurationDialog)
    dialog.turn_count.setText('4')
    dialog.wire_cross_section.setText('.01')
    dialog.insulation_thickness.setText('.005')
    qtbot.mouseClick(dialog.apply_button,Qt.MouseButton.LeftButton)
    assert page.session.part(KEY)[TURN_FIELD]==4
    assert page.session.path.read_bytes()==original
    page.session.undo();assert page.session.document==before
    page.select_part(OWNER,emit=False)
    assert page._winding_context() is None


def test_stale_transverse_dialog_rejects_changed_draft(page,qtbot):
    dialog=page.configure_winding()
    dialog.turn_count.setText('4')
    dialog.wire_cross_section.setText('.01')
    dialog.insulation_thickness.setText('.005')
    page.session.set_dimension(('parts',KEY,'winding_band_width_mm'),1.1)
    before=deepcopy(page.session.document)
    qtbot.mouseClick(dialog.apply_button,Qt.MouseButton.LeftButton)
    assert page.session.document==before
    assert 'draft changed' in dialog.validation.text()
