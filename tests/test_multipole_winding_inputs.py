"""Multipole engineering inputs preserve optical controls and atomic history."""
from copy import deepcopy

import pytest
from PySide6.QtWidgets import QLabel

from temsim.gui.multipole_winding_dialog import MultipoleWindingConfigurationDialog
from temsim.gui.part_model_editor import PartModelEditorPage
from temsim.module_manifest import stage_manifest_text
from temsim.parameter_impact import describe_parameter_impact
from temsim.parameter_semantics import describe_parameter
from temsim.part_model_3d import part_dimension_specs
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.multipole_current import (
    MATRIX_FIELD,BIAS_FIELD,REFERENCE_ENERGY_FIELD,REFERENCE_ORIENTATION_FIELD,
    KIND_FIELD,CALIBRATION_KIND,FIELDS as CURRENT_FIELDS,NOTE_FIELD as CURRENT_NOTE,
    STATUS_FIELD as CURRENT_STATUS,
)
from temsim.multipole_winding import (
    FIELDS,TURN_FIELD,AREA_FIELD,INSULATION_FIELD,BASIS_FIELD,STATUS_FIELD,NOTE_FIELD,
    area_field,turn_field,multipole_winding_definition,
)


PATH=INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml'
KEY='probe_qpol_quadrupole'
HP='probe_hp1_hexapole'


@pytest.fixture(autouse=True)
def no_source_or_wave(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Input transactions must not emit or transport a source')
    monkeypatch.setattr('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden)
    monkeypatch.setattr('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden)


@pytest.fixture(scope='module')
def baseline():
    return PartModelDocument(PATH)


@pytest.fixture
def document(baseline):
    result=deepcopy(baseline)
    result.configure_multipole_winding(KEY,turn_count_per_coil=None)
    return result


def configure(document,key=KEY,turns=4,area=.01,insulation=.005,**kwargs):
    return document.configure_multipole_winding(key,turn_count_per_coil=turns,
        wire_cross_section_mm2=area,insulation_thickness_mm=insulation,**kwargs)


def electrical(document,key=KEY):
    configure(document,key,turns=2 if key==HP else 4)
    candidate=deepcopy(document.document)
    part=next(p for p in candidate['parts'] if p['key']==key)
    count=1 if key==HP else 2
    part.update({MATRIX_FIELD:[[.15]] if count==1 else [[.15,-.01],[.02,.18]],
        BIAS_FIELD:[0.]*count,REFERENCE_ENERGY_FIELD:300000.,REFERENCE_ORIENTATION_FIELD:0.,
        KIND_FIELD:CALIBRATION_KIND,CURRENT_STATUS:'test_reference',
        CURRENT_NOTE:'Synthetic input fixture, not a field-strength calibration.'})
    document._commit_component_operation(candidate)


def test_new_auxiliary_recipe_keeps_native_params_and_noop_history(document):
    original=deepcopy(document.part(KEY));history=len(document._history)
    configure(document)
    current=deepcopy(document.part(KEY))
    assert current[TURN_FIELD]==4 and current[AREA_FIELD]==.01
    assert multipole_winding_definition(current)['total_turn_count']==32
    assert {k:v for k,v in current.items() if k not in FIELDS|{'parameter_metadata'}}=={
        k:v for k,v in original.items() if k not in FIELDS|{'parameter_metadata'}}
    assert len(document._history)==history+1
    configure(document)
    assert document.part(KEY)==current and len(document._history)==history+1
    document.undo();assert document.part(KEY)==original
    document.redo();assert document.part(KEY)==current


def test_hp_uses_only_original_canonical_N_and_area(document):
    configure(document,HP,turns=2,area=.01)
    part=document.part(HP)
    assert part['hexapole_winding_turn_count_per_coil']==2
    assert part['hexapole_winding_wire_cross_section_mm2']==.01
    assert TURN_FIELD not in part and AREA_FIELD not in part
    assert multipole_winding_definition(part)['electrical_channel_count']==1


@pytest.mark.parametrize(('turns','area','insulation'),[
    (True,.01,.005),(0,.01,.005),(1.5,.01,.005),(4,-1.,.005),
    (4,float('nan'),.005),(4,.01,-.1),(4,100.,.005),(None,.01,None),
    (None,None,.005),(1000000,.01,.005),
])
def test_bad_recipe_preserves_document_and_undo(document,turns,area,insulation):
    before=deepcopy(document.document);history=deepcopy(document._history)
    with pytest.raises((ValueError,TypeError)):
        configure(document,turns=turns,area=area,insulation=insulation)
    assert document.document==before and document._history==history


def test_known_N_unknown_area_and_clear_remove_electrical_definition(document):
    electrical(document)
    configure(document,area=None,insulation=None)
    assert not multipole_winding_definition(document.part(KEY))['detailed_cad']
    configure(document,turns=None,area=None,insulation=None)
    assert not (FIELDS|CURRENT_FIELDS).intersection(document.part(KEY))


@pytest.mark.parametrize('bias',[[0.],[0.,float('nan')],[True,0.],'.1,.2'])
def test_bad_signed_bias_preserves_document(document,bias):
    electrical(document);before=deepcopy(document.document)
    with pytest.raises((ValueError,TypeError)):
        configure(document,current_bias_a=bias)
    assert document.document==before


def test_bias_changes_keep_matrix_native_commands_and_geometry(document):
    electrical(document);before=deepcopy(document.part(KEY))
    configure(document,current_bias_a=[-.2,.3])
    after=document.part(KEY)
    assert after[BIAS_FIELD]==[-.2,.3] and after[MATRIX_FIELD]==before[MATRIX_FIELD]
    assert after[CURRENT_STATUS]=='manual_current_edit'
    assert {f:after.get(f) for f in FIELDS}=={f:before.get(f) for f in FIELDS}
    for field in ('model','orientation_rad','local_center_z_mm','mechanical_clear_bore_diameter_mm'):
        assert after.get(field)==before.get(field)


def test_zero_bias_noop_preserves_absence_and_full_provenance(document):
    electrical(document)
    candidate=deepcopy(document.document)
    next(p for p in candidate['parts'] if p['key']==KEY).pop(BIAS_FIELD)
    document._commit_component_operation(candidate)
    before=deepcopy(document.document);history=len(document._history)
    configure(document,current_bias_a=[0.,0.])
    assert document.document==before and len(document._history)==history


def test_geometry_and_N_stale_fit_area_only_keeps_it(document):
    electrical(document);matrix=deepcopy(document.part(KEY)[MATRIX_FIELD])
    document.set_dimension(('parts',KEY,AREA_FIELD),.012)
    assert document.part(KEY)[CURRENT_STATUS]=='test_reference'
    document.set_dimension(('parts',KEY,TURN_FIELD),5.)
    assert document.part(KEY)[CURRENT_STATUS]=='stale_geometry_or_turn_edit'
    assert document.part(KEY)[MATRIX_FIELD]==matrix
    electrical(document)
    candidate=deepcopy(document.document)
    next(p for p in candidate['parts'] if p['key']==KEY)[BASIS_FIELD]=7.
    document._commit_component_operation(candidate)
    assert document.part(KEY)[CURRENT_STATUS]=='stale_geometry_or_turn_edit'


def test_native_strength_commands_keep_fit_but_reference_parameters_stale_it(document):
    from temsim.part_model_document import _invalidate_multipole_geometry_calibrations
    electrical(document)
    before=deepcopy(document.document);candidate=deepcopy(before)
    part=next(p for p in candidate['parts'] if p['key']==KEY)
    part['strength_m2']=.3
    part.setdefault('model',{}).setdefault('params',{})['skew_strength_m2']=-.2
    part['orientation_rad']=.3
    part['model']['params']['orientation_rad']=.2
    _invalidate_multipole_geometry_calibrations(before,candidate)
    assert part[CURRENT_STATUS]=='test_reference'
    part['model']['params']['effective_length_mm']=7.
    _invalidate_multipole_geometry_calibrations(before,candidate)
    assert part[CURRENT_STATUS]=='stale_geometry_or_turn_edit'


def test_current_recipe_is_retained_inactive_for_replaced_CAD(document):
    from temsim.multipole_current import validate_multipole_current_inputs
    electrical(document)
    part=deepcopy(document.part(KEY));matrix=deepcopy(part[MATRIX_FIELD])
    part['model_3d']={'base':{'kind':'box'}}
    document._commit_part(KEY,part)
    assert document.part(KEY)[MATRIX_FIELD]==matrix
    assert multipole_winding_definition(document.part(KEY)) is None
    assert validate_multipole_current_inputs(document.part(KEY)) is None
    restored=deepcopy(document.part(KEY));restored['model_3d']={'base':{'kind':'existing'}}
    document._commit_part(KEY,restored)
    assert document.part(KEY)[MATRIX_FIELD]==matrix
    assert document.part(KEY)[CURRENT_STATUS]=='stale_geometry_or_turn_edit'


def test_legacy_hp_API_updates_new_electrical_status_and_clears_all(document):
    electrical(document,HP)
    matrix=deepcopy(document.part(HP)[MATRIX_FIELD])
    document.configure_probe_winding(HP,turn_count_per_coil=3,wire_cross_section_mm2=.01)
    assert document.part(HP)[CURRENT_STATUS]=='stale_geometry_or_turn_edit'
    assert document.part(HP)[MATRIX_FIELD]==matrix
    document.configure_probe_winding(HP,turn_count_per_coil=None,wire_cross_section_mm2=None)
    assert not (FIELDS|CURRENT_FIELDS).intersection(document.part(HP))
    assert 'hexapole_winding_turn_count_per_coil' not in document.part(HP)


def test_numeric_dimensions_keep_reference_readonly_and_route_ampere_bias(document):
    electrical(document)
    specs=part_dimension_specs(document.document,KEY)
    derived=next(s for s in specs if s.path==('derived',KEY,'multipole_winding_wire_diameter_mm'))
    assert not derived.editable and derived.value>0
    assert all(not s.editable for s in specs if s.path[2] in {MATRIX_FIELD,REFERENCE_ENERGY_FIELD,REFERENCE_ORIENTATION_FIELD})
    with pytest.raises(ValueError,match='read-only'):
        document.set_dimension(('parts',KEY,MATRIX_FIELD,0,0),.2)
    document.set_dimension(('parts',KEY,BIAS_FIELD,1),-.4)
    assert document.part(KEY)[BIAS_FIELD]==[0.,-.4]


def test_dialog_readback_and_atomic_commit(document,qtbot):
    electrical(document)
    by_key={p['key']:p for p in document.document['parts']}
    dialog=MultipoleWindingConfigurationDialog(document.part(KEY),document.part(KEY),by_key,
        runtime_values={'strength_m2':2.,'skew_strength_m2':3.})
    qtbot.addWidget(dialog)
    assert isinstance(dialog.wire_diameter,QLabel)
    assert 'A/(m⁻²)' in dialog.reference.toPlainText()
    assert '0.27' in dialog.reference.toPlainText() and '0.58' in dialog.reference.toPlainText()
    dialog.current_bias[0].setText('-.3')
    assert dialog.apply_button.isEnabled() and dialog.values()['current_bias_a']==[-.3,0.]
    dialog.wire_cross_section.setText('100')
    assert not dialog.apply_button.isEnabled()


def test_primary_dialog_has_one_channel_and_honest_unknown_size(document,qtbot):
    electrical(document,HP)
    configure(document,HP,turns=2,area=None,insulation=None)
    part=document.part(HP);by_key={p['key']:p for p in document.document['parts']}
    dialog=MultipoleWindingConfigurationDialog(part,part,by_key,runtime_values={'strength_m3':10.})
    qtbot.addWidget(dialog)
    assert len(dialog.current_bias)==1 and 'A/(m⁻³)' in dialog.reference.toPlainText()
    assert 'unknown' in dialog.packing.toPlainText().lower()


def test_auxiliary_hexapole_readback_rotates_native_coefficients(document,qtbot):
    import numpy as np
    key='probe_hpol_hexapole'
    electrical(document,key)
    part=document.part(key);by_key={p['key']:p for p in document.document['parts']}
    dialog=MultipoleWindingConfigurationDialog(part,part,by_key,
        runtime_values={'strength_m3':2.,'skew_strength_m3':0.,'orientation_rad':np.pi/6})
    qtbot.addWidget(dialog)
    line=next(line for line in dialog.reference.toPlainText().splitlines() if line.startswith('Current readback (A):'))
    np.testing.assert_allclose([float(v) for v in line.split(':',1)[1].split(',')],[-.02,.36],atol=1e-12)


def test_primary_current_readback_remains_scalar_when_native_angle_changes(document,qtbot):
    electrical(document,HP)
    part=document.part(HP);by_key={p['key']:p for p in document.document['parts']}
    dialog=MultipoleWindingConfigurationDialog(part,part,by_key,
        runtime_values={'strength_m3':10.,'orientation_rad':.3})
    qtbot.addWidget(dialog)
    assert dialog.apply_button.isEnabled()
    assert 'Current readback (A): 1.5' in dialog.reference.toPlainText()


def test_primary_readback_labels_unrepresentable_skew_drive(document,qtbot):
    electrical(document,HP)
    part=document.part(HP);by_key={p['key']:p for p in document.document['parts']}
    dialog=MultipoleWindingConfigurationDialog(part,part,by_key,
        runtime_values={'strength_m3':10.,'skew_strength_m3':1.})
    qtbot.addWidget(dialog)
    assert 'no second physical winding basis' in dialog.reference.toPlainText()
    assert 'Current readback (A):' not in dialog.reference.toPlainText()


def test_scalar_quadrupole_readback_uses_zero_native_skew_and_saved_bias(document,qtbot):
    electrical(document)
    part=deepcopy(document.part(KEY));part[BIAS_FIELD]=[.02,-.03]
    dialog=MultipoleWindingConfigurationDialog(part,part,{KEY:part},runtime_values={'strength_m2':2.})
    qtbot.addWidget(dialog)
    assert 'zero skew command' in dialog.reference.toPlainText()
    line=next(line for line in dialog.reference.toPlainText().splitlines() if line.startswith('Current readback (A):'))
    assert [float(v) for v in line.split(':',1)[1].split(',')]==[.32,.01]


def test_primary_native_orientation_refreshes_module_and_assembly_mesh(document,qtbot,monkeypatch):
    configure(document,HP,turns=2)
    page=PartModelEditorPage();qtbot.addWidget(page)
    page.session=document;page._selected_key=HP
    values={HP:{'orientation_rad':0.,'strength_m3':10.}}
    page._runtime_values=values
    monkeypatch.setattr(page,'_model_runtime_values',lambda:values)
    before=page._runtime_mesh_dependencies()
    whole=page._runtime_mesh_dependencies(whole_assembly=True)
    values[HP]['orientation_rad']=.2
    assert page._runtime_mesh_dependencies()!=before
    assert page._runtime_mesh_dependencies(whole_assembly=True)!=whole
    after=page._runtime_mesh_dependencies()
    values[HP]['strength_m3']=20.
    assert page._runtime_mesh_dependencies()==after


def test_editor_routes_auxiliary_control_and_rejects_stale_dialog(document,qtbot,monkeypatch):
    configure(document)
    page=PartModelEditorPage();qtbot.addWidget(page)
    page.session=document;page._selected_key=KEY
    monkeypatch.setattr(page,'_model_runtime_values',lambda:{})
    part,owner,_=page._winding_context()
    assert part['key']==KEY and owner['key']==KEY
    dialog=page.configure_winding()
    assert isinstance(dialog,MultipoleWindingConfigurationDialog)
    document.set_dimension(('parts',KEY,AREA_FIELD),.012)
    dialog._apply()
    assert 'draft changed' in dialog.validation.text().lower()
    assert dialog.isVisible()
    dialog.reject()


def test_meaning_and_impact_distinguish_CAD_from_current_source(document):
    electrical(document);part=document.part(KEY)
    meaning=describe_parameter(part,('parts',KEY,BIAS_FIELD,0))
    assert meaning.unit=='A' and 'm^-2' in meaning.description
    settings={'round_lens_field_model':'copper_coil'}
    active=describe_parameter_impact(part,('parts',KEY,BIAS_FIELD,0),simulation_mode='unified',round_lens_settings=settings)
    assert active.status=='active'
    geometry=describe_parameter_impact(part,('parts',KEY,AREA_FIELD),simulation_mode='unified',round_lens_settings=settings)
    assert geometry.status=='inactive' and 'homogenized' in geometry.detail


def test_clear_survives_text_staging_and_real_save_reload(document,tmp_path):
    electrical(document)
    target=tmp_path/'multipole.toml'
    document.save_copy(target)
    saved=PartModelDocument(target)
    saved.configure_multipole_winding(KEY,turn_count_per_coil=None)
    assert saved.updates()[('parts',KEY,MATRIX_FIELD)] is None
    saved.save()
    loaded=PartModelDocument(target)
    assert not (FIELDS|CURRENT_FIELDS).intersection(loaded.part(KEY))
    assert loaded.part('sample')==document.part('sample')
