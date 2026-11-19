"""Closed-coil tails bind current inputs across specimen cache boundaries."""
from copy import deepcopy
from dataclasses import replace

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.calculation_cache import calculation_signatures,incident_field_dependencies,matching_products
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.immutable_json import thaw_json
from temsim.optics.column import default_state
from temsim.simulation_modes import switch_mode
from temsim.transverse_current import BIAS_FIELD,MATRIX_FIELD,REFERENCE_ENERGY_FIELD
from temsim.transverse_winding import AREA_FIELD,INSULATION_FIELD,TURN_FIELD


@pytest.fixture(autouse=True)
def identities_never_solve_or_sample_fields(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Cache identity must not emit, transport, capture or sample a field')
    monkeypatch.setattr('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden)
    monkeypatch.setattr('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden)
    monkeypatch.setattr('temsim.physics.transverse_coil_field.capture_transverse_coil_fields',forbidden)
    monkeypatch.setattr('temsim.physics.transverse_coil_field.TransverseCoilPotential.sample_jet',forbidden)


@pytest.fixture
def state():
    catalog=AssemblyCatalog()
    state=default_state()
    catalog.apply(state,catalog.default_selection())
    switch_mode(state,'unified')
    apply_physical_layout_to_state(state)
    state.round_lens_field_model='copper_coil'
    return state


def edit_host(state,key,field):
    rows=[]
    for part in state._resolved_assembly.parts:
        if part.key!=key:
            rows.append(part)
            continue
        data=thaw_json(part.data)
        if field==MATRIX_FIELD:
            data[field][0][0]+=.001
        elif field==BIAS_FIELD:
            data[field][0]+=.001
        elif field==TURN_FIELD:
            data[field]+=1
        else:
            data[field]*=1.001
        rows.append(replace(part,data=data))
    state._resolved_assembly=replace(state._resolved_assembly,parts=tuple(rows))


def test_all_calibrated_hosts_are_retained_including_downstream_and_gun(state):
    dependency=incident_field_dependencies(state)['all_z_transverse_sources']
    hosts={row['host_key']:row for row in dependency['rows']}
    assert {'feg_deflector','feg_stigmator','probe_dph2_coils','diffraction_stigmator'}<=hosts.keys()
    assert hosts['diffraction_stigmator']['assembled_center_z_mm']>state.sample.z_mm
    assert dependency['implementation'] and dependency['source_schema']
    shared=hosts['image_diffraction_deflector']['shared_logical_drives']
    assert 'descan_deflector' in shared
    assert 'probe_dp12_scan_deflector' in hosts['ac_deflector']['shared_logical_drives']


@pytest.mark.parametrize('host',['feg_deflector','probe_dph2_coils','diffraction_stigmator'])
@pytest.mark.parametrize('field',[TURN_FIELD,MATRIX_FIELD,BIAS_FIELD,AREA_FIELD,INSULATION_FIELD])
def test_all_z_source_edits_invalidate_incident_and_specimen_products(state,host,field):
    before=calculation_signatures(state)
    edit_host(state,host,field)
    after=calculation_signatures(state)
    reused=matching_products(before,after)
    assert not {'incident','elastic','eds','sample_region'}&reused


def test_live_shared_drive_changes_all_z_dependency_without_field_query(state):
    before=incident_field_dependencies(state)['all_z_transverse_sources']
    descan=next(component for component in state.corrector_elements if component.key=='descan_deflector')
    descan.kick_x_mrad+=.001
    after=incident_field_dependencies(state)['all_z_transverse_sources']
    assert before!=after


def test_dp12_service_bias_and_installation_gate_remain_ac_host_dependencies(state):
    before=incident_field_dependencies(state)['all_z_transverse_sources']
    channel=next(component for component in state.deflectors if component.key=='probe_dp12_scan_deflector')
    channel.alignment_x_mrad+=.001
    after=incident_field_dependencies(state)['all_z_transverse_sources']
    assert before!=after
    state.probe_corrector_installed=not state.probe_corrector_installed
    assert incident_field_dependencies(state)['all_z_transverse_sources']!=after


def test_reference_energy_pose_and_time_remain_source_dependencies(state):
    before=incident_field_dependencies(state)['all_z_transverse_sources']
    state.simulation_time_s=float(getattr(state,'simulation_time_s',0.))+.001
    assert incident_field_dependencies(state)['all_z_transverse_sources']!=before
    before=incident_field_dependencies(state)['all_z_transverse_sources']
    edit_host(state,'diffraction_stigmator',REFERENCE_ENERGY_FIELD)
    assert incident_field_dependencies(state)['all_z_transverse_sources']!=before
    before=calculation_signatures(state)
    parts=tuple(replace(part,data={**part.data,'offset_x_mm':.001})
        if part.key=='diffraction_stigmator' else part for part in state._resolved_assembly.parts)
    state._resolved_assembly=replace(state._resolved_assembly,parts=parts)
    assert calculation_signatures(state)['incident']!=before['incident']


def test_source_model_version_invalidates_its_inventory(state,monkeypatch):
    before=incident_field_dependencies(state)['all_z_transverse_sources']
    from temsim.physics import transverse_coil_field
    monkeypatch.setattr(transverse_coil_field,'SCHEMA',transverse_coil_field.SCHEMA+'-changed')
    assert incident_field_dependencies(state)['all_z_transverse_sources']!=before


def test_gaussian_sources_preserve_native_dependency_contract(state):
    state.round_lens_field_model='gaussian'
    dependency=incident_field_dependencies(state)
    assert 'all_z_transverse_sources' not in dependency
    assert set(dependency)=={'schema','plane_z_mm','rows','coupled_recipes'}
