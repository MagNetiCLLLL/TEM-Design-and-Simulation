"""Every multipole current tail is a source/cache dependency, without a solve."""
from dataclasses import replace

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.calculation_cache import incident_field_dependencies,calculation_signatures,matching_products
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.immutable_json import thaw_json
from temsim.optics.column import default_state
from temsim.simulation_modes import switch_mode
from temsim.multipole_current import (
    MATRIX_FIELD,BIAS_FIELD,REFERENCE_ENERGY_FIELD,REFERENCE_ORIENTATION_FIELD,
    KIND_FIELD,CALIBRATION_KIND,STATUS_FIELD as CURRENT_STATUS,NOTE_FIELD as CURRENT_NOTE,
)
from temsim.multipole_winding import (
    INSULATION_FIELD,BASIS_FIELD,STATUS_FIELD,NOTE_FIELD,turn_field,area_field,
)


TARGETS=('probe_hp1_hexapole','probe_qpol_quadrupole')


@pytest.fixture(autouse=True)
def identities_never_solve_or_sample(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Cache identity must not emit, transport, capture or sample a field')
    monkeypatch.setattr('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden)
    monkeypatch.setattr('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden)
    monkeypatch.setattr('temsim.physics.transverse_coil_field.TransverseCoilPotential.sample_jet',forbidden)
    monkeypatch.setattr('temsim.physics.multipole_coil_field.capture_multipole_coil_fields',forbidden)


@pytest.fixture(scope='module')
def base():
    state=default_state();catalog=AssemblyCatalog()
    catalog.apply(state,catalog.default_selection());switch_mode(state,'unified')
    apply_physical_layout_to_state(state)
    state.round_lens_field_model='copper_coil'
    rows=[]
    for record in state._resolved_assembly.parts:
        if record.key not in TARGETS:
            rows.append(record);continue
        part=thaw_json(record.data)
        count=1 if record.key==TARGETS[0] else 2
        part.update({turn_field(part):2,area_field(part):.01,INSULATION_FIELD:.005,BASIS_FIELD:30. if count==1 else 0.,
            STATUS_FIELD:'engineering_reconstruction_not_oem',NOTE_FIELD:'Synthetic cache input; not OEM data.',
            MATRIX_FIELD:[[1e-8]] if count==1 else [[1e-6,0.],[0.,1e-6]],BIAS_FIELD:[0.]*count,
            REFERENCE_ENERGY_FIELD:300000.,REFERENCE_ORIENTATION_FIELD:0.,KIND_FIELD:CALIBRATION_KIND,
            CURRENT_STATUS:'test_reference',CURRENT_NOTE:'Synthetic cache input, no field fit.'})
        rows.append(replace(record,data=part))
    state._resolved_assembly=replace(state._resolved_assembly,parts=tuple(rows))
    return state


@pytest.fixture
def state(base):
    assembly=base._resolved_assembly
    model=base.round_lens_field_model
    installed=base.probe_corrector_installed
    components={c.key:c for name in ('stigmators','corrector_elements','deflectors') for c in getattr(base,name,())}
    attributes={key:dict(vars(value)) for key,value in components.items()}
    yield base
    base._resolved_assembly=assembly;base.round_lens_field_model=model;base.probe_corrector_installed=installed
    for key,component in components.items():
        vars(component).clear();vars(component).update(attributes[key])


def edit(state,key,field):
    rows=[]
    for record in state._resolved_assembly.parts:
        if record.key!=key:rows.append(record);continue
        part=thaw_json(record.data)
        if field=='turns':part[turn_field(part)]+=1
        elif field=='area':part[area_field(part)]*=1.001
        elif field==MATRIX_FIELD:part[field][0][0]*=1.001
        elif field==BIAS_FIELD:part[field][0]+=.001
        else:part[field]+= .001
        rows.append(replace(record,data=part))
    state._resolved_assembly=replace(state._resolved_assembly,parts=tuple(rows))


def test_inventory_includes_fixed_physical_inputs_and_both_source_classes(state):
    dependency=incident_field_dependencies(state)['all_z_multipole_sources']
    hosts={row['host_key']:row for row in dependency['rows']}
    assert set(TARGETS)<=hosts.keys()
    assert dependency['implementation'] and dependency['source_schema']
    assert len(hosts[TARGETS[0]]['closed_circuits'])==6
    assert len(hosts[TARGETS[1]]['closed_circuits'])==8
    assert 'hexapole_winding_turn_count_per_coil' in hosts[TARGETS[0]]['source_inputs']


@pytest.mark.parametrize('key',TARGETS)
@pytest.mark.parametrize('field',['turns','area',INSULATION_FIELD,BASIS_FIELD,MATRIX_FIELD,BIAS_FIELD,REFERENCE_ENERGY_FIELD])
def test_all_source_edits_invalidate_incident_and_specimen_results(state,key,field):
    before=calculation_signatures(state)
    edit(state,key,field)
    assert not {'incident','elastic','eds','sample_region'}&matching_products(before,calculation_signatures(state))


def test_live_native_coefficients_installation_and_pose_remain_dependencies(state):
    previous=incident_field_dependencies(state)['all_z_multipole_sources']
    component=next(c for c in state.corrector_elements if c.key==TARGETS[1])
    component.strength_m2+=.1
    assert incident_field_dependencies(state)['all_z_multipole_sources']!=previous
    previous=incident_field_dependencies(state)['all_z_multipole_sources']
    state.probe_corrector_installed=not state.probe_corrector_installed
    assert incident_field_dependencies(state)['all_z_multipole_sources']!=previous
    previous=incident_field_dependencies(state)['all_z_multipole_sources']
    rows=tuple(replace(p,data={**p.data,'offset_x_mm':.001}) if p.key==TARGETS[0] else p
        for p in state._resolved_assembly.parts)
    state._resolved_assembly=replace(state._resolved_assembly,parts=rows)
    assert incident_field_dependencies(state)['all_z_multipole_sources']!=previous


def test_source_schema_changes_invalidate_without_query(state,monkeypatch):
    from temsim.physics import multipole_coil_field
    before=incident_field_dependencies(state)['all_z_multipole_sources']
    monkeypatch.setattr(multipole_coil_field,'SCHEMA',multipole_coil_field.SCHEMA+'-edited')
    assert incident_field_dependencies(state)['all_z_multipole_sources']!=before


def test_primary_native_orientation_invalidates_all_z_dependency(state):
    previous=incident_field_dependencies(state)['all_z_multipole_sources']
    hp=next(c for c in state.corrector_elements if c.key==TARGETS[0])
    hp.orientation_rad+=.01
    assert incident_field_dependencies(state)['all_z_multipole_sources']!=previous


def test_gaussian_dependency_contract_stays_native(state):
    state.round_lens_field_model='gaussian'
    data=incident_field_dependencies(state)
    assert 'all_z_multipole_sources' not in data
    assert set(data)=={'schema','plane_z_mm','rows','coupled_recipes'}
