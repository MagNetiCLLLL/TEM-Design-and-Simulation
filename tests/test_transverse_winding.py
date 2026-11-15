"""Fine-wire authoring, complete circuits and fixed-envelope CAD checks."""
from collections import Counter
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pytest

from temsim.cad_catalog import is_reconstruction_region, native_reconstruction, part_solids
from temsim.cad_mesh import _manifold, meshes_from_solids
from temsim.cad_multipoles import _stigmator_windings, deflector_insert_winding_solids
from temsim.cad_transverse_winding import transverse_bulk_shape, transverse_winding_solids
from temsim.module_manifest import read_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.transverse_winding import (
    AREA_FIELD,FIELDS,INSULATION_FIELD,NOTE_FIELD,STATUS_FIELD,TURN_FIELD,SOURCE_NOTE,
    estimate_transverse_winding,supported_transverse_winding,transverse_winding_definition,
    transverse_winding_envelopes,transverse_winding_owner_key,
)


@pytest.fixture(scope='module')
def hosts():
    result={}
    for name in ('column/C3_ProbeCorrector.toml','gun/FEG.toml',
                 'project_and_recording_system/NoEnergyFilter.toml'):
        result.update({p['key']:p for p in read_document(INSTRUMENT_CONFIG_ROOT/name)['parts']})
    return result


def detailed(part,n=2,area=.02):
    return {**part,TURN_FIELD:n,AREA_FIELD:area,INSULATION_FIELD:.005,
            STATUS_FIELD:'engineering_reconstruction_not_oem',NOTE_FIELD:SOURCE_NOTE}


@pytest.mark.parametrize('key',['probe_dph2_coils','probe_dp22_coils','condenser_deflector',
    'condenser_stigmator','objective_stigmator','diffraction_stigmator',
    'feg_deflector','feg_stigmator'])
def test_supported_material_hosts_have_closed_circuits_and_safe_finewire_initialization(hosts,key):
    p=hosts[key];before=deepcopy(p)
    assert supported_transverse_winding(p)
    coils=transverse_winding_envelopes(p)
    estimate=estimate_transverse_winding(p,.02 if 'stigmator' in key else .01)
    current={**p,**estimate['fields']};d=transverse_winding_definition(current)
    assert estimate['wire_diameter_mm']==pytest.approx(2*math.sqrt(estimate['wire_cross_section_mm2']/math.pi))
    assert estimate['actual_metal_fill_fraction']<=.60+1e-14
    assert d['coil_count']==len(coils) and d['total_turn_count']==len(coils)*d['turn_count_per_coil']
    assert set(c['group'] for c in coils)=={'x','y'}
    for group in ('x','y'):
        values=[c for c in coils if c['group']==group]
        assert sum(c['current_sign'] for c in values)==0
    assert not estimate['automatic_geometry_binding'] and p==before
    for field in ('local_start_z_mm','local_center_z_mm','local_end_z_mm'):
        assert current[field]==p[field]


@pytest.mark.parametrize('key',['probe_qpol_quadrupole','probe_qph2_quadrupole','probe_hpol_hexapole'])
def test_nonmaterial_channels_do_not_acquire_independent_windings(hosts,key):
    p=hosts[key]
    assert not supported_transverse_winding(p) and transverse_winding_envelopes(p)==()
    with pytest.raises(ValueError,match='material'):
        transverse_winding_definition(detailed(p))


def test_known_N_unknown_size_retains_bulk_and_unchanged_provenance(hosts):
    p=detailed(hosts['probe_dp22_coils']);p.pop(AREA_FIELD)
    d=transverse_winding_definition(p)
    assert d['turn_count_per_coil']==2 and not d['detailed_cad']
    assert d['wire_diameter_mm'] is None and transverse_winding_solids(p) is None
    p[TURN_FIELD]=1000000
    assert not transverse_winding_definition(p)['detailed_cad']


@pytest.mark.parametrize('changes',[
    {TURN_FIELD:True},{TURN_FIELD:0},{TURN_FIELD:1.5},{TURN_FIELD:100000},
    {AREA_FIELD:0},{AREA_FIELD:float('nan')},{AREA_FIELD:True},{AREA_FIELD:100.},
    {INSULATION_FIELD:-1},{INSULATION_FIELD:float('inf')},{STATUS_FIELD:'measured'},
    {NOTE_FIELD:''},{'transverse_winding_unknown':4},
])
def test_invalid_inputs_reject_without_mutating_existing_state(hosts,changes):
    p=detailed(hosts['probe_dph2_coils']);p.update(changes);before=deepcopy(p)
    with pytest.raises(ValueError):transverse_winding_definition(p)
    assert p==before


def test_area_without_N_fails_and_replaced_geometry_remains_inactive(hosts):
    p=detailed(hosts['probe_dph2_coils']);p.pop(TURN_FIELD)
    with pytest.raises(ValueError):transverse_winding_definition(p)
    p['model_3d']={'base':{'kind':'box'}}
    assert transverse_winding_definition(p) is None


@pytest.mark.parametrize('key',['probe_dph2_coils','probe_dp22_coils','condenser_stigmator'])
def test_two_turn_wires_are_closed_inside_original_bulk_and_partition_all_material(hosts,key):
    p=detailed(hosts[key]);d=transverse_winding_definition(p)
    specs=transverse_winding_solids(p);wires=[s for s in specs if s.material=='copper_winding']
    assert len(wires)==2*len(d['coils'])
    first=[s for s in specs if s.parameters['coil_index']==1]
    wire_bodies=[_manifold(s.shape,16) for s in first if s.material=='copper_winding']
    assert all(len(body.decompose())==1 and body.volume()>0 for body in wire_bodies)
    bulk=_manifold(transverse_bulk_shape(d['coils'][0]),16)
    for body in wire_bodies:
        assert (body-bulk).volume()<1e-9
    assert (wire_bodies[0]^wire_bodies[1]).volume()<1e-12
    remaining=_manifold(next(s.shape for s in first if s.material=='insulator'),16)
    assert remaining.volume()+sum(b.volume() for b in wire_bodies)==pytest.approx(bulk.volume(),rel=2e-7,abs=1e-9)
    meshes=meshes_from_solids(first,16)
    for mesh in meshes:
        edges=Counter(tuple(sorted((int(a),int(b)))) for face in mesh.faces for a,b in zip(face,np.roll(face,-1)))
        assert all(n==2 for n in edges.values())


def test_saddle_bulk_geometry_matches_old_allocation(hosts):
    p=hosts['probe_dp22_coils']
    new=transverse_winding_envelopes(p)
    old=deflector_insert_winding_solids(p,hosts)
    for c,spec in zip(new,old):
        # Use the same converged angular grid: the new wire display explicitly
        # refines thin bands, whereas the old bulk display used a coarse grid.
        first=_manifold(transverse_bulk_shape(c),4096);second=_manifold(spec.shape,4096)
        assert first.volume()==pytest.approx(second.volume(),rel=1e-12)


@pytest.mark.parametrize('key',['probe_dp22_coils','condenser_stigmator','feg_stigmator'])
def test_native_and_export_catalogs_keep_passives_and_all_wire_regions(hosts,key):
    original=hosts[key];p=detailed(original);by_key={**hosts,key:p}
    current=part_solids(p,by_key)
    old=part_solids({k:v for k,v in original.items() if k not in FIELDS},hosts)
    passive={s.region:s for s in old if s.material!='copper_winding'}
    indexed={s.region:s for s in current}
    assert all(indexed[region]==spec for region,spec in passive.items())
    assert len([s for s in current if s.region.startswith('winding_turn_transverse_')])==2*len(transverse_winding_envelopes(p))
    assert native_reconstruction(p,by_key) is not None


def test_material_classification_is_bounded_without_rebuilding_CAD(hosts,monkeypatch):
    p=detailed(hosts['probe_dp22_coils'])
    def forbidden(*args,**kwargs):raise AssertionError('Classification must not build all wire solids')
    monkeypatch.setattr('temsim.cad_transverse_winding.transverse_winding_solids',forbidden)
    assert is_reconstruction_region(p,'winding_turn_transverse_coil_4_000002')
    assert is_reconstruction_region(p,'winding_insulation_transverse_coil_1')
    assert not is_reconstruction_region(p,'winding_turn_transverse_coil_5_000001')
    assert not is_reconstruction_region(p,'winding_turn_transverse_coil_1_000003')


def test_explicit_owner_reuses_existing_field_channel(hosts):
    assert transverse_winding_owner_key(hosts['probe_dp22_coils'])=='probe_dp22_deflector'
    assert transverse_winding_owner_key(hosts['condenser_stigmator'])=='condenser_stigmator'


def test_dense_saddle_insulation_keeps_positive_faces_and_closed_material_volume(hosts):
    # A dense second-axis saddle previously generated exactly collinear faces
    # after the many-wire Boolean. Keep its complete physical turn population.
    p=detailed(hosts['beam_deflector'],n=216,area=.01)
    specs=[s for s in transverse_winding_solids(p) if s.parameters['coil_index']==3]
    assert len(specs)==217
    meshes=meshes_from_solids(specs,16)
    for mesh in meshes:
        triangles=mesh.vertices[mesh.faces]
        assert np.all(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)>0)
    insulation=_manifold(specs[-1].shape,16)
    assert insulation.volume()>0 and str(insulation.status())=='Error.NoError'
