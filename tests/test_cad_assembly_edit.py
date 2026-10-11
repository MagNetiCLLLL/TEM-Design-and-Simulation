"""One captured assembly, not all catalog combinations or guessed placements."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest
import tomli_w

from temsim.cad_assembly_edit import build_current_assembly_edit_catalog, assembly_edit_context_is_current
from temsim.cad_edit_validation import build_edit_validation_request, finalize_edit_validation
from temsim.column.module_assembly import ModuleDefinition, ModulePart, AssemblyPart, ResolvedAssembly, _freeze
from temsim.component_operations import make_component
from temsim.part_model_document import PartModelDocument


def fixture(tmp_path):
    modules, parts, paths, captured, origins = [], [], [], {}, {}
    for key, origin, center in [('a', 0., 8.5), ('b', 10., .5)]:
        row = make_component(key='body_'+key, shape='box', center_z_mm=center,
                             length_mm=1., width_mm=2., height_mm=2.)
        geometry = dict(length_mm=10., vacuum_drift_inner_diameter_mm=20., vacuum_liner_wall_thickness_mm=.25)
        document = dict(format_version=1, coordinate_system='module_local_z_mm',
            module=dict(key=key, type='mechanical', name=key), geometry=geometry,
            ports=dict(entrance=dict(interface='beam', local_z_mm=0.), exit=dict(interface='beam', local_z_mm=10.)), parts=[row])
        path = tmp_path/(key+'.toml')
        path.write_text(tomli_w.dumps(document), encoding='utf-8')
        captured[path] = path.read_bytes()
        origins[path] = origin
        paths.append(path.name)
        p = ModulePart(row['key'], row['name'], row['branch'], center-.5, center, center+.5,
                       1., row['parent_key'], _freeze(row))
        modules.append(ModuleDefinition('mechanical', key, 'beam', 0., 'beam', 10., 10.,
                                        (p,), path.name, _freeze(geometry)))
        parts.append(AssemblyPart(key, path.name, p.key, p.name, p.branch, p.start_z_mm+origin,
            p.center_z_mm+origin, p.end_z_mm+origin, p.length_mm, p.parent_key, p.data))
    assembly = ResolvedAssembly(tuple(modules), tuple(parts), 20., tmp_path,
        tuple((p.stem, p.name) for p in captured), (), ())
    session = PartModelDocument(tmp_path/'a.toml')
    return session, dict(project_root=tmp_path, project_paths=paths, module_origins=origins,
        coordinate_source_bytes=captured, assembly=assembly, runtime_values={}, aperture_indices={})


def test_current_assembly_uses_fixed_translations_real_liners_and_rejects_changed_sources(tmp_path):
    session, context = fixture(tmp_path)
    session.update_model_3d('body_a', {'base': {'length_mm': 6.}})
    catalog = build_current_assembly_edit_catalog(session, **context)
    assert len(catalog['assemblies']) == 1
    assembly = catalog['assemblies'][0]
    assert assembly['material_liner_scope'] == 'physical_adapter'
    assert assembly['placements'][:2] == [dict(module_key='a', offset_mm=[0., 0., 0.]), dict(module_key='b', offset_mm=[0., 0., 10.])]
    liner = catalog['modules'][-1]
    assert liner['key'].startswith('@physical_liners:') and liner['solids']
    assert assembly['placements'][-1]['offset_mm'] == [0., 0., 0.]
    assert len({s['instance_key'] for s in liner['solids']}) == len(liner['solids'])
    assert catalog['edit_context']['needs_reassemble']
    current_args = {k: v for k, v in context.items() if k not in {'project_root', 'project_paths'}}
    assert assembly_edit_context_is_current(catalog, **current_args, full_source_check=True)
    (tmp_path/'b.toml').write_bytes((tmp_path/'b.toml').read_bytes()+b'\n')
    assert not assembly_edit_context_is_current(catalog, **current_args)
    with pytest.raises(ValueError, match='source changed'):
        build_current_assembly_edit_catalog(session, **context)


def test_exact_occ_finds_overlap_between_different_modules_after_draft_edit(tmp_path):
    root = Path(__file__).parents[1]
    interpreter = root/'environment/cad/Scripts/python.exe'
    if not interpreter.is_file():
        pytest.skip('Optional isolated CAD geometry interpreter is not installed')
    session, context = fixture(tmp_path)
    session.update_model_3d('body_a', {'base': {'length_mm': 6.}})
    catalog = build_current_assembly_edit_catalog(session, **context)
    request = build_edit_validation_request(session.document, catalog, request_id='cross-module')
    source = tmp_path/'request.json'
    source.write_text(json.dumps(request), encoding='utf-8')
    output = tmp_path/'response'
    process = subprocess.run([str(interpreter), str(root/'scripts/cad_geometry_worker.py'),
        str(source), '--output', str(output)], capture_output=True, text=True, timeout=45)
    assert process.returncode == 0, process.stdout+process.stderr
    result = finalize_edit_validation(request, json.loads((output/'result.json').read_text()), document=session.document)
    assert not result['can_apply']
    pairs = result['geometry_report']['assemblies'][0]['collisions']
    pair = next(p for p in pairs if {p['first']['key'], p['second']['key']} == {'body_a', 'body_b'})
    assert {pair['first']['module_key'], pair['second']['module_key']} == {'a', 'b'}
    assert pair['intersection_volume_mm3'] == pytest.approx(4.)


def test_uninstalled_source_is_explicit_module_only_without_active_coordinates(tmp_path):
    session, context = fixture(tmp_path)
    session.save_copy(tmp_path/'external.toml')
    catalog = build_current_assembly_edit_catalog(session, **context)
    assert catalog['edit_context']['scope'] == 'module_only'
    assert 'assemblies' not in catalog and len(catalog['modules']) == 1


def test_subassembly_draft_is_reembedded_once_in_owner_not_duplicated(tmp_path):
    session, context = fixture(tmp_path)
    from temsim.component_operations import translated_component
    parent = session.document
    child_part = make_component(key='insert', shape='box', center_z_mm=1., length_mm=.5, width_mm=.2, height_mm=.2)
    child = dict(format_version=1, coordinate_system='module_local_z_mm', module=dict(type='subassembly', key='insert_definition'), parts=[child_part])
    child_path = tmp_path/'child.toml'
    child_path.write_text(tomli_w.dumps(child), encoding='utf-8')
    parent['subassemblies'] = [dict(key='insert_subassembly', file='child.toml', placement=dict(mode='fixed', origin_z_mm=3.))]
    (tmp_path/'a.toml').write_text(tomli_w.dumps(parent), encoding='utf-8')
    context['coordinate_source_bytes'].update({tmp_path/'a.toml': (tmp_path/'a.toml').read_bytes(), child_path: child_path.read_bytes()})
    context['module_origins'][child_path] = 3.
    child_session = PartModelDocument(child_path)
    child_session.update_model_3d('insert', {'base': {'width_mm': .4}})
    catalog = build_current_assembly_edit_catalog(child_session, **context)
    owner = next(m for m in catalog['modules'] if m['key'] == 'a')
    inserted = [s for s in owner['solids'] if s['key'] == 'insert']
    assert len(inserted) == 1
    assert inserted[0]['shape']['offset_mm'] == [0., 0., 4.]
    assert inserted[0]['shape']['shape']['size_mm'][0] == .4
    request = build_edit_validation_request(child_session.document, catalog)
    assert request['binding_report']['module_key'] == 'a'
    assert request['draft_digest'] == catalog['edit_context']['draft_sha256']
    assert len({p['key'] for p in catalog['edit_context']['binding_document']['parts']}) == 2


def test_exact_occ_detects_other_module_in_aperture_without_material_collision(tmp_path):
    from temsim.cad_geometry import annulus, transformed
    from temsim.cad_optical_binding import canonical_digest
    root = Path(__file__).parents[1]
    interpreter = root/'environment/cad/Scripts/python.exe'
    if not interpreter.is_file():
        pytest.skip('Optional isolated CAD geometry interpreter is not installed')
    key = 'condenser_aperture_2'
    part = dict(key=key, aperture_plate_form='perforated_strip', local_center_z_mm=20.,
        optical_reference_local_z_mm=20., plate_thickness_mm=.2, aperture_hole_diameters_mm=[.1])
    document = dict(module=dict(key='owner'), parts=[part])
    owner = dict(key='owner', source_document_sha256=canonical_digest(document), solids=[dict(
        key=key, region='plate', material='metal', status='configured_geometry',
        shape=annulus(19.9, 20.1, .05, 2.))])
    # Owner module is translated by (2,3,100). The second module's plug
    # lands on the same global point via a different local origin. Its radius
    # is much smaller than the aperture, so material/material overlap is zero.
    neighbor = dict(key='neighbor', solids=[dict(key='plug', region='body', material='metal',
        status='configured_geometry', shape=transformed(annulus(9.95, 10.05, 0., .01), offset=(6., -2., 0.)))])
    catalog = dict(schema='temsim-analytic-material-cad-v1', units='mm', modules=[owner, neighbor],
        assemblies=[dict(key='installed', placements=[dict(module_key='owner', offset_mm=[2., 3., 100.]),
            dict(module_key='neighbor', offset_mm=[-4., 5., 110.])])])
    request = build_edit_validation_request(document, catalog, request_id='cross-module-passage')
    assert request['optical_probes'][0]['assembly_key'] == 'installed'
    assert request['binding_report']['optical_references'][0]['global_origin_mm'] == [2., 3., 120.]
    source = tmp_path/'probe-request.json'
    source.write_text(json.dumps(request), encoding='utf-8')
    output = tmp_path/'probe-response'
    process = subprocess.run([str(interpreter), str(root/'scripts/cad_geometry_worker.py'),
        str(source), '--output', str(output)], capture_output=True, text=True, timeout=45)
    assert process.returncode == 0, process.stdout+process.stderr
    receipt = json.loads((output/'result.json').read_text())
    result = finalize_edit_validation(request, receipt, document=document)
    assert not result['geometry_report']['assemblies'][0]['collisions']
    assert not result['can_apply']
    probe, = result['probe_results']
    assert probe['coordinate_frame'] == 'global_column_mm'
    assert probe['module_offset_mm'] == [2., 3., 100.]
    blocker, = probe['material_blockers']
    assert blocker['module_key'] == 'neighbor'
    assert blocker['solid_id'] == 'plug/body'
    assert blocker['volume_mm3'] == pytest.approx(3.141592653589793e-5, rel=1e-8)
    # A valid hash cannot make a module-only receipt count as assembly proof.
    forged = json.loads(json.dumps(receipt))
    forged['result']['probe_results'][0]['coordinate_frame'] = 'module_local_mm'
    forged['result']['report_sha256'] = canonical_digest({k:v for k,v in forged['result'].items() if k!='report_sha256'})
    with pytest.raises(ValueError, match='assembly coordinates'):
        finalize_edit_validation(request, forged, document=document)
