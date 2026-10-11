"""Optical datums and CAD receipts must stay in the same explicit frame."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import tomllib

import numpy as np
import pytest

from temsim.cad_geometry import box, annulus
from temsim.cad_optical_binding import build_optical_binding_report, canonical_digest
from temsim.cad_edit_validation import (
    build_edit_validation_request, finalize_edit_validation, validation_is_current,
)


def aperture_fixture():
    part = dict(key='condenser_aperture_2', aperture_plate_form='perforated_strip',
        local_center_z_mm=20., optical_reference_local_z_mm=20.,
        local_start_z_mm=19., local_end_z_mm=21., plate_thickness_mm=.2,
        mechanical_outer_diameter_mm=4., aperture_hole_diameters_mm=[.1, .2])
    doc = dict(module=dict(key='fixture'), parts=[part])
    catalog = dict(key='fixture', source_document_sha256=canonical_digest(doc),
        solids=[dict(key=part['key'], region='plate', material='metal', status='configured_geometry',
                     shape=box((0., 0., 20.), (4., 4., .2)))])
    return doc, catalog


def receipt(request, *, collision=False, probe_failure=False):
    reports = [dict(key=m['key'], collisions=[{'volume_mm3': 1.}] if collision else [],
                    unresolved_solids=[]) for m in request['catalog']['modules']]
    result = dict(schema='temsim-cad-geometry-validation-v1', units='mm',
        catalog_sha256=canonical_digest(request['catalog']),
        binding_report_sha256=request['binding_report_sha256'],
        geometry_report=dict(status='completed', assemblies=reports),
        probe_results=[dict(id=p['id'], module_key=p['module_key'], kind=p['kind'],
            status='failed' if probe_failure else 'passed', probe_volume_mm3=.001,
            uncovered_volume_mm3=None, material_blockers=[dict(solid_id='a/b', volume_mm3=.01)] if probe_failure else [],
            unresolved_intersections=[]) for p in request['optical_probes']])
    result['report_sha256'] = canonical_digest(result)
    return dict(status='complete', request_sha256=canonical_digest(request),
        draft_digest=request['draft_digest'], request_id=request.get('request_id'), result=result)


def test_aperture_selected_opening_and_pose_share_module_coordinates_without_mutating_inputs():
    doc, catalog = aperture_fixture()
    doc['parts'][0].update(offset_x_mm=2., rotation_y_mrad=100.)
    catalog['source_document_sha256'] = canonical_digest(doc)
    before = deepcopy((doc, catalog))
    report = build_optical_binding_report(doc, catalog, aperture_indices={'condenser_aperture_2': 1})
    row, probe = report['optical_references'][0], report['optical_probes'][0]
    np.testing.assert_allclose(row['origin_mm'], (2., 0., 20.))
    np.testing.assert_allclose(row['normal'], (np.sin(.1), 0., np.cos(.1)))
    assert row['acceptance']['diameter_mm'] == .2
    np.testing.assert_allclose(probe['shape']['offset_mm'], row['origin_mm'])
    np.testing.assert_array_equal(probe['shape']['rotation'], np.eye(3))
    assert row['mechanical_reference_normal'] == [0., 0., 1.]
    assert 'optical-axis tilt is excluded' in probe['scope']
    assert (doc, catalog) == before


def test_optical_reference_tilts_but_material_passage_keeps_its_own_datum():
    doc, catalog = aperture_fixture()
    doc['parts'][0].update(optical_reference_local_z_mm=20.1,
                           offset_x_mm=2., rotation_y_mrad=100.)
    catalog['source_document_sha256'] = canonical_digest(doc)
    report = build_optical_binding_report(doc, catalog)
    row, probe = report['optical_references'][0], report['optical_probes'][0]
    np.testing.assert_allclose(row['origin_mm'], (2.+.1*np.sin(.1), 0., 20.+.1*np.cos(.1)))
    np.testing.assert_allclose(probe['shape']['offset_mm'], (2., 0., 20.1))
    np.testing.assert_array_equal(probe['shape']['rotation'], np.eye(3))
    assert any('independent of the fixed mechanical plate' in note for note in report['limitations'])


def test_mechanical_step_copy_never_acquires_source_optical_role():
    doc, catalog = aperture_fixture()
    part = doc['parts'][0]
    part.update(mechanical_only=True, mechanical_part_role='custom_mechanical_copy',
                model_3d=dict(base=dict(kind='step', asset_json='uninterpreted here')))
    catalog['source_document_sha256'] = canonical_digest(doc)
    report = build_optical_binding_report(doc, catalog)
    assert not report['optical_references'] and not report['optical_probes']
    assert report['mechanical_only'][0]['optical_role'] is None


def test_new_hole_cannot_be_claimed_to_update_aperture_acceptance_even_when_old_passage_is_clear():
    doc, catalog = aperture_fixture()
    doc['parts'][0]['model_3d'] = {'features': [dict(kind='hole', enabled=True)]}
    catalog['source_document_sha256'] = canonical_digest(doc)
    request = build_edit_validation_request(doc, catalog)
    result = finalize_edit_validation(request, receipt(request), document=doc)
    assert not result['can_apply']
    assert result['blockers'][0]['code'] == 'acceptance_geometry_edit_requires_optical_review'
    assert not result['electromagnetic_fields_recomputed']


def test_camera_uses_square_not_inscribed_circular_acceptance():
    doc = dict(module=dict(key='camera_module'), parts=[dict(key='camera', mechanical_profile='camera_sensor_plane',
        outer_width_mm=10., optical_reference_local_z_mm=30., local_center_z_mm=30.1,
        local_start_z_mm=30., local_end_z_mm=30.2)])
    catalog = dict(key='camera_module', solids=[dict(key='camera', region='sensor', material='silicon',
        status='configured_geometry', shape=box((0., 0., 30.1), (10., 10., .2)))])
    report = build_optical_binding_report(doc, catalog)
    assert report['optical_references'][0]['acceptance']['kind'] == 'square'
    probe = report['optical_probes'][0]
    assert probe['shape']['shape']['kind'] == 'box'
    assert probe['shape']['shape']['size_mm'] == [10., 10., .001]
    assert probe['expected_solid_ids'] == ['camera/sensor']


def test_filter_sensor_faces_are_mounted_once_and_alignment_is_only_a_guide():
    from temsim.cad_catalog import module_catalog
    from temsim.energy_filter_model_3d import energy_filter_frame_mm, FILTER_TO_COLUMN_ROTATION, energy_filter_origin_mm
    doc = tomllib.loads((Path(__file__).parents[1]/'configs/subassemblies/energy_filter.toml').read_text())
    # A resolved subassembly is a valid standalone module for this pure check.
    doc['module'] = dict(key='filter_fixture')
    catalog = module_catalog(doc)
    report = build_optical_binding_report(doc, catalog)
    by_key = {p['key']: p for p in doc['parts']}
    sensor = next(r for r in report['optical_references'] if r['key'] == 'energy_filter_zebra')
    origin, basis = energy_filter_frame_mm(by_key['energy_filter_zebra'], by_key)
    mount = np.asarray(FILTER_TO_COLUMN_ROTATION)
    np.testing.assert_allclose(sensor['origin_mm'], mount@origin+(0., 0., energy_filter_origin_mm(by_key)))
    np.testing.assert_allclose(sensor['basis'], mount@basis)
    probes = [p for p in report['optical_probes'] if p['part_key'] == 'energy_filter_zebra']
    assert len(probes) == by_key['energy_filter_zebra']['strip_count']
    assert all('alignment' not in p['id'] for p in probes)
    assert any(r['region'] == 'alignment_active_area' and not r['material_sensor'] for r in sensor['readout_regions'])


def test_reference_host_is_not_falsely_a_physical_host():
    parts = [dict(key='image_hp1_hexapole', optical_reference_local_z_mm=25., local_center_z_mm=25.),
             dict(key='image_dph1_deflector', optical_reference_host_key='image_hp1_hexapole',
                  length_mm=2., optical_reference_local_z_mm=0.)]
    doc = dict(module=dict(key='fixture'), parts=parts)
    report = build_optical_binding_report(doc, dict(key='fixture', solids=[]))
    channel = next(r for r in report['optical_references'] if r['key'] == 'image_dph1_deflector')
    assert channel['origin_mm'] == [0., 0., 25.]
    assert channel['physical_host_key'] is None and channel['optical_reference_only']
    assert channel['physical_host_solid_ids'] == []


@pytest.mark.parametrize('failure', ['collision', 'probe'])
def test_exact_geometry_conflicts_never_become_success(failure):
    doc, catalog = aperture_fixture()
    request = build_edit_validation_request(doc, catalog)
    result = finalize_edit_validation(request, receipt(request, collision=failure=='collision', probe_failure=failure=='probe'), document=doc)
    assert not result['can_apply'] and result['status'] == 'conflict'


def test_stale_draft_operating_state_missing_probe_and_mismatched_source_are_rejected():
    doc, catalog = aperture_fixture()
    request = build_edit_validation_request(doc, catalog)
    assert validation_is_current(request, doc)
    assert not validation_is_current(request, doc, aperture_indices={'condenser_aperture_2': 1})
    with pytest.raises(ValueError, match='stale'):
        finalize_edit_validation(request, receipt(request), document=doc, runtime_values={'x': 1})
    evidence = receipt(request)
    evidence['result']['probe_results'] = []
    evidence['result']['report_sha256'] = canonical_digest({k:v for k,v in evidence['result'].items() if k!='report_sha256'})
    with pytest.raises(ValueError, match='missing'):
        finalize_edit_validation(request, evidence)
    changed = deepcopy(doc)
    changed['parts'][0]['plate_thickness_mm'] = .3
    with pytest.raises(ValueError, match='different document'):
        build_optical_binding_report(changed, catalog)


def test_real_occ_distinguishes_clear_aperture_from_noncolliding_obstruction(tmp_path):
    root = Path(__file__).parents[1]
    interpreter = root/'environment/cad/Scripts/python.exe'
    if not interpreter.is_file():
        pytest.skip('Optional isolated CAD geometry interpreter is not installed')
    doc, catalog = aperture_fixture()
    catalog['solids'][0]['shape'] = annulus(19.9, 20.1, .05, 2.)
    for blocked in (False, True):
        if blocked:
            doc['parts'].append(dict(key='plug', mechanical_only=True, mechanical_part_role='custom_mechanical_copy'))
            catalog['solids'].append(dict(key='plug', region='body', material='metal', status='configured_geometry',
                shape=annulus(19.95, 20.05, 0., .01)))
            catalog['source_document_sha256'] = canonical_digest(doc)
        request = build_edit_validation_request(doc, catalog, request_id=f'passage-{blocked}')
        input_path = tmp_path/f'{blocked}.json'
        output = tmp_path/f'result-{blocked}'
        input_path.write_text(json.dumps(request), encoding='utf-8')
        completed = subprocess.run([str(interpreter), str(root/'scripts/cad_geometry_worker.py'),
            str(input_path), '--output', str(output)], capture_output=True, text=True, timeout=45)
        assert completed.returncode == 0, completed.stdout+completed.stderr
        result = finalize_edit_validation(request, json.loads((output/'result.json').read_text()), document=doc)
        assert result['can_apply'] is (not blocked)
        # The plug fits inside the bore without touching the material plate;
        # an ordinary material/material collision test alone would miss it.
        assert not result['geometry_report']['assemblies'][0]['collisions']
        assert bool(result['probe_results'][0]['material_blockers']) is blocked
