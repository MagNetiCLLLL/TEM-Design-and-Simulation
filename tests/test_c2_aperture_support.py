"""C2 functional attachment, unchanged working opening and tube entry."""
from collections import Counter
from copy import deepcopy
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.cad_aperture_support import (
    PREFIX, FIELDS, aperture_support_solids, aperture_support_meshes,
    validate_aperture_support,
)
from temsim.cad_auxiliary import aperture_solids
from temsim.cad_catalog import posed_solids
from temsim.cad_geometry import annulus, box, transformed
from temsim.cad_mesh import _manifold
from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, decorate_physical_tube_row, tube_solids
from temsim.module_manifest import read_document
from temsim.part_model_3d import part_model_from_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT


KEY = "condenser_aperture_2"
PATH = INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml"


@pytest.fixture
def document():
    return read_document(PATH)


def _part(document):
    return next(p for p in document["parts"] if p["key"] == KEY)


def _mat(shape):
    return _manifold(shape, 64)


def _tube(document, runtime=None):
    g = document["geometry"]
    parts = [SimpleNamespace(key=p["key"], module_key="column", data=p,
                             center_z_mm=float(p["local_center_z_mm"])+450.) for p in document["parts"]]
    module = SimpleNamespace(key="column", geometry=g, parts=parts)
    row = dict(key=CONTINUOUS_TUBE_KEY, mechanical_profile="vacuum_liner",
               local_start_z_mm=g["c1_c2_objective_vacuum_tube_start_z_mm"]+450.,
               local_end_z_mm=g["c1_c2_objective_vacuum_tube_end_z_mm"]+450.,
               mechanical_inner_diameter_mm=g["c1_c2_objective_vacuum_tube_inner_diameter_mm"],
               mechanical_outer_diameter_mm=g["c1_c2_objective_vacuum_tube_outer_diameter_mm"])
    decorated = decorate_physical_tube_row(row, SimpleNamespace(modules=[module], parts=parts), runtime_values=runtime)
    return _mat(tube_solids(decorated)[0].shape), decorated


def test_saved_opt_in_retains_optical_and_mechanism_dimensions(document):
    p = _part(document)
    assert {k for k in p if k.startswith(PREFIX)} == FIELDS
    assert (p["local_start_z_mm"], p["local_center_z_mm"], p["local_end_z_mm"],
            p["optical_reference_local_z_mm"], p["length_mm"]) == (305., 315., 325., 315., 20.)
    assert (p["plate_thickness_mm"], p["mechanical_outer_diameter_mm"],
            p["mechanical_bore_diameter_mm"], p["vacuum_inner_diameter_mm"], p["maximum_radius_mm"]) == (.2, 80., 4., 5., 2.)
    assert not any(k in p for k in ("aperture_hole_diameters_mm", "aperture_hole_diameters_um", "hole_diameters_mm", "hole_diameters_um"))
    original = deepcopy(document)
    specs = aperture_support_solids(p, {"radius_mm": .05})
    assert [s.region for s in specs] == ["body", "connection_rod", "fixing_screw"]
    assert all(s.status == "engineering_reconstruction" and not s.parameters["field_source_added"] for s in specs)
    assert all(s.parameters["working_hole_count"] == "selected_only_total_unknown" for s in specs)
    assert document == original


def test_three_materials_closed_positive_disjoint_and_attached(document):
    p = _part(document)
    specs = aperture_support_solids(p, {"radius_mm": .05})
    meshes, _ = aperture_support_meshes(p, 64, {"radius_mm": .05})
    solids = [_mat(s.shape) for s in specs]
    for mesh, solid in zip(meshes, solids):
        edges = Counter(tuple(sorted((int(a), int(b)))) for face in mesh.faces for a, b in zip(face, np.roll(face, -1)))
        assert set(edges.values()) == {2}
        assert solid.volume() > 0 and not mesh.is_exact
    for a, b in combinations(solids, 2):
        assert (a ^ b).volume() == pytest.approx(0., abs=1e-8)
    # Plate underside exactly rests on the rod; the head underside on its top.
    assert meshes[0].vertices[:, 2].max() == pytest.approx(meshes[1].vertices[:, 2].min())
    assert meshes[2].vertices[:, 2].min() == pytest.approx(314.3)
    assert meshes[2].vertices[:, 2].max() == pytest.approx(meshes[1].vertices[:, 2].max())
    # One connected assembly under zero-gap nominal contacts; no hidden nut.
    combined = solids[0]+solids[1]+solids[2]
    assert combined.volume() == pytest.approx(sum(s.volume() for s in solids), abs=1e-8)


@pytest.mark.parametrize("radius,dx,dy", [(0., 0., 0.), (.05, .2, -.1), (2., 0., 0.)])
def test_exact_existing_working_hole_and_only_one_separate_fixing_bore(document, radius, dx, dy):
    p = _part(document)
    runtime = dict(radius_mm=radius, offset_x_mm=dx, offset_y_mm=dy)
    plate, rod, screw = aperture_support_solids(p, runtime)
    area_factor = 64/2*np.sin(2*np.pi/64)
    expected = (6.5*17.75-area_factor*(radius**2+.5**2))*.2
    assert _mat(plate.shape).volume() == pytest.approx(expected, abs=1e-9)
    if radius:
        hole = transformed(annulus(314.9,315.1,0.,radius),offset=(dx,dy,0.))
        assert (_mat(plate.shape)^_mat(hole)).volume() == pytest.approx(0., abs=1e-9)
    for spec in (rod,screw):
        beam = transformed(annulus(305.,325.,0.,max(radius,.05)),offset=(dx,dy,0.))
        assert (_mat(spec.shape)^_mat(beam)).volume() == pytest.approx(0., abs=1e-9)
    meshes, _ = aperture_support_meshes(p,64,runtime)
    fix = meshes[0].surfaces["fixing_bore"]
    assert all(path[0] != "runtime" for path in fix["parameter_paths"])
    if radius:
        assert ("runtime",KEY,"radius_mm") in meshes[0].surfaces["working_opening"]["parameter_paths"]


def test_unknown_working_hole_stays_unspecified_and_runtime_does_not_resize_outline(document):
    p = _part(document)
    none = aperture_support_solids(p)
    small = aperture_support_solids(p, {"radius_mm": .05})
    large = aperture_support_solids(p, {"radius_mm": .1})
    assert none[0].parameters["working_diameter_mm"] is None
    assert small[1:] == large[1:] or [s.shape for s in small[1:]] == [s.shape for s in large[1:]]
    assert _mat(none[0].shape).volume()-_mat(small[0].shape).volume() == pytest.approx(32*np.sin(2*np.pi/64)*.05**2*.2)
    for field in ("strip_half_width_mm","strip_y_bounds_mm"):
        assert none[0].parameters[field] == small[0].parameters[field] == large[0].parameters[field]


def test_selected_declared_hole_does_not_invent_other_positions(document):
    p = _part(document);p["aperture_hole_diameters_mm"]=[.02,.08]
    selected = aperture_support_solids(p, {"radius_mm":.9},1)
    assert selected[0].parameters["working_diameter_mm"] == .08
    assert selected[0].parameters["working_hole_count"] == "selected_only_total_unknown"
    with pytest.raises(ValueError): aperture_support_solids(p,aperture_index=2)


def test_production_native_cad_and_pose_share_the_same_recipe(document):
    p = _part(document);p.update(offset_x_mm=.3,offset_y_mm=-.15,rotation_x_mrad=25.)
    runtime={"radius_mm":.05};rows={p["key"]:p for p in document["parts"]}
    specs=posed_solids(p,rows,runtime=runtime)
    model=part_model_from_document(document,KEY,include_children=False,angular_segments=64,runtime_values={KEY:runtime})
    assert {s.region for s in specs} == {m.region for m in model.meshes}
    for spec,mesh in zip(specs,model.meshes):
        bound=np.asarray(_mat(spec.shape).bounding_box()).reshape(2,3)
        np.testing.assert_allclose(np.vstack((mesh.vertices.min(0),mesh.vertices.max(0))),bound,atol=1e-10)
    tube,_ = _tube(document,{KEY:runtime})
    for spec in specs:
        assert (tube ^ _mat(transformed(spec.shape,offset=(0.,0.,450.)))).volume() == pytest.approx(0.,abs=1e-8)


def test_default_tube_slot_and_outward_insertion_path_clear_all_three_materials(document):
    p=_part(document);rows={p["key"]:p for p in document["parts"]}
    validate_aperture_support(p,rows,document["geometry"])
    tube,decorated=_tube(document)
    slot=next(s for s in decorated["mechanical_material_tube_aperture_slots"] if s["aperture_key"]==KEY)
    assert slot["clearance_mm"] == .2
    np.testing.assert_allclose(slot["local_bounds_mm"],[[-3.45,-3.45,314.7],[3.45,9.8,315.3]],atol=1e-12)
    for travel in (0.,2.,8.,20.):
        for spec in aperture_support_solids(p):
            placed=transformed(spec.shape,offset=(0.,travel,450.))
            assert (tube^_mat(placed)).volume() == pytest.approx(0.,abs=1e-8)
    # Side-entry material cut must not remove the opposite wall.
    assert (tube^_mat(box((0.,-9.4,765.),(.05,.05,.05)))).volume() > 0


@pytest.mark.parametrize("name,value", [
    ("plate_half_width_mm",0.),("plate_lower_y_mm",1.),("rod_start_y_mm",14.),
    ("rod_end_y_mm",41.),("rod_width_mm",2.),("rod_thickness_mm",20.),
    ("screw_center_y_mm",3.),("screw_shank_diameter_mm",2.5),
    ("screw_head_thickness_mm",False),("screw_head_diameter_mm",float('nan')),
    ("rod_thickness_mm",10**1000),
])
def test_invalid_explicit_geometry_rejected(document,name,value):
    p=_part(document);p[PREFIX+name]=value
    with pytest.raises(ValueError): aperture_support_solids(p)


def test_incomplete_unknown_or_wrong_host_and_tube_collision_rejected(document):
    p=_part(document)
    for change in (lambda q:q.pop(PREFIX+'source'),lambda q:q.update(aperture_support_typo_mm=1.),
                   lambda q:q.update(key='objective_aperture'),lambda q:q.update(aperture_support_source_note='')):
        q=deepcopy(p);change(q)
        with pytest.raises(ValueError):validate_aperture_support(q)
    q=deepcopy(p);q['offset_y_mm']=-2.
    with pytest.raises(ValueError,match='outside the continuous tube'):
        validate_aperture_support(q,{KEY:q},document['geometry'])
    with pytest.raises(ValueError,match='fit the declared C2 strip'):
        aperture_support_solids(p,dict(radius_mm=.05,offset_x_mm=4.))


def test_parameter_changes_and_legacy_recipe_remain_supported(document):
    p=_part(document);before=aperture_support_solids(p)
    p[PREFIX+'rod_thickness_mm']=3.
    after=aperture_support_solids(p)
    assert after[0].shape==before[0].shape or _mat(after[0].shape).volume()==pytest.approx(_mat(before[0].shape).volume())
    assert _mat(after[1].shape).volume() > _mat(before[1].shape).volume()
    legacy={k:v for k,v in p.items() if not k.startswith(PREFIX)}
    assert validate_aperture_support(legacy) is None
    assert aperture_support_solids(legacy) is None
    assert len(aperture_solids(legacy))==1


def test_copy_and_axial_placement_keep_relative_details_without_unrelated_tube_constraints(document):
    from temsim.part_model_document import PartModelDocument
    target=PartModelDocument(PATH)
    key=target.copy_component_from(document,KEY,'c2_aperture_copy',2500.,include_children=False)
    assert target.part(key)[PREFIX+'screw_center_y_mm']==13.
    target.set_dimension(('parts',key,PREFIX+'rod_start_y_mm'),5.)
    # This independent copy is far beyond the actual tube; its radial design
    # is not constrained by the tube merely because they share a module file.
    target.validate()
    original=deepcopy(target.document)
    with pytest.raises(ValueError,match='outside the continuous tube'):
        target.place_component(key,315.)
    assert target.document==original
    model=part_model_from_document(target.document,key,include_children=False)
    plate=next(m for m in model.meshes if m.region=='body')
    assert (plate.vertices[:,2].min()+plate.vertices[:,2].max())/2==pytest.approx(2500.)


def test_cad_scale_rebuilds_material_slot_with_fixed_millimetre_clearance(document):
    from temsim.part_model_document import PartModelDocument
    draft=PartModelDocument(PATH)
    draft.update_model_3d(KEY,{'transform':{'scale_xy':[2.,1.5],'offset_mm':[.3,-.15,0.],
                                          'rotation_deg':[0.,0.,15.]}})
    draft.validate()
    rows={p['key']:p for p in draft.document['parts']}
    tube,decorated=_tube(draft.document)
    slot=next(s for s in decorated['mechanical_material_tube_aperture_slots'] if s['aperture_key']==KEY)
    assert slot['clearance_mm']==.2
    np.testing.assert_allclose(slot['local_bounds_mm'],[[-6.7,-5.075,314.7],[6.7,9.8,315.3]],atol=1e-12)
    # Optical CAD rotation remains optical; mechanical offset is applied once.
    np.testing.assert_allclose(slot['rotation_local_to_global'],np.eye(3),atol=1e-12)
    np.testing.assert_allclose(slot['translation_local_mm'],[.3,-.15,0.],atol=1e-12)
    specs=posed_solids(rows[KEY],rows)
    for spec in specs:
        assert (tube^_mat(transformed(spec.shape,offset=(0.,0.,450.)))).volume()==pytest.approx(0.,abs=1e-8)


@pytest.mark.parametrize('updates',[
    {'transform':{'scale_xy':[1.,.5]}},
    {'base':{'kind':'box','width_mm':8.,'height_mm':18.,'length_mm':.2}},
])
def test_scaled_rod_collision_or_unqualified_replacement_is_rejected_atomically(updates):
    from temsim.part_model_document import PartModelDocument
    draft=PartModelDocument(PATH);before=deepcopy(draft.document)
    with pytest.raises(ValueError,match='outside the continuous tube'):
        draft.update_model_3d(KEY,updates)
    assert draft.document==before and not draft.dirty and not draft.can_undo


def test_copied_cad_rotation_and_offset_use_actual_bounds_for_tube_guard(document):
    from temsim.part_model_document import PartModelDocument
    draft=PartModelDocument(PATH)
    key=draft.copy_component_from(document,KEY,'placed_aperture_copy',2500.,include_children=False)
    before=deepcopy(draft.document)
    with pytest.raises(ValueError,match='outside the continuous tube'):
        draft.update_model_3d(key,{'transform':{'rotation_deg':[0.,0.,45.],'scale_xy':[2.,1.],
                                              'offset_mm':[0.,0.,-2185.]}})
    assert draft.document==before
    # Same axial placement safely outside the tube's radial cylinder is valid.
    draft.update_model_3d(key,{'transform':{'rotation_deg':[0.,0.,45.],'scale_xy':[2.,1.],
                                          'offset_mm':[200.,0.,-2185.]}})
    draft.validate()


def test_real_step_replacement_is_allowed_outside_tube_and_rejected_atomically_inside(document, tmp_path):
    import json
    from pathlib import Path
    import subprocess
    from temsim.part_model_document import PartModelDocument
    from temsim.step_assets import canonical_json

    root = Path(__file__).resolve().parents[1]
    cad = root / 'environment/cad/Scripts/python.exe'
    if not cad.exists():
        pytest.skip('Optional isolated OpenCascade runtime is not installed')

    def worker(name, request):
        source, output = tmp_path / f'{name}.json', tmp_path / f'{name}-result.json'
        source.write_text(canonical_json(request), encoding='utf-8')
        result = subprocess.run([str(cad), str(root / 'scripts/cad_geometry_worker.py'),
                                 '--request', str(source), '--response', str(output)],
                                capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        receipt = json.loads(output.read_text(encoding='utf-8'))
        assert receipt['status'] == 'complete', receipt
        return receipt['result']

    step = tmp_path / 'offset-source.step'
    worker('export', dict(op='export_step', path=str(step), solids=[
        dict(key='source', region='body', name='Source material', material='copper',
             color=[.8, .4, .2, 1.],
             shape=dict(kind='box', center_mm=[3., -2., 20.], size_mm=[10., 8., 6.]))]))
    asset = worker('import', dict(op='import_step', path=str(step)))
    draft = PartModelDocument(PATH)
    before = deepcopy(draft.document)
    with pytest.raises(ValueError, match='outside the continuous tube'):
        draft.import_step_asset(asset, mode='replace', target_key=KEY, registration='center')
    assert draft.document == before and not draft.dirty and not draft.can_undo

    key = draft.copy_component_from(document, KEY, 'step_aperture_copy', 2500., include_children=False)
    draft.import_step_asset(asset, mode='replace', target_key=key, registration='center')
    draft.validate()
    model = part_model_from_document(draft.document, key, include_children=False)
    points = np.vstack([mesh.vertices for mesh in model.meshes])
    np.testing.assert_allclose([points.min(0), points.max(0)],
                               [[-5., -4., 2497.], [5., 4., 2503.]], atol=1e-7)
    before = deepcopy(draft.document)
    with pytest.raises(ValueError, match='outside the continuous tube'):
        draft.place_component(key, 315.)
    assert draft.document == before
