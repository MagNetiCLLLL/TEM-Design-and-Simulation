"""The diagnostic audit must not silently accept broken material shells."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.cad_geometry import SolidSpec, box
from temsim.cad_mesh import meshes_from_solids


@pytest.fixture(scope="module")
def audit():
    path = Path(__file__).resolve().parents[1]/"scripts/audit_column_model_details.py"
    spec = importlib.util.spec_from_file_location("column_model_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def closed_box():
    mesh, = meshes_from_solids((SolidSpec("fixture", "body", "electrode",
                                        box((0., 0., 1.), (2., 2., 2.))),))
    return mesh


def test_known_closed_box_has_positive_volume_and_no_topology_issues(audit, closed_box):
    before = closed_box.vertices.copy(), closed_box.faces.copy()
    health = audit.mesh_health(closed_box)
    assert health["boundary_edges"] == 0
    assert health["nonmanifold_edges"] == 0
    assert health["signed_volume_mm3"] == pytest.approx(8.)
    assert health["issues"] == []
    np.testing.assert_array_equal(closed_box.vertices, before[0])
    np.testing.assert_array_equal(closed_box.faces, before[1])


def test_open_material_is_an_issue_even_with_positive_signed_volume(audit, closed_box):
    opened = replace(closed_box, faces=closed_box.faces[:-1])
    health = audit.mesh_health(opened)
    assert health["boundary_edges"] == 3
    assert health["signed_volume_mm3"] > 0
    assert "open_material_boundary" in health["issues"]


def test_reversed_closed_material_is_an_issue(audit, closed_box):
    reversed_box = replace(closed_box, faces=closed_box.faces[:, ::-1])
    health = audit.mesh_health(reversed_box)
    assert health["boundary_edges"] == 0
    assert health["signed_volume_mm3"] == pytest.approx(-8.)
    assert health["issues"] == ["nonpositive_material_volume"]


@pytest.mark.parametrize("wireframe", [False, True])
def test_planar_reference_is_exempt_only_when_explicitly_a_wireframe(audit, closed_box, wireframe):
    plane = replace(closed_box, vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                    faces=np.array([[0, 1, 2]]), wireframe=wireframe)
    health = audit.mesh_health(plane)
    assert health["boundary_edges"] == 3 and health["signed_volume_mm3"] == 0
    assert health["issues"] == ([] if wireframe else
                                 ["open_material_boundary", "nonpositive_material_volume"])


@pytest.mark.parametrize("cad_failure", [False, True])
def test_summary_includes_synthesized_mesh_issues_and_cad_failures(audit, closed_box, monkeypatch, cad_failure):
    import temsim.assembly_model_3d as assembly_models
    import temsim.cad_catalog as catalog
    import temsim.field_vector_geometry as field_geometry

    source = dict(key="fixture", local_start_z_mm=0., local_center_z_mm=1., local_end_z_mm=2.)
    part = SimpleNamespace(key="fixture", name="Fixture", data=source, parent_key=None,
                           module_key="fixture_module", start_z_mm=0., center_z_mm=1., end_z_mm=2.)
    assembly = SimpleNamespace(modules=[SimpleNamespace(key="fixture_module")])
    synthesized = replace(closed_box, key="synthetic_liner", faces=closed_box.faces[:, ::-1])
    monkeypatch.setattr(assembly_models, "_contexts", lambda _: (
        (part,), {"fixture_module": {"parts": [source]}}))
    monkeypatch.setattr(assembly_models, "assembly_model_fingerprint", lambda *args: "unchanged")
    monkeypatch.setattr(assembly_models, "assembly_model_from_assembly", lambda *args, **kwargs:
                        SimpleNamespace(meshes=(closed_box, synthesized), errors=()))
    monkeypatch.setattr(field_geometry, "_runtime_geometry", lambda _: {})
    def cad_specs(*args, **kwargs):
        if cad_failure:
            raise ValueError("Deliberate CAD construction failure")
        return ()
    monkeypatch.setattr(catalog, "part_solids", cad_specs)
    report, _ = audit.inspect(SimpleNamespace(_resolved_assembly=assembly))
    assert report["part_count"] == 1 and report["mesh_count"] == 2
    assert report["statuses"] == {"cad_error" if cad_failure else "rendered": 1}
    assert report["synthesized_statuses"] == {"mesh_issue": 1}
    assert report["mesh_issues"] == [{"key": "synthetic_liner", "region": "body",
                                      "issues": ["nonpositive_material_volume"]}]
    assert report["cad_errors"] == ([{"key": "fixture", "error": "Deliberate CAD construction failure"}]
                                     if cad_failure else [])


def test_diagnostic_lighting_uses_face_normals_without_changing_geometry_or_material(audit):
    triangles = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                          [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.]]])
    before = triangles.copy()
    material = (.8, .6, .4, 1.)
    colours = audit._face_colours(triangles, material)
    assert colours.shape == (2, 3)
    assert np.all(colours[0] > colours[1])
    assert np.all(colours >= 0) and np.all(colours <= material[:3])
    np.testing.assert_allclose(colours[0]/np.asarray(material[:3]),
                               np.repeat(colours[0, 0]/material[0], 3))
    np.testing.assert_array_equal(triangles, before)


def test_optical_reference_does_not_hide_declared_material_omission(audit, monkeypatch):
    import temsim.assembly_model_3d as assembly_models
    import temsim.cad_catalog as catalog
    import temsim.field_vector_geometry as field_geometry

    source = dict(key="transverse_detector", local_start_z_mm=0.,
                  local_center_z_mm=0., local_end_z_mm=0.)
    part = SimpleNamespace(key=source["key"], name="Detector", data=source, parent_key=None,
                           module_key="fixture", start_z_mm=0., center_z_mm=0., end_z_mm=0.)
    assembly = SimpleNamespace(modules=[SimpleNamespace(key="fixture")])
    monkeypatch.setattr(assembly_models, "_contexts", lambda _: (
        (part,), {"fixture": {"parts": [source]}}))
    monkeypatch.setattr(assembly_models, "assembly_model_fingerprint", lambda *args: "unchanged")
    monkeypatch.setattr(assembly_models, "assembly_model_from_assembly", lambda *args, **kwargs:
                        SimpleNamespace(meshes=(), errors=()))
    monkeypatch.setattr(field_geometry, "_runtime_geometry", lambda _: {})
    material = SolidSpec(part.key, "sensor", "silicon", box((2., 0., 0.), (1., 1., 1.)))
    monkeypatch.setattr(catalog, "part_solids", lambda *args, **kwargs: (material,))
    report, _ = audit.inspect(SimpleNamespace(_resolved_assembly=assembly))
    assert report["parts"][0]["omission_reason"].startswith("zero-thickness")
    assert report["statuses"] == {"omitted_material": 1}
    assert report["material_omissions"] == [part.key]


def test_contact_tile_uses_pixel_depth_when_triangle_centres_give_the_wrong_order(audit):
    # Both triangles project onto the same XY area. Red is nearer on the right,
    # blue on the left; one whole-triangle painter order cannot represent this.
    sloping = SimpleNamespace(vertices=np.array([[-1., -1., -1.], [1., -1., 1.], [-1., 1., -1.]]),
                              faces=np.array([[0, 1, 2]]), color=(1., 0., 0., 1.), wireframe=False)
    flat = SimpleNamespace(vertices=np.array([[-1., -1., 0.], [1., -1., 0.], [-1., 1., 0.]]),
                           faces=np.array([[0, 1, 2]]), color=(0., 0., 1., 1.), wireframe=False)
    before = sloping.vertices.copy(), flat.vertices.copy()
    image = audit._raster_tile((sloping, flat), np.eye(3), size=64)
    reversed_order = audit._raster_tile((flat, sloping), np.eye(3), size=64)
    np.testing.assert_array_equal(image, reversed_order)
    assert image[52, 46, 0] > 0 and image[52, 46, 2] == 0
    assert image[46, 18, 2] > 0 and image[46, 18, 0] == 0
    np.testing.assert_array_equal(sloping.vertices, before[0])
    np.testing.assert_array_equal(flat.vertices, before[1])
