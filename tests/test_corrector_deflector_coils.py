"""Approximate mechanical windings retain open passages and honest provenance."""

from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from temsim.corrector_model_3d import corrector_deflector_meshes, deflector_coil_meshes
from temsim.part_model_3d import part_model_from_document


def _part(channel="probe_dph2_deflector", length=14.5):
    return {
        "key": channel + "_coils", "name": channel + " winding insert",
        "mechanical_profile": "corrector_deflector_coils",
        "mechanical_only": True,
        "material_class": "copper_winding",
        "local_start_z_mm": 650.0, "local_end_z_mm": 650.0 + length,
        "local_center_z_mm": 650.0 + length / 2, "length_mm": length,
        "mechanical_inner_diameter_mm": 22.2,
        "mechanical_outer_diameter_mm": 23.8,
    }


def _check_closed_loop(mesh):
    assert np.isfinite(mesh.vertices).all()
    assert mesh.faces.dtype.kind in "iu"
    assert mesh.faces.min() >= 0 and mesh.faces.max() < len(mesh.vertices)
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert (np.linalg.norm(cross, axis=1) > 0).all()
    edges = Counter(tuple(sorted(edge)) for a, b, c in mesh.faces
                    for edge in ((a, b), (b, c), (c, a)))
    assert set(edges.values()) == {2}
    # A connected solid loop has genus one, unlike a filled cylindrical sector.
    assert len(mesh.vertices) - len(edges) + len(mesh.faces) == 0
    shifted = triangles - mesh.vertices.mean(axis=0)
    volume = np.einsum("ij,ij->i", shifted[:, 0], np.cross(shifted[:, 1], shifted[:, 2])).sum() / 6
    assert volume > 0
    return volume


@pytest.mark.parametrize("channel,length", [
    ("probe_dph2_deflector", 14.5), ("probe_dph1_deflector", 14.5),
    ("probe_dp22_deflector", 14.6), ("probe_dp21_deflector", 14.6),
    ("probe_dp11_deflector", 14.5),
])
def test_coil_parts_have_four_closed_separate_loops_and_keep_document(channel, length):
    row = _part(channel, length)
    drive = {"key": channel, "layout_role": "control_channel",
             "physical_host_key": row["key"]}
    document = {"parts": [row, drive]}
    before = deepcopy(document)
    model = part_model_from_document(document, row["key"], angular_segments=16)
    assert len(model.meshes) == 4
    assert {m.region for m in model.meshes} == {
        "x_winding_1", "x_winding_2", "y_winding_1", "y_winding_2"}
    for mesh in model.meshes:
        _check_closed_loop(mesh)
        assert not mesh.is_exact and mesh.material_class == "copper_winding"
        assert "not OEM" in mesh.description and "not mesh" in mesh.description
        assert mesh.key == row["key"]
        assert mesh.vertices[:, 2].min() == pytest.approx(650.0, rel=0, abs=1e-12)
        assert mesh.vertices[:, 2].max() == pytest.approx(650.0 + length, rel=0, abs=1e-12)
    assert model.meshes[0].color == model.meshes[1].color
    assert model.meshes[2].color == model.meshes[3].color
    assert model.meshes[0].color != model.meshes[2].color
    assert any(f"Linked field source: {channel}." in note for note in model.notes)
    assert any("Display construction" in note for note in model.notes)
    assert document == before


@pytest.mark.parametrize("segments", [3, 32, 128])
def test_entire_triangle_surfaces_fit_annular_layers_and_leave_the_pipe_open(segments):
    meshes, _ = corrector_deflector_meshes(_part(), segments)
    radii = []
    for mesh in meshes:
        _check_closed_loop(mesh)
        points = mesh.vertices[:, :2]
        radius = np.linalg.norm(points, axis=1)
        assert radius.max() <= 11.9 + 1e-12
        # Check chord interiors, not just vertices of polygonal radial surfaces.
        triangles = points[mesh.faces]
        minimum = np.inf
        for a, b in ((0, 1), (1, 2), (2, 0)):
            left, delta = triangles[:, a], triangles[:, b] - triangles[:, a]
            square = np.sum(delta * delta, axis=1)
            t = np.clip(np.divide(-np.sum(left * delta, axis=1), square,
                                  out=np.zeros_like(square), where=square > 0), 0, 1)
            minimum = min(minimum, np.linalg.norm(left + t[:, None] * delta, axis=1).min())
        assert minimum >= 11.1 - 1e-12  # Declared 22.2 mm material bore stays open.
        radii.append((minimum, radius.max()))
    assert max(radii[i][1] for i in (0, 1)) < min(radii[i][0] for i in (2, 3))
    centers = [mesh.vertices[:, :2].mean(axis=0) for mesh in meshes]
    assert np.allclose(centers[0], -centers[1], atol=1e-12)
    assert np.allclose(centers[2], -centers[3], atol=1e-12)
    assert abs(np.dot(centers[0], centers[2])) < 1e-12


@pytest.mark.parametrize("change", [
    {"mechanical_inner_diameter_mm": 0.0},
    {"mechanical_outer_diameter_mm": 22.2},
    {"mechanical_outer_diameter_mm": float("nan")},
    {"local_end_z_mm": 650.0},
])
def test_invalid_coil_geometry_is_rejected(change):
    with pytest.raises(ValueError):
        corrector_deflector_meshes({**_part(), **change})


def test_original_channel_stays_nonmaterial_even_with_new_profile_metadata():
    channel = {**_part(), "key": "probe_dph2_deflector",
               "layout_role": "control_channel"}
    model = part_model_from_document({"parts": [channel]}, channel["key"])
    assert not model.meshes
    assert any("no independent material body" in note for note in model.notes)


def _native_part(**changes):
    return {**_part(), "mechanical_profile": "deflector_coil_insert",
            "winding_layer_thickness_mm": .32, "winding_angular_span_deg": 120.,
            "winding_band_width_mm": 1.16, **changes}


def test_native_dimensions_reproduce_legacy_mesh_and_use_same_public_entry():
    assert corrector_deflector_meshes is deflector_coil_meshes
    old, _ = corrector_deflector_meshes(_part())
    native, _ = deflector_coil_meshes(_native_part())
    for left, right in zip(old, native):
        np.testing.assert_allclose(left.vertices, right.vertices, rtol=0, atol=2e-13)
        np.testing.assert_array_equal(left.faces, right.faces)
        assert left.color == right.color


@pytest.mark.parametrize("field,value", [("winding_layer_thickness_mm", .5),
    ("winding_layer_thickness_mm", 0), ("winding_angular_span_deg", 180),
    ("winding_angular_span_deg", False), ("winding_band_width_mm", 8),
    ("winding_band_width_mm", float("nan")), ("winding_band_width_mm", "1.0")])
def test_native_profile_rejects_impossible_winding_dimensions(field, value):
    with pytest.raises(ValueError, match="winding"):
        deflector_coil_meshes(_native_part(**{field: value}))


def test_native_profile_requires_explicit_dimensions_and_changes_closed_solids():
    with pytest.raises(ValueError, match="winding_layer_thickness_mm"):
        deflector_coil_meshes({**_part(), "mechanical_profile": "deflector_coil_insert"})
    meshes, _ = deflector_coil_meshes(_native_part(winding_layer_thickness_mm=.25,
        winding_angular_span_deg=100., winding_band_width_mm=.8))
    for mesh in meshes:
        _check_closed_loop(mesh)
    assert np.max(np.linalg.norm(meshes[0].vertices[:, :2], axis=1)) == pytest.approx(11.35)


def test_legacy_long_coil_retains_per_layer_implicit_band_width():
    meshes, notes = corrector_deflector_meshes(_part(length=100.))
    assert sorted(set(meshes[0].vertices[:, 2]))[1] == pytest.approx(650. + .2 * 11.1)
    assert sorted(set(meshes[2].vertices[:, 2]))[1] == pytest.approx(650. + .2 * (11.1 + .6 * .8))
    assert any("Legacy file" in note for note in notes)


@pytest.mark.parametrize("band", [1.16, 4., 5., 6.])
def test_wide_side_bands_converge_to_annular_loop_volume_without_closing_window(band):
    row = _native_part(winding_band_width_mm=band)
    before = deepcopy(row)
    errors = []
    for count in (32, 128, 512):
        meshes, _ = deflector_coil_meshes(row, count)
        level_errors = []
        for index, mesh in enumerate(meshes):
            # Independent annular-sector volume minus the open central window.
            inner = 11.1 if index < 2 else 11.58
            outer = inner + .32
            span = np.deg2rad(120.)
            window_span = span - 2 * band / ((inner + outer) / 2)
            exact = .5 * (outer**2 - inner**2) * (
                span * 14.5 - window_span * (14.5 - 2 * band))
            volume = _check_closed_loop(mesh)
            assert volume <= exact * (1 + 1e-12)
            level_errors.append(abs(volume / exact - 1))
            points = mesh.vertices[:, :2]
            assert np.linalg.norm(points, axis=1).max() <= outer + 1e-12
            # Radial clearance applies to triangle interiors, not only their
            # vertices. The nearest point on a triangle is on one of its edges
            # for these annular cells, which never surround the optical axis.
            triangles = points[mesh.faces]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                left, delta = triangles[:, a], triangles[:, b] - triangles[:, a]
                square = np.einsum("ij,ij->i", delta, delta)
                t = np.clip(np.divide(-np.einsum("ij,ij->i", left, delta), square,
                    out=np.zeros_like(square), where=square > 0), 0, 1)
                assert np.linalg.norm(left + t[:, None] * delta, axis=1).min() >= inner - 1e-12
        errors.append(max(level_errors))
    assert errors[2] < errors[1] < errors[0]
    assert errors[0] < .025
    assert errors[-1] < .001
    assert row == before


def test_wide_band_saved_draft_reopens_and_keeps_native_part_and_module_visible(tmp_path):
    from temsim.part_model_document import PartModelDocument
    from temsim.part_model_3d import module_model_from_document
    from temsim.paths import INSTRUMENT_CONFIG_ROOT

    source = INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml"
    draft = PartModelDocument(source)
    key = "probe_dph2_coils"
    old_channel = deepcopy(draft.part("probe_dph2_deflector"))
    old_row = deepcopy(draft.part(key))
    old_support = tuple(mesh for mesh in part_model_from_document(draft.document, key).meshes
                        if mesh.material_class != "copper_winding")
    draft.set_dimension(("parts", key, "winding_band_width_mm"), 6.)
    destination = tmp_path / "wide-band.toml"
    draft.save_copy(destination)
    reopened = PartModelDocument(destination)
    assert reopened.part("probe_dph2_deflector") == old_channel
    assert reopened.part(key) == {**old_row, "winding_band_width_mm": 6.}
    model = part_model_from_document(reopened.document, key)
    assert len(model.meshes) == 9
    new_support = tuple(mesh for mesh in model.meshes if mesh.material_class != "copper_winding")
    for before, after in zip(old_support, new_support, strict=True):
        np.testing.assert_array_equal(before.vertices, after.vertices)
        np.testing.assert_array_equal(before.faces, after.faces)
    # Render the same source row through the module path without generating all
    # unrelated objective and detector geometry in this focused regression.
    isolated = {"parts": [deepcopy(reopened.part(key))]}
    isolated["parts"][0].pop("parent_key", None)
    assert len(module_model_from_document(isolated).meshes) == 9


def test_wide_band_honours_maximum_requested_tessellation():
    meshes, _ = deflector_coil_meshes(_native_part(winding_band_width_mm=6.), 4096)
    for mesh in meshes:
        _check_closed_loop(mesh)
        # Each of at most 4096 angular cells has two radial and four axial
        # corner coordinates; the bounded grid keeps allocation predictable.
        assert len(mesh.vertices) <= 8 * (4096 + 1)
