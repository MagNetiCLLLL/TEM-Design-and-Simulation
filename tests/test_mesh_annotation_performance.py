"""Bulk annotation preserves the existing face and editable edge contracts.

Timing is measured separately rather than asserted on shared CI hardware.
The small scalar oracle deliberately retains the prior boundary definition,
including non-manifold edges and disconnected boundary chains.
"""

from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from temsim.part_model_3d import revolve_section
from temsim.part_model_features import (
    _normal_data, _semantic_edges, annotate_legacy_mesh,
)


def _scalar_edges(vertices, faces, groups, surfaces):
    adjacency = defaultdict(list)
    for face, group in zip(faces, groups):
        for a, b in zip(face, np.roll(face, -1)):
            adjacency[tuple(sorted((int(a), int(b))))].append(str(group))
    boundaries = defaultdict(set)
    for edge, owners in adjacency.items():
        distinct = tuple(sorted(set(owners)))
        if len(distinct) > 1:
            boundaries[distinct].add(edge)
    records = []
    for pair, segments in sorted(boundaries.items()):
        neighbours = defaultdict(set)
        for a, b in segments:
            neighbours[a].add(b)
            neighbours[b].add(a)
        chains = []
        unused = set(segments)
        while unused:
            candidates = {v for edge in unused for v in edge}
            ends = [v for v in candidates if len(neighbours[v]) != 2]
            first = min(ends or candidates, key=lambda v: tuple(vertices[v]))
            chain, current = [first], first
            while True:
                available = [v for v in neighbours[current] if tuple(sorted((current, v))) in unused]
                if not available:
                    break
                following = min(available, key=lambda v: tuple(vertices[v]))
                unused.remove(tuple(sorted((current, following))))
                chain.append(following)
                current = following
                if current == first or len(neighbours[current]) != 2:
                    break
            chains.append(vertices[chain])
        chains.sort(key=lambda points: tuple(points.mean(axis=0)))
        paths = tuple(dict.fromkeys(path for surface in pair for path in surfaces[surface]["parameter_paths"]))
        label = " / ".join(surfaces[surface]["label"] for surface in pair)
        for index, points in enumerate(chains):
            records.append({"id": "|".join(pair) + f":{index}", "label": label,
                            "vertices": points, "surface_ids": pair, "parameter_paths": paths})
    return tuple(records)


def _assert_edges_equal(actual, expected):
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected, strict=True):
        assert result.keys() == reference.keys()
        for name in result:
            if name == "vertices":
                np.testing.assert_array_equal(result[name], reference[name])
                assert not result[name].flags.writeable
            else:
                assert result[name] == reference[name]


@pytest.mark.parametrize("segments,shuffle", [(8, False), (32, True), (257, True)])
def test_stepped_bore_annotation_preserves_segment_order_and_complete_edge_records(segments, shuffle):
    # Several shoulders, both a tapered and cylindrical bore, and a re-entrant
    # outer profile exercise separate axial bands of the same surface kind.
    section = [(10, 2), (10, 8), (11, 8), (11, 6), (13, 6),
               (14, 9), (16, 9), (16, 3), (14, 3), (13, 2)]
    mesh = revolve_section(section, key="stepped", angular_segments=segments)
    if shuffle:
        order = np.random.default_rng(108).permutation(len(mesh.faces))
        mesh = replace(mesh, faces=mesh.faces[order])
    part = {"key": "stepped", "length_mm": 6,
            "mechanical_outer_diameter_mm": 18, "mechanical_bore_diameter_mm": 4}
    result = annotate_legacy_mesh(mesh, part)
    centers, normals = _normal_data(mesh.vertices, mesh.faces)
    base = np.where(np.abs(normals[:, 2]) > 1 - 1e-9,
                    np.where(normals[:, 2] > 0, "axial_positive", "axial_negative"),
                    np.where(np.einsum("ij,ij->i", centers[:, :2], normals[:, :2]) < 0, "inner", "outer"))
    expected = base.astype(object)
    triangles = mesh.vertices[mesh.faces]
    for group in ("outer", "inner", "axial_negative", "axial_positive"):
        indices = np.flatnonzero(base == group)
        bands = sorted({(float(triangles[index, :, 2].min()), float(triangles[index, :, 2].max()))
                        for index in indices})
        if len(bands) > 1:
            for index in indices:
                band = (float(triangles[index, :, 2].min()), float(triangles[index, :, 2].max()))
                expected[index] = f"{group}:segment:{bands.index(band)}"
    np.testing.assert_array_equal(result.face_groups, expected)
    assert result.vertices is mesh.vertices and result.faces is mesh.faces
    assert not result.face_groups.flags.writeable
    for group, surface in result.surfaces.items():
        if group.startswith("inner"):
            assert surface["parameter_paths"] == (("parts", "stepped", "mechanical_bore_diameter_mm"),)
        elif group.startswith("outer"):
            assert surface["parameter_paths"] == (("parts", "stepped", "mechanical_outer_diameter_mm"),)
        else:
            assert surface["parameter_paths"] == (
                ("parts", "stepped", "length_mm"),
                ("parts", "stepped", "mechanical_bore_diameter_mm"),
                ("parts", "stepped", "mechanical_outer_diameter_mm"))
    _assert_edges_equal(result.edges, _scalar_edges(
        mesh.vertices, mesh.faces, result.face_groups, result.surfaces))
    assert annotate_legacy_mesh(result, part) is result


@pytest.mark.parametrize("shuffle", [False, True])
def test_boundary_owners_preserve_nonmanifold_and_disconnected_chain_semantics(shuffle):
    vertices = np.array([(0, 0, 0), (2, 0, 0), (1, 1, 0),
                         (1, -1, 0), (1, 0, 1), (4, 0, 0),
                         (6, 0, 0), (5, 1, 0), (5, -1, 0)], float)
    faces = np.array([(0, 1, 2), (1, 0, 3), (0, 1, 4),
                      (5, 6, 7), (6, 5, 8), (2, 1, 0)], int)
    groups = np.array(["cut", "cap", "inner", "cap", "cut", "cut"])
    metadata = {name: {"label": name, "parameter_paths": (("parts", "body", name),)}
                for name in set(groups)}
    if shuffle:
        order = np.random.default_rng(44).permutation(len(faces))
        faces, groups = faces[order], groups[order]
    result = _semantic_edges(vertices, faces, groups, metadata)
    _assert_edges_equal(result, _scalar_edges(vertices, faces, groups, metadata))
    assert {row["surface_ids"] for row in result} == {("cap", "cut", "inner"), ("cap", "cut")}


def test_empty_and_single_owner_edges_remain_absent():
    vertices = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], float)
    faces = np.array([(0, 1, 2)], int)
    assert _semantic_edges(vertices, faces, np.array(["same"]), {}) == ()
    assert _semantic_edges(vertices, faces[:0], np.array([], dtype=str), {}) == ()
