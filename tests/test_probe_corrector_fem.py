"""Registered geometry, fixed windings and magnetostatic sign/constraint checks."""

import json

import numpy as np
import pytest
from scipy.constants import mu_0

from temsim.module_manifest import read_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics import probe_corrector_fem as fem
from temsim.physics.corrector_vacuum_field import AirPotential, vacuum_field


@pytest.mark.parametrize(
    "name",
    [
        "C3_ProbeCorrector",
        "C3_ProbeCorrector_ImageCorrector",
        "C3_ProbeCorrector_BCOR",
    ],
)
def test_installed_joint_circuit_preserves_sources_and_named_materials(name):
    doc = read_document(INSTRUMENT_CONFIG_ROOT / f"column/{name}.toml")
    parts = {p["key"]: p for p in doc["parts"]}
    recipe = fem.circuit(doc)
    assert len(recipe["solids"]) == 40
    for key in fem.HP:
        part = parts[key]
        assert (
            part["mechanical_outer_diameter_mm"],
            part["mechanical_inner_diameter_mm"],
            part["mechanical_hexapole_pole_length_mm"],
        ) == (152, 8, 30)
        pins = [s for s in recipe["solids"] if s["name"].startswith(key + "/pole_pin_")]
        assert len(pins) == 6 and all(s["material"] == "permenorm_5000_h2" for s in pins)
        assert all(s["curve"]["source_url"].startswith("https://") for s in pins)
        coils = recipe["sources"][key]
        assert coils["turns"] == 2 and len(coils["coils"]) == 6
        assert [abs(c["current_sign"]) for c in coils["coils"]] == [1] * 6
    for key in fem.TL:
        assert recipe["sources"][key]["turns"] == 19
        for role in ("return_plates", "pole_sleeves"):
            assert len([s for s in recipe["solids"] if s["name"].startswith(key + "_" + role + "/")]) == 2


def test_native_air_potential_affine_reproduction():
    xyz = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    # Include distant tetrahedra to exercise the cell search.
    corners = np.vstack((xyz, xyz + 3))
    grad = np.array([0.3, -0.6, 1.2])
    f = AirPotential(
        dict(air_xyz_m=corners, air_tet=np.arange(8).reshape(2, 4), air_u_A=corners @ grad + 1.0)
    )
    points = np.array([[0.2, 0.2, 0.3], [3.1, 3.2, 3.3]])
    np.testing.assert_allclose(f.evaluate(points), 1 + points @ grad, atol=1e-13)
    np.testing.assert_allclose(f.evaluate(points, gradient=True), [grad, grad], atol=1e-13)


def test_short_exterior_mesh_blends_before_its_air_boundary(monkeypatch):
    from temsim.physics import corrector_vacuum_field as bore

    corners = np.array([[[0, 0, 0.70], [0.001, 0, 0.70], [0, 0.001, 0.70], [0, 0, 0.80]]])
    air_z_min, air_z_max = 0.615, 0.885
    sample_z = []

    class ZeroPotential:
        def evaluate(self, points):
            sample_z.extend(points[:, 2])
            return np.zeros(len(points))

    monkeypatch.setattr(bore, "AirPotential", lambda solution: ZeroPotential())
    monkeypatch.setattr(bore, "_coil_b", lambda points, solution: np.zeros_like(points))
    monkeypatch.setattr(bore, "_induced_cuda", lambda points, *args, **kwargs: np.zeros_like(points))
    solution = dict(
        corners_m=corners,
        moments_Am2=np.zeros((1, 3)),
        air_xyz_m=np.array([[0, 0, air_z_min], [0, 0, air_z_max]]),
    )
    _, _, field = bore.harmonic_table(solution, 0.005, grid_half=1, angles=6)
    assert air_z_min < min(sample_z) < 0.65 and 0.85 < max(sample_z) < air_z_max
    assert np.all(field == 0)


def test_circuit_accepts_identity_in_saved_document_and_rejects_unmodelled_pose():
    doc = read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    row = next(p for p in doc["parts"] if p["key"] == "probe_hp1_hexapole")
    reference = fem.circuit(doc)
    row["model_3d"] = {"transform": {"offset_mm": [0, 0, 0], "scale_xy": [1, 1], "rotation_deg": [0, 0, 0]}}
    assert fem.circuit(doc) == reference
    row["model_3d"]["transform"]["scale_xy"] = [1, 1.1]
    with pytest.raises(ValueError, match="centered installed geometry"):
        fem.circuit(doc)


def test_uniform_linear_iron_matches_one_free_node_solution(monkeypatch):
    xyz = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0.25, 0.25, 0.25]])
    cells = np.array([[4, 1, 2, 3], [4, 0, 2, 3], [4, 0, 1, 3], [4, 0, 1, 2]])
    labels = np.array(["test iron", "air", "air", "air"])
    recipe = dict(
        sources={"winding": {"turns": 1}},
        solids=[
            dict(
                name="test iron",
                material="fixture",
                curve=dict(h_a_per_m=[0.0, 100.0], b_t=[0.0, 200 * mu_0]),
            )
        ],
    )
    monkeypatch.setattr(
        fem, "impressed_h", lambda recipe, p, c: np.tile([c["winding"], 0.0, 0.0], (len(p), 1))
    )
    # On the diagonal exterior face the central node has gradient (-4,-4,-4).
    # On the three coordinate faces its gradients are respectively 4 e_i.
    # Relative permeability 2 inside the first cell gives u_center=-4/144.
    mesh = xyz, cells, labels, np.arange(4)
    checkpoint = {}
    for sign in (-1.0, 1.0):
        air = {}
        _, mom, r = fem.solve(
            recipe, mesh, {"winding": sign}, air_solution=air, potential_checkpoint=checkpoint
        )
        np.testing.assert_allclose(mom[0], sign * np.array([8 / 9, -1 / 9, -1 / 9]) / 24, atol=1e-10)
        assert r["residual"] < 1e-10
        assert air["air_u_A"][-1] == pytest.approx(-sign / 36)
        _, reversed_mom, reversed_report = fem.solve(
            recipe, mesh, {"winding": -sign}, potential_seed_a=checkpoint["free_potential_a"]
        )
        np.testing.assert_allclose(reversed_mom, -mom, atol=1e-12)
        assert reversed_report["residual"] < 1e-10
    with pytest.raises(ValueError, match="continuation seed"):
        fem.solve(recipe, mesh, {"winding": 1.0}, potential_seed_a=np.zeros(2))


def test_flat_tetrahedron_preserves_hanging_edge_interpolation():
    xyz = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.25, 0, 0], [0.0, 1.0, 0], [0.0, 0.0, 1]])
    good, p = fem._conforming_nodes(xyz, np.array([[0, 1, 2, 3], [0, 2, 3, 4]]), [])
    assert good.tolist() == [False, True]
    np.testing.assert_allclose(p @ np.array([2.0, 6.0, 4.0, 5.0]), [2.0, 6.0, 3.0, 4.0, 5.0])


def test_far_dipole_exterior_strength_and_sign():
    # 1 A m^2 in a tiny tetrahedron approaches the axial dipole field.
    offsets = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * 1e-5
    solution = dict(
        corners_m=offsets[None],
        moments_Am2=np.array([[0.0, 0, 1.0]]),
        recipe=json.dumps({"sources": {}}),
        report=json.dumps({"currents_A": {}}),
    )
    points = np.array([[0.0, 0, 1.0], [1.0, 0, 0.0]])
    actual = vacuum_field(points, solution, backend="cpu")
    np.testing.assert_allclose(actual[:, 2], [2 * mu_0 / (4 * np.pi), -mu_0 / (4 * np.pi)], rtol=1e-9)
