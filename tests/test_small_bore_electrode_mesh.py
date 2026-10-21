"""Small bores retain their geometry without refining a whole thick ring uniformly."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.physics import closed_gun_field as closed
from temsim.physics import planar_gun_field as planar
from temsim.physics.electrode_mesh import MAX_VERTICES
from temsim.physics.grounded_tip_field import refine_electrode_axes


@pytest.mark.parametrize("builder,axes", [
    (closed.closed_field_request, closed.mesh_axes),
    (lambda gun: planar.planar_field_request(gun, cathode_boundary="planar_equipotential"), planar.mesh_axes),
])
def test_tiny_bore_preserves_faces_and_local_resolution_with_bounded_growth(builder, axes):
    gun = FieldEmissionGun()
    sizes = []
    for diameter in (1e-3, 1e-6, 1e-9):
        gun.extractor.mechanical_clear_bore_diameter_mm = diameter
        request = builder(gun)
        original = deepcopy(request)
        r, z = axes(request)
        sizes.append(len(r)*len(z))
        assert request == original
        assert sizes[-1] < MAX_VERTICES
        assert np.all(np.diff(r) > 0) and np.all(np.diff(z) > 0)
        ring = request["rings"][0]
        assert ring["inner_m"] == diameter*.5e-3
        for row in request["rings"]:
            assert row["inner_m"] in r and row["outer_m"] in r
            assert row["start_m"] in z and row["stop_m"] in z
        for face in (ring["start_m"], ring["stop_m"]):
            # Every wholly contained cell within two radii of each face is
            # still resolved at the requested bore scale, in the actual grid.
            inside = (z[:-1] >= face-2*ring["inner_m"]) & (z[1:] <= face+2*ring["inner_m"])
            assert inside.any()
            assert np.max(np.diff(z)[inside]) <= ring["inner_m"]/8*1.001
    # A million-fold shrink is not a million-fold memory demand.
    assert max(sizes) < 2*min(sizes)


def test_closed_small_bore_field_keeps_equipotentials_and_strict_domain():
    gun = FieldEmissionGun()
    gun.extractor.mechanical_clear_bore_diameter_mm = 1e-6
    # Four cells are an explicit bounded regression setting, not a change to
    # the production default of eight. The original 49-ray test covers eight.
    field = closed.build_closed_gun_field(gun, cells_per_bore=4)
    assert field.report["linear_residual"] <= 1e-9
    assert field.report["vertices"] < MAX_VERTICES
    assert field.request["rings"][0]["inner_m"] == 5e-10
    for ring in field.request["rings"]:
        if ring["start_m"] == ring["stop_m"]:
            continue
        point = [[(ring["inner_m"]+ring["outer_m"])/2, 0.,
                  (ring["start_m"]+ring["stop_m"])/2]]
        voltage, electric = field.interpolate(point)
        np.testing.assert_allclose(voltage, ring["potential_rise_v"], atol=1e-7, rtol=0.)
        np.testing.assert_allclose(electric, 0., atol=1e-4, rtol=0.)
    with pytest.raises(ValueError, match="outside"):
        field.interpolate([[0., 0., field.z[-1]+.01]])


def test_grounded_tip_refines_small_bands_without_refining_whole_coarse_intervals():
    r = np.array([0., 1e-12, 1e-9, .001, .005, .01, .03])
    z = np.array([-.001, 0., 1e-9, .005, .01, .015, .05, .2])
    rings = [("extractor", .01, .015, 5e-10, .01, 4000.)]
    rr, zz = refine_electrode_axes(r, z, rings, 8)
    assert len(rr)*len(zz) < MAX_VERTICES
    assert np.all(np.isin(r, rr)) and np.all(np.isin(z, zz))
    assert rings[0][3] in rr
    assert np.min(np.diff(rr)) > 0 and np.min(np.diff(zz)) > 0
    assert (rr[0], rr[-1], zz[0], zz[-1]) == (r[0], r[-1], z[0], z[-1])


def test_mesh_budget_is_not_silently_expanded_for_small_holes(monkeypatch):
    gun = FieldEmissionGun()
    gun.extractor.mechanical_clear_bore_diameter_mm = 1e-6
    request = closed.closed_field_request(gun)
    monkeypatch.setattr(closed, "MAX_VERTICES", 1000)
    with pytest.raises(ValueError, match="vertex budget"):
        closed.mesh_axes(request)
    request = planar.planar_field_request(gun, cathode_boundary="planar_equipotential")
    request["numerics"]["maximum_vertices"] = 1000
    with pytest.raises(ValueError, match="vertex budget"):
        planar.mesh_axes(request)


def test_grounded_refinement_rejects_unrepresentable_counts_before_integer_conversion():
    with pytest.raises(ValueError, match="unrepresentable node count"):
        refine_electrode_axes(np.array([0., .002, .01, .03]),
            np.array([0., .01, .015, .2]),
            [("extractor", 1e-300, 2e-300, 1e-300, .01, 4000.)], 8)


@pytest.mark.parametrize("cells,expected_shape", [(0, (1030, 2765)), (8, (1055, 3208)), (64, (1523, 8077))])
def test_supported_surface_resolution_above_closed_field_budget_still_constructs_axes(monkeypatch, cells, expected_shape):
    """Capture immediately before tensor allocation; do not run a large FEM."""
    from dataclasses import replace
    from temsim import module_manifest
    from temsim.optics.electron_gun.tip_assembly import model_from_part
    from temsim.physics import grounded_tip_field as grounded
    gun = FieldEmissionGun()
    model = model_from_part(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    gun.emitter.surface_model = replace(model, field_numerics=replace(
        model.field_numerics, radial_nodes=1024, axial_nodes=2048, electrode_cells_per_bore=cells)).validate()
    request = grounded.field_request(gun)
    axes = []

    class Captured(Exception):
        pass

    def capture(r, z, **kwargs):
        axes.extend((r, z))
        raise Captured

    monkeypatch.setattr(np, "meshgrid", capture)
    with pytest.raises(Captured):
        grounded.GroundedTipField(request)
    assert tuple(map(len, axes)) == expected_shape  # Existing supported HEAD grids.
    r, z = axes
    assert len(r)*len(z) > MAX_VERTICES
    assert np.isfinite(r).all() and np.isfinite(z).all()
    assert np.all(np.diff(r) > 0) and np.all(np.diff(z) > 0)
    for _, start, stop, inner, outer, _ in request["rings"]:
        assert inner in r and outer in r and start in z and stop in z


def test_long_channel_potential_converges_against_independent_uniform_grid():
    """Numerical fixture only, not qualification of a nanometre instrument."""
    from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField
    from threadpoolctl import threadpool_limits
    ring = dict(key="test", start_m=.005, stop_m=.021, inner_m=.001,
                outer_m=.003, potential_rise_v=1000.)
    request = dict(rings=[ring], domain=dict(entrance_m=0., exit_m=.03, outer_radius_m=.004),
                   numerics=dict(cells_per_bore=4, maximum_vertices=MAX_VERTICES))

    def solve(r, z):
        rr, zz = np.meshgrid(r, z, indexing="ij")
        fixed, values = (zz == 0.) | (zz == .03), np.zeros_like(rr)
        values[:, -1] = 2000.
        metal = (rr >= .001) & (rr <= .003) & (zz >= .005) & (zz <= .021)
        fixed |= metal
        values[metal] = 1000.
        with threadpool_limits(limits=1):
            solved = AxisymmetricCutField(r, z, fixed, values)
        assert solved.residual <= 1e-9
        return planar.PlanarGunField(request, r, z, solved.nodal_voltage)

    points = np.array([[radius, 0., z] for radius in (0., .0003, .0007)
                       for z in (.0021, .0033, .0047, .0053, .0067, .013,
                                 .0193, .0207, .0213, .0227, .0271)])
    reference = solve(np.linspace(0., .004, 129), np.linspace(0., .03, 961))
    expected_v, expected_e = reference.interpolate(points)
    errors, field_errors = [], []
    for cells in (4, 8, 16):
        request["numerics"]["cells_per_bore"] = cells
        actual_v, actual_e = solve(*planar.mesh_axes(request)).interpolate(points)
        errors.append(np.max(abs(actual_v-expected_v)))
        field_errors.append(np.max(abs(actual_e-expected_e)))
    assert errors[0] < 3.
    assert errors[1] < .5*errors[0] and errors[2] < .5*errors[1]
    assert errors[-1] < .4  # 0.02 percent of the imposed 2 kV difference.
    assert field_errors[2] < field_errors[1] < field_errors[0]
