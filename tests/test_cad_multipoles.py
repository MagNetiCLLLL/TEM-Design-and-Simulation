"""Independent material occupancy checks for generic analytical CAD solids."""
from copy import deepcopy
from pathlib import Path
import tomllib

import numpy as np
import pytest

from temsim.cad_multipoles import deflector_insert_solids, multipole_solids
from temsim.corrector_model_3d import winding_dimensions


def _contains(shape, points):
    kind = shape["kind"]
    if kind == "transform":
        local = (points-np.asarray(shape["offset_mm"]))@np.asarray(shape["rotation"])
        return _contains(shape["shape"], local)
    if kind == "difference":
        inside = _contains(shape["base"], points)
        for tool in shape["tools"]:
            inside &= ~_contains(tool, points)
        return inside
    if kind == "union":
        return np.logical_or.reduce([_contains(child, points) for child in shape["shapes"]])
    if kind == "box":
        return np.all(abs(points-shape["center_mm"]) < np.asarray(shape["size_mm"])/2, axis=1)
    radius = np.linalg.norm(points[:, :2], axis=1)
    if kind == "revolve":
        # These builders' revolutions are rectangles in the z/r plane.
        section = np.asarray(shape["section_zr_mm"])
        return ((points[:, 2] > section[:, 0].min()) & (points[:, 2] < section[:, 0].max())
                & (radius > section[:, 1].min()) & (radius < section[:, 1].max()))
    if kind == "annular_sector":
        angle = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        delta = (angle-shape["angle_deg"]+180) % 360-180
        return ((points[:, 2] > shape["start_mm"]) & (points[:, 2] < shape["end_mm"])
                & (radius > shape["inner_mm"]) & (radius < shape["outer_mm"])
                & (abs(delta) < shape["span_deg"]/2))
    raise AssertionError(f"Unexpected untested shape {kind}")


def _row(key="probe_hp2_hexapole", inner=24., outer=75.):
    return dict(key=key, local_start_z_mm=685.4, local_end_z_mm=703.6,
        mechanical_inner_diameter_mm=inner, mechanical_outer_diameter_mm=outer,
        mechanical_clear_bore_diameter_mm=20.)


@pytest.mark.parametrize("key", ["unknown_hexapole", "objective_stigmator", "physical_quadrupole"])
def test_unknown_internals_keep_only_saved_envelope_bore_open_and_source_unchanged(key):
    row = _row(key)
    before = deepcopy(row)
    solids = multipole_solids(row, {key: row})
    assert row == before
    assert len(solids) == 1
    assert solids[0].region == "body" and solids[0].material == "unspecified"
    assert solids[0].status == "configured_envelope"
    assert "does not assert" in solids[0].note
    assert solids[0].parameters["geometry_semantics"] == "envelope_only_not_material_distribution"
    assert "physical_pole_count" not in solids[0].parameters
    rng = np.random.default_rng(749)
    points = rng.uniform([-40., -40., 684.], [40., 40., 705.], (70000, 3))
    occupancy = np.column_stack([_contains(s.shape, points) for s in solids])
    assert not np.any(occupancy.sum(axis=1) > 1)
    assert np.all(occupancy.sum(axis=0) > 0), "Every declared solid must have nonzero occupied volume"
    material = points[occupancy.any(axis=1)]
    assert (np.linalg.norm(material[:, :2], axis=1) > 12).all()
    assert (np.linalg.norm(material[:, :2], axis=1) < 37.5).all()
    assert ((material[:, 2] > 685.4) & (material[:, 2] < 703.6)).all()
    axis = np.column_stack((np.zeros(200), np.zeros(200), np.linspace(684., 705., 200)))
    assert not any(_contains(s.shape, axis).any() for s in solids)


def test_physical_pole_count_is_not_inferred_from_active_harmonic_and_channels_remain_empty():
    row = _row("physical_hexapole")
    row["physical_pole_count"] = 12
    # Even an isolated count cannot establish cores, coil dimensions, housing
    # or supports. A declared physical construction would need its own adapter.
    assert len(multipole_solids(row, {})) == 1
    assert multipole_solids(row, {})[0].status == "configured_envelope"
    for key in ("probe_qph2_quadrupole", "probe_hpol_hexapole", "probe_dph2_deflector", "descan_deflector"):
        assert multipole_solids(_row(key), {}) == ()
    assert multipole_solids(_row("objective_lens"), {}) is None


@pytest.mark.parametrize("key", ["probe_hp1_hexapole", "probe_hp2_hexapole", "copied_hp"])
def test_source_backed_cescor_hp_has_six_joined_pins_bulk_coils_and_one_yoke(key):
    row = _row(key)
    if key == "copied_hp":
        row["geometry_template_key"] = "probe_hp2_hexapole"
    before = deepcopy(row)
    solids = multipole_solids(row, {})
    assert row == before and len(solids) == 13
    assert {s.region for s in solids} == ({"return_yoke"}
        | {f"pole_pin_{i+1}" for i in range(6)} | {f"coil_{i+1}" for i in range(6)})
    assert all(s.parameters["physical_pole_count"] == 6 for s in solids)
    assert all(s.evidence[0].endswith("Mueller2006a.pdf") for s in solids)
    assert all("not used" in s.parameters["dimension_status"] for s in solids)
    assert {s.material for s in solids} == {"soft_magnetic", "copper_winding"}
    for solid in solids:
        if solid.region.startswith("pole_pin_"):
            assert solid.shape["shape"]["kind"] == "union"
    yoke = next(s for s in solids if s.region == "return_yoke")
    section = np.asarray(yoke.shape["section_zr_mm"])
    assert section[:, 1].max() == 37.5  # saved OD; never the paper's 152 mm
    assert section[:, 0].min() == 685.4 and section[:, 0].max() == 703.6
    rng = np.random.default_rng(759)
    p = rng.uniform([-38., -38., 685.], [38., 38., 704.], (120000, 3))
    occupied = np.column_stack([_contains(s.shape, p) for s in solids])
    assert np.all(occupied.sum(axis=0) > 0)
    assert not np.any(occupied.sum(axis=1) > 1)
    material = p[occupied.any(axis=1)]
    assert (np.linalg.norm(material[:, :2], axis=1) > 12.).all()
    assert (np.linalg.norm(material[:, :2], axis=1) < 37.5).all()
    assert ((material[:, 2] > 685.4) & (material[:, 2] < 703.6)).all()


def test_declared_stigmator_coil_count_does_not_create_poles_or_optional_hardware():
    row = _row("objective_stigmator")
    # Coil count without the corresponding construction is still unresolved.
    row["coil_count"] = 8
    assert multipole_solids(row, {})[0].status == "configured_envelope"
    row["coil_topology"] = "two_interleaved_four_pole_windings"
    before = deepcopy(row)
    solids = multipole_solids(row, {})
    assert row == before and len(solids) == 8
    assert {s.material for s in solids} == {"copper_winding"}
    assert {s.region for s in solids} == {f"coil_{i+1}" for i in range(8)}
    assert all("physical_pole_count" not in s.parameters for s in solids)
    assert all(s.evidence and "not OEM" in s.note for s in solids)
    rng = np.random.default_rng(781)
    p = rng.uniform([-38., -38., 685.], [38., 38., 704.], (90000, 3))
    occupied = np.column_stack([_contains(s.shape, p) for s in solids])
    assert np.all(occupied.sum(axis=0) > 0)
    assert not np.any(occupied.sum(axis=1) > 1)
    material = p[occupied.any(axis=1)]
    assert (np.linalg.norm(material[:, :2], axis=1) > 12.).all()
    assert (np.linalg.norm(material[:, :2], axis=1) < 37.5).all()
    assert ((material[:, 2] > 685.4) & (material[:, 2] < 703.6)).all()
    for bad in (True, 5, 8.5, 0, 12):
        with pytest.raises(ValueError, match="coil_count=8"):
            multipole_solids({**row, "coil_count": bad}, {})


def _stigmator_row(**changes):
    return {**_row("objective_stigmator"), "coil_count": 8,
            "coil_topology": "two_interleaved_four_pole_windings", **changes}


def test_stigmator_basis_rotation_moves_material_without_changing_axial_or_field_inputs():
    row = _stigmator_row(strength_x_percent=12., strength_y_percent=-7.,
                        effective_length_mm=6., channel_x_angle_deg=0., channel_y_angle_deg=45.)
    baseline = multipole_solids(row, {})
    # Historical rows lacking explicit angles retain the same 0/45 geometry.
    historical = multipole_solids(_stigmator_row(), {})
    assert [s.shape for s in historical] == [s.shape for s in baseline]
    turned = {**row, "channel_x_angle_deg": 17., "channel_y_angle_deg": 62.}
    before = deepcopy(turned)
    rotated = multipole_solids(turned, {})
    assert turned == before
    assert [s.region for s in rotated] == [s.region for s in baseline]
    angle = np.radians(17.)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    rng = np.random.default_rng(1401)
    points = rng.uniform([-38., -38., 685.], [38., 38., 704.], (20000, 3))
    for index, (old, new) in enumerate(zip(baseline, rotated)):
        np.testing.assert_array_equal(_contains(old.shape, points),
                                      _contains(new.shape, points @ rotation.T))
        assert new.parameters["coil_group"] == "xy"[index % 2]
        assert new.parameters["basis_angle_deg"] == (17., 62.)[index % 2]
        assert new.shape["shape"] == old.shape["shape"]  # Same axial/radial allocation.
    assert any(not np.array_equal(_contains(old.shape, points), _contains(new.shape, points))
               for old, new in zip(baseline, rotated))


@pytest.mark.parametrize("angles", [(10., 40.), (44., 45.), (15., -15.)])
def test_nonorthogonal_stigmator_basis_keeps_disjoint_copper_at_one_axial_station(angles):
    row = _stigmator_row(channel_x_angle_deg=angles[0], channel_y_angle_deg=angles[1])
    solids = multipole_solids(row, {})
    # Sample each actual end band, including very narrow coils that a uniform
    # volume sampler could miss. No other declared coil may occupy that band.
    local_points = []
    for solid in solids:
        base = solid.shape["shape"]["base"]
        centre, size = np.asarray(base["center_mm"]), np.asarray(base["size_mm"])
        point = centre + [0., 0., .44*size[2]]
        rotation = np.asarray(solid.shape["rotation"])
        local_points.append(point @ rotation.T)
        # The entire box's angular half-width is below half its nearest gap.
        radial_inner = centre[0]-size[0]/2
        half_width = np.degrees(np.arctan2(size[1]/2, radial_inner))
        assert 2*half_width < solid.parameters["minimum_coil_separation_deg"]
        assert centre[2] == pytest.approx(694.5)
        assert radial_inner > 12.
        assert np.hypot(centre[0]+size[0]/2, size[1]/2) < 37.5
    occupied = np.column_stack([_contains(s.shape, np.asarray(local_points)) for s in solids])
    np.testing.assert_array_equal(occupied, np.eye(8, dtype=bool))


@pytest.mark.parametrize("angles", [(0., 90.), (20., 110.), (0., 0.),
                                    (float("nan"), 45.), (0., float("inf")), (True, 45.)])
def test_stigmator_geometry_rejects_dependent_or_nonfinite_structural_bases(angles):
    with pytest.raises(ValueError, match="independent|finite"):
        multipole_solids(_stigmator_row(channel_x_angle_deg=angles[0], channel_y_angle_deg=angles[1]), {})


@pytest.mark.parametrize("key", ["image_hp1_hexapole", "image_hp2_hexapole"])
def test_cetcor_hp_is_explicit_family_inference_preserving_saved_envelope(key):
    document = tomllib.loads(Path("configs/instruments/column/C3_ImageCorrector.toml").read_text(encoding="utf-8"))
    row = next(p for p in document["parts"] if p["key"] == key)
    before = deepcopy(row)
    solids = multipole_solids(row, {})
    assert row == before and len(solids) == 13
    assert {s.region for s in solids} == ({"return_yoke"}
        | {f"pole_pin_{i+1}" for i in range(6)} | {f"coil_{i+1}" for i in range(6)})
    assert all(s.status == "engineering_reconstruction_family_inference" for s in solids)
    assert all(s.evidence[0] == "https://doi.org/10.1016/j.phpro.2008.07.093" for s in solids)
    assert all("unavailable" in s.note and "not a verified OEM" in s.note for s in solids)
    assert all(s.parameters["inner_radius_mm"] == 4.0 for s in solids)
    assert all(s.parameters["outer_radius_mm"] == 90.0 for s in solids)
    assert all(s.parameters["pole_axial_length_mm"] == 30.0 for s in solids)
    yoke = next(s for s in solids if s.region == "return_yoke")
    section = np.asarray(yoke.shape["section_zr_mm"])
    assert section[:, 0].min() == row["local_start_z_mm"]
    assert section[:, 0].max() == row["local_end_z_mm"]
    rng = np.random.default_rng(908)
    p = rng.uniform([-91., -91., row["local_start_z_mm"]-1.],
                    [91., 91., row["local_end_z_mm"]+1.], (90000, 3))
    occupied = np.column_stack([_contains(s.shape, p) for s in solids])
    assert np.all(occupied.sum(axis=0) > 0)
    assert not np.any(occupied.sum(axis=1) > 1)
    radii = np.linalg.norm(p[occupied.any(axis=1), :2], axis=1)
    assert np.all(radii > 4.0) and np.all(radii < 90.0)
    # Naming the optical harmonic alone still cannot invent this structure.
    unknown = {**row, "key": "unknown_image_hexapole"}
    assert multipole_solids(unknown, {})[0].status == "configured_envelope"
    channel = next(p for p in document["parts"] if p["key"] == key.replace("hp", "dph").replace("hexapole", "deflector"))
    assert multipole_solids(channel, {}) == ()


@pytest.mark.parametrize("legacy", [False, True])
def test_native_insert_retains_exact_layers_band_windows_and_driver_identity(legacy):
    row = _row("probe_dph2_coils", inner=22.2, outer=23.8)
    row["mechanical_profile"] = "corrector_deflector_coils" if legacy else "deflector_coil_insert"
    if not legacy:
        row.update(winding_layer_thickness_mm=.32, winding_angular_span_deg=120., winding_band_width_mm=1.16)
    before = deepcopy(row)
    layer, span, band = winding_dimensions(row)
    solids = deflector_insert_solids(row, {"probe_dph2_deflector": {"physical_host_key": row["key"]}})
    assert row == before and len(solids) == 4
    for i, solid in enumerate(solids):
        base, tool = solid.shape["base"], solid.shape["tools"][0]
        expected_inner = 11.1 if i < 2 else 11.9-layer
        actual_band = min(.08*18.2, .2*expected_inner) if legacy else band
        assert base["inner_mm"] == pytest.approx(expected_inner)
        assert base["outer_mm"]-base["inner_mm"] == pytest.approx(layer)
        assert base["span_deg"] == span
        assert tool["start_mm"] == pytest.approx(685.4+actual_band)
        assert solid.parameters["linked_field_channels"] == ["probe_dph2_deflector"]
        # Mid-window is genuinely empty, end band remains material.
        radius = (base["inner_mm"]+base["outer_mm"])/2
        theta = np.radians(base["angle_deg"])
        p = np.array([[radius*np.cos(theta), radius*np.sin(theta), 694.5],
                      [radius*np.cos(theta), radius*np.sin(theta), 685.4+actual_band/2]])
        assert _contains(solid.shape, p).tolist() == [False, True]


def test_actual_toml_scan_and_shift_tilt_coils_use_saved_finite_positions():
    document = tomllib.loads(Path("configs/instruments/column/C3_ProbeCorrector.toml").read_text(encoding="utf-8"))
    by_key = {row["key"]: row for row in document["parts"]}
    for key in ("beam_deflector", "ac_deflector", "condenser_deflector", "image_diffraction_deflector"):
        row = by_key[key]
        before = deepcopy(row)
        solids = multipole_solids(row, by_key)
        windings = [solid for solid in solids if solid.material == "copper_winding"]
        assert len(windings) == 8 and len(solids) == 20 and row == before
        supports = [solid for solid in solids if solid.material != "copper_winding"]
        assert len(supports) == 12 and {solid.material for solid in supports} == {"insulator"}
        assert not any(s.region in {"housing", "return_yoke"} for s in solids)
        length = row.get("mechanical_coil_length_mm", row["effective_thickness_mm"])
        centres = row["interaction_centers_local_z_mm"]
        for index, centre in enumerate(centres):
            selected = [s for s in windings if s.region.startswith(f"coil_{index+1}_")]
            assert len(selected) == 4
            for s in selected:
                assert s.shape["base"]["start_mm"] == centre-length/2
                assert s.shape["base"]["end_mm"] == centre+length/2
        # Deterministic cross-sectional material occupancy catches radial-layer
        # or neighbouring-coil overlap without relying on the display mesh.
        rng = np.random.default_rng(59)
        outer = row["mechanical_outer_diameter_mm"]/2
        p = rng.uniform([-outer, -outer, row["local_start_z_mm"]],
                        [outer, outer, row["local_end_z_mm"]], (30000, 3))
        occupied = np.column_stack([_contains(s.shape, p) for s in solids])
        assert not np.any(occupied.sum(axis=1) > 1)
        assert np.all(np.linalg.norm(p[occupied.any(axis=1), :2], axis=1) >= row["mechanical_clear_bore_diameter_mm"]/2)


def test_bad_bore_and_finite_coil_extents_are_not_silently_repaired():
    with pytest.raises(ValueError, match="clear bore"):
        multipole_solids(_row(inner=18.), {})
    row = _row("beam_deflector")
    row.update(interaction_centers_local_z_mm=[685.4, 703.6], mechanical_coil_length_mm=8.)
    with pytest.raises(ValueError, match="leaves"):
        multipole_solids(row, {})


def test_optical_deflector_centre_alone_does_not_invent_a_physical_coil_pair():
    row = _row("beam_deflector")
    row["optical_reference_local_z_mm"] = 694.5
    solids = multipole_solids(row, {})
    assert len(solids) == 1 and solids[0].status == "configured_envelope"


def test_gun_dimension_aliases_and_filter_branch_ownership():
    for path in Path("configs/instruments/gun").glob("*.toml"):
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        for row in document["parts"]:
            if row["key"].endswith(("deflector", "stigmator")):
                before = deepcopy(row)
                solids = multipole_solids(row, {})
                assert solids and row == before
                assert solids[0].parameters["outer_radius_mm"] == row["outer_diameter_mm"]/2
                if row["key"].endswith("deflector"):
                    assert solids[0].parameters["mechanical_coil_length_mm"] == row["active_length_mm"]
    # Even without a complete annular envelope the dedicated electrostatic
    # branch builder, not the magnetic family, must receive this raw part.
    assert multipole_solids(dict(key="camera_deflector", branch_path_only=True), {}) is None
