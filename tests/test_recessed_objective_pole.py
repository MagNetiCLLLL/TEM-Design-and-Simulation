"""The mirrored objective poles share one section across CAD, FEM and EDS."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.path import Path as PolygonPath

from temsim.cad_round_lenses import round_lens_solids
from temsim.detector.eds_photon_transport import objective_pole_occluders_from_state
from temsim.module_manifest import read_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.axisymmetric_magnetostatics import _part_mask
from temsim.physics.lens_field_provider import _part_geometry_data
from temsim.pole_geometry import RECESSED_FIELDS


COLUMNS = INSTRUMENT_CONFIG_ROOT / "column"
UPPER = "objective_upper_pole"
LOWER = "objective_lower_pole"


@pytest.mark.parametrize("path", sorted(COLUMNS.glob("*.toml")), ids=lambda p: p.stem)
def test_native_section_and_fem_agree_through_the_stem_well_shoulder_and_bore(path):
    document = read_document(path)
    rows = {row["key"]: row for row in document["parts"]}
    upper, lower = rows[UPPER], rows[LOWER]
    geom = document["geometry"]
    start, end = upper["local_start_z_mm"], upper["local_end_z_mm"]
    solid, = round_lens_solids(upper, rows)
    section = np.array(solid.shape["section_zr_mm"])
    lower_solid, = round_lens_solids(lower, rows)
    lower_section = np.array(lower_solid.shape["section_zr_mm"])
    midplane = rows["objective_lens"]["local_center_z_mm"]
    expected = section.copy()
    expected[:, 0] = 2 * midplane - section[:, 0]
    np.testing.assert_allclose(lower_section, expected, atol=1e-10, rtol=0)
    assert lower_section[:, 0].min() - section[:, 0].max() == pytest.approx(5.4)
    assert "assumed to mirror" in lower["mechanical_geometry_source"]
    flange = upper["mechanical_outer_diameter_mm"]
    assert upper["pole_stem_outer_diameter_mm"] / flange == pytest.approx(.23, abs=.01)
    assert upper["pole_rear_recess_diameter_mm"] / flange == pytest.approx(.52, abs=.01)
    assert upper["pole_mounting_shank_inner_diameter_mm"] / flange == pytest.approx(.075)
    assert start == geom["c1_c2_objective_vacuum_tube_end_z_mm"]
    assert upper["pole_vacuum_connector_outer_diameter_mm"] == geom["c1_c2_objective_vacuum_tube_outer_diameter_mm"]
    assert 2 * min(section[:, 1]) == geom["c1_c2_objective_vacuum_tube_inner_diameter_mm"]
    assert sorted(section[section[:, 0] == end, 1]) == [2.88, 4.]
    field_part = dict(key=UPPER, start_z_mm=start, end_z_mm=end, data=upper)
    # A transverse line through the open well crosses the stem and outer ring.
    coordinates = np.array([0., 2.87, 2.9, 10.9, 11.1, 24.9, 25.1, 45.9, 46.1])
    mask = _part_mask(field_part, coordinates * .001, (start + 20) * .001)
    assert mask.tolist() == [False, False, True, True, False, False, True, True, False]
    z, r = np.meshgrid(np.linspace(start-.137, end+.193, 197), np.linspace(.013, 48.237, 139))
    contains = PolygonPath(section).contains_points(np.column_stack((z.ravel(), r.ravel())))
    lower_part = dict(key=LOWER, start_z_mm=lower["local_start_z_mm"],
                      end_z_mm=lower["local_end_z_mm"], data=lower)
    for part, stations in ((field_part, z), (lower_part, 2 * midplane - z)):
        np.testing.assert_array_equal(_part_mask(part, r*.001, stations*.001), contains.reshape(r.shape))
        assert set(RECESSED_FIELDS) <= _part_geometry_data(SimpleNamespace(data=part["data"], key=part["key"])).keys()


def test_crossing_photons_see_two_rings_and_the_axial_channel_is_open():
    rows = {r["key"]: r for r in read_document(COLUMNS / "C2.toml")["parts"]}
    def installed(key):
        row = rows[key]
        return SimpleNamespace(key=key, data=row, definition_id=key,
                               start_z_mm=row["local_start_z_mm"], end_z_mm=row["local_end_z_mm"])
    state = SimpleNamespace(_resolved_assembly=SimpleNamespace(part=installed))
    upper, lower = objective_pole_occluders_from_state(state)
    for pole, rear, direction in ((upper, upper.z_min_mm, 1.), (lower, lower.z_max_mm, -1.)):
        intervals = pole.ray_intervals([0., 0., rear + direction * 20.], [1., 0., 0.], 51.)
        np.testing.assert_allclose([(i.entry_distance_mm, i.exit_distance_mm) for i in intervals],
                                   [(2.88, 11.), (25., 46.)])
        for radius in (0., 2.87):
            assert not pole.ray_intervals([radius, 0., rear - direction], [0., 0., direction], 65.)
        assert pole.ray_intervals([3., 0., rear - direction], [0., 0., direction], 65.)


@pytest.mark.parametrize("key", (UPPER, LOWER))
def test_well_depth_edit_round_trips_and_an_invalid_edit_preserves_the_draft(tmp_path, key):
    draft = PartModelDocument(COLUMNS / "C3_ProbeCorrector.toml")
    field = "pole_rear_recess_depth_mm"
    draft.set_dimension(("parts", key, field), 16.)
    updated = deepcopy(draft.part(key))
    with pytest.raises(ValueError, match="rear well"):
        draft.set_dimension(("parts", key, field), 60.)
    assert draft.part(key) == updated
    path = tmp_path / "column.toml"
    draft.save_copy(path)
    restored = PartModelDocument(path).part(key)
    assert restored[field] == 16.
    section = round_lens_solids(restored, {})[0].shape["section_zr_mm"]
    floor = restored["local_start_z_mm"] + 28. if key == UPPER else restored["local_end_z_mm"] - 28.
    assert [floor, 25.] in section
