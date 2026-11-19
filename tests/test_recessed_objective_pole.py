"""The upper objective has two distinct material rings across its rear well."""
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
KEY = "objective_upper_pole"


@pytest.mark.parametrize("path", sorted(COLUMNS.glob("*.toml")), ids=lambda p: p.stem)
def test_native_section_and_fem_agree_through_the_stem_well_shoulder_and_bore(path):
    document = read_document(path)
    rows = {row["key"]: row for row in document["parts"]}
    upper = rows[KEY]
    geom = document["geometry"]
    start, end = upper["local_start_z_mm"], upper["local_end_z_mm"]
    solid, = round_lens_solids(upper, rows)
    section = np.array(solid.shape["section_zr_mm"])
    flange = upper["mechanical_outer_diameter_mm"]
    assert upper["pole_stem_outer_diameter_mm"] / flange == pytest.approx(.23, abs=.01)
    assert upper["pole_rear_recess_diameter_mm"] / flange == pytest.approx(.52, abs=.01)
    assert upper["pole_mounting_shank_inner_diameter_mm"] / flange == pytest.approx(.075)
    assert start == geom["c1_c2_objective_vacuum_tube_end_z_mm"]
    assert upper["pole_vacuum_connector_outer_diameter_mm"] == geom["c1_c2_objective_vacuum_tube_outer_diameter_mm"]
    assert 2 * min(section[:, 1]) == geom["c1_c2_objective_vacuum_tube_inner_diameter_mm"]
    assert sorted(section[section[:, 0] == end, 1]) == [2.88, 4.]
    field_part = dict(key=KEY, start_z_mm=start, end_z_mm=end, data=upper)
    # A transverse line through the open well crosses the stem and outer ring.
    coordinates = np.array([0., 2.87, 2.9, 10.9, 11.1, 24.9, 25.1, 45.9, 46.1])
    mask = _part_mask(field_part, coordinates * .001, (start + 20) * .001)
    assert mask.tolist() == [False, False, True, True, False, False, True, True, False]
    z, r = np.meshgrid(np.linspace(start-.137, end+.193, 197), np.linspace(.013, 48.237, 139))
    contains = PolygonPath(section).contains_points(np.column_stack((z.ravel(), r.ravel())))
    np.testing.assert_array_equal(_part_mask(field_part, r*.001, z*.001), contains.reshape(r.shape))
    assert set(RECESSED_FIELDS) <= _part_geometry_data(SimpleNamespace(data=upper, key=KEY)).keys()


def test_crossing_photons_see_two_rings_and_the_axial_channel_is_open():
    rows = {r["key"]: r for r in read_document(COLUMNS / "C2.toml")["parts"]}
    def installed(key):
        row = rows[key]
        return SimpleNamespace(key=key, data=row, definition_id=key,
                               start_z_mm=row["local_start_z_mm"], end_z_mm=row["local_end_z_mm"])
    state = SimpleNamespace(_resolved_assembly=SimpleNamespace(part=installed))
    upper, _ = objective_pole_occluders_from_state(state)
    intervals = upper.ray_intervals([0., 0., upper.z_min_mm+20.], [1., 0., 0.], 51.)
    np.testing.assert_allclose([(i.entry_distance_mm, i.exit_distance_mm) for i in intervals], [(2.88, 11.), (25., 46.)])
    for radius in (0., 2.87):
        assert not upper.ray_intervals([radius, 0., upper.z_min_mm-1.], [0., 0., 1.], 65.)
    assert upper.ray_intervals([3., 0., upper.z_min_mm-1.], [0., 0., 1.], 65.)


def test_well_depth_edit_round_trips_and_an_invalid_edit_preserves_the_draft(tmp_path):
    draft = PartModelDocument(COLUMNS / "C3_ProbeCorrector.toml")
    field = "pole_rear_recess_depth_mm"
    draft.set_dimension(("parts", KEY, field), 16.)
    updated = deepcopy(draft.part(KEY))
    with pytest.raises(ValueError, match="rear well"):
        draft.set_dimension(("parts", KEY, field), 60.)
    assert draft.part(KEY) == updated
    path = tmp_path / "column.toml"
    draft.save_copy(path)
    restored = PartModelDocument(path).part(KEY)
    assert restored[field] == 16.
    section = round_lens_solids(restored, {})[0].shape["section_zr_mm"]
    assert [restored["local_start_z_mm"]+12.+16., 25.] in section
