"""Installed local material bores are distinct from electron-accessible bores."""

from dataclasses import replace

import numpy as np
import pytest

from temsim.column.module_assembly import _module_vacuum_segments
from temsim.optics.column import default_state
from temsim.physics.closed_gun_field import _physical_liner_rows
from temsim.physics.column_wall import _expanded_profile_axis, clip_column_wall
from temsim.physics.column_wave import _wave_bores


@pytest.fixture(scope="module")
def state():
    return default_state()


def _diameters_at(segments, z_mm):
    return [segment.inner_diameter_mm for segment in segments
            if segment.start_z_mm <= z_mm <= segment.end_z_mm]


@pytest.mark.parametrize("key,diameter", (
    ("condenser_aperture_2", 5.), ("condenser_deflector", 5.),
    ("condenser_lens_3", 5.76), ("condenser_stigmator", 5.76),
    ("condenser_aperture_3", 5.), ("beam_deflector", 5.), ("adapter_lens", 5.76),
    ("probe_tl22_lens", 5.76),
))
def test_condenser_material_details_preserve_resolved_particle_and_wave_passage(state, key, diameter):
    part = state._resolved_assembly.part(key)
    z = np.array([part.center_z_mm-.05, part.center_z_mm+.05])
    assert _diameters_at(state._resolved_assembly.vacuum_bore_segments, part.center_z_mm) == [diameter]
    x = np.tile([diameter/2-.01, diameter/2+.01], (2, 1))*1e-3
    alive, stop, _ = clip_column_wall(state, z, x, np.zeros_like(x))
    assert alive.tolist() == [True, False]
    assert stop[1] == pytest.approx(z[0])
    walls, posed = _wave_bores(state, *z)
    assert posed == ()
    _, interval, nodes = _expanded_profile_axis(z, walls)
    np.testing.assert_allclose(interval, diameter/2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(nodes, diameter/2, rtol=0, atol=1e-12)
    # Working aperture clipping remains an additional independent operation.
    # The material tube ID is not substituted for a tighter local constraint.


@pytest.mark.parametrize("key", (
    "probe_hp1_hexapole", "probe_hp2_hexapole", "probe_dp11_coils",
    "probe_dp21_coils", "probe_dp22_coils",
    "probe_dph1_coils", "probe_dph2_coils",
))
def test_corrector_local_twenty_mm_bore_is_inside_narrower_continuous_tube(state, key):
    assembly = state._resolved_assembly
    part = next(part for part in assembly.parts if part.key == key)
    assert part.data["vacuum_inner_diameter_mm"] == 20.
    assert part.data["mechanical_clear_bore_diameter_mm"] == 20.
    assert _diameters_at(assembly.vacuum_bore_segments, part.center_z_mm) == [5.76]
    liner, = [segment for segment in assembly.vacuum_liner_segments
               if segment.key == "@vacuum_liner:c1_c2_to_upper_objective"]
    assert liner.start_z_mm <= part.start_z_mm < part.end_z_mm <= liner.end_z_mm
    assert liner.inner_diameter_mm == 5.76
    assert liner.outer_diameter_mm == 19.2 < part.data["mechanical_clear_bore_diameter_mm"]


def test_particle_and_wave_masks_share_the_actual_two_point_eight_eight_mm_radius(state):
    part = next(part for part in state._resolved_assembly.parts if part.key == "probe_hp1_hexapole")
    z = np.array((part.start_z_mm + .1, part.end_z_mm - .1))
    x = np.array([[2.87, 2.89, 9.], [2.87, 2.89, 9.]]) * 1e-3
    alive, stop, keys = clip_column_wall(state, z, x, np.zeros_like(x))
    assert alive.tolist() == [True, False, False]
    np.testing.assert_allclose(stop[1:], z[0], rtol=0, atol=1e-12)
    assert keys[1:] == ["column_wall", "column_wall"]
    walls, posed = _wave_bores(state, *z)
    assert posed == ()
    _, interval_radii, node_radii = _expanded_profile_axis(z, walls)
    np.testing.assert_allclose(interval_radii, 2.88, rtol=0, atol=1e-12)
    np.testing.assert_allclose(node_radii, 2.88, rtol=0, atol=1e-12)


def test_grounded_electric_boundary_uses_same_inner_tube_surface(state):
    rows = _physical_liner_rows(state.electron_gun, None)
    part = next(part for part in state._resolved_assembly.parts if part.key == "probe_hp1_hexapole")
    row, = [row for row in rows if row["start_m"] < part.center_z_mm * 1e-3 < row["stop_m"]]
    assert row["inner_m"] == pytest.approx(2.88e-3)
    assert row["outer_m"] == pytest.approx(9.6e-3)


def test_material_side_slots_and_outer_profile_do_not_silently_enlarge_beam_bore(state):
    assembly = state._resolved_assembly
    module = next(module for module in assembly.modules if module.type == "column")
    part = next(part for part in assembly.parts if part.module_key == module.key)
    origin = part.center_z_mm - float(part.data["local_center_z_mm"])
    old = _module_vacuum_segments(module, origin, assembly.parts)
    geometry = dict(module.geometry)
    geometry["mechanical_material_tube_aperture_slot_clearance_mm"] = .4
    geometry["mechanical_material_tube_outer_profile_z_od_mm"] = (
        (0., 19.2), (370., 19.2), (380., 17.), (550., 17.), (560., 19.2), (1084.5, 19.2))
    new = _module_vacuum_segments(replace(module, geometry=geometry), origin, assembly.parts)
    assert new == old


def test_resolver_reacts_to_each_independent_bore_without_rewriting_material_metadata(state):
    assembly = state._resolved_assembly
    part = next(part for part in assembly.parts if part.key == "probe_hp1_hexapole")
    module = next(module for module in assembly.modules if module.key == part.module_key)
    origin = part.center_z_mm - float(part.data["local_center_z_mm"])
    geometry = dict(module.geometry)
    geometry["c1_c2_objective_vacuum_tube_inner_diameter_mm"] = 6.
    changed = _module_vacuum_segments(replace(module, geometry=geometry), origin, assembly.parts)
    assert _diameters_at(changed, part.center_z_mm) == [6.]
    tighter = replace(part, data={**part.data, "vacuum_inner_diameter_mm": 4.})
    parts = tuple(tighter if candidate.key == part.key else candidate for candidate in assembly.parts)
    changed = _module_vacuum_segments(module, origin, parts)
    assert _diameters_at(changed, part.center_z_mm) == [4.]
    assert part.data["vacuum_inner_diameter_mm"] == 20.
    assert part.data["mechanical_clear_bore_diameter_mm"] == 20.
    assert module.geometry["c1_c2_objective_vacuum_tube_inner_diameter_mm"] == 5.76
