"""Field-aware checkpoints preserve source ownership and finite-domain meaning."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state
from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.gun_magnetic_potential import GunMagneticPotential
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.potential_column_segments import plan_potential_column_segments
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
from temsim.physics.unified_field_runtime import capture_unified_field_runtime


def _snapshot(*providers):
    return ElectromagneticPotentialSnapshot(capture_physical_scene(SimpleNamespace(lenses=[])),
        tuple(PotentialContribution(provider.source_id, (provider.source_id,), provider, "magnetic")
              for provider in providers))


def _round(key="round", registration=None, upper_mm=12.):
    native = FrozenAnalyticField(key, ((.3, 0., .006),), registration or CoordinateRegistration(),
                                 (-12., upper_mm), .003)
    return HighOrderRoundLensPotential(native, axial_domain="gaussian_all_z")


def _compact(registration=None):
    native = FrozenPosedMultipole("dipole", "dipole", registration or CoordinateRegistration(),
        ((1, 0, -.3), (0, 1, .2)), "uniform", 0., .006, (-12., 12.), .003, 3e-22)
    return AnalyticPotentialDomain(AnalyticMagneticPotential(native))


def test_gaussian_nominal_exit_keeps_its_nonzero_tail_and_snapshot_identity():
    snapshot = _snapshot(_round())
    point = np.array((.0001, .0002, .019))
    before = snapshot.sample(point)
    plan = plan_potential_column_segments(snapshot, .002, .025, maximum_segment_m=.01)
    marker, = plan.markers
    assert marker.kind == "nominal_gaussian_exit"
    assert marker.z_m == .012 and not marker.complete_global_z_face
    assert marker.z_m in plan.stops_m
    assert plan.stops_m[-1] == .025
    assert max(np.diff((.002, *plan.stops_m))) <= .01
    assert plan.snapshot_digest == snapshot.numerical_digest
    assert np.linalg.norm(before["B_t"]) > 0.
    for key, value in before.items():
        np.testing.assert_array_equal(snapshot.sample(point)[key], value)
    assert "no field removal" in plan.to_dict()["field_policy"]
    assert "no budget renewal" in plan.to_dict()["history_policy"]
    with pytest.raises(FrozenInstanceError):
        plan.stops_m = ()


def test_compact_support_entry_and_exit_are_real_model_faces():
    provider = _compact()
    snapshot = _snapshot(provider)
    plan = plan_potential_column_segments(snapshot, -.02, .02, maximum_segment_m=.03)
    assert [(row.z_m, row.kind, row.complete_global_z_face) for row in plan.markers] == [
        (-.012, "compact_support_entry", True), (.012, "compact_support_exit", True)]
    np.testing.assert_array_equal(provider.sample((.02, -.02, .013))["B_t"], 0.)
    assert set(row.z_m for row in plan.markers) <= set(plan.stops_m)


@pytest.mark.parametrize("tilt", [0., .03])
def test_gaussian_entry_and_exit_are_saved_without_dropping_entrance_or_exit_tails(tilt):
    rotation = Rotation.from_rotvec((tilt, 0., 0.)).as_matrix()
    registration = CoordinateRegistration((0., 0., .1), tuple(map(tuple, rotation)))
    snapshot = _snapshot(_round(registration=registration))
    plan = plan_potential_column_segments(snapshot, .07, .13, maximum_segment_m=.05)
    label = "nominal_gaussian" if tilt == 0. else "nominal_oblique_gaussian"
    assert [(row.z_m, row.kind, row.complete_global_z_face) for row in plan.markers] == [
        (.1-.012*rotation[2, 2], label+"_entry", False),
        (.1+.012*rotation[2, 2], label+"_exit", False)]
    assert set(row.z_m for row in plan.markers) <= set(plan.stops_m)
    assert plan.to_dict()["schema"] == "potential-column-segment-plan-v2"
    for z in (.08, .12):
        assert np.linalg.norm(snapshot.sample((0., 0., z))["B_t"]) > 0.


def test_oblique_support_markers_are_not_mislabelled_common_global_z_faces():
    rotation = Rotation.from_rotvec((.03, .02, .01)).as_matrix()
    registration = CoordinateRegistration((.001, -.002, .1), tuple(map(tuple, rotation)))
    provider = _compact(registration)
    plan = plan_potential_column_segments(_snapshot(provider), .07, .13, maximum_segment_m=.05)
    assert all(not item.complete_global_z_face for item in plan.markers)
    assert {item.kind for item in plan.markers} == {
        "nominal_oblique_compact_support_entry", "nominal_oblique_compact_support_exit"}
    assert plan.markers[-1].z_m == .1+.012*rotation[2, 2]


def test_arbitrarily_small_nonzero_tilt_stays_nominal():
    rotation = Rotation.from_rotvec((1e-15, 0., 0.)).as_matrix()
    registration = CoordinateRegistration(rotation_local_to_global=tuple(map(tuple, rotation)))
    plan = plan_potential_column_segments(_snapshot(_compact(registration)), -.02, .02,
                                          maximum_segment_m=.05)
    assert all(not item.complete_global_z_face for item in plan.markers)


def test_reversed_axis_orders_faces_in_forward_global_z():
    registration = CoordinateRegistration((0., 0., .1), ((-1., 0., 0.), (0., 1., 0.), (0., 0., -1.)))
    plan = plan_potential_column_segments(_snapshot(_compact(registration)), .07, .13,
                                          maximum_segment_m=.05)
    assert [(item.z_m, item.kind) for item in plan.markers] == [
        (.1-.012, "compact_support_entry"), (.1+.012, "compact_support_exit")]


def test_gun_soft_support_includes_edge_width_without_modifying_coefficients():
    provider = GunMagneticPotential("gun", ((.1, .003, .0002, 1., 2., 3., 4., 5.),), .002)
    plan = plan_potential_column_segments(_snapshot(provider), .09, .11, maximum_segment_m=.02)
    assert [row.z_m for row in plan.markers] == [.1-.003-.0002, .1+.003+.0002]
    assert all(row.complete_global_z_face for row in plan.markers)
    assert provider.terms[0][-5:] == (1., 2., 3., 4., 5.)


def test_bounded_native_domain_is_not_claimed_to_be_zero_exterior():
    native = _round().native
    provider = HighOrderRoundLensPotential(native, axial_domain="native_support")
    plan = plan_potential_column_segments(_snapshot(provider), -.02, .02, maximum_segment_m=.03)
    assert {row.kind for row in plan.markers} == {"validity_domain_entry", "validity_domain_exit"}
    with pytest.raises(ValueError, match="outside"):
        provider.sample((0., 0., .013))


def test_adjacent_marker_and_observation_planes_are_never_epsilon_merged():
    stop = float(np.nextafter(.012, np.inf))
    plan = plan_potential_column_segments(_snapshot(_round()), .01, stop, maximum_segment_m=.01)
    assert plan.stops_m == (.012, stop)
    assert plan.stops_m[-1].hex() == stop.hex()


def test_same_plane_and_coincident_markers_have_no_spurious_steps():
    snapshot = _snapshot(_round("first"), _round("second"))
    plan = plan_potential_column_segments(snapshot, .01, .02, maximum_segment_m=.1)
    assert len(plan.markers) == 2
    assert plan.stops_m == (.012, .02)
    same = plan_potential_column_segments(snapshot, .012, .012, maximum_segment_m=.1)
    assert same.stops_m == () and same.markers == ()


@pytest.mark.parametrize("start,stop,maximum", [
    (True, .1, .01), (0., np.inf, .01), (0., .1, np.nan), (0., .1, False),
    (.1, 0., .01), (0., .1, 0.), (0., .1, -.1), ("0", .1, .01),
    (-1e308, 1e308, 1.), (0., 1., 1e-12),
])
def test_invalid_or_unbounded_schedule_is_rejected_before_allocation(start, stop, maximum):
    with pytest.raises(ValueError):
        plan_potential_column_segments(_snapshot(), start, stop, maximum_segment_m=maximum)


def test_plain_mock_cannot_supply_potential_identity():
    with pytest.raises(ValueError, match="exact captured"):
        plan_potential_column_segments(SimpleNamespace(contributions=()), 0., 1., maximum_segment_m=.1)


@pytest.mark.parametrize("column", [option.name for option in AssemblyCatalog().columns])
def test_seven_columns_plan_retained_gaussian_and_compact_sources_without_changes(column):
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection("FEG", column, "No Energy Filter"))
    state.simulation_mode = "unified"
    runtime = capture_unified_field_runtime(state)
    identity = runtime.digest
    plan = plan_potential_column_segments(runtime.snapshot, runtime.start_z_m,
        runtime.stop_z_m, maximum_segment_m=.032)
    assert plan.stops_m[-1] == runtime.stop_z_m
    assert np.all(np.diff((runtime.start_z_m, *plan.stops_m)) > 0.)
    assert np.max(np.diff((runtime.start_z_m, *plan.stops_m))) <= .032
    assert {row.z_m for row in plan.markers} <= set(plan.stops_m)
    assert any(row.kind == "nominal_gaussian_exit" for row in plan.markers)
    assert any(row.kind == "nominal_gaussian_entry" for row in plan.markers)
    assert any(row.kind == "compact_support_exit" for row in plan.markers)
    assert "instrument_electric" in plan.sources_without_axial_marker
    assert runtime.digest == identity
    runtime.require_state(state)
    assert plan.to_dict() == plan_potential_column_segments(runtime.snapshot,
        runtime.start_z_m, runtime.stop_z_m, maximum_segment_m=.032).to_dict()
    assert plan.digest == plan_potential_column_segments(runtime.snapshot,
        runtime.start_z_m, runtime.stop_z_m, maximum_segment_m=.032).digest
