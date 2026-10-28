"""Small analytical checks for the incident-column acceptance inventory."""
from types import SimpleNamespace
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from temsim.optics.crossover_acceptance import evaluate_crossover_chain, regions_for_state


_spec = importlib.util.spec_from_file_location("compare_particle_optical_models",
    Path(__file__).resolve().parents[1] / "scripts/compare_particle_optical_models.py")
_comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_comparison)


def drift(*, count=16, root=2., z=None):
    z = np.arange(5.) if z is None else np.asarray(z, float)
    phi = np.arange(count)*2*np.pi/count
    tx = np.broadcast_to(.002*np.cos(phi), (len(z), count)).copy()
    ty = np.broadcast_to(.002*np.sin(phi), (len(z), count)).copy()
    dz = (z-root)[:, None]*1e-3
    cp = SimpleNamespace(z_mm=z, x_m=dz*tx+3e-6, y_m=dz*ty-4e-6,
                         tx_rad=tx, ty_rad=ty)
    return cp, np.ones(tx.shape, bool), np.ones(count), np.arange(count)+100


def evaluate(cp, mask, weights, identities, **kwargs):
    return evaluate_crossover_chain(cp, mask, weights, identities, source_current_a=7e-9,
        regions={"test_interval": dict(bounds_mm=[0., 4.], requirement="required_2d_waist")},
        maximum_spacing_mm=1., **kwargs)


@pytest.fixture(scope="module")
def physical_regions():
    """Use installed optical boundaries without tracing or applying a fit."""
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    for part in (*state.deflectors, *state.stigmators, *state.corrector_elements):
        if hasattr(part, "scan_enabled"):
            part.scan_enabled = False
        if hasattr(part, "wobble_enabled"):
            part.wobble_enabled = False
    result = {}
    for selected in ("normal", "large_angle"):
        state.nanoprobe_range = selected
        result[selected] = regions_for_state(state)
    return result


def synthetic_column(regions, waist_regions):
    boundaries = sorted({z for row in regions.values() for z in row["bounds_mm"]})
    roots = [sum(regions[key]["bounds_mm"])/2 for key in waist_regions]
    z = np.unique(np.r_[np.arange(boundaries[0], boundaries[-1], .5), boundaries,
                        [plane for root in roots for plane in (root-.01, root, root+.01)]])
    # Two-dimensional analytic envelopes with consistent mechanical slopes.
    # Only these supplied roots are waist candidates; no fields are evaluated.
    polynomial = np.poly1d(np.poly(roots))
    radius = polynomial(z)*1e-8
    derivative = np.polyder(polynomial)(z)*1e-5  # dx/dz: metres per metre
    phi = np.arange(16)*2*np.pi/16
    cp = SimpleNamespace(z_mm=z,
        x_m=radius[:, None]*np.cos(phi)+3e-6, y_m=radius[:, None]*np.sin(phi)-4e-6,
        tx_rad=derivative[:, None]*np.cos(phi), ty_rad=derivative[:, None]*np.sin(phi))
    return cp, np.ones(cp.x_m.shape, bool), np.ones(16), np.arange(16)


@pytest.mark.parametrize("selected,waists,passed,c2_status,c3_status", [
    ("normal", ("c2_to_c3",), True, "PASS_AT_BOUNDED_SAMPLING", "PASS_AT_BOUNDED_SAMPLING"),
    ("large_angle", ("c2_to_c3",), False, "PASS_AT_BOUNDED_SAMPLING", "FAIL_MISSING_EXECUTED_2D_WAIST"),
    ("normal", ("c2_to_c3", "c3_to_adl"), False, "PASS_AT_BOUNDED_SAMPLING", "FAIL_FORBIDDEN_WAIST"),
    ("large_angle", ("c2_to_c3", "c3_to_adl"), True, "PASS_AT_BOUNDED_SAMPLING", "PASS_AT_BOUNDED_SAMPLING"),
    ("normal", ("c3_to_adl",), False, "FAIL_MISSING_EXECUTED_2D_WAIST", "FAIL_FORBIDDEN_WAIST"),
    ("large_angle", ("c3_to_adl",), False, "FAIL_MISSING_EXECUTED_2D_WAIST", "PASS_AT_BOUNDED_SAMPLING"),
])
def test_actual_component_intervals_require_the_selected_nanoprobe_topology(
        physical_regions, selected, waists, passed, c2_status, c3_status):
    regions = physical_regions[selected]
    cp, masks, weights, identities = synthetic_column(regions, waists)
    result = evaluate_crossover_chain(cp, masks, weights, identities,
        source_current_a=7e-9, regions=regions)
    assert result["status"] == "OBSERVED_AT_BOUNDED_SAMPLING"
    assert result["required_topology_satisfied_at_sampling"] is passed
    assert result["intervals"]["c2_to_c3"]["status"] == c2_status
    assert result["intervals"]["c3_to_adl"]["status"] == c3_status
    assert {row["region"] for row in result["executed_waists"]} == set(waists)


@pytest.mark.parametrize("damage", ["missing_endpoint", "coverage_gap", "membership_change"])
def test_topology_cannot_pass_when_the_large_angle_capture_is_unresolved(physical_regions, damage):
    regions = physical_regions["large_angle"]
    cp, masks, weights, identities = synthetic_column(regions, ("c2_to_c3", "c3_to_adl"))
    if damage == "membership_change":
        # A downstream loss must remain visible without erasing either waist.
        masks[cp.z_mm > regions["c3_to_adl"]["bounds_mm"][1], 0] = False
    else:
        selected = np.ones(len(cp.z_mm), bool)
        if damage == "missing_endpoint":
            selected[0] = False
        else:
            left = regions["c3_to_adl"]["bounds_mm"][0]
            selected[(cp.z_mm > left+1) & (cp.z_mm < left+3)] = False
        cp = SimpleNamespace(**{key: value[selected] for key, value in vars(cp).items()})
        masks = masks[selected]
    result = evaluate_crossover_chain(cp, masks, weights, identities,
        source_current_a=7e-9, regions=regions)
    assert result["status"] == "UNRESOLVED"
    assert not result["required_topology_satisfied_at_sampling"]
    assert {row["region"] for row in result["executed_waists"]} == {"c2_to_c3", "c3_to_adl"}


def test_executed_waist_keeps_chief_and_current_and_refuses_rank_two_map():
    cp, masks, weight, identities = drift()
    originals = {key: getattr(cp, key).copy() for key in ("x_m", "y_m", "tx_rad", "ty_rad")}
    result = evaluate(cp, masks, weight, identities)
    assert result["status"] == "OBSERVED_AT_BOUNDED_SAMPLING"
    assert result["intervals"]["test_interval"]["status"] == "PASS_AT_BOUNDED_SAMPLING"
    waist, = result["executed_waists"]
    assert waist["z_mm"] == 2.
    assert waist["actual"]["statistics"]["mean_x_m"] == pytest.approx(3e-6)
    assert waist["actual"]["current_a"] == pytest.approx(7e-9)
    assert waist["actual"]["alpha95_mrad"] == pytest.approx(1e3*np.arctan(.002))
    assert waist["actual"]["nominal_pupil_edge_mrad"] is None
    assert result["angular_relationships"][0]["common"]["alpha95_ratio"] == pytest.approx(1.)
    assert result["angular_relationships"][0]["empirical_phase_fit"]["status"] == "UNDERDETERMINED"
    assert result["specimen"]["status"] == "ENDPOINT_OBSERVATION_NOT_TWO_SIDED_WAIST"
    json.dumps(result, allow_nan=False)
    for key, expected in originals.items():
        np.testing.assert_array_equal(getattr(cp, key), expected)
    np.testing.assert_array_equal(weight, np.ones(16))


def test_downstream_loss_does_not_erase_upstream_actual_waist():
    cp, masks, weights, identities = drift(root=1.)
    phi = np.arange(8)*2*np.pi/8
    cp.x_m[:, 8:] = 3e-6+1e-7*np.cos(phi)
    cp.y_m[:, 8:] = -4e-6+1e-7*np.sin(phi)
    cp.tx_rad[:, 8:] = 0.
    cp.ty_rad[:, 8:] = 0.
    masks[3:, :8] = False
    result = evaluate(cp, masks, weights, identities)
    waist, = result["executed_waists"]
    assert waist["z_mm"] == 1.
    assert waist["actual"]["current_a"] == pytest.approx(7e-9)
    assert waist["common"]["current_a"] == pytest.approx(3.5e-9)
    assert not waist["common_2d_waist_at_same_bracket"]
    assert result["common_cohort"]["source_ids"] == identities[8:].tolist()
    assert result["coverage"]["population_change_intervals_mm"] == [[2., 3.]]
    assert result["status"] == "UNRESOLVED"


def test_sampling_and_line_focus_cannot_pass_as_a_resolved_2d_waist():
    cp, masks, weights, identities = drift(z=[0., .5, 1., 3., 4.])
    result = evaluate(cp, masks, weights, identities)
    assert result["status"] == "UNRESOLVED"
    assert not result["coverage"]["spacing_satisfied"]
    assert result["unresolved_candidates"]
    cp, masks, weights, identities = drift()
    cp.y_m[:] = np.linspace(-1e-6, 1e-6, len(weights))
    cp.ty_rad[:] = 0.
    result = evaluate(cp, masks, weights, identities)
    assert result["executed_waists"] == []
    assert result["unresolved_candidates"]


@pytest.mark.parametrize("invalid", ["resurrection", "duplicate_id", "nonfinite_live"])
def test_invalid_capture_is_rejected(invalid):
    cp, masks, weights, identities = drift()
    if invalid == "resurrection":
        masks[1, 0] = False
    elif invalid == "duplicate_id":
        identities[0] = identities[1]
    else:
        cp.tx_rad[1, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate(cp, masks, weights, identities)


def comparison_chain(scale=1.):
    cp, masks, weights, identities = drift()
    phi = np.arange(len(weights))*2*np.pi/len(weights)
    # A tangential emittance offset gives an analytical nonzero waist while
    # retaining the same exact focus and current-carrying source cohort.
    cp.x_m = 3e-6 + scale*(cp.x_m-3e-6+1e-9*np.sin(phi))
    cp.y_m = -4e-6 + scale*(cp.y_m+4e-6-1e-9*np.cos(phi))
    cp.tx_rad *= scale
    cp.ty_rad *= scale
    return evaluate(cp, masks, weights, identities)


def test_legacy_comparison_uses_current_weighted_rms_and_reports_d95_without_mutating_capture():
    reference, candidate = comparison_chain(), comparison_chain(9.)
    frozen = deepcopy((reference, candidate))
    result = _comparison.compare_crossover_chains(reference, candidate)
    assert result["status"] == "PASS_AT_BOUNDED_SAMPLING"
    waist, = result["matched_waists"]
    assert waist["common"]["rms_radius_nm"]["ratio"] == pytest.approx(9.)
    assert waist["common"]["diameter95_nm"]["ratio"] == pytest.approx(9.)
    assert result["same_common_cohort"]
    assert (reference, candidate) == frozen
    assert _comparison.compare_crossover_chains(reference, comparison_chain(11.))["status"] == "FAIL"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("difference", ["missing_waist", "changed_region", "changed_cohort", "extra_candidate"])
def test_legacy_comparison_cannot_hide_topology_population_or_unresolved_candidate(difference):
    reference = comparison_chain()
    candidate = deepcopy(reference)
    if difference == "missing_waist":
        candidate["executed_waists"] = []
    elif difference == "changed_region":
        candidate["executed_waists"][0]["region"] = "another_component_interval"
    elif difference == "changed_cohort":
        candidate["common_cohort"]["original_weights"][0] *= .5
    else:
        candidate["unresolved_candidates"] = [dict(z_mm=3.5, kind="radial_candidate")]
    result = _comparison.compare_crossover_chains(reference, candidate)
    assert result["status"] == ("UNRESOLVED" if difference == "extra_candidate" else "FAIL")
    assert result["candidate_waist_count"] == len(candidate["executed_waists"])
    assert result["unresolved_candidates"]["candidate"] == candidate["unresolved_candidates"]


def test_legacy_comparison_does_not_create_a_floor_for_zero_size_reference():
    cp, masks, weights, identities = drift()
    reference = evaluate(cp, masks, weights, identities)
    result = _comparison.compare_crossover_chains(reference, comparison_chain())
    assert result["status"] == "FAIL"
    size = result["matched_waists"][0]["common"]["rms_radius_nm"]
    assert not size["within_limit"]
    with pytest.raises(ValueError):
        _comparison.compare_crossover_chains(reference, reference, maximum_size_ratio=float("nan"))


def test_unresolved_or_unsampled_waist_is_not_claimed_as_confirmed_missing():
    reference = comparison_chain()
    candidate = deepcopy(reference)
    candidate["executed_waists"] = []
    candidate["status"] = "UNRESOLVED"
    candidate["unresolved_candidates"] = [dict(z_mm=2., kind="radial_candidate")]
    result = _comparison.compare_crossover_chains(reference, candidate)
    assert result["status"] == "UNRESOLVED"
    assert result["observed_count_differences"][0]["status"] == "UNRESOLVED"
    assert not result["ordered_topology_matches"]
    candidate["unresolved_candidates"] = []
    candidate["intervals"]["test_interval"]["status"] = "UNRESOLVED_COVERAGE"
    assert _comparison.compare_crossover_chains(reference, candidate)["status"] == "UNRESOLVED"


def test_legacy_no_added_diagnostic_does_not_retune_fields_or_mutate_inputs():
    inputs = dict(lenses=[dict(key="objective_lens", percent=51., z_mm=1599., cs_mm=1.2),
                          dict(key="condenser_lens_2", percent=43., z_mm=700., cs_mm=None)],
        corrector_elements=[dict(key="probe_hp1_hexapole", strength_m3=12.)],
        probe_aberrations={"A1": [1., 2.]}, image_aberrations={},
        lens_intrinsic_aberrations={"objective_lens": {"B2": [1., 0.]}},
        chromatic_aberration_enabled=True)
    original = deepcopy(inputs)
    result = _comparison.without_added_aberrations(inputs)
    assert inputs == original
    assert result["corrector_elements"] == original["corrector_elements"]
    assert all(row["cs_mm"] == 0. for row in result["lenses"])
    assert [(row["percent"], row["z_mm"]) for row in result["lenses"]] == [(51., 1599.), (43., 700.)]
    assert not result["probe_aberrations"] and not result["lens_intrinsic_aberrations"]
    assert not result["chromatic_aberration_enabled"]


def test_cached_legacy_source_reuses_exact_energy_offsets_ids_weights_and_clock(tmp_path):
    from temsim.optics.column import default_state
    state = default_state()
    inputs = json.loads(json.dumps(state.to_dict()))
    prefix, cache = tmp_path/"saved", tmp_path/"gun.npz"
    ids = np.array([2, 7, 19], dtype=np.int64)
    weights, offsets = np.array([.1, .2, .7]), np.array([-.1, .01, .2])
    initial = (np.array([-1e-6, 0., 2e-6]), np.array([1e-4, -2e-4, 3e-4]),
               np.array([2e-7, 1e-7, -1e-7]), np.array([-2e-4, 3e-4, 1e-4]),
               np.array([1e-8, 1.1e-8, 1.2e-8]))
    alive = np.ones(3, bool)
    identity = _comparison._digest_arrays(*initial, offsets, weights, alive, ids)
    prefix.with_suffix(".inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
    prefix.with_suffix(".json").write_text(json.dumps(dict(source_population_identity=identity)), encoding="utf-8")
    kinetic = float(state.electron_gun.nominal_exit_energy_ev)+offsets
    np.savez(prefix.with_suffix(".npz"), z_mm=[state.electron_gun.exit_plane_z_mm, 500.],
             source_id=ids, weight=weights, source_energy_offset_ev=offsets,
             kinetic_energy_ev=np.tile(kinetic+np.spacing(kinetic), (2, 1)))
    data = dict(zip(("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"), initial))
    data.update(ray_id=ids, weight=weights, energy_offset_ev=offsets, alive=alive,
                blocked_z_mm=np.full(3, np.nan), blocked_key=np.full(3, ""))
    np.savez(cache, **data)
    returned, _, source = _comparison.read_source_result(prefix, cache)
    assert returned == inputs
    assert source["source_hash"] == identity
    np.testing.assert_array_equal(source["kinetic"], float(state.electron_gun.nominal_exit_energy_ev)+offsets)
    for actual, expected in zip(source["initial"], initial):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source["identity"][0], ids)
    np.testing.assert_array_equal(source["weights"], weights)
    np.savez(prefix.with_suffix(".npz"), z_mm=[state.electron_gun.exit_plane_z_mm, 500.],
             source_id=ids, weight=weights, source_energy_offset_ev=offsets,
             kinetic_energy_ev=np.tile(kinetic+.01, (2, 1)))
    with pytest.raises(ValueError, match="kinetic energies"):
        _comparison.read_source_result(prefix, cache)
    data["weight"] = weights*2
    np.savez(cache, **data)
    with pytest.raises(ValueError, match="fingerprint"):
        _comparison.read_source_result(prefix, cache)
