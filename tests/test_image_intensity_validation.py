"""Evidence collection contracts; these fixtures do not claim real transport."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate_image_intensity_convergence import (
    arguments, array_digest, canonical_digest, configure_source, emitted_arrays,
    matrix_pairs, run_identifier, sampling_prefix_check, screen_arrays, source_arrays,
    validate_launch_against_gun, write_npz,
)
from temsim.component_keys import FLUORESCENT_SCREEN


@pytest.fixture
def retained_section():
    ids = np.array([17, 21, 35, 44], dtype=np.int64)
    weights = np.array([.1, .2, 0., .7])
    x = np.array([1.0000000001e-8, 2e-8, -3e-8, 4e-8])
    y = np.array([-.12345678901e-8, 1e-8, 2e-8, -4e-8])
    bundle = SimpleNamespace(ray_id=ids.copy(), x_m=x.copy(), y_m=y.copy(),
        tx_rad=np.array([.001, .002, .003, .004]), ty_rad=np.zeros(4),
        weight=weights.copy(), energy_offset_ev=np.array([-.2, -.1, .1, .2]),
        alive=np.array([True, True, False, True]),
        flight_time_s=np.array([1e-11, 2e-11, np.nan, 4e-11]))
    gun = SimpleNamespace(exit_bundle=bundle, emitted_current_a=2e-10,
        blocked_key=("", "", "gun_aperture", ""),
        blocked_z_mm=np.array([np.nan, np.nan, 120., np.nan]),
        emission_reference={"position_m": np.column_stack((x, y, np.zeros(4)))})
    cp = SimpleNamespace(z_mm=np.array([1500., 1700.]),
        x_m=np.stack((x, x)), y_m=np.stack((y, y)),
        tx_rad=np.stack((bundle.tx_rad, bundle.tx_rad)), ty_rad=np.zeros((2, 4)),
        kinetic_energy_ev=np.full((2, 4), 300000.),
        flight_time_s=np.array([[1e-8, 2e-8, np.nan, 4e-8], [2e-8, 3e-8, np.nan, 5e-8]]))
    branch = SimpleNamespace(source_ray_id=ids.copy(), ray_weight=weights.copy(),
        blocked_key=np.array([FLUORESCENT_SCREEN, "condenser_aperture", FLUORESCENT_SCREEN,
                              FLUORESCENT_SCREEN], dtype=object),
        blocked_z=np.array([1700., 900., 1700., 1700.]), alive=np.zeros(4, dtype=bool),
        # Display rows intentionally have different, lower-precision values.
        x=np.full((2, 4), 7., dtype=np.float32), y=np.full((2, 4), 8., dtype=np.float32))
    result = SimpleNamespace(simulation=SimpleNamespace(gun_trace=gun,
        section_checkpoint=SimpleNamespace(segments=[SimpleNamespace(checkpoints=cp, branch=branch)])))
    return result, source_arrays(result), 1700., 2e-10


def test_default_cli_preserves_legacy_matrix_and_uses_separate_sobol_output():
    old, new = arguments([]), arguments(["--source-mode", "sobol-antithetic"])
    assert old.source_mode == "legacy" and old.source_seed == 0
    assert old.counts == [3000, 12000, 48000]
    assert old.steps == [.05, .025, .0125, .00625]
    assert old.fixed_population == 3000
    assert new.counts == [4096, 16384, 65536] and new.fixed_population == 4096
    assert new.steps == [.00625]
    assert old.output != new.output


@pytest.mark.parametrize("options", [
    ["--source-mode", "sobol-antithetic", "--counts", "3000"],
    ["--source-mode", "sobol-antithetic", "--fixed-population", "3000"],
    ["--source-mode", "sobol-antithetic", "--counts", str(2**21)],
    ["--source-seed", "1"], ["--source-seed", "-1"],
    ["--source-mode", "sobol-antithetic", "--source-seed", str(2**32)],
    ["--prepare-only", "--emit-only"],
])
def test_cli_rejects_ambiguous_or_invalid_source_protocol(options):
    with pytest.raises(SystemExit):
        arguments(options)


def test_legacy_does_not_silently_replace_an_explicit_source_rule():
    emitter = SimpleNamespace(quadrature=None, sampling=object())
    with pytest.raises(ValueError, match="recorded default"):
        configure_source(emitter, "legacy", 0)
    emitter.sampling = None
    assert configure_source(emitter, "legacy", 0) == dict(mode="legacy", record=None)
    emitter.quadrature = object()
    with pytest.raises(ValueError, match="product quadrature"):
        configure_source(emitter, "legacy", 0)


def test_explicit_sampling_rule_binds_seed_and_mode_to_protocol_identity():
    from temsim.optics.electron_gun.emitter import ColdFieldEmitter
    emitter = ColdFieldEmitter(0., 1., 1., ray_count=16)
    old = configure_source(emitter, "legacy", 0)
    first = configure_source(emitter, "sobol-antithetic", 0)
    second = configure_source(emitter, "sobol-antithetic", 1)
    assert first["record"] == dict(seed=0, schema="tip-sobol-antithetic-v1")
    assert second["record"]["seed"] == emitter.sampling.seed == 1
    assert len({canonical_digest(rule) for rule in (old, first, second)}) == 3


def test_independent_population_and_step_axes_share_one_cell_without_cartesian_expansion():
    cells = matrix_pairs([3000, 12000, 48000], [.05, .025, .0125, .00625], 3000, .00625)
    assert cells == [(3000, .05), (3000, .025), (3000, .0125), (3000, .00625),
                     (12000, .00625), (48000, .00625)]
    assert len(cells) == len(set(cells))


def test_distinct_representable_steps_cannot_overwrite_one_run_file():
    step = .025
    adjacent = np.nextafter(step, np.inf)
    assert run_identifier(3000, step, "baseline") != run_identifier(3000, adjacent, "baseline")


def test_screen_collection_uses_exact_checkpoint_and_unrenormalised_first_hits(retained_section):
    result, reference, target, current = retained_section
    before = deepcopy(result)
    arrays = screen_arrays(result, target, reference, current)
    cp = result.simulation.section_checkpoint.segments[-1].checkpoints
    np.testing.assert_array_equal(arrays["xy_m"], np.column_stack((cp.x_m[-1], cp.y_m[-1])))
    assert arrays["xy_m"].dtype == np.float64
    assert arrays["xy_m"][0, 0] != float(np.float32(cp.x_m[-1, 0]))
    np.testing.assert_array_equal(arrays["survivor"], [True, False, False, True])
    np.testing.assert_array_equal(arrays["weight"], [.1, .2, 0., .7])
    np.testing.assert_array_equal(arrays["blocked_key"],
                                  result.simulation.section_checkpoint.segments[-1].branch.blocked_key)
    np.testing.assert_array_equal(arrays["blocked_z_mm"], [1700., 900., 1700., 1700.])
    assert arrays["blocked_key"].dtype.kind == "U"
    assert not arrays["final_alive"].any()
    assert arrays["weight"][arrays["survivor"]].sum() == pytest.approx(.8)
    assert current * 1e12 * arrays["weight"][arrays["survivor"]].sum() == pytest.approx(160.)
    np.testing.assert_array_equal(result.simulation.gun_trace.exit_bundle.weight,
                                  before.simulation.gun_trace.exit_bundle.weight)
    np.testing.assert_array_equal(cp.x_m, before.simulation.section_checkpoint.segments[-1].checkpoints.x_m)


@pytest.mark.parametrize("mutation, message", [
    ("plane", "exact checkpoint"), ("float32", "float64"),
    ("no_hits", "physical screen hits"), ("ids", "ancestry"),
    ("weights", "probabilities"), ("current", "current changed"),
])
def test_screen_evidence_rejects_changed_measurement_contract(retained_section, mutation, message):
    result, reference, target, current = retained_section
    segment = result.simulation.section_checkpoint.segments[-1]
    if mutation == "plane":
        segment.checkpoints.z_mm[-1] = np.nextafter(target, np.inf)
    elif mutation == "float32":
        segment.checkpoints.x_m = segment.checkpoints.x_m.astype(np.float32)
        segment.checkpoints.y_m = segment.checkpoints.y_m.astype(np.float32)
    elif mutation == "no_hits":
        segment.branch.blocked_key[:] = "condenser_aperture"
    elif mutation == "ids":
        segment.branch.source_ray_id[[0, 1]] = segment.branch.source_ray_id[[1, 0]]
    elif mutation == "weights":
        segment.branch.ray_weight *= .9
    else:
        result.simulation.gun_trace.emitted_current_a = np.nextafter(current, np.inf)
    with pytest.raises(AssertionError, match=message):
        screen_arrays(result, target, reference, current)


@pytest.mark.parametrize("field", ["gun_exit_weight", "gun_exit_energy_offset_ev",
                                   "gun_exit_ray_id", "gun_exit_flight_time_s", "emission_position_m"])
def test_source_identity_binds_probabilities_energy_ancestry_clocks_and_emission(retained_section, field):
    _, reference, _, _ = retained_section
    changed = {key: value.copy() for key, value in reference.items()}
    changed[field].flat[0] += 1
    assert array_digest(changed) != array_digest(reference)
    assert array_digest(dict(reversed(tuple(reference.items())))) == array_digest(reference)


def test_source_evidence_rejects_non_numeric_emission_payload(retained_section):
    result, _, _, _ = retained_section
    result.simulation.gun_trace.emission_reference["invalid"] = None
    with pytest.raises(ValueError, match="numerical arrays"):
        source_arrays(result)


def test_numeric_evidence_roundtrip_preserves_full_precision_and_weights(retained_section, tmp_path):
    result, reference, target, current = retained_section
    arrays = screen_arrays(result, target, reference, current)
    path = tmp_path / "screen.npz"
    write_npz(path, arrays)
    with np.load(path, allow_pickle=False) as saved:
        assert array_digest(dict(saved)) == array_digest(arrays)
        assert saved["xy_m"].dtype == np.float64
        np.testing.assert_array_equal(saved["weight"], reference["gun_exit_weight"])
    assert not path.with_suffix(".npz.tmp").exists()


def test_prefix_report_distinguishes_exact_launch_from_actual_gun_drift():
    launch = dict(ray_id=np.arange(4, dtype=np.int64), x_m=np.arange(4)*1e-9,
                  y_m=np.zeros(4), tx_rad=np.arange(4)*1e-4, ty_rad=np.zeros(4),
                  energy_offset_ev=np.arange(4)*.01, weight=np.full(4, .25))
    small = {key: value[:2].copy() for key, value in launch.items()}
    small["weight"] *= 2
    gun_big = {"gun_exit_"+key: value.copy() for key, value in launch.items()}
    gun_big.update(gun_exit_alive=np.ones(4, dtype=bool), gun_exit_flight_time_s=np.full(4, 1e-8))
    gun_small = {key: value[:2].copy() for key, value in gun_big.items()}
    gun_big["gun_exit_x_m"] += 1e-12
    report = sampling_prefix_check(small, launch, gun_small, gun_big)
    assert report["full_emission_prefix_equal"]
    assert not report["gun_exit"]["fields"]["x_m"]["equal"]
    assert report["gun_exit"]["fields"]["x_m"]["rms"] == pytest.approx(1e-12)
    assert report["gun_exit"]["common_survivors"] == 2
    canonical_digest(report)
    changed = {key: value.copy() for key, value in launch.items()}
    changed["energy_offset_ev"][0] += .001
    assert not sampling_prefix_check(small, changed)["full_emission_prefix_equal"]


def test_emission_only_retains_full_actual_launch_and_checks_executed_lineage():
    from temsim.optics.electron_gun.emitter import ColdFieldEmitter
    emitter = ColdFieldEmitter(0., 1., 1., ray_count=16)
    launch = emitted_arrays(emitter)
    assert len(launch["ray_id"]) == 16
    assert launch["weight"].sum() == 1.
    source = {key: value.copy() for key, value in launch.items() if key.startswith("emission_")}
    source["gun_exit_weight"] = launch["weight"].copy()
    validate_launch_against_gun(launch, source)
    source["emission_position_m"][0, 0] += 1e-12
    with pytest.raises(AssertionError, match="Executed gun emission"):
        validate_launch_against_gun(launch, source)
