"""Convergence gates compare frozen physics and retain numerical failures."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest

spec=spec_from_file_location("particle_count_convergence",Path(__file__).resolve().parents[1]/"scripts/validate_particle_count_convergence.py")
module=module_from_spec(spec)
spec.loader.exec_module(module)


def report(count, **changes):
    point=dict(alpha95_mrad=10.,rms_radius_nm=.2,diameter95_nm=.6,current_a=1e-5,local_focus_offset_nm=0.)
    point.update(changes)
    return dict(rays=count,step_mm=.05,physical_input_signature="frozen",specimen=point,
                execution_identity=dict(solver="fixed",external_inputs="fixed"))


def test_only_numerical_counts_and_column_step_are_excluded_from_physical_identity():
    base=dict(step_mm=.05,electron_gun=dict(ray_count=193,emission_energy_ev=.3,energy_spread_fwhm_ev=.3),
              lens=dict(percent=30.,z_mm=500.,length_mm=10.),gun_trace_step_mm=.2)
    numerical=deepcopy(base);numerical["step_mm"]=.025;numerical["electron_gun"]["ray_count"]=3073
    assert module.physical_input_signature(base)==module.physical_input_signature(numerical)
    for field in ("percent","z_mm","length_mm"):
        changed=deepcopy(numerical);changed["lens"][field]+=1.
        assert module.physical_input_signature(base)!=module.physical_input_signature(changed)
    changed=deepcopy(numerical);changed["electron_gun"]["energy_spread_fwhm_ev"]=.31
    assert module.physical_input_signature(base)!=module.physical_input_signature(changed)


def test_angle_uses_nonzero_finer_population_scale_and_focus_absolute_budget():
    a=report(769);b=report(3073,alpha95_mrad=10.03,local_focus_offset_nm=.26)
    original=deepcopy((a,b))
    result=module.compare_counts(a,b)
    assert not result["passed"]
    assert result["metrics"]["alpha95_mrad"]["relative_change"]==pytest.approx(.03/10.03)
    assert not result["metrics"]["alpha95_mrad"]["passed"]
    assert not result["metrics"]["local_focus_offset_nm"]["passed"]
    assert (a,b)==original
    assert module.compare_counts(a,report(3073,alpha95_mrad=10.01,local_focus_offset_nm=.1))["passed"]


def test_real_current_and_size_changes_cannot_be_hidden_or_nonfinite_promoted():
    a=report(769);b=report(3073,current_a=.9e-5,rms_radius_nm=.25,local_focus_offset_nm=None)
    result=module.compare_counts(a,b)
    assert not result["passed"]
    assert not result["metrics"]["current_a"]["passed"]
    assert not result["metrics"]["rms_radius_nm"]["passed"]
    assert not result["metrics"]["local_focus_offset_nm"]["passed"]
    assert not module.compare_counts(a,report(3073,alpha95_mrad=0.))["passed"]
    assert not module.compare_counts(a,report(3073,alpha95_mrad=None))["passed"]
    import numpy as np
    from types import SimpleNamespace
    cp=SimpleNamespace(**{key:np.zeros((1,2)) for key in
        ("x_m","y_m","tx_rad","ty_rad","flight_time_s","kinetic_energy_ev")})
    cp.x_m[0,1]=np.nan
    with pytest.raises(ValueError,match="Nonfinite"):
        module.require_finite_population(cp,np.ones((1,2),bool),np.ones(2))
    module.require_finite_population(cp,np.array([[True,False]]),np.ones(2))


def test_mixed_physics_steps_or_nonincreasing_counts_cannot_claim_count_convergence():
    a=report(769)
    for key,value in (("physical_input_signature","changed"),("step_mm",.025),("rays",193)):
        b=report(3073);b[key]=value
        with pytest.raises(ValueError):
            module.compare_counts(a,b)
    b=report(3073);b["execution_identity"]["solver"]="different"
    result=module.compare_counts(a,b)
    assert result["numerical_metrics_passed"] and not result["passed"]
    assert result["execution_identity_status"]=="MISSING_OR_DIFFERENT"
    b=report(3073);b.pop("execution_identity")
    assert not module.compare_counts(a,b)["passed"]


benchmark_spec = spec_from_file_location("benchmark_unified_column",
    Path(__file__).resolve().parents[1] / "scripts/benchmark_unified_column.py")
benchmark = module_from_spec(benchmark_spec)
benchmark_spec.loader.exec_module(benchmark)


def _column_arrays():
    import numpy as np
    values = {key: np.zeros((2, 2), dtype=np.float64) for key in benchmark.PHASE_KEYS}
    values.update(z_mm=np.array([0., 1.]), alive=np.ones((2, 2), bool),
                  weight=np.array([.4, .6]), source_id=np.array([10, 20], dtype=np.int64),
                  source_energy_offset_ev=np.array([-.1, .1]))
    return values


def test_column_accuracy_gates_are_strict_and_bytes_are_not_elementwise_equality():
    import numpy as np
    old = _column_arrays()
    for key, (atol, rtol) in benchmark.TOLERANCES.items():
        changed = deepcopy(old)
        changed[key][0, 0] = atol * .5
        result = benchmark.compare_arrays(changed, old)
        assert result["passed"] and not result["all_arrays_bitwise_equal"]
        assert result["arrays"][key]["max_absolute_difference"] == atol * .5
        changed[key][0, 0] = atol * 2.
        assert not benchmark.compare_arrays(changed, old)["passed"]
        changed[key][0, 0] = np.nan
        assert not benchmark.compare_arrays(changed, changed)["passed"]
    for key in benchmark.EXACT_KEYS:
        changed = deepcopy(old)
        changed[key].flat[0] = False if key == "alive" else changed[key].flat[0] + 1
        assert not benchmark.compare_arrays(changed, old)["passed"]
    signed = deepcopy(old)
    signed["x_m"][0, 0] = -0.0
    result = benchmark.compare_arrays(signed, old)
    assert result["passed"] and not result["arrays"]["x_m"]["bitwise_equal"]
    # Only jointly inactive NaNs are legitimate stored missing states.
    old["alive"][0, 0] = False
    old["x_m"][0, 0] = np.nan
    assert benchmark.compare_arrays(old, old)["passed"]
    old["x_m"][0, 0] = np.inf
    assert not benchmark.compare_arrays(old, old)["passed"]


def test_cached_gun_source_and_original_nominal_energy_cannot_be_replaced():
    import numpy as np
    reference = _column_arrays()
    reference["kinetic_energy_ev"][0] = 999.  # Stored checkpoint K is deliberately unrelated.
    source = {key: reference[key][0].copy() for key in benchmark.PHASE_KEYS[:-1]}
    source.update(energy_offset_ev=reference["source_energy_offset_ev"].copy(),
                  weight=reference["weight"].copy(), alive=reference["alive"][0].copy(),
                  ray_id=reference["source_id"].copy(), blocked_z_mm=np.full(2, np.nan),
                  blocked_key=np.array(["", ""]))
    identity = benchmark._digest_arrays(*(source[key] for key in benchmark.SOURCE_KEYS))
    baseline = dict(rays=2, source_population_identity=identity)
    frozen = dict(source_population_identity=identity)
    observed, initial = benchmark.check_source(source, reference, baseline, frozen, nominal_exit_energy_ev=300000.)
    assert observed == identity
    assert benchmark.bytes_equal(initial, 300000. + source["energy_offset_ev"])
    assert not benchmark.bytes_equal(initial, reference["kinetic_energy_ev"][0])
    for key in ("ray_id", "weight", "energy_offset_ev", "flight_time_s"):
        changed = deepcopy(source)
        changed[key][0] += 1
        with pytest.raises(ValueError, match="hash changed"):
            benchmark.check_source(changed, reference, baseline, frozen, nominal_exit_energy_ev=300000.)
    changed = deepcopy(reference)
    changed["x_m"][0, 0] += 1e-9
    with pytest.raises(ValueError, match="differs from the archived entrance"):
        benchmark.check_source(source, changed, baseline, frozen, nominal_exit_energy_ev=300000.)


def test_column_snapshot_accepts_only_implementation_change_not_physics_or_external_data():
    from temsim.instrument_snapshot import InstrumentSnapshot
    inputs = dict(step_mm=.05, lens=dict(percent=30.))
    external = ({"role": "model", "path": "fixture", "content_hex": None, "sha256": "fixed"},)
    historical = InstrumentSnapshot(inputs, external, "old-solver")
    current = InstrumentSnapshot(inputs, external, "new-solver")
    baseline = dict(physical_input_signature=module.physical_input_signature(inputs),
                    snapshot_digest=historical.digest, execution_identity=module.execution_identity(historical))
    benchmark.check_snapshot_identity(inputs, baseline, historical, current)
    changed = dict(step_mm=.05, lens=dict(percent=31.))
    with pytest.raises(ValueError, match="input graph changed"):
        benchmark.check_snapshot_identity(inputs, baseline, historical, InstrumentSnapshot(changed, external, "new"))
    with pytest.raises(ValueError, match="external input identities changed"):
        benchmark.check_snapshot_identity(inputs, baseline, historical, InstrumentSnapshot(inputs, (), "new"))
    with pytest.raises(ValueError, match="physical inputs"):
        benchmark.check_snapshot_identity(changed, baseline, historical, current)
    damaged = deepcopy(baseline)
    damaged["execution_identity"]["solver"] = "unrelated"
    with pytest.raises(ValueError, match="disagrees"):
        benchmark.check_snapshot_identity(inputs, damaged, historical, current)


def test_column_timing_excludes_old_total_and_evidence_is_never_overwritten(tmp_path):
    plan = {key: key for key in benchmark.PLAN_KEYS}
    baseline = dict(rays=2, execution_identity=dict(solver="native"))
    timing = dict(stages=[dict(label="complete_column_batch", seconds=1143.2257047)],
                  baseline_elapsed_s=3539., optimized_implementation="native", column_batch_populations=[2],
                  plans=[{key: plan[key] for key in benchmark.PLAN_KEYS[:-1]}],
                  same_physical_input_signature=True, same_source_population=True, same_observation_mesh=True)
    assert benchmark.baseline_column_seconds(timing, baseline, plan) == 1143.2257047
    for rows in ([], timing["stages"] * 2, [dict(label="complete_column_batch", seconds=float("nan"))]):
        changed = deepcopy(timing)
        changed["stages"] = rows
        with pytest.raises(ValueError):
            benchmark.baseline_column_seconds(changed, baseline, plan)
    output = tmp_path / "replay.json"
    paths = benchmark.reserve_output(output)
    before = paths["reservation"].read_bytes()
    with pytest.raises(FileExistsError, match="Preserve"):
        benchmark.reserve_output(output)
    assert paths["reservation"].read_bytes() == before
    occupied = tmp_path / "occupied.json"
    occupied.with_suffix(".npz").write_bytes(b"original")
    with pytest.raises(FileExistsError):
        benchmark.reserve_output(occupied)
    assert occupied.with_suffix(".npz").read_bytes() == b"original"
