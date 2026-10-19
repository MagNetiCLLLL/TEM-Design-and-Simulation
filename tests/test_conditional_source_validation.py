"""Conditional-source bookkeeping; these fixtures do not simulate propagation."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate_conditional_source_convergence import (
    CHANNELS, CASES, arguments, boundary_uncertainty, case_state, c2_margin,
    explicit_probe_table, quadrature_arrays, query_table_arrays, terminal_arrays,
)


def test_global_queries_deduplicate_and_always_retain_every_base_endpoint():
    base = np.arange(20, dtype=float).reshape(4, 5)/20
    data = query_table_arrays(base, np.array([2, 2, 1]), np.array([.3, .3, .4]))
    assert len(data["query_base_index"]) == 10
    assert len(data["anchor_rows"]) == 8
    assert data["requested_rows"][0] == data["requested_rows"][1]
    for i in range(0, len(data["coordinates"]), 2):
        np.testing.assert_array_equal(data["coordinates"][i], data["coordinates"][i+1])
        np.testing.assert_array_equal(data["signs"][i:i+2], [1, -1])
    np.testing.assert_array_equal(data["query_radial_u"][data["anchor_rows"]], np.tile([0., 1.], 4))
    assert not data["weights"].any()
    row = data["requested_rows"][0]
    np.testing.assert_array_equal(data["coordinates"][2*row], [*base[2, :2], .3, *base[2, 2:]])


def test_explicit_boundary_source_preserves_zero_current_role_and_identity():
    data = query_table_arrays(np.full((4, 5), .5), np.array([0]), np.array([.5]))
    table = explicit_probe_table(data, "test actual emission table; no propagation claim")
    assert table.role == "boundary-probes" and table.total == len(data["weights"])
    assert sum(table.weights) == 0.
    np.testing.assert_array_equal(table.ray_ids, data["ray_ids"])


def test_margin_uses_actual_float64_plane_and_zero_weight_stops():
    aperture = SimpleNamespace(key="condenser_aperture_2", radius_mm=.05,
                               offset_x_mm=.01, offset_y_mm=0., z_mm=765.)
    cp = SimpleNamespace(z_mm=np.array([450., 765.]), x_m=np.array([[0., 0.], [.01e-3, .07e-3]]),
        y_m=np.zeros((2, 2)), tx_rad=np.zeros((2, 2)), ty_rad=np.zeros((2, 2)),
        kinetic_energy_ev=np.full((2, 2), 300000.), flight_time_s=np.full((2, 2), 1e-8))
    branch = SimpleNamespace(source_ray_id=np.array([2, 3]), ray_weight=np.zeros(2),
        alive=np.array([True, False]), blocked_key=["", aperture.key],
        blocked_z=np.array([np.nan, 765.]), x=np.full((2, 2), 999., dtype=np.float32))
    result = SimpleNamespace(simulation=SimpleNamespace(section_checkpoint=SimpleNamespace(
        segments=[SimpleNamespace(checkpoints=cp, branch=branch)])))
    arrays = terminal_arrays(result, 765.)
    np.testing.assert_allclose(c2_margin(arrays, aperture), [.05, -.01])
    arrays["blocked_key"][1] = "gun_stop"
    with pytest.raises(ValueError, match="upstream"):
        c2_margin(arrays, aperture)
    with pytest.raises(AssertionError, match="exact plane"):
        terminal_arrays(result, 764.)


@pytest.mark.parametrize("indices,radial", [
    ([-1], [.5]), ([4], [.5]), ([1.], [.5]), ([0, 1], [.5]),
    ([0], [float("nan")]), ([0], [-.1]), ([0], [1.1]),
])
def test_query_table_rejects_invalid_node_mapping(indices, radial):
    with pytest.raises(ValueError):
        query_table_arrays(np.full((4, 5), .5), np.asarray(indices), np.asarray(radial))


def test_query_cache_keys_preserve_adjacent_float_radial_nodes_and_all_base_anchors():
    a, b = .5, np.nextafter(.5, 1.)
    data = query_table_arrays(np.full((4, 5), .5), np.array([0, 0]), np.array([a, b]))
    assert data["requested_rows"][0] != data["requested_rows"][1]
    assert np.array_equal(data["query_radial_u"][data["requested_rows"]], [a, b])
    assert np.array_equal(data["ray_ids"], np.arange(len(data["weights"])))


def test_final_quadrature_retains_all_gauss_measure_and_zero_current_endpoint_anchors():
    from scripts.source_radial_quadrature import GaussTable
    base = np.arange(20, dtype=float).reshape(4, 5)/20
    # Each base gets the full [0,1] measure; these different radial locations
    # are bookkeeping fixtures, not a claim of actual aperture transmission.
    positive = GaussTable(base_index=np.repeat(np.arange(4), 2), cell_index=np.zeros(8, dtype=int),
        angular_cdf=np.tile([.2, .8], 4), weight=np.full(8, 1/8),
        expected_pass=np.arange(64).reshape(8, 8)%3 == 0, diagnostics={})
    data = quadrature_arrays(base, positive)
    assert len(data["weights"]) == 32
    assert data["weights"].sum() == 1
    assert np.array_equal(data["weights"][:16], np.repeat(positive.weight/2, 2))
    assert np.array_equal(data["weights"][16:], np.zeros(16))
    assert np.array_equal(data["cell_index"][16:], np.full(16, -1))
    assert np.array_equal(data["expected_pass"], positive.expected_pass)
    assert np.array_equal(data["signs"], np.tile([1, -1], 16))
    for node in range(8):
        expected = np.r_[base[positive.base_index[node], :2], positive.angular_cdf[node],
                         base[positive.base_index[node], 2:]]
        assert np.array_equal(data["coordinates"][2*node:2*node+2], [expected, expected])
    for b in range(4):
        assert data["weights"][data["base_index"] == b].sum() == .25
    assert np.array_equal(data["angular_cdf"][16:], np.repeat(np.tile([0., 1.], 4), 2))


@pytest.mark.parametrize("damage,error", [
    ("float32", "float64"), ("length", "correspondence"), ("nonterminal", "exact plane"),
])
def test_terminal_reader_rejects_display_precision_and_misaligned_solver_arrays(damage, error):
    cp = SimpleNamespace(z_mm=np.array([765.]), **{k: np.zeros((1, 3), dtype=np.float64) for k in
        ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s")})
    branch = SimpleNamespace(source_ray_id=np.arange(3), ray_weight=np.zeros(3), alive=np.ones(3, bool),
        blocked_key=["", "", ""], blocked_z=np.full(3, np.nan))
    if damage == "float32":
        cp.x_m = cp.x_m.astype(np.float32)
    elif damage == "length":
        branch.ray_weight = np.zeros(2)
    else:
        cp.z_mm[-1] = 765.0001
    result = SimpleNamespace(simulation=SimpleNamespace(section_checkpoint=SimpleNamespace(
        segments=[SimpleNamespace(checkpoints=cp, branch=branch)])))
    with pytest.raises(AssertionError, match=error):
        terminal_arrays(result, 765.)


@pytest.mark.parametrize("damage,exception,match", [
    ("classification", AssertionError, "disagrees"),
    ("stop_plane", AssertionError, "physical plane"),
    ("nonfinite", ValueError, "finite"),
])
def test_margin_requires_actual_physical_clipping_and_plane_agreement(damage, exception, match):
    aperture = SimpleNamespace(key="condenser_aperture_2", z_mm=765., radius_mm=.05,
                              offset_x_mm=.01, offset_y_mm=0.)
    data = dict(x_m=np.array([0., .1])*1e-3, y_m=np.zeros(2), alive=np.array([True, False]),
        blocked_key=np.array(["", aperture.key]), blocked_z_mm=np.array([np.nan, 765.]))
    if damage == "classification":
        data["alive"][1] = True
    elif damage == "stop_plane":
        data["blocked_z_mm"][1] = 764.
    else:
        data["x_m"][0] = np.nan
    with pytest.raises(exception, match=match):
        c2_margin(data, aperture)


def test_case_state_changes_only_requested_c2_aberration_and_preserves_projector():
    from copy import deepcopy
    original = SimpleNamespace(lenses=[SimpleNamespace(key="condenser_lens_2", percent=23.23, cs_mm=8.),
        SimpleNamespace(key="diffraction_lens", percent=44., cs_mm=3.)],
        lens_intrinsic_aberrations={"condenser_lens_2": {"a1_mm": .2, "b2_mm": -.3}})
    snapshot = SimpleNamespace(restore=lambda: deepcopy(original))
    out = case_state(snapshot, "A1_plus0.03mm", {"diffraction_lens":44.}, 8.)
    assert out.lens_intrinsic_aberrations["condenser_lens_2"] == {"a1_mm":.23, "b2_mm":-.3}
    assert original.lens_intrinsic_aberrations["condenser_lens_2"]["a1_mm"] == .2
    assert out.lenses[0].cs_mm == original.lenses[0].cs_mm
    with pytest.raises(AssertionError, match="projector controls changed"):
        case_state(snapshot, "baseline", {"diffraction_lens":45.}, 8.)


def test_observed_anchor_error_is_scaled_by_actual_two_sided_root_slope():
    c = CHANNELS.index("baseline:+1")
    cache = {(0, .49): np.full(len(CHANNELS), -.02), (0, .51): np.full(len(CHANNELS), .02)}
    solution = SimpleNamespace(roots=[dict(base_index=0, channel_index=c, channel_key=CHANNELS[c],
        u=.5, bracket_width=1e-10)], evaluator=SimpleNamespace(cache=cache), partitions=[object()],
        diagnostics={"angular_cdf_tolerance":1e-8})
    tracer = SimpleNamespace(anchor_drift=np.full((2, len(CHANNELS)), 2e-10))
    report = boundary_uncertainty(solution, tracer)
    assert report["roots"][0]["local_slope_mm_per_cdf"] == pytest.approx(2.)
    assert report["roots"][0]["anchor_cdf_uncertainty_estimate"] == pytest.approx(1e-10)
    assert report["roots"][0]["bracket_plus_anchor_estimate"] == pytest.approx(2e-10)
    assert "not a bound" in report["scope"]
    assert report["anchor_drift_within_requested_u_tol"]


@pytest.mark.parametrize("argv", [["--base-count", "3"], ["--base-count", "12"],
    ["--u-tol", "0"], ["--step", "0"], ["--seed", "-1"], ["--scan-points", "2"]])
def test_invalid_protocol_settings_are_rejected_before_physical_execution(argv):
    with pytest.raises(SystemExit):
        arguments(argv)


@pytest.fixture
def mocked_screen_collector(monkeypatch, tmp_path, request):
    """Real emitter/table/file bookkeeping, explicitly mocked transport only."""
    from copy import deepcopy
    import scripts.validate_conditional_source_convergence as module
    from scripts.source_radial_quadrature import GaussTable
    from temsim.optics.electron_gun.emitter import ColdFieldEmitter
    import temsim.instrument_snapshot as snapshots
    import temsim.particle_section_io as section_io
    import temsim.simulation_pipeline as pipeline
    independent_count, orbit_size = 4, getattr(request, "param", 2)
    base = np.full((independent_count, 5), .5)
    if orbit_size != 2:
        from scripts.source_azimuth_orbit import expand_common_azimuth
        base = expand_common_azimuth(base, orbit_size).base_points
    expanded_count = len(base)
    angular = np.tile([.2, .8], expanded_count)
    table = GaussTable(base_index=np.repeat(np.arange(expanded_count), 2),
        cell_index=np.tile([0, 1], expanded_count),
        angular_cdf=angular, weight=np.full(2*expanded_count, 1/(2*expanded_count)),
        expected_pass=np.repeat((angular < .7)[:, None], len(CHANNELS), axis=1), diagnostics={})
    state = SimpleNamespace(electron_gun=SimpleNamespace(emitter=ColdFieldEmitter(0., 1., 1.),
        exit_plane_z_mm=450., trace_step_mm=.2, drift_step_mm=2.), lenses=[SimpleNamespace(key="condenser_lens_2", cs_mm=8., percent=23.23),
        SimpleNamespace(key="diffraction_lens", cs_mm=3., percent=44.)],
        lens_intrinsic_aberrations={}, fluorescent_screen=SimpleNamespace(z_mm=1500.),
        condenser_aperture_2=SimpleNamespace(key="condenser_aperture_2", z_mm=765.,
                                           radius_mm=.05, offset_x_mm=0., offset_y_mm=0.))

    class Snapshot:
        def __init__(self, value):
            self.state = deepcopy(value)
            record = value.electron_gun.emitter.cdf_quadrature
            self.digest = module.canonical_digest(dict(table=None if record is None else record.digest,
                cs=value.lenses[0].cs_mm, intrinsic=value.lens_intrinsic_aberrations,
                projector=value.lenses[1].percent))

        def restore(self):
            return deepcopy(self.state)

        def to_dict(self):
            return {"mock_transport_only": True, "digest": self.digest}

    monkeypatch.setattr(snapshots, "capture_instrument_snapshot", Snapshot)
    calls, archives = [], {}

    def calculate(run_state, target, existing_result=None, **_kwargs):
        calls.append(float(target))
        emitter = run_state.electron_gun.emitter
        emitted = emitter.emit()  # Execute the production source, not a synthetic source override.
        u = np.array(emitter.cdf_quadrature.coordinates)[:, 2]
        margin = .04*(.7-u)
        passed = margin >= 0
        blocked = np.where(passed, "", "condenser_aperture_2")
        alive = passed.copy()
        if target == 1500.:
            # One independently stopped downstream ray must reduce exposure,
            # not be renormalized into the other survivors.
            blocked[0] = "downstream_stop"
            alive[0] = False
        original = dict(gun_exit_ray_id=emitted.ray_id, gun_exit_weight=emitted.weight,
                        gun_exit_alive=np.ones(len(u), dtype=bool))
        end = dict(x_m=(.05-margin)*1e-3, y_m=np.zeros(len(u)), tx_rad=emitted.tx_rad,
            ty_rad=emitted.ty_rad, kinetic_energy_ev=np.full(len(u), 300000.),
            flight_time_s=np.full(len(u), 1e-8), source_id=emitted.ray_id,
            weight=emitted.weight, alive=alive, blocked_key=blocked,
            blocked_z_mm=np.where(passed, np.nan, 765.))
        cp = SimpleNamespace(z_mm=np.array([target]), **{name: values[None, :] for name, values in end.items()
            if name in ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s")})
        branch = SimpleNamespace(source_ray_id=emitted.ray_id, ray_weight=emitted.weight,
            alive=alive, blocked_key=blocked, blocked_z=end["blocked_z_mm"])
        simulation = SimpleNamespace(gun_trace=SimpleNamespace(emitted_current_a=1e-5),
            metrics={"section_gun_reused":existing_result is not None},
            section_checkpoint=SimpleNamespace(segments=[SimpleNamespace(checkpoints=cp, branch=branch)]))
        return SimpleNamespace(source=original, end=end, simulation=simulation)

    def save(result, path, **_kwargs):
        path.write_bytes(b"mock transport archive; never scientific evidence")
        archives[str(path)] = result

    monkeypatch.setattr(pipeline, "calculate_particle_section", calculate)
    monkeypatch.setattr(section_io, "save_section_result", save)
    monkeypatch.setattr(section_io, "load_section_result", lambda path: archives[str(path)])
    monkeypatch.setattr(module, "source_arrays", lambda result: result.source)
    monkeypatch.setattr(module, "screen_arrays", lambda result, *_args: dict(result.end, survivor=result.end["alive"]))
    tracer = SimpleNamespace(snapshot=Snapshot(state), base_points=base, output=tmp_path,
        manifest=dict(protocol_digest="mock-protocol", implementation="mock-transport", sources={}, runs={},
                      base_count=independent_count, expanded_base_count=expanded_count),
        controls={"diffraction_lens":44.}, c2_cs=8., guard=lambda:None,
        anchor_reference=None, anchor_drift=None)
    tracer.observe_anchors = lambda anchors: module.PhysicalBoundaryTracer.observe_anchors(tracer, anchors)
    if orbit_size != 2:
        tracer.manifest["azimuth_orbit_size"] = orbit_size
    return module, tracer, table, calls


def test_final_collector_preserves_raw_weights_real_table_and_downstream_loss(mocked_screen_collector):
    module, tracer, table, calls = mocked_screen_collector
    module.collect_screen_group(tracer, None, table, 2, "partition-id")
    assert len(calls) == 9  # one gun and C2+screen for each of four cases
    assert len(tracer.manifest["runs"]) == 4
    for row in tracer.manifest["runs"].values():
        assert row["screen_probability"] == .4375
        assert row["actual_c2_membership_matches"]
        with np.load(tracer.output/row["particles_path"], allow_pickle=False) as data:
            assert data["weight"].sum() == 1.
            assert np.count_nonzero(data["weight"] == 0) == 16
            assert data["weight"][data["survivor"]].sum() == .4375


def test_complete_resume_reuses_verified_transport_and_recovers_final_anchor_drift(mocked_screen_collector):
    module, tracer, table, calls = mocked_screen_collector
    module.collect_screen_group(tracer, None, table, 2, "partition-id")
    tracer.anchor_reference = tracer.anchor_reference+1e-6
    tracer.anchor_drift[:] = 0
    module.collect_screen_group(tracer, None, table, 2, "partition-id")
    assert len(calls) == 9
    assert np.allclose(tracer.anchor_drift, 1e-6, rtol=0, atol=1e-17)
    assert tracer.manifest["sources"]["M4_G2"]["final_anchor_max_drift_mm"] == pytest.approx(1e-6)


@pytest.mark.parametrize("artifact", ["screen", "c2", "source_snapshot", "source_archive", "table", "protocol"])
def test_resume_rejects_tampered_evidence_before_new_transport(mocked_screen_collector, artifact):
    module, tracer, table, calls = mocked_screen_collector
    module.collect_screen_group(tracer, None, table, 2, "partition-id")
    run = next(iter(tracer.manifest["runs"].values()))
    source = tracer.manifest["sources"]["M4_G2"]
    key = {"screen":run["particles_path"], "c2":run["c2_path"],
           "source_snapshot":source["snapshot_path"], "source_archive":source["archive_path"],
           "table":source["table_path"]}.get(artifact)
    if key is not None:
        with (tracer.output/key).open("ab") as f:
            f.write(b"changed")
    else:
        tracer.manifest["protocol_digest"] = "changed-protocol"
    with pytest.raises(ValueError, match="changed"):
        module.collect_screen_group(tracer, None, table, 2, "partition-id")
    assert len(calls) == 9


def test_tracer_guard_rejects_production_implementation_change(monkeypatch):
    import scripts.validate_conditional_source_convergence as module
    import temsim.calculation_manifest as manifests
    monkeypatch.setattr(manifests, "solver_source_identity", lambda:"new-code")
    fake = SimpleNamespace(manifest={"implementation":"original-code"})
    with pytest.raises(RuntimeError, match="implementation changed"):
        module.PhysicalBoundaryTracer.guard(fake)


def _orbit_gauss_table(expanded_count):
    from scripts.source_radial_quadrature import GaussTable
    angular = np.tile([.2, .8], expanded_count)
    return GaussTable(base_index=np.repeat(np.arange(expanded_count), 2),
        cell_index=np.zeros(2*expanded_count, dtype=int), angular_cdf=angular,
        weight=np.full(2*expanded_count, 1/(2*expanded_count)),
        expected_pass=np.repeat((angular < .7)[:, None], len(CHANNELS), axis=1),
        diagnostics={})


@pytest.mark.parametrize("k", [2, 8, 16])
def test_orbit_collector_preserves_independent_measure_all_signs_and_zero_anchors(k):
    from scipy.stats import qmc
    from scripts.source_azimuth_orbit import expand_common_azimuth
    from temsim.optics.electron_gun.emitter import ColdFieldEmitter, EmissionCDFQuadrature
    independent = qmc.Sobol(5, scramble=True, seed=31, bits=30).random_base2(2)
    orbit = expand_common_azimuth(independent, k)
    expanded_count = len(independent)*k//2
    data = quadrature_arrays(orbit.base_points, _orbit_gauss_table(expanded_count))
    positive = data["weights"] > 0
    assert positive.sum() == 4*expanded_count
    assert (~positive).sum() == 4*expanded_count
    np.testing.assert_array_equal(data["signs"], np.tile([1, -1], len(data["weights"])//2))
    np.testing.assert_array_equal(data["coordinates"][::2], data["coordinates"][1::2])
    np.testing.assert_array_equal(data["angular_cdf"][~positive],
                                  np.repeat(np.tile([0., 1.], expanded_count), 2))
    parent = orbit.independent_base_index[data["base_index"]]
    np.testing.assert_allclose(np.bincount(parent, weights=data["weights"]),
                               np.full(len(independent), 1/len(independent)), rtol=0, atol=1e-15)
    assert data["weights"].sum() == pytest.approx(1., abs=1e-15)
    emitter = ColdFieldEmitter(0., 1., 1.)
    emitter.cdf_quadrature = EmissionCDFQuadrature.from_arrays(data["coordinates"], data["weights"],
        signs=data["signs"], ray_ids=data["ray_ids"], construction="Analytical source-only orbit fixture")
    emitter.ray_count = emitter.cdf_quadrature.total
    emitted = emitter.emit()
    np.testing.assert_array_equal(emitted.weight, data["weights"])
    np.testing.assert_array_equal(emitted.ray_id, data["ray_ids"])
    np.testing.assert_array_equal(emitted.energy_offset_ev[::2], emitted.energy_offset_ev[1::2])
    for field in ("x_m", "y_m", "tx_rad", "ty_rad"):
        values = getattr(emitted, field)
        np.testing.assert_array_equal(values[::2], -values[1::2])


def test_orbit_actual_source_map_cancels_harmonics_but_retains_real_shift_and_loss():
    from scripts.source_azimuth_orbit import expand_common_azimuth
    from temsim.optics.electron_gun.emitter import ColdFieldEmitter
    from temsim.optics.electron_gun.tip_joint_sampling import phase_space_from_quantiles
    independent = np.array([[.6, .07, .21, .3, .7], [.4, .17, .33, .7, .3],
                            [.7, .23, .61, .4, .6], [.3, .41, .81, .6, .4]])
    orbit = expand_common_azimuth(independent, 8)
    data = quadrature_arrays(orbit.base_points, _orbit_gauss_table(len(orbit.base_points)))
    emitter = ColdFieldEmitter(0., 1., 1.)
    x, y, tx, ty, energy = phase_space_from_quantiles(emitter, data["coordinates"])
    sign, weight = data["signs"], data["weights"]
    x, y, tx, ty = (value*sign for value in (x, y, tx, ty))
    unit_direction = (tx+1j*ty)/np.maximum(np.hypot(tx, ty), np.finfo(float).tiny)
    for harmonic in (2, 4, 6):
        assert abs(weight @ unit_direction**harmonic) < 2e-15
    # This analytical, non-equivariant map models a true scalar optical shift.
    # It is not a real column simulation or an assertion of its convergence.
    source_x, source_y = x/1e-9, y/1e-9
    radial_angle2 = (tx*1e3)**2+(ty*1e3)**2
    actual_x, actual_y = source_x+.7*radial_angle2, source_y-.23
    expected_centroid = np.array([.7*(weight @ radial_angle2), -.23])
    np.testing.assert_allclose([weight @ actual_x, weight @ actual_y], expected_centroid,
                               rtol=0, atol=2e-15)
    assert expected_centroid[0] > 0
    fixed_acceptance = actual_x > expected_centroid[0]+.2
    transmitted = weight[fixed_acceptance].sum()
    assert 0 < transmitted < 1
    assert weight.sum() == 1
    # Do not rotate the output, remove its physical mean or normalize survivors.
    measured_centroid = weight[fixed_acceptance] @ actual_x[fixed_acceptance]/transmitted
    assert measured_centroid > expected_centroid[0]+.2


@pytest.fixture
def prepared_orbit_collector(monkeypatch, tmp_path):
    """Exercise real prepare/archive/protocol code; no physical propagation."""
    import json
    import os
    # main() changes CLI process settings. Restore original values, including
    # absent keys, before later tests use an already initialized Numba pool.
    for name in ('TEMSIM_CPU_THREADS', 'NUMBA_NUM_THREADS', 'OMP_NUM_THREADS',
                 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'QT_QPA_PLATFORM'):
        monkeypatch.setenv(name, os.environ.get(name, 'offscreen' if name == 'QT_QPA_PLATFORM' else '1'))
    import scripts.validate_conditional_source_convergence as module
    import temsim.instrument_snapshot as snapshots
    import temsim.calculation_manifest as manifests
    screen = SimpleNamespace(z_mm=1500., point_spread_model="gaussian",
        point_spread_sigma_x_mm=.6, point_spread_sigma_y_mm=.6, point_spread_rotation_deg=0.,
        point_spread_status="provisional_model_parameter", point_spread_source="Test fixture only")
    state = SimpleNamespace(electron_gun=SimpleNamespace(emitted_current_a=1e-5, trace_step_mm=.2, drift_step_mm=2.),
        condenser_aperture_2=SimpleNamespace(z_mm=765.), fluorescent_screen=screen)
    snapshot = SimpleNamespace(digest="unchanged-optical-input", to_dict=lambda:{"fixture":True})
    monkeypatch.setattr(module, "configure_inputs", lambda *_args:(state, {"diffraction_lens":44.}, 8.))
    monkeypatch.setattr(snapshots, "capture_instrument_snapshot", lambda _state:snapshot)
    monkeypatch.setattr(manifests, "solver_source_identity", lambda:"unchanged-implementation")
    parse = module.arguments

    def prepare(name, *, m=4, k=8, resume=False):
        output = tmp_path/name
        cli = ["--output", str(output), "--stage", "prepare", "--base-count", str(m),
               "--azimuth-orbit-size", str(k), "--backend", "CPU", "--threads", "1"]
        if resume:
            cli.append("--resume")
        args = parse(cli)
        monkeypatch.setattr(module, "arguments", lambda:args)
        module.main()
        return output, json.loads((output/"manifest.json").read_text())

    return module, prepare


def test_prepare_tracks_independent_sobol_prefix_separately_from_expanded_orbit(prepared_orbit_collector):
    module, prepare = prepared_orbit_collector
    small_path, small = prepare("small", m=4, k=8)
    large_path, large = prepare("large", m=8, k=8)
    assert small["base_count"] == 4 and small["expanded_base_count"] == 16
    assert large["base_count"] == 8 and large["expanded_base_count"] == 32
    for field in ("base_points_path", "expanded_base_points_path"):
        with np.load(small_path/small[field], allow_pickle=False) as a, np.load(large_path/large[field], allow_pickle=False) as b:
            np.testing.assert_array_equal(a["points"], b["points"][:len(a["points"])])
    with np.load(small_path/small["azimuth_orbit_map_path"], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["independent_base_index"], np.repeat(np.arange(4), 4))
        np.testing.assert_array_equal(data["half_orbit_index"], np.tile(np.arange(4), 4))
    assert small["physical_protocol_digest"] == large["physical_protocol_digest"]
    assert small["protocol_digest"] != large["protocol_digest"]
    assert small["protocol"]["azimuth_orbit_schema"] == small["azimuth_orbit"]["schema"]
    assert small["physical_protocol"]["azimuth_orbit_schema"] == small["azimuth_orbit"]["schema"]
    assert "source_azimuth_orbit.py" in small["protocol"]["helper_sha256"]
    for path_key, sha_key in (("base_points_path", "base_points_sha256"),
                             ("expanded_base_points_path", "expanded_base_points_sha256"),
                             ("azimuth_orbit_map_path", "azimuth_orbit_map_sha256")):
        assert module.digest_file(small_path/small[path_key]) == small[sha_key]


def test_orbit_refinement_changes_request_identity_without_changing_physical_input(prepared_orbit_collector):
    module, prepare = prepared_orbit_collector
    _, k2 = prepare("k2", k=2)
    _, k8 = prepare("k8", k=8)
    assert k2["physical_protocol_digest"] == k8["physical_protocol_digest"]
    assert k2["protocol_digest"] != k8["protocol_digest"]
    assert module.group_label(4, 2, 2) == "M4_G2"
    assert module.group_label(4, 8, 2) == "M4_K8_G2"
    assert module.group_label(4, 16, 4) == "M4_K16_G4"
    prepare("k8", k=8, resume=True)
    with pytest.raises(ValueError):
        prepare("k8", k=16, resume=True)


@pytest.mark.parametrize("field,value", [("base_count", 8), ("expanded_base_count", 32),
    ("azimuth_orbit_size", 16), ("azimuth_orbit", {"schema":"changed-method"})])
def test_resume_rejects_relabelled_orbit_counts_and_method(prepared_orbit_collector, field, value):
    module, prepare = prepared_orbit_collector
    output, record = prepare("changed")
    record[field] = value
    module.write_json(output/"manifest.json", record)
    with pytest.raises(ValueError):
        prepare("changed", resume=True)


@pytest.mark.parametrize("field", ["expanded_base_points_path", "azimuth_orbit_map_path"])
def test_resume_rejects_changed_expanded_orbit_evidence(prepared_orbit_collector, field):
    _module, prepare = prepared_orbit_collector
    output, record = prepare("damaged")
    with (output/record[field]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        prepare("damaged", resume=True)


@pytest.mark.parametrize("mocked_screen_collector", [8, 16], indirect=True)
def test_expanded_collector_keeps_independent_counts_weights_and_actual_loss(mocked_screen_collector):
    module, tracer, table, calls = mocked_screen_collector
    k = tracer.manifest["azimuth_orbit_size"]
    expanded_count = 4*k//2
    module.collect_screen_group(tracer, None, table, 2, "partition-id")
    group = f"M4_K{k}_G2"
    source = tracer.manifest["sources"][group]
    assert len(calls) == 9
    assert source["base_count"] == 4
    assert source["expanded_base_count"] == expanded_count
    assert source["azimuth_orbit_size"] == k
    assert source["total_probability"] == 1.
    with np.load(tracer.output/source["table_path"], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["independent_base_index"], data["base_index"]//(k//2))
        np.testing.assert_array_equal(data["half_orbit_index"], data["base_index"]%(k//2))
        np.testing.assert_allclose(np.bincount(data["independent_base_index"], weights=data["weights"]),
                                   np.full(4, .25), rtol=0, atol=1e-15)
        original_weights = data["weights"].copy()
    for run_id, row in tracer.manifest["runs"].items():
        assert run_id.startswith(group+"_")
        assert row["base_count"] == 4
        assert row["expanded_base_count"] == expanded_count
        assert row["screen_probability"] == .5-1/(4*expanded_count)
        with np.load(tracer.output/row["particles_path"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["weight"], original_weights)
            assert np.count_nonzero(data["weight"] == 0) == 4*expanded_count
    module.collect_screen_group(tracer, None, table, 2, "partition-id")
    assert len(calls) == 9
