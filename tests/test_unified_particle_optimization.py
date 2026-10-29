"""Contact acceleration parity against the pre-optimization material law.

These are bounded particle-only fixtures. A contact-fraction tolerance covers
floating-point ordering, never a changed blocked population or material host.
"""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.posed_wave_aperture import PosedWaveAperture
from temsim.physics.posed_wave_hardware import PosedWaveBore
from temsim.physics import unified_field_particle as particle


def reference_contact(pieces, before, after):
    """Original NumPy contact composition, independent of accelerated dispatch."""
    fractions = np.full(len(before), np.inf)
    labels = np.full(len(before), "", dtype=object)
    for key, piece in pieces:
        trial = PosedWaveAperture.first_contact_fraction(piece, before*1000, after*1000)
        outside = ~piece.transmission_mask(before[:, 0]*1000, before[:, 1]*1000, before[:, 2]*1000)
        trial = np.where(outside, 0., trial)
        selected = trial < fractions
        fractions[selected], labels[selected] = trial[selected], key
    return fractions, labels


def registration(angle=0., origin=(0., 0., 0.)):
    cs, sn = np.cos(angle), np.sin(angle)
    return CoordinateRegistration(origin, ((cs, 0., sn), (0., 1., 0.), (-sn, 0., cs)))


def global_points(local_mm, frame):
    return np.asarray(local_mm) @ frame.rotation_array.T*.001 + frame.origin_array_m


def assert_contacts_equal(pieces, start, stop):
    expected, expected_names = reference_contact(pieces, start, stop)
    actual, names = particle._contact(pieces, start, stop)
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    np.testing.assert_array_equal(names, expected_names)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    return actual, names


@pytest.mark.parametrize("angle", [0., .13, -.47])
@pytest.mark.parametrize("primitive", [PosedWaveAperture, PosedWaveBore])
def test_contact_random_chords_keep_blocked_population_and_first_host(angle, primitive):
    frame = registration(angle, (1.7e-5, -2.1e-5, .003))
    piece = primitive("opening", 450., 452., .03, frame)
    other = primitive("shoulder", 451., 452.5, .04, frame)
    rng = np.random.default_rng(41802)
    start = np.c_[rng.uniform(-.07, .07, (97, 2)), rng.uniform(448., 454., 97)]
    stop = start + rng.uniform((-.09, -.09, -4.), (.09, .09, 4.), (97, 3))
    assert_contacts_equal((("opening-host", piece), ("shoulder-host", other)),
                          global_points(start, frame), global_points(stop, frame))


@pytest.mark.parametrize("primitive,blocked", [(PosedWaveAperture, False), (PosedWaveBore, True)])
def test_aperture_and_bore_keep_distinct_exact_radius_entrance_rules(primitive, blocked):
    piece = primitive("edge", 0., 2., 1., CoordinateRegistration())
    # Exactly one millimetre; no pose roundoff enters this boundary decision.
    position = np.array([[.001, 0., .001]])
    fraction, labels = assert_contacts_equal((("edge-host", piece),), position, position)
    assert np.isfinite(fraction[0]) == blocked
    assert labels[0] == ("edge-host" if blocked else "")


def test_zero_opening_endpoint_contacts_reverse_chords_and_equal_host_ties():
    frame = CoordinateRegistration()
    closed = PosedWaveAperture("closed", 1., 2., 0., frame)
    start = np.array([[0., 0., 0.], [0., 0., .003], [0., 0., .001]])
    stop = np.array([[0., 0., .001], [0., 0., .002], [0., 0., .001]])
    fractions, labels = assert_contacts_equal((("first", closed), ("same-place-second", closed)), start, stop)
    np.testing.assert_array_equal(fractions, [1., 1., 0.])
    np.testing.assert_array_equal(labels, ["first", "first", "first"])


def test_contact_empty_batch_empty_geometry_and_nonfinite_chords_keep_reference_semantics():
    piece = PosedWaveBore("tube", 0., 2., 1., CoordinateRegistration())
    assert_contacts_equal((("tube", piece),), np.empty((0, 3)), np.empty((0, 3)))
    assert_contacts_equal((), np.zeros((2, 3)), np.ones((2, 3)))
    start = np.array([[np.nan, 0., .001], [0., 0., np.inf], [0., 0., .001]])
    stop = np.array([[0., 0., .001], [0., 0., .001], [0., np.nan, .001]])
    with np.errstate(invalid="ignore"):
        assert_contacts_equal((("tube", piece),), start, stop)


def test_nearly_parallel_tilted_slab_keeps_original_axial_threshold():
    angle = .35
    frame = registration(angle)
    piece = PosedWaveAperture("slab", 1., 2., .1, frame)
    start = np.array([[.02, 0., .99], [.02, 0., 1.], [.02, 0., 2.]])
    for local_dz in (0., .5e-14, 2e-14, -2e-14):
        stop = start + np.array([.2, 0., local_dz])
        assert_contacts_equal((("slab", piece),), global_points(start, frame), global_points(stop, frame))


@pytest.mark.parametrize("dynamic", [False, True])
def test_short_real_transport_keeps_phase_energy_clock_and_mechanical_stop_state(monkeypatch, dynamic):
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    from temsim.physics.planar_gun_field import PlanarGunField
    from temsim.physics.potential_snapshot import PotentialContribution
    from test_unified_field_particle import controlled, execute
    from test_unified_field_runtime import parked_state

    state = parked_state()
    state.step_mm = .1
    angle = .05
    frame = registration(angle)
    centre = np.array([0., 0., .451])
    frame = registration(angle, tuple(centre-frame.rotation_array@centre))
    aperture = PosedWaveAperture("tilted-iris", 450.8, 451.2, .025, frame)
    bore = PosedWaveBore("wide-bore", 450., 452., .3, CoordinateRegistration())
    radius, z = np.array([0., .02]), np.array([.449, .453])
    phi = np.broadcast_to(200000.+2e6*(z-.45), (2, 2)).copy()
    potential = ElectricPotentialSnapshot.from_provider(PlanarGunField({}, radius, z, phi))
    runtime = controlled(state, (PotentialContribution("electric", ("electric",), potential, "electric"),),
                         (aperture, bore))
    if dynamic:
        from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
        from test_dynamic_potential import coil
        runtime = replace(runtime, snapshot=DynamicPotentialSnapshot(runtime.snapshot, (coil(),)))
    initial = np.array([[0., 0., .001, -.002], [1e-5, -1e-5, -.001, .001],
                        [6e-5, 0., 0., 0.], [4e-4, 0., 0., 0.]])
    energy = np.array([100000., 300000., 200000., 300000.])
    _, actual, actual_energy = execute(state, runtime, initial, monkeypatch, energy=energy, step=.1)
    actual_stops = particle.merge_unified_stops(actual[0], np.ones(4, bool), np.full(4, np.nan), [""]*4)
    monkeypatch.setattr(particle, "_contact", reference_contact)
    _, expected, expected_energy = execute(state, runtime, initial, monkeypatch, energy=energy, step=.1)
    expected_stops = particle.merge_unified_stops(expected[0], np.ones(4, bool), np.full(4, np.nan), [""]*4)
    np.testing.assert_array_equal(actual[0], expected[0])
    for have, want in zip(actual[1:6], expected[1:6]):
        np.testing.assert_allclose(have, want, rtol=2e-14, atol=2e-18, equal_nan=True)
    np.testing.assert_allclose(actual_energy, expected_energy, rtol=2e-14, atol=2e-9, equal_nan=True)
    np.testing.assert_array_equal(actual_stops[0], expected_stops[0])
    np.testing.assert_allclose(actual_stops[1], expected_stops[1], rtol=0., atol=2e-11, equal_nan=True)
    assert actual_stops[2] == expected_stops[2]
    assert np.any(actual_stops[0]) and np.any(~actual_stops[0])
    assert "tilted-iris" in actual_stops[2] and "wide-bore" in actual_stops[2]


def compiled_geometry(rows):
    from temsim.physics.potential_geometry import PotentialBoundaryHost, PotentialGeometrySnapshot
    from temsim.physics.unified_particle_contacts import compile_contact_geometry
    geometry = PotentialGeometrySnapshot(-1., 1., tuple(
        PotentialBoundaryHost(key, (key,), (piece,)) for key, piece in rows))
    return geometry, compile_contact_geometry(geometry)


def test_compiled_geometry_cache_and_active_selection_preserve_tilted_infinite_slabs():
    from temsim.physics.unified_particle_contacts import compile_contact_geometry
    rows = (("near", PosedWaveBore("near", 1., 2., 1., CoordinateRegistration())),
            ("far", PosedWaveBore("far", 100., 102., 1., CoordinateRegistration())),
            ("tilted", PosedWaveBore("tilted", 100., 102., 1., registration(.1))))
    geometry, compiled = compiled_geometry(rows)
    assert compile_contact_geometry(geometry) is compiled
    active = compiled.active(.0015, .002)
    assert [key for key, _ in active] == ["near", "tilted"]
    assert len(active) == 2
    for array in (compiled.origins_mm, compiled.native, active.indices):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_native_aligned_contact_path_runs_without_numpy_primitive(monkeypatch):
    from temsim.physics import unified_particle_contacts as contacts
    if contacts._native_contact is None:
        pytest.skip("Numba acceleration is unavailable; reference fallback remains covered")
    piece = PosedWaveAperture("iris", 1., 2., 1., CoordinateRegistration())
    _, compiled = compiled_geometry((("iris", piece),))
    before = np.array([[.0001, 0., 0.], [.002, 0., 0.], [0., 0., .0015]])
    after = np.array([[.0002, 0., .003], [.002, 0., .003], [.002, 0., .0015]])
    expected, labels = reference_contact(compiled, before, after)

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("Unambiguous aligned boundaries should use native contact")

    monkeypatch.setattr(PosedWaveAperture, "first_contact_fraction", unexpected_fallback)
    actual, actual_labels = contacts.contact(compiled.active(0., .003), before, after)
    np.testing.assert_array_equal(actual_labels, labels)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("primitive", [PosedWaveAperture, PosedWaveBore])
def test_compiled_contact_keeps_ulp_neighbours_of_circle_and_slab_edges(primitive):
    from temsim.physics.unified_particle_contacts import contact
    radius = .037
    piece = primitive("edge", 1., 2., radius, CoordinateRegistration())
    _, compiled = compiled_geometry((("edge", piece),))
    angles = np.linspace(0., 2*np.pi, 17)
    radii = (np.nextafter(radius, 0.), radius, np.nextafter(radius, np.inf))
    points = np.array([(r*np.cos(angle), r*np.sin(angle), z)
        for r in radii for angle in angles
        for z in (np.nextafter(1., 0.), 1., np.nextafter(1., np.inf), 1.5,
                  np.nextafter(2., 0.), 2., np.nextafter(2., np.inf))])*.001
    for start, stop in ((points, points),
                        (points-np.array([0., 0., .001]), points),
                        (points, points+np.array([0., 0., .001]))):
        expected, labels = reference_contact(compiled, start, stop)
        actual, actual_labels = contact(compiled.active(-.001, .004), start, stop)
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
        np.testing.assert_array_equal(actual_labels, labels)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_subclass_fallback_and_native_equal_contacts_keep_original_host_order():
    from temsim.physics.unified_particle_contacts import contact

    class SpecializedBore(PosedWaveBore):
        pass

    frame = CoordinateRegistration()
    native = PosedWaveBore("native", 1., 2., 1., frame)
    special = SpecializedBore("special", 1., 2., 1., frame)
    before, after = np.array([[.002, 0., 0.]]), np.array([[.002, 0., .003]])
    for rows in ((("special", special), ("native", native)),
                 (("native", native), ("special", special))):
        expected, labels = reference_contact(rows, before, after)
        actual, actual_labels = contact(rows, before, after)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_labels, labels)
        assert actual_labels[0] == rows[0][0]


def test_compiled_geometry_keeps_authoritative_fallback_without_numba(monkeypatch):
    from temsim.physics import unified_particle_contacts as contacts
    piece = PosedWaveAperture("fallback", 1., 2., .1, registration(.1))
    _, compiled = compiled_geometry((("fallback", piece),))
    start, stop = np.array([[.0002, 0., .001]]), np.array([[.0002, 0., .002]])
    expected, labels = reference_contact(compiled, start, stop)
    monkeypatch.setattr(contacts, "_native_contact", None)
    actual, actual_labels = contacts.contact(compiled, start, stop)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_labels, labels)


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_input_dtype_promotion_keeps_original_millimetre_conversion(dtype):
    from temsim.physics.unified_particle_contacts import contact
    scale = .001 if dtype is np.float32 else 1.
    piece = PosedWaveAperture("dtype", 450.*scale*1000, 452.*scale*1000,
                              .03*scale*1000, CoordinateRegistration())
    rng = np.random.default_rng(41802)
    start = (np.c_[rng.uniform(-.07, .07, (97, 2)), rng.uniform(448., 454., 97)]*scale).astype(dtype)
    stop = (start+rng.uniform((-.09, -.09, -4.), (.09, .09, 4.), (97, 3))*scale).astype(dtype)
    rows = (("dtype", piece),)
    expected, labels = reference_contact(rows, start, stop)
    actual, actual_labels = contact(rows, start, stop)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_labels, labels)


def test_final_restart_box_contains_every_sample_and_rejected_contact_query(monkeypatch):
    from test_unified_field_particle import controlled, execute
    from test_unified_field_runtime import parked_state

    state = parked_state()
    piece = PosedWaveAperture("iris", 451., 451.1, .015, CoordinateRegistration())
    runtime = controlled(state, pieces=(piece,))
    initial = np.array([[0., 0., .001, -.002], [2e-5, 0., .03, .02]])
    queried = []
    queries_by_node = []
    original_sampler, original_contact = particle._sampler, particle._contact

    def recording_sampler(snapshot):
        sample = original_sampler(snapshot)

        def observed(points, *args):
            queried.append(np.asarray(points).copy())
            # Exact node queries precede checkpoint publication. One-sided
            # RK endpoints must remain separate from this physical plane.
            plane = float(points[0, 2])
            if np.all(points[:, 2] == plane):
                queries_by_node.append((plane, len(queried)))
            return sample(points, *args)

        return observed

    def recording_contact(pieces, before, after):
        queried.extend((before.copy(), after.copy()))
        return original_contact(pieces, before, after)

    monkeypatch.setattr(particle, "_sampler", recording_sampler)
    monkeypatch.setattr(particle, "_contact", recording_contact)
    _, result, _ = execute(state, runtime, initial, monkeypatch, step=.2)
    saved = result[-1].unified_state
    assert saved.active[-1].tolist() == [True, False]
    points = np.concatenate(queried)
    assert np.all(points >= saved.swept_bounds_m[-1, 0])
    assert np.all(points <= saved.swept_bounds_m[-1, 1])
    for row, plane_mm in enumerate(result[-1].z_mm):
        plane = float(plane_mm)*.001
        prefixes = [stop for z, stop in queries_by_node if z == plane]
        assert prefixes
        prefix = np.concatenate((np.array([[0., 0., .45]]), *queried[:max(prefixes)]))
        expected = np.array((prefix.min(axis=0), prefix.max(axis=0)))
        np.testing.assert_array_equal(saved.swept_bounds_m[row], expected)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("history_policy", ["checkpoints", "sampled"])
def test_checkpoint_storage_keeps_every_force_contact_and_complete_restart(monkeypatch, dynamic, history_policy):
    """Storage cannot change field work, a later loss, or the reuse proof."""
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    from temsim.physics.planar_gun_field import PlanarGunField
    from temsim.physics.potential_snapshot import PotentialContribution
    from temsim.physics.unified_particle_restart import STATE_ARRAY_FIELDS
    from test_unified_field_particle import controlled
    from test_unified_field_runtime import parked_state
    from test_unified_particle_prefix import compact

    state = parked_state()
    state.step_mm = .05
    state.history_step_mm = .3
    iris = PosedWaveAperture("iris", 451., 451.1, .015, CoordinateRegistration())
    radius, axial = np.array([0., .02]), np.array([.449, .453])
    phi = np.broadcast_to(200000.+2e6*(axial-.45), (2, 2)).copy()
    electric = ElectricPotentialSnapshot.from_provider(PlanarGunField({}, radius, axial, phi))
    runtime = controlled(state, (compact(1e-6, 450.),
        PotentialContribution("electric", ("electric",), electric, "electric")), (iris,))
    if dynamic:
        from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
        from test_dynamic_potential import coil
        runtime = replace(runtime, snapshot=DynamicPotentialSnapshot(runtime.snapshot, (coil(),)))
    plan = particle.build_unified_plan(state, 450., 452., checkpoint_z_mm=(450., 450.5, 451., 452.),
                                       _captured_runtime=runtime)
    phase = (np.array([1e-6, 3e-5]), np.array([.001, .0]), np.zeros(2), np.zeros(2))
    energy, times = np.array([200000., 300000.]), np.array([2e-9, 2.1e-9])
    counts = {"sample": 0, "contact": 0}
    original_sampler, original_contact = particle._sampler, particle._contact

    def counted_sampler(snapshot):
        sample = original_sampler(snapshot)
        def counted(*args, **kwargs):
            counts["sample"] += 1
            return sample(*args, **kwargs)
        return counted

    def counted_contact(*args):
        counts["contact"] += 1
        return original_contact(*args)

    monkeypatch.setattr(particle, "_sampler", counted_sampler)
    monkeypatch.setattr(particle, "_contact", counted_contact)
    full_energy, sparse_energy = [], []
    full = particle.execute_unified_plan(state, plan, *phase, initial_time_s=times,
        initial_kinetic_energy_ev=energy, return_flight_times=True, energy_output=full_energy)
    full_counts = counts.copy()
    counts.update(sample=0, contact=0)
    sparse = particle.execute_unified_plan(state, plan, *phase, initial_time_s=times,
        initial_kinetic_energy_ev=energy, return_flight_times=True, energy_output=sparse_energy,
        history_policy=history_policy)
    assert counts == full_counts
    assert len(sparse[0]) < len(full[0])
    rows = np.searchsorted(full[0], sparse[0])
    for have, expected in zip(sparse[1:6], full[1:6]):
        np.testing.assert_array_equal(have, expected[rows])
    np.testing.assert_array_equal(sparse_energy[0], full_energy[0][rows])
    for name in ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev"):
        np.testing.assert_array_equal(getattr(sparse[-1], name), getattr(full[-1], name))
    for name in STATE_ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(sparse[-1].unified_state, name),
                                      getattr(full[-1].unified_state, name))
    assert sparse[-1].unified_state.spatial_dependency_identities == full[-1].unified_state.spatial_dependency_identities
    cp = sparse[-1]
    assert cp.unified_state.active[1].tolist() == [True, True]
    assert cp.unified_state.active[-1].tolist() == [True, False]
    assert cp.unified_state.blocked_key == ("", "iris")
    assert np.isfinite(cp.kinetic_energy_ev[2, 1])  # Exact material-face energy.
    row = 1
    resumed_energy = []
    resumed = particle.execute_unified_plan(state, plan,
        *(getattr(cp, key)[row] for key in ("x_m", "tx_rad", "y_m", "ty_rad")),
        start_index=int(np.searchsorted(plan.z_mm, cp.z_mm[row])), unified_resume=(cp, row),
        initial_time_s=cp.flight_time_s[row], initial_kinetic_energy_ev=cp.kinetic_energy_ev[row],
        return_flight_times=True, energy_output=resumed_energy, history_policy=history_policy)
    retained_start = np.searchsorted(sparse[0], cp.z_mm[row])
    for have, expected in zip(resumed[1:6], sparse[1:6]):
        np.testing.assert_array_equal(have, expected[retained_start:])
    np.testing.assert_array_equal(resumed_energy[0], sparse_energy[0][retained_start:])


def test_sparse_history_keeps_unrequested_receiver_plane_and_full_default(monkeypatch):
    from temsim.physics.recording_clipping import clip_recording_planes
    from test_unified_field_particle import controlled
    from test_unified_field_runtime import parked_state

    state = parked_state()
    state.step_mm = .1
    receiver = state.recording_planes[0]
    receiver.z_mm = 450.73
    receiver.geometry, receiver.inner_diameter_mm, receiver.outer_width_mm = "disk", 0., .02
    runtime = controlled(state)
    plan = particle.build_unified_plan(state, 450., 452., checkpoint_z_mm=(450., 452.),
                                       _captured_runtime=runtime)
    phase = (np.array([0., 2e-5]), np.zeros(2), np.zeros(2), np.zeros(2))
    full = particle.execute_unified_plan(state, plan, *phase, initial_time_s=np.zeros(2))
    sparse = particle.execute_unified_plan(state, plan, *phase, initial_time_s=np.zeros(2),
                                           history_policy="checkpoints")
    np.testing.assert_array_equal(full[0], plan.z_mm)
    predecessor = plan.z_mm[np.searchsorted(plan.z_mm, receiver.z_mm)-1]
    np.testing.assert_array_equal(sparse[0], [450., predecessor, receiver.z_mm, 452.])
    assert receiver.z_mm not in sparse[-1].z_mm
    stopped = []
    for result in (full, sparse):
        stopped.append(clip_recording_planes(state, result[0], result[1], result[3],
            np.ones(2, bool), np.full(2, np.nan), ["", ""]))
    for have, expected in zip(stopped[1], stopped[0]):
        np.testing.assert_array_equal(have, expected)
    np.testing.assert_array_equal(stopped[1][0], [False, True])
    assert stopped[1][1][0] == receiver.z_mm
    sparse_budget = particle.unified_history_memory_bytes(state, plan, 2, history_policy="checkpoints")
    monkeypatch.setattr(particle, "MAX_UNIFIED_HISTORY_BYTES", sparse_budget)
    with pytest.raises(ValueError, match="history requires"):
        particle.execute_unified_plan(state, plan, *phase, initial_time_s=np.zeros(2))
    particle.execute_unified_plan(state, plan, *phase, initial_time_s=np.zeros(2), history_policy="checkpoints")
    with pytest.raises(ValueError, match="history policy"):
        particle.execute_unified_plan(state, plan, *phase, history_policy="approximate")


def test_history_budget_covers_actual_immutable_checkpoint_construction_peak():
    """Measure constructor coexistence, including its still-live arguments."""
    import tracemalloc
    from types import SimpleNamespace
    from temsim.physics.core import PropagationCheckpoints
    from temsim.physics.unified_particle_restart import UnifiedParticleCheckpointState

    count, rows = 257, 129
    plan = SimpleNamespace(z_mm=np.linspace(0., 1., 513), checkpoint_index=np.arange(0, 513, 4))
    state = SimpleNamespace(recording_planes=())
    budget = particle.unified_history_memory_bytes(state, plan, count, history_policy="checkpoints")
    tracemalloc.start()
    try:
        phase = np.zeros((rows, count, 4))
        clock, kinetic = np.zeros((rows, count)), np.full((rows, count), 300000.)
        work, active = np.zeros((rows, count)), np.ones((rows, count), bool)
        internal = UnifiedParticleCheckpointState(np.full(count, 300000.), np.zeros(count), work, active,
            np.full(count, np.nan), ("",)*count, "0"*64, "0"*64, 0., np.zeros((rows, 2, 3)), ("0"*64,)*rows)
        selected = np.arange(rows)
        cp = PropagationCheckpoints(plan.z_mm[plan.checkpoint_index],
            phase[selected, :, 0], phase[selected, :, 2], phase[selected, :, 1], phase[selected, :, 3],
            clock[selected], kinetic[selected], unified_state=internal)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert cp.x_m.shape == (rows, count)
    assert peak <= budget
