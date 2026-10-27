"""Declared residual Cc: independent ray/phase formula and execution ownership."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import hbar

from temsim.physics.core import build_propagation_plan, execute_propagation_plan
from temsim.physics.intrinsic_aberrations import (
    FrozenIntrinsicAberrationKick, apply_intrinsic_wave, capture_chromatic_kicks,
)
from temsim.optics.lens_focal_length import electron_momentum
from temsim.physics.tip_gun_wave import _momentum_velocity
from test_magnetic_lens_aberration import _RoundLens, _state
from test_intrinsic_aberration_operator import packet


def fixture(cc=2.):
    lens = _RoundLens(peak_t=0., focal_mm=10.)
    lens.cs_mm, lens.cc_mm = 0., cc
    state = _state(lens)
    state.chromatic_aberration_enabled = True
    return state


def plan_for(state):
    return build_propagation_plan(state, -.1, .1, (),
        save_z_mm=(-.1, 0., .1), checkpoint_z_mm=(0., .1))


@pytest.mark.parametrize("energy_ev", [299000., 300000., 301000.])
def test_ray_wave_screen_equals_independent_focal_perturbation(energy_ev):
    state = fixture()
    kick, = capture_chromatic_kicks(state, -.1, .1)
    wave = packet()
    p = float(_momentum_velocity(energy_ev)[0])
    wavelength = 2*np.pi*hbar/p
    result, record = apply_intrinsic_wave(wave, wavelength, kick,
        momentum_kg_m_s=p, reference_momentum_kg_m_s=p)
    strength = .002*(energy_ev-300000.)/300000./.01**2
    xy = wave.coordinates_m()
    expected = wave.full_amplitude(wavelength)*np.exp(
        1j*np.pi/wavelength*strength*(xy[0]**2+xy[1]**2))
    np.testing.assert_allclose(result.full_amplitude(wavelength), expected, rtol=1e-11, atol=1e-14)
    ray = kick.apply_canonical(np.array([wave.origin_m[0]]), np.array([.003]),
        np.array([wave.origin_m[1]]), np.array([-.002]),
        momentum_kg_m_s=electron_momentum(energy_ev/1000.))
    np.testing.assert_allclose([ray[1][0]-.003, ray[3][0]+.002],
        strength*wave.origin_m, rtol=2e-10, atol=1e-18)
    assert result.probability == pytest.approx(wave.probability, abs=1e-14)
    np.testing.assert_allclose(result.tilt_rad-wave.tilt_rad, strength*wave.origin_m,
        rtol=2e-10, atol=1e-18)
    assert record["cc_mm"] == 2.
    if energy_ev == 300000.:
        assert result is wave


def test_mixed_energy_plan_and_checkpoint_resume_execute_cc_exactly_once():
    state = fixture()
    plan = plan_for(state)
    x, y = np.array([1e-5, 2e-5, 3e-5]), np.array([2e-5, -1e-5, 4e-5])
    energy = np.array([299000., 300000., 301000.])
    trace = execute_propagation_plan(state, plan, x, x*0, y, y*0,
                                    initial_kinetic_energy_ev=energy)
    strength = .002*(energy-300000.)/300000./.01**2
    np.testing.assert_allclose(trace[-1].tx_rad[-1], strength*x, rtol=3e-13, atol=1e-19)
    np.testing.assert_allclose(trace[-1].ty_rad[-1], strength*y, rtol=3e-13, atol=1e-19)
    cp = trace[-1]
    resumed = execute_propagation_plan(state, plan, cp.x_m[0], cp.tx_rad[0],
        cp.y_m[0], cp.ty_rad[0], initial_kinetic_energy_ev=energy,
        start_index=int(np.flatnonzero(plan.z_mm == 0.)[0]), include_initial_plane_kicks=False)
    for name in ("x_m", "tx_rad", "y_m", "ty_rad", "kinetic_energy_ev"):
        np.testing.assert_allclose(getattr(resumed[-1], name)[-1], getattr(cp, name)[-1], rtol=2e-13)


def test_zero_disabled_missing_and_mapped_cc_do_not_add_estimates(monkeypatch):
    state = fixture(0.)
    assert not capture_chromatic_kicks(state, -1., 1.)
    state.lenses[0].cc_mm = None
    assert not capture_chromatic_kicks(state, -1., 1.)
    state.lenses[0].key = "objective_lens"
    assert -dict(capture_chromatic_kicks(state, -1., 1.)[0].coefficients)["c1_mm"] == 2.
    state.chromatic_aberration_enabled = False
    assert not capture_chromatic_kicks(state, -1., 1.)
    state.chromatic_aberration_enabled = True
    monkeypatch.setattr("temsim.physics.lens_field_provider.active_mapped_providers",
        lambda state: [SimpleNamespace(lens_key="objective_lens")])
    assert not capture_chromatic_kicks(state, -1., 1.)


def test_cc_plan_archive_and_signature_preserve_energy_contract():
    from temsim.particle_section_io import _pack_record_graph, _unpack_record_graph
    state = fixture()
    plan = plan_for(state)
    arrays = {}
    graph = _pack_record_graph(plan, arrays)
    restored = _unpack_record_graph(graph, arrays, maximum_unpacked_bytes=1<<24)
    assert restored.intrinsic_aberration_kicks == plan.intrinsic_aberration_kicks
    assert restored.intrinsic_aberration_kicks[0].chromatic_reference_energy_ev == 300000.
    state.lenses[0].cc_mm = 20.
    assert plan_for(state).signature != plan.signature
    state.chromatic_aberration_enabled = False
    assert not plan_for(state).intrinsic_aberration_kicks


def test_static_and_chromatic_actions_of_same_lens_both_invalidate_cached_prefix():
    from temsim.physics.core import propagation_plan_common_prefix_nodes
    state = fixture()
    key = state.lenses[0].key
    state.lens_intrinsic_aberrations = {key: {"a1_mm": .001}}
    original = plan_for(state)
    state.lens_intrinsic_aberrations[key]["a1_mm"] = .002
    changed = plan_for(state)
    assert len(changed.intrinsic_aberration_kicks) == 2
    # The cache retains a two-node safety margin before a changed action.
    assert propagation_plan_common_prefix_nodes(original, changed) == 0
    state.lens_intrinsic_aberrations[key]["a1_mm"] = .001
    state.lenses[0].cc_mm = 20.
    assert propagation_plan_common_prefix_nodes(original, plan_for(state)) == 0


def test_coincident_specimen_loss_adds_only_increment_and_keeps_physical_plane():
    from temsim.physics.intrinsic_aberrations import specimen_chromatic_energy_change
    state = fixture()
    state.sample = SimpleNamespace(z_mm=0.)
    x, y = np.array([2e-5, -3e-5]), np.array([-4e-5, 1e-5])
    incoming = np.array([300100., 299900.])
    outgoing = incoming-20.
    dx, dy = specimen_chromatic_energy_change(state, x, y, incoming, outgoing)
    increment = .002*(-20.)/300000./.01**2
    np.testing.assert_allclose(dx, increment*x, rtol=1e-15)
    np.testing.assert_allclose(dy, increment*y, rtol=1e-15)
    kick, = capture_chromatic_kicks(state, -1., 1.)
    before = kick.apply_canonical(x, x*0, y, y*0,
        momentum_kg_m_s=np.array([electron_momentum(v/1000.) for v in incoming]))
    expected = .002*(outgoing-300000.)/300000./.01**2
    np.testing.assert_allclose(before[1]+dx, expected*x, rtol=3e-12)
    np.testing.assert_allclose(before[3]+dy, expected*y, rtol=3e-12)
    for value in specimen_chromatic_energy_change(state, x, y, incoming, incoming):
        np.testing.assert_array_equal(value, np.zeros(2))
    # A downstream lens must consume outgoing energy at its own plane;
    # an upstream lens must not be reapplied at the specimen.
    for lens_z in (-.1, .1):
        state.lenses[0].z_mm = lens_z
        for value in specimen_chromatic_energy_change(state, x, y, incoming, outgoing):
            np.testing.assert_array_equal(value, np.zeros(2))


def test_cc_roundoff_at_endpoint_is_owned_by_incident_not_skipped_on_resume():
    state = fixture()
    state.lenses[0].z_mm = 1599.2
    state.lens_intrinsic_aberrations = {state.lenses[0].key: {"a1_mm": .001}}
    before = build_propagation_plan(state, 1599., 1599.2, checkpoint_z_mm=(1599.2,))
    after = build_propagation_plan(state, 1599.2, 1599.4, checkpoint_z_mm=(1599.4,))
    assert len(before.intrinsic_aberration_kicks) == len(after.intrinsic_aberration_kicks) == 2
    x, zero = np.array([1e-5]), np.zeros(1)
    first = execute_propagation_plan(state, before, x, zero, zero, zero,
                                    initial_kinetic_energy_ev=np.array([300100.]))
    checkpoint = first[-1]
    second = execute_propagation_plan(state, after, checkpoint.x_m[-1], checkpoint.tx_rad[-1],
        checkpoint.y_m[-1], checkpoint.ty_rad[-1], initial_kinetic_energy_ev=np.array([300100.]),
        include_initial_plane_kicks=False)
    expected = x*(.002*(100./300000.)-.001e-3)/.01**2
    np.testing.assert_allclose(checkpoint.tx_rad[-1], expected, rtol=1e-12)
    np.testing.assert_allclose(second[-1].tx_rad[-1], expected, rtol=1e-12)


def test_specimen_increment_preserves_stopped_unknown_energy_and_rejects_live_unknown():
    from temsim.physics.intrinsic_aberrations import specimen_chromatic_energy_change
    state = fixture()
    state.sample = SimpleNamespace(z_mm=0.)
    x, y = np.array([1e-5, np.nan]), np.array([2e-5, np.nan])
    before, after = np.array([300100., np.nan]), np.array([300080., np.nan])
    for output in specimen_chromatic_energy_change(state, x, y, before, before, active_mask=[True, False]):
        np.testing.assert_array_equal(output, np.zeros(2))
    dx, dy = specimen_chromatic_energy_change(state, x, y, before, after, active_mask=[True, False])
    assert dx[1] == dy[1] == 0.
    assert dx[0] == pytest.approx(1e-5*.002*(-20.)/300000./.01**2)
    with pytest.raises(ValueError, match="known positive"):
        specimen_chromatic_energy_change(state, x, y, before, after, active_mask=[True, True])


def test_finite_specimen_handoff_uses_source_indexed_energy_change(monkeypatch):
    import temsim.specimen.downstream_transport as module
    from test_downstream_transport import _test_state, _test_simulation, _test_transport, _stub_times
    state, simulation, transport = _test_state(), _test_simulation(), _test_transport()
    lens = _RoundLens(peak_t=0., focal_mm=10.)
    lens.z_mm, lens.cc_mm = 1., 2.
    state.lenses, state.chromatic_aberration_enabled = [lens], True
    simulation.incident.kinetic_energy_ev = np.array([[200100., np.nan, 199900.]])
    terminal = transport.terminal_electrons
    # Isolate reference-plane matching; the downstream handoff itself is real.
    class IdentityMatching:
        def __init__(self, state):
            pass
        def to_plane(self, position, direction, *args, **kwargs):
            return position, direction, 0.
    monkeypatch.setattr(module, "SpecimenFieldTransport", IdentityMatching)
    rows = []
    def capture(s, start, stop, x, tx, y, ty, events, energy, **kwargs):
        rows.append((x.copy(), tx.copy(), y.copy(), ty.copy()))
        z = np.array([start, stop])
        return (z, *(np.tile(v, (2, 1)) for v in (x, tx, y, ty)),
                _stub_times(z, tx, ty, energy, **kwargs))
    monkeypatch.setattr(module, "propagate", capture)
    for name in ("clip_recording_planes", "clip_column_wall"):
        monkeypatch.setattr(module, name, lambda s, z, x, y, a, b, k: (a, b, k))
    module.build_geometric_specimen_exit(state, simulation, transport, stop_z_mm=2.)
    assert len(rows) == 2
    for row, index in zip(rows, (0, 1)):
        x, tx, y, ty = row
        source_index = terminal.source_ray_index[index]
        delta = (terminal.kinetic_energy_ev[index]-simulation.incident.kinetic_energy_ev[0, source_index])/200000.
        slope = terminal.direction[index, :2]/terminal.direction[index, 2]
        np.testing.assert_allclose(tx, slope[0]+x*.002*delta/.01**2, rtol=1e-13)
        np.testing.assert_allclose(ty, slope[1]+y*.002*delta/.01**2, rtol=1e-13)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., True, "2"])
def test_invalid_cc_rejected(value):
    with pytest.raises(ValueError, match="Cc"):
        capture_chromatic_kicks(fixture(value), -1., 1.)


def test_off_energy_cuda_screen_matches_cpu_without_recentering():
    from test_wave_device import _cuda
    cp = _cuda()
    kick, = capture_chromatic_kicks(fixture(), -1., 1.)
    x, y = np.array([1e-5, -4e-5, 3e-5]), np.array([-2e-5, 5e-5, 1e-5])
    p = np.array([electron_momentum(kv) for kv in (299., 300., 301.)])
    args = x, np.zeros(3), y, np.zeros(3)
    cpu = kick.apply_canonical(*args, momentum_kg_m_s=p)
    gpu = kick.apply_canonical(*(cp.asarray(a) for a in args), momentum_kg_m_s=cp.asarray(p))
    for left, right in zip(cpu, gpu):
        np.testing.assert_allclose(left, cp.asnumpy(right), rtol=3e-12, atol=1e-18)
