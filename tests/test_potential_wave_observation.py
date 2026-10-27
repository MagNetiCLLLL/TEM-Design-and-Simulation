"""Controlled observation planes preserve raw phase, losses and checkpoint identity."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.constants import c, e, hbar, m_e

from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.potential_wave_observation import (
    PotentialAperturePlane, observe_potential_plane, observe_potential_mode,
    observe_potential_checkpoint, potential_observation_nodes,
)
from temsim.physics.tip_gun_wave import TipGunCheckpoint, TIP_REFERENCE
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_reference import AxialWaveReference
from test_potential_wave_carrier import _wave, _snapshot, _multipole


def _mode():
    return WaveMode(_wave(n=32, width=2e-7, curvature=-100.), .27, TIP_REFERENCE, "controlled:0", 300.,
                    AxialWaveReference(2e-12, 3e-25))


@pytest.mark.parametrize("with_geometry", [False, True])
@pytest.mark.parametrize("policy", ["sampling_retry", "step_start_and_retry", "step_boundaries_and_retry"])
def test_support_timing_policy_is_persistent_and_inferred_from_actual_legacy_stages(tmp_path, with_geometry, policy):
    from temsim.immutable_json import thaw_json
    from temsim.physics.potential_geometry import PotentialGeometrySnapshot, observe_potential_geometry_checkpoint
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from test_potential_geometry import _checkpoint
    snapshot, entrance = _snapshot(), _checkpoint()
    geometry = PotentialGeometrySnapshot(1., 1.000003, ())
    options = {"maximum_step_m": 5e-7, "support_projection_policy": policy}

    def observe(checkpoint, z, **overrides):
        selected = dict(options, **overrides)
        if with_geometry:
            return observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (z,), **selected)[0]
        return observe_potential_checkpoint(snapshot, checkpoint, (z,), **selected)[0]

    first = observe(entrance, 1.000001, support_ledger=NumericalSupportLedger(.01, 1e-7))
    assert first.record["potential_support_policy"]["policy"] == policy
    store = ExecutedWaveStore(tmp_path, snapshot.numerical_digest, 32*1024**2)
    key = store.key("support-policy", with_geometry, policy)
    store.put(key, first)
    restored = store.get(key)
    continued = observe(restored, 1.000002)
    assert continued.record["potential_support_policy"] == first.record["potential_support_policy"]
    assert continued.record["support_ledger"]["maximum_l2"] == .01
    for alternative in {"sampling_retry", "step_start_and_retry", "step_boundaries_and_retry"}-{policy}:
        for z in (1.000001, 1.000002):
            with pytest.raises(ValueError, match="support projection policy changed"):
                observe(restored, z, support_projection_policy=alternative)
    legacy = thaw_json(restored.record)
    legacy.pop("potential_support_policy")
    inferred = observe(replace(restored, record=legacy), 1.000002)
    assert inferred.record["potential_support_policy"] == first.record["potential_support_policy"]
    stages = legacy["modes"][0]["stages"]
    assert stages and stages[0]["support_projection_policy"] == policy
    stages.append({**stages[0], "support_projection_policy": next(value for value in
        ("sampling_retry", "step_start_and_retry", "step_boundaries_and_retry") if value != policy)})
    with pytest.raises(ValueError, match="Historical numerical support policies conflict"):
        observe(replace(restored, record=legacy), 1.000002)


def test_material_support_policy_is_evidence_only_without_new_projection():
    from temsim.immutable_json import thaw_json
    from temsim.physics.potential_support_lineage import capture_inherited_support
    from temsim.physics.potential_wave_policy import potential_support_policy
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from test_potential_geometry import _checkpoint
    snapshot = _snapshot()
    first, = observe_potential_checkpoint(snapshot, _checkpoint(), (1.000001,),
        maximum_step_m=5e-7, support_ledger=NumericalSupportLedger(.01, 1e-7),
        support_projection_policy="step_boundaries_and_retry")
    prefix = capture_inherited_support(first, snapshot.numerical_digest)
    context = thaw_json(first.record)
    context.pop("support_ledger")
    context.update(inherited_potential_support=prefix, support_prefix_digest=prefix["digest"])
    assert potential_support_policy({}, context) == first.record["potential_support_policy"]
    with pytest.raises(ValueError, match="disabled after material"):
        potential_support_policy({"support_projection_policy": "step_boundaries_and_retry"}, context)


@pytest.mark.parametrize("with_geometry", [False, True])
@pytest.mark.parametrize("order", [4, 6, 8])
def test_quartic_split_policy_is_persistent_and_cannot_change_on_resume(with_geometry, order, tmp_path):
    from temsim.physics.potential_geometry import PotentialGeometrySnapshot, observe_potential_geometry_checkpoint
    snapshot = _snapshot()
    entrance = TipGunCheckpoint(BeamState((_mode(),), TIP_REFERENCE), 1000., 1e-12,
                                {"source_scope": "controlled operator policy regression"})
    geometry = PotentialGeometrySnapshot(1., 1.000003, ())
    split_key = "quartic_scalar_splitting" if order == 4 else "higher_order_scalar_splitting"
    options = dict(maximum_step_m=5e-7, scalar_splitting=True, kinetic_order=order, **{split_key: True})

    def observe(checkpoint, z, **overrides):
        selected = dict(options, **overrides)
        if with_geometry:
            return observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (z,), **selected)[0]
        return observe_potential_checkpoint(snapshot, checkpoint, (z,), **selected)[0]

    first = observe(entrance, 1.000001)
    assert first.record["potential_operator_policy"]["policy"][split_key] is True
    store = ExecutedWaveStore(tmp_path, snapshot.numerical_digest, 20*1024**2)
    key = store.key("quartic-policy", with_geometry)
    store.put(key, first)
    restored = store.get(key)
    continued = observe(restored, 1.000002)
    assert continued.record["potential_operator_policy"] == first.record["potential_operator_policy"]
    assert continued.beam.total_weight == pytest.approx(entrance.beam.total_weight, abs=2e-13)
    for z in (1.000001, 1.000002):
        with pytest.raises(ValueError, match="operator policy changed"):
            observe(restored, z, **{split_key: False})
    malformed = dict(restored.record, potential_operator_policy=None)
    with pytest.raises(ValueError, match="operator policy changed"):
        observe(replace(restored, record=malformed), 1.000002)
    legacy = dict(restored.record)
    legacy.pop("potential_operator_policy")
    with pytest.raises(ValueError, match="Historical executed checkpoints"):
        observe(replace(restored, record=legacy), 1.000002)


@pytest.mark.parametrize("options", [
    {"quartic_scalar_splitting": True},
    {"quartic_scalar_splitting": True, "scalar_splitting": True, "kinetic_order": 2},
    {"quartic_scalar_splitting": 1},
    {"higher_order_scalar_splitting": True},
    {"higher_order_scalar_splitting": True, "kinetic_order": 6},
    {"higher_order_scalar_splitting": 1},
])
def test_observation_validates_quartic_policy_even_without_a_propagation_step(options):
    with pytest.raises(ValueError, match="splitting"):
        observe_potential_plane(_snapshot(), _mode().plane, 1., (1.,), energy_ev=300000., **options)


def test_v1_operator_policy_retains_historical_flags_and_cannot_claim_new_kinetic_orders():
    from temsim.immutable_json import json_digest
    from temsim.physics.potential_wave_policy import potential_wave_operator_policy
    old = {"kinetic_order": 4, "reference_quadrature": "cf4", "scalar_splitting": True,
           "quartic_scalar_splitting": True}
    receipt = {"schema": "potential-wave-operator-policy-v1", "policy": old, "digest": json_digest(old)}
    restored = potential_wave_operator_policy(old, {"potential_operator_policy": receipt})
    assert restored["policy"]["quartic_scalar_splitting"] is True
    assert restored["policy"]["higher_order_scalar_splitting"] is False
    for wrong in (6, 4.):
        changed = dict(old, kinetic_order=wrong)
        corrupted = dict(receipt, policy=changed, digest=json_digest(changed))
        with pytest.raises(ValueError):
            potential_wave_operator_policy(old, {"potential_operator_policy": corrupted})


def test_historical_scalar_split_receipt_is_preserved_instead_of_relabelled_false():
    from temsim.immutable_json import thaw_json
    from temsim.physics.potential_wave_policy import potential_wave_operator_policy
    entrance = TipGunCheckpoint(BeamState((_mode(),), TIP_REFERENCE), 1000., 1e-12,
                                {"source_scope": "actual historical metadata shape regression"})
    actual, = observe_potential_checkpoint(_snapshot(), entrance, (1.000001,), scalar_splitting=True)
    history = thaw_json(actual.record)
    history.pop("potential_operator_policy")
    stage = history["modes"][0]["stages"][0]
    assert stage["exact_scalar_potential_strang"] is True
    assert stage["residual_steps"][0]["scalar_exact_strang"] is True
    policy = potential_wave_operator_policy({"scalar_splitting": True}, history)
    assert policy["policy"]["scalar_splitting"] is True
    assert policy["policy"]["quartic_scalar_splitting"] is False
    with pytest.raises(ValueError, match="Historical operator policy changed"):
        potential_wave_operator_policy({}, history)
    stage.pop("exact_scalar_potential_strang")
    assert potential_wave_operator_policy({"scalar_splitting": True}, history) == policy
    stage["exact_scalar_potential_strang"] = False
    with pytest.raises(ValueError, match="invalid or inconsistent"):
        potential_wave_operator_policy({"scalar_splitting": True}, history)


@pytest.mark.parametrize("changed", [{"kinetic_order": 2}, {"reference_quadrature": "cf4"}])
def test_historical_actual_operator_order_and_quadrature_cannot_change(changed):
    from temsim.immutable_json import thaw_json
    entrance = TipGunCheckpoint(BeamState((_mode(),), TIP_REFERENCE), 1000., 1e-12,
                                {"source_scope": "historical stage identity regression"})
    actual, = observe_potential_checkpoint(_snapshot(), entrance, (1.000001,), **changed)
    history = thaw_json(actual.record)
    history.pop("potential_operator_policy")
    historical = replace(actual, record=history)
    with pytest.raises(ValueError, match="Historical operator policy changed"):
        observe_potential_checkpoint(_snapshot(), historical, (1.000001,))
    restored, = observe_potential_checkpoint(_snapshot(), historical, (1.000001,), **changed)
    assert all(restored.record["potential_operator_policy"]["policy"][key] == value
               for key, value in changed.items())


def test_exact_nodes_include_compact_faces_observations_and_aperture_once():
    native = replace(_multipole().base.field, native_support_mm=(1000.001, 1000.003))
    snapshot = _snapshot(AnalyticPotentialDomain(AnalyticMagneticPotential(native)))
    aperture = PotentialAperturePlane("iris", 1.0000023, 1e-6)
    nodes = potential_observation_nodes(snapshot, 1., (1.000002, 1.000004), maximum_step_m=1.1e-6,
                                         apertures=(aperture,))
    assert not nodes.flags.writeable
    for z in (1., 1.000001, 1.000002, aperture.z_m, 1.000003, 1.000004):
        assert np.count_nonzero(nodes == z) == 1
    assert np.max(np.diff(nodes)) <= 1.1e-6*(1+1e-9)


def test_arbitrary_observation_sequence_keeps_original_carriers_and_complex_phase():
    mode, snapshot = _mode(), _snapshot()
    results = observe_potential_plane(snapshot, mode.plane, 1., (1., 1.000001, 1.000002),
        energy_ev=300000., maximum_step_m=5e-7)
    np.testing.assert_array_equal(results[0].plane.amplitude, mode.plane.amplitude)
    assert all(row.plane.curvature_m1 is not None for row in results)
    direct, = observe_potential_plane(snapshot, mode.plane, 1., (1.000002,),
        energy_ev=300000., maximum_step_m=5e-7)
    np.testing.assert_allclose(results[-1].plane.amplitude, direct.plane.amplitude, rtol=2e-11, atol=2e-13)
    np.testing.assert_allclose(results[-1].plane.basis_m, direct.plane.basis_m, rtol=1e-13)
    assert results[-1].metadata["production_admission"] is False
    with pytest.raises(TypeError):
        results[-1].metadata["output_z_m"] = 0.


def test_neighbouring_float_observation_and_aperture_keep_exact_planes_and_finite_budget():
    aperture_z = 1.0000001
    observation_z = np.nextafter(aperture_z, np.inf)
    aperture = PotentialAperturePlane("nearby", aperture_z, 1e-3)
    result, = observe_potential_plane(_snapshot(), _mode().plane, 1., (observation_z,),
        energy_ev=300000., maximum_step_m=5e-8, apertures=(aperture,))
    assert result.z_m == observation_z
    assert result.metadata["aperture_events"][0]["z_m"] == aperture_z
    stages = result.metadata["stages"]
    assert stages[-1]["axial_nodes_m"] == (aperture_z, observation_z)
    assert sum(stage["series_amplitude_error_bound"] for stage in stages) <= 1e-10
    assert result.plane.probability == pytest.approx(_mode().plane.probability, abs=1e-12)


@pytest.mark.parametrize("blocked", [False, True])
def test_aperture_keeps_absolute_probability_weight_and_no_normalization(blocked):
    mode, snapshot = _mode(), _snapshot()
    center = (1e-3, 0.) if blocked else (1e-7, 0.)
    aperture = PotentialAperturePlane("iris", 1.000001, 2e-7, center)
    raw, = observe_potential_plane(snapshot, mode.plane, 1., (aperture.z_m,),
        energy_ev=300000., maximum_step_m=5e-7, apertures=(aperture,))
    output, = observe_potential_mode(snapshot, mode, 1., (aperture.z_m,),
        maximum_step_m=5e-7, apertures=(aperture,))
    np.testing.assert_allclose(output.mode.plane.amplitude*np.sqrt(output.mode.weight_per_reference_electron),
                               raw.plane.amplitude*np.sqrt(mode.weight_per_reference_electron), atol=2e-16)
    assert output.mode.weight_per_reference_electron == pytest.approx(.27*raw.plane.probability, abs=1e-15)
    assert raw.metadata["input_probability"] == pytest.approx(raw.metadata["output_probability"]+raw.metadata["aperture_loss"], abs=2e-13)
    assert raw.metadata["normalization_applied"] is False
    assert tuple(raw.metadata["applied_event_ids"]) == ("iris",)
    assert raw.metadata["all_stopped"] is blocked


def test_complete_absorption_does_not_create_future_current_or_an_invented_wave_phase():
    mode = _mode()
    stop = PotentialAperturePlane("stop", 1., 1e-7, (1e-3, 0.))
    rows = observe_potential_mode(_snapshot(), mode, 1., (1., 1.000001, 1.000002),
        maximum_step_m=1e-6, apertures=(stop,))
    assert all(row.mode.weight_per_reference_electron == 0. for row in rows)
    assert all(row.mode.plane.probability == 0. for row in rows)
    assert all(row.metadata["all_stopped"] for row in rows)
    assert all(len(row.metadata["aperture_events"]) == 1 for row in rows)
    assert not rows[-1].metadata["stages"]


def test_mode_action_and_time_use_constant_energy_without_entering_transverse_amplitude():
    mode = _mode()
    first, second = observe_potential_mode(_snapshot(), mode, 1., (1.000001, 1.000003), maximum_step_m=1e-6)
    kinetic = mode.energy_kev*1e3*e
    momentum = np.sqrt(kinetic*(kinetic+2*m_e*c*c))/c
    speed = momentum*c*c/(kinetic+m_e*c*c)
    for row in (first, second):
        distance = row.z_m-1.
        assert row.mode.axial_reference.flight_time_s == pytest.approx(mode.axial_reference.flight_time_s+distance/speed, rel=2e-15)
        assert row.mode.axial_reference.longitudinal_action_j_s == pytest.approx(mode.axial_reference.longitudinal_action_j_s+momentum*distance, rel=2e-15)
        assert row.metadata["longitudinal_carrier_in_amplitude"] is False
        assert row.mode.scattering_history == mode.scattering_history


def test_existing_checkpoint_store_preserves_complex_field_then_resumes(tmp_path):
    mode, snapshot = _mode(), _snapshot()
    entrance = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), 1000., 1e-12,
        {"source_scope": "controlled test mode, not a physical gun"})
    intermediate, = observe_potential_checkpoint(snapshot, entrance, (1.000001,), maximum_step_m=5e-7)
    store = ExecutedWaveStore(tmp_path, snapshot.numerical_digest, 20*1024**2)
    key = store.key("controlled-potential-observation", intermediate.digest)
    store.put(key, intermediate)
    loaded = store.get(key)
    for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(loaded.beam.modes[0].plane, name), getattr(intermediate.beam.modes[0].plane, name))
    resumed, = observe_potential_checkpoint(snapshot, loaded, (1.000002,), maximum_step_m=5e-7)
    direct, = observe_potential_checkpoint(snapshot, entrance, (1.000002,), maximum_step_m=5e-7)
    first, second = resumed.beam.modes[0], direct.beam.modes[0]
    np.testing.assert_allclose(first.plane.amplitude*np.sqrt(first.weight_per_reference_electron),
                               second.plane.amplitude*np.sqrt(second.weight_per_reference_electron), rtol=3e-11, atol=3e-13)
    assert first.axial_reference.flight_time_s == pytest.approx(second.axial_reference.flight_time_s, rel=2e-13)
    assert first.axial_reference.longitudinal_action_j_s == pytest.approx(second.axial_reference.longitudinal_action_j_s, rel=2e-13)


def test_fresh_process_restores_source_bound_complex_state_and_consumed_budgets(tmp_path):
    """A clean interpreter must need only the retained snapshot and disk store."""
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.wave_grid import WaveGridNumerics
    from test_potential_wave_carrier import _pure_hex

    snapshot = _snapshot(_pure_hex(5e7))
    mode = replace(_mode(), plane=_wave(n=128, pitch=1e-7, width=3e-7), energy_kev=200.)
    entrance = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), 1., 1e-12,
        {"scope": "controlled source for fresh-process persistence"})
    middle_z, final_z = .00101, .00102
    aperture = PotentialAperturePlane("recorded-prefix-plane", middle_z, 1e-4)
    options = dict(maximum_step_m=1e-5, scalar_splitting=True,
        support_projection_policy="step_start_and_retry",
        grid_numerics=WaveGridNumerics(maximum_pixels=512), apertures=(aperture,))
    initial_support, initial_phase = NumericalSupportLedger(1e-5, 1e-9), KineticPhaseLedger(.01)
    middle, expected = observe_potential_checkpoint(snapshot, entrance, (middle_z, final_z),
        support_ledger=initial_support, kinetic_phase_ledger=initial_phase, **options)
    support = NumericalSupportLedger.from_dict(middle.record["support_ledger"])
    phase = KineticPhaseLedger.from_dict(middle.record["kinetic_phase_ledger"])
    assert support.events and support.used_l2 > 0. and support.used_probability > 0.
    assert phase.events and phase.used_phase_rad > 0.
    assert middle.record["applied_event_ids"] == (aperture.event_id,)

    store = ExecutedWaveStore(tmp_path / "waves", snapshot.numerical_digest, 32*1024**2)
    middle_key = store.key("fresh-process-input", middle.digest)
    entrance_key = store.key("fresh-process-entrance", entrance.digest)
    output_key = store.key("fresh-process-output", middle.digest)
    direct_key = store.key("fresh-process-direct", entrance.digest)
    store.put(middle_key, middle)
    store.put(entrance_key, entrance)
    stored_digest = store.get(middle_key).digest
    request = {"snapshot": snapshot.to_dict(), "middle_key": middle_key, "entrance_key": entrance_key,
        "output_key": output_key, "direct_key": direct_key,
        "middle_z_m": middle_z, "final_z_m": final_z, "prefix_digest": stored_digest}
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    # This child imports no test fixtures and receives no live Python object.
    script = tmp_path / "continue_saved_potential.py"
    script.write_text('''from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, StaticPolynomialGauge
from temsim.physics.potential_phase_budget import KineticPhaseLedger
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.potential_wave_observation import observe_potential_checkpoint, PotentialAperturePlane
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_grid import WaveGridNumerics

folder = Path(sys.argv[1])
request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
snapshot = ElectromagneticPotentialSnapshot.from_dict(request["snapshot"])
store = ExecutedWaveStore(folder / "waves", snapshot.numerical_digest, 32*1024**2)
middle = store.get(request["middle_key"])
assert middle.digest == request["prefix_digest"]
options = dict(maximum_step_m=1e-5, scalar_splitting=True,
    support_projection_policy="step_start_and_retry",
    grid_numerics=WaveGridNumerics(maximum_pixels=512),
    apertures=(PotentialAperturePlane("recorded-prefix-plane", request["middle_z_m"], 1e-4),))
rejected = []
for name, replacement in (("support_ledger", NumericalSupportLedger(1e-5, 1e-9)),
                          ("kinetic_phase_ledger", KineticPhaseLedger(.01))):
    try:
        observe_potential_checkpoint(snapshot, middle, (request["final_z_m"],),
            **options, **{name: replacement})
    except ValueError as exc:
        assert "cannot be reset" in str(exc)
        rejected.append(name)
    else:
        raise AssertionError("A restored calculation accepted a reset budget")
changed = replace(snapshot, gauge=StaticPolynomialGauge(gradient_t_m=(1e-12, 0., 0.)))
try:
    observe_potential_checkpoint(changed, middle, (request["final_z_m"],), **options)
except ValueError as exc:
    assert "snapshot changed" in str(exc)
    rejected.append("changed_snapshot")
else:
    raise AssertionError("A restored checkpoint accepted a different source")
resumed, = observe_potential_checkpoint(snapshot, middle, (request["final_z_m"],), **options)
_, direct = observe_potential_checkpoint(snapshot, store.get(request["entrance_key"]),
    (request["middle_z_m"], request["final_z_m"]),
    support_ledger=NumericalSupportLedger(1e-5, 1e-9),
    kinetic_phase_ledger=KineticPhaseLedger(.01), **options)
store.put(request["output_key"], resumed)
store.put(request["direct_key"], direct)
(folder / "child_report.json").write_text(json.dumps({"pid": os.getpid(), "rejected": rejected,
    "snapshot_digest": snapshot.numerical_digest, "restored_digest": middle.digest}), encoding="utf-8")
''', encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    child = subprocess.run([sys.executable, "-X", "utf8", str(script), str(tmp_path)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    report = json.loads((tmp_path / "child_report.json").read_text(encoding="utf-8"))
    assert report["pid"] != os.getpid()
    assert report["snapshot_digest"] == snapshot.numerical_digest
    assert report["restored_digest"] == stored_digest
    assert report["rejected"] == ["support_ledger", "kinetic_phase_ledger", "changed_snapshot"]

    for key in (output_key, direct_key):
        result = store.get(key)
        actual, reference = result.beam.modes[0], expected.beam.modes[0]
        for name in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
            np.testing.assert_allclose(getattr(actual.plane, name), getattr(reference.plane, name),
                rtol=2e-13, atol=2e-15)
        # Carrier arrays, absolute scalar action and both complex quadratures
        # are retained separately; no piston fit or weight normalization occurs.
        np.testing.assert_allclose(actual.plane.amplitude*np.sqrt(actual.weight_per_reference_electron),
            reference.plane.amplitude*np.sqrt(reference.weight_per_reference_electron),
            rtol=3e-10, atol=3e-13)
        assert actual.weight_per_reference_electron == pytest.approx(reference.weight_per_reference_electron, abs=2e-14)
        assert actual.axial_reference.flight_time_s == pytest.approx(
            reference.axial_reference.flight_time_s, rel=2e-15)
        axial_phase_difference = (actual.axial_reference.longitudinal_action_j_s
            - reference.axial_reference.longitudinal_action_j_s)/hbar
        # Sliced addition may differ by one float ulp in the large absolute
        # action. Include its physical phase, rather than fitting it away.
        assert abs(axial_phase_difference) < 1e-6
        absolute_actual = actual.plane.amplitude*np.sqrt(actual.weight_per_reference_electron)*np.exp(1j*axial_phase_difference)
        absolute_reference = reference.plane.amplitude*np.sqrt(reference.weight_per_reference_electron)
        assert np.linalg.norm(absolute_actual-absolute_reference)/np.linalg.norm(absolute_reference) < 1e-6
        assert result.reference_current_a == entrance.reference_current_a
        assert result.record["applied_event_ids"] == (aperture.event_id,)
        restored_support = NumericalSupportLedger.from_dict(result.record["support_ledger"])
        restored_phase = KineticPhaseLedger.from_dict(result.record["kinetic_phase_ledger"])
        assert restored_support.events[:len(support.events)] == support.events
        assert restored_phase.events[:len(phase.events)] == phase.events
        assert restored_support.maximum_l2 == initial_support.maximum_l2
        assert restored_support.maximum_probability == initial_support.maximum_probability
        assert support.used_l2 <= restored_support.used_l2 <= initial_support.maximum_l2
        assert support.used_probability <= restored_support.used_probability <= initial_support.maximum_probability
        assert phase.used_phase_rad < restored_phase.used_phase_rad <= initial_phase.maximum_phase_rad
        assert restored_phase.maximum_phase_rad == initial_phase.maximum_phase_rad
        assert restored_phase.binding["snapshot_digest"] == snapshot.numerical_digest
        assert restored_phase.last_z_m == final_z
        assert result.record["modes"][0]["aperture_loss"] == 0.


def test_resumed_aperture_id_is_not_reapplied_or_reported_twice():
    mode = _mode()
    aperture = PotentialAperturePlane("prior_stop", 1., 1e-7, (1e-3, 0.))
    result, = observe_potential_mode(_snapshot(), mode, 1., (1.,), maximum_step_m=1e-6,
        apertures=(aperture,), already_applied_event_ids=("prior_stop",))
    assert result.mode.weight_per_reference_electron == mode.weight_per_reference_electron
    assert not result.metadata["aperture_events"]


def test_checkpoint_resume_rejects_changed_potential_or_prior_aperture_geometry():
    from temsim.physics.potential_snapshot import StaticPolynomialGauge
    mode, snapshot = _mode(), _snapshot()
    entrance = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), 1000., 1e-12, {"scope": "controlled"})
    event = PotentialAperturePlane("iris", 1.000001, 1e-5)
    middle, = observe_potential_checkpoint(snapshot, entrance, (event.z_m,), maximum_step_m=5e-7, apertures=(event,))
    altered = replace(snapshot, gauge=StaticPolynomialGauge(gradient_t_m=(1e-12, 0., 0.)))
    with pytest.raises(ValueError, match="snapshot changed"):
        observe_potential_checkpoint(altered, middle, (1.000002,), maximum_step_m=5e-7)
    with pytest.raises(ValueError, match="aperture geometry changed"):
        observe_potential_checkpoint(snapshot, middle, (1.000002,), maximum_step_m=5e-7,
            apertures=(replace(event, radius_m=2e-5),))


@pytest.mark.parametrize("observations", [(), (1.001, 1.0005), (.9,), (1., np.nan)])
def test_invalid_observation_requests_reject_before_propagation(observations):
    with pytest.raises(ValueError, match="Observation"):
        potential_observation_nodes(_snapshot(), 1., observations, maximum_step_m=1e-6)


def test_duplicate_or_outside_aperture_events_are_rejected():
    event = PotentialAperturePlane("iris", 1., 1e-6)
    with pytest.raises(ValueError, match="unique"):
        potential_observation_nodes(_snapshot(), 1., (1.000001,), maximum_step_m=1e-6, apertures=(event, event))
    with pytest.raises(ValueError, match="executed interval"):
        potential_observation_nodes(_snapshot(), 1., (1.000001,), maximum_step_m=1e-6,
            apertures=(replace(event, z_m=2.),))


def test_unified_output_enters_existing_material_action_and_absolute_detector_without_legacy_pupil(monkeypatch):
    from temsim.detector.wave_readout import read_wave_detector, WaveReadoutOptions
    from temsim.physics import specimen_wave_transport, wave_imaging
    from temsim.physics.specimen_wave_channels import _attenuate_zero_loss
    from temsim.physics.wave_grid import WaveGridNumerics
    from test_specimen_wave_channels import material
    from test_wave_detector_readout import detector
    monkeypatch.setattr(wave_imaging, "explicit_pupil_spectrum", lambda *a, **k: pytest.fail("Legacy pupil must not be reapplied"))
    monkeypatch.setattr(specimen_wave_transport, "_propagate_column", lambda *a, **k: pytest.fail("This single material action must not rerun legacy optics"))
    mode, snapshot = _mode(), _snapshot()
    entrance = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), 1000., 1e-12,
        {"source_scope": "controlled single-mode material-interface fixture"})
    transported, = observe_potential_checkpoint(snapshot, entrance, (1.000001,), maximum_step_m=5e-7)
    before = transported.beam.modes[0]
    sigma = wave_imaging.interaction_constant_rad_per_v_angstrom(before.energy_kev)
    phase = .23
    axis = np.linspace(-5e-6, 5e-6, 33)
    potential = np.full((33, 33), phase/sigma)
    after_phase, action = specimen_wave_transport._material_phase(before, potential, axis, axis,
        sigma, 1., numerics=WaveGridNumerics(specimen_phase_method="galerkin", compute_backend="cpu"),
        cancelled=lambda: False)
    np.testing.assert_allclose(after_phase.plane.amplitude, before.plane.amplitude*np.exp(1j*phase), atol=2e-13)
    assert action["carrier_retained"] is True
    attenuated, ledger = _attenuate_zero_loss(after_phase, np.ones(after_phase.plane.amplitude.shape, bool), 30., material())
    survival = np.exp(-30*(1/100+1/200+1/400))
    assert attenuated.weight_per_reference_electron == pytest.approx(before.weight_per_reference_electron*survival, abs=2e-15)
    specimen = replace(transported, beam=BeamState((attenuated,), TIP_REFERENCE), record={
        "scope": "one prepared thin specimen action plus known material attenuation; not full multislice",
        "upstream_digest": transported.digest, "material_action": action, "loss_ledger": ledger})
    sensor = detector()
    sensor.z_mm = specimen.plane_z_mm
    sensor.outer_width_mm = .01
    options = WaveReadoutOptions(intensity=True, phase=True, complex_amplitude=True)
    original = read_wave_detector(transported, sensor, options)
    readout = read_wave_detector(specimen, sensor, options)
    assert readout.optical_probability.sum() == pytest.approx(attenuated.weight_per_reference_electron, abs=2e-14)
    assert readout.detected_probability.sum() == pytest.approx(attenuated.weight_per_reference_electron, abs=2e-14)
    np.testing.assert_allclose(readout.modes[0].complex_cell_amplitude,
        original.modes[0].complex_cell_amplitude*np.sqrt(survival)*np.exp(1j*phase), atol=2e-13)
    assert readout.modes[0].axial_reference == before.axial_reference
