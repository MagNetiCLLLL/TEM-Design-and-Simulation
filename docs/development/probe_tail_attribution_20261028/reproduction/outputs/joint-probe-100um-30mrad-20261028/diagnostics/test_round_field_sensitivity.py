"""Analytical/mocked explicit radial-order diagnostics; no ray propagation."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

FILE = Path(__file__).with_name("run_round_field_sensitivity.py")
spec = importlib.util.spec_from_file_location("round_field_diagnostic", FILE)
diagnostic = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diagnostic
spec.loader.exec_module(diagnostic)


def runtime():
    from temsim.physics.physical_scene import capture_physical_scene
    from temsim.physics.potential_geometry import PotentialGeometrySnapshot
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
    from temsim.physics.unified_field_runtime import UnifiedFieldRuntime
    from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
    from temsim.physics.lens_field_provider import FrozenAnalyticField, CoordinateRegistration
    from temsim.physics.planar_gun_field import PlanarGunField
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    scene = capture_physical_scene(SimpleNamespace(lenses=[SimpleNamespace(key=key, z_mm=float(i))
        for i, key in enumerate(("a", "gun"))]))
    native = FrozenAnalyticField("a", ((.3, 0., .006),), CoordinateRegistration(), (-80., 200.), .006)
    field = HighOrderRoundLensPotential(native, 3, .004, axial_domain="gaussian_all_z")
    r, z = np.array((0., .003)), np.array((0., .002))
    electric = ElectricPotentialSnapshot.from_provider(PlanarGunField({}, r, z, np.zeros((2, 2))))
    snapshot = ElectromagneticPotentialSnapshot(scene, (
        PotentialContribution("round:a", ("a",), field, "magnetic"),
        PotentialContribution("electric:gun", ("gun",), electric, "electric")))
    geometry = PotentialGeometrySnapshot(-.05, .05, (), source_scene_geometry_digest=scene.geometry_digest)
    return UnifiedFieldRuntime(snapshot, geometry, "0"*64, dict(round_radial_order=3,
        empirical_aberrations_applied=False, equivalent_image_maps_applied=False))


@pytest.mark.parametrize("order", [3, 5, 7])
def test_runtime_override_is_explicit_immutable_and_changes_only_round_order(order):
    base = runtime()
    before = base.to_dict()
    changed, manifest = diagnostic.variant_runtime(base, order)
    assert base.to_dict() == before
    assert changed.source_identity == base.source_identity
    assert changed.geometry is base.geometry
    assert changed.snapshot.scene is base.snapshot.scene
    assert changed.snapshot.physical_digest == base.snapshot.physical_digest
    assert changed.snapshot.contributions[1] is base.snapshot.contributions[1]
    assert manifest[1]["changed"] is False
    assert manifest[1]["original_digest"] == manifest[1]["executed_digest"]
    assert all(row["exact"] for row in manifest[0]["native_axis_checks"].values())
    assert changed.snapshot.contributions[0].potential.radial_order == order
    if order == 3:
        assert changed is base and changed.digest == base.digest
    else:
        assert changed.digest != base.digest
        assert changed.metadata["explicit_diagnostic_override"]["executed_order"] == order
        assert changed.metadata["round_radial_order"] == order


def test_added_fifth_order_matches_independent_gaussian_coefficient():
    base = runtime()
    changed, _ = diagnostic.variant_runtime(base, 5)
    old = base.snapshot.contributions[0].potential
    new = changed.snapshot.contributions[0].potential
    radius, sigma, b0 = .002, .006, .3
    before, after = old.sample([radius, 0., 0.]), new.sample([radius, 0., 0.])
    fourth_axial_derivative = 3*b0/sigma**4
    assert after["A_t_m"][1]-before["A_t_m"][1] == pytest.approx(fourth_axial_derivative*radius**5/384, rel=2e-12)
    assert after["B_t"][2]-before["B_t"][2] == pytest.approx(fourth_axial_derivative*radius**4/64, rel=2e-12)
    assert after["B_t"][0] == before["B_t"][0] == 0.
    for field in (old, new):
        jet = field.sample_jet([radius, 0., .001])
        jac = jet["jac_A_t"]
        curl = [jac[2, 1]-jac[1, 2], jac[0, 2]-jac[2, 0], jac[1, 0]-jac[0, 1]]
        np.testing.assert_allclose(curl, field.sample([radius, 0., .001])["B_t"], rtol=2e-14)


@pytest.mark.parametrize("value", [1, 4, 9, True, 5., None])
def test_unsupported_order_is_rejected_without_coercion(value):
    with pytest.raises(ValueError, match="radial order"):
        diagnostic.variant_runtime(runtime(), value)


def test_base_with_empirical_or_unknown_order_is_rejected():
    base = runtime()
    with pytest.raises(ValueError, match="additional"):
        diagnostic.variant_runtime(replace(base, metadata={**base.metadata, "empirical_aberrations_applied": True}), 5)
    with pytest.raises(ValueError, match="order 3"):
        diagnostic.variant_runtime(replace(base, metadata={**base.metadata, "round_radial_order": 5}), 5)


def arrays():
    angles = np.arange(20)*2*np.pi/20
    offsets = np.array([-1., 0., 1.])
    radii = np.array([.3, .2, .3])*1e-9
    result = dict(x_m=radii[:, None]*np.cos(angles), y_m=radii[:, None]*np.sin(angles),
        tx_rad=np.tile(.03*np.cos(angles), (3, 1)), ty_rad=np.tile(.03*np.sin(angles), (3, 1)),
        flight_time_s=np.full((3, 20), 1e-6), kinetic_energy_ev=np.full((3, 20), 300000.),
        alive=np.ones((3, 20), bool), z_mm=offsets*1e-6, integration_z_mm=offsets*1e-6,
        source_id=np.arange(20), weights=np.full(20, .05), terminal_alive=np.ones(20, bool),
        blocked_z_mm=np.full(20, np.nan), blocked_key=np.full(20, ""))
    result.update(gun_exit_x_m=np.full(20, 1e-6), gun_exit_kinetic_energy_ev=np.full(20, 300000.),
        emission_position_m=np.zeros((20, 3)))
    return result


@pytest.mark.parametrize("key", ["x_m", "kinetic_energy_ev", "flight_time_s", "weights", "alive", "blocked_key"])
def test_baseline_gate_detects_one_changed_element(key):
    original = arrays()
    changed = deepcopy(original)
    if key == "alive":
        changed[key][0, 0] = False
    elif key == "blocked_key":
        changed[key] = np.full(20, "new")
    else:
        changed[key].flat[0] = np.nextafter(changed[key].flat[0], np.inf)
    assert diagnostic.baseline_comparison(original, original)["passed"]
    assert not diagnostic.baseline_comparison(changed, original)["passed"]


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    import temsim.instrument_snapshot
    import temsim.physics.unified_field_runtime
    base = runtime()
    raw = arrays()
    directory = tmp_path/"donor"
    directory.mkdir()
    (directory/"report.json").write_text("{}")
    snapshot = SimpleNamespace(digest=base.source_identity, to_dict=lambda: {"synthetic": True})
    state = SimpleNamespace(sample=SimpleNamespace(upper_surface_z_mm=0.))
    donor = SimpleNamespace(state=state, snapshot=snapshot, arrays=raw, directory=directory,
        files_sha256={"report.json": diagnostic._hash(directory/"report.json")}, planes=raw["z_mm"],
        report={"receipt": {"source_current_a": 1e-5, "batches": [{"field_audit": {"identities": {
            "runtime_digest": base.digest, "potential_digest": base.snapshot.numerical_digest,
            "geometry_digest": base.geometry.digest}}}]}})
    monkeypatch.setattr(temsim.instrument_snapshot, "capture_instrument_snapshot", lambda _: snapshot)
    monkeypatch.setattr(temsim.physics.unified_field_runtime, "capture_instrument_snapshot", lambda _: snapshot)
    monkeypatch.setattr(temsim.physics.unified_field_runtime, "get_unified_field_runtime", lambda _: base)
    prepared = []
    def prepare(donor, base, order):
        prepared.append(order)
        variant, manifest = diagnostic.variant_runtime(base, order)
        return SimpleNamespace(unified_runtime=variant), dict(order=order, provider_manifest=manifest)
    monkeypatch.setattr(diagnostic, "prepare_plan", prepare)
    return donor, prepared


def test_complete_order3_replay_is_required_before_fifth_order(tmp_path, orchestration):
    donor, calls = orchestration
    def executor(donor, plan, cancelled):
        assert not cancelled()
        output = deepcopy(donor.arrays)
        if plan.unified_runtime.metadata["round_radial_order"] == 5:
            output["x_m"] *= 1.01
        return output, {"column_batches": 1, "gun_traces": 0}
    report = diagnostic.run(donor, tmp_path/"run", executor=executor)
    assert report["status"] == "COMPLETE"
    assert calls == [3, 5]
    assert report["stages"][0]["baseline_reproduction"]["passed"]
    assert report["stages"][1]["change_from_order3"]["maximum_relative_d95_change"] > 0
    assert report["source_input_unchanged"]


def test_baseline_mismatch_forbids_higher_order_and_preserves_failed_evidence(tmp_path, orchestration):
    donor, calls = orchestration
    def executor(donor, plan, cancelled):
        output = deepcopy(donor.arrays)
        output["x_m"][1, 0] = np.nextafter(output["x_m"][1, 0], np.inf)
        return output, {}
    report = diagnostic.run(donor, tmp_path/"run", executor=executor)
    assert report["status"] == "FAILED"
    assert calls == [3]
    assert report["stages"][0]["status"] == "FAILED"
    assert not report["stages"][0]["baseline_reproduction"]["passed"]
    assert (tmp_path/"run/order-3/particles.npz").exists()
    assert not (tmp_path/"run/order-5").exists()


def test_preexisting_cancel_and_thread_budget_fail_without_propagation(tmp_path, orchestration):
    donor, calls = orchestration
    flag = tmp_path/"cancel"
    flag.touch()
    report = diagnostic.run(donor, tmp_path/"cancelled", cancel_file=flag,
        executor=lambda *a: pytest.fail("No executor should run"))
    assert report["status"] == "CANCELLED" and calls == []
    with pytest.raises(ValueError, match="four threads"):
        diagnostic.run(donor, tmp_path/"unbounded", threads=5)
    assert not (tmp_path/"unbounded").exists()


@pytest.mark.parametrize("cancel_requested", [True, False])
def test_production_runtime_cancellation_is_distinguished_from_unrelated_failure(tmp_path, orchestration, cancel_requested):
    donor, calls = orchestration
    flag = tmp_path/"cancel"
    def executor(*args):
        if cancel_requested:
            flag.touch()
        raise RuntimeError("Superseded optical tuning request")
    report = diagnostic.run(donor, tmp_path/"run", executor=executor, cancel_file=flag)
    assert report["status"] == ("CANCELLED" if cancel_requested else "FAILED")
    assert calls == [3]
    assert "Superseded optical tuning request" in report["error"]
    assert json.loads((tmp_path/"run/order-3/report.json").read_text())["status"] == report["status"]
