"""Intrinsic screens survive persistence and cannot be lost in radial dispatch."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from temsim.artifact_store import ArtifactIntegrityError, ArtifactStore, INCIDENT_SEED_CODEC
from temsim.immutable_json import thaw_json
from temsim.physics.intrinsic_aberrations import FrozenIntrinsicAberrationKick
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.posed_aberrations import FrozenLensAberrationKick
from test_shared_field_archives import storage_result


def kicks():
    rotation = Rotation.from_rotvec((0., .001, .1)).as_matrix()
    registration = CoordinateRegistration((1e-7, -2e-7, 0.), tuple(map(tuple, rotation)))
    intrinsic = FrozenIntrinsicAberrationKick("fixture_lens", .002, .01,
        (("a1_mm", .002), ("a1_azimuth_deg", 13.), ("b2_mm", .3)), registration)
    spherical = FrozenLensAberrationKick("fixture_cs", .002, 1e6, registration)
    return intrinsic, spherical


def populated_store(tmp_path, result, chromatic=False):
    intrinsic, spherical = kicks()
    if chromatic:
        intrinsic = replace(intrinsic, coefficients=(("c1_mm", -2.),),
                            chromatic_reference_energy_ev=300000.)
    plan = replace(result.simulation.incident_plan,
        intrinsic_aberration_kicks=(intrinsic,), posed_spherical_kicks=(spherical,))
    simulation = replace(result.simulation, incident_plan=plan)
    store = ArtifactStore(tmp_path, quota_bytes=100_000_000)
    store.put_incident_simulation_seed(result.calculation_manifest, simulation)
    return store, simulation


@pytest.mark.parametrize("chromatic", [False, True])
def test_incident_seed_preserves_intrinsic_and_posed_cs_actions(storage_result, tmp_path, chromatic):
    store, original = populated_store(tmp_path, storage_result, chromatic)
    restored = store.get_incident_simulation_seed(storage_result.calculation_manifest)
    original_plan, restored_plan = original.incident_plan, restored.incident_plan
    assert restored_plan.intrinsic_aberration_kicks == original_plan.intrinsic_aberration_kicks
    assert restored_plan.posed_spherical_kicks == original_plan.posed_spherical_kicks
    coordinates = (np.array((1e-6, -2e-6)), np.array((1e-5, -2e-5)),
                   np.array((3e-6, 4e-6)), np.array((2e-5, 1e-5)))
    for before, after in zip((*original_plan.intrinsic_aberration_kicks, *original_plan.posed_spherical_kicks),
                             (*restored_plan.intrinsic_aberration_kicks, *restored_plan.posed_spherical_kicks)):
        expected = before.apply_canonical(*coordinates, momentum_kg_m_s=original_plan.reference_momentum_kg_m_s*1.001)
        actual = after.apply_canonical(*coordinates, momentum_kg_m_s=restored_plan.reference_momentum_kg_m_s*1.001)
        for x, y in zip(expected, actual):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("damage", ("missing_intrinsic", "missing_posed", "negative_focal", "coefficient_shape", "duplicate_cs", "missing_energy_reference"))
def test_current_seed_cannot_drop_or_invent_aberration_actions(storage_result, tmp_path, damage):
    store, _ = populated_store(tmp_path, storage_result)
    manifest = storage_result.calculation_manifest
    bundle = store.get_array_bundle(manifest, product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"], codec=INCIDENT_SEED_CODEC)
    metadata = thaw_json(bundle.metadata)
    if damage == "missing_intrinsic":
        metadata.pop("plan_intrinsic_aberration_kicks")
    elif damage == "missing_posed":
        metadata.pop("plan_posed_spherical_kicks")
    elif damage == "negative_focal":
        metadata["plan_intrinsic_aberration_kicks"][0]["focal_m"] = -1.
    elif damage == "coefficient_shape":
        metadata["plan_intrinsic_aberration_kicks"][0]["coefficients"] = [1., 2.]
    elif damage == "missing_energy_reference":
        metadata["plan_intrinsic_aberration_kicks"][0].pop("chromatic_reference_energy_ev")
    else:
        metadata["plan_intrinsic_aberration_kicks"][0]["coefficients"]["c3_mm"] = 1.
    store.put_array_bundle(manifest, product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"], codec=INCIDENT_SEED_CODEC,
        arrays=dict(bundle.arrays), metadata=metadata)
    with pytest.raises(ArtifactIntegrityError):
        store.get_incident_simulation_seed(manifest)


def test_previous_seed_codec_remains_history_only(storage_result, tmp_path):
    store, _ = populated_store(tmp_path / "current", storage_result)
    manifest = storage_result.calculation_manifest
    bundle = store.get_array_bundle(manifest, product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"], codec=INCIDENT_SEED_CODEC)
    old = ArtifactStore(tmp_path / "historical", quota_bytes=100_000_000)
    metadata = thaw_json(bundle.metadata)
    metadata.pop("plan_intrinsic_aberration_kicks")
    codec = "incident-simulation-seed-v5-electromagnetic-energy"
    old.put_array_bundle(manifest, product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"], codec=codec,
        arrays=dict(bundle.arrays), metadata=metadata)
    assert old.get_incident_simulation_seed(manifest) is None
    assert old.get_array_bundle(manifest, product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"], codec=codec) is not None


@pytest.mark.parametrize("kind", ("a1_mm", "c1_mm", "c5_mm", "posed_cs"))
def test_radial_prefix_hands_unsupported_aberrations_to_full_column(monkeypatch, kind):
    from temsim.physics.radial_column_wave import _round_column_prefix
    intrinsic, spherical = kicks()
    if kind != "posed_cs":
        intrinsic = replace(intrinsic, coefficients=((kind, 1.),))
    plan = SimpleNamespace(intrinsic_aberration_kicks=(intrinsic,) if kind != "posed_cs" else (),
                           posed_spherical_kicks=(spherical,) if kind == "posed_cs" else ())
    monkeypatch.setattr("temsim.physics.column_wave._prepare_column", lambda *args: (plan, (), {}, ()))
    checkpoint = SimpleNamespace(plane_z_mm=1.)
    z, modes, record = _round_column_prefix(SimpleNamespace(), checkpoint, 3.)
    assert z == checkpoint.plane_z_mm and modes == ()
    assert "two-dimensional column operator" in record["reason"]
