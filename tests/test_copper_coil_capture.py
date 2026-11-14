"""Installed winding ownership and immutable capture; no ray propagation."""
from copy import copy, deepcopy
from dataclasses import replace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.copper_coil_field_provider import CopperCoilAssemblyPotential
from temsim.physics.lens_field_provider import FrozenAnalyticField, resolve_runtime_lens_field_provider
from temsim.physics.potential_capture import capture_round_lens_potential, capture_selected_column_potentials
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot


@pytest.fixture(scope="module")
def installed():
    state = default_state()
    state.simulation_mode = "unified"
    state.round_lens_field_model = "copper_coil"
    return state


def _copy_state(state):
    result = copy(state)
    result.lenses = deepcopy(state.lenses)
    result._runtime_copper_coil_provider_cache = {}
    result._runtime_lens_field_provider_cache = {}
    return result


def _change_part(state, key, **changes):
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, **changes}) if part.key == key else part
        for part in state._resolved_assembly.parts))


def test_every_installed_round_lens_captures_owned_windings_without_gaussians(installed):
    lenses = tuple(lens.key for lens in installed.lenses if lens.enabled)
    snapshot = capture_selected_column_potentials(installed, round_lens_keys=lenses)
    assert set(row.source_id for row in snapshot.contributions) == set(lenses)
    for row in snapshot.contributions:
        assert type(row.potential) is CopperCoilAssemblyPotential
        assert row.owner_ids == (row.source_id,)
        assert row.potential.provenance["reference_ampere_turns_at_100_percent"] == 3000.
        assert row.potential.provenance["reference_status"] == "engineering_reference_not_oem_not_particle_qualified"
        assert "optical tilt does not imply a mechanically tilted copper winding" in row.potential.provenance["source_registration_policy"]
        assert not row.potential.to_metadata()["pole_piece_magnetism"]
    clone = ElectromagneticPotentialSnapshot.from_dict(snapshot.to_dict())
    assert clone.numerical_digest == snapshot.numerical_digest
    point = np.array([[0., 0., 1.6]])
    np.testing.assert_array_equal(clone.sample(point)["B_t"], snapshot.sample(point)["B_t"])
    # Uncompiled providers remain complete in the production sampler.
    from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler
    sampled = compile_unified_potential_sampler(snapshot, prefer_plane=True).sample(point)
    np.testing.assert_array_equal(sampled["B_t"], snapshot.sample(point)["B_t"])


def test_objective_winding_uses_split_material_intervals_and_conserves_ampere_turns(installed):
    row = capture_round_lens_potential(installed, "objective_lens")
    coils = row.potential.coils
    assert len(coils) == 2
    assert coils[0].center_z_m+coils[0].length_m/2 < coils[1].center_z_m-coils[1].length_m/2
    assert sum(coil.length_m for coil in coils) == pytest.approx(.290)
    excitation = row.potential.provenance
    expected = excitation["reference_ampere_turns_at_100_percent"]*excitation["excitation_percent"]*.01*excitation["polarity"]
    assert sum(coil.ampere_turns for coil in coils) == pytest.approx(expected)
    gap = (coils[0].center_z_m+coils[1].center_z_m)/2
    # The physical central gap is not silently filled with copper.
    row.potential.sample([[coils[0].inner_radius_m+.001, 0., gap]])


def test_current_and_explicit_reference_changes_invalidate_frozen_source(installed):
    state = _copy_state(installed)
    key = "objective_lens"
    lens = next(lens for lens in state.lenses if lens.key == key)
    original = capture_round_lens_potential(state, key).potential
    lens.percent *= .5
    lowered = capture_round_lens_potential(state, key).potential
    assert lowered.digest != original.digest
    point = [[0., 0., 1.7]]
    np.testing.assert_allclose(lowered.sample(point)["B_t"], .5*original.sample(point)["B_t"], rtol=2e-15)
    _change_part(state, key, air_core_reference_ampere_turns=6000.)
    changed = capture_round_lens_potential(state, key).potential
    assert changed.digest != lowered.digest
    np.testing.assert_allclose(changed.sample(point)["B_t"], original.sample(point)["B_t"], rtol=2e-15)


def test_missing_reference_or_winding_does_not_fall_back_to_gaussian(installed):
    state = _copy_state(installed)
    key = "objective_lens"
    _change_part(state, key, air_core_reference_ampere_turns=None)
    with pytest.raises(ValueError, match="reference_ampere_turns"):
        capture_round_lens_potential(state, key)
    _change_part(state, key, air_core_reference_ampere_turns=3000.)
    coil = next(part for part in state._resolved_assembly.parts if part.data.get("field_source_key") == key
                and part.data.get("mechanical_profile") == "magnetic_excitation_coil")
    _change_part(state, coil.key, field_source_key="unowned-test-coil")
    with pytest.raises(ValueError, match="no explicitly owned winding"):
        capture_round_lens_potential(state, key)


def test_fixed_gaussian_remains_same_and_link_scales_internal_offsets_and_widths(installed):
    state = _copy_state(installed)
    state.round_lens_field_model = "gaussian"
    key = "objective_lens"
    native = next(lens for lens in state.lenses if lens.key == key)
    baseline = FrozenAnalyticField.from_provider(resolve_runtime_lens_field_provider(state, key, native))
    owner = next(part for part in state._resolved_assembly.parts if part.key == key)
    _change_part(state, key, gaussian_reference_coil_length_mm=owner.data["gaussian_reference_coil_length_mm"]*.5)
    fixed = FrozenAnalyticField.from_provider(resolve_runtime_lens_field_provider(state, key, native))
    np.testing.assert_array_equal(fixed.terms_t_m, baseline.terms_t_m)
    state.gaussian_geometry_coupling = "axial_length"
    linked = FrozenAnalyticField.from_provider(resolve_runtime_lens_field_provider(state, key, native))
    for prefix, start in (("upper", 0), ("lower", len(native.upper_gaussian))):
        center = getattr(native, prefix+"_field_center_z_mm")*.001
        for old, new in zip(baseline.terms_t_m[start:start+len(getattr(native, prefix+"_gaussian"))],
                            linked.terms_t_m[start:start+len(getattr(native, prefix+"_gaussian"))]):
            assert new[0] == old[0]
            assert new[1]-center == pytest.approx(2*(old[1]-center), abs=3e-16)
            assert new[2] == 2*old[2]


def test_copper_field_is_not_silently_admitted_to_coherent_operator(installed):
    from temsim.physics.potential_wave_carrier import _admit
    snapshot = capture_selected_column_potentials(installed, round_lens_keys=("objective_lens",))
    with pytest.raises(ValueError, match="not admitted"):
        _admit(snapshot)


def test_mixed_native_and_copper_sources_keep_exact_uncompiled_contribution(installed):
    from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler
    from temsim.physics.analytic_potential import AnalyticMagneticPotential
    from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
    key = next(item.key for item in installed.stigmators if item.enabled)
    snapshot = capture_selected_column_potentials(installed, round_lens_keys=("objective_lens",), auxiliary_keys=(key,))
    # Production capture adds the compact-domain adapter before compilation.
    snapshot = ElectromagneticPotentialSnapshot(snapshot.scene, tuple(
        replace(row, potential=AnalyticPotentialDomain(row.potential))
        if type(row.potential) is AnalyticMagneticPotential else row for row in snapshot.contributions))
    sampler = compile_unified_potential_sampler(snapshot, prefer_plane=True)
    assert sampler.compiled is not None
    assert any(type(row.potential) is CopperCoilAssemblyPotential for row in sampler.remaining)
    points = np.array([[0., 0., 1.7], [1e-7, -2e-7, 1.7]])
    for name in ("A_t_m", "B_t", "E_v_per_m", "phi_v"):
        np.testing.assert_allclose(sampler.sample(points)[name], snapshot.sample(points)[name], rtol=1e-14, atol=1e-25)


def test_nonrigid_winding_shape_and_boolean_diameter_are_rejected(installed):
    state = _copy_state(installed)
    key = "objective_lens"
    coil = next(part for part in state._resolved_assembly.parts if part.data.get("field_source_key") == key
                and part.data.get("mechanical_profile") == "magnetic_excitation_coil")
    _change_part(state, coil.key, mechanical_inner_diameter_mm=True)
    with pytest.raises(ValueError, match="diameters"):
        capture_round_lens_potential(state, key)
    _change_part(state, coil.key, mechanical_inner_diameter_mm=coil.data["mechanical_inner_diameter_mm"],
                 model_3d={"base": {"kind": "box"}})
    with pytest.raises(ValueError, match="non-rigid CAD"):
        capture_round_lens_potential(state, key)
