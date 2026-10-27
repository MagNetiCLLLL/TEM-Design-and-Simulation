"""Worker capture preserves real native/FEM source ownership and inputs."""
from dataclasses import replace
from types import MethodType

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.component_representation import SHARED_DEFLECTOR_HOSTS
from temsim.optics.column import default_state
from temsim.physics.potential_capture import capture_round_lens_potential, capture_selected_column_potentials


VARIANTS = ("C2", "C3", "C3_ImageCorrector", "C3_ProbeCorrector",
            "C3_ProbeCorrector_ImageCorrector", "C3_BCOR", "C3_ProbeCorrector_BCOR")


@pytest.fixture(scope="module")
def catalog():
    return AssemblyCatalog()


@pytest.fixture(scope="module", params=VARIANTS)
def assembled(catalog, request):
    state = default_state()
    option = next(option for option in catalog.columns if option.file == f"column/{request.param}.toml")
    catalog.apply(state, AssemblySelection("FEG", option.name, "No Energy Filter"))
    state.image_diffraction_deflector.upper_x_mrad = .02
    state.descan_deflector.enabled = True
    state.descan_deflector.kick_x_mrad = -.01
    return state


def _all_supported_keys(state):
    lenses = tuple(lens.key for lens in state.lenses if lens.enabled)
    # These two objects are declared reference coordinates, without their own
    # finite magnetic source. Descan instead shares the Image/Diffraction host.
    references = {"image_sad_plane", "probe_dp12_scan_deflector"}
    auxiliary = tuple(component.key for component in (
        *state.stigmators, *state.corrector_elements, *state.deflectors)
        if component.enabled and component.key not in references | set(SHARED_DEFLECTOR_HOSTS))
    return lenses, auxiliary


def test_seven_columns_capture_every_requested_native_source_and_shared_descan_once(assembled):
    from temsim.physics.instrument_magnetic import column_dipole_fields
    state = assembled
    lenses, auxiliary = _all_supported_keys(state)
    snapshot = capture_selected_column_potentials(state, round_lens_keys=lenses, auxiliary_keys=auxiliary)
    by_id = {row.source_id: row for row in snapshot.contributions}
    assert len(by_id) == len(snapshot.contributions)
    assert set(lenses) <= set(by_id)
    captured_aux = {row.potential.field.component_key for row in snapshot.contributions if row.source_id not in lenses}
    assert captured_aux == set(auxiliary)
    host = state.image_diffraction_deflector
    expected_coils = {coil.key: coil for coil in column_dipole_fields(state) if coil.key.startswith(host.key+":")}
    captured_coils = {key: row for key, row in by_id.items() if key.startswith(host.key+":")}
    assert set(captured_coils) == set(expected_coils) and len(captured_coils) == 2
    assert not any("descan_deflector" in row.source_id for row in snapshot.contributions)
    assert snapshot.scene.owner_for("descan_deflector") == host.key
    for key, row in captured_coils.items():
        assert row.owner_ids == (key,)
        assert set(row.potential.field.drive_keys) == {host.key, "descan_deflector"}
        point = np.array((0., 0., expected_coils[key].event_z_mm*.001))
        np.testing.assert_allclose(row.potential.sample(point)["B_t"],
                                   expected_coils[key].field_at_global_positions_t(point), atol=1e-18)
    # Every captured native potential agrees with the actual same captured
    # field at an interior point; no unrelated optical settings are retuned.
    for row in snapshot.contributions:
        field = row.potential.field
        radius = min(field.radial_support_m*.1, 1e-6)
        point = np.array((radius, -.7*radius, np.mean(field.native_support_mm)*.001))
        point = point@field.registration.rotation_array.T+field.registration.origin_array_m
        np.testing.assert_allclose(row.potential.sample(point)["B_t"], field.field_at_global_positions_t(point), rtol=2e-13, atol=2e-17)


def test_seven_columns_reject_shared_channel_and_reference_plane_as_independent_source(assembled):
    with pytest.raises(ValueError, match="select shared physical host"):
        capture_selected_column_potentials(assembled, auxiliary_keys=("descan_deflector",))
    with pytest.raises(ValueError, match="unsupported auxiliary"):
        capture_selected_column_potentials(assembled, auxiliary_keys=("probe_dp12_scan_deflector",))


def test_linear_fem_capture_preserves_cached_raw_potential_and_excitation():
    from test_magnetic_circuits import _state, _shared_rows
    from temsim.physics.axisymmetric_magnetostatics import solve_geometry_field_details, _solve_bound_geometry
    from temsim.physics.lens_field_provider import resolve_runtime_lens_field_provider
    state = _state(_shared_rows())
    provider = resolve_runtime_lens_field_provider(state, "a", state.lenses[0])
    legacy, solved = solve_geometry_field_details(provider.binding, state.lens_field_map_descriptors["a"])
    initial_misses = _solve_bound_geometry.cache_info().misses
    row = capture_round_lens_potential(state, "a")
    assert _solve_bound_geometry.cache_info().misses == initial_misses
    np.testing.assert_array_equal(row.potential.a_phi_tm, solved.a_phi_tm)
    assert row.potential.source_scale == 1.
    assert row.owner_ids == ("a",)
    assert row.potential.provenance["solve_source_sha256"] == legacy.provenance.source_sha256
    assert row.potential.provenance["settings"]["ampere_turns"] == 100
    point = np.array((.0003, .0004, -.005))
    before = row.potential.sample(point)
    state.lenses[0].percent = 50.
    changed = capture_round_lens_potential(state, "a")
    assert _solve_bound_geometry.cache_info().misses == initial_misses
    assert changed.potential.digest != row.potential.digest
    assert changed.potential.source_scale == .5
    for name, value in before.items():
        np.testing.assert_allclose(changed.potential.sample(point)[name], .5*value, atol=1e-18)
        np.testing.assert_array_equal(row.potential.sample(point)[name], value)
    for recipe in state.lens_field_map_descriptors.values():
        recipe["relative_permeability"] = 80.
    rematerialized = capture_round_lens_potential(state, "a")
    assert _solve_bound_geometry.cache_info().misses == initial_misses+1
    assert rematerialized.potential.digest != changed.potential.digest


def test_joint_nonlinear_capture_counts_total_once_even_when_member_selected_first():
    from test_nonlinear_magnetostatics import _shared
    state = _shared()
    snapshot = capture_selected_column_potentials(state, round_lens_keys=("b", "a"))
    row, = snapshot.contributions
    assert row.source_id == "joint_nonlinear:a+b"
    assert row.owner_ids == ("a", "b")
    assert row.potential.source_scale == 1.
    assert row.potential.provenance["settings"]["channel_ampere_turns"] == {"a": 7000., "b": 7000.}
    same = capture_selected_column_potentials(state, round_lens_keys=("a",))
    assert row.potential.digest == same.contributions[0].potential.digest
    assert np.any(snapshot.sample((.0003, .0004, -.005))["B_t"])
    previous = row.potential.to_dict()
    state.lenses[1].percent = 50.
    changed = capture_selected_column_potentials(state, round_lens_keys=("a", "b"))
    assert len(changed.contributions) == 1
    assert changed.numerical_digest != snapshot.numerical_digest
    assert changed.contributions[0].potential.provenance["settings"]["channel_ampere_turns"]["b"] == 3500.
    assert row.potential.to_dict() == previous


def test_b_only_import_is_rejected_instead_of_reconstructed(tmp_path):
    from test_lens_field_provider import _write_axisymmetric_map
    from temsim.physics.lens_field_provider import bind_imported_lens_field_map, load_magnetic_field_map
    state = default_state()
    lens = state.objective_lens
    path = tmp_path/"b_only.npz"
    binding = _write_axisymmetric_map(path, state, lens)
    field_map = load_magnetic_field_map(path, geometry_binding=binding,
        provenance_kind="fem", reference_excitation_percent=100.)
    bind_imported_lens_field_map(state, lens.key, field_map, native_provider=lens)
    with pytest.raises(ValueError, match="B-only maps"):
        capture_round_lens_potential(state, lens.key)


def test_overridden_native_lens_is_rejected_and_unselected_multipole_does_not_block_capture():
    state = default_state()
    objective = state.objective_lens
    objective.magnetic_field_t = MethodType(lambda self, z: np.ones_like(z), objective)
    with pytest.raises(ValueError, match="field law is not admitted"):
        capture_round_lens_potential(state, objective.key)
    # No round lens was selected, so this overridden law is irrelevant here.
    other = state.objective_stigmator
    other.quadrupole_tensor_m2 = MethodType(lambda self, z: (z, z, z), other)
    selected = capture_selected_column_potentials(state, auxiliary_keys=(state.condenser_stigmator.key,))
    assert [row.source_id for row in selected.contributions] == [state.condenser_stigmator.key]
    with pytest.raises(ValueError, match="native Gaussian"):
        capture_selected_column_potentials(state, auxiliary_keys=(other.key,))


def test_bcor_capture_rejects_class_level_changes_to_the_native_field_law(catalog, monkeypatch):
    from temsim.optics.hexapole import HexapoleComponent
    from temsim.optics.round_lens import RoundLensComponent
    state = default_state()
    option = next(option for option in catalog.columns if option.file == "column/C3_BCOR.toml")
    catalog.apply(state, AssemblySelection("FEG", option.name, "No Energy Filter"))
    # Populate the immutable original-method contracts before replacement.
    capture_selected_column_potentials(state, round_lens_keys=("image_tl31_lens",),
                                      auxiliary_keys=("image_hp3_hexapole",))
    with monkeypatch.context() as patch:
        patch.setattr(HexapoleComponent, "hexapole_strength_components_m3", lambda self, z: (z*0., z*0.))
        with pytest.raises(ValueError, match="native Gaussian"):
            capture_selected_column_potentials(state, auxiliary_keys=("image_hp3_hexapole",))
    with monkeypatch.context() as patch:
        patch.setattr(RoundLensComponent, "magnetic_field_t", lambda self, z: np.zeros_like(z))
        with pytest.raises(ValueError, match="field law is not admitted"):
            capture_round_lens_potential(state, "image_tl31_lens")


def test_selected_native_parameter_changes_detach_and_invalidate_snapshot():
    state = default_state()
    keys = (state.objective_lens.key,)
    old = capture_selected_column_potentials(state, round_lens_keys=keys)
    before = old.to_dict()
    state.objective_lens.percent *= .9
    new = capture_selected_column_potentials(state, round_lens_keys=keys)
    assert new.numerical_digest != old.numerical_digest
    state.objective_lens.upper_a_mm *= 1.1
    resized = capture_selected_column_potentials(state, round_lens_keys=keys)
    assert resized.numerical_digest != new.numerical_digest
    assert old.to_dict() == before


def test_explicit_electric_capture_freezes_actual_provider_without_gun_solve(monkeypatch):
    from test_electric_potential_snapshot import planar, points
    from temsim.physics import instrument_electric
    provider = planar()
    called = []
    def capture(state):
        called.append(state)
        return provider
    monkeypatch.setattr(instrument_electric, "capture_instrument_electric_field", capture)
    state = default_state()
    snapshot = capture_selected_column_potentials(state, include_electric=True)
    assert called == [state]
    row, = snapshot.contributions
    assert row.source_id == "instrument_electric" and row.kind == "electric"
    phi, electric = provider.interpolate(points())
    sampled = snapshot.sample(points())
    np.testing.assert_array_equal(sampled["phi_v"], phi)
    np.testing.assert_array_equal(sampled["E_v_per_m"], electric)
    assert not np.any(sampled["B_t"])


@pytest.mark.parametrize("options", [
    {"round_lens_keys": "objective_lens"}, {"round_lens_keys": ("objective_lens", "objective_lens")},
    {"round_lens_keys": ("missing",)}, {"auxiliary_keys": ("missing",)},
    {"include_electric": "false"},
])
def test_invalid_selections_fail_explicitly(options):
    with pytest.raises(ValueError):
        capture_selected_column_potentials(default_state(), **options)
