"""Physical invariants and guarded acceptance for the read-only CM study."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.layout import (
    C3Hardware,
    CorrectorAssembly,
    LayoutConfiguration,
    build_optics_layout,
    resolve_installed_image_corrector_exit_mechanical_axis,
    resolve_objective_entrance_mechanical_axis,
    resolve_objective_exit_mechanical_axis,
    resolve_post_objective_deflector_mechanical_axis,
)
from temsim.column.module_assembly import resolve_module_assembly
from temsim.column.state_layout import layout_configuration_from_state
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.condenser_objective_design import (
    CondenserObjectiveTargets,
    condenser_objective_position_bounds,
    evaluate_condenser_objective_design,
    solve_condenser_objective_design,
)


def test_cm_bounds_follow_real_ac_body_and_upper_pole_faces():
    state = default_state()
    bounds = condenser_objective_position_bounds(state, clearance_mm=0.1)
    half = state.mini_condenser.length_mm / 2
    assert bounds.minimum_center_z_mm - half == pytest.approx(
        state.ac_deflector.lower_surface_z_mm + 0.1)
    assert bounds.maximum_center_z_mm + half == pytest.approx(
        state.objective_lens.upper_pole_piece_center_z_mm
        - state.objective_lens.upper_pole_piece_axial_length_mm / 2 - 0.1)
    with pytest.raises(ValueError, match="mechanical envelope"):
        evaluate_condenser_objective_design(state, center_z_mm=bounds.maximum_center_z_mm + 1)
    state.mini_condenser.enabled = False
    with pytest.raises(ValueError, match="installed, enabled"):
        evaluate_condenser_objective_design(state)


def test_isolated_round_lens_reversal_preserves_circular_beam_size():
    state = default_state()
    for lens in state.lenses:
        lens.enabled = lens is state.mini_condenser
    covariance = np.diag((1e-12, 1e-12, 1e-6, 1e-6))
    result = evaluate_condenser_objective_design(
        state, excitation_magnitude_percent=90,
        input_covariance=covariance,
    )
    # Changing the sign of an isolated axisymmetric field changes rotation,
    # not radial lens power. It cannot alone implement two probe modes.
    assert result.micro.rms_radius_m == pytest.approx(result.nano.rms_radius_m, rel=1e-8)
    assert result.micro.rms_mechanical_angle_rad == pytest.approx(
        result.nano.rms_mechanical_angle_rad, rel=1e-8)
    assert result.polarity_focusing_integral_difference_t2_mm == pytest.approx(0, abs=1e-12)
    assert result.micro.mechanical_transfer[0, 1] == pytest.approx(
        -result.nano.mechanical_transfer[0, 1], abs=1e-9)


def test_same_magnitude_combines_with_objective_without_changing_live_inputs():
    state = default_state()
    before_cm = vars(state.mini_condenser).copy()
    before_objective = vars(state.objective_lens).copy()
    before_tip = state.electron_gun.to_dict()
    result = evaluate_condenser_objective_design(state, excitation_magnitude_percent=90)
    assert result.micro.polarity == 1 and result.nano.polarity == -1
    assert result.micro.mechanical_transfer.shape == (4, 4)
    assert result.micro.canonical_output_transfer.shape == (4, 4)
    assert 0 < result.cm_objective_normalized_overlap < 0.02
    assert result.polarity_focusing_integral_difference_t2_mm > 0
    assert not result.targets_met and result.unmet
    assert not result.micro.mechanical_transfer.flags.writeable
    assert vars(state.mini_condenser) == before_cm
    assert vars(state.objective_lens) == before_objective
    assert state.electron_gun.to_dict() == before_tip


def test_unattainable_targets_return_best_candidate_with_unmet_and_cancel():
    state = default_state()
    result = solve_condenser_objective_design(
        state, candidate_count=2, excitation_magnitude_percent=90,
        input_covariance=np.diag((1e-12, 1e-12, 1e-6, 1e-6)),
        targets=CondenserObjectiveTargets(1e-15, 1e-15),
    )
    assert len(result.candidates) == 2
    assert result.best.objective_value > 1
    assert not result.best.targets_met
    assert len(result.best.unmet) == 2
    with pytest.raises(InterruptedError):
        solve_condenser_objective_design(state, candidate_count=2, cancel_check=lambda: True)


def test_geometry_bound_field_models_are_not_translated_as_analytic_profiles():
    state = default_state()
    state.simulation_mode = "linear_geometry"
    with pytest.raises(ValueError, match="geometry-matched"):
        evaluate_condenser_objective_design(state)


# Expected physical order comes from each selected manifest, not a universal
# assumption about the placement of the Condenser Stigmator upstream of AC.
_COLUMN_LAYOUTS = (
    ("C2", C3Hardware.TWO_CONDENSER, CorrectorAssembly.NO_CORRECTOR, "cetcor", False),
    ("C3", C3Hardware.THREE_CONDENSER, CorrectorAssembly.NO_CORRECTOR, "cetcor", False),
    ("C3 + Probe Corrector", C3Hardware.THREE_CONDENSER, CorrectorAssembly.PROBE_CORRECTOR, "cetcor", True),
    ("C3 + Image Corrector", C3Hardware.THREE_CONDENSER, CorrectorAssembly.IMAGE_CORRECTOR, "cetcor", False),
    ("C3 + Probe Corrector + Image Corrector", C3Hardware.THREE_CONDENSER, CorrectorAssembly.DOUBLE_CORRECTOR, "cetcor", False),
    ("C3 + B-COR", C3Hardware.THREE_CONDENSER, CorrectorAssembly.IMAGE_CORRECTOR, "bcor", False),
    ("C3 + Probe Corrector + B-COR", C3Hardware.THREE_CONDENSER, CorrectorAssembly.DOUBLE_CORRECTOR, "bcor", True),
)


def _selected_column_layout(row):
    name, hardware, corrector, design, ac_first = row
    bare = LayoutConfiguration(
        c3_hardware=hardware,
        c3_excited=hardware is C3Hardware.THREE_CONDENSER,
        corrector=corrector,
        image_corrector_design=design,
    )
    state = default_state()
    assembly = AssemblyCatalog().apply(
        state, AssemblySelection("FEG", name, "No Energy Filter")
    )
    # Nondefault operating settings must survive a read-only layout query.
    state.condenser_stigmator.strength_x_percent = 3.25
    state.mini_condenser.percent += 0.5
    configuration = layout_configuration_from_state(state)
    return state, assembly, configuration, bare, ac_first


@pytest.fixture(scope="module", params=_COLUMN_LAYOUTS, ids=lambda row: row[0])
def column_layout(request):
    return _selected_column_layout(request.param)


def _assert_manifest_envelopes(layout, assembly):
    by_key = {component.key: component for component in layout}
    sample_z = assembly.part("sample").center_z_mm
    for part in assembly.parts:
        if part.key not in by_key:
            assert part.data.get("mechanical_only") or part.data.get("branch_path_only")
            continue
        component = by_key[part.key]
        assert sample_z - component.local_s_center_mm == pytest.approx(part.center_z_mm)
        assert sample_z - component.mechanical.end_s_mm == pytest.approx(part.start_z_mm)
        assert sample_z - component.mechanical.start_s_mm == pytest.approx(part.end_z_mm)
    return by_key


def test_bare_layout_preserves_selected_manifest_geometry_for_every_column(column_layout):
    state, assembly, configuration, bare, _ = column_layout
    before = capture_instrument_snapshot(state).digest
    bare_assembly = resolve_module_assembly(bare)
    assert bare_assembly.selected_module_paths == assembly.selected_module_paths
    bare_rows = _assert_manifest_envelopes(build_optics_layout(bare), bare_assembly)
    runtime_rows = _assert_manifest_envelopes(build_optics_layout(configuration), assembly)
    assert bare_rows.keys() == runtime_rows.keys()
    for key in bare_rows:
        assert bare_rows[key].mechanical == runtime_rows[key].mechanical
        assert bare_rows[key].local_s_center_mm == runtime_rows[key].local_s_center_mm
    assert capture_instrument_snapshot(state).digest == before


def test_objective_entrance_follows_real_order_and_keeps_stigmator_position(column_layout):
    state, assembly, configuration, bare, ac_first = column_layout
    before = capture_instrument_snapshot(state).digest
    first = ("ac_deflector", "condenser_stigmator") if ac_first else (
        "condenser_stigmator", "ac_deflector")
    expected_order = (*first, "mini_condenser", "objective_upper_pole", "sample")
    source_z = assembly.part("feg_tip").center_z_mm
    for selected in (configuration, bare):
        axis = resolve_objective_entrance_mechanical_axis(selected)
        assert axis.axis_order == expected_order
        for key in expected_order:
            part = assembly.part(key)
            placement = axis.by_key[key]
            assert (placement.start_z_mm, placement.center_z_mm, placement.end_z_mm) == pytest.approx(
                (part.start_z_mm - source_z, part.center_z_mm - source_z,
                 part.end_z_mm - source_z))
        for upstream, downstream in zip(expected_order, expected_order[1:]):
            assert axis.clearance_between(upstream, downstream) == pytest.approx(
                assembly.part(downstream).start_z_mm - assembly.part(upstream).end_z_mm)
        assert axis.by_key["sample"].center_z_mm - axis.by_key["condenser_stigmator"].center_z_mm == pytest.approx(147.7)
    assert capture_instrument_snapshot(state).digest == before


def test_objective_entrance_still_rejects_real_overlap_for_every_column(column_layout):
    state, assembly, configuration, _, _ = column_layout
    before = capture_instrument_snapshot(state).digest
    stigma = assembly.part("condenser_stigmator")
    shift = assembly.part("ac_deflector").center_z_mm - stigma.center_z_mm
    overlapping = replace(stigma, start_z_mm=stigma.start_z_mm + shift,
        center_z_mm=stigma.center_z_mm + shift, end_z_mm=stigma.end_z_mm + shift)
    changed_assembly = replace(assembly, parts=tuple(
        overlapping if part.key == stigma.key else part for part in assembly.parts))
    with pytest.raises(ValueError, match="Unapproved mechanical overlap:.*(?:ac_deflector|condenser_stigmator)"):
        resolve_objective_entrance_mechanical_axis(
            replace(configuration, resolved_assembly=changed_assembly))
    assert capture_instrument_snapshot(state).digest == before


def test_layout_and_entrance_support_no_argument_public_entry_points():
    assert build_optics_layout() == build_optics_layout(LayoutConfiguration())
    assert resolve_objective_entrance_mechanical_axis() == resolve_objective_entrance_mechanical_axis(LayoutConfiguration())


def test_exit_diagnostics_support_no_argument_public_entry_points():
    for helper in (resolve_objective_exit_mechanical_axis,
                   resolve_post_objective_deflector_mechanical_axis):
        assert helper() == helper(LayoutConfiguration())
    assert resolve_installed_image_corrector_exit_mechanical_axis() == (
        resolve_installed_image_corrector_exit_mechanical_axis(
            LayoutConfiguration(corrector=CorrectorAssembly.IMAGE_CORRECTOR)))


@pytest.mark.parametrize("column", [row for row in _COLUMN_LAYOUTS
    if row[2] not in (CorrectorAssembly.IMAGE_CORRECTOR, CorrectorAssembly.DOUBLE_CORRECTOR)],
    ids=lambda row: row[0])
def test_image_exit_requires_an_installed_corrector(column):
    _, hardware, corrector, design, _ = column
    configuration = LayoutConfiguration(c3_hardware=hardware,
        c3_excited=hardware is C3Hardware.THREE_CONDENSER,
        corrector=corrector, image_corrector_design=design)
    with pytest.raises(ValueError, match="requires an installed Image Corrector"):
        resolve_installed_image_corrector_exit_mechanical_axis(configuration)


def _configuration_with_part_replacements(configuration, assembly, *parts):
    replacements = {part.key: part for part in parts}
    changed = replace(assembly, parts=tuple(
        replacements.get(part.key, part) for part in assembly.parts))
    return replace(configuration, resolved_assembly=changed)


def _part_at_center(part, center_z_mm):
    shift = center_z_mm - part.center_z_mm
    return replace(part, start_z_mm=part.start_z_mm + shift,
                   center_z_mm=center_z_mm, end_z_mm=part.end_z_mm + shift)


def _assert_axis_matches_physical_parts(axis, assembly, keys):
    # Expected positions come from the selected assembly, independent of the
    # helper's candidate list, sorting and shared-channel handling.
    expected = sorted((assembly.part(key) for key in keys),
                      key=lambda part: part.center_z_mm)
    assert axis.axis_order == tuple(part.key for part in expected)
    assert set(axis.by_key) == set(keys)
    source_z = assembly.part("feg_tip").center_z_mm
    for part in expected:
        placement = axis.by_key[part.key]
        assert (placement.start_z_mm, placement.center_z_mm,
                placement.end_z_mm) == pytest.approx((
                    part.start_z_mm - source_z, part.center_z_mm - source_z,
                    part.end_z_mm - source_z))
    for upstream, downstream in zip(expected, expected[1:]):
        assert axis.clearance_between(upstream.key, downstream.key) == pytest.approx(
            downstream.start_z_mm - upstream.end_z_mm)


def test_objective_exit_diagnostics_use_shared_hardware_once_for_every_column(column_layout):
    state, assembly, configuration, bare, _ = column_layout
    before = capture_instrument_snapshot(state).digest
    exit_keys = {"sample", "objective_aperture", "objective_lower_pole",
                 "objective_stigmator", "image_diffraction_deflector"}
    post_keys = {"objective_stigmator", "image_diffraction_deflector"}
    for selected in (configuration, bare):
        for helper, keys in (
            (resolve_objective_exit_mechanical_axis, exit_keys),
            (resolve_post_objective_deflector_mechanical_axis, post_keys),
        ):
            axis = helper(selected)
            _assert_axis_matches_physical_parts(axis, assembly, keys)
            assert "descan_deflector" not in axis.by_key
            assert axis.axis_order.count("image_diffraction_deflector") == 1
    assert capture_instrument_snapshot(state).digest == before


def test_post_objective_diagnostic_follows_edited_positions_and_dimensions(column_layout):
    state, assembly, configuration, _, _ = column_layout
    before = capture_instrument_snapshot(state).digest
    stigma = assembly.part("objective_stigmator")
    host = assembly.part("image_diffraction_deflector")
    baseline = resolve_post_objective_deflector_mechanical_axis(configuration)
    baseline_gap = baseline.clearance_between(*baseline.axis_order)

    # Exchanging separated physical bodies must reverse their displayed order;
    # the stale Descan channel coordinates must not create another body.
    exchanged = _configuration_with_part_replacements(configuration, assembly,
        _part_at_center(stigma, host.center_z_mm),
        _part_at_center(host, stigma.center_z_mm))
    swapped = resolve_post_objective_deflector_mechanical_axis(exchanged)
    assert swapped.axis_order == tuple(reversed(baseline.axis_order))
    assert swapped.clearance_between(*swapped.axis_order) == pytest.approx(baseline_gap)
    _assert_axis_matches_physical_parts(swapped, exchanged.resolved_assembly,
        {"objective_stigmator", "image_diffraction_deflector"})

    # A 2 mm longer body consumes 1 mm more space on each side of its centre.
    longer = replace(stigma, start_z_mm=stigma.start_z_mm - 1,
                     end_z_mm=stigma.end_z_mm + 1, length_mm=stigma.length_mm + 2)
    resized = resolve_post_objective_deflector_mechanical_axis(
        _configuration_with_part_replacements(configuration, assembly, longer))
    assert resized.axis_order == baseline.axis_order
    assert resized.clearance_between(*resized.axis_order) == pytest.approx(baseline_gap - 1)
    assert capture_instrument_snapshot(state).digest == before


@pytest.mark.parametrize("helper", (
    resolve_objective_exit_mechanical_axis,
    resolve_post_objective_deflector_mechanical_axis,
), ids=("objective-exit", "post-objective"))
def test_objective_exit_diagnostics_still_reject_real_body_overlap(column_layout, helper):
    state, assembly, configuration, _, _ = column_layout
    before = capture_instrument_snapshot(state).digest
    overlap = _part_at_center(assembly.part("objective_stigmator"),
                              assembly.part("image_diffraction_deflector").center_z_mm)
    changed = _configuration_with_part_replacements(configuration, assembly, overlap)
    with pytest.raises(ValueError, match="Unapproved mechanical overlap:.*(?:objective_stigmator|image_diffraction_deflector)"):
        helper(changed)
    assert capture_instrument_snapshot(state).digest == before


@pytest.fixture(scope="module", params=[row for row in _COLUMN_LAYOUTS
    if row[2] in (CorrectorAssembly.IMAGE_CORRECTOR, CorrectorAssembly.DOUBLE_CORRECTOR)],
    ids=lambda row: row[0])
def image_corrector_layout(request):
    return _selected_column_layout(request.param)


def test_image_corrector_exit_checks_materials_not_field_channels(image_corrector_layout):
    state, assembly, configuration, bare, _ = image_corrector_layout
    before = capture_instrument_snapshot(state).digest
    required_materials = {
        "image_diffraction_deflector", "image_tl11_lens", "image_tl12_lens",
        "image_tl21_lens", "image_tl22_lens", "image_hp1_hexapole",
        "image_hp2_hexapole", "image_adapter_lens", "selected_area_aperture",
        "image_dp11_coils", "image_dp12_coils", "image_dph1_coils",
        "image_dp21_coils", "image_dp22_coils", "image_dph2_coils",
        "image_ish_coils", "image_dsh_coils",
    }
    if configuration.image_corrector_design == "bcor":
        required_materials.update({"image_tl31_lens", "image_tl32_lens",
                                   "image_hp3_hexapole"})
    field_channels = {
        "image_hpol_hexapole", "image_qpol_quadrupole",
        "image_dp11_deflector", "image_dp12_deflector",
        "image_dph1_deflector", "image_dph2_deflector",
        "image_dp21_deflector", "image_dp22_deflector",
        "image_ish_deflector", "image_dsh_deflector", "image_dstg_quadrupole",
    }
    for selected in (configuration, bare):
        axis = resolve_installed_image_corrector_exit_mechanical_axis(selected)
        assert required_materials <= set(axis.by_key)
        assert field_channels.isdisjoint(axis.by_key)
        assert "image_sad_plane" not in axis.by_key
        assert len(axis.axis_order) == len(set(axis.axis_order))
        centers = [axis.by_key[key].center_z_mm for key in axis.axis_order]
        assert centers == sorted(centers)
        assert axis.clearance_between("image_dp11_coils", "image_tl11_lens") == pytest.approx(0.5)
        assert axis.clearance_between("image_dp21_coils", "image_dp22_coils") == pytest.approx(0.5)
        for number in (1, 2):
            coil_key = f"image_dph{number}_coils"
            placement = axis.by_key[coil_key]
            assert placement.nested_parent_key == f"image_hp{number}_hexapole"
            assert placement.nesting_reason
            assert coil_key not in axis.axis_order
        # The physical aperture stays at the selected real observation plane;
        # omitting a virtual marker must not erase the actual aperture body.
        aperture = assembly.part("selected_area_aperture")
        assert axis.by_key[aperture.key].center_z_mm == pytest.approx(
            aperture.center_z_mm - assembly.part("feg_tip").center_z_mm)
    assert capture_instrument_snapshot(state).digest == before


def test_image_corrector_exit_follows_aperture_position_and_size(image_corrector_layout):
    state, assembly, configuration, _, _ = image_corrector_layout
    before = capture_instrument_snapshot(state).digest
    baseline = resolve_installed_image_corrector_exit_mechanical_axis(configuration)
    aperture = assembly.part("selected_area_aperture")
    assert baseline.axis_order[-1] == aperture.key
    previous_key = baseline.axis_order[-2]
    gap = baseline.clearance_between(previous_key, aperture.key)

    moved = _part_at_center(aperture, aperture.center_z_mm + 2)
    translated = resolve_installed_image_corrector_exit_mechanical_axis(
        _configuration_with_part_replacements(configuration, assembly, moved))
    assert translated.axis_order == baseline.axis_order
    assert translated.clearance_between(previous_key, aperture.key) == pytest.approx(gap + 2)

    longer = replace(aperture, start_z_mm=aperture.start_z_mm - 1,
                     end_z_mm=aperture.end_z_mm + 1, length_mm=aperture.length_mm + 2)
    resized = resolve_installed_image_corrector_exit_mechanical_axis(
        _configuration_with_part_replacements(configuration, assembly, longer))
    assert resized.axis_order == baseline.axis_order
    assert resized.clearance_between(previous_key, aperture.key) == pytest.approx(gap - 1)
    assert capture_instrument_snapshot(state).digest == before


def test_image_corrector_exit_still_rejects_real_aperture_lens_overlap(image_corrector_layout):
    state, assembly, configuration, _, _ = image_corrector_layout
    before = capture_instrument_snapshot(state).digest
    aperture = _part_at_center(assembly.part("selected_area_aperture"),
                              assembly.part("image_tl12_lens").center_z_mm)
    changed = _configuration_with_part_replacements(configuration, assembly, aperture)
    with pytest.raises(ValueError, match="Unapproved mechanical overlap:.*(?:selected_area_aperture|image_tl12_lens)"):
        resolve_installed_image_corrector_exit_mechanical_axis(changed)
    assert capture_instrument_snapshot(state).digest == before


def test_image_corrector_coil_material_length_controls_clearance(image_corrector_layout):
    state, assembly, configuration, _, _ = image_corrector_layout
    before = capture_instrument_snapshot(state).digest
    coil = assembly.part("image_dp21_coils")
    assert coil.length_mm == 14.0
    assert coil.data["winding_axial_length_mm"] == 3.0
    longer = replace(coil, data={**coil.data, "winding_axial_length_mm": 3.5})
    axis = resolve_installed_image_corrector_exit_mechanical_axis(
        _configuration_with_part_replacements(configuration, assembly, longer))
    assert axis.by_key[coil.key].end_z_mm - axis.by_key[coil.key].start_z_mm == pytest.approx(3.5)
    # Centres are 3.5 mm apart: 3.5 mm + 3 mm copper leaves 0.25 mm.
    assert axis.clearance_between(coil.key, "image_dp22_coils") == pytest.approx(0.25)
    touching = replace(coil, data={**coil.data, "winding_axial_length_mm": 4.0})
    axis = resolve_installed_image_corrector_exit_mechanical_axis(
        _configuration_with_part_replacements(configuration, assembly, touching))
    assert axis.clearance_between(coil.key, "image_dp22_coils") == pytest.approx(0.0)
    crossing = replace(coil, data={**coil.data, "winding_axial_length_mm": 4.2})
    with pytest.raises(ValueError, match="Unapproved mechanical overlap:.*image_dp21_coils.*image_dp22_coils"):
        resolve_installed_image_corrector_exit_mechanical_axis(
            _configuration_with_part_replacements(configuration, assembly, crossing))
    assert capture_instrument_snapshot(state).digest == before


def test_image_corrector_coil_end_cover_consumes_real_clearance(image_corrector_layout):
    from temsim.deflector_support_geometry import END_STATUS, SUPPORT_PROFILE, SUPPORT_SOURCE

    state, assembly, configuration, _, _ = image_corrector_layout
    before = capture_instrument_snapshot(state).digest
    coil = assembly.part("image_dp21_coils")
    # Explicit synthetic support edit: copper and its 14 mm field allocation
    # stay fixed; only a downstream 0.25 mm material cover is added.
    supported_data = {
        **coil.data,
        "winding_support_profile": SUPPORT_PROFILE,
        "winding_support_carrier_thickness_mm": 0.04,
        "winding_support_cover_thickness_mm": 0.04,
        "winding_support_end_cover_thickness_mm": 0.25,
        "winding_support_end_cover_side": "downstream",
        "winding_support_end_cover_geometry_status": END_STATUS,
        "winding_support_end_cover_source": SUPPORT_SOURCE,
        "winding_support_end_cover_source_note": "Synthetic regression geometry, not OEM evidence.",
    }
    covered = replace(coil, data=supported_data)
    axis = resolve_installed_image_corrector_exit_mechanical_axis(
        _configuration_with_part_replacements(configuration, assembly, covered))
    assert axis.by_key[coil.key].end_z_mm - axis.by_key[coil.key].start_z_mm == pytest.approx(3.25)
    assert axis.clearance_between(coil.key, "image_dp22_coils") == pytest.approx(0.25)
    crossing = replace(coil, data={**supported_data,
        "winding_support_end_cover_thickness_mm": 0.6})
    with pytest.raises(ValueError, match="Unapproved mechanical overlap:.*image_dp21_coils.*image_dp22_coils"):
        resolve_installed_image_corrector_exit_mechanical_axis(
            _configuration_with_part_replacements(configuration, assembly, crossing))
    assert capture_instrument_snapshot(state).digest == before


def test_image_corrector_replacement_uses_declared_envelope_not_inactive_winding(image_corrector_layout):
    state, assembly, configuration, _, _ = image_corrector_layout
    before = capture_instrument_snapshot(state).digest
    coil = assembly.part("image_dp12_coils")
    replacement = replace(coil, data={
        **coil.data,
        "winding_axial_length_mm": 999.0,
        "model_3d": {
            "schema_version": 1,
            "base": {"kind": "box", "width_mm": 1.0,
                     "height_mm": 1.0, "length_mm": 2.0},
            "features": [],
        },
    })
    axis = resolve_installed_image_corrector_exit_mechanical_axis(
        _configuration_with_part_replacements(configuration, assembly, replacement))
    actual = axis.by_key[coil.key]
    source_z = assembly.part("feg_tip").center_z_mm
    # This helper validates the declared installation envelope, not the posed
    # replacement CAD. Inactive native winding settings must not be evaluated.
    assert (actual.start_z_mm, actual.center_z_mm, actual.end_z_mm) == pytest.approx((
        coil.start_z_mm - source_z, coil.center_z_mm - source_z,
        coil.end_z_mm - source_z))
    assert capture_instrument_snapshot(state).digest == before


@pytest.mark.parametrize("number", (1, 2))
def test_image_corrector_nested_winding_cannot_leave_hexapole(image_corrector_layout, number):
    state, assembly, configuration, _, _ = image_corrector_layout
    before = capture_instrument_snapshot(state).digest
    coil = assembly.part(f"image_dph{number}_coils")
    parent = assembly.part(f"image_hp{number}_hexapole")
    outside = _part_at_center(coil, parent.end_z_mm)
    with pytest.raises(ValueError, match=f"Nested component {coil.key} is not fully contained"):
        resolve_installed_image_corrector_exit_mechanical_axis(
            _configuration_with_part_replacements(configuration, assembly, outside))
    assert capture_instrument_snapshot(state).digest == before


@pytest.mark.parametrize("installation", ["integrated", "standalone"])
def test_mini_fallback_poles_use_selected_installation_gap_and_bore(installation):
    from temsim.column.layout import _build_optics_layout_metadata
    from temsim.optics.mini_condenser import MINI_CONDENSER_DEFINITION

    original = MINI_CONDENSER_DEFINITION.geometry_for(installation)
    geometry = replace(original, pole_gap_mm=original.pole_gap_mm + 0.4,
                       bore_diameter_mm=original.bore_diameter_mm + 0.3)
    definition = replace(MINI_CONDENSER_DEFINITION,
                         **{f"{installation}_geometry": geometry})
    configuration = LayoutConfiguration(mini_condenser_component=definition,
        corrector=CorrectorAssembly.PROBE_CORRECTOR if installation == "integrated"
        else CorrectorAssembly.NO_CORRECTOR)
    rows = {row.key: row for row in _build_optics_layout_metadata(configuration)}
    upper, lower = rows["mini_condenser_upper_pole"], rows["mini_condenser_lower_pole"]
    assert upper.mechanical.start_s_mm - lower.mechanical.end_s_mm == pytest.approx(geometry.pole_gap_mm)
    for pole in (upper, lower):
        assert pole.mechanical_shape.active_diameter_mm == geometry.bore_diameter_mm
        assert pole.mechanical_shape.axial_length_mm == pytest.approx(
            (geometry.mechanical_length_mm - geometry.pole_gap_mm) / 2)
    assert rows["mini_condenser"].mechanical_shape.active_length_mm == geometry.pole_gap_mm


def test_ac_descan_field_metadata_matches_finite_shared_stations(column_layout):
    from temsim.physics.instrument_magnetic import column_dipole_fields

    state, assembly, _, _, _ = column_layout
    fields = column_dipole_fields(state)
    for key in ("ac_deflector", "descan_deflector"):
        status = assembly.part(key).data["field_geometry_status"]
        assert "finite_uniform_fields" in status and "not_measured_coil_map" in status
        assert "ideal_kicks" not in status
    for host in (state.ac_deflector, state.image_diffraction_deflector):
        stations = [field for field in fields if field.key.startswith(f"{host.key}:")]
        assert len(stations) == 2
        assert [field.event_z_mm for field in stations] == [host.upper_z_mm, host.lower_z_mm]
        for field in stations:
            assert (field.upper_m - field.lower_m) * 1000 == pytest.approx(15.0)
    assert not any(field.key.startswith("descan_deflector:") for field in fields)
    for field in fields:
        if field.key.startswith("image_diffraction_deflector:"):
            assert set(field.drive_keys) == {"image_diffraction_deflector", "descan_deflector"}
    tl12 = next((part for part in assembly.parts if part.key == "probe_tl12_lens"), None)
    if tl12 is not None:
        assert tl12.name == "TL12 Transfer Lens"
