"""Reviewed real-device checks that cannot establish GPU evidence on CPU CI.

These cases stay collected and their failures still fail the lane. Only a
normal pytest skip may be recorded separately when the runner explicitly
allows GPU skips. Policy/emulation tests are deliberately not included.
"""

GPU_TEST_FUNCTIONS = {
    "test_intrinsic_aberration_integration.py": (
        "test_full_column_wave_uses_intrinsic_screen_and_resumes_once_on_gpu",
    ),
    "test_intrinsic_aberration_operator.py": (
        "test_real_cuda_intrinsic_action_matches_cpu_complex_field",
    ),
    "test_intrinsic_ray_device.py": (
        "test_real_cuda_each_intrinsic_term_keeps_cpu_action",
        "test_real_cuda_combined_intrinsic_action_with_analytic_and_multipole_gauge",
        "test_real_cuda_native_field_and_potential_derivatives_match_cpu",
        "test_real_cuda_legacy_cs_geometric_screen_remains_identical",
        "test_real_cuda_zero_and_invalid_input_have_cpu_semantics",
    ),
    "test_intrinsic_gpu_segments.py": (
        "test_native_actions_and_intrinsics_execute_once_on_gpu",
        "test_gpu_events_preserve_electric_energy_and_clock_then_resume",
        "test_empty_saved_history_and_event_only_requests",
        "test_cancel_keeps_input_arrays_unchanged",
        "test_posed_cs_then_tilted_intrinsic_uses_real_fast_cuda_spans",
        "test_core_require_gpu_records_complete_intrinsic_execution",
    ),
    "test_intrinsic_particle_backend.py": (
        "test_top_level_backend_runs_sparse_intrinsic_events_on_real_cuda",
        "test_electric_intrinsic_gpu_preserves_energy_time_and_after_screen_resume",
    ),
    "test_particle_gpu_reference.py": (
        "test_gpu_electric_interpolation_matches_scalar_cells_and_boundaries",
        "test_native_magnetic_gpu_matches_reference_and_keeps_inputs",
        "test_native_electric_gpu_matches_reference_energy_and_flight_time",
        "test_aligned_magnetic_intervals_keep_timing_and_stopped_nan_rays",
        "test_zero_interval_records_actions_once_before_checkpoint",
        "test_gpu_cancellation_after_device_actions_preserves_original_inputs",
    ),
    "test_part_model_gpu_view.py": (
        "test_real_gpu_runtime_failure_falls_back_in_the_same_frame",
        "test_real_gpu_keeps_cad_triangles_and_interior_lighting",
        "test_real_gpu_fitted_small_part_keeps_detail_at_column_length_coordinates",
        "test_real_gpu_camera_changes_reuse_geometry_without_cpu_rasterization",
        "test_real_gpu_exact_face_picking_preserves_display_and_reuses_pick_buffer",
        "test_real_gpu_section_invalidates_picking_and_preserves_source_faces",
        "test_real_gpu_edge_pick_does_not_select_occluded_back_rim",
        "test_real_gpu_uses_visible_depth_range_when_fitting_a_tiny_component",
        "test_real_gpu_uses_temporary_cpu_for_unresolvable_depth_then_resumes_gpu",
    ),
    "test_radial_fftlog.py": (
        "test_explicit_cuda_has_same_complex_field_and_no_implicit_fallback",
    ),
    "test_wave_support.py": (
        "test_cupy_compaction_matches_cpu_if_available",
        "test_cupy_spectral_projection_matches_cpu_if_available",
    ),
    "test_plane_wave_storage.py": (
        "test_real_gpu_public_input_and_default_replace_own_their_storage",
        "test_real_gpu_owned_operator_output_avoids_copy_and_preserves_input",
        "test_real_gpu_norm_has_no_stale_cache_and_retains_double_precision",
        "test_real_gpu_canonical_chain_matches_cpu_and_keeps_retained_input",
    ),
    "test_radial_device_residency.py": (
        "test_full_complex_phase_cs_stop_stay_on_device_until_handoff",
        "test_existing_hankel_host_contract_and_explicit_native_return",
        "test_device_expansion_refinement_and_angular_order_preserve_field",
        "test_device_sampling_and_domain_guards_remain_active",
        "test_production_round_prefix_uploads_once_returns_host_and_keeps_losses",
        "test_device_failures_propagate_without_cpu_retry",
    ),
    "test_actual_probe_diagnostics.py": (
        "test_actual_gpu_continuation_matches_cpu_without_replacing_source",
    ),
    "test_compute_backend.py": (
        "test_auto_cuda_ray_trace_matches_cpu_with_energy_spread",
    ),
    "test_canonical_action.py": (
        "test_cuda_lct_and_gauge_keep_complete_complex_field_on_device",
    ),
    "test_column_acceptance_window.py": (
        "test_aperture_refinement_uses_global_gpu_policy",
    ),
    "test_electrostatic_column_transport.py": (
        "test_cuda_electric_batches_preserve_every_ray_and_report_real_completed_counts",
    ),
    "test_galerkin_potential.py": (
        "test_gpu_galerkin_matches_dense_complex_action_without_iteration_transfers",
        "test_gpu_galerkin_resource_and_physics_failures_do_not_fallback",
        "test_gpu_execution_retry_uses_original_host_wave_only_for_resource_failure",
    ),
    "test_galerkin_specimen.py": (
        "test_gpu_material_phase_interpolation_and_bandlimit_preserve_carrier_and_weights",
        "test_gpu_material_restarts_full_quadrature_and_phase_after_resource_failure",
        "test_gpu_repeated_material_operators_release_private_split_blocks",
        "test_gpu_retired_material_pool_preserves_live_device_output",
    ),
    "test_gpu_capture_contract.py": (
        "test_actual_gpu_capture_matches_cpu_cube_and_detector_reintegration",
    ),
    "test_multislice.py": (
        "test_cupy_multislice_matches_complex128_cpu_reference",
    ),
    "test_stem_cuda_pipeline.py": (
        "test_explicit_cuda_keeps_stem_arrays_resident_until_one_bulk_transfer",
        "test_resident_cuda_frozen_phonon_detector_signals_match_cpu_reference",
        "test_resident_cuda_projected_phase_object_matches_cpu_reference",
        "test_dynamic_detector_centres_match_cpu_even_after_partial_cuda_resource_failure",
    ),
    "test_wave_device.py": (
        "test_real_gpu_column_matches_cpu_complex_field_and_host_checkpoint",
        "test_real_gpu_physical_annulus_and_column_aperture_masks",
        "test_real_gpu_fft_larger_than_cache_quota_executes_without_raising_memory_budget",
        "test_real_gpu_segmented_modes_reuse_one_pool_and_match_cpu",
    ),
    "test_wave_fft.py": (
        "test_cupy_tem_image_fft_matches_numpy_reference",
        "test_cupy_stem_detector_fft_matches_numpy_reference",
    ),
    "test_wave_aperture_edge.py": (
        "test_device_field_estimate_matches_cpu",
    ),
    "test_wave_grid.py": (
        "test_cuda_fourier_refinement_and_nonlinear_carriers_preserve_phase_and_rejections",
    ),
    "test_tilted_column_wave.py": (
        "test_real_gpu_tilted_objective_matches_cpu_complex_checkpoint",
        "test_real_column_higher_order_magnetic_residual_runs_on_cpu_and_gpu",
    ),
    "test_tilted_spherical_column.py": (
        "test_real_gpu_tilted_cs_matches_cpu_and_mechanical_ray_response",
        "test_captured_default_electric_tail_tilted_cs_cpu_gpu",
    ),
    "test_posed_aberration_wave.py": (
        "test_real_gpu_tilted_screen_matches_cpu_absolute_complex_field",
    ),
    "test_wave_magnetic_residual.py": (
        "test_actual_gpu_matches_cpu_full_complex_envelope",
    ),
    "test_posed_multipole_wave.py": (
        "test_real_gpu_native_multipole_residual_matches_cpu",
    ),
    "test_tilted_multipole_column.py": (
        "test_actual_tilted_stigmator_gpu_and_cpu_wave_match_particle_centroid",
    ),
    "test_posed_wave_aperture.py": (
        "test_gpu_uses_same_swept_geometry_and_contacts",
    ),
    "test_posed_wave_hardware.py": (
        "test_real_cuda_bore_mask_matches_ray_contacts_and_cpu",
    ),
    "test_tilted_aperture_column.py": (
        "test_closed_tilted_plate_only_absorbs_local_slab_on_actual_gpu",
        "test_tilted_c2_finite_shoulder_diffraction_cpu_gpu_complex_parity",
        "test_tilted_c2_broad_beam_total_transmission_stabilizes_on_gpu",
    ),
}
GPU_TEST_IDS = frozenset(
    f"tests/{filename}::{name}"
    for filename, names in GPU_TEST_FUNCTIONS.items()
    for name in names
)
# This test also has mandatory CPU parameters; never exclude its whole function.
GPU_PARAMETER_IDS = frozenset({
    "tests/test_electrostatic_column_transport.py::"
    "test_closed_electric_and_magnetic_actions_match_reference_backend[CUDA GPU]",
})


def is_gpu_hardware_test(node: str) -> bool:
    return node.split("[", 1)[0] in GPU_TEST_IDS or node in GPU_PARAMETER_IDS


def is_complete_skip(outcomes: dict) -> bool:
    """Neither missing phases nor failed teardown are an optional-device skip."""
    return outcomes in (
        {"setup": "skipped", "teardown": "passed"},
        {"setup": "passed", "call": "skipped", "teardown": "passed"},
    )


def is_gpu_unavailable_reason(reason: str) -> bool:
    """Match reviewed device/dependency admission messages, not arbitrary skips."""
    reason = reason.removeprefix("Skipped: ")
    # The rendering fixture also falls back after shader or runtime failures.
    # Only device/context admission failures may be NOT_RUN on hosted CI;
    # never admit the generic "Real GPU rendering unavailable" prefix.
    graphics_prefix = "Real GPU rendering unavailable: CPU rendering: "
    if reason.startswith(graphics_prefix):
        detail = reason.removeprefix(graphics_prefix)
        if detail in {
            "No OpenGL context is available",
            "CAD GPU rendering requires OpenGL 4.1",
            "No offscreen OpenGL surface is available",
        }:
            return True
        software_prefix = "No hardware OpenGL renderer is available: "
        if detail.startswith(software_prefix):
            renderer = detail.removeprefix(software_prefix).lower()
            return renderer == "unknown" or any(token in renderer for token in (
                "llvmpipe", "softpipe", "swiftshader", "software", "gdi generic",
                "microsoft basic render", "mesa offscreen",
            ))
        return False
    return reason in {
        "CUDA device unavailable", "NOT_RUN: actual CUDA device unavailable",
        "No CUDA hardware", "CuPy CUDA backend unavailable", "No CUDA device",
        "CUDA is unavailable", "A CUDA device is required for complex128 parity",
        "optional real CUDA verification requires CuPy", "CuPy found no CUDA device",
        "Real CuPy CUDA device is unavailable", "No accessible CUDA runtime",
    } or reason.startswith((
        "could not import 'cupy':", "CUDA runtime unavailable:", "CuPy CUDA unavailable:",
    ))
