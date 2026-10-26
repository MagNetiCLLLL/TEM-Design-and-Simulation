# Historical wave-entry tests and HANDOFF v2

Three baseline tests used synthetic incident ray bundles/statistics to call
the formerly available specimen-plane pupil implementation:

- `test_a5_state_profile_cache_and_production_stem_phase`
- `test_tem_reference_cif_frozen_phonons_average_intensities_on_finite_roi_grid`
- `test_stem_atomistic_frozen_phonons_average_detector_intensities`

Their original versions are retained in Git baseline
`5eac9855ff8eefa2a3d68ddba3029f7c23e03b0c`. The initial broad-audit source-gate
failures remain in the HANDOFF ledger and are not rewritten as historical
passes.

The current tests first assert that production rejects the synthetic source.
They then override admission **only inside their individual pytest fixtures**
to retain the existing algorithm checks: A5 changes phase rather than amplitude,
profiles/cache identities preserve the coefficients, and frozen-phonon outputs
average intensities rather than coherent amplitudes. Names explicitly say
`isolated` or `historical`; physical parameters, expected values and tolerances
remain unchanged. No production gate, GUI, CLI or profile admits this override.

These tests do not demonstrate the new gun-to-specimen, coil-driven STEM or
post-specimen detector adapters, which remain separate unfinished requirements.

## Retired files: 15 October 2026

The following files were removed from the working tree. Their exact pre-removal
versions remain available at commit
`a0478f12ab21c84ea4f27bef72454d791e897c06`:

| Removed file | Reason and retained coverage |
| --- | --- |
| [scripts/validate_illumination.py](https://github.com/MagNetiCLLLL/TEM-Design-and-Simulation/blob/a0478f12ab21c84ea4f27bef72454d791e897c06/scripts/validate_illumination.py) | Its PASS condition required two `wp03_convergence_` properties produced only by the withdrawn historical suite. Neither its default nor extended selection included that producer, so even passing selected tests could not satisfy the report gate. It is no longer a validation entry point. |
| [tests/historical/illumination_wp03_contract.py](https://github.com/MagNetiCLLLL/TEM-Design-and-Simulation/blob/a0478f12ab21c84ea4f27bef72454d791e897c06/tests/historical/illumination_wp03_contract.py) | Already outside default pytest discovery; it expected independent specimen-entrance pupils to enter production. Current mathematics and explicit rejection tests remain in the files listed below. |
| [tests/historical/cache_projection_assumptions.py](https://github.com/MagNetiCLLLL/TEM-Design-and-Simulation/blob/a0478f12ab21c84ea4f27bef72454d791e897c06/tests/historical/cache_projection_assumptions.py) | Already outside default pytest discovery; it assumed a legacy source could be reprojected and an arbitrary projector field-map change could preserve upstream products without proving field support. Current cache tests enforce source admission and scoped invalidation. |

Retained regression entry points:

- Mathematics: [test_illumination_modes.py](../test_illumination_modes.py)
  checks pupil amplitude/intensity semantics, current quantiles, incoherent mode
  addition and analytic Gaussian LCT propagation;
  [test_effective_gun_source.py](../test_effective_gun_source.py) retains Gaussian
  covariance, mode weights and energy-quadrature checks without activating an
  independent downstream source.
- Source admission: [test_source_admission.py](../test_source_admission.py) and
  [test_tip_source_contract.py](../test_tip_source_contract.py) reject independent
  pupil/exit sources, including attempts to bypass the gun through cached data.
- Cache behavior: [test_calculation_cache_reuse.py](../test_calculation_cache_reuse.py)
  retains `test_projector_field_map_descriptor_with_unknown_support_invalidates_upstream_products`
  and `test_new_tem_calculation_rejects_legacy_source_before_cache_reprojection`,
  alongside reuse checks for changes with known local scope.

Removing the historical files does not replace the withdrawn convergence
evidence with a new PASS or qualify a complete physical source-to-image chain.
