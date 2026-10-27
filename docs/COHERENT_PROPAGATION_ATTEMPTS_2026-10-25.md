# Coherent propagation attempts and deferred method review

Date: 2026-10-25. Status: **Further optimization deferred at the user's request.**

The current whole-column method has not met the requested low-memory,
through-column exploratory workflow. The user asked to retain the attempts and
revisit the computational method later. Do not describe the work below as a
successful gun-to-specimen or gun-to-screen calculation. No further propagation
or numerical changes should be inferred from this handoff.

## Intended calculation and preserved constraints

The working point is the archived, screen-prepared STEM Nanoprobe Normal
configuration, with the actual coherent Tip boundary, accelerating gun and
captured electromagnetic fields. The requested sequence is gun to specimen to
main fluorescent screen, with intermediate complex-wave checkpoints. Restarting
from a synthetic specimen probe is outside this scope.

Lens settings, field definitions, geometry, apertures, source parameters and
aberration ownership were retained. Numerical projections must preserve their
actual surviving absolute weight and record their losses separately from physical
absorption. No intensity renormalization, centroid fit, removal of defocus/tilt,
or phase fit may be used to disguise an approximation.

The latest priority is one exact nominal emission energy. The nine-energy
mixture was deferred. The specimen target is **1599.2 mm** and the main screen
plane is **2714.15 mm**. The archived specimen is vacuum; reaching it would not
validate material scattering. The screen is retracted, so its requested result
would be the incident wave, not an inserted-screen readout. The inserted HAADF
stop at **2594.15 mm** must still execute on a successful downstream run.

## Emission energy is separate from FWHM

`electron_gun.emitter.emission_energy_ev` is the nominal initial kinetic energy.
`electron_gun.emitter.energy_spread_fwhm_ev` describes the energy-distribution
width. In the current cold-field-gun configuration both happen to equal
**0.3 eV**, but they are independent inputs. The width was never substituted for
the nominal energy or set to zero to obtain one sample.

The explicit `source_nominal_energy_preview=True` option uses the configured
nominal energy and labels the result **configured FWHM not integrated**. It
requires one energy sample and is not accuracy qualified. Tests also use
nominal energy **2 eV** with unchanged **0.3 eV FWHM** to prevent accidental
coupling of the two fields. The original nine quadrature energies contain no
exact 0.3 eV member; selecting or relabelling one would not implement this source.

Actual nominal-energy gun propagation reached **450 mm**, with kinetic energy
**300.0003000000031 keV** and absolute weight **0.998074156513386** per reference
electron. This includes the actual accelerating boundary and gun propagation.
The static FEM field was reusable; old nine-energy wavefunctions were not.

## Methods and improvements already attempted

The executed column route uses the captured scalar/vector potentials, a
quadratic canonical carrier and high-order residual propagation through **H8**.
The transverse complex envelope uses bounded Cartesian Fourier grids. Physical
geometry, electric reference energy/clock, kinetic-phase accounting and dynamic
scan fields remain in the execution path.

The attempts included:

- Segmented disk checkpoints, complete-plane indices, exact cache replay and
  independent checksum/history-prefix validation.
- Controlled spatial and spectral projection, weighted resampling, numerical
  boundary tapering and resource-aware retries. An accepted projection has an
  explicit absolute probability cost and complex L2 cost.
- Opt-in segment-entrance grid reduction, followed by admission of the actual
  physical operator before committing the proposed wave and history.
- Opt-in quadratic phase-carrier reselection with a compensating envelope.
  Temporary preparation runs on CPU at up to twice the column pixel limit.
  Complete-field representation discrepancies are separately recorded as
  estimates, not certified propagated error bounds.
- Immutable-array check reuse, same-plane scalar-potential reuse and avoiding
  known-zero dynamic-coil arrays after their domain checks. These reduce local
  overhead but do not remove the underlying bandwidth demand.
- Explicit single-nominal-energy execution, independent of the configured FWHM.
  Real execution exposed and fixed missing preview-flag forwarding in both
  near-tip source reconstruction and the GUI source preflight. Full source
  identity checking remains active.
- Adjustable exploratory probability/L2 allowances. The default preview remains
  **0.01 / 0.5**. Explicit previews now permit at most **0.5 / 4**; strict-mode
  and original gun settings are unchanged. Increasing an already issued ledger
  is rejected. New allowances require a fresh column branch from the actual gun.

Detailed earlier experiments and local comparisons are retained in
[Weighted coherent segments](WEIGHTED_COHERENT_SEGMENTS.md). These local results
do not establish whole-column convergence.

## Actual propagation milestones

All lengths in this table are millimetres. P and L2 denote the explicitly
declared column allowances, not measured errors. A pixel cap is per axis.

| Attempt | Source and column settings | Last complete saved plane | Outcome |
| --- | --- | ---: | --- |
| actual23 | Original nine energies; P=0.01, L2=0.5 | 530.7142857142857 | Exact-cache continuation recovered a missing mode after an unexplained earlier process exit. All nine modes then completed this plane. |
| actual24 | Same nine modes and allowances; segment-entrance projection | 558.75 | Cooperatively stopped. Weight 0.9902804452862601; numerical P cost 0.008167043192946838; L2 cost 0.21056641994671413. |
| actual25 | New nominal source; 1024 pixels, 5 mm, P=0.1, L2=0.5 | No new gun exit | Source reconstruction omitted the explicit nominal flag and correctly hit the ordinary finite-spread rejection. Fixed before actual26. |
| actual26 | One nominal energy; 1024 pixels, 5 mm, P=0.1, L2=0.5 | 558.75 | At approximately 577.625 mm the operator required 1056 pixels, above the 1024 cap. This was a documented grid-budget failure, not exhaustion of the total P/L2 allowances. |
| actual27 | Exact actual26 gun cache; 2048 pixels, 2 mm, P=0.1, L2=0.5 | 530.7142857142857 | Cooperatively stopped for the next explicit preview branch. Weight 0.9975513695192405; P cost 0.0005227869940720269; combined L2 cost 0.04743987859535327. |
| actual28 | New actual nominal gun execution; 2048 pixels, 5 mm, P=0.5, L2=4 | 631.7499999999999 | Saved a complete plane, then encountered repeated sampling pressure in the C2 entrance region. The process later exited with code 1 and no final Python failure receipt. Its final exit cause is unknown. |

The last complete actual28 checkpoint has:

- Digest `1391d9d314d667dffe4ed891aa37af0380ec3bb3b850c636e3be3895b81ef51c`.
- Absolute weight **0.7656216638753734**.
- Numerical probability loss **0.232452492637828** per reference electron.
- Combined projection/representation L2 cost **1.4195172489599732**, including
  an explicitly uncertified representation estimate of approximately
  **5.563536419005605e-7**.
- **Zero physical column absorption** through this saved prefix. The large
  reduction is numerical; it must not be presented as real lens or aperture loss.

Progress logs mention a later step at approximately **633.432185 mm**, and a
residual substage at approximately **634.053931 mm**. Neither is a complete
published observation plane. The original result file still says `running`
because no final report was written. Preserve that original evidence and use
`actual28_process_exit_observation.json` for the external exit observation.
Do not diagnose OOM, device exhaustion or exhausted error allowance from this
exit code alone. No task propagation process remained at the final check.

**Neither the specimen nor the screen was reached, for either the original
nine-energy mixture or the nominal-energy branch.**

## What the current evidence says about the method

The sharp deterioration around 632–634 mm is not explained by the number of
energy samples: it also occurs for the single actual nominal-energy mode.
Simply allowing more numerical loss enabled additional distance but ultimately
discarded a substantial part of the wave. This is not an adequate precision
strategy for future phase-plate matching or instrument optimization.

The region is the entrance of **C2**, whose Gaussian centres are 637.5, 650 and
662.5 mm. At 631.75 mm the C2 on-axis field is about 0.03282 T, while the C1 tail
is negligible. There is no hard aperture or changing bore radius at 632 mm.
The 631.75 mm checkpoint is an artificial subdivision of C2's support interval,
not a physical field boundary. Resource retries begin before that checkpoint,
so merely relocating it is not an established solution.

A proposed quadratic-carrier focus singularity was checked and was not found
at this saved plane: carrier curvature is about **7.507739 m^-1**, and the next
5 mm canonical map has `A+BQ` singular values about **1.036206**, condition
number 1. A reference-ray caustic near 671.53 mm is a separate, unvalidated
prediction for the full nonlinear wave.

Observed failures involve residual-envelope Nyquist checks, repeated spectrum
projection, boundary tapering and step reduction. High-order C2 phase and the
available transverse bandwidth are plausible limitations, but their separate
contributions have not been fully isolated. At the saved plane the input
boundary statistic, about 4.01e-11, also exceeds the phase-reselection helper's
1e-12 prerequisite, so that optional optimization declines the input. This is
a representation prerequisite, not proof that the physical model is invalid.

## Candidate directions for future work, not completed features

1. Reassess the representation and numerical method before increasing loss
   allowances again. Use short C2 intervals and compare complete complex fields
   on common physical coordinates, without fitting phase, centroid or norm.
2. Investigate a snapshot-bound axisymmetric radial executor for genuinely
   axisymmetric intervals. RadialWave, FFTLog and complex Cartesian/radial
   handoff primitives already exist. The old radial-column shortcut uses legacy
   equivalent optics/empirical Cs and is **not** a drop-in unified H8 executor.
   A new route needs the same captured phi/A_theta, geometry, weight, phase,
   timing and numerical histories, followed by validated return to 2D before
   unsupported non-axisymmetric elements. No radial admission was run here.
3. Investigate controlled low-band execution of high-order scalar phase without
   allocating a full high-resolution propagated plane first. Distinguish
   `exp(i R* Phi R)` (finite-basis Hamiltonian evolution) from
   `R* exp(i Phi) R` (phase action followed by projection). The former conserving
   norm does not prove zero truncation error; the latter needs explicit
   probability/L2 accounting and independently checked quadrature.
4. Investigate carrier preparation and chart selection at genuine difficult
   interfaces, retaining the complete compensating phase and unchanged physical
   admission gates. A canonical-only leap through C2 would omit its nonzero
   high-order residual and is not an accepted workaround.

No replacement method has been selected or validated. Any future approximation
must be visible, independently configurable, and excluded from high-accuracy
qualification until its own validation is complete.

## Retained evidence and validation

Local research root (outside Git):

`F:/TEMForge_research/2026-10-25_segmented_phase/`

Key retained files and directories:

- `actual26_single_nominal_preview_*`: actual nominal gun, resource failure and
  complete saved prefix. Independent audit checked 6 manifests and 41 payloads.
- `actual27_single_nominal_preview_*`: same-gun cache reuse and cooperative stop.
  Independent audit checked 8 manifests and 51 payloads. Its new column entries
  share `actual26_single_nominal_preview_checkpoints` under distinct keys.
- `actual28_single_nominal_preview_*`, including its separate checkpoints,
  immutable runner, source identity, progress, residual and pressure diagnostics.
  Independent audit verified 8 manifests and 51 payloads (378,886,928 bytes),
  with intact histories and a flux-accounting residual of about 1.924e-13.
  The last complete wave is 1980 by 1980 cells.
- `actual28_process_exit_observation.json`: external termination observation,
  deliberately separate from the unchanged original report.
  `actual28_single_nominal_preview_termination_observation.json` additionally
  records the independent audit's termination observation and original-report hash.
- `actual28_631_carrier_chart_audit.json` and
  `actual28_c2_segment_geometry_audit.json`: geometry/carrier findings above.
- `actual28_readonly_coefficient_costs.json`: small CPU cost inspection;
  not a whole-column performance benchmark.
- `nominal_single_gun_field_cache_audit.json`: verified reuse of the static FEM
  field without invoking a new solve.
- `scalar_phase_low_resolution_advisory.md`: deferred scalar-projection method
  notes; no implementation or whole-column validation is claimed.
- `audit_nominal_preview_results.py --run-prefix <run> --rehash`: independent
  checksum and accepted-prefix audit, using only that run's referenced entries.

Frozen implementation digests:

| Snapshot | Digest |
| --- | --- |
| segmented_snapshot_25 | `e906b26ea243783215ab9654b6ed5565e360617da93045c03ed38766e6b5e134` |
| segmented_snapshot_26 | `9254476b5c309e22da5d334d93f892c6a147fff0319a9af4d09e2b2427e2e236` |
| segmented_snapshot_28 | `b1ab57a62c15ac5be1c114ec54cb42b2a59850a12a4cb62480d6f4e257d966d8` |

Completed targeted regression evidence includes:

- `nominal_source_final_regression.xml`: **103 passed**, covering nominal-source,
  request, GUI and source-preflight routes after the execution fixes.
- `nominal_accelerating_tip.xml`: **50 passed**, including real captured-field
  Robin propagation to 4 micrometres. This overlaps the source tests above.
- `extended_exploratory_budget_limits_cpu.xml`: **78 passed**, including explicit
  wider limits, unchanged strict/default settings and refusal to widen an
  existing allocation.
- `nominal_final_integration.xml`: **293 passed** at its earlier recorded source
  revision; it predates the actual25 entry-path fixes and is not their evidence.
- Earlier local CPU/CUDA physical-operator comparisons are described in
  `WEIGHTED_COHERENT_SEGMENTS.md`. They do not establish full-column accuracy.

Do not add these counts as unique tests or treat passing regressions as proof of
through-column completion. An initial wider-budget test invocation accidentally
included a small CUDA test and was stopped; the final 78-test report explicitly
excludes CUDA and is the completed CPU evidence for that change.

Project changes and local evidence are retained. No commit, upload, GUI operation
or shutdown was performed as part of this handoff.
