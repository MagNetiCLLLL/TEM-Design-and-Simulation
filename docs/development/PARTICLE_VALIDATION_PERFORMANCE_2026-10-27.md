# Complete-source particle validation performance

The subsequent [cached-source column optimization report](UNIFIED_COLUMN_PERFORMANCE_2026-10-27.md)
documents the next optimization of the 1,143-second incident-column stage.
The measurements below remain the preserved complete-source baseline.

## Scope and status

This work responds to the unacceptable runtime of the 12,289-particle
Finite-Field Normal validation. It preserves the optical controls, component
positions and dimensions, source distribution, physical currents, integration
step and error tolerances. Only added empirical aberrations are disabled;
effects produced by the retained electromagnetic fields remain active.

The timed complete-source validation spans actual tip emission through the
450-mm gun exit to **specimen entrance at 1599.1999975 mm**. "Column" in the
timings means this entire incident column. The separate downstream-to-screen
crossover validation in the optical report is not included in these timings.

The original complete calculation finished normally in **3,539.376 s
(58.990 min)**. Its source and every observed particle state remain archived.
The optimized complete-source run finished successfully in **1,487.115 s
(24.785 min)**: **2.380 times faster**, a **57.98% reduction**. The eleven
saved observation/source arrays match the original **byte for byte**.

The calculation is classical particle transport. No GUI, coherent propagation,
optical fitting, source pruning or current renormalization is involved.

## Completed full-scale comparison

| Measurement | Original | Optimized |
| --- | ---: | ---: |
| Complete emitted particles | 12,289 | 12,289 |
| Maximum column step (mm) | 0.05 | 0.05 |
| Complete integration nodes | 23,923 | 23,923 |
| Checkpoints / published observations | 891 / 777 | 891 / 777 |
| Incident-column batches | 15 | 1 |
| Numerical thread budget | 1 | 4 |
| Source-through-specimen validation (s) | 3,539.376 | 1,487.115 |

The new stage counters measured **337.351 s** for the complete fresh gun,
**1.711 s** for column-plan preparation and **1,143.226 s** for incident-column
propagation. The remaining time covers validation setup, source-exit archival
and statistics. Both total counters stop before the final compressed observation
archive and before the comparative readback. No saved gun state was substituted
for the fresh source in this timed run.

This before/after result includes removal of the earlier single-thread
diagnostic cap. It is not an isolated code-only speedup or a benchmark of the
GUI's default settings. It is one complete run per configuration, not a timing
distribution. The incident column still accounts for about 77% of the new
total; this validation remains a roughly 25-minute computation.

All six `777 x 12289` position/slope/energy/time arrays, the survival masks,
observation Z, original IDs, weights and source-energy offsets are elementwise
equal and also equal as contiguous bytes, with matching shapes and dtypes.
Every maximum absolute difference is zero. The full source fingerprint,
physical-input signature, field identity, geometry and complete integration
and checkpoint meshes match the preserved baseline. All 12,289 particles
survive to specimen entrance at the unchanged 10-microampere current.
The source emission mean and spread setting remain independently 0.3 eV.

Whole-process peak working set was **2,523,742,208 bytes (about 2.35 GiB)**,
including source, execution, output and comparison activity. This is distinct
from the 2-GiB retained-history admission limit and is not a claimed measured
reduction from the original process, whose peak was not captured.

The actual optimized solver identity is
`23de8dc3fc9352dde109b75444d4890315eb4b674350a868d23a126a6192a24c`;
the source fingerprint is
`3ada33dbccd7a2820b74a9aaa6e8a277a4721a17afa7559539cb47891a435a02`.
The complete receipt is `complete-performance-comparison.json` in the local
evidence directory listed below. The process exited successfully.

## Identified costs and changes

### Avoid repeated complete-column traversals for exact observations

The original observer requested a complete trajectory at every integration
node, then kept only a smaller set of observations. For this source its six
full-history arrays alone require about **14.111 GB**. The memory guard reduced
the batch size to 870 particles, requiring **15 complete axial traversals**.
Each traversal repeatedly prepares and samples the same fields and geometry.

`execute_unified_plan` now has an explicit `history_policy="checkpoints"`.
It still executes every original integration node, all RK4 stages, per-node
potential queries and mechanical contacts. It retains exact observations and
complete restart state, including energy, clock, induction work, active flags,
physical stops, swept bounds and spatial dependency identities.

The complete validation uses **23,923 integration nodes, 891 checkpoints and
777 published planes**. Its six retained checkpoint history arrays require
about **0.526 GB**, before internal-state copies and working buffers. The memory
gate includes immutable checkpoint constructor temporaries and uses the same
estimator as batching. The corrected executor estimate for this plan is
**1,940,828,040 bytes**; the final observation-publication estimate is
**1,394,113,602 bytes**. Prior batches' observations are reserved while a later
batch runs. Both stages fit the 2-GiB history/observation budget, admitting this
complete source in one batch. This is not a total process-RSS cap: caller-owned
gun histories, field providers, Python and external caches are separate.
Inputs too large for one pass still use bounded batches on the same complete-
population mesh. No samples are discarded or downsampled.

Inserted receiver planes retain the adjacent integration row used by the
existing clipping interpolation, preserving its operation order at an edge.
The ordinary GUI default remains full history, so arbitrary-plane inspection
is not silently replaced with sparse interpolation. The audit and differential
field-response helper explicitly request checkpoint storage.

### Exact field shortcuts

- Gun magnetic potentials with exactly zero drive coefficients return zero
  field after the original finite-coordinate and bore-domain checks.
- An electric-potential batch wholly inside its already declared grounded
  continuation returns its existing constant potential and zero electric field.
  Tail and mixed-domain batches keep their original interpolation.
- Packed gun magnetic queries skip only sources whose captured finite support
  is completely outside the batch Z range. Inclusive support faces, original
  summation order, non-finite query handling and native posed-field fallback
  remain intact. No weak-field threshold or additional tail truncation is used.

The existing native potential sampler remains selected. A separate plane
sampler experiment was not enabled unconditionally: it improved a larger
short-segment batch but slowed a small batch and changed last-bit arithmetic.

### Correct the diagnostic thread limit

The earlier headless command set `OMP_NUM_THREADS=1`. In this project that
limits the complete numerical job, not just BLAS. It is not evidence of the
GUI's default parallel throughput. The validation command now exposes
`--cpu-threads`; stricter external limits remain respected. The complete
comparison uses four numerical threads and serial BLAS.

The discrete-gradient gun integrator retains its global convergence/error
barrier across row chunks. Its tolerance, accepted-step logic and physical
plane roots are unchanged. The existing rejected-step half-step reuse was
already present and was not implemented a second time.

## Bounded measurements before the full comparison

An 8,193-particle real-source run through the actual 24-mm aperture and beyond
25 mm gave the following results. It is a gun prefix, not a full-gun or
whole-column benchmark.

| Run | Total seconds | Batch DG seconds |
| --- | ---: | ---: |
| Original implementation, one thread | 456.269 | 431.844 |
| Original implementation, four threads | 189.759 | 165.277 |
| Exact magnetic-support shortcut, four threads | 195.757 | 165.663 |

The thread comparison measured a **2.40-fold** prefix speedup. All 20 saved
arrays and the 2,212 accepted-step state digests matched. The source emission
mean and energy-spread setting were independently 0.3 eV in all runs.
The support shortcut did **not** demonstrate additional gun-prefix speedup.
Its field-query microbenchmark is not a whole-integrator speedup claim.
Individual prefix timings include initialization/native compilation and some
overlapping task activity; they are not statistically replicated measurements.

In the single-thread profile, batch discrete-gradient updates accounted for
94.65% of wall time; physical-plane root finding accounted for only 0.93%.
Those nested timings must not be added together. This identifies the implicit
gun updates as a remaining cost without relaxing their accuracy requirements.

A process-local prototype experiment reduced a separate 3,073-particle short
column segment by about 13.6%, with equal saved states. Its more general
nonzero gun-potential optimization was not retained in production; the archived
zero-drive case exercises the applicable shortcut. This is a bounded prototype
result, not an independently measured production or full-scale speedup.

## Verification

- 79 tests passed across unified beam audit, particle optimization, prefix
  continuation and field-particle execution; one existing finite-coil response
  test also passed.
- 45 electric/gun-potential tests passed.
- 27 gun instrument-field and parallel-step tests passed.
- 14 exact gun, count-convergence, downstream and diagnostic-ray tests passed;
  CI test inventory also passed separately.

After the prepared-gun identity and memory-peak refinements below, 52 audit
and optimization tests plus one measured immutable-checkpoint allocation test
passed. This is an overlapping follow-up scope, not 53 additional distinct
tests to add to the earlier counts.

These checks cover static and dynamic fields, physical contact and receiver
edges, actual energy/time, stopped source columns, complete and sliced source
identity, cancellation, partial failures, retained restart state and unchanged
full-history defaults. A test interrupted by concurrent source edits correctly
failed the execution-identity guard; its uninterrupted rerun passed.

The complete optimized run freezes the implementation and rejects changes in
the old field identity, geometry, full integration mesh or checkpoint mesh.
It regenerates the entire gun source. Final verification compares all original
IDs, weights, energy offsets, observation planes, six state arrays and survival
masks. Numerical solver identity is expected to change after code edits;
physical identity and measured states must be checked independently.

The first optimized attempt completed its fresh gun in 323.668 seconds, then
the strict plan guard rejected a changed scene identity before any column
integration. Gun preparation leaves derived vacuum attributes on its temporary
state. All 40 actual potential contribution digests, geometry and integration
nodes were unchanged, but reusing this prepared state differed from the old
observer's clean snapshot restoration. The direct observer now restores the
original column state once and keeps the actually executed gun particles.
A real preparation/restoration preflight matches all seven frozen identities.
The stopped attempt is archived separately and is not counted as a completed
performance result. See `prepared-gun-restored-column-preflight.json` and
`sparse-memory-budget-review.md`.

## Evidence and limits

Local raw data are under `outputs/extended-optical-validation-20261027/`:
`count-n12289-h005.*`, `frozen-12289-plan.json`,
`gun-performance-comparison.json`, `GUN_PERFORMANCE_ASSESSMENT.md`,
`performance-short-segment/`, and `sparse-history-implementation.md`.
The completed full run uses `count-n12289-h005-optimized.*`; its comparative
receipt is `complete-performance-comparison.json`. `optimized-gun-exit.npz`
also preserves the independently verified complete source boundary.

Performance acceptance must not be confused with optical qualification.
The original 3,073-to-12,289 source-count refinement still fails the declared
angle, radius and local-focus convergence gates. The current setting also
does not meet the requested 30-mrad physical pupil edge. Faster execution
does not change those results. See the
[particle validation report](FINITE_FIELD_PARTICLE_VALIDATION_2026-10-27.md).
