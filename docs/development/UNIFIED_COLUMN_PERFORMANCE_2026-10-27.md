# Cached-source unified particle column optimization

## Scope

This follow-up optimizes the 1,143.225705-second column stage measured in
[the preceding performance report](PARTICLE_VALIDATION_PERFORMANCE_2026-10-27.md).
The benchmark reuses the actual 12,289-particle gun exit and propagates from
450 mm to specimen entrance at 1599.1999975 mm. It does not regenerate emission,
change lens controls, move components, reduce particle count, increase the
0.05-mm maximum step, or replace actual particle states with a fitted probe.
The source mean emission energy and its separate energy-spread setting both
remain 0.3 eV. Only classical particles are calculated.

Performance equivalence and legacy-model compatibility are separate checks.
The user's approximately tenfold waist-size allowance applies to the comparison
between physical models. It is not an error budget for this optimization.

## Implementation

- Compile static axial derivatives and aligned multipole polynomial coefficients
  once per exact Z, rather than independently for every particle and repeated
  RK query. Each immutable sampler retains at most 16 exact-Z coefficient sets.
  Adjacent floating-point values used at one-sided support faces remain distinct.
- Evaluate the resulting field at every actual transverse particle position.
  Never cache a particle field value across RK trial states. Every query retains
  its source-domain and bore checks, including checks after a cache hit.
- Keep original evaluation for tilted or off-axis sources, nonplanar queries,
  selected-source requests, near-boundary ambiguity and extreme arithmetic.
  Time-dependent coils and electric/FEM providers retain their own evaluations;
  no field tail or small coefficient is discarded.
- Avoid redundant swept-bound scans when the complete set of trial or endpoint
  positions has already been recorded. Include rejected mechanical-contact
  trials and preserve complete checkpoint bounds and restart dependencies.
- Avoid allocating and multiplying an absent induction-work array in static
  fields. Preserve the original operation order whenever that array is present.

The particle executor selects this sampler; other callers retain their explicit
backend choice. The full-history default, sparse observation policy, RK4 mesh,
material contacts, energy, flight time, particle IDs, weights and real stops
keep their existing contracts.

## Measurement and integrity gates

`scripts/benchmark_unified_column.py` checks the real source fingerprint,
complete input graph, content-pinned external inputs, field and geometry
identities, all 23,923 integration nodes, 891 checkpoints and 777 published
observations. Initial energy is reconstructed with the original expression,
`nominal_exit_energy_ev + original energy_offset_ev`, rather than reusing a
checkpoint energy that may contain one ULP from potential subtraction.

The timer surrounds only column execution, like the preceding 1,143.225705-s
stage. Source loading, plan preparation, comparison and archive compression are
reported separately or excluded. Both runs use four numerical threads and
serial BLAS. These are individual measurements, not a runtime distribution.

The frozen pre-execution gates are `abs(actual-reference) <= atol +
rtol*abs(reference)`:

| State | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: |
| X/Y position | 5e-15 m | 5e-10 |
| X/Y slope | 5e-13 rad | 5e-10 |
| Flight time | 5e-21 s | 5e-12 |
| Kinetic energy | 2e-8 eV | 0 |

Z, survival masks, original IDs, weights and energy offsets must match exactly.
The script distinguishes numerical agreement from actual byte equality,
preserves raw outputs even on a failed comparison, refuses to overwrite prior
evidence, and checks that source files do not change during execution.

## Results

The frozen full-column replay completed successfully in **332.302849 s
(5 min 32 s)**, compared with **1,143.225705 s (19 min 03 s)** for the native
sampler baseline: **3.4403 times faster**, a **70.93% runtime reduction**.
Both measurements contain the same incident column, all 12,289 original
particles and every original integration node. Loading a saved source does
not artificially inflate this ratio: neither column timer includes gun
propagation. Reusing that source separately avoids the preceding measured
337.351-s fresh gun calculation.

Archive compression took another 13.260 s and is excluded from propagation
timing. The optimized timer includes a lightweight ten-second progress and
code-identity monitor. No other simulation or regression process ran during
the formal measurement. This is a backend measurement, not a GUI startup or
image-generation benchmark.

Every fixed numerical gate passed across all 777 x 12,289 states:

| Array | Maximum absolute difference | Byte equality |
| --- | ---: | --- |
| X | 1.437e-18 m | No |
| Y | 1.938e-18 m | No |
| X slope | 1.700e-16 rad | No |
| Y slope | 1.735e-16 rad | No |
| Flight time | 0 s | Yes |
| Kinetic energy | 0 eV | Yes |
| Z, survival, weights, IDs, source energy offsets | 0 | Yes |

Static coefficient aggregation changes floating-point summation order;
therefore the position and slope arrays are numerically equivalent within
the declared gates, not byte-identical. All 12,289 particles survive at the
unchanged total current of 10 microamperes. No source weight is renormalized.
Specimen RMS radius is **0.285042252 nm**, D95 **0.540665522 nm**, and alpha95
**9.842057978 mrad**. These remain diagnostic statistics, not proof of the
requested 30-mrad physical pupil edge or source-count convergence.

The optimized implementation identity is
`8824c04dbb44ef360f5e3ea03b9f326d4f34b95df9dc89111968d7def21238ef`.
The unchanged real source identity is
`3ada33dbccd7a2820b74a9aaa6e8a277a4721a17afa7559539cb47891a435a02`.
Field and geometry digests remain
`29b1ccece2e501a831ae1839b2ff4f51b2b977d809f992806a8b92e009593f86` and
`8d1c16ba257f61320f6535911c9bef08b22e61210a76058e5aff8cfb45557b74`.

## Regression evidence

**191 distinct tests passed** in the final source configuration:

- 143 tests covering native and plane samplers, particle execution, sparse
  observations, complete checkpoint bounds and safe prefix continuation.
- 47 additional tests covering remote Gaussian domains, small CPU coefficient
  operators, dynamic electric/gauge fields, benchmark guards and crossover
  acceptance.
- The final 15-test crossover file rerun includes one additional diagnostic
  isolation test; its other 14 tests overlap the preceding 47.

The small coefficient-operator tests are not new coherent propagations. No
coherent pipeline or GUI was run. An earlier test run during concurrent source
edits correctly rejected the changed implementation; the final frozen rerun
passed. Tests preserve dynamic timing, real clipping, rejected RK trials,
one-sided support faces, stale-cache rejection and strict finite-value domains.

## Legacy particle compatibility

The legacy comparison uses the same cached source, exact observation planes,
current optical controls and geometry. It is separate from the strict native
unified comparison above. Both legacy and unified models retain finite optical
extent; the legacy model is not the thin-lens Ideal Optics tier.

The dense archived 193-particle comparison has 5,348 exact observation planes,
with at most 0.5-mm spacing and local refinement. Both models identify five
main waists in the same order: C1-C2, C2-C3, corrector internal, AC pair and
Mini Condenser-to-specimen. Their observed positions are 503.2, 721, 1202, 1400
and 1585.9 mm. Every corresponding RMS radius satisfies the factor-ten limit.

The new 12,289-particle legacy replay uses exactly the same 777 planes as the
completed optimized unified run. The primary physical diagnostic explicitly
disables added legacy Cs and chromatic/other empirical aberrations in a private
copy; native field strengths and all geometry remain fixed. A second replay
retains the archived legacy objective Cs of 1.2 mm and is labeled separately.
Neither replay changes the unified inputs. All original IDs, weights, gun-exit
states and survival at every observed plane match.

The no-added-aberration reference gives the following local two-direction
minima; every corresponding RMS radius meets the factor-ten limit:

| Interval | Legacy Z (mm) | Unified Z (mm) | Legacy RMS (nm) | Unified RMS (nm) | RMS ratio | D95 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1-C2 | 503.25 | 503.25 | 0.196368073 | 0.196369506 | 1.0000073 | 1.0000106 |
| C2-C3 | 720.75 | 720.75 | 2.10181781 | 2.09491614 | 0.9967163 | 0.9950400 |
| Corrector internal | 1202 | 1202 | 2155.56048 | 2155.20749 | 0.9998362 | 0.9994997 |
| AC pair | 1400.05 | 1400.00 | 40.6521454 | 24.8489012 | 0.6112568 | 0.8076366 |
| Mini Condenser-specimen | 1585.9 | 1585.9 | 105.011974 | 99.6026655 | 0.9484886 | 0.9257362 |

The specimen endpoint RMS changes from 7.37056660 to 0.285042252 nm across
these different models, a ratio of 0.0386730; D95 changes from 32.4929948 to
0.540665522 nm, a ratio of 0.0166394. With archived legacy Cs retained, the
endpoint RMS is 7.30024720 nm instead. Endpoint size is not itself proof of a
two-sided focus.

**The full high-count topology qualification remains UNRESOLVED**, although
the five confirmed principal waists retain their component intervals and
relative order. The largest observation gap is 2 mm, exceeding the unchanged
0.5-mm coverage requirement. A further unified radial candidate occurs about
2.75 nm before specimen entrance. Separately archived 0.25-nm local sampling
places its minimum at 1599.199994750 mm with RMS 0.284651483 nm, but the two
transverse covariance eigenvalues have opposite signs (approximately
-3.077e-15 and +2.931e-15 m rad). It is not a confirmed common two-direction
focus. There is no matching legacy fine-grid capture, so this candidate is
retained explicitly instead of silently counted as an extra or missing
confirmed crossover. The comparison code also distinguishes unresolved
sampling/count differences from demonstrated topology failures.

The legacy CPU column takes 10.622 s with archived added terms and 10.922 s
without them; serialization takes another approximately 13.7 s. These are
contextual measurements, not the denominator of the 3.4403-fold improvement.
The unified executor remains substantially slower than the legacy compiled
parameterized integrator. It retains shared three-dimensional potentials,
natural high-order field effects, strict provider-domain checks and actual
mechanical/restart state. These model and execution differences prevent
treating the legacy timing as an equivalent-physics speed claim.

Historical approximately 30-mrad control settings are a separate reference,
not substitutes for the unchanged current settings. On the archived dense
193-particle captures, these historical controls give five legacy waists versus
six unified waists, with a specimen RMS radius ratio of about 1,325.5. That
comparison fails the topology/size requirement. This pre-existing difference
is not repaired or concealed by the performance optimization. No lens strengths
or component positions were adjusted.

The current source-count convergence and physical pupil-edge qualification
also remain unresolved as recorded in the preceding optical reports. A
matching high-count legacy downstream-to-screen capture was not produced;
the present timing and model comparison stop at specimen entrance.

## Local evidence

Raw caches and trajectories remain local under
`outputs/extended-optical-validation-20261027/`,
`outputs/plane-coefficient-optimization-20261027/`, and
`outputs/column-performance-20261027/`. They are not Git deliverables.
Only project implementation, reusable diagnostics, tests and this maintained
summary are committed.

The formal receipt and arrays are `column-optimized-n12289-h005.json/.npz`;
`column-preflight.json` records the independent non-executing guard check.
`legacy-high-count-comparison.json/.md` compare both legacy cases directly
against the new optimized arrays. `legacy-topology-comparison.json/.md`
describe the separate dense 193-particle and historical-control evidence.
JUnit receipts are `regressions-core.xml`, `regressions-additional.xml` and
`regressions-final-comparison.xml`. The short-segment sampler timings are
`short-results.json` in the plane-coefficient evidence directory.
