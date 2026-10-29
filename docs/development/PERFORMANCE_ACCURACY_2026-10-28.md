# Unified benchmark, experiment CLI and measured optimization

The completed before/after comparison passes exact same-input equivalence.
Warm median elapsed time falls by **23.1% for Nanoprobe transport**, **19.0% for
bounded continuation**, and **21.1% for scan calibration**. The implemented
optimization targets the measured electric-potential interpolation hotspot.
These results apply to the declared 49-particle classical workload; they do not
establish convergence or physical qualification.

[Machine-readable evidence](performance_accuracy_20261028.json) retains all
timing samples, input and implementation identities, array fingerprints,
observables, resource measurements, qualification status and CLI proof.
Usage is documented in [Unified benchmarks](../UNIFIED_BENCHMARKS.md) and
[Experiment CLI](../EXPERIMENT_CLI.md).

## Delivered behavior

- The benchmark suite runs Nanoprobe, real saved-prefix continuation and scan
  calibration in separate owned processes. It separates first-call, warm and
  dependency-checked reuse timings from preparation, observation and archive I/O.
  Deadlines, complete run inventories, failure receipts and isolated scan caches
  prevent incomplete work from being reported as a successful comparison.
- The classical experiment CLI supports `plan`, `run`, `resume` and `export`
  using the existing recipe and `.temexp` formats. Completed successful or failed
  points are saved atomically. Resume requires the same inputs, code, observer
  and tolerances, and does not recalculate retained points. Repeated complete
  resume preserves the sensitivity inventory rather than differentiating
  previously derived sensitivities again.
- The common observer separates execution, same-input repeatability, numerical
  convergence and physical qualification. It preserves source IDs, original
  weights, interception, phase, clock, energy and checkpoint evidence. Missing
  measurements remain unavailable. Scalar comparison uses zero absolute and
  relative tolerance; retained numeric arrays require exact fingerprints.
- The earlier finite-waist boundary and automatic-scan cancellation repairs
  remain in place and are covered by the integrated regression run.

## Measurement design

| Setting | Executed value |
| --- | --- |
| Host | Windows 11; AMD Ryzen 9 9950X3D, 16 cores / 32 logical processors |
| Runtime | Python 3.12.3; NumPy 2.4.6; SciPy 1.18.0; Numba 0.66.0; PySide6 6.8.3 |
| CPU allocation | 8 numerical threads; BLAS serial |
| Scan response allocation | 4 independent processes; 17 ordered differential traces |
| Source and column integration | 49 actual emitted particles; 0.05 mm column step |
| Retained column history | 0.5 mm; distinct from the integration step |
| Per-case sampling | 1 first call, 3 warm recomputations, 3 result/cache reuses |
| Nanoprobe endpoint | Nominal specimen upper surface at 1599.1999975 mm |
| Continuation | Saved prefix at 460 mm; terminal plane at 470 mm |
| Coherent calculations | Explicitly `NOT_RUN`; usable-CUDA restriction retained |

Each accepted suite contains 21 declared timed operations. Prefix generation,
save/load, setup and observations have separate receipts. Continuation actually
starts at the saved prefix, with start index 200 and 200 new intervals instead
of 400 complete local-plan intervals. No entrance restart is counted as resume.
Continuation is local pre-specimen transport, not a specimen probe measurement.

First-call means the first measured operation in a fresh worker after declared
preparation. It does not mean cold operating-system or Numba disk caches. Warm
transport may reuse legitimate gun caches. Optical controls are unchanged:
there is no fitting, refocusing, source replacement, current renormalization,
aperture enlargement, field-tail cutoff or integration-step increase.

Baseline and candidate timing suites ran separately, without concurrent project
tests or other project calculations. Three warm samples support descriptive
ratios, not a confidence interval or a guarantee for other machines or workloads.
The complete physical/numerical graphs and external content identities match.
The comparator also requires identical actual scan scheduling, including the
original measurement receipts retained by memory/disk cache hits.

## Measured bottleneck and implementation

A separate diagnostic Nanoprobe `cProfile` pass recorded 61.336 seconds of
profiled time. Unified potential sampling accumulated 31.530 seconds; electric
snapshot sampling accounted for 11.943 seconds, including 6.549 seconds in its
bilinear interpolator. This identified repeated small-array interpolation and
dispatch within field evaluation as a concrete optimization target.

`electric_potential_snapshot.py` now uses a compiled per-point loop for the
existing bilinear representation. It reuses the already calculated NumPy radius,
keeps float64 arithmetic and operation order, and disables `fastmath`. The original
NumPy implementation remains the fallback. Domain admission, cell selection,
continuation rules, finite-element/cut-cell evaluation, scalar potential and its
gradient are unchanged. No new approximation or optical parameter is introduced.

Analytic-potential, gradient, line-integral, nonuniform-cell, boundary,
signed-zero, shape, domain and fallback regressions pass. Additional exploratory
operator checks compared 82,680 synthetic points and 67,081 retained trajectory
points exactly. The production-suite comparison below is the acceptance evidence;
operator checks alone are not a full-column speedup measurement.

The diagnostic suite containing the profile was not an accepted timing baseline:
observer and continuation-fixture issues were repaired before collecting the
accepted baseline. Only `baseline-final` and `optimized-final` contribute to the
following ratios. Parent scan profiling cannot attribute time inside scan child
processes; parent waits are not treated as physical-kernel hotspots.

## Performance results

Times are seconds. Warm and reuse values are medians of three samples; first-call
values are single observations. Speedup is baseline divided by candidate.

| Case | Phase | Baseline | Optimized | Speedup |
| --- | --- | ---: | ---: | ---: |
| Nanoprobe | First call | 50.4061 | 43.0791 | 1.170x |
| Nanoprobe | Warm | 45.1606 | 34.7474 | 1.300x |
| Nanoprobe | Reuse | 3.0920 | 3.2092 | 0.963x |
| Continuation | First call | 2.7929 | 2.9318 | 0.953x |
| Continuation | Warm | 2.9252 | 2.3695 | 1.235x |
| Continuation | Reuse | 2.6333 | 2.0711 | 1.271x |
| Scan calibration | First call | 81.5963 | 64.4141 | 1.267x |
| Scan calibration | Warm | 79.3534 | 62.6114 | 1.267x |
| Scan calibration | Reuse | 2.1353 | 1.7301 | 1.234x |

Warm ranges were 44.3705–45.9058 versus 31.4523–35.7124 seconds for Nanoprobe,
2.7724–3.0817 versus 2.3545–2.4195 seconds for continuation, and
68.6983–88.1054 versus 58.9889–69.5163 seconds for scan calibration.

Not every phase improved: Nanoprobe reuse was 3.8% slower and continuation's
first call was 5.0% slower. No blanket speedup is claimed for cached results or
short operations dominated by setup and identity checks. Timing variation also
prevents attributing every phase's change solely to the interpolation kernel.

| Case | Baseline peak worker RSS (MiB) | Optimized peak worker RSS (MiB) |
| --- | ---: | ---: |
| Nanoprobe | 560.79 | 561.87 |
| Continuation | 271.12 | 269.40 |
| Scan calibration | 192.01 | 192.14 |

These are the largest sampled peaks across each case's timed operations,
sampled every 20 ms. They exclude descendants, including scan response workers,
and may miss brief peaks. They do not establish a memory reduction. VRAM is
unmeasured, not zero.

## Numerical equivalence and physical limits

Both suites completed with particle repeatability and suite acceptance `PASS`.
The independent comparator reports particle equivalence `PASS` and scan
equivalence `PASS`, including all four response matrices and the residual.
Source IDs, original weights and energy offsets, stops, masks, phase histories,
existing clock/energy ledgers and exact checkpoints remain identical.

| Nanoprobe upper-surface observable | Before and after | Absolute difference |
| --- | ---: | ---: |
| Current-weighted alpha95 | 9.96935661178633 mrad | 0 |
| D95 | 0.5555806898010609 nm | 0 |
| RMS radius | 0.1559004765887315 nm | 0 |
| Surviving absolute current | 0.00001 A | 0 |

Existing unknown kinetic-energy values in the original gun-history prefix remain
unknown and are explicitly marked `PARTIAL_RETAINED`. Their exact prefix
provenance is checked; they are not replaced with invented nominal energies.
Current-carrying column and checkpoint energy remains subject to finite-positive
validation, and original arrays remain fingerprinted.

Numerical convergence for this workload is `NOT_RUN`; physical qualification is
`UNRESOLVED`. The retained history does not supply the exact focus triple or all
condenser boundary/guard checkpoints needed for those qualifications. This round
does not certify the default 3,000-particle population, source-count convergence,
step convergence, specimen scattering, coherent propagation, OEM performance or
hardware behavior. A faster identical calculation does not resolve these limits.

The baseline uses observer v2 and the candidate v3. The only admitted migration
is the documented unchanged upper-surface and section-terminal contracts.
The comparator records this decision explicitly and still requires exact planes,
inputs, arrays and scalar values. The newly added sample-reference-plane contract
is not retroactively assigned to old observations.

## Actual CLI lifecycle

An actual single-point 49-particle experiment executed all six commands:
`plan`, `run`, `resume`, CSV export with provenance manifest, JSON export and
readable-report export. All returned zero without creating a `QApplication`.
Complete resume invoked the calculator zero times and retained the same point
digest, `6f4955d7f5d0f85d40b3acb50d82eeb40d4656b48c8bc90246b7ea1aeb9b7266`.

This ordinary calculation retains the exact optical reference plane at
1599.2 mm, rather than the upper surface used by the Nanoprobe benchmark. Its
report correctly identifies `sample_reference_plane`, with D95
0.5773127894445536 nm, RMS radius 0.1564683455672042 nm and current 0.00001 A.
These are incident statistics for the retracted/vacuum specimen state. Focus and
topology are `NOT_EVALUATED`; a single run's repeatability is `NOT_EVALUATED`.
The CLI run overlapped functional tests, so its duration is excluded from all
benchmark ratios.

## Validation and evidence locations

| Validation | Passed | Failures | Notes |
| --- | ---: | ---: | --- |
| Integrated fields, transport, scans, infrastructure and prior repairs | 458 | 0 | 774.19 s; before final observer/inventory/resume hardening |
| Final observer and calculation-performance coverage | 81 | 0 | 4.94 s |
| Final benchmark/comparison hardening | 92 | 0 | 8.88 s |
| Final CLI, sweep execution and experiment records | 58 | 0 | 433.33 s |

These groups overlap and must not be summed as distinct tests. Reported warnings
are upstream Pydantic deprecations and existing pytest `record_property`/JUnit
format warnings. Compilation and `git diff --check` also pass.

Full local receipts remain under `outputs/optimization-round-20261028/`:
`baseline-final/`, `optimized-final/`, `comparison-final/`, `validation.xml`,
`cli-final-plan.temexp`, `cli-final-run.temexp`, `cli-final-proof.json` and
`cli-final-report.{csv,json,txt}` with the CSV manifest. Earlier diagnostic
directories are retained but excluded from the accepted performance comparison.
The compact JSON beside this report preserves the principal evidence in the
repository; full local archives remain in the output directory.

Implementation identities recorded by the accepted suites are:

- Baseline: `3ed407a3e78f1130470a9d6bd064499b4d8a3dd0ef1acaa48690fec187be1c63`.
- Optimized: `f83cdadcf327ec51561a4cc942aa8a93e5e456fce51d0c025f2ec3beef675737`.

## Follow-up priorities

1. Extend this fixed-input benchmark to larger declared source populations and
   independent step refinement, with explicit time budgets and separate physical
   acceptance. Do not extrapolate the 49-particle ratios to 3,000 particles.
2. Instrument scan child processes and re-profile the optimized sampler before
   selecting another kernel. Scan calibration remains the longest measured warm
   case; parent waiting time alone cannot identify its next bottleneck.
3. Investigate remaining field sampling/assembly and identity-check overhead
   only where the refreshed profile justifies it, retaining exact source,
   interception, domain and cache-admission semantics.
