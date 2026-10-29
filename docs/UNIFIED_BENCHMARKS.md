# Unified performance and accuracy benchmarks

Run the production benchmark suite from the project virtual environment. Every
case receives a fresh owned worker process. The output must be a new directory;
existing evidence is never overwritten.

```powershell
.\.venv\Scripts\python.exe scripts\run_benchmark_suite.py --output outputs\benchmark-baseline --cases nanoprobe --rays 49 --step-mm 0.05 --threads 8 --budget-seconds 1200 --profile
```

To run all declared cases, omit `--cases`. To measure the complete 3,000-particle
population, set `--rays 3000` and choose a suitable explicit time budget. The
default 49 particles bound the workload; they do not establish source sampling
convergence or qualify the stored Nanoprobe preset.

## Cases and timing semantics

| Case | Measured operation | Preparation and reuse |
| --- | --- | --- |
| `nanoprobe` | Production tip emission, physical gun and shared-potential incident-column transport to the specimen upper surface. | Default Finite-Field Optics / Nanoprobe / Diffraction lens controls, `Numba CPU`, static raster drives. First call in a fresh worker; three warm recomputations with legitimate upstream caches; three dependency-checked same-result reuses. |
| `continuation` | Production continuation from a saved local prefix to a bounded pre-specimen terminal plane; default 460 to 470 mm. | The existing `continuation_planes` helper selects two 10 mm spans after the gun exit, shortening them only when needed to remain before the specimen. Physical tip-to-prefix generation, save and each load are timed separately. Cold means first continuation after that preparation in a fresh worker. Three warm continuations and three terminal reuses retain original particles, weights, stops and prefix checkpoints. |
| `scan_calibration` | Actual finite-coil shared-field AC/Descan response measurement and drive solution. | A private response-cache directory starts empty. Warm recomputations clear only this isolated response cache. Reuse alternates real memory and disk response lookup. These local differential probes are not an emitted beam. |
| `coherent` | `NOT_RUN`, with an explicit reason. | A future separately scoped coherent fixture requires usable CUDA before execution. This declaration never launches a coherent pipeline. |

All cases preserve the stored optical controls. There is no fitting, refocusing,
source replacement, aperture enlargement, dropped particle population or weak
field truncation. Raster drives are disabled explicitly for the two static beam
cases; their underlying coils, alignment and geometry remain production inputs.
Scan calibration is an independent case with raster channels enabled.
History is retained at `max(step_mm, 0.5)` mm, matching the existing high-accuracy
request and design-snapshot contract. This controls saved history spacing; the
integration step remains the explicitly requested `step_mm`.
The fixture explicitly records `High accuracy` before its first input snapshot,
matching the existing archive-load quality contract. Continuation cold, warm and
reuse requests all prepare their state from the same checked prefix archive;
the terminal result remains the reused calculation for the reuse phase.

Warm recomputation is not a new process and may reuse legitimate gun and field
caches. The outer-call counters record whether gun and column transport actually
executed. A cold worker does not mean cold filesystem, operating-system or Numba
disk caches. The continuation case does not claim that prefix preparation is free.
Its receipt requires the actual resume plane to equal the saved prefix, a positive
start index and fewer executed intervals than the complete local plan. An entrance
restart or complete recomputation cannot pass as continuation. This case measures
local reuse; only `nanoprobe` covers propagation to the specimen upper surface.

The worker enforces a cooperative deadline covering setup, calculations,
observations, archives and the optional profile pass. The supervisor enforces a
hard timeout after `--cleanup-grace-seconds` (default 30), using the existing owned
process-tree cleanup. Unconfirmed cleanup blocks subsequent cases. Partial runs
retain an atomic `RUNNING`, `FAILED`, `CANCELLED` or `TIMED_OUT` receipt.

## Evidence

- `suite.json`: configuration, all case receipts, supervisor exits and status.
  `status` describes execution. `particle_repeatability` and `acceptance_status`
  separately report strict joint acceptance; requested particle cases require
  `PASS` for exit code 0, even if every worker completed. Scan-only/coherent-only
  suites report `NOT_APPLICABLE` for this particle check. Physical qualification
  remains independent and is never promoted by this acceptance.
  Every executed case must retain the complete requested cold/warm/reuse run
  inventory, with unique repeat indices and finite elapsed times; a missing case
  or repetition cannot be hidden by another case's successful observations.
  Every executed case must also contain the complete configured cold/warm/reuse
  inventory with unique phase/repeat identities, complete runs and finite times;
  another passing case cannot conceal an empty or incomplete case.
- `<case>/inputs.json`: immutable input graph, external content identities and
  solver implementation identity.
- `<case>/case.json`: per-operation elapsed time, separate preparation and
  observation time, original-source evidence, production counters and limitations.
- `<case>/stdout.log`, `stderr.log`: worker diagnostics.
- Particle cases save `terminal.temresult`; continuation also saves
  `prefix.temresult`. Archive I/O is outside transport timing and reported.
- `accuracy-summary.json` and `.txt`: the common physical-observable and
  repeatability report. Calibration retains an independent matrix/response
  summary and is never represented as a full emitted-ray observation.
- With `--profile`, `<case>/profile.prof` and `profile.txt` contain a separate
  additional pass, including the top 30 entries by own and cumulative time.
  The profile pass never contributes to cold/warm/reuse timing statistics.
  Profiling covers the benchmark worker only. Scan calibration may delegate its
  physical response calculation to a child process, whose Python calls are not
  captured by this profile; parent `readline`, wait or lock time is not a physical
  kernel hotspot. Total operation elapsed time still includes that wait.

When `psutil` is installed, benchmark-worker RSS is sampled every 20 ms; descendant
processes are excluded, including scan-response workers. Brief peaks can
be missed. If unavailable, the report records `null`, `NOT_MEASURED` and a reason;
it never reports zero as a measurement. VRAM is explicitly unmeasured for these
classical CPU cases. Thread budgets use the existing numerical job controller;
BLAS remains serial within its policy.

Input preparation, snapshots, observers and archives have separate durations.
Nested production stage timings overlap with total operation time and must not
be added together. Execution completion, numerical repeatability/convergence
and physical qualification are distinct statuses. No passing runtime result can
promote an unqualified optical topology or missing scientific evidence.

Before collecting a baseline, finish all source edits. Changing implementation
identity during a worker invalidates the run. Compare the same physical and
numerical input identities, particle population, observation planes and backend;
record changed implementation identities explicitly. Preserve the baseline and
use another new directory for an optimized version.

## Compare complete baseline and candidate suites

```powershell
.\.venv\Scripts\python.exe scripts\compare_benchmark_suites.py outputs\benchmark-baseline outputs\benchmark-candidate --output outputs\benchmark-comparison
```

The output directory must be new. `comparison.json` and `comparison.txt` retain
the exact-equivalence result and before/after median seconds and speedup for each
case and phase. Both suites must be complete, with identical configuration,
case/phase/repeat inventories and CPU allocations. Saved inputs are checked against
their archive digests and their execution receipts. Implementation digests are
recorded separately and may differ between the suites.
Scan comparison also requires the same actual response execution mode, worker
count and trace count, including the original measurement receipts on cache hits.

Particle comparison rebuilds strict reports from every retained run and uses
`performance_accuracy.compare_reports`; no failed or missing run is dropped.
Scan comparison independently binds the captured physical/numerical graph and
external inputs, requires exact matrix fingerprints and finite residual equality
within and across both suites, and checks the actual cache source for each run.
Actual scan execution mode, worker count and trace count must also match;
equal parent thread budgets alone do not establish equal child-process resources.
Receipt wall-clock values are excluded from that resource identity.
Observer versions must be known and consistent. The only admitted version change
is v2 to v3 for unchanged `specimen_upper_surface` or `section_terminal` observations;
all input, raw-array, plane-position and scalar comparisons remain strict. The new
`sample_reference_plane` kind requires v3 on both sides. Unknown versions and
version downgrades are rejected. Original suite receipts are never rewritten.
The complete inventory includes an explicit unmeasured coherent declaration when
present. A coherent-only comparison is `NOT_EVALUATED`, never `PASS`.

Exit code 0 means same-input equivalence passed; 1 means failed or not evaluated;
2 means invalid command usage or an unavailable new output directory. A positive
speedup is a timing ratio, not evidence of convergence or physical qualification.
The comparison reads evidence only and does not restore inputs, run a solver or
change the source suites.

Observer v3 adds exact sample-reference-plane observations for ordinary classical
experiments. Its specimen-upper-surface and bounded-section-terminal contracts
are unchanged from v2. The comparator explicitly permits v2-to-v3 only for these
two unchanged plane kinds and records that compatibility decision. Unknown
versions, reverse migration, or a v2 claim of the new plane kind are rejected.
Exact input, plane position, raw fingerprints and zero-tolerance scalar checks
remain mandatory.
