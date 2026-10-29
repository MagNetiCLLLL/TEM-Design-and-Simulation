# Headless classical experiments

`python -m temsim.experiment_cli` provides `plan`, `run`, `resume` and `export`
without constructing a `QApplication`. An installed package also provides
`tem-design-simulation-experiment`. Run commands with the project interpreter
and `PYTHONPATH=src` when working from the checkout.

This entry uses the existing `DesignRecipe`, `ParameterSweep` and `.temexp`
records. It does not load executable expressions from input JSON, replace the
physical source, fit optical settings, or enable coherent calculations. Plans
with coherent readout flags or a source other than `classical_particles` are
rejected explicitly and left unchanged.

## Plan before running

Create `axes.json` with existing registered runtime parameter paths, explicit
numeric values and units. This single-point example observes the stored C2
setting rather than performing an optimization:

```json
[
  {
    "path": "lenses[condenser_lens_2].percent",
    "values": [13.899040164772698],
    "unit": "%"
  }
]
```

From PowerShell in the project checkout:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m temsim.experiment_cli plan --default-nanoprobe --rays 49 --step-mm 0.05 --axes axes.json --output plan.temexp
.\.venv\Scripts\python.exe -m temsim.experiment_cli run plan.temexp --output run.temexp --timeout-seconds 600
```

The default fixture shares `benchmark_suite.configured_state`: the default
assembly and stored Finite-Field Optics / Normal Nanoprobe / Diffraction
presets, classical Numba CPU execution, and disabled raster/wobble for a
declared static observation. The source population and integration step must
be supplied explicitly. The existing Design Explorer high-accuracy contract
sets stored trajectory spacing to `max(step_mm, 0.5)` mm. This observation
spacing is distinct from the integration step and cannot resolve every focus
or crossover. The simulator's ordinary default population remains 3,000;
`--rays 49` is this example's explicit experiment budget.

Alternatively, preserve a saved Design Explorer recipe and its source,
geometry, scan, backend and numerical settings:

```powershell
.\.venv\Scripts\python.exe -m temsim.experiment_cli plan --recipe saved-recipe.json --axes axes.json --output captured-plan.temexp
```

Do not pass `--rays` or `--step-mm` with `--recipe`. Numerical requests already
belong to that recipe. Per-point reconstruction preserves its optical
controls and never silently refocuses a deliberate parameter perturbation.

Multiple axes form a Cartesian product. Values within one axis must be
unique. Plans default to at most 16 points; `--maximum-points` may explicitly
raise this limit to 256. A plan performs input/identity preflight and writes
an ordinary zero-point experiment record; it performs no particle propagation.

Optional tolerances use a JSON list, supplied by `plan --tolerances file.json`:

```json
[
  {"metric": "sample_surviving_current_pa", "minimum": 1.0}
]
```

This is an example user-selected bound, not an instrument specification.
Tolerance bounds are inclusive. Missing measurements fail their declared
tolerance rather than being treated as zero or as a pass.

## Checkpoints, cancellation and resume

Each successful or failed point is committed atomically before the next point
begins. Failed points retain the reason and carry no successful metric values.
The CLI uses the existing scalar experiment format; it does not put full ray
or wave arrays in the checkpoint. The `point_observer` evidence retains the
separate execution, numerical and physical status supplied by the common
performance/accuracy observer.

`run` always requires a new output path. It validates the captured plan and
runs its points afresh. `resume` is the explicit operation that updates an
existing checkpoint and computes only missing points:

```powershell
.\.venv\Scripts\python.exe -m temsim.experiment_cli resume run.temexp --timeout-seconds 600
```

The complete inputs, runtime, implementation, observer identity, axes and
tolerances must match. Changed inputs or code require a newly captured plan.
Retained failed points remain failed; resume does not silently reinterpret or
retry them. A failed final write stops execution and leaves the last valid
atomic checkpoint. External edits to a live checkpoint are detected before
replacement. Concurrent CLI writers are blocked by a sibling `.lock` file;
after a process is forcibly terminated, remove that lock only after confirming
that no writer remains.

Ctrl+C requests cooperative cancellation at the next existing solver boundary.
The default wall-time allowance is 600 seconds per invocation; an explicit
resume begins a new declared allowance. The timeout cannot interrupt a native
kernel or guarantee an immediate process exit. A cancelled in-flight point is
not retained. A hard process termination can lose only work since the last
completed-point publication, and does not turn partial work into a successful
point. An in-progress checkpoint may have fewer points than its plan even when
its `cancelled` flag is false; completion is determined from both counts.

Exit codes are:

| Code | Meaning |
| --- | --- |
| 0 | Command completed; no failed points or failed declared metric tolerances |
| 1 | At least one failed point or failed declared metric tolerance |
| 2 | Invalid inputs, changed identity, incompatible record, or I/O failure |
| 124 | Cooperative invocation timeout |
| 130 | User interruption/cancellation |

Exit code 0 is a software-execution outcome. It is not a physical qualification.

## Export retained evidence

```powershell
.\.venv\Scripts\python.exe -m temsim.experiment_cli export run.temexp --format csv --output observations.csv
.\.venv\Scripts\python.exe -m temsim.experiment_cli export run.temexp --format json --output joint-report.json
.\.venv\Scripts\python.exe -m temsim.experiment_cli export run.temexp --format report --output joint-report.txt
```

CSV export includes the existing JSON provenance manifest. JSON and readable
reports use the shared performance/accuracy report model, with a separate case
for each parameter point. Different inputs are never compared as repeated
executions of one case. A single observation does not establish repeatability;
unmeasured RSS/VRAM remains unavailable rather than zero. Exports read saved
observations without invoking the solver and refuse existing output files.

Ordinary classical calculations retain an exact incident checkpoint at
`sample.z_mm`, the optical reference plane. When the specimen upper surface is
not retained, the observer may report this separately as `sample_reference_plane`
only after the plan, incident trajectory and checkpoint endpoints all agree
exactly. These are incident optical statistics, including when the specimen is
retracted; they do not establish specimen-surface focus or specimen scattering.
Focus and topology remain `NOT_EVALUATED` for that observation. Upper-surface
observations take priority when the exact checkpoint exists.

Point `duration_s` includes input preparation, calculation and metric
extraction, excluding observer and checkpoint serialization time. Optional
observer overhead is recorded separately as `evidence.observer_duration_s`.
Use the dedicated benchmark suite for controlled cold/warm/reuse timing studies.

## Scientific limits

The stored unified default preset has a historical 49-particle calibration
record, not established source-sampling or topology qualification. Its current
catalog labels remain `NOT_TESTED` for those checks. The later 193-particle
direct-focus candidate also remains short of the joint probe-size requirement;
its historical evidence is not a new run under repaired observer code. Neither
a successful sweep, an optimizer status nor a saved single-point report changes
those limits.

Source weights, physical interception and deliberately changed focus remain
observable. Candidate acceptance requires its own declared focus, topology,
angle, current and size constraints together with independent numerical
refinement. Coherent wave calculations remain outside this classical CLI and
subject to the project's separate usable-CUDA policy.
