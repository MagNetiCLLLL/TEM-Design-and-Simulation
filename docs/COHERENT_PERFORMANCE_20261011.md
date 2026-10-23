# Measured coherent propagation performance improvements (2026-10-11)

This work first measured fixed inputs, then changed wave storage, radial GPU data residency, and checkpoint readout.
It did not change the electron source, lens/deflector/aperture strengths, sampling grids, error thresholds, or numerical precision.
The shared CPU/GPU settings at the top remain in use; no calculation buttons or device selectors were added.

## Measured speed

Windows, NVIDIA GeForce RTX 5090, CuPy 14.1.1, float64/complex128.
Timing tasks ran sequentially; GPU timings synchronized after completion. Warm-up and measured runs were separate.

| Measurement scope | Before | After | Speedup |
|---|---:|---:|---:|
| Actual in-column continuation of a fixed Si wave, 3×6720×6720, including file verification, loading, and device transfer | 9.421 s | 6.157 s | 1.53× |
| Verification, 128² display, and full-grid beam centroid/width statistics for the same Si checkpoint | 12.027 s | 8.095 s | 1.49× |
| Radial operator benchmark, 1,048,576 points, 6 drift/C3/aperture operations | 1267.009 ms | 40.748 ms | 31.09× |
| Cartesian operator benchmark, 1536², 6 canonical propagations | 52.493 ms | 47.824 ms | 1.10× |

These scopes differ: their speedup factors cannot be multiplied or treated as a speedup for the complete microscope calculation.
The Si in-column continuation used an existing GPU-executed wave as a fixed input boundary and advanced only 0.001 mm,
using the current default instrument state saved at the start of this work. It did not rerun the entire tip-to-specimen path.
The old Si instrument snapshot was incompatible with the current format, so the program's input checks were not bypassed and old physical parameters were not migrated.
The before/after continuations used exactly the same current instrument snapshot and old wave boundary. This is a local performance/regression test,
not evidence that the old wave was produced by the current default instrument.

The Si timing is the median of 3 runs after the first execution; the readout test is the median of 3 sequential runs.
The operating-system file cache was not cleared, so these results do not represent cold-disk startup speed.
The small Cartesian benchmark used 12 comparisons with alternating order in the same process to reduce GPU clock and scheduling fluctuations.
The radial benchmark used scaled operator inputs suitable for verification, rather than real microscope source parameters; inputs were identical before and after.

## Stage timings and reasons for the changes

The independent profile of the actual Si in-column continuation is listed below. Timings include nested calls and must not be summed row by row:

| Call | Before | After |
|---|---:|---:|
| Load, validate, and construct execution modes | 5.296 s | 2.688 s |
| Array-file SHA-256 | 2.611 s / 30 calls | 1.347 s / 15 calls |
| Wave construction/storage initialization | 2.575 s | 1.669 s |
| Probability statistics | 1.530 s / 32 calls | 0.969 s / 25 calls |
| Canonical propagation operator | 0.476 s | 0.458 s |
| Device-to-host transfer | 0.490 s | 0.467 s |

Preparation of the in-column physical plan took 0.343 s, essentially unchanged before and after, and was not counted toward the propagation speedup.
Profiling showed that budget preflight loaded all complex arrays and execution read them again.
Preflight now reads dimension metadata; complete files and actual grids are still verified before a mode is used.
Corrupted checkpoints still raise errors.

1. **Wave storage**: public constructors continue to copy inputs; newly generated internal GPU arrays transfer ownership,
   and arrays are reused when only metadata such as origin/curvature changes. Finite values, coordinates, and curvature are still checked.
   Immutable CPU arrays can cache probability; callers can modify GPU arrays, so their probability is recalculated every time.
   GPU probability uses a double-precision fused reduction, avoiding intermediate magnitude and squared arrays.
2. **Radial propagation**: each mode uploads on entry; propagation, phase, spherical aberration, apertures, sampling checks, and refinement stay on the GPU,
   with download at the final handoff to Cartesian calculation. Benchmark upload/download volumes each fell from 192 MiB to 24 MiB.
   Non-axisymmetric and other unsupported cases still use the original two-dimensional path; no physical interactions or sampling checks were removed.
3. **Checkpoint readout**: validated immutable snapshots are reused within a request, with a cache limit of 128 MiB,
   also constrained by the existing readout and RAM budgets. The cache is released when the request ends, and the next request verifies the files again.
   Modes exceeding the cache limit remain streamed, with display and statistics combined into one pass.
   Si readout array verification/loading fell from 51 to 36 calls, and mode construction from 9 to 6 calls;
   the actual additional snapshot cache for this large-grid test was 0 bytes.

## Agreement and scope

- All 3 complex-amplitude arrays in the Si continuation were equal element by element, with identical intensities, weights, energies, grids, origins,
  curvatures, and tilts. Comparison did not first fit phase, shift the waves, or renormalize them.
- Si display density was bitwise identical, and integrated probability, beam centroid, beam width, and energy statistics were identical.
- The radial operator's complex-field L2 difference with radial integration weights was **1.66×10⁻¹⁵**, its probability difference was 0, and its radial grid was bitwise identical.
  Division by radius at very small radii amplifies FFT rounding error, so bitwise equality of that complex array was not claimed.
- The Cartesian operator's maximum complex-field difference was 0, and its probability-reduction difference was **2.22×10⁻¹⁶**.
- Tests for file integrity, insufficient sampling, physical truncation, CPU/GPU comparison, cancellation, cache release, and memory budgets were retained.
  CI without a GPU may mark only explicitly registered hardware tests as not run; execution failures must not be treated as passes.

Final integration coverage comprised 458 distinct cases, with none skipped. The first run had 457 passes and 1 failure from an old memory expectation:
the small-RAM budget must reserve an additional 64 MiB for the per-request cache. After the assertion was corrected, the 2 related budget tests passed;
the other cases that had already passed were not rerun. The 3 new test files were registered in the existing acceptance scopes.
An outdated numerical-test premise already present in the baseline was also corrected: an FFT-friendly grid now fits within the original 8 MiB limit,
so the test limit for the rejection path was changed to 6 MiB. This changes only test input, not the software's calculation budget.
The old radial-routing test gained its missing mode identity and isolated a separate aperture optimization path that this test did not validate.

## Reproduction and original evidence

The working directory is the project root, with interpreter `.venv\Scripts\python.exe`.
The actual workspace before these changes, including the uncommitted edits present at the time, is saved in
`outputs/wave-speed-20261011/baseline/`; Git HEAD was not substituted for the baseline.
Source-file hashes and confirmation that configuration files were unchanged are in `source-manifest.json`, one directory above that baseline.

SHA-256 of the fixed current instrument snapshot:
`fc723070bb4d6baab9e0d3356778eec30ffd259e476961028239586d95697af4`.
Si input checkpoint digest:
`3caca88a40bc7c267a8123dc86fa9d23fca17e73e870ce214e4ef168f049f832`.

```powershell
.venv/Scripts/python.exe scripts/benchmark_wave_hotpaths.py --source-root outputs/wave-speed-20261011/baseline/src --fixture outputs/gpu-coherent-20261004/si-public-final-gpu-v2 --instrument outputs/wave-speed-20261011/benchmark-instrument.json --output outputs/wave-speed-20261011/column/before --profile
.venv/Scripts/python.exe scripts/benchmark_wave_hotpaths.py --source-root src --fixture outputs/gpu-coherent-20261004/si-public-final-gpu-v2 --instrument outputs/wave-speed-20261011/benchmark-instrument.json --output outputs/wave-speed-20261011/column/after --profile
.venv/Scripts/python.exe scripts/compare_wave_hotpaths.py outputs/wave-speed-20261011/column/before outputs/wave-speed-20261011/column/after --output outputs/wave-speed-20261011/column/comparison.json
```

Complete local evidence is in `outputs/wave-speed-20261011/`:

- `column/{before,after}/`: individual timings, independent stage profiles, all complex arrays, and device execution records.
- `column/comparison.json`: per-array comparisons without phase alignment.
- `radial/`: radial reproduction script, before/after waves, timings, and comparison results.
- `plane/alternating.json`: small-grid GPU measurements in alternating order.
- `checkpoint/si-{baseline,optimized,comparison}.json`: actual Si readout, input hashes, density, and statistics comparisons.
- `integration-tests.xml`, `memory-budget-rerun.xml`, `validation-summary.json`: integration, targeted reruns, and combined results, retaining the initial failure record.

The original large wave arrays remain in the local output directory and are not included in the source repository.
