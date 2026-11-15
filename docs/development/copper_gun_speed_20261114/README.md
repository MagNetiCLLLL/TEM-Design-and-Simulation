# Copper Gun field and integration execution

This change accelerates the existing air-core copper-coil law and native Gun
discrete-gradient integration. It does not tune the optics, change source
sampling, alter physical boundaries, truncate magnetic tails, or relax any
integration or energy check. Preview and High accuracy use exactly 1024 original
positive-current particles from the preserved input snapshot.

"Unchanged controls" means the independent declared lens, corrector, source and
geometry settings. The existing High accuracy workflow additionally solves
automatic AC/Descan scanning calibration. Its derived coupling/readback values
and actual dynamic scan fields are recorded separately below. The original
Preview and High outputs are not a fixed-field step-convergence comparison.

## Measured cause

An isolated copy of the pre-optimization solver was instrumented without
changing its numerical statements. A real 1024-particle run accepted 64 Gun
steps in 137.2748423 s. It made 192 full/half/fine DG calls, with no rejected
step or failed implicit solve in this interval. Profiling attributed 134.934 s
to 384 instrument magnetic-field calls. These called 5376 annular-coil field
evaluations; axial derivatives consumed 131.217 s cumulatively (106.015 s self).
The expensive work was repeated at the same quadrature order inside the
implicit full/half/fine evaluations.

The unchanged closed electric-field cache has 313 radial and 2911 axial nodes.
Its initial 149 cells, from 0 to 2.3 mm, have 15--15.625 micrometre spacing.
Accepted axial increments in the extractor can be substantially smaller than
a cell: the original impulse cap and four error gates still determine the
population step. Mesh spacing alone does not identify the limiting gate or
predict full-job time. The globally smallest cells are beyond the 450 mm Gun
exit and do not explain the current near-cathode progression.

The previous magnetic packing path did not support copper coils, so these
steps used the general Python/NumPy DG evaluator. The previous worker split
also gave a 1024-particle bundle only one worker. The all-Z magnetic support
is physical input and is retained. It is not a fictitious stopping plane.

The first reference report failed an observer-only momentum comparison after
the run: it normalized an already normalized direction a second time. A new,
read-only receipt reconstructs momentum from the original slopes and verifies
all launch momentum bytes exactly. The original failure and raw results remain
unaltered. This diagnostic error did not alter transport or its timing.

## Execution changes

- Compile the same axial derivative polynomials and all 144 quadrature nodes
  using a common-factor recurrence and the original NumPy pairwise sum tree.
  Keep all six near-axis terms and the original far-field branch. No fast-math
  transformations are enabled. Unsafe ranges and unsupported compilation
  return to the original direct-power NumPy evaluator.
- Bound exact repeated-Z caching and return independently owned arrays. Include
  the helper implementation bytes and compiler arithmetic/version metadata in
  numerical field identities so old field/transport caches cannot acquire
  reuse authority after a kernel change.
- Pack supported copper sources into the existing implicit DG execution path,
  preserving instrument source order, per-owner winding order, and full tails.
  Unsupported poses, trial domains, or ambiguous convergence decisions repeat
  the identical native step before any trial is applied.
- Recompile the three composed cross-module JIT callers in each new process;
  retain compiled machine code in that process, and keep same-file scalar
  kernel caches. This avoids trusting persistent caller code after an imported
  helper changes. An existing stale numerical result was not reproduced; this
  is a dependency-integrity safeguard, not a claimed observed cache failure.
- Use 256-row work blocks for copper, retaining a single population time step,
  clock, and maximum residual barrier at every implicit iterate. Other field
  models retain their existing worker policy.
- Add optional context-local observation of original source arrays, step gates,
  retries, accepted states, timing, and compilation/fallback dispatch. Observers
  never return a numerical decision or become part of editable optics.

## Preserved numerical input

Raw execution evidence is under `outputs/copper-gun-speed-20261114/`. The
`baseline-input` manifest hashes the historical input, request and independent
source arrays. IDs, weights, emission coordinates, directions, energies and
times are compared exactly; no finite-N recentering or width adjustment occurs.
The configured energy width is 0.3 eV FWHM. The existing finite population has
launch kinetic energies 0.015094567656436075--0.7013491466779715 eV.

The captured assembly has three condenser lenses and the probe corrector.
Its specimen is not inserted; this is an optical-reference particle workflow,
not a Si material scan or a coherent calculation. The Gun exit is 450 mm and
the specimen reference is 1599.2 mm. Preview retains the 0.5 mm column step
and 2 mm history interval. The existing High accuracy request policy uses a
0.05 mm column step and 0.5 mm history interval. Both retain the original Gun
0.2 mm trace-step ceiling, unchanged optical controls and unchanged tolerances.

Full-result and final validation receipts are recorded separately from the
bounded reference profile. A bounded profile is not a complete Preview or
High accuracy result. Field microbenchmark speedups are not full-job speedups.

## Frozen implementation and software checks

The final raw Python implementation identity is
`50af4539a3dcd8e97b5ab7d586c13111659057fb98d3ada7a76c64abfc18ef35`.
The final focused regression passed **234 tests in 83.60 s**, including field
operators, fresh-process JIT dependencies, strict fallback, DG parity, observer
noninterference, source-prefix preservation, recoverable numeric persistence,
and cache identity. The suite uses mathematical arrays and mocked workflows;
it does not substitute small physical sources for the 1024-particle trials.

Independent input review verified all five preserved source/input files against
their originals, exact Gun/optical/assembly/sample graph preservation, unchanged
50 tracked configuration/profile files, and 14 original numerical criteria.
The existing High accuracy request resolution is 0.05 mm, not 0.1 mm.

One-process JIT startup was measured separately on a seven-row mathematical
fixture: 14.26 s for the first call and 0.394 ms median for resident subsequent
calls. This is a compiler startup/operator test, not finite-source throughput.

## Matched 1024-particle reference interval

Both solvers accepted 64 steps with zero retry and 192 full/half/fine DG calls.
All executed launch arrays are byte-identical. Every optimized DG dispatch used
the copper compiled channel. The accepted/rejected classifications match.

| Measurement | Original | Optimized |
| --- | ---: | ---: |
| Profile wall time, including initialization/JIT | 137.2748423 s | 26.9027660 s |
| Accepted-step median excluding first step | 2.1570958 s | 0.1850585 s |

This is a **5.10x measured bounded-interval wall speedup** and **11.66x resident
accepted-step median speedup**. It is not a full-column speedup estimate.
Evaluation-order roundoff is visible: maximum relative dt difference is
8.90e-16, maximum saved coordinate difference 3.71e-22 m, and maximum relative
momentum-vector difference 3.32e-15. These are far below the preserved original
error budgets. Numerical trajectories and time steps are not claimed to be
bitwise equal; the physical source is.

## Completed Preview

The official Preview worker completed the optical-reference workflow in
3210.5337729 s (53.51 min), including source/Gun and both column segments.
All 1024 original positive-current particles arrived at the Gun exit and
specimen reference; the effective specimen count is 1024. The absolute source
current remains 1e-5 A and source weight sums to one. The downstream fluorescence
screen correctly intercepts the post-specimen path; its loss ledger is not
renormalized into surviving current.

Gun exit kinetic energies are 300000.01509460155--300000.70134922117 eV.
The maximum work-relative exit energy error is 9.0978574e-8 eV, below the
unchanged 0.001 eV budget. Executed column energies and pre-stop flight times
are finite; column clocks are monotone and the full incident clock validator
passes. Unknown energy values in Gun drawing-history rows remain unknown,
while the actual Gun states and exit energies retain their native checks.
The formal recoverable Gun/incident store round-trip passes, and original
emission fields and optical controls remain byte-exact/unchanged respectively.

The full Gun accepted 3502 population steps, made 9444 attempts and rejected
5942 trials under the original gates. Transverse momentum failed in 5775 of
those rejections; gate failure counts overlap. The full/half/fine trials made
22390 DG calls and spent 2958.0653171 s in those calls. This retained numerical
work explains why the full cold start still takes tens of minutes despite the
per-step acceleration; the integration is not replaced by coarse transport.
Gun plane-crossing root finds also perform 26744 smaller DG reintegrations.
Across all Gun DG dispatches, 48622 of 49134 use compiled copper (98.96%);
512 strictly repeat the native calculation for the recorded combined
trial-domain/roundoff admission reason. That reason cannot be split further
from the saved counters.
An independent read-only audit checks the stored arrays, including exact
Gun-exit/incident and specimen/post-specimen clock continuity, all 4,978,688
required positive finite energy values, preserved unknown Gun-prefix values,
the detector's physical 10 microampere interception, and closed current ledgers.
It finds no implicit-solve failure. Rejected attempts consumed 1998.585 s
against 962.532 s for accepted attempts, so the original error control is a
substantial remaining cost.

## High accuracy numerical execution and archive recovery

The normal High accuracy numerical worker reached its full endpoint in
2909.128643 s. It used 1024 original particles, reused the same native Gun
arrays, integrated the incident column with 23013 nodes and a nominal 0.05 mm
maximum step, and retained 1024 effective particles at the specimen.

The first validation script then rejected two `lower_coil_gain` changes in AC
and Descan. Existing component `to_dict` methods and calculation-cache rules
explicitly define these as compatibility readbacks of automatically solved
2x2 matrices, not independent user controls. All other optical/source controls
were unchanged. The rejection and raw records remain intact. This validation
script initially preserved only the formal Gun/incident seed; the completed
post-specimen arrays were not archived before this guard ran.

This distinction does not make the scan fields equal: scanning is enabled in
the preserved inputs. The original Preview uses its unresolved initializer,
whereas High performs its normal automatic 17-chief response solve. For example,
the upper AC command at the recorded origin is approximately
`(-5e-5, -4.921875e-5) rad` in Preview and
`(-1.9017386e-8, -2.2934426e-8) rad` in High. The resulting specimen coordinates
cannot be used to claim fixed-field convergence. No scan is disabled and no
default lens or correction strength is retuned to hide this difference.

The recovery procedure separately checks every independent input and saves all
resolved coupling/command matrices and field metadata. It restores the genuine
High specimen checkpoint through the official codec, retrieves the existing
calibration without any new chief trace, and requires the entire captured
runtime, dynamic coils, geometry and numerical field identity to match the
original High exactly before calling the official downstream rebuild.
The matching runtime digest is
`50c1d6995691954e597bfc8c58ad0c9e65f343aeeffeb596426f0bae7be579a2`;
the numerical snapshot digest is
`0d893312694840ecd630e0ac962f376dba01e76d943c9f56400c640739f9422e`.
Source emission, Gun transport, pre-specimen propagation and new calibration
traces are explicitly forbidden during this recovery. Its real full-result
archive is written before statistical verification so that post-specimen
evidence is retained if another analysis check fails.

The downstream rebuild completed in **1108.1023425 s**. The formal 2,924,108,230
byte archive was written and read back exactly before the remaining guards.
Independent audit confirms all 63 original upstream array fields, all source
fields, the entire native Gun and the actual runtime are unchanged. Specimen
checkpoint position, slopes, energy and clock join the post branch exactly.
All 1024 particles are physically absorbed at the fluorescence screen at
2714.15 mm; the absolute intercepted current is 1e-5 A and ledger residual is
zero. No new source, Gun, pre-specimen or chief-ray operation was attempted.
The complete archive contains the executed post arrays; the rebuild API only
retains restart authority for Gun and incident checkpoints, not internal post
checkpoints.

## Fixed actual field: Preview versus High accuracy

A separate Preview control uses the genuinely completed High result and its
resolved automatic scan calibration, while preserving its own official
runtime/source identities. It is not a relabelling of the earlier fresh
Preview. Independent comparison verifies every actual provider payload in
order, the complete dynamic laws, gauge, geometry and policies. Their common
operator witness digest is
`1146c2766ad13b8fdc73808642ff19776f80eb691e3b1853251c4bc260954d7f`.

The official Preview request uses 0.5 mm steps and 2 mm history. Its worker
completed in **158.8868834 s**, reusing the genuine Gun and freshly executing
the entire incident and post column. Execution starts at index zero at 450 mm
and 1599.2 mm respectively, with 2578 incident nodes and 2839 post nodes.
Only the Gun-exit integration boundary and upstream Gun display history were
reused; no High fine-grid column or downstream prefix was substituted. All
102 numeric archive arrays were independently checksum-verified, and all 26
recorded geometry/energy/clock round-trip fields match.

At the specimen, paired by the same 1024 source IDs:

| Quantity | Maximum absolute Preview minus High difference |
| --- | ---: |
| x | 1.3604311e-15 m |
| y | 1.3704193e-15 m |
| flight time | 3.2673634e-21 s |
| kinetic energy | 0 eV, byte-exact |

The project's specimen-plane D95 is 570.954465681 nm for High and
570.954463798 nm for this fixed-field Preview. These values describe the
unchanged copper optical-reference setup; they do not qualify source-count
convergence, an optimized/atomic probe, dual-condenser performance, material
scattering or coherent imaging. No focus or probe-size adjustment was made.

## Retained evidence

`validation/manifest.json` inventories the curated raw receipts and their
SHA-256 hashes. It includes the preserved source/input files, compressed exact
input, cache-free original reference Python package, frozen implementation,
regression XML, executed comparison receipts, independent audits and recovery
scripts. The original validation failure remains included and is never
overwritten by a success receipt.

Large complete native archives remain local:

- `outputs/copper-gun-speed-20261114/high-post-recovered/high-post-recovered.temsim`
- `outputs/copper-gun-speed-20261114/fixed-high-field-preview-execution-1/fixed-high-field-preview.temsim`

Their complete checksums are retained in the curated manifest. Both restore
through the official typed archive loader. This round executes classical CPU
particle transport; it requests no new coherent calculation and changes no
CUDA admission policy.
