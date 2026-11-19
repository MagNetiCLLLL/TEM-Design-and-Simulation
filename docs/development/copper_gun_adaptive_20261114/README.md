# Copper Gun accepted-step feedback

This numerical execution change preserves the default independent optical
controls, original 1024-particle source and 0.3 eV FWHM energy law, physical
boundaries, quadrature and every original accuracy threshold. It changes the
first trial proposed after a successful geometry-field vacuum step. All four
original full/half/fine gates, the shared population residual, halving retry,
half-to-full retry reuse and lack of geometry-field energy projection remain.

## Where the 5942 rejected trials occurred

The completed reference implementation is
`50af4539a3dcd8e97b5ab7d586c13111659057fb98d3ada7a76c64abfc18ef35`.
Its complete observer contains 9444 attempts: 3502 accepted, 5942 rejected,
with no implicit-convergence failure or nonfinite gate trial.
Rejected attempts consumed 1998.585 of 2961.117 recorded seconds (67.49%).

| Starting axial band | Accepted | Rejected | Attempt time | Share of all attempt time |
| --- | ---: | ---: | ---: | ---: |
| 0.5--1.1 mm | 728 | 4875 | 1853.835 s | 62.61% |
| 0.65--1.0 mm, subset of the preceding band | 528 | 3785 | 1537.781 s | 51.93% |
| 14--25 mm | 180 | 568 | 230.164 s | 7.77% |

The exclusive largest normalized rejecting gate was transverse momentum in
5663 trials (95.30%), XYZ position in 172 and XY position in 107. Nonexclusive
failed-gate counts overlap and must not be added as separate rejected trials.

Each outer step recomputed a spatial/field/impulse ceiling and discarded the
smaller duration just accepted. After 1103 steps needing retries, the next
first trial was always more than 1.5 times the previous successful duration;
the median ratio was 64.025 and maximum 290.303. Of these, 1088 returned to
within 5% of the preceding initial proposal. This is direct evidence of
repeated restart-and-halve work, rather than a failed implicit solve.

The beam at the 0.788--0.804 mm recorded snapshot had maximum radius 53.6 nm,
inside the 250 micrometre extractor bore and first 15.625 micrometre radial
mesh cell. It was not near-wall interception. The bunch spans about 16.3
micrometres axially, similar to local axial cells. Interpolation-interface
effects are plausible but are not individually identified by this observer.
The empirical transverse normalized-error halving order is about 1.03, not
uniformly three; a cubic error law is not assumed for these production fields.

## Numerical policy

`SurfaceStepPredictor` lives only inside one executed static geometry-field
vacuum trace. The next first trial is bounded by all freshly computed original
spatial, impulse and captured-field scope caps, and by remembered successful
step feedback. The damped scalar gain is limited to 1--1.25 after acceptance;
its cube-root expression is a feedback heuristic, not a new error tolerance
or a claim about the local convergence order. Rejected trials still halve by
the original code and satisfy the original four gates before acceptance.
Retry and active-population changes temporarily suppress growth. Provider or
clock discontinuity clears the trace-local proposal context. Stochastic medium
transport retains its prior selection path.

Already accepted steps never automatically shrink their subsequent suggestion:
otherwise a nearly constant normalized error of 0.99 can make successful step
durations form a convergent geometric sum and stall the floating-point clock.
A 400-step pure-record regression demonstrates continued progress.

Plane localization additionally reuses only exactly repeated fractions inside
one root solve, with the same single-ray start, duration, fields and tolerance.
It never substitutes a batch endpoint. Fresh complete native-provider packing
admission rejects unknown nested providers, instance callbacks or unsupported
backends; these keep the original repeated root evaluation. The original Brent
fraction tolerance and both DG half steps are unchanged.

## Frozen implementation and validation

The new tested implementation is
`6169010971ca2aeb082c8f0936b043f2ad4eb886cdfd07991c69bdda32a7ffca`.
The method schema is `geometry-dg-transverse-step-doubled-v3-trace-local-proposal`
and proposal schema is `surface-dg-bounded-feedback-proposal-v2`.
All eleven audited numerical constants match the reference AST exactly.
No default configuration or profile is edited.

The final targeted software suite passed **304 tests in 107.23 s**. It includes
all four unchanged error gates, successful-step feedback, growth/cap bounds,
clock continuity and extremes, active-mask/provider changes, strict nested
provider admission, exact root reuse, field/DG parity, source identities and
numeric cache persistence. Pure-array and mocked checks do not substitute for
the required new full 1024-source physical runs.

Raw evidence and executed results are retained under
`outputs/copper-gun-adaptive-20261114/`. The old reference package and raw source
arrays are separately frozen. New-solver Preview starts with no old Gun or
particle result; any High reuse must come from this new solver's completed
Preview. Each quality writes its complete formal numeric archive before
statistics, control guards and incident-seed checks. Independent user inputs
are checked separately from the existing derived AC/Descan calibration
readbacks, and actual dynamic fields are recorded explicitly.

## Completed Preview comparison

The new Preview completed the full optical column, saved its complete formal
result, read that result back exactly and retained all 1024 positive-current
originals at the specimen plane. Emitted, surviving and effective populations
are all 1024; absolute specimen-plane current remains 10 microamperes and the
source-weight sum is one. The original source IDs, launch positions, momenta,
energies, emission times and weights are byte-identical to the reference.

| Observed quantity | Reference 50af | Adaptive 6169 |
| --- | ---: | ---: |
| Accepted Gun steps | 3502 | 3801 |
| Trial attempts | 9444 | 3984 |
| Rejected trials | 5942 | 183 |
| Surface-step full/half/fine DG calls | 22390 | 11769 |
| Recorded attempt time | 2961.117 s | 1255.557 s |
| Rejected-attempt time | 1998.585 s | 62.292 s |
| Full Preview worker time to result callback | 3210.534 s | 1512.922 s |
| Invocation including result persistence | 3213.832 s | 1526.433 s |

Rejections fell by 96.92% and the observed surface-step DG calls by 47.44%.
The recorded Preview worker speed ratio is 2.122. Accepted steps increased by
8.54%; the saving comes from avoiding repeated failed proposals. These are two
actual executions with different cold-start costs, not a repeatability or
controller-only timing guarantee. The DG-call count excludes extra plane-root
calls, and the attempt timing excludes downstream column work and persistence.

| Disjoint active start-Z midpoint band | Old rejected | New rejected |
| --- | ---: | ---: |
| 0--0.5 mm | 260 | 12 |
| 0.5--1.1 mm | 4875 | 19 |
| 1.1--14 mm | 205 | 38 |
| 14--25 mm | 568 | 98 |
| 25--450.1 mm | 34 | 16 |

The last new rejected trial starts at Z=42.709782 mm. The 183 rejections occur
in 134 steps: 102 halve once, 18 twice, 11 three times and three four times.
All proposals respect their original freshly computed physical ceiling,
halving is exact, clocks are exactly continuous and no implicit/nonfinite
trial failure occurs. No initial trial grows immediately after a retried
step; other interstep growth is at most 1.25.

Largest accepted normalized gate ratios are 0.053813 for total momentum,
0.985363 for XYZ position, 0.997578 for transverse momentum and 0.478462 for
XY position. Each is below the unchanged acceptance threshold of one. Gun
exit work-relative energy error is at most 8.0793e-8 eV, below the original
0.001 eV budget. Column clocks remain monotone, and the physical loss ledger
closes exactly. The downstream fluorescent screen absorbs the full post
branch as designed; this is retained as physical loss rather than restored
current.

The independently parsed raw observer, comparison brief and static comparison
plot remain in `outputs/copper-gun-adaptive-20261114/adaptive-preview-high-1024/`.
Changed accepted meshes require a separate endpoint comparison; local gate
passes alone are not a particle-count or working-point convergence claim.

## Independent endpoint audit

The official typed Preview archive was independently read, with 63 saved-array
comparisons against the actual result data. Original evidence was read as
checksum-verified historical arrays; no old Gun was relabelled as a result of
the new solver. The normal Preview actual field snapshot and mechanical
geometry match the reference exactly, so this particular old/new comparison
does not mix automatic Preview and High scan-field settings.

| Maximum absolute old/new difference | At Gun exit | At specimen |
| --- | ---: | ---: |
| X | 0.262289 nm | 0.121222 nm |
| Y | 0.199132 nm | 0.112459 nm |
| Arrival clock | 0.566544 fs | 0.566544 fs |

Gun exit maximum slope differences are 0.769106 nrad in X and 1.063164 nrad in
Y; maximum exit energy difference is 2.64263e-8 eV. The specimen's weighted
centred RMS radius changes from 160.559619 to 160.571093 nm (0.00715%).
Per-ray specimen RMS differences are 0.020974 nm in X and 0.020888 nm in Y.
These are measured differences between two accepted temporal meshes, not a
new global-error bound. IDs, weights, survival flags and physical loss events
agree. No focus, atomic probe, source-count convergence or material-imaging
qualification is inferred from this optical-reference validation.

## Completed High accuracy workflow

High accuracy also completed the full requested ray workflow and its official
numeric archive readback. Worker-to-result time is 2629.547 s (43.826 minutes);
the invocation including result persistence is 2703.879 s. The original
quality policy applies a 0.05 mm maximum column step, with 23013 incident and
27809 post integration nodes. Gun integration retains its separate original
limits. This mode reuses the identical completed Gun object from the new 6169
Preview; it does not reuse the old 50af transport or rerun/modify the source.

All 1024 original positive-current particles survive to the specimen, with
effective population 1024, absolute current 10 microamperes and weight sum one.
The original source bytes and independent instrument controls remain unchanged.
Energy and flight-time verification passes, both physical loss ledgers close
exactly and the final post current is physically intercepted at the same
fluorescent screen. No cancellation, worker error or persistence error occurred.

Normal Preview and High accuracy resolve their automatic AC/Descan calibration
differently. Their independent hardware inputs are checked against the frozen
reference, and each actual drive matrix/runtime is retained. A normal-mode
comparison is therefore not used to claim fixed-field spatial-step convergence.
Both are complete classical optical-reference runs through the actual worker;
no material scattering, coherent wave, Si scan or native-GUI interaction is
claimed by this validation.

Independent High review verifies the full archive/file checksums, 63 native
disk-to-typed-archive arrays and 54 Gun/source/history/exit/event/common-clock
arrays against the new Preview. All 54 arrays and all Gun metadata are
byte-identical, with no new Gun observer trace or attempt. The specimen
checkpoint, incident endpoint and post entry agree exactly in position,
direction, energy and time. The independent complete input graph and hardware
controls agree before and after High execution. This separately verifies
legitimate current-solver reuse rather than relying only on a status flag.

## Fixed-actual-field step comparison

A separate completed Preview control retains the High run's exact native
static providers, dynamic drive laws, gauge and physical boundaries. It reuses
only the identical new Gun, then newly executes the entire 0.5 mm coarse
incident and post column, rather than resuming a fine High incident mesh.
The actual operator witness matches High exactly, with 2578 incident and
2839 post nodes. Worker time is 215.464 s; the 900 s cooperative limit did
not trigger. No source, Gun, field solve or calibration chief ray was run.
The complete 387418038-byte native archive passed 26 incident/post-array
roundtrips. Fifteen original launch fields and nine Gun exit fields are
byte-identical to High, with the same 1024 original IDs and absolute current.

| Specimen checkpoint difference, coarse versus fine | Maximum absolute | RMS |
| --- | ---: | ---: |
| X | 1.360585e-15 m | 3.746175e-16 m |
| Y | 1.370354e-15 m | 3.759756e-16 m |
| X slope | 1.002447e-14 rad | 2.732915e-15 rad |
| Y slope | 9.954323e-15 rad | 2.742171e-15 rad |
| Arrival clock | 3.042370e-21 s | 1.143774e-21 s |
| Kinetic energy | 0 eV | 0 eV |

Source weights, specimen survival/stop flags and post-column screen
interception agree exactly. These differences use actual corresponding
native checkpoints without recentering, refocusing or fine-trajectory
interpolation. They measure sensitivity to these two column steps under the
same field; they do not establish source-count convergence, a global error
bound or agreement with a measured magnetic field.

## Evidence and reproduction

[The curated manifest](validation/manifest.json) records checksums for copied
compact evidence, the three complete native result archives and larger raw
local records. [The comparison plot](validation/preview-rejection-comparison.png)
shows disjoint start-Z bands and recorded trial times. It is based on complete
raw observers with no dropped attempt records. The input/control, endpoint,
High-reuse and fixed-field audits are independently retained alongside the
final 304-test report and the 13 pure fixed-field preparation contract tests.

The reference source archive contains exactly the 781 captured 50af files,
with cache files excluded and each source checksum verified. The original
input JSON is stored as lossless gzip with its uncompressed checksum; the
source NPZ and original input manifest are copied unchanged. Script `.txt`
files preserve executed source bytes. Restore them to their original paths
listed by the manifest, and restore the input files under `baseline-input`,
before reproducing. Keep all historical outputs and use a new output folder:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.venv/Scripts/python.exe -u outputs/copper-gun-adaptive-20261114/run_particle_validation.py --expected-implementation 6169010971ca2aeb082c8f0936b043f2ad4eb886cdfd07991c69bdda32a7ffca --output outputs/copper-gun-adaptive-20261114/reproduction-new --quality both --execute
```

This requests 1024 original positive-current particles and the complete
classical workflows. It rejects a different source implementation. Old native
results remain historical evidence; they are never retagged for current-solver
reuse. New coherent work remains subject to the separate usable-CUDA policy.
