# Corrector crossover refinement and finite-waist interpretation

Date: 2026-10-28. This report reads immutable saved trajectories. It performs
no new propagation, changes no field or source, and installs no preset. The
separate [continuation record](FINITE_FIELD_DIRECT_FOCUS_CONTINUATION_2026-10-28.md)
owns production probe-size results and subsequent optimization stages.

## Physical acceptance and what a waist means

Retain the user's Normal entrance: the required C2-C3 crossover, an active
weak C3, and no independent C3-ADL crossover. The finite C3 field and following
fields jointly form the next source image. Require one resolved source-waist
region between TL22/TL21 and one inside the AC pair. Count other source-waist
regions separately. Mini Condenser plus Upper Objective must form the final
focus directly at the specimen upper surface, without an additional upstream
source crossover. The geometric midpoint 1208.15 mm is not a measured optical
symmetry plane or an OEM position requirement.

The selected finite source need not have exactly simultaneous zero derivatives
in every transverse direction. A nonzero directional focus separation is not
automatically two source crossovers or a topology failure. Conversely, a radial
minimum or a coarse three-plane minimum cannot conceal an unresolved line focus.
Report the two-dimensional covariance, directional separation, finite waist
depth, actual particle population, and numerical resolution together. No new
roundness, exact common-zero, or dimensionless depth-ratio limit is imposed here.
The sub-nanometre specimen-size and surface-focus checks remain separate and
more restrictive than identifying a broad intermediate source-waist region.

## Evidence and reproducibility

The audit scripts and JSON results are under
`outputs/finite-field-corrector-rematch-20261028/`:

- `audit_two_dimensional_waists_v1.py` and
  `two-dimensional-waist-audit-v1.json`: full two-dimensional moments and
  executed three-plane brackets for `central-angle-matched-v5`, the stage-1
  proposal, and `first-focus-hp-optimized-v1` production.
- `audit_waist_depth_comparison_v2.py` and `waist-depth-comparison-v2.json`:
  finite-depth interpretation and the old Normal comparison.
- `extra-waist-planes-v1.json` and its `.manifest.json`: 2,705 requested new
  observation planes, not executed results.

The source identity is
`4e3bac1af215cfe7b8589e64363541ba2039654fb01be49e8a5abee3017c0815`.
The unchanged production implementation identity is
`f34f2ff6f73532f29faa3652fd88dd902625bd248d2196e5d94c3f0b08c5cd21`.
The JSON files record observation hashes, original IDs and weights.

All 193 original particles survive every saved plane in v5 and stage-1
production. Their actual current remains 10 microampere. The stage-1 fast
proposal has zero diagnostic current and no mechanical interception model;
its complete phase array is nevertheless bitwise equal to the independently
completed stage-1 production array for this particular setting. This comparison
does not qualify later proposals or remove their production-stop checks.

The old comparison is
`outputs/finite-field-joint-preset-20261028/legacy-normal-complete193/`.
It replays historical `custom` controls with the current implementation and
the same original 193 source columns. Nineteen particles are physically stopped
upstream; the stable central/AC population is 174 and 9.015544 microampere.
The table therefore compares each setting's actual local population, not an
artificially identical survivor subset. Original weights are never rescaled
physically; normalization is only for reporting conditional moments.

## Two-dimensional diagnostic

Let R be the centered position covariance, C the symmetric position-slope
covariance, and T the slope covariance, using the same positive-current IDs
throughout each local interval. The exact local kinematic identity is
`dR/dz = 2 C`. In a straight-drift diagnostic,
`R(d) = R + 2 d C + d^2 T`.

The eigenvalues of `-T^(-1/2) C T^(-1/2)` give the extrema of the stationary
distances over all fixed transverse directions. This is more informative than
testing only the radial trace or principal directions of R at one plane. It
is a local moment extrapolation, not an executed finite-field continuation.

At the local radial minimum, define the reporting scale
`depth = RMS radius / RMS slope`. It is the straight-drift distance over which
the radial variance increases by its minimum value. It is neither a quantum
depth of focus nor an accepted tolerance. The directional depth ranges are
also saved in JSON. The reported intermediate minima below are local drift
diagnostics from executed phase; they are not extra observed checkpoints.

| Case / region | Radial minimum Z, mm | RMS radius, nm | RMS angle, mrad | Depth, mm | Directional split, micrometre | Split / depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Old Normal / central | 1202.046073 | 2029.277 | 3.746697 | 0.541618 | 61.875 | 0.11424 |
| v5 / central | 1207.955592 | 12703.884 | 2.159321 | 5.883278 | 777.129 | 0.13209 |
| Stage 1 / central | 1207.847231 | 15181.011 | 2.159902 | 7.028566 | 931.996 | 0.13260 |
| Old Normal / AC | 1400.000604 | 2.182 | 1.873304 | 0.001165 | 0.1084 | 0.09306 |
| v5 / AC | 1400.278088 | 301.140 | 1.687743 | 0.178427 | 17.257 | 0.09672 |
| Stage 1 / AC | 1400.417734 | 450.010 | 1.691139 | 0.266099 | 25.026 | 0.09405 |

The relative directional separation is comparable to the old Normal reference.
The much larger absolute intermediate radii and depths are a distinct finding;
they must not be confused with evidence for two independent source crossovers.
The source populations and field models differ as stated above, so this table
does not isolate a single cause or prove that a particular aberration dominates.

For stage 1, the local directional stationary planes span
1207.380387-1208.312383 mm centrally and 1400.404780-1400.429805 mm at AC.
The eigenvalue sign changes in the saved executed planes give compatible
interpolated limits. Those intervals motivate refinement across the entire
waist region rather than sampling only the radial minimum.

## Existing sampling does not settle every acceptance check

The production captures have 3,696 checkpoints and a maximum gap of
0.499870 mm. Existing strict three-plane covariance-growth detection yields:

| Region | v5 | Stage-1 production | Interpretation |
| --- | --- | --- | --- |
| C2-C3 | One bounded-sampling 2D bracket | Same bracket | Required Normal waist is observed; local refinement can improve its position. |
| C3-ADL | No radial or directional sign change | Same | No separate waist detected at current sampling; C3 remains active. |
| Central TL pair | No accepted strict bracket; one radial candidate | Same | Dense 2D characterization is required; a strict point test can miss a finite broad waist. |
| AC pair | No accepted strict bracket; one radial candidate | One strict bracket at 1400.4 mm | The different count is sampling dependent; neither count alone resolves directional structure. |
| Mini body to specimen | One coarse bracket near the final focus | No extra 2D bracket | Distinguish final defocus from an additional upstream optical branch. |

For v5, the detector labels the sample-minus-200-nm checkpoint as a minimum
because its left neighbor is much farther upstream. That label does not place
the true waist 200 nm before the surface. The surface itself is diverging in
both directions; its local stationary offsets are about -1415.4 and -1330.6 nm.
It is an incorrectly placed final focus, not evidence that the old separate
1585.7 mm crossover has returned.

Stage 1 has a radial surface-focus offset of about +0.0686 nm, but directional
offsets span -7.6674 to +7.8349 nm. A nearly zero trace does not establish
common surface focus. The surface is an endpoint: upstream measurements alone
cannot give an executed two-sided bracket. No downstream point through or
past the specimen is silently substituted for incident vacuum optics.

## Requested executed refinement

The supplied extra-plane array preserves the existing full-column observations
and adds targeted measurements:

- Central: 1207.0-1208.6 mm at 0.005 mm, with 0.001 mm neighborhoods around
  the two stage-1 directional limits.
- AC: 1399.9-1400.7 mm at 0.002 mm, with 0.00025 mm neighborhoods around
  the stage-1 directional limits.
- C3-ADL: 0.25 mm survey spacing; Mini body entrance to specimen: 0.1 mm
  survey spacing, plus the final-focus planes.
- C2-C3: a local 0.002 mm region and a narrow 5 nm-spaced root neighborhood.
  Coarse interpolation of ordered covariance eigenvalues can exaggerate a
  very small directional separation, so this region is measured directly.
- Specimen: upstream offsets 200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1
  and 0 nm, plus wider offsets through 10,000 nm to cover the prior v5 focus.

All planes are at or before 1599.1999975 mm. These are observation requests,
not a hidden extension into the sample and not physics results. A new proposal
may move the waist outside a focused window; inspect the full survey and adapt
the windows before claiming resolution. At a population transition, analyze
stable local cohorts rather than final survivors only.

Use finer integration separately from denser observations; adding checkpoints
alone is not a 0.025 mm half-step convergence result. Report actual changes in
directional limits, sizes, angle, current and final focus when the parent task
executes those checks. Until then, the finite internal waist pattern is a
plausible one-region interpretation awaiting refinement, and no candidate in
this audit is qualified against D95 below 0.7 nm.

## Completed dense stage-3 observation

The subsequent production case `first-focus-coupled-refined-v3` requested
6,217 planes and saved 6,335 executed checkpoints, including instrument
boundaries. All 193 original particles survive every plane; the actual current
is 10 microampere. The maximum integration step is still 0.05 mm. This is a
denser observation, not a 0.025 mm half-step or source-count convergence check.
Its input SHA256, `09fa92864515586bb7489a7a1bff7bf9987cdeb7f8851a49eaf59c499dc70eb4`,
matches the frozen best candidate from `fast-third-stage-v1`.

The saved-only analysis is `audit_refined_coupled_v3.py` and
`refined-coupled-waist-audit-v3.json` in the same output directory. The script
loaded the then-unapplied direct-focus policy from the independently tested
ordinary copy under `direct-focus-gate-validation-v2/`; that audit did not
modify production source or relabel the original physical calculation's
implementation receipt.
Set `TEMSIM_PROJECT_ROOT` to the repository when reproducing that read-only
analysis so the isolated code reads the original configuration files.

### Agreement with the coarse stage-3 proposal

All 3,696 coarse observation planes occur exactly in the dense capture. Their
original source IDs, weights and initial phase arrays match. The largest
absolute differences over those common planes and all particles are
`1.399e-16 m` in X, `1.335e-16 m` in Y, `2.695e-14 rad` in X slope and
`2.839e-14 rad` in Y slope. These tiny differences are consistent with the
extra checkpoints changing the integration partition; their cause has not
been isolated by an independent integrator experiment.

| Specimen observable | Coarse zero-current proposal | Dense actual production | Difference |
| --- | ---: | ---: | ---: |
| alpha95, mrad | 29.988771329704647 | 29.988771329705315 | +6.68e-13 |
| D95, nm | 5.133828223389049 | 5.133827962044907 | -2.61e-7 |
| RMS radius, nm | 1.392492362514604 | 1.392492365203105 | +2.69e-9 |
| Radial covariance focus offset, nm | -3.572011995 | -3.572016108 | -4.11e-6 |

This validates agreement of these two saved calculations at their shared
planes and this working point. It does not give future proposals a physical
interception exemption or establish source/integration convergence.

### What the added planes resolve

The following root coordinates remain interpolated moment diagnostics inside
the listed **executed** brackets. Their extra displayed digits are useful for
comparison, not a claim of that many digits of physical accuracy.

| Region | Radial stationary Z, mm | Executed radial bracket width | Directional separation | Finite depth | Separation / depth |
| --- | ---: | ---: | ---: | ---: | ---: |
| C2-C3 | 687.721486190 | 5 nm | 4.994 nm | 85.527 nm | 0.05839 |
| Central TL pair | 1207.849396427 | 5 micrometre | 929.992 micrometre | 7.015450 mm | 0.13256 |
| AC pair | 1400.495480916 | 2 micrometre | 25.036 micrometre | 0.266520 mm | 0.09394 |
| Final near-surface focus | 1599.199993928 | 3 nm | 0.281 nm | 85.510 nm | 0.00328 |

The central directional transitions are individually bracketed within
1207.3834-1207.3844 and 1208.3134-1208.3144 mm. AC transitions occur within
1400.482-1400.484 and 1400.506-1400.508 mm. The central and AC depth ratios
remain close to the old Normal values, 0.11424 and 0.09306 respectively.
There is one radial source-image region in each required interval, with
finite directional separation inside it. These data do not justify counting
each directional transition as an independent source crossover.

The denser observations remove much of the earlier axial-location uncertainty
and show that directional splitting persists when it is directly bracketed.
They do **not** make the existing strict three-adjacent-plane point counter
sampling invariant: its C2-C3 and AC counts change from one to zero, while the
central count remains zero. No lens or source changed between those two saved
stage-3 calculations. A neighboring checkpoint can lie closer to one direction's
minimum even though both calculations describe the same finite waist region.
Consequently the strict counter still reports unresolved topology. Neither
reusing the coarse PASS nor declaring that the physical crossover disappeared
is justified. A future finite-region acceptance rule must retain the measured
directional separation and numerical resolution; this audit introduces no
new hard split threshold.

The C3-ADL survey has no radial or directional converging-to-diverging
transition at its 0.25 mm bounded spacing. No distinct old 1585.7 mm source
waist appears between Mini Condenser and specimen. The only new terminal
candidate is the final focus approximately 3.572 nm before the surface.

### Direct-focus verdict and remaining rejection

The near-surface radial transition is bracketed at 5 and 2 nm before the
specimen. Both points lie outside the declared 1 nm terminal-focus window.
The isolated direct-focus gate therefore returns **UNRESOLVED, passed=false**,
retains that candidate, and proposes additional executed planes at
1599.199993553, 1599.199993928 and 1599.199994303 mm. Those planes were not
executed by this audit. The gate does not use the optimizer's 500 nm screening
window as an acceptance allowance.

The specimen is already diverging in both transverse directions. Local
directional stationary offsets are approximately -3.7120 and -3.4312 nm,
so a small directional separation does not cure the common defocus. The
surface-focus criterion is missed, and actual D95 is 5.133828 nm, well above
the strict 0.7 nm limit. This candidate remains unqualified independently of
the intermediate finite-waist interpretation. The near-surface defocus is
reported separately from a distinct extra upstream optical branch.

## Isolated finite-region observer validation

A separate opt-in observer was subsequently developed and tested in an
ordinary source copy, without modifying the then-frozen production implementation
or the existing three-adjacent-plane counter. Its proposal, implementation
contract, isolated validation receipt and new saved-array replay are under
`outputs/finite-field-corrector-rematch-20261028/finite-waist-region-proposal-v2/`.
This is independent of the direct-focus/D95 gate patch.

The observer searches actual sides around one isolated radial minimum until
`Csym` is negative definite on the left and positive definite on the right,
and the full position covariance exceeds the same executed representative
plane in every direction on both sides. It records the finite region span,
directional transitions and explicit non-adjacent side indices. It does not
claim simultaneous zero derivatives at one point. Search cannot cross another
radial trace reversal, component bounds or interception. Multiple directional
refocus events inside the proposed bracket remain unresolved; earlier
directional envelope maxima outside it are recorded separately.

Thirty new synthetic tests and nine unchanged historical counter tests pass
in the isolated copy. The tests cover rotation and sampling refinement,
fixed-width line-focus rejection, two separate waists, repeated directional
transitions, separate earlier directional maxima, stationary plateaus,
explicit brackets, source identities/current denominators and missing data.

The new saved-only replay gives C2-C3 = 1, central = 1, AC = 1 and C3-ADL = 0
for both coarse and dense stage-3 data. Legacy Normal also has one central
and one AC region; its C2-C3 interval remains unresolved because interception
changes the observed cohort there. No eventual-survivor substitution is made.
Dense central covariance-growth sides are 1206.802434195-1208.801912225 mm,
containing directional transitions within 1207.3834-1208.3144 mm. Dense AC
sides are 1400.468-1400.520 mm, containing transitions within
1400.482-1400.508 mm. These are finite source-image regions, not OEM absolute
Z requirements or proof that the geometric center 1208.15 mm is optical
symmetry.

The dense Mini Condenser body-to-specimen interval contains one final region
whose measured sides are 10 and 1 nm before the sample. An earlier directional
envelope maximum near 1531 mm is not a second waist. The radial focus remains
3.572 nm upstream and D95 remains 5.134 nm: resolving the finite-region count
does not qualify the candidate or override the independent surface-focus
gate. No historical receipt or strict-counter result above is rewritten.

At that development stage, the observer remained unapplied and explicitly
opt in. Consumers assuming adjacent side planes still needed adaptation;
the subsequent integration below supplied it. Adding the module changes the
solver source identity and cache keys even though it only observes statistics,
so the physical calculations remained frozen until they finished.

### Qualification consumer integration

A subsequent independent consumer patch connects this observer to the new
Unified Nanoprobe qualification path; it is no longer only an unused helper.
The three-layer application order, exact base/final file hashes, tests and
saved replay are recorded under
`outputs/finite-field-corrector-rematch-20261028/finite-region-consumer-proposal-v1/`.
The layers were initially applied to a fresh ordinary isolated copy while
production source remained frozen. New state-selected condenser/GUI validation
also requests central and AC observations; generic corrector and whole-chain
APIs select the new detector explicitly and label their new report schemas.
Old point results remain diagnostic, and all finite consumers use explicit
bracket sides and all observations inside them.

The overlaid copy passes 175 pure tests and 13 selected mocked production/joint
entry checks. Known aperture loss away from a waist does not become a
100-percent-transmission requirement: actual stable segments retain original
IDs, weights and the complete-source current denominator. Loss attribution
uses captured masks and opening geometry, not a plane coincidence alone.
Full-inventory status remains separate from required-region evidence, and
unattributed interception is explicitly rejected by the new CLI qualification.

The authoritative saved consumer replay is `saved-consumer-replay-v2.json`.
It uses the original 0.5 mm maximum spacing for both condenser and whole-chain
evaluation. All condenser guard planes, 649.75, 650, 915, 1088.2 and
1088.45 mm, exist in both captures. The preceding v1 replay accidentally used
1.0 mm for the condenser and is superseded; its missing-guard diagnosis must
not be used. At the original spacing both captures pass corrector finite-region
qualification and the required C2-C3/central/AC count checks. The remaining
direct-focus, sample defocus and D95 failures are preserved, so neither
candidate becomes qualified merely through the detector correction.

### Reviewed consumer v2 and the later focused candidate

The first consumer patch is superseded by
`finite-region-consumer-proposal-v2/finite-region-consumers-v2.unapplied.patch`.
Independent review found that a known interception across a forbidden C3-ADL
radial bracket could disappear when the interval was split into stable
segments. The revised consumer retains complete-capture adjacent-cohort
brackets as negative evidence at prohibited intervals and component boundaries.
They cannot be used to manufacture a required positive waist across clipping.
The same review corrected real source-ID forwarding into direct-focus reports
and explicit identity validation before terminal endpoint handling. Targeted
independent tests of these corrections pass.

The authoritative consumer replay is now
`finite-region-consumer-proposal-v2/saved-consumer-replay-v3.json`, retaining
the 0.5 mm contract and including the actual
`first-focus-auxiliary-refined-v4` capture. That case passes the new condenser,
central/AC and required-region whole-chain gates, with no extra direct-focus
branch. Its sample focus and alpha95 pass independently. The only remaining
qualification failure is the strict size limit: actual D95 is approximately
2.435050 nm, above 0.7 nm. The previous dense stage-3 case remains rejected
for its approximately -3.572 nm focus offset and approximately 5.134 nm D95.
These are new saved-array observations; the physical source identity, original
193 particles/weights and original calculation receipts are not relabeled.

The later completed 0.025 mm integration case,
`first-focus-auxiliary-half-step-v1`, was also replayed with the final applied
consumer tree in `saved-consumer-step-comparison-v4.json`. Both it and the
0.05 mm case pass the condenser, corrector, direct-focus and required-region
chain gates at the same 0.5 mm observation spacing. In both, the only remaining
qualification failure is D95. The original physical implementation is still
`f34f2ff6...`; the new observer implementation is recorded separately as
`7dbc9aab...`. This comparison introduces no new propagation and does not
relabel either physical result.

### Final completed candidate observation

The final actual production case `first-focus-final-refined-v5` was replayed
without new propagation or any parameter change. Its new report is
`finite-region-consumer-proposal-v2/saved-final-candidate-replay-v5.json`.
The report preserves the original physical implementation `f34f2ff6...`, the
original observation/report file hashes, all 193 source IDs and weights, and
the separate qualification observer implementation `7dbc9aab...`.

| Final requirement | Saved-array result |
| --- | --- |
| Condenser topology, central/AC finite regions and required chain | Pass |
| Direct focus without an extra pre-specimen waist | Pass |
| Surface focus | Pass; local variance offset +0.000517 nm |
| alpha95 | Pass; 29.999865624 mrad |
| Source support/current | 193 of 193; complete-source current fraction 1 |
| Strict D95 below 0.7 nm | Fail; D95 2.199562064 nm |

The independent covariance focus offset is approximately +0.000517 nm as
well. The only new qualification failure is the strict D95 size criterion;
the final candidate remains unqualified. This observation establishes the
required finite-source topology at the declared sampling, not an exact
common stationary point, an OEM absolute-Z match, or a qualified scanning
image preset. No failed physical candidate was installed as a qualified
default by this audit.

### Production integration

After all frozen physical calculations finished, the parent task applied the
three verified patches to the actual workspace in order: direct-focus gate
v3, finite-region observer v2, and finite-region consumers v2. Exact byte
guards passed for all 12 affected files before and after application. The
parent then registered five new tests in the existing acceptance scopes,
including the four optical qualification tests and the separately developed
scan-background test. No optical default or physical candidate was promoted
by that registration.

The source identities now have three distinct meanings:

- `f34f2ff6f73532f29faa3652fd88dd902625bd248d2196e5d94c3f0b08c5cd21`:
  the frozen implementation that produced the saved physical trajectories.
- `7dbc9aab0bd4ee5e23c6ea3113754ab408f6992731fc661efdb382b9e2a28ec8`:
  the fully applied isolated observer used for the final saved-array replays.
- `1c8a91c1cf5007de6035becd106149300cbd97114ced39fcb6d4e306645644c3`:
  the integrated workspace immediately after acceptance-test registration.

The last identity differs because the implementation hash also includes the
acceptance registry. Historical physical receipts retain their original
identity. The new qualification paths are integrated into the project; the
earlier unapplied/copy-only statements above describe their development
stages. Final workspace regression outcomes and the push receipt are recorded
in the parent continuation report, rather than asserted from the older
isolated test runs.
