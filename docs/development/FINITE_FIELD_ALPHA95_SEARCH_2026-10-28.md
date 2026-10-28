# Finite-field Nanoprobe Normal alpha95 search, 2026-10-28

## Target and status

The selected target is **alpha95 = 30 mrad** and **spatial D95 < 0.7 nm** at
the specimen upper surface. Both statistics refer to the original current
weights that physically arrive at that plane. There is no transmitted-current
floor or additional RMS-radius gate. The physical pupil edge is a separate
observable, not the selected 30 mrad target.

C2/C3 are the primary convergence-angle controls. Mini Condenser and Upper
Objective compensate specimen focus. Corrector strengths, orientation and
entrance/output matching address aberrations. Coupling between these controls
is measured rather than assumed absent.

**The new 30 mrad preset is not qualified or installed.** The existing default
is unchanged. A small probe at approximately 10 mrad does not satisfy this
target. A local fit or a historical source subset does not qualify a new
operating point.

## Executed evidence

All observations below use the current production classical-particle solver,
193 complete source particles, original weights and ancestry, a 0.05 mm column
integration step unless explicitly labelled otherwise, and the specimen upper
surface at 1599.1999975 mm. Raster
and wobble are disabled on diagnostic copies. No coherent calculation or
new scanning image was executed in this search.

The complete gun source is restored only through the production section codec
and dependency checks. Column-prefix reuse is decided by production validation;
no manually selected downstream ray bundle is supplied as a new source.

| Actual observation | alpha95 (mrad) | D95 (nm) | RMS radius (nm) | Local radial waist offset (nm) | Specimen current (microampere) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing default, 193 particles | 9.819338 | 0.509216 | 0.171771 | +4.286412 | 10.000000 |
| High-angle restart | 29.315612 | 183.290986 | 34.558246 | -750.562326 | 9.015544 |
| Executed HP strength/orientation and Objective candidate | 29.327973 | 59.559740 | 15.374536 | -16.391093 | 9.015544 |
| Same HP candidate, half integration step (0.025 mm) | 29.327973 | 59.558536 | 15.374554 | -16.419553 | 9.015544 |
| Executed eleven-control joint candidate | 30.038653 | 313.049684 | 74.148574 | -3808.474345 | 9.015544 |

The emitted and gun-exit currents are both 10 microampere in these observations.
The high-angle observations retain 174 of the 193 source particles at the
specimen: 19 are physically intercepted at the C2 aperture, accounting for
0.984456 microampere. Transmission is 90.155440% relative to both emitted and
gun-exit current. These are finite-population measurements, not converged
current estimates. No stopped particle is restored or assigned extra weight.

The measured condenser Normal topology passes at the bounded observation
spacing. The complete crossover chain and the internal corrector waist remain
unresolved, so this is not whole-column acceptance. The half-step observation
changes D95 by only 0.001204 nm, with unchanged source/current and specimen
survivors. This tested step difference does not explain the approximately
59.56 nm width. It is one numerical comparison, not general convergence of
every prospective working point. Source-count refinement, resolved corrector
conjugacies and driven scan validation remain prerequisites for a qualified
default.

## Controlled differences

Each following observation independently adds 0.001 percentage point to one
lens at the same high-angle restart. All retain the same 174 specimen arrivals.
These derivatives belong to that restart, not to the later corrected candidate.

| Control | Measured alpha95 (mrad) | Measured D95 (nm) | Local alpha95 derivative (mrad / percentage point) |
| --- | ---: | ---: | ---: |
| C2 | 29.318001 | 183.381800 | +2.388480 |
| C3 | 29.315184 | 183.379813 | -0.428250 |
| TL22 | 29.317215 | 185.209659 | +1.602650 |
| TL21 | 29.314413 | 190.578435 | -1.198974 |
| ADL | 29.315602 | 183.312832 | -0.010087 |
| TL12 | 29.316446 | 183.807063 | +0.833943 |
| Mini Condenser magnitude | 29.314137 | 188.677204 | -1.475423 |

The optical roles are not assigned by whichever lens has the largest numerical
derivative. The C2/C3 pair remains responsible for the convergence target;
changes elsewhere require rechecking the coupled angle.

## Residual diagnosis and model limits

For the executed 59.56 nm candidate, a purely diagnostic best scalar defocus
removal changes RMS radius from 15.3745 to only 15.3717 nm. This local
scalar-defocus diagnostic explains little of the remaining width; it is not
a new Objective-field execution or evidence about all possible nonlinear
refocusing changes.

An empirical polynomial fit of transverse position against the actual
specimen angular coordinates leaves approximately 14.55 nm RMS after terms
through degree three, 6.81 nm through degree four, and 0.33 nm through degree
five. A source-ID holdout check leaves approximately 0.39 nm for the fifth-degree
fit. This identifies substantial higher-order structure in this measured
population. It is **not** a physical correction, a new ray distribution, or
proof that a feasible set of lens strengths can realize the fitted map.

The local TL22/TL21 perturbations predominantly change lower-order structure
and focus. Larger six-control affine extrapolations can predict improvement,
but none establishes the selected size target while preserving the angle.
An unconstrained least-squares example predicts D95 about 7.74 nm only while
reducing alpha95 to about 24.58 mrad and leaving the stated trust bounds; it is
rejected as a candidate. No prediction is promoted to measured evidence.

Adding measured C2/C3, ADL/TL12, and Mini Condenser responses produced an
eleven-control affine candidate with predicted alpha95 = 30 mrad and D95
about 41.59 nm. C3 is the outer matching control, C2 explicitly solves the
full-model angle target, and Upper Objective solves radial focus. Mini
Condenser supplies the measured joint-focus/output-matching direction. The
other controls optimize the spatial shape. Their angular coupling is reported
without imposing an extra current floor or a requirement that coupling vanish.
The predicted directional waist offsets remain approximately +62.88 and
-63.69 nm; radial focus cancellation alone is not common two-dimensional
focus. Independent full execution returned alpha95 = 30.038653 mrad, D95 =
313.049684 nm and a radial waist offset of -3808.474345 nm, with the same
174 arrivals and original current. The prediction therefore failed badly
for this large combined step, especially in focus. The point is rejected;
it neither replaces the better measured candidate nor changes the default.
This failure is evidence against extrapolating that local Jacobian over the
chosen trust box, not proof that all strength combinations are infeasible.

## Local provenance

Local raw reports, complete input states, original source arrays, checkpoints,
and versioned diagnostic scripts are under
`outputs/finite-field-joint-preset-20261028/` (ignored execution artifacts).
The main executed folders are `baseline193`, `high-angle-restart`,
`high-angle-corrector-candidate`, `high-c2-plus`, `high-c3-plus`,
`tl22-plus-v2`, `tl21-plus-v2`, `high-adl-plus`, `high-tl12-plus`,
`high-mini-plus`, `high-angle-corrector-half-step`, and
`high-angle-joint-candidate`.
Earlier `tl22-plus` and `tl21-plus` folders
record an input-allowlist failure before execution and are not measurements.

- Solver identity: `f34f2ff6f73532f29faa3652fd88dd902625bd248d2196e5d94c3f0b08c5cd21`
- Complete source identity: `4e3bac1af215cfe7b8589e64363541ba2039654fb01be49e8a5abee3017c0815`
- Gun dependency signature: `0dce9e48f36759d5b609d09197eb8fd8ec35d5d9f1c2d17f8dd61bb513f7162f`

See the [control contract](../OPTICAL_CONTROL_ROLES.md), the updated
[acceptance contract](FINITE_FIELD_OPTICAL_ACCEPTANCE_CONTRACT_2026-10-27.md),
the [corrector initialization and angular-transfer comparison](CORRECTOR_ANGLE_TRANSFER_COMPARISON_2026-10-28.md),
and the separate [software repair record](FINITE_FIELD_REVIEW_REPAIRS_2026-10-28.md).
