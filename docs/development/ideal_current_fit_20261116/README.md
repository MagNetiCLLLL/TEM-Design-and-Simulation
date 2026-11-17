# Ideal-current coil fitting and bounded validation

Study started on 2026-11-16 and closed on 2026-11-17.

## Outcome

The fixed-envelope current-only search **did not recover the archived Gaussian
working field or its linear transfer**. No fitted candidate was installed as a
default. This is a failed bounded search, not a proof that every possible
fixed-envelope current distribution is infeasible. No focused probe, BF/ADF
scan or diffraction image is claimed for these candidates.

The software now supports explicit internal forward/reverse current groups
inside ideal vacuum coils. The current, geometry and tolerance defaults remain
unchanged. No resistance, constant-power, thermal, magnetic-core or pole-piece
model is introduced. Other field modes preserve these electrical recipes as
dormant data and do not execute them.

## What was held fixed

The reference is the captured original geometry and controls of `mixed-run-v3`,
in `round-lens-field-match/captured-field-coefficients.json`. The winding inner
and outer radii, axial intervals, poses, physical turns and conductor-area
records are unchanged. Objective lens has the existing 580 + 580 modeled turns;
the earlier photograph-based 586-turn reference was not silently restored.
The earlier resized inverse designs are historical studies and are excluded
from this restricted baseline.

Every current group represents complete closed turns or a closed subwinding.
Different steady currents are never assigned to successive sides of one
unbranched wire. A cell's physical turn count is an integer; numerical
quadrature points are not additional turns. The homogenized cross-section
partition does not certify that each cell boundary corresponds to an existing
individual CAD wire. The CAD envelope is retained, and actual wire-addressed
wiring would need a separate turn-membership design.

No source law, 0.3 eV FWHM, energy boundary, particle weight, solver tolerance
or aperture boundary was retuned. A changed current distribution changes
all-Z magnetic tails, including the Gun. Previously transported particles
cannot serve as fresh transport evidence for such a candidate.

## Field and transfer results

The original uniform ideal-coil field had the same one-point Objective lens
calibration but a very different shape: the total round-lens integral of
`Bz^2` was 136.55 times the old Gaussian reference. One-point field agreement
therefore did not preserve focusing power or convergence angle.

Internal signed-current SVD fits retained the original envelopes and counts:

| Joint fit | Full-axis q24 relative L2 field error | Largest segment current | q12-to-q24 relative field difference |
| --- | ---: | ---: | ---: |
| SVD cutoff 1e-8 | 8.2591% | 1.1460e9 A | 2.35e-10 |
| SVD cutoff 1e-10 | 4.6027% | 9.2231e10 A | 1.92e-8 |
| SVD cutoff 1e-14 | 1.2007% | 8.1209e14 A | 4.02e-5 |

These are conditioning experiments, not relaxed project tolerances. Larger
opposing currents improve the fitted residual while amplifying quadrature and
floating-point cancellation. MP80 checks found a TL22 native binary64 axis-field
error of 1.74e-5 T for the aggressive 1e-14 per-owner solution. It was not admitted.
The joint fit also does not preserve each lens's independent control response.

A separate bounded search used 30 internal groups and moderate currents to
match the radial Larmor matrices and unwrapped rotation between Gun exit,
corrector entrance, corrector exit and specimen surface. Its maximum current
was 2442.67 A. The final objective residual was 1.22089786354 at a 0.05 mm mesh
and 1.22089786358 at 0.025 mm: the numerical mesh comparison is stable, but the
reference transfer is not matched. Specimen `Bz` is approximately 0.4443 T
versus 1.1138 T, and the final interval has an additional focusing branch
signal. See `fixed-envelope-segmented-current/round-transfer/final-conclusion.json`.
This constant-energy, axisymmetric linear operator excludes the actual Gun
acceleration, nonlinear corrector, finite source, apertures and aberrations;
it cannot qualify specimen focus or D95.

For HP1/HP2, two complete N1 groups within each original N2 radial winding,
using the implemented equal-area radial partition, gave approximately
26.7448% / 53.2909% longitudinal-hexapole-profile L2 errors. Exploratory
unequal-area partitions improved these to 26.23% / 52.87%, but those are not
admitted recipes: the implemented partition preserves integer turns and the
original turn density in every cell. Evaluating the unchanged actual CAD loops
separately gave 26.93% / 53.42%. Quadrature refinement was already small;
none of these explored two-current profiles reproduced the old narrow field.
The HP reports distinguish homogenized groups from actual CAD turn membership.

## Earlier actual source and mixed-scan limit

Before the restricted segmentation study, the recorded full source trial used
3000 original positive-current particles. All 3000 reached the specimen
entrance with 10 microampere total current. The specimen-surface D95 was
178.4799 nm, RMS radius 50.2378 nm and axis alpha95 0.179526 mrad. These are
results for the archived unsplit ideal-coil inputs, not the new candidates.
Its input snapshot, current accounting and backend receipts are preserved in
`mixed-run-v3/`; large particle caches remain local.

The covariance-matched Gaussian packet lift could not admit that executed
source under its existing coherence model. A new unit-balanced uncertainty
guard reports this limitation before ill-conditioned packet algebra. It does
not change successful lift formulas, tolerances, beam energies or source
covariance. This rejection is a limitation of this lift construction, not a
claim that the physical classical source cannot exist. No BF/ADF or diffraction
result was produced from this failed admission.

## Using the new electrical input

In Model Inspector, select the ideal prescribed-current vacuum-coil mode,
select a round lens, and open **Internal coil currents...**. Generate integer
turn groups and edit signed current ratios, or import a validated JSON recipe.
The displayed actual current includes the existing lens percentage and polarity.
The original master current remains the lens control; each internal group
uses `I_group = ratio * I_master`. Applying a recipe does not move or repack the
winding. Cancelling leaves inputs untouched. Uniform ratios of one use the
original field source exactly. Zero net signed NI does not mean zero field
when opposing groups occupy different parts of the original winding.

The Python entry point is `temsim.ideal_coil_segments.set_current_segments`.
HP1/HP2 use the separate `temsim.ideal_transverse_segments` recipe and setter;
their physical normal/skew channels are not repurposed as internal layers.
Recipes are included in saved inputs and source/backend/cache identities.
An electrical change invalidates incompatible runtimes and transported results.

## Reproduction and validation scope

Use the project virtual environment, from the repository root:

```powershell
.venv/Scripts/python.exe docs/development/ideal_current_fit_20261116/run_validation.py
.venv/Scripts/python.exe docs/development/ideal_current_fit_20261116/audit_evidence.py
.venv/Scripts/python.exe docs/development/ideal_current_fit_20261116/verify_packaging.py
.venv/Scripts/python.exe docs/development/ideal_current_fit_20261116/expand_evidence.py
.venv/Scripts/python.exe docs/development/ideal_current_fit_20261116/fixed-envelope-segmented-current/round-study/fit.py
.venv/Scripts/python.exe docs/development/ideal_current_fit_20261116/fixed-envelope-segmented-current/round-study/joint.py
.venv/Scripts/python.exe docs/development/ideal_current_fit_20261116/fixed-envelope-segmented-current/round-study/audit_precision.py
```

The first command is a bounded source-free test selection. Its emission/Gun
transport guard prevents accidental finite-source execution. CUDA checks use
actual device field operators on synthetic phase arrays; GUI checks run
offscreen. These are not new coherent propagation, finite-source convergence,
Preview/High accuracy full-column certification or scanning-image evidence.
The archived validation-exclusions ledger retains earlier accidentally selected
small-source unit cases and failed intermediate attempts; those are not used
as physical qualification. Final machine-readable receipts record the executed
selection and result without summing duplicate test runs.

The final executed selection passed **228 tests, with zero failures, errors or
skips** on the RTX 5090 system. The wheel was built offline and its installed
target imported successfully outside the checkout, including the bundled
eight-atom Si CIF. That packaging check uses an explicitly bound installed
prefix and the host virtual environment's dependencies; it is not a fresh
dependency installation. Source compilation and the Git whitespace check
also passed. See `validation/field-tests.json`, `validation/packaging.json`
and `validation/evidence-identity.json` for the bounded scope and identities.

The field scripts can be expensive and overwrite their generated study files.
Large JSON snapshots are stored losslessly as `.json.gz`; `expand_evidence.py`
restores their exact original bytes before inspection or dependent field scripts.
Their compact frozen geometry/field inputs are tracked, so reproducing the
field fit does not require a particle checkpoint. `executed-fit-v1.py` is
preserved unchanged; only repository-root discovery is made portable in
`fit.py`. Checksums distinguish these copies and the original evidence.
