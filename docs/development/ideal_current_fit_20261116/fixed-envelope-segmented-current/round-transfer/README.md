# Bounded fixed-envelope current-only linear-transfer search

**Result: failed to match the reference; do not install this candidate.**

The search preserves all 13 original round-lens winding envelopes, poses,
integer physical turns and conductor-metal-area records. It uses four internal
axial cells in C1 lens and two in each original interval of every other round
lens: 30 independent signed currents in total. Every cell has a positive integer
turn allocation; its cross-section fraction equals its turn fraction. No
virtual turns, extra winding envelope, resistance, constant-power constraint,
magnetic core or pole piece is introduced.

This is a source-free, axisymmetric, constant-300-keV linear-operator study.
It emits no particles and runs no wave calculation. The physical source energy
distribution, project tolerances, production defaults and source caches are
not changed. A matrix fit would still require actual finite-source validation
before any focus, D95, transmission or crossover claim.

## Bounded experiment and outcome

Eight explicit signed-current seeds were each limited to 500 evaluations on a
0.2 mm matrix-integration mesh. The best seed received 500 evaluations on a
0.05 mm mesh, followed by a separate 0.025 mm verification. The objective
includes three standalone radial transfer matrices, signed unwrapped Larmor
rotation, and four endpoint-field comparisons. Canonical rotation and both
endpoint magnetic-field conversions to mechanical slopes are retained. The
current bounds and objective weights are experiment settings, not changes to
project numerical tolerances or a thermal model.

- Final objective norm: 1.22089786354 at 0.05 mm and 1.22089786358 at 0.025 mm.
- Maximum absolute cell current: 2442.67475 A.
- Gun-exit to corrector-entrance radial residual norm: 1.07032303.
- Corrector-entrance to corrector-exit radial residual norm: 0.00441140.
- Corrector-exit to specimen-upper-surface radial residual norm: 0.43659901.
- Specimen field: 0.44432474 T; archived reference: 1.11383501 T.
- Last-interval unwrapped fundamental phase: 18.04248877 rad versus reference
  6.37806783 rad. The extra focusing branch is another reason not to install.

The normalized radial-matrix mesh differences are below 1.41e-8; accurate
integration of this failed candidate does not make it agree with the reference.
On the separate finite-domain axial-field check, q12 to q24 source quadrature
differs by 2.09e-9 in relative L2, whereas the Gaussian-field error is 5.1248.
The raw residual Jacobian condition is 1.37e9, including near-redundancies from
the symplectic matrix constraints. It does not establish a unique inverse.

This bounded local search is not a mathematical proof that all possible
fixed-envelope current distributions are impossible. It establishes that the
executed candidates have not restored the archived working point. Opposite
currents reverse the signed field but never reverse the positive B-squared
focusing term. No physical source or scanning calculation is justified by
this candidate, and none was run.

## Evidence and portable reproduction

`final-conclusion.json` contains the original winding identities and their hash,
all final physical cells, parser-ready signed-current recipes, current bounds,
optimizer termination, matrix residuals and mesh comparisons. Its recipe is
marked failed and is evidence only. `metrics.json` is a compact summary;
`recipes.json` preserves the same evidence recipes with an explicit failed
status. `executed-checksums.json` preserves the original complete output manifest. The top-level `checksums.json` covers the curated files; only source-root discovery is changed in the portable script copies.

`design_transfer.py`, `design_transfer_v2.py` and `finalize_v2.py` reproduce the
executed first search, bounded signed-seed search and final verification. They
read only the frozen `captured-field-coefficients.json` and
`candidate-currents.csv` from `round-lens-field-match`, plus the pure field
helpers in `../round-study/fit.py`. They do not load the particle boundary or
large transport caches. To relocate them into tracked development evidence,
set their reference paths to copied frozen input JSON/CSV and retain the native
project linear-operator import. The captured JSON contains both Gaussian terms
and native annular-coil geometry/current metadata, so no source-cache restore
is needed to reproduce these field and matrix operators.

The earlier `candidate.json` and the intermediate `candidate-v2.json` remain
failed/intermediate search records. Neither is an approved operating point.
