# Canonical C3 + Probe Corrector diagnostics

This opt-in benchmark uses the actual 29 static magnetic-source definitions of
STEM Nanoprobe Normal and Large Angle, between 600 mm and the specimen entrance
near 1599.2 mm. It retains native axial lens excitation and explicit physical
apertures. It is a controlled-source calculation, not a converged physical-gun
population, a complete electric-and-magnetic column, an OEM calibration or a
production-mode qualification. AC scan and wobble are explicitly parked.
The separate 25-particle hybrid-source pilot retains the physical FEG and
legacy upstream propagation; its input is not substituted into this metrology.

## Fixed canonical input and reference

Both reference planes are immersed in magnetic field. Mechanical slopes alone
are therefore not a canonical chart. With electron charge q=-e and one frozen
common gauge, the measurement uses

```
P_perp / p_ref = (p_actual * u_perp - e * A_perp) / p_ref,
u_perp = (tx, ty) / sqrt(1 + tx^2 + ty^2).
```

The reference momentum is fixed at 300 keV. A central ray and four pairs of
canonical perturbations at four decreasing scales establish the raw 4 by 4
transfer matrix. The result is never symplectified. Scaled symplectic defect,
perturbation convergence, matrix conditioning and the input sheet's Lagrangian
defect are recorded separately.

The inverse nominal transfer defines a synthetic input sheet that focuses in
the linear map at the declared output plane. A fixed 192-point training pupil
and an independent 161-point holdout pupil, plus the chief ray, sample that
sheet. Its radius is 0.01 in normalized canonical transverse momentum. This is
a declared input construction, not a subsequent subtraction of measured focus.
The input is reused without refocusing after a drive or thickness change.

At the observation plane the diagnostic fits raw positions as

```
(x, y) = constant_intercept - grad_(P_perp/p_ref) W.
```

The two constant-gradient terms preserve chief/tilt. C1, C3 and all other fitted
terms are retained. An acceptable finite polynomial fit grants only that
finite-pupil gradient representation; independent field-order, trajectory-step,
energy-step and wave-admission checks remain necessary.

## Chromatic response and observable budget

The canonical energy family holds physical input canonical momentum and input
position fixed, while the same current field and reference momentum are used
for every energy. Energy changes therefore require different mechanical slopes
at the entrance. The measured quantity is named `Cc_response`, not an
unqualified field-free fixed-angle Cc. A separate fixed-mechanical family is
reported as a comparison and is explicitly ineligible for the immersed-source
scalar-wavefront qualification.

Three successively halved energy offsets are required. In addition to retaining
the raw changes in Cc, their image consequence at the declared pupil edge and
largest energy offset must fit the predeclared observation budget:

```
image_error_m = abs(delta_Cc_mm) * 1e-3 * pupil_radius * max(abs(delta_E/E)).
```

The default image budget is 1 nm. The coefficient-percentage check alone cannot
qualify the response. No phase budget or coherent-wave qualification follows
from this image-space check.

## Physical drives, causal replay and cross-coupling

The four simulator channels comprise fourteen independent physical controls:
DPH1/DPH2 X and Y, condenser-stigmator X and Y, HP1/HP2 normal and skew,
and HPC/HPOL normal and skew. These names express simulator assignments, not
verified vendor channel wiring.

Each drive is measured at positive, negative, positive half-step and negative
half-step. The unchanged upstream trajectory may be reused only before both
the old and new compact field-support intervals. Source digests verify that
only the selected physical sources changed. Every downstream support face and
aperture remains an integration boundary. No empirical aberration is injected.

The complete fifteen-coefficient response and constant intercept are retained,
including unwanted C1/C3 and other cross-coupling. Rank is compared against the
signed/half-step uncertainty; a deficient response must be reported instead of
repaired with an invented channel mapping. A combined small-control holdout
checks actual raw image displacements at unseen pupil points. HP2 effective
magnetic length is varied separately from 6.0 to 7.2 mm while retaining
excitation, the input sheet and reference. This is a finite-field-length
perturbation; it does not rebuild a magnetic circuit from an edited external
mechanical envelope.

## Numerical and physical limits

The Lorentz operator is fully relativistic and uses the frozen potential's
curl. Compact-field faces use the interior one-sided sample for each adjacent
integration interval; observation coordinates are unchanged. A regression
compares this rule with the independent exact uniform-dipole arc.

Strict DOP853 is used for canonical metrology. In the actual column a coarse
fixed RK4 reference produced percent-scale symplectic defects and was rejected;
reducing only reference perturbation size did not cure that integration error.
The accepted adaptive reference retains every physical input and has no matrix
projection or output centering. An optional exact native sampler accelerates
the same admitted formulas and does not loosen integration tolerances.

Gaussian radial field order and trajectory integration are refined independently.
The on-axis field is unchanged. Energy derivatives at those resolutions use
the same fixed energy pair, while the nominal three-pair scan establishes the
independent energy-step gate. Unknown FEM exterior, electric-field omissions,
spin, radiation, space charge, source-current convergence and dynamic scanning
are not covered by this static magnetic pilot.

The stored reduced canonical ray action is a ray quantity. It is not a sampled
coherent phase plate. A coherent comparison requires its own source wave,
gauge/reference conventions, support and sampling gates, and actual wave
propagation. The present diagnostic does not authorize retiring empirical
production screens or changing a preset.

## Recorded bounded results

The 12 tests in `tests/test_potential_probe_canonical.py` passed in 949.229 s
on the development CPU. The figures below use the stated 300 keV controlled
canonical family, not a physical FEG distribution or a field-free standard
aberration measurement. All values are deterministic simulation evidence.

| Observable | Nanoprobe Normal | Nanoprobe Large Angle |
| --- | ---: | ---: |
| Canonical C3 response, radial field order 7 (mm) | -16.85508242 | -11.18661850 |
| Canonical Cc response, order 7 and relative energy pair +/-0.0005 (mm) | -19.98117898 | -17.02495906 |
| Nominal raw transfer scaled symplectic defect | 3.68e-10 | 5.11e-10 |
| Maximum seven-energy gradient holdout RMS (m) | 1.130e-11 | 6.168e-13 |
| Last energy-derivative refinement image estimate (m) | 4.796e-12 | 3.491e-12 |
| Field order 5 to 7 maximum raw image change (m) | 4.920e-17 | 1.952e-17 |
| Joint-control holdout RMS (m) | 1.092e-11 | 5.365e-13 |
| Joint-control maximum raw position change (m) | 1.121e-9 | 6.766e-10 |
| HP2 effective-length change: canonical C3 response change (mm) | +0.08032169 | +0.00267980 |

The strict adaptive trajectory comparisons are already at their numerical
floor: changing the maximum step from 0.25 through 0.125 to 0.0625 mm changes
raw image coordinates by less than 8.3e-18 m. This is an agreement of these
numerical solutions, not a claim of physical accuracy at that length scale.
The independent field expansion, source assumptions and model calibration
remain relevant limitations.

All four channel target matrices have two resolved singular directions in
both modes when compared with their positive/negative half-step uncertainty.
The full 15 by 14 response matrix, constant intercept, individual drive units,
uncertainty and held-out predictions are retained in the test evidence. This
does not establish an OEM sensitivity hierarchy: at the existing, differently
unitized calibration steps, A1 fine has a larger target response per step than
A1 coarse, and the two A2 fine components have markedly different responses.
No channel is renamed, rescaled or reassigned from this observation. These
local matrices support bounded future design studies only when their declared
source, gauge, field, pupil and control-range restrictions are retained.
