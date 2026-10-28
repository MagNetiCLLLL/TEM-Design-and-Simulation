# CEOS probe-corrector ray-path audit

Date: 2026-10-28. This is a primary-source and configuration audit. No optical
field, geometry, production code, source population or preset was changed.
No particle or coherent propagation was executed for this document.

## Applicable manufacturer reference

The simulated ordering most closely follows the two-hexapole CESCOR diagram.
CEOS describes CESCOR as a two-stage hexapole corrector. DCOR/ASCOR adds design
freedoms that suppress sixfold astigmatism and correct fourth-order axial
aberrations. Similar names in a service interface do not establish those
capabilities in the simulator. The simulator's uncalibrated auxiliary channels
and provisional magnetic profiles are not a validated DCOR/ASCOR field model.
Sources: [CEOS CESCOR](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/cescor)
and [CEOS DCOR/ASCOR](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor).

The official diagrams inspected visually were:

- [Muller et al. (2006), Figure 2, printed page 444 / PDF page 3](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=3).
- [Lu, CEOS-hosted MC2025 workshop, page 2, CESCOR-STEM panel](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/MC2025_CEOS_PL_20250831.pdf#page=2).

Local reference copies and rendered pages are in the ignored
`outputs/ceos-reference-20261028/` directory. They remain unmodified external
material. Neither diagram is a dimensioned production drawing.

## Two different families of intersections

The colored 2025 CESCOR panel distinguishes the axial ray from the selected
field ray. Counting their zeros separately gives:

| Ray family | Intermediate axial zeros shown | End condition |
| --- | --- | --- |
| Axial ray / source-image family | Three: after CL3 and before ADL; between TL22 and TL21; near DP12 between the scan coils | Focus at the specimen |
| Selected field ray / aperture-image family | Three: HP2 midplane; HP1 midplane; the objective front-focal/aperture region | Nonzero field height at the specimen |

Thus the HP2-to-HP1 core contains one source-image crossover, not three.
The two hexapole-plane field-ray zeros are different conjugate planes and must
not be added to a source-waist count. The source-image crossing downstream of
TL12 is distinct from the later field-ray zero near the objective. This count
covers the diagram's displayed range, not every crossover in the microscope.
Source: [2025 diagram, page 2](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/MC2025_CEOS_PL_20250831.pdf#page=2).

The 2006 reference allows the microscope mini lens to replace TL11 and scan
coils to implement DP12. Its depicted aperture-conjugate alignment connects
the HP midplanes to the objective coma-free aperture plane; the downstream
transfer magnification is adjustable. This does not require equal source-image
angles or a whole-corrector minus-identity map. C5-free alignment changes the
aperture-plane relation, as discussed below.
Source: [Muller et al., pages 443-444](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=2).

## Normal-mode compatibility and exact-position limits

The 2025 schematic labels its upstream lens **CL3** and draws a source-image
crossover before ADL. Neither reference assigns the project's Normal/Large
labels to that drawing. A name match alone therefore cannot establish the
corresponding physical element or force a change of operating branch.

The user confirmed on 2026-10-28 that the Normal entrance is retained and the
corrector interior follows the CEOS matching principle. The subsequent
clarification defines C3 as weak because no additional crossover occurs
between C3 and the next optical element. C3 is active: its finite field and
the following optics jointly produce the next crossover. Retain C3's profile
and overlapping fringe fields rather than replacing it by zero excitation
or an independently acting thin lens. The existing required C2-C3 crossover
and forbidden C3-ADL crossover remain in effect for this assembly.

The selected internal pattern is exactly one resolved two-dimensional
source-image waist between TL22/TL21 and one inside the AC scan-coil pair.
Every other actual waist and unresolved interval is recorded. This selected
physical contract does not claim full diagram equivalence, an OEM branch
name, or matching absolute dimensions. See the
[control contract](../OPTICAL_CONTROL_ROLES.md).

The user's later clarification makes direct Mini Condenser plus Upper
Objective focusing mandatory: the next focus after that combination must be
the specimen upper surface, with no distinct extra pre-specimen crossover.
Older strong-branch results with such a waist are historical counterexamples,
not candidate qualification. The final surface-focus tolerance remains a
separate numerical check.

Public schematics supply relative locations. They do not specify OEM absolute
Z coordinates, a crossover-position tolerance, or an all-microscope waist
count. Keep the current fixed geometry for a strength-only study and report
both geometric-reference error and the difference from the executed old Normal
reference. Do not silently equate those two references.

## Current configuration: a substantive symmetry mismatch

The following source-absolute positions are the unchanged default assembly
coordinates, obtained by adding the 450 mm gun offset to
[C3_ProbeCorrector.toml](../../configs/instruments/column/C3_ProbeCorrector.toml).

| Element/reference | Z, mm | Configuration detail |
| --- | ---: | --- |
| HP2 | 1144.5 | Effective field length 6 mm |
| TL22 | 1166.35 | Field width parameter 6 mm; negative polarity |
| TL pair geometric midpoint | 1208.15 | Arithmetic midpoint, not a measured waist |
| TL21 | 1249.95 | Field width parameter 6 mm; positive polarity |
| HP1 | 1300.85 | Effective field length 10 mm |
| TL12 | 1366.3 | Downstream matching lens |
| Upper/lower AC coil reference centers | 1387.5 / 1412.5 | Midpoint 1400 mm |
| Mini Condenser | 1522.5 | Nanoprobe negative polarity |
| Specimen upper surface in the paired run | 1599.1999975 | Actual evaluation plane |

The HP centers lie 63.65 and 92.70 mm from the TL-pair midpoint. Their effective
lengths differ, and the TL22/TL21 field amplitudes at equal percent also differ.
This assembly is not a literal reflection-symmetric core. Its historical
near-unit HP position relay does not remove that discrepancy. Strength changes
may improve optical matching with this fixed geometry, but exact geometrical
reflection symmetry cannot be created merely by changing excitations.
The 1208.15 mm geometric midpoint is consequently not a verified optical
symmetry plane. Optical symmetry or equivalent cancellation must be tested
using propagated maps and aberration responses, not assigned by midpoint alone.

The fresh old-Normal control replay found the central source-waist bracket at
1202.178571 mm, 5.971429 mm upstream of the geometric TL midpoint. The old
strong-C3 reference near 1208.148342 mm belongs to another branch and must not
replace it. An optimization that preserves the old Normal waist position
exactly is therefore a different experiment from one that targets the TL
geometric midpoint exactly.

CEOS identifies focusing the central image at the symmetry plane as a way to
minimize fourth-order three-lobe aberration D4. It also identifies hexapole
strength and transfer magnification as spherical-aberration controls, with
additional coupled settings for fifth order. This makes internal matching a
specific candidate explanation to test; it does not establish that a measured
fourth-order residual in this simulator is caused solely by its displaced
central waist. Source: [CEOS residual-aberration discussion](https://www.ceos-gmbh.de/en/produkte/residualsCEXCOR).

## What fixed-geometry field freedoms can address

For C5-free alignment, the 2006 example jointly adjusts TL11, TL12 and
Objective: one changes C5 while two compensate focus and preserve C3
correction. With fixed HP strengths, the specimen-to-central-image
magnification is retained. HP1 is then imaged between the objective coma-free
aperture plane and specimen, rather than exactly onto that aperture plane.
The demonstrated experiment is CETCOR; the paper explicitly discusses its
STEM analogue. Source:
[Muller et al., pages 447-448, Figure 5](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=7).

For this simulator, that supports retaining TL12, Mini Condenser and Objective
as coupled fitting freedoms alongside the main HP and internal transfer
controls. It does not turn Mini Condenser into the primary convergence-angle
control. The selected C2/C3 role and actual alpha95 constraint still apply.
Do not simultaneously freeze the illustrated coma-free aperture conjugacy
and require C5-free alignment without testing whether both are attainable.
Report the achieved aperture image plane and driven scan response.

D4 cancellation requires the correct optical matching, not simply more
hexapole strength. In the current asymmetric assembly, the number and rank of
independent controls needed to suppress its complex residual must be measured.
Two scalar controls need not remove two residual components while preserving
focus, alpha95 and both source images if their measured responses are dependent.

Additional HP/QP channels are legitimate response candidates: the CEOS-hosted
workshop identifies extra multipoles and transfer elements for alignment and
parasitic-aberration adjustment. It does not supply a calibrated HPC/HPol/QPC
to D4/C5 map for this implementation. Source:
[2025 workshop, page 2](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/MC2025_CEOS_PL_20250831.pdf#page=2).
Small signed perturbations should establish whether those channels add an
independent useful direction after preserving the required first-order
conditions. Quadrupoles can also introduce or split two-dimensional waists;
their inclusion requires checking both principal directions. These are local
fitting criteria, not a claim that arbitrary auxiliary strengths guarantee
DCOR-like fourth-order correction or a sub-0.7 nm result.

## Relationship to the 854 nm result

### Length and magnification limits of an A5 explanation

The 2006 design uses 30 mm mechanical pole pieces. Those dimensions are not
directly interchangeable with this model's 6/10 mm effective Gaussian lengths.
The paper relates A5 to axial-ray height `M_TL f_OL`, hexapole length and transfer
lens contributions. Its main-field approximations give
`|A5| proportional to (C3_OL / (M_TL f_OL))^2`, and, at maintained C3 correction,
`I_HP proportional to L_HP^(-3/2)` and `A5 proportional to L_HP`. The latter
approximation fails for sufficiently short elements because transfer-lens
contributions matter. Increased magnification also increases chromatic cost.
Direct A5 correction is discussed using dodecapoles, not ordinary quadrupoles.
Source: [Muller et al., pages 443 and 451-452](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=10).

Consequently, field tuning can change relevant optical ratios, but the source
does not guarantee that arbitrary fixed unequal geometries can cancel every
residual. Equally, unequal lengths do not prove this model cannot meet D95
below 0.7 nm. Establish the actual aberration response before assigning a
failure to A5 or proposing geometry changes. The paper's probe metrics are
not interchangeable with this project's finite-current D95.

The [finite-waist audit](CORRECTOR_CROSSOVER_REFINEMENT_2026-10-28.md) separately
compares directional splitting with intermediate waist depth. Such splitting
does not by itself count as an additional source crossover.

The existing strict same-input comparison gives spatial D95 0.589372 nm for
`custom` and 854.265910 nm for `unified`, using the same old HP strengths and
174 specimen survivors. Their shared 1400 mm AC plane has RMS radius 2.458080
versus 686.101566 nm. Thus the discrepancy is already substantial before the
Mini Condenser and Objective. The newly stronger default HP settings are not
the independent variable in this comparison.

Both same-control replays also record a bounded-sampling waist near 1585.703521
mm between the Mini Condenser and specimen. The published CESCOR axial ray
does not show an additional source crossing in that interval. A full-source
current-weighted waist and one Gaussian fundamental-ray zero are not identical
observables, so this is an audit flag, not by itself proof of a propagation bug.
The old Normal chain contains five observed intermediate waist brackets in the
full simulated incident column; merely matching that number does not establish
matching CEOS relative positions. Whole-chain status remains `UNRESOLVED`.
Measured evidence and limitations are in the
[same-control comparison](CORRECTOR_ANGLE_TRANSFER_COMPARISON_2026-10-28.md).

## Constraints for the next strength search

1. Preserve the selected Normal condenser topology and C3's active combined
   field. Require exactly one source-image waist between TL22/TL21 and one
   inside the AC coil pair. Resolve both transverse directions. Detect all
   additional waists and report their locations rather than hiding them behind
   a total-count match.
2. Measure the HP-to-HP and HP-to-downstream aperture imaging separately
   from full-source waists. Use the current finite-field propagator; do not
   transplant a legacy paraxial map. Report positions, angular distributions,
   magnifications and residuals rather than imposing a whole-corrector unit
   map or freezing a coma-free image plane during C5-free tuning.
3. Use C2/C3 as the primary angle controls; include ADL, TL22/TL21, TL12,
   hexapoles and available probe trim fields for measured matching and
   aberration residuals. Use Mini Condenser/Objective for specimen focus.
   Perturb and retrace locally before attempting large combined changes.
4. Keep the geometric midpoint, any measured optical symmetry plane and old
   Normal central waist as explicitly different references. There is no
   published CEOS tolerance that justifies calling the geometric or historical
   value exact for this provisional assembly.
5. Accept only simultaneously measured alpha95 near 30 mrad, spatial D95 below
   0.7 nm, the selected crossover pattern, and numerical/source convergence.
   Keep original source weights and report physical losses. Static waist
   matching does not qualify the driven scan pivot or a scanning image.
6. Reject a distinct additional source waist between Mini Condenser and the
   specimen. The combined downstream fields must directly focus at the
   specimen upper surface, as subsequently required by the user. Do not
   preserve the historical extra-waist branch merely to match an old count.

This audit does not claim an OEM-identical layout, a corrected 30 mrad probe,
or a new qualified preset.
