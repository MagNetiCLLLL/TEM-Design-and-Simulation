# Corrector rematching: causal checks and unresolved topology

Date: 2026-10-28. The requested target remains specimen-surface alpha95 =
30 mrad and spatial D95 < 0.7 nm, using actual transmitted original current
weights. All probe internal field strengths are now authorized tuning controls.
C2/C3 remain the primary angle controls; Mini Condenser and Objective provide
focus compensation. No new preset has been installed by this investigation.

## Findings

The 854 nm result is reproducible but is not the size of every finite-field
working point. It belongs to the strict old-Normal-control replay in the new
field model. The previous fitted finite-field candidate already reduced D95
to 59.56 nm near alpha95 29.33 mrad, which is still far from acceptance.

The present investigation supports three separate observations:

1. A real Objective-only adjustment removes much of the apparent defocus,
   reducing the strict replay's D95 from 854.27 to 305.51 nm. It does not remove
   the remaining aberrated distribution.
2. Local differential transport predicts a small near-axis image, while the
   actual finite-amplitude beam departs strongly from that prediction, already
   at the AC plane. A large first-order focus error alone cannot describe the
   observed broadening. The remainder includes chromatic and finite-amplitude
   effects; this experiment does not uniquely isolate one aberration order.
3. The current assembly and old Normal crossover path are not a literal copy
   of the published symmetric CEOS optical diagram. Internal matching and
   higher-order aberration correction must be checked together.

The [field implementation audit](CORRECTOR_FIELD_IMPLEMENTATION_AUDIT_2026-10-28.md)
found no demonstrated sign, unit or double-counting defect. It records the
different natural round-lens aberrations and the limits of the engineering
multipole fringe model. Its 92 field-operator tests passed. These are separate
from the 92 GUI/topology checks recorded earlier; neither suite qualifies a
physical probe.

## Executed Objective-only causal trial

The complete validated 193-particle gun source, all original weights, energies
and clocks were reused. Exactly the same 174 particles reach the specimen;
the same 19 are intercepted at C2 Aperture. Source current is 10 microampere,
specimen current 9.015544 microampere, and transmission 90.155440 percent.
No current floor or artificial particle selection is applied.

Only Objective excitation changes: 68.9801 to 68.94637 percent, a change of
-0.03373 percentage point. C2/C3, HP fields, geometry and source are unchanged.
The maximum integration step is 0.05 mm. The new run uses the standard 3,696
stored observation planes, while the earlier strict model comparison used a
3,994-plane grid with extra local observers. This is a physical focus test,
not an exact identical-observer numerical-error study.

| Actual executed result | Original strict unified replay | Objective-only trial |
| --- | ---: | ---: |
| alpha95, mrad | 29.330967704 | 29.323115581 |
| D95, nm | 854.265910425 | 305.505956710 |
| RMS radius, nm | 188.773703128 | 82.982903845 |
| Local covariance-drift focus estimate, nm | -9442.929506 | +3.506165 |

The covariance-drift value is a diagnostic estimate, not a two-sided
finite-field waist qualification. The actual trial agrees closely with the
earlier offline scalar-drift prediction, D95 305.729811 nm. The latter is not
itself a physical corrected probe. The full crossover chain remains unresolved,
and this trial does not satisfy size, angle or joint qualification.

Raw evidence:
`outputs/finite-field-corrector-rematch-20261028/legacy-refocus/`.
The one-column measurement completed in 178.545 seconds, using one CPU thread,
zero new gun executions and zero coherent calculations.

## Executed local transport diagnostic

One production `trace_unified_response` execution propagated 17 zero-current
differential probes about the original weighted gun-exit mean. The perturbations
are plus/minus 10 nm in x/y and plus/minus 1 nrad in slopes, with half-amplitude
partners. All diagnostic rays survive. The mean source energy and flight time
define this local map; the actual full source remains mixed-energy.

This measurement uses the unchanged strict old-Normal unified controls. It
does not replace, shrink or reweight the physical source. Comparison with real
particles retains the same original surviving IDs at each observed plane.

| Same-cohort centered RMS, nm | AC at 1400 mm | Specimen |
| --- | ---: | ---: |
| Actual finite-source beam | 686.101566 | 188.773703 |
| Local half-amplitude linear prediction | 3.781739 | 0.204173 |
| Actual minus local prediction | 682.713037 | 188.601164 |

The sample difference between full- and half-amplitude map predictions is
0.102967 nm centered RMS. The small local prediction is therefore not a
converged sub-nanometre beam result. Its diagnostic D95 is approximately
0.734315 nm and is not an executed full-source probe. The original helper's
un-centered sample residual, 188.743151 nm, must not be confused with the
centered value in this table.

The 65.046-second execution and independent read-only audit are stored in
`outputs/finite-field-corrector-rematch-20261028/local-transport-old-normal/`
and `local-transport-independent-audit.json`. No field coefficients were changed.

## Confirmed Normal entrance and CEOS internal matching

The [official-diagram audit](CEOS_PROBE_CORRECTOR_REFERENCE_2026-10-28.md)
distinguishes three intermediate source-image crossings from the selected
field ray's aperture-plane crossings. Only one source image is inside the
HP2-to-HP1 core, between TL22 and TL21; another is near DP12 inside the scan
coil pair. Public schematics do not establish OEM absolute Z coordinates.

The old Normal central waist bracket is at 1202.178571 mm, while the TL
geometric midpoint is 1208.15 mm. The 5.971429 mm difference is a concrete
matching question. HP centers are also unequally spaced around that midpoint
and have different effective lengths. Exact geometric reflection symmetry
cannot be created by field-strength changes alone. The 1208.15 mm arithmetic
midpoint is not a validated optical symmetry plane; report geometric-reference
error separately from executed optical matching and the old Normal waist.

CEOS describes minimizing fourth-order three-lobe aberration by placing the
intermediate source image at the core symmetry plane, and tuning fifth-order
spherical aberration through transfer settings and HP/focus changes. These
are useful physical tests, not proof that this assembly has a reachable
qualified solution. Source:
[CEOS residual-aberration discussion](https://www.ceos-gmbh.de/en/produkte/residualsCEXCOR).

The user has confirmed retaining the Normal entrance and matching the
corrector interior. This supersedes the earlier pending branch question.
The user's additional clarification is physical rather than label-based:
the project's names may differ from the diagram; C3 is weak, with no extra
crossover before the next optical element, and its field together with the
following optics forms the next crossover. C3 remains active with its finite
profile and overlapping fringes. No C3-ADL crossover is introduced merely to
match the schematic's CL3 label.

The internal acceptance pattern requires exactly one resolved two-dimensional
source-image waist between TL22/TL21 and exactly one inside the AC coil pair.
All other actual waist brackets and unresolved intervals must be counted and
reported. This preserves the selected Normal branch without claiming that the
entire vendor drawing or unspecified OEM absolute positions are reproduced.

The user's subsequent clarification additionally requires the Mini Condenser
plus Upper Objective fields to focus directly onto the specimen upper surface,
without a distinct extra pre-specimen crossover. New accepted candidates must
use that branch. The earlier strong-branch waist near 1585.7 mm is retained
only as diagnostic history; it is a rejection condition for a new preset,
irrespective of final D95. Resolve the final specimen-focus bracket and its
numerical tolerance separately from such an additional waist.

Use bounded, retraced adjustments of ADL,
TL22/TL21, TL12, HP1/HP2 and the existing auxiliary HPC, HPol and quadrupole
fields as necessary. Record both reference-plane and actual waist positions,
all additional waist brackets, and the aperture conjugacies. Do not impose a
whole-corrector minus-identity map or equal angles at successive source images.
The [fixed-geometry freedom discussion](CEOS_PROBE_CORRECTOR_REFERENCE_2026-10-28.md#what-fixed-geometry-field-freedoms-can-address)
also distinguishes the illustrated coma-free aperture alignment from C5-free
alignment. Do not freeze one aperture conjugate plane as an extra hard
constraint when deliberately testing the latter; measure its actual location
and requalify the scan response.

Acceptance still requires the full source, numerical and source-count
convergence, selected crossover pattern, angle, specimen size and driven-scan
validation. No new scanning image, accepted preset or Git upload is claimed.

## Identities

- Solver: `f34f2ff6f73532f29faa3652fd88dd902625bd248d2196e5d94c3f0b08c5cd21`.
- Complete source: `4e3bac1af215cfe7b8589e64363541ba2039654fb01be49e8a5abee3017c0815`.
- Gun section SHA256: `0d62afe83714e7e3745354ffccdcfa15b923dad0ed6a282a92c4d43f8fe04bcb`.

Production source was kept fixed during both physical executions. Diagnostic
scripts and actual input files are retained in the ignored output directory.
