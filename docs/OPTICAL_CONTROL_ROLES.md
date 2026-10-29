# Optical control roles and illumination constraints

This is the project's control contract clarified by the user on 2026-10-27.
It describes the intended TEM/STEM operating roles, not a universal vendor
current table or evidence that every stored working point has been qualified.
The roles are coupled: changing one lens can also change transmission, source
imaging, convergence, aberrations and downstream focus.

## Controls

| Component | Intended role | Required interpretation |
| --- | --- | --- |
| Gun Lens | Control the current admitted to the column by changing how the gun beam passes through the gun aperture. | Calculate physical aperture interception and retain absolute source weights. Changing lens strength does not itself change the emitted current or justify renormalizing transmitted particles. |
| C1 Lens | Control microscope **spot size** through source imaging/demagnification, and the current delivered below C1. | Spot size is the electron-microscope operating term. It is not the literal specimen-plane spot radius, a raster pixel size, or a claim that every vendor uses the same numbered scale. Preserve any existing scale; do not invent an index-to-current calibration. |
| C2 and C3 | Jointly control probe convergence semi-angle in Nanoprobe mode. | Measure the executed beam at the specified specimen plane with the declared current-weighted angular statistic. A saved target or a single lens percentage is not a measurement. |
| C3 | Select the Normal or Large angle optical branch through its excitation and resulting crossover topology. | C3 range selection is distinct from the Mini Condenser's Nanoprobe/Microprobe selection. Absolute excitation percentage alone does not establish the branch. |
| Mini Condenser | Select the probe/illumination mode, Nanoprobe or Microprobe. | Preserve the current magnitude unless a deliberate tuning operation changes it. Existing mode-specific polarity rules remain in force. A change can require downstream focus revalidation. |
| Mini Condenser and Objective | Jointly form the probe. | Account for both fields, their geometry, and the incoming beam. Do not infer a focused probe from a mode label. |
| Objective | Control whether the probe is focused on the specimen. | Evaluate focus at the actual specimen entrance/reference plane and report which plane is used. Preserve defocus when the user deliberately perturbs a lens. |
| Probe Corrector | Correct aberrations while maintaining the required optical conjugacies. | Compare the executed source-image waist between TL22 and TL21 with the executed waist between the two AC scan coils, but do not assume equal semi-angles. Whole-corrector mechanical faces and fringe-field reference planes are supplementary observations. Report actual transmitted populations and a common surviving cohort with original weights. |
| AC Scan | A virtual control channel of the Probe Corrector, acting through the existing two physical scan coils. | Its common tilt centre is to coincide with the front focal plane of the downstream Mini Condenser plus Upper Objective combination, converting a beam tilt into a probe shift at the specimen. Keep this distinct from the upstream C3 Lens and do not add another physical field for the virtual channel. |

The stored presets can assign several controls together. Their values are
reference starting points, not independent guarantees for each role. AC/Descan
calibration changes scan-drive response matrices, not these illumination lens
excitations. Descan remains a control channel on the existing shared
Image/Diffraction physical deflector; it is not an extra magnetic element.

For the current preset search, use C2 and C3 as the primary controls for the
specimen convergence semi-angle. Use the Mini Condenser and Upper Objective
combination to compensate focus at the specimen surface, with corrector
entrance/output matching and aberration correction measured separately. Their
physical coupling must be reported; it does not make Mini Condenser or Upper
Objective the primary angle-fitting controls. During an aberration-only
adjustment, hold C2/C3 fixed and check that a smaller probe was not obtained
merely by reducing the convergence angle.

## Required crossover topology

For C3 + Probe Corrector in STEM Nanoprobe:

| Range | C2 to C3 | C3 to Adapter Lens (ADL) |
| --- | --- | --- |
| Normal | A crossover is required. | No crossover; C3 is weak in the project's convention. |
| Large angle | A crossover is required. | A crossover is required; C3 is strong in the project's convention. |

More generally, strong excitation means that the executed beam forms a
crossover after the lens and before the next optical lens; weak excitation
means no crossover in that interval. Weak does not mean disabled. Use an
executed two-dimensional, current-weighted waist and sufficient axial sampling;
a crossing of projected rays or an unresolved interval cannot certify topology.

Any future working-point optimization must check the selected topology together
with convergence angle, specimen focus, transmission and its numerical error
budget. It must not obtain a passing result by dropping weak particles,
recentering the beam, changing source weights, or silently refocusing each
member of an independent lens-perturbation experiment.

## Corrector crossover comparison

The user clarified the two comparison locations on 2026-10-27. The Normal-mode
C3-to-ADL interval must remain free of crossovers. The next downstream
source-image crossover is sought inside the corrector, between TL22 and TL21;
the second crossover is sought between the upper and lower AC scan
coils. Determine both from executed particle trajectories. A geometric midpoint,
an interpolated candidate or a projected-ray intersection alone is insufficient.

On 2026-10-28 the user confirmed keeping the Normal entrance while matching
the corrector interior to the CEOS optical principle. Device names need not
map one-to-one to the manufacturer's schematic. In physical terms, C3 is weak:
there is no additional crossover between C3 and the next optical element.
C3 remains active, and its field together with downstream fields forms the
next crossover. Preserve its finite profile and overlapping fringes; weak
excitation is neither zero field nor an independently acting thin-lens rule.

The internal acceptance pattern is exactly one resolved, two-dimensional
source-image waist between TL22 and TL21 and exactly one within the two AC
scan-coil reference centers. Count and report all other actual waist brackets
and unresolved intervals throughout the incident column. A matching total
count alone is insufficient. This is an internal matching contract, not a
claim that the full published upstream ray path or OEM absolute positions
have been reproduced.

The user's subsequent 2026-10-28 clarification makes the downstream branch
mandatory: Mini Condenser and Upper Objective must focus directly onto the
specimen upper surface, with no separate source-image crossover before that
surface. A distinct additional waist between Mini Condenser and specimen is
a rejection condition, even if the final spot is small. Resolve the final
surface-focus bracket and numerical tolerance separately from an additional
upstream waist. The older strong downstream branch with an extra crossover
is historical diagnostic evidence, not an acceptable branch for new presets.

With the current fixed geometry, 1208.15 mm is only the arithmetic midpoint
of the TL22/TL21 reference centers. The core's unequal HP spacing and field
lengths mean that this point is not a validated optical symmetry plane.
Report the measured waist position, its difference from this midpoint, and
its difference from the old Normal reference separately. Determine useful
optical matching from actual finite-field transfer and aberration responses;
do not silently impose 1208.15 mm as an OEM requirement. See the
[CEOS reference audit](development/CEOS_PROBE_CORRECTOR_REFERENCE_2026-10-28.md).

CEOS's [original optical diagram and accompanying text, Figure 2 and
pp. 443-444](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=3)
place source images at the central transfer-doublet symmetry plane and near DP12.
They also describe replacing TL11 with the microscope mini condenser and
emulating DP12 using weighted offsets on the two scan coils. This identifies
the intended reference locations; it does not specify the simulated coordinates
or prove that their semi-angles are equal.

The user's later clarification asks whether equal semi-angles are physically
required, rather than imposing equality as a settled design constraint. Treat
the ratio as a diagnostic by default. Equal semi-angles must not be inferred
from the separate unit-magnitude HP2-to-HP1 relay, and a non-unit ratio alone
does not justify moving components or changing effective field lengths. Any
deliberate equal-angle experiment must declare that additional constraint and
independently verify the corrector conjugacies and aberrations.

Check the actual current-weighted angular distributions at both waists, and
separately report changes caused by physical clipping. Retain
centroids, mean tilts, source energies and unscaled particle weights. Static
waist measurements disable raster and wobble on a detached diagnostic state;
the common tilt centre and probe-shift response require a separate driven test.
In particular, a static source-image waist is not automatically the virtual
pivot of trajectories produced by different scan commands. The axial ray and
the selected field ray in the CEOS diagram describe different conjugate-plane
families. State the entrance/output reference planes, retained fringe fields
and measured sample angle when assessing a front-focal-plane claim.

For the strength search requested on **2026-10-28**, the current target is
**alpha95 = 30 mrad at the specimen upper surface**, a **100 um C2 aperture
diameter**, and the **native D95 waist at that surface**. The user's latest
instruction removes the earlier D95 < 0.7 nm hard limit. Report and minimize
D95 without imposing an undeclared size cap. Both 95% statistics use the original positive-current weights
that physically reach that plane: alpha95 is the angular containment semi-angle
about the current-weighted chief direction; D95 is the spatial containment
diameter about the current-weighted beam centroid. Report RMS size separately;
the previous 0.2 nm candidate-screening value is not an additional user-selected
acceptance limit. There is no prescribed transmitted-current limit. Report the absolute current
and interception losses; neither acceptance filtering nor arbitrary current
normalization is permitted.

This latest selection supersedes the **2026-10-27 physical pupil-edge 30 mrad
objective for new tuning**. Keep the physical pupil edge as a separate measured
observable; do not infer it from alpha95 or the maximum sampled particle angle.
Historical edge-acceptance reports retain their original definitions and do not
qualify the new alpha95 working point. Follow the remaining topology, source,
focus, current, convergence, and driven-scan requirements in the
[optical acceptance contract](development/FINITE_FIELD_OPTICAL_ACCEPTANCE_CONTRACT_2026-10-27.md)
before claiming joint qualification. The older alpha95 fitters still require
those independent checks and must not silently impose their extra historical
size/current gates on this user-selected specification.

Use `scripts/measure_optical_acceptance.py` for the separate physical-pupil
measurement and incident-column joint report. It brackets actual source-to-column
mechanical interception with zero-current diagnostic rays and reports source
filling and current quantiles separately. Its reference-source, nominal-energy
boundary is not a complete mixed-energy finite-source or scan qualification.
The older `qualify_finite_field_nanoprobe.py --target` remains an explicitly
labelled alpha95 diagnostic and must not certify a 30 mrad aperture-edge target.

## Unified D95 probe-waist definition

The user's subsequent 2026-10-28 clarification requires a consistent size and
focus definition. For the current 100 um C2 / 30 mrad alpha95 probe target,
the specimen probe waist is the minimum of **native current-weighted D95(z)**,
and that minimum must lie at the specimen upper surface within the declared
numerical placement tolerance. The latest instruction removes the former
0.7 nm size limit; no default absolute D95 acceptance cap applies. Report the
actual D95, D90 and RMS separately. Any future optional size cap must be
explicitly selected and recorded, never silently inherited from old reports.

Before resuming joint tuning, perform the requested monochromatic-source
control and distinguish source-energy effects, gun numerical precision and
field-model sensitivity. Hold physical controls fixed during attribution and
report the actual alpha95 in every control experiment. Do not silently retune
an energy/control experiment to 30 mrad; retain 30 mrad as the subsequent
working-point target. Changing the source energy law is a declared physical
control experiment, not fixed-source numerical convergence.

Do not substitute an RMS minimum, an RMS covariance-waist estimate, a smoothed
quantile, or a fitted parabolic vertex for this D95-waist condition. Earlier
mixed RMS-focus/D95-size searches are historical proposals, not qualification
under the clarified definition. RMS focus and RMS convergence are no longer
hard gates for this probe target; numerical convergence must instead check
native D95, its observed minimum-plane bracket, alpha95 and actual current.

Determine D95(z) from actual unified-field propagation at explicitly saved
planes and each plane's complete actual positive-current population. Preserve
all original source IDs, weights, energies and contact records. A minimum at
a scan boundary, an unresolved plateau or competing minimum, or a nearby
population change cannot certify a continuous optical waist. A scan extending
past the sample reference plane is a vacuum-probe optical diagnostic; it does
not validate specimen scattering. Upstream conjugacy/topology observations
remain separate evidence and do not prove the specimen D95-waist location.

The existing engineering placement budget is +/-1 nm. Report the actual
executed bracket and its sampling resolution; refine the local observations
before qualification. This is a numerical tolerance, not a claim to prove an
exact real-number minimum at an exactly equal Z coordinate.

## Qualification boundary and this maintenance change

The current Finite-Field Normal preset has a fixed 49-particle, approximately
10 mrad focus record at 0.05 and 0.025 mm integration steps. Its record explicitly
marks source-sampling convergence and condenser topology as `NOT_TESTED`.
See [the recorded scope](development/FINITE_FIELD_NANOPROBE_DEFAULT_2026-10-27.md).
That focus record does not prove either crossover condition above. Historical
legacy-field topology results do not qualify the shared-potential executor.

The scan-routing, gun-exit, validation and reference-label repairs do not retune
the stored lens strengths, change apertures, alter source sampling or change the
integration accuracy. They preserve these control roles and make reference
claims explicit. A separate optical qualification is still required before
declaring the present finite-field preset compliant with the topology contract.
