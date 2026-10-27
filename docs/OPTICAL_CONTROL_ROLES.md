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

The user clarified that the 30 mrad specimen target means the physical pupil /
calibrated diffraction-disk edge semi-angle. Report alpha95 separately; do not
substitute it or the maximum sampled particle angle for an unknown pupil edge.
There is no prescribed transmitted-current limit. Report the absolute current
and interception losses; neither acceptance filtering nor arbitrary current
normalization is permitted. The existing alpha95-based fitters do not yet
qualify this edge target. Follow the
[optical acceptance contract](development/FINITE_FIELD_OPTICAL_ACCEPTANCE_CONTRACT_2026-10-27.md)
before joint excitation and geometry tuning.

Use `scripts/measure_optical_acceptance.py` for the separate physical-pupil
measurement and incident-column joint report. It brackets actual source-to-column
mechanical interception with zero-current diagnostic rays and reports source
filling and current quantiles separately. Its reference-source, nominal-energy
boundary is not a complete mixed-energy finite-source or scan qualification.
The older `qualify_finite_field_nanoprobe.py --target` remains an explicitly
labelled alpha95 diagnostic and must not certify a 30 mrad aperture-edge target.

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
