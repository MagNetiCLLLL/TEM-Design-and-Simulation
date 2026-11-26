# Optical control roles and illumination constraints

The operating rules below are specific to the installed column and selected
mode. Changing one lens can change transmission, source imaging, convergence,
aberrations and downstream focus together.

## Controls

| Component | Intended role | Required interpretation |
| --- | --- | --- |
| Gun Lens | Set the illumination of the downstream apertures and hence the current admitted to the column. | Apply physical interception at the gun electrodes, gun aperture and C1 aperture. The 1000 µm C1 aperture can select the beam. Measure the current intercepted by each element and retain original source weights. |
| C1 Lens | Control microscope **spot size** through source imaging/demagnification, and the current delivered below C1. | In the requested Thermo Fisher convention, spot size is a dimensionless integer operating index, not a specimen-plane diameter or raster pixel size. Preserve any existing scale; a calibrated index may select coupled condenser settings, but no unmeasured index-to-current table may be invented. |
| C2 and C3, when installed | Jointly control probe convergence semi-angle in three-condenser Nanoprobe mode. | A physical two-condenser column instead uses its actual C1/C2 illumination system; do not require or synthesize a missing C3. Measure the executed beam at the specified specimen plane with the declared current-weighted angular statistic. A saved target or a single lens percentage is not a measurement. |
| C3 | Select the Normal or Large angle optical branch in hardware that contains C3. | A physical two-condenser column has neither branch. C3 range selection is distinct from the Mini Condenser's Nanoprobe/Microprobe selection. Absolute excitation percentage alone does not establish the branch. |
| Mini Condenser | Select the probe/illumination mode, Nanoprobe or Microprobe. | Preserve the current magnitude unless a deliberate tuning operation changes it. Existing mode-specific polarity rules remain in force. A change can require downstream focus revalidation. |
| Mini Condenser and Objective | Jointly form the probe. | Account for both fields, their geometry, and the incoming beam. Do not infer a focused probe from a mode label. |
| Objective | Control whether the probe is focused on the specimen. | Evaluate focus at the actual specimen entrance/reference plane and report which plane is used. Preserve defocus when the user deliberately perturbs a lens. |
| Probe Corrector | Correct aberrations while maintaining the required optical conjugacies. | For the specified STEM Nanoprobe branch, compare the source-image waist between TL22 and TL21 with the TL12 output waist downstream of AC. Measure the angular distribution at each waist with original current weights. Microprobe requires its own source-image and pupil-plane survey. |
| AC Scan / DP12 | Two physical scan windings also supply the DP12 alignment drive. | For the specified STEM Nanoprobe branch, set the common X/Y scan pivot downstream of the lower winding, at the TL12 output waist and the specimen-referenced front focal plane of Mini Condenser plus Upper Objective. Measure the two-axis scan response there and at the specimen. Scan and alignment share the physical coil limits. |

The stored presets can assign several controls together. Their values are
reference starting points, not independent guarantees for each role. AC/Descan
calibration changes scan-drive response matrices, not these illumination lens
excitations. Descan remains a control channel on the existing shared
Image/Diffraction physical deflector; it is not an extra magnetic element.

### Operational fine focus in three-condenser Nanoprobe Normal

For the +50 V joint finite-element working point, use the 100 µm diameter
C2 aperture and the same 100,000 original positive-current tip particles in
each comparison. Propagate through the physical apertures at their installed
positions and retain measured changes in transmission.

During probe tuning, **any installed hexapole supply of the probe corrector**
may be varied independently or in combination. The joint field with the HPC
windings includes sources for HP1, HP2, HPC X/Y and HPol X/Y. The older
1.743 µm working point is its zero-HPC reference. See the
[hexapole inventory](NANOPROBE_NORMAL_CROSSOVERS.md#how-hp1-and-hp2-affect-convergence).
Apply each trial excitation to the winding and magnetic-material model.
After each change, refocus with the installed
objective supply driving both objective packs together. Keep the aligned
condenser, ADL, transfer-lens, Mini Condenser and scan settings fixed during
this corrector comparison. An intentional change to the illumination or
transfer match starts a new calibration.

Minimize the **full transmitted beam's native D95 at the specimen upper
surface**. Retain the two intermediate waists near 1220 and 1439 mm and
check their physical intervals and conjugacies using the
[crossover guide](NANOPROBE_NORMAL_CROSSOVERS.md). Record the signed beam
width slope from the C3 lens exit to the ADL entrance. Outward motion was a
proxy for stronger ADL excitation. Evaluate the actual ADL winding current
and magnetic field when comparing illumination settings. Both converging and
diverging segments are admissible if the gap contains no crossover.

Measure alpha95, the populated aperture-rim angles, the RMS waist, total
current and current within a **50 nm radius** of the fixed axis as
diagnostics. A second disk of the same radius about the current centroid
can quantify the effect of a lateral shift. These measurements have no target
values in this size optimization. In a driven scan calibration, record the
centroid and incidence response to the same X/Y commands before and after
the change.

## Physical two-condenser Nanoprobe requested on 2026-11-12

The `C2` assembly is a distinct physical column with C1 and C2 as its only
main condenser lenses. Its installed Mini Condenser and Objective remain
downstream focusing components. This assembly has no C3 lens, C3 aperture,
Adapter Lens or probe-corrector relay/multipoles in its resolved layout.
Disabled compatibility objects in a saved software state do not represent
installed hardware and must not contribute a field or mechanical stop.

This configuration has one Nanoprobe illumination branch. Normal and Large
angle are C3-dependent choices and are not applicable to it. A retained range
value from a three-condenser state must not select a second two-condenser
Nanoprobe preset, re-enable absent hardware or block the valid mode pair.
Keep physical two-condenser installation separate from disabling excitation
on an installed C3 in a three-condenser column. In particular, the latter's
C2 compensation and saved C3 strengths do not define this assembly.

Match the incident optics using only installed elements. C1 retains its
source/current and dimensionless microscope spot-size roles; C2 controls the
incident convergence in conjunction with the actual upstream source image.
Adjust the Mini Condenser and Upper Objective fields deliberately to focus
directly on the specimen upper surface. A separate bounded source-image waist
between Mini Condenser and that surface invalidates the direct-focus branch.
Do not impose a C2-to-C3 waist, a C3-to-ADL condition, corrector-internal waist
counts or a TL12 output waist downstream of AC when those components
are absent. Any driven AC shift/tilt assessment remains a separate measurement
of the installed scan coils and downstream combined field.

Use an assembly-specific Nanoprobe working point. A three-condenser or
probe-corrected preset is at most a declared search seed, not a calibrated
two-condenser default. Diffraction is an independent downstream projector
selection; its lens strengths cannot establish or repair incident probe focus.
Apply a C2 Nanoprobe/Diffraction pair only through compatible installed
devices, preserving source law, absolute weights and physical interception.

The initial focus-only request introduced no numerical convergence-angle or
probe-size pass threshold. The subsequent request on 2026-11-12 requires
current-weighted alpha95 of at least 10 mrad, with a 70 um default C2 aperture
diameter for this two-condenser Nanoprobe working point. Keep the earlier
2.2 mrad, 100 um input as a historical comparison, not a qualifying high-angle
default. Change installed condenser working points explicitly; retain the
source distribution and absolute aperture losses. Do not obtain the larger
angle by silently broadening emission or counting zero-current support rays.

Report the actual current-weighted alpha95, angular RMS, native
D95, RMS radius, absolute current and source/surviving/effective populations.
Locate the native D95 minimum and RMS minimum independently in executed
through-focus observations; an RMS covariance estimate alone does not certify
D95 placement. Use the actual sample centre and active thickness to determine
the upper surface, rather than a historical module coordinate. The common
minimum source budget and numerical refinement rules below still apply.

## Microprobe basic tuning requested on 2026-11-11

Start from the current, explicitly identified instrument configuration. Reverse
the Mini Condenser polarity relative to Nanoprobe while preserving its current
magnitude for the initial comparison. The software's wiring convention is
positive for Microprobe and negative for Nanoprobe; these signed values are not
an OEM current-sign calibration. Keep this mode selection distinct from the
C3 Normal/Large angle branch.

Use C2 and C3 jointly to change the specimen-plane illumination diameter while
retaining approximately parallel incident electrons. Initially keep C1, Mini
magnitude, Objective, corrector drives, source law and physical apertures fixed.
Report any subsequent adjustment to them as a separate design step. Do not
replace this starting point with a historical Microprobe preset implicitly.

Measure native current-weighted D95, RMS radius, mechanical alpha95 and angular
RMS at the actual specimen upper surface. Quantify the local axial change of
the illumination area and distinguish radial vergence from azimuthal rotation
in the Objective immersion field. A small canonical-angle residual or zero
radial curvature alone does not establish mechanical parallelism. Microprobe
parallel illumination does not require a Nanoprobe-sized waist at the specimen.

Recheck the internal corrector source images when C2/C3 change. The existing
precise CEOS-matching contract below concerns STEM Nanoprobe; a Microprobe
illumination candidate does not inherit that qualification or the historical
Microprobe crossover baseline merely because its area and angle are suitable.
Keep a basic parallel-illumination candidate distinct from a fully matched
corrector operating preset. The initial measured family is recorded in the
[Microprobe study](development/microprobe_20261111/README.md).

The subsequent [Microprobe conjugacy study](development/microprobe_match_20261111/README.md)
uses mode-specific source-image counts and independent native pupil/steering
measurements. A parallel TEM source image upstream of the specimen is distinct
from the direct-to-specimen Nanoprobe focus requirement. A source-image waist,
an aperture conjugate, and a driven virtual pivot are different optical objects.
Do not impose either the Nanoprobe AC-waist rule or a historical Microprobe
reference count on a newly measured Microprobe family. Distinguish bounded
two-dimensional source images from radial-only caustics and retain unresolved
directional evidence explicitly. Matching paraxial conjugacy does not establish
complete hexapole aberration correction.

### Probe corrector in conventional TEM illumination

Qualify the present nearly parallel TEM family by its incident illumination:
diameter, mechanical angular distribution, spatial uniformity, local direction,
physical interception and shift/tilt response. The pre-specimen probe corrector
acts on the illumination/probe-forming channel; a post-specimen image corrector
acts on the TEM imaging channel. See the manufacturer descriptions of
[DCOR/ASCOR](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor)
and [CETCOR](https://www.ceos-gmbh.de/en/produkte/temcorrectors/cetcor).

There is no established universal TEM hexapole excitation ratio for this model.
The measured 10% family is an empirical illumination candidate, not an OEM
setting or proof of active spherical-aberration cancellation. Do not reject a
conventional TEM illumination point solely because it lacks STEM Nanoprobe
aberration qualification. Conversely, acceptable D95 and alpha95 do not prove
uniform illumination or cancellation of nonlinear aberrations. FEI's
[multipole-corrector disclosure](https://patents.google.com/patent/US20210159044A1/en)
identifies unequal hexapole excitations, beam radii and asymmetric transfer
optics as possible sources of uncancelled aberrations.

The [Cooper et al. holography experiment](https://doi.org/10.1063/1.3195088)
did not use its installed probe corrector, but also used a special Lorentz lens
configuration with the objective and third condenser lens off. This is not a
universal ordinary-TEM prescription. A statement that a corrector was not used
does not specify that every adapter/transfer lens was de-energized. Preserve
the physical relay and distinguish fixed-control multipole comparisons from
separately retuned illumination candidates. New finite-source comparisons must
retain at least 1000 original positive-current particles and all absolute losses.

## Transverse field sampling during corrector adjustment

Before interpreting an asymmetric probe as an optical alignment or corrector
error, compare the current-weighted centroid, the brightest patch and the full
two-dimensional landing map at a fixed specimen plane. Hold the emitter, stops,
ordinary lenses, corrector currents and source IDs fixed for the numerical
comparison. Use an explicitly recorded transverse interpolation method. In the
registered CESCOR vacuum bore, four-node cubic interpolation on the 0.30 and
0.15 mm grids reproduced the same pattern. Bilinear interpolation on the
square grid gave one side of the hexapole pattern excess current.

For a nominal threefold field, compare the current map with its 120° rotation
about the installed axis. This tests equality between the three directions.
The third angular harmonic of the measured landings and the fitted A₂
quadratures measure different properties and must be reported separately.
Keep physical asymmetries in the working field. A half-turn of the registered
vector field or an HP/HPol-off trace at fixed relay currents can locate a
suspect asymmetry without changing the operating point.

The earlier [HPol calibration and focal-plane measurements](development/nanoprobe_si110_20261122/fei-corrector-a2/README.md)
used 50,000 original tip particles. The current Nanoprobe comparison uses the
source count specified under operational fine focus above.

For A₂ tuning, measure both quadratures against independent changes of the
two fine winding currents using the validated interpolation. Check the
selected currents against the original settings at a finer field pitch.
Record the fit over the full transmitted cone and the selected imaging cone,
the specimen D95, convergence, absolute enclosed currents and axial focus.
Changing the material mesh tests a separate source of error. Any residual
smaller than the variation with mesh or fit aperture is specific to the
quoted model. [CESCOR interpolation controls and measured landing maps](development/cescor_direct_probe_20261121/README.md#why-the-bright-patch-appears-off-axis)

## Current-control families requested on 2026-10-29

Joint illumination design must test a family of settings, not just one narrow
probe. Gun Lens and C1/spot-size controls should change transmitted current
while causing only small changes to specimen alpha95 and native-D95 waist
position. Measure isolated control changes with other inputs frozen first.
Then record any deliberate coupled compensation as a separate, explicitly
calibrated family. Do not silently refocus an isolated perturbation or present
a compensated response as passive optical insensitivity.

Keep C2/C3 as the primary convergence-angle controls. Normal uses active weak
C3 without a C3-to-ADL crossover; Large angle uses stronger C3 and the required
additional crossover. For installed finite-length lenses, measure both crossover
intervals from the upstream lens exit to the downstream lens entrance. Normal
requires exactly one two-dimensional C2-to-C3 source image in that
open gap. Report waists inside the lens bodies separately. Measure the illuminated
C2 aperture radius alongside the specimen angular distribution when the two
condenser supplies change. The 100 µm value specifies its diameter. Count
any interception at the physical foil, allowing zero intercepted current.
An underfilled aperture may leave its nominal edge without a measurable
incident direction. Validate corrector internal source images independently.
For the specified Nanoprobe setting, TL12 must form its output waist directly
at the common scan pivot downstream of the lower AC winding. Check for a
separate earlier waist from the entrance of TL12 through the scan assembly.
On the requested operating branch, each round electromagnetic lens may form
zero or one two-dimensional beam waist. Retain every measured minimum in the
diagnostic, including those that disqualify a branch. Attribute a waist using
the incoming convergence and the overlapping focusing fields. A waist can lie
inside a downstream component's housing. Dipole deflectors translate the
centroid position and direction. Stigmators correct differential focusing in
the transverse axes. Neither receives a separate round-lens waist assignment.
Check both transverse widths before counting a projected line crossing.
Each AC station applies a common first-order transverse deflection. Changing
the relative drive moves the effective pivot without moving the windings.
At the target plane, opposite commands along each scan axis must leave the
centroid stationary and generate two independent tilt directions. At the
specimen, report both the resulting displacement and the mechanical change
in incidence angle in the objective immersion field.

If excitation changes do not provide the requested Gun Lens current control,
assess physical axial placement with the full electrostatic field, mechanical
apertures and vacuum geometry updated consistently. Explicitly state whether
the Gun Aperture stays fixed or moves with the electrode. Do not use a field
offset alone to represent moving a metal electrode. On 2026-10-29 the user
explicitly allowed the C1 aperture size as an additional design variable.
Record that change and its physical interception separately from a fixed-size
Gun Lens perturbation. Keep the selected 100 um C2 aperture diameter and the
source distribution fixed unless separately authorized.

Later on 2026-10-29, the user explicitly authorized changes to the original
electron-source emission parameters for probe-tail research. Keep the original
source as a named comparison and prepare a new input snapshot for each source
law. The October source-law study requested isolated angular-distribution and
energy-width changes, with separate lens compensation to restore its reference
alpha95 and surface waist. For a new source study, propagate the complete
changed source and report physical losses. Do not remove selected
specimen-plane particles or present that deletion as a new source calculation.
Narrowing the source angular distribution at unchanged emitted current and
source size changes the assumed source brightness; distinguish this design
assumption from physical aperture losses and from measured gun performance.
The 100 um C2 aperture constraint and minimum source sampling still apply.

For Gun Lens and C1 sensitivity studies, report actual currents, angular
changes and native D95 waist displacements for each control perturbation.

The AC common tilt centre must still match the downstream front focal plane.
Two-coil position-plus-tilt compensation can produce small specimen angle
changes without satisfying that condition. Report such compensation separately;
it must not replace the original conjugacy check. Declare the coordinates of
the local transfer map with blocks A, B, C and D. A canonical/Larmor
front-focal relation requires D near zero and full-rank B in both directions;
report the separate mechanical-angle response at the specimen. A claim of
mechanically pure tilt-to-shift requires a small mechanical D residual, not
just canonical cancellation. A source-image conjugate with B near zero is a
different condition. Do not require B and D to vanish at the same input plane:
that would make the transfer singular, and a measured finite-source waist
does not impose that simultaneous condition. Finite-source and driven scan
measurements remain necessary when nonlinear aberrations are significant.
For this Nanoprobe setting, locate the common plane downstream of the AC
windings. Match the static TL12 output waist and the driven scan pivot to
the Mini Condenser plus Upper Objective front focal plane referred to the
specimen upper surface.
Locate the waist with scan and wobble off. Locate the focal plane
independently with both axes of a point-tilt perturbation in the combined
downstream lens field. Repeat the transfer in the complete installed field
to quantify overlapping upstream fringes. Verify the common tilt center
separately by driving the two deflectors. A mechanical midpoint, a static
waist or a root found only in the full column cannot establish all three
conditions. Report plane offsets, numerical resolution and both canonical
and mechanical angular residuals in the objective immersion field.

The [objective focus and STEM raster study](development/cescor_direct_probe_20261121/scan-focus-calibration.md)
holds the scan calibration fixed during objective fine focus. It measures how
much the specimen pixel scale, scan rotation and scan-dependent beam angle
change. Those measurements address the operating range around a reference
alignment separately from the nominal three-plane match.

Thermo Fisher's [Krios training packet, pp. 13-15](https://cryoem.jhmi.edu/wp-content/uploads/2022/07/Krios-Customer-Packet.pdf#page=13)
illustrates C1 source-image changes, aperture interception and condenser zoom.
The primary [STEM experiment by Radic et al.](https://doi.org/10.1017/S143192762002440X)
describes discrete spot numbers and coupled C1-C2 imaging: larger spot numbers
reduce the maximum admitted current in the discussed FEI configuration.
These sources support the operating interpretation, not an OEM calibration
of this simulator or a universal current ratio between neighbouring indices.

During condenser zoom calibration, use the C2 and C3 lenses as the primary
controls for the specimen convergence semi-angle. Measure the combined Mini
Condenser and Upper Objective focus, corrector entrance/output matching and
aberrations separately. During a corrector-only adjustment, keep the condenser
settings fixed and report the measured angular distribution alongside D95.

For a C2/C3 zoom comparison, measure the moving C2-to-C3 source image
and the first downstream source image formed with C3 and ADL energized.
Report their axial displacements separately. Keep the corrector drives fixed
while changing the condenser pair, then use the installed shared Objective
drive to place the final Mini Condenser / Objective waist at the sample
entrance. Verify this placement before comparing specimen images. An
on/off pair for the main hexapoles at each setting measures their signed
C3 contribution. Report this coefficient separately from the probe current
and spot size, which can change with semiangle. See the [20 mrad comparison](development/probe_zoom20/README.md).

## Required crossover topology

For C3 + Probe Corrector in STEM Nanoprobe:

| Range | C2 to C3 | C3 to Adapter Lens (ADL) |
| --- | --- | --- |
| Normal | A crossover is required. | No crossover; C3 is weak in the project's convention. |
| Large angle | A crossover is required. | A crossover is required; C3 is strong in the project's convention. |

For the three-condenser Normal setting, use the [Nanoprobe Normal crossover and conjugate-plane guide](NANOPROBE_NORMAL_CROSSOVERS.md) for each downstream interval, the scan pivot and the terminal specimen focus.

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

The corrector-specific locations and C3 branch requirements in this section
apply to the installed three-condenser probe-corrected assembly. They do not
add missing hardware or crossover requirements to a physical C1/C2 column;
the direct Mini Condenser-to-specimen focus requirement still applies there.

The Normal-mode C3-to-ADL interval must remain free of crossovers.
Measure the internal source image between TL22 and TL21. The November 21
clarification places the TL12 output waist at the virtual tilt plane downstream
of the lower AC winding. Measure both waists with executed finite-source
trajectories and locate the pivot independently with applied scan commands.

On 2026-10-28 the user confirmed keeping the Normal entrance while matching
the corrector interior to the CEOS optical principle. Device names need not
map one-to-one to the manufacturer's schematic. In physical terms, C3 is weak:
there is no additional crossover between C3 and the next optical element.
C3 remains active, and its field together with downstream fields forms the
next crossover. Preserve its finite profile and overlapping fringes; weak
excitation is neither zero field nor an independently acting thin-lens rule.

The internal acceptance pattern is exactly one resolved, two-dimensional
source-image waist between TL22 and TL21 and a direct TL12 output waist at
the downstream pivot. Count and report all other actual waist brackets
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

At the specimen, alpha95 is the semi-angle enclosing 95% of the transmitted
current about its weighted chief direction. For a physical rim measurement,
identify electrons by their radius at the C2 aperture, state the radial band,
then measure their directions at the specimen. An underfilled rim may have
too little current for a measured direction. Report the two angle measurements
separately. The full-beam surface D95 selects the working point.

The [physical-pupil measurement study](development/FINITE_FIELD_OPTICAL_ACCEPTANCE_CONTRACT_2026-10-27.md)
documents a separate nominal-energy diagnostic. Its script,
`scripts/measure_optical_acceptance.py`, brackets mechanical interception
with zero-current rays and reports the physical-source filling separately.

## Unified D95 probe-waist definition

At each saved plane, compute the current-weighted centroid of the arriving
electrons. D95 is twice the 95th weighted percentile of their distances from
that centroid. Keep original source IDs, currents, energies and interception
records. Calculate D95 directly from the transverse landing coordinates.

After objective fine focus, bracket the minimum of full-beam D95(z) on both
sides of the specimen upper surface. Report the measured offset, axial step
and any unresolved plateau, competing minimum or change in population.
Extension through the nominal sample in vacuum measures optical focus.
The axial step measures resolution along z. The transverse coordinates and
current weights determine the lateral diameter. Locate RMS minima separately
and identify the population and centering convention for each reported current.

## Minimum source sampling for subsequent physical tests

On 2026-10-29 the user required **at least 1000 original positive-current
source particles for all subsequent finite-source physical calculations**,
including quick tests, sensitivity studies, tuning prescreens and finite-current
scanning diagnostics. The general starting count is **1024** when the selected
source sampler supports it. The current three-condenser probe comparison
uses a fixed **100,000** emitted particles at every setting. A source-count
convergence study uses separate, explicitly identified runs.

Explicitly record any numerical count change in a new input snapshot. Preserve
the source law, original weights and physical aperture interception; do not
duplicate rays or add zero-current probes to meet the minimum. For explicit
source tables, keep the declared count and table dimensions consistent.
Report the actual emitted positive-current population, surviving population
and effective weighted sample size separately. A clipped population can be
smaller than the emitted count; increase the source budget when its statistical
support is insufficient, without renormalizing the transmitted current.

The 1000-particle minimum is an execution baseline, not a convergence claim.
Further source-count refinement remains necessary before accepting the
stability of native D95, alpha95, its focus position or transmitted current.
Historical 193/769-particle results retain their original inputs and evidence
limits; they are not sufficient sampling for new working-point qualification.
Small chief-response measurements and pure-array/mocked software regressions
are separate diagnostic categories and do not substitute for finite-source runs.

## Qualification boundary and this maintenance change

The October 27 Finite-Field Normal preset had a fixed 49-particle, approximately
10 mrad focus record at 0.05 and 0.025 mm integration steps. Its record explicitly
marks source-sampling convergence and condenser topology as `NOT_TESTED`.
See [the recorded scope](development/FINITE_FIELD_NANOPROBE_DEFAULT_2026-10-27.md).
That focus record does not prove either crossover condition above. Historical
legacy-field topology results do not qualify the shared-potential executor.

The scan-routing, gun-exit, validation and reference-label repairs do not retune
the stored lens strengths, change apertures, alter source sampling or change the
integration accuracy. The October maintenance record documents that preset
and the scope of its topology checks.

## Fixed-target geometric probe correction

Use the specimen upper surface as the physical target. At that plane, report
the full-beam D95 and absolute current within the fixed-axis 50 nm radius
disk. Label any additional centroid-centered disk. Compute each disk current
by summing the weights of its arriving electrons. Record the center, beam
centroid, and total transmitted
current so a centroid shift can be distinguished from a change in diameter.
The [November 1 particle calibration](development/PROBE_CORRECTOR_PARTICLE_CALIBRATION_2026-11-01.md)
records the measurement conventions used for that archived experiment.

Use `scripts/measure_crossover_cs.py` with a complete vacuum instrument
snapshot for an independent geometric pencil and finite-source waist survey.
Each coefficient belongs to its named plane, incident reference family, angular
chart and fitted pupil radius. A local crossover value includes all preceding
fields. Evaluate HP1/HP2 attribution by changing their real excitations and
retaining the common input and target. Intervals without a resolved 2D waist,
intercepted pencils and failed holdouts carry no accepted coefficient. The
gun-to-exit accelerating region requires its own valid emission-angle chart.

### Joint central-current and D95 calibration

For each corrector setting, use the objective supply to minimize native
full-beam D95 at the same specimen surface. Compare disk currents and D50
there using the same emitted population and the same disk definitions. In a
through-focus plot, label the axial positions of the D95 and RMS minima and
the disk-current maxima separately. Give the axial sampling and brackets for
resolved extrema. Determine the sign of the physical sextupoles' spherical
aberration contribution with a separate reference-pencil measurement.

### Fixed-source HP tuning

Use the common 100,000-particle source for the current Nanoprobe comparison.
Keep its IDs, weights, initial phase space, energy and clocks identical
across settings. For a separate measurement of the main pair's contribution
to specimen Cs, compare HP1/HP2 off and on with the other supplies fixed.
Report the fitted coefficient separately from the objective-refocused size
and currents used to compare probe settings.

### Mini Condenser and Objective joint focus

During corrector adjustment, hold Mini Condenser excitation fixed and find
the finite two-dimensional waist downstream in the combined field. Measure
D95, RMS radius and the fixed-axis 50 nm disk current at that waist and separately
at the specimen upper surface. Record the radial
bracket, directional extent, through-focus extrema, convergence angle and
absolute transmitted current. A separate bounded waist upstream of the
intended image invalidates direct placement on this branch.

Use the measured placement error to change objective excitation, propagate
the same physical source again, and relocate the waist after every optical
change. The installed operating control drives both objective lobes.
Experiments changing only the upper field amplitude must be labeled as
detached model perturbations. The command
`scripts/measure_probe_crossover.py TRIAL REPORT.json` observes a completed
fixed-source calibration trial using its own verified original checkpoint.
An ambiguous or unbounded region gives an inconclusive result.

For targeted adjustments from the 300 kV R2 particle working point, use the
[probe-corrector channel rules and signed coupling matrix](development/PROBE_ABERRATION_CHANNEL_RULES.md).
Each proposed control combination requires a new propagated measurement at
the working point, including the protected aberrations and the native probe.

### Probe image preservation

For the staged probe-corrector work, archive each generated current image
with a distinct stage name, its raw absolute-current kernel, source and input
identities, sample-plane metrics and crossover evidence. Append an image
history entry and retain earlier images. Curated evidence belongs in the
repository for Git delivery. See the
[stage 1 and 2 archive](development/probe_stage12_20261101/README.md).
