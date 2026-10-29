# Illumination control investigation and maintenance repairs

Date: 2026-10-29. Status: **NOT_QUALIFIED**. No default preset was installed.
The completed calculations below are classical vacuum-probe diagnostics, not
specimen images or coherent-wave calculations. Numerical proposal searches,
isolated control perturbations and nominal focus compensation are recorded
separately. The user selected sensitivity reporting without acceptance
thresholds and authorized C1 aperture size as an additional design variable.

## Software repairs and verification

Four repairs preserve physical inputs and source
weights:

1. `direct_focus_topology.py` retains unresolved two-dimensional waist evidence.
   A narrow radial endpoint crossing no longer hides an unbounded directional
   waist. The genuine common surface-boundary exception requires the actual
   directional covariance condition; entrance minima and directional
   cancellation are rejected.
2. `performance_accuracy.py` uses the population at the requested saved plane.
   Particles stopped later no longer disappear retrospectively from that
   plane's current, angle or size statistics. Endpoint survival is independently
   checked against the source and physical stop ledger.
3. The scan calibration panel restarts the same pending operation after a
   readout change, preserving an explicit user request. Disabling automatic
   readout does not cancel an explicit operation. Generation and physical-input
   guards remain active.
4. A bounded explicit AC calibration now runs with raster switches off when
   the physical coils lie inside the observation domain. Previously this
   request silently returned before reaching the explicit-capture branch.
   Ordinary unused-scanner preparation and upstream-only cutoff requests still
   skip calibration; the repair does not turn on raster scanning.

The validation environment now declares its existing high-precision test
dependency, `mpmath`, and CI includes the previously omitted joint-probe and
experiment-observation test scopes. Routing fixtures provide the unified
execution receipts required by the current production guard. A positive
finite-region fixture now actually has a common two-dimensional surface waist;
the former astigmatic fixture remains as a negative regression.

**656 unique targeted tests passed** across the affected policy, diagnostics,
waist, observation, benchmark and GUI tests, including all 50 tests of the
final unified calibration-control module. This is not a full-CI pass claim.
No broad coherent suite was executed. New coherent calculations still require
a verified usable CUDA backend.

## Meaning of current controls

Thermo Fisher's [Krios training packet, pp. 13–15](https://cryoem.jhmi.edu/wp-content/uploads/2022/07/Krios-Customer-Packet.pdf#page=13)
shows C1 source imaging and aperture interception within a condenser zoom
system. [Radic et al., original STEM research](https://doi.org/10.1017/S143192762002440X)
describes discrete spot numbers and coupled C1–C2 imaging; increasing the spot
number reduces maximum current in the discussed FEI configuration. This supports
a dimensionless operating index, not a literal specimen diameter, a universal
current ratio or a calibrated index table for this simulator.

The simulator's C1 aperture is at **445 mm**, upstream of the C1 magnetic lens
center at **500 mm**. Its original radius is **2 mm**. Gun Lens changes can alter
the population admitted there. C1-lens current control must instead be checked
against its actual upstream fringe and downstream apertures, including the
fixed **100 micrometre diameter C2 aperture at 765 mm**.

The unchanged flat-tip source law has a 5 nm emitting-region FWHM (the stored
parameter is named virtual-source FWHM), 1 mrad per-axis pre-truncation emission
slope scale, 3 mrad radial cutoff, 0.3 eV nominal emission energy and 0.3 eV
energy-spread FWHM.
All measurements retain the emitted IDs and absolute weights. No transmission
normalization, tail deletion or source retuning was used.

The 1 mrad setting acts at low-energy emission, not at the 300 keV gun exit.
The emitter, initial-momentum construction, parameter descriptions and source
editor consistently use that meaning. At 0.3 eV its small-angle transverse
energy scale is approximately 3e-7 eV per axis before truncation. This very
narrow initial transverse momentum is one relevant input to the underfilled
C1 aperture; the lens field and emitting-region width also matter. There is
no evidence here of an angle-unit implementation bug, and its meaning was
not silently changed to an exit-angle setting.

## Gun strength and physical placement with the original aperture

Six actual 769-particle gun observations used the same emitted population.
Every case transmitted all 769 particles and **10 microampere** through C1.
The table reports the radius about the physical aperture axis; the angle is at
the gun exit and must not be confused with the specimen convergence angle.

| Gun voltage relative to extractor, kV | Lens center, mm | Maximum C1 radius, micrometre | Gun-exit alpha95, mrad |
| ---: | ---: | ---: | ---: |
| 1.14 | 18.0 | 0.758545 | 0.000238818 |
| 1.20 | 18.0 | 0.799455 | 0.000300018 |
| 1.26 | 18.0 | 0.839367 | 0.000359912 |
| 1.20 | 15.0 | 1.467990 | 0.001336947 |
| 1.20 | 16.5 | 1.153730 | 0.000851378 |
| 1.20 | 18.0, repeat | 0.799455 | 0.000300018 |

The gun aperture remained at 24 mm. Moving the lens updated its metal geometry,
vacuum bore, assembly and field consistently. The tested 15–18 mm interval
covers the available placement range with at least 1 mm body gaps; the strict
noncontact interval is 14 < center < 19 mm. These trials do not establish a
global impossibility for other geometries or voltages.

A separate cheap axis-variational screen combined two closed Laplace solutions
using their fixed-geometry voltage linearity. Twelve mathematical voltage
points from -3.9 to 20 kV relative to the extractor were considered; this is
**not a hardware voltage rating**. The largest screened source-support radius
was about 12.56 micrometre after the emitted-energy check, still far below the
original 2000 micrometre aperture radius. At 1.2 kV its predicted 0.799700
micrometre maximum agreed with the actual 0.799455 micrometre result. This is
candidate-screening evidence only, not fresh finite-particle voltage validation.

## Isolated column sensitivity with original apertures

These seven native finite-field runs used 193 particles and a maximum step of
0.2 mm. No control was automatically refocused. All retained **10 microampere**.
Waist offsets are the smallest sampled native D95 positions, not interpolated
continuous minima. Sparse outer brackets limit the two C2 focus observations.

| Isolated perturbation | Surface alpha95, mrad | Surface D95, nm | Observed waist offset, nm | Neighbor bracket, nm |
| --- | ---: | ---: | ---: | --- |
| Baseline | 29.999861 | 2.209807 | +15 | +10 to +20 |
| C1 -5% | 29.381806 | 2.391682 | +25 | +20 to +30 |
| C1 +5% | 30.176507 | 2.039220 | +20 | +15 to +25 |
| C2 -1% | 29.211957 | 7.810654 | +100 | +95 to +1000 |
| C2 +1% | 30.792404 | 6.565260 | -100 | -1000 to -95 |
| C3 -1% | 30.043586 | 4.585047 | +70 | +65 to +75 |
| C3 +1% | 29.955711 | 2.967594 | -35 | -40 to -30 |

This supports C2/C3 as the angle-and-entrance-matching pair. It does not show
independent C1 current control. A wider first-order screen of C1 from 0 to 100%
also remained underfilled at C2, with a maximum sampled radius around
26.75 micrometre. That screen is not a finite-source qualification.

## Separately compensated original-aperture focus

With C2/C3 fixed, the objective-only proposal changed Upper Objective from
29.293344328471793% to **29.293371898678036%**. The independent actual run with
the original **769-particle** population and **0.05 mm** maximum step measured:

| Observable | Result |
| --- | ---: |
| Specimen upper-surface alpha95 | 29.769629 mrad |
| Surface D95 | 2.183034 nm |
| Sample current | 10 microampere |
| Smallest observed native D95 | 2.173276 nm |
| Observed minimum offset from upper surface | -0.2 nm |
| Executed neighbor bracket | -0.4 to 0 nm |

The through-focus observation spacing was 0.2 nm. This places an observed
minimum near the upper surface at this numerical setting; it does not prove
source-count or integration-step convergence. In particular, D95 differs
between the 193-particle and 769-particle runs. The historical 0.7 nm hard size
cap is not applied under the latest recorded optical contract.

## Explicit C1 aperture design trial

The user then authorized changing C1 aperture size. A **0.3 micrometre radius**
was selected as a diagnostic design candidate from the saved physical aperture
arrival distributions. The existing hard-edge aperture model was then rerun;
this is not post hoc particle deletion. The source law, C2 aperture and all
other geometry were retained. Source count was 193 and maximum step 0.2 mm.

All five isolated cases used the same frozen objective-only proposal above.
Only the named gun or C1-lens strength was perturbed; none was refocused.

| Case | Sample current, microampere | Surface alpha95, mrad | Surface D95, nm | Observed D95 waist offset, nm |
| --- | ---: | ---: | ---: | ---: |
| Base, Gun 1.20 kV | 6.476684 | 15.554154 | 1.190080 | -30 |
| Gun 1.14 kV | 6.943005 | 15.509996 | 1.209062 | -25 |
| Gun 1.26 kV | 6.062176 | 15.803374 | 1.194879 | -25 |
| C1 -5% | 6.476684 | 15.172993 | 1.041082 | -25 |
| C1 +5% | 6.476684 | 15.759184 | 1.343608 | -35 |

Each waist has a 10 nm neighbor bracket at the 5 nm sampling used here.
Gun current control is now observed through C1 interception. C1-lens current
control is still absent. The reduced D95 cannot be presented as improvement
at 30 mrad: this aperture-only change reduced the angle to about 15.5 mrad.
The submicrometre opening remains a model design candidate, not a qualified
instrument aperture specification.

The first explicit C2/C3 compensation proposal used **C2 21.32039505593406%**
and **C3 5.044564335771807%**, leaving Mini Condenser and Upper Objective fixed.
Its actual run recovered **30.203924 mrad**, with **6.476684 microampere**,
surface D95 **2.654096 nm**, and an observed minimum at **-15 nm**
(neighbor bracket -20 to -10 nm). This is a separate jointly adjusted point,
not the isolated gun response or a qualified new default.

A subsequent actual local fit retained C3 and adjusted C2 to
**21.25079393298154%** and Upper Objective to **29.29337773651361%**.
At 193 particles and 0.2 mm maximum step it measured alpha95
**30.00000003 mrad**, D95 **2.333733 nm**, and **6.476684 microampere**.
The independent 0.2 nm native-D95 observation grid found its minimum at
**-1.0 nm**, with neighbor bracket **-1.2 to -0.8 nm**. This is a nominal
candidate fit, not passive insensitivity or proof of numerical convergence.

An independent **769-particle** count check retained every physical setting
and the same 0.2 mm maximum step. It measured **29.679247 mrad**,
**2.264675 nm** surface D95 and **6.345904 microampere**. Its smallest observed
D95 was 2.078591 nm at **+7.0 nm**, bracket **+6.8 to +7.2 nm**.
The count change alone moves the observed minimum by about 8 nm and changes
the transmitted current by about 2.0%. Source-count convergence is therefore
not established; the fitted 193-particle point must not be promoted to a default.

The final nominal point also received four **isolated** sensitivity runs at
193 particles, with its C2/C3 and Objective settings frozen. These are actual
native propagation, not compensation lookups:

| Case | Current, microampere | alpha95, mrad | D95, nm | Native minimum offset, nm | Neighbor bracket, nm |
| --- | ---: | ---: | ---: | ---: | --- |
| Nominal | 6.476684 | 30.000000 | 2.333733 | -1.0 | -1.2 to -0.8 |
| Gun -5%, 1.14 kV | 6.943005 | 29.792218 | 2.372425 | +1 | 0 to +5 |
| Gun +5%, 1.26 kV | 6.062176 | 30.646709 | 2.710748 | +15 | +10 to +20 |
| C1 -5% | 6.476684 | 29.415933 | 2.196790 | +20 | +15 to +25 |
| C1 +5% | 6.476684 | 30.340694 | 2.341168 | +1 | 0 to +5 |

Gun current varies by about +7.2%/-6.4% in these samples, with angle changes
of about -0.69%/+2.16% and observed waist displacements of +2/+16 nm from
the nominal observation. C1 current remains unchanged despite angle/focus
changes. No pass/fail sensitivity threshold is imposed. The perturbation
profiles are sampled more coarsely than the nominal dense profile, as their
explicit brackets show.

At the final nominal's actual C2-aperture plane, the maximum surviving ray
radius is approximately **25.415 micrometre**, below the fixed 50 micrometre
aperture radius. A proposed further C2/ADL rematching search was stopped before
any trial because its simplified one-axis operator omitted installed skew
terms. It produced no new candidate or negative feasibility proof. A faithful
joint search must retain those terms and all required corrector conjugacies;
the current data do not justify a calibrated spot-number/current table.

## Corrector and scan evidence

See the independent [internal-waist audit](ILLUMINATION_INTERNAL_WAISTS_2026-10-29.md)
for the actual frozen baseline capture. The selected Normal intervals have
one C2–C3 region, none between C3 and ADL, one central TL22–TL21 region and
one AC-region waist. The AC radial bracket is 1408.51–1408.52 mm. A broad,
shallow region near HP1 is retained in the inventory; it is not sufficient
evidence of another source-image crossover. These observations have not been
repeated for every new aperture and compensated control point.

The [CEOS original Figure 2 and accompanying text](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=3)
identify central and output source-image references and emulation of the
DP12 alignment deflector by weighted offsets on two scan coils. This does not
identify DP12 with every raster command's virtual pivot. A static source waist,
an image conjugate and a driven tilt pivot require separate measurements.
They cannot be inferred from the manufacturer's label or a projected-ray
crossing.

Nine native chief traces measured the existing two-coil compensation response.
For the 64 by 64, 0.02 nm pixel-center range, the linear prediction gives a
maximum corner mechanical tilt change of about **0.1003 microradian**.
This is not an executed finite-current scan. The actual distributed-coil
response was extended through a separately measured zero-drive static map;
no coil-support edge was imposed as a pivot.

The existing AC-region local map has a near-zero B image-conjugate candidate
around 1408.081 mm, while its D singular values remain around 11. This does
not establish a pure-tilt-to-shift front-focal plane inside that interval.
However, only the static output waist is required to be inside the AC region:
the user did not impose that location on the mathematical virtual pivot.
The downstream region must also be searched. Two-coil position-plus-tilt
cancellation at one reference plane can correspond to a pure virtual tilt
at another plane; the proper static transfer must demonstrate that relation.
Canonical and mechanical angles must remain distinct inside the objective
magnetic field. Therefore the nonzero AC-region D block alone is not evidence
that the requested overall architecture is impossible or has failed.

The downstream search subsequently found the current combination's virtual
pivot near **1584.32735 mm**, about 14.87265 mm before the upper surface.
An independent native **nine-chief** response run at the new clipped nominal
point verified a common two-axis position minimum at that executed plane,
with approximately 4.32e-7 nm position-response residual per 0.64 nm axis
command and full-rank angular response. This tiny computed residual is not a
claim of physical positioning accuracy. It is a local response at 0.2 mm
maximum step and 1e-6 rad perturbations; neither scale has been converged.
It supports the existing downstream virtual-pivot architecture. The separate
weak-objective front-focal proposals are not required by the absence of an
AC-region pivot and were not installed.

The new nominal point was also propagated with its actual finite-current
population at optical zero, four axis points and four corners of the requested
64 by 64, 0.02 nm raster's pixel-center range (edges at +/-0.63 nm).
The detached instrument received a newly executed native canonical calibration;
each fixed command then used the actual two physical coils. This was a
vacuum-probe 9-position diagnostic, not a 4096-pixel specimen image.

| Finite-current scan observation | Measured range or maximum |
| --- | ---: |
| Surface D95 | 2.332536 to 2.334928 nm |
| Surface alpha95 | 29.99999976 to 30.00000030 mrad |
| Sample current at every point | 6.476684 microampere |
| Maximum requested-position error | 0.000515101 nm |
| Maximum mechanical mean-slope change | 0.100357378 microradian |
| Surviving original particles | 125 of 193 at every point |

Every lost particle was intercepted at C1: 68 particles, carrying
3.523316 microampere. Scanning introduced no additional loss. Fourteen
original source, gun-exit, identity and weight arrays matched the nominal
input exactly at all nine points. Production dependency checks admitted
cached gun and upstream segments for the subsequent points; actual changed
finite-field transport restarted at 1375 mm. These observations establish
local static scanning behavior at the declared numerical setting, not a
dynamic-raster or source-count/step-converged scan qualification. No historical
0.7 nm size cap is applied.

The same chief linearization predicts approximately 336 nm D95 when applied
to the actual finite source, versus 2.21 nm in its actual propagation. The
position residual is strongly associated with third-order angular terms.
Consequently, first-order candidate fits cannot qualify finite-source focus,
corrector source images or scanning performance.

## Evidence and outstanding qualification

The compact [machine-readable summary](illumination_control_audit_20261029.json)
records completed observations. Raw input snapshots, original particle arrays,
receipts and local drivers are retained under
`outputs/illumination-control-20261029/`. These ignored local outputs are not
an archived or published reproducibility bundle. Historical input migration
retained the original digest and declared current implementation, path and
line-ending changes; historical solver identity was not forged.
After the final calibration-entry repair, unchanged nominal graphs were
explicitly imported into new `clipped-nominal-current-n193/` and
`clipped-nominal-current-n769/` input snapshots. Their migration records retain
both implementation identities and verify identical input graphs and external
dependencies. Historical observations retain their original identities;
the imports are not new physical executions or qualification results.

Outstanding items are C1/spot-size current control, the complete new
aperture/control family, actual Normal/Large-angle family topology, and
numerical/source-count convergence of focus, static corrector waists and
driven scans. The common downstream pivot and finite-current nine-point
response now have native evidence at one numerical setting; this does not
qualify every family member or a full dynamic image. A nominal focus or
matching selected waist counts does not establish the remaining conditions.
