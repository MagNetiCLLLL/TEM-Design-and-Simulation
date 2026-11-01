# Mini Condenser and Objective crossover calibration

## Fixed-source operating result

Setting A places the representative plane of the measured finite 2D waist at
the specimen entrance, **1599.1999975 mm**. At that exact plane the same 2,000
original particles deliver **1.575 µA inside a 50 pm radius**, **2.701 nm
D95**, **1.901 nm RMS radius**, **29.749 mrad alpha95** and **10.000 µA
transmitted current**. Setting C gives 1.855 µA with a wider 2.756 nm D95.
The fixed-source search did not establish a joint optimum of all three axial
criteria at the specimen. The chosen A configuration favors the smaller D95
among the verified high-current settings with a physically placed waist.

| Actual setting or adjustment | Waist Z relative to entrance (nm) | D95 at waist (nm) | RMS radius at waist (nm) | 50 pm current at waist (µA) | D95 at entrance (nm) | 50 pm current at entrance (µA) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J0, before the local paired-HP adjustment | −22.0 | 2.746 | 2.067 | 0.640 | 2.009 | 2.115 |
| A, after paired strength, balance and rotation adjustment | −5.5 | 2.701 | 1.901 | 1.575 | 2.533 | 1.555 |
| A, after objective placement | 0.0 | 2.701 | 1.901 | 1.575 | 2.701 | 1.575 |
| C, stronger core after objective placement | 0.0 | 2.756 | 1.932 | 1.855 | 2.756 | 1.855 |
| Near-zero C3 experiment, objective held | −112.1 | 4.356 | 2.901 | 0.020 | 9.265 | 1.300 |
| Same near-zero drive, D95 minimum first placed at the surface | −16.6 | 4.356 | 2.901 | 0.020 | 3.894 | 0.025 |
| Same near-zero drive, actual 2D waist placed with the common objective | 0.0 | 4.356 | 2.901 | 0.020 | 4.356 | 0.020 |
| Same near-zero drive, actual 2D waist placed with the upper field alone | 0.0 | 4.357 | 2.901 | 0.020 | 4.357 | 0.020 |

Each correction round locates the actual combined-field image and measures
its local width. It replays the original gun-exit packet
through its own optics, uses its own upstream checkpoint to survey
1522.50–1602.55 mm, and adaptively refines the native observation grid to
0.1 nm. The image requires convergence followed by divergence in both
transverse covariance directions, stable particle membership and measured
growth on both sides. The interpolated radial root is used only to place
additional measurement planes. Final numbers refer to executed checkpoints.

For A the radial sign transition lies between −0.1 and 0.0 nm, and the
two-direction finite waist region spans **−5.0 to +4.8 nm**. Zero placement
error refers to the sampled representative. With the objective and all other
settings frozen, the 50 pm current peaks at **1.615 µA at +2.6 nm**, whereas
D95 reaches its observed minimum of **1.962 nm at +23.2 nm**. The signed
RMS-minus-D95 focus offset is −23.2 nm with an executed difference bracket
of −23.4 to −23.0 nm. Halving the maximum propagation step from 0.050 to
0.025 mm leaves the representative plane, the 1.575 µA disk current and
that offset unchanged. D95 changes by 0.000017 nm and RMS radius by less
than 0.000001 nm. Both step sizes used the identical frozen gun outlet.
Each equal-current particle contributes 0.005 µA.

A further iteration began from the placed A objective setting. Corrections
to the paired drive, relative HP1 gain and orientation, auxiliary hexapoles
and TL21/TL22 moved the free waists of D and E to −125.0 nm and +104.3 nm.
At the specimen before repositioning, D gave D95 = 7.060 nm and 0.005 µA
in the disk. At its own upstream waist the same fields gave 2.735 nm and
1.705 µA. After one measured objective correction the D waist lay at the
specimen, where D95 = 2.734 nm, RMS radius = 1.893 nm and disk current =
1.705 µA. E also landed on the specimen, with D95 = 2.746 nm, RMS radius =
1.933 nm and disk current = 1.435 µA. These iterations show why observing
only the stationary sample would misrank a promising change to the corrector.

An additional paired-hexapole search explicitly favored a shorter signed
RMS-minus-D95 focus separation. It reduced the absolute distance to about
0.5 nm, with a 6.853 nm D95 and only 0.465 µA at the waist. A less extreme
choice achieved about 20 nm, 2.920 nm D95 and 1.695 µA. The retained
operating setting uses smaller current-containment diameter at the placed
waist. The full table preserves both lower-separation alternatives.

Setting A uses HP1 = 4,634,567.145 m⁻³, HP2 = 5,347,687.020 m⁻³,
HP1 orientation = 0.01891965741 rad, HP2 orientation = 0 rad and common
Objective excitation = **29.2934367286%**. TL22 and TL21 are 53.5070985053%
and 59.7053035190%. At a fixed 0.55 µm launch radius, the matched reference
pencil gives **+6.486820 mm with both main hexapoles off** and
**−0.182829 mm with them on**, an effective change of **−6.669649 mm**.
At 0.1375 µm the on-state coefficient is −0.186943 mm. The full-pupil on
fit retains C5 = +196.227 mm and a 4.19 pm holdout RMS.

In the near-zero experiment, changing the upper field alone by the
equivalent of −0.0003622304 excitation percentage points places the
representative at the surface while preserving the complete lower field.
After this placement the independent full-pupil pencil gives C3 =
−0.00002340 mm and the smaller pencil gives −0.003837 mm. This upper-only
amplitude perturbation is a model experiment. Installed profiles use the
common Objective excitation, which scales both lobes. Refocusing the same
near-zero branch with the common excitation gives −0.00002552 mm and
−0.003839 mm at the two pupils.

[Through-focus figure](probe_joint_d95_20261101/placed-waist-focus.png) ·
[Native probe current map](probe_joint_d95_20261101/placed-A-incident-probe.png) ·
[Round-by-round measurements](probe_joint_d95_20261101/crossover-calibration.csv) ·
[Per-lens cumulative spherical aberrations](probe_joint_d95_20261101/lens-crossover-survey.csv) ·
[GUI survey file](probe_joint_d95_20261101/lens-crossover-survey.json)

The full A column includes 14 lens intervals from the gun exit through the
projector section. The Normal branch has one measured 2D waist between C2
and C3 and none between C3 and ADL. The mechanical Objective reference at
1599.2 mm cuts through the final 2D region. The isolated Mini-to-Objective
interval lacks a bounding divergent side. The separate combined-field row
extends across this reference and uses the measured target pencil at the
identical representative plane. C1 and C2 crossover full-pupil cumulative
Cs values are +3.537 and +586.820 mm in their respective outgoing angular
charts. The TL21 crossover lacks an admitted fit at either pupil. The last
projector interval has unresolved particle losses. Blank entries preserve
these distinctions. Each coefficient describes the full transport to its
own plane, with its own pupil and angular magnification.

The [scan-ready A profile](../../outputs/probe-joint-d95-20261101/placed-A-profiles/scan-ready.toml)
and [static inspection profile](../../outputs/probe-joint-d95-20261101/placed-A-profiles/static-inspection.toml)
were exported separately and read back against the executed optics. The
scan profile requires fresh scan calibration before image generation. The
[incident current kernel](../../outputs/probe-joint-d95-20261101/round2-placed-A-2000/target-current.npz)
contains absolute current per 0.02 nm pixel, out-of-frame current and the
original surviving positions and weights. The 5.12 nm plotted field contains
9.865 µA with another 0.135 µA outside the display. The cyan circle marks
the fixed 50 pm radius and the dotted circle marks D95 about the measured
centroid. The observed threefold tails remain visible around the bright
core. The experimental optical scope is
the classical incident probe. A specimen image and atomic image resolution
have not been validated.

## Physical question

At the specimen entrance and at the freely located combined-field crossover, measure absolute current inside a
50 pm radius about the fixed commanded axis and the diameter containing 95%
of the transmitted current about its measured centroid. Compare the target
with both native extrema of the same frozen optical setting over ±100 nm.
Compare feasible excitation settings at the same source count separately.
The secondary objective is to reduce the absolute distance between the RMS
radius minimum and the D95 minimum while retaining specimen performance.
A zero distance is not required.

Longitudinal separation alone cannot identify spherical aberration. Consider
a symmetric cohort carrying 96% of the current through a crossover at z = 0
and a second symmetric cohort carrying 4% through z = L. Give all rays the
same absolute slope a and propagate them along straight lines. The centroid
stays on the axis. D95 is zero at z = 0, while the squared RMS radius is

```text
R_rms² = a² [0.96 z² + 0.04 (z − L)²].
```

Its minimum occurs at z = 0.04 L despite completely linear transport.
The regression uses the native weighted quantile and L = 1 mm. For the
actual instrument the causal test compares the signed cubic term of a
controlled nominal-energy pencil with HP1 and HP2 on and off at fixed
matching settings. Total transmitted current and disk current are separate
observables. With complete transmission the former remains 10 µA on every
plane, whereas the latter changes at focus.

## Scope and procedure

All new physical calculations used the static unified-field Lorentz particle
executor. Source particles originate at the original 300 kV tip setting, with
unchanged energies, absolute weights, apertures, geometry and source law.
The 100 µm C2 aperture and Normal Nanoprobe branch are retained. The angle
criterion is 29.7–30.3 mrad for the measured current-weighted alpha95.
Radii, diameters and current are evaluated before material interaction.
The images intended for this working point require their own scan and
specimen validation.

Start from the previous balanced working point. Measure responses to paired
HP strength, HP1 relative strength and orientation, objective excitation,
auxiliary sextupoles and TL22/TL21. Local response models can propose a
change in excitation and physical focal placement. Each reported operating
point is a new transport through the actual fields. No extrapolated focus,
quantile smoothing or reduced particle pencil supplies a finite-source
acceptance result. Frozen gun handoffs in full-column replays verify the
source dependency signature, IDs, weights, coordinates, energies and clocks
before propagation.

The final instruction fixes the source at 2,000 original positive-current particles. All settings, objective comparisons and final profiles below use the identical source. Other-count experiments started earlier were stopped or archived and are excluded from the final ranking. Through-focus reports distinguish an exact common sampled extremum from a
1% relative performance allowance. They retain boundary and population
checks. A physical continuation from a verified original checkpoint at
−100 nm adds 0.1 nm observations near the waists. It checks the repeated
target measurement and does not constitute an independent upstream
integration-step test.

[Optical control contract](../OPTICAL_CONTROL_ROLES.md) ·
[Prior particle calibration](PROBE_CORRECTOR_PARTICLE_CALIBRATION_2026-11-01.md) ·
[Müller et al. on current containment and probe size](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf)



## Direct negative spherical response at fixed controls

The reference drive in this sweep is the rematched working point J0.
Only the two main sextupole strengths are scaled. HP1/HP2 balance and
relative orientation, TL21/TL22, auxiliary multipoles, objective, nominal
energy, 100 µm aperture, axial target and source are fixed. The physical
field digests identify the changed sextupoles. Every row retains the same
2,000 original particles and 10 µA at the specimen. H0 is a causal control,
and its measured alpha95 of 29.593 mrad is outside the operating band.

| Paired drive relative to J0 | Geometric total C3, 0.55 µm launch radius (mm) | Sample D95 (nm) | Current in 50 pm (µA) |
| --- | ---: | ---: | ---: |
| 0, main sextupoles off | +6.486818 | 391.131 | 0.025 |
| 0.98824477, finite-pupil C3 null experiment | +0.00000429 | 9.2649 | 1.300 |
| 0.999 | −0.142257 | 2.4115 | 2.105 |
| 1.000, J0 | −0.155562 | 2.0094 | 2.115 |
| 1.001 | −0.168881 | 2.2312 | 2.035 |

The main pair contributes **−6.642380 mm** to the sample-plane geometric
C3 at J0, taking the difference between matched H1 and H0. A 0.1% change
in drive near that point changes C3 by approximately 0.0133 mm. The
zero-current, single-virtual-point pencil uses identical inputs across
branches, outgoing mechanical slopes, a full bivariate polynomial through
degree five and independent holdouts. It is separate from the finite-current
source used for the D95 and disk-current columns.

The C3 null is local to the specified finite pupil and chart. A separate
0.1375 µm launch radius gives −0.003809 mm there. At the 0.55 µm launch
radius, the J0 fits retain a fifth-order round coefficient of about +195 mm
and a holdout RMS of 4.16 pm. The sample-plane drift, finite source, other
aberrations and finite-pupil weighting remain measurable when the cubic term
is balanced. The null comparison freezes the objective, as required for
causal attribution. Its 9.265 nm diameter is the actual size at the target,
rather than a separately refocused minimum.
