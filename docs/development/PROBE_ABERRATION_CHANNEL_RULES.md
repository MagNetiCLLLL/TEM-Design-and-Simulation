# Probe-corrector channel rules and physical-drive qualification

The historical response table below applies to the 300 kV, Normal Nanoprobe,
two-main-hexapole assembly at the specimen entrance of the R2-width-placed
working point. Its channels are perturbations of the unified model field;
they are not all verified independent windings or supplies. A fitted aberration
is an observable of the whole column. Multiple excitations can contribute to
the same observable, with different directions and side effects.

The [fixed-axis A5 continuation](probe_a5_physical_20261101/README.md) supersedes
any interpretation of the table as a hardware control map. Main-hexapole
normal/skew directions can require changing a fixed installation angle;
HPC/HPol hosts and electrical quadratures remain unverified. The upper-only
objective experiment is not an independent ordinary supply. Use the qualified
subset below for continuation from the accepted A5 point.

## Control selection

The last column gives the predicted in-axis response in pm for an independent
10 pm edge-intercept request at 30 mrad, first X and then Y for a directional
term. The bounded local solve uses all 29 actuators, penalizes other
coefficients and chief-ray shift/tilt, and limits each drive to four of the
declared diagnostic steps. A rounded value of 10 alone does not establish a
usable large-signal correction. For B4, A4 and C5 even this small requested
increment was not reached within that drive budget.

| Target | Start with these parts | Couplings and compensation to check | Predicted in-axis response (pm) |
| --- | --- | --- | --- |
| C1, defocus | Upper Objective field for final focus. | TL22, TL12 and TL21 have large focus responses and also change the corrector match. Re-observe the real Mini Condenser + Objective waist after positioning. | 10.0 |
| A1, twofold astigmatism | Condenser stigmator X/Y for fine trim. DPH1/DPH2 provide the existing coarse channel. QPH1, QPH2 and QPol also respond. | DP21/DP22 displacements in the operating hexapoles give particularly strong A1 feed-down. Protect chief shift, tilt, B2 and S3 before using this route. Each installed Q channel has a single strength axis. | 10.0 / 10.0 |
| A2, threefold astigmatism | HPC/HPol normal and skew for fine trim. Main HP1/HP2 Cartesian balance for coarse changes. | The largest direct step responses were from HP1/HP2. Protect C3 and D4. HPC also affects A5. Use HPol to recover A2 when using HPC for higher-order work. | 10.0 / 10.0 |
| B2, axial coma | DP11 X/Y is a comparatively selective direct lever. Combine with DPH1/DPH2 and DP21/DP22 to control the trajectory through the main hexapoles. | DP21 produces particularly large A1 spillover. Include A1, chief position and chief tilt as explicit constraints; include the stigmator for compensation if necessary. | 10.0 / 10.0 |
| C3, spherical aberration | Common HP1/HP2 drive with matched TL21/TL22. | Confirm negative contribution at fixed target and matching settings. Protect A2, D4, C5, A5, angle and focus. TL12 can also move C3 and C1 strongly. | 10.0 |
| S3, star aberration | Paired DP21/DP22 and DPH drives, with QPH1/QPH2/QPol and the condenser stigmator as compensators. | These controls also generate much larger A1 or B2 responses. Solve two independent S3 axes while protecting both lower-order terms and the chief ray. | 10.0 / 10.0 |
| A3, fourfold astigmatism | QPC is the strongest direct lever at this setting. QPH1 and the dipole/other quadrupole combination supply additional directions. | QPC also changes B2 and B4. One QPC strength cannot specify both A3 components. There is no independently driven probe octupole in this assembly. | 10.0 / 9.9 |
| B4, fourth-order axial coma | Investigate QPC/QPH1 together with DP21/DP22 and DPH. | Large collateral A3, A1 or B2 responses. The local solve produced only about half the requested response; maximum collateral components were 2.2 and 1.5 pm. An independent B4 knob remains unqualified. | 5.2 / 5.1 |
| D4, fourth-order three-lobe aberration | Change the main hexapole balance/orientation and the TL21/TL22 match. The two HPC quadratures are useful for higher-order threefold trim. | Compensate A2 using HPol and monitor A5, C3 and C1. Compare both second- and fourth-order threefold signals across pupil radii. | 10.0 / 10.0 |
| A4, fivefold astigmatism | Weak responses from QPH2, DP22 and DPH2 through the existing operating fields. | No probe decapole drive exists. Target-axis response is comparable to collateral change for these bounded moves; the installed channels do not provide a demonstrated selective A4 adjustment here. | 2.2 / 0.8 |
| C5, fifth-order spherical aberration | Explore HP-pair and transfer-lens/aperture-conjugate matching, including TL12, under C3 and focus compensation. | HP2 normal had the largest measured direct step response. C3 and A2 also respond much more strongly. Keep angle and transmission fixed. Independent C5 control remains limited in this local range. | 3.9 |
| A5, sixfold astigmatism | Use a calibrated combination. HPC normal/skew had the largest direct step response; HP1/HP2 and TL21/TL22 also contribute. | HPol and the main pair recover A2 and D4; use the transfer lenses and focus drive to hold C1, C3 and C5. Verify both A5 components and finite-source tails. No independently driven probe dodecapole exists. | 10.0 / 10.0 |

The photographed DCORPRIME service labels describe composite controls, and
their winding gains are not available for this simulator. The manufacturer's
two-hexapole analysis identifies both the main hexapole fields and the
intermediate transfer lenses as contributors to intrinsic A5. The strong
HPC response in this table was measured about the specific powered R2
configuration. The original simulator A1/A2 coarse/fine assignments are
documented separately. [Müller et al., 2006, pp. 451–453](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=10)
· [CEOS CESCOR](https://ceos-gmbh.de/en/produkte/stemkorrektoren/cescor)
· [Simulator A1/A2 controls](../corrector_channels.md)
· [Photographed service-control inventory](../probe_corrector_reference.md#additional-dcorprime-reference)

At the 29.72 mrad joint setting, the measured normal/skew condenser
stigmator response has rank two and condition number 1.00 in the signed
specimen-plane A1 chart. Measure both columns on the active state and solve
the two signed equations together, since each modeled drive changes
both coefficient components. Monitor C3, both A5 components and the native
core angular moments separately. The [focused A1 experiment](probe_a1_trim/README.md)
records the measured pair, protected terms, intermediate move and field identities.

## Verified A5 adjustment

The sole aberration target was the sixfold term. At the physical specimen
entrance the full-pupil geometric magnitude fell from **55.686 to 33.468 mm**
(39.9%). The quarter-pupil fit gave 55.690 and 33.471 mm. Independent rays
used to validate the compensated setting found a maximum collateral vector
change of **2.69 pm** at 30 mrad among the other eleven requested terms and
the extra C52/C54 terms, below the predeclared 5 pm bound. C3 was
−0.195372 mm before and −0.195387 mm afterwards.

| Native observable | Starting point | Accepted A5 setting |
| --- | ---: | ---: |
| Original particles / transmitted current | 2,000 / 10.00 µA | 2,000 / 10.00 µA |
| Crossover representative relative to surface | 0.0 nm | −0.1 nm |
| Sample alpha95 | 29.723 mrad | 29.777 mrad |
| Sample D95 / RMS radius | 3.017 / 1.845 nm | 2.502 / 1.710 nm |
| Sample current in the axial 50 pm radius | 2.360 µA | 2.330 µA |
| Normalized sixth moment across the displayed square | 0.172 | 0.140 |
| Normalized sixth moment in the 0.1–1.0 nm annulus | 0.237 | 0.145 |
| Absolute current beyond 1 nm | 1.445 µA | 1.200 µA |
| Normalized sixth moment beyond 1 nm | 0.313 | 0.424 |

The accepted setting reduces the full-field sixfold signal and the total
current at large radius. The remaining current beyond 1 nm is more
sixfold-oriented; its absolute sixth-harmonic current increases from
0.452 to 0.508 µA. Image panels use the same fixed axis, 20 pm square
pixels, absolute color scale and 5.12 / 0.44 nm fields as the R2 image.
The particle data and signed pencil coefficients are independent observables.
[Specimen current image](probe_channels_20261101/incident-probe-02.png)
· [Before/after coefficients](probe_channels_20261101/accepted-a5-coefficients.csv)
· [Acceptance measurements](probe_channels_20261101/accepted-a5.json)
· [Restore-tested vacuum working point](probe_stage12_20261101/inputs/portable-probe.temwp)

## Measurement and use

1. Freeze the specimen plane, geometry, source, energy and numerical settings.
   Change one physical drive with both signs and propagate the same
   monochromatic reference family. Fit the two-dimensional ray intercept
   against outgoing mechanical slopes about the chief in fixed laboratory
   axes. Use all polynomial monomials through degree seven, including terms
   outside the named set, with separate training and holdout rings.
2. Project each homogeneous vector polynomial onto the gradient of
   `r^(n+1) Re(C exp(-im phi))/(n+1)`. This specifies the component signs and
   normalization of coma and star terms as well as spherical and astigmatic
   terms. Report complex Cartesian components in length units and scale by
   `0.03^n` to compare equivalent displacement at the same pupil radius.
   The chart describes cumulative geometric ray behavior. Production current
   comes only from the original 2,000 positive-current particles.
3. Use both signed components for a directional aberration. Choose a
   bounded combination from the matrix, protect every other named term,
   the additional fifth-order harmonics C52/C54, chief coordinates and
   convergence. Repropagate and measure the candidate. Repeat the reference
   fit at a smaller pupil and finer integration step for a material change.
   After a substantial adjustment, refresh the response about the new setting.
4. Locate the real 2D crossover over the combined Mini Condenser / Objective
   interval and compare it with the actual specimen entrance. Report
   absolute current inside 50 pm, native D95, RMS radius, angular harmonics,
   transmitted current and alpha95. Show the native specimen-plane current
   image for the user's next aberration diagnosis.

Excitation increments used for the signed matrix were 1,000 m⁻³ for each
Cartesian sextupole component, 0.25 m⁻² for each fixed-axis quadrupole,
0.001 mrad for each deflector axis, 0.01 percentage points for each
condenser-stigmator axis, and 0.001 percentage points for ADL/TL21/TL22/TL12.
The separate upper-objective model experiment used 0.00002 equivalent
excitation percentage points on the upper field amplitude alone. The
installed shared Objective percentage scales both lobes.

The matrix records the central signed response in pm per stated step,
together with the chief shift in pm and chief tilt in µrad. The long table
also gives coefficients per unit of physical drive and the even part of the
positive/negative perturbation. The selectivity record gives both target
axes, the maximum other component and the largest participating drives.
Drive coordinates are relative to the specific R2 operating point.
[Signed response matrix](probe_channels_20261101/signed-channel-matrix.csv)
· [Per-drive coefficients](probe_channels_20261101/channel-responses-long.csv)
· [Bounded selectivity calculations](probe_channels_20261101/local-selectivity.json)

The 58 signed perturbations and their unperturbed reference produced 59
distinct physical field digests. The independent position holdout RMS ranged
from 0.260 to 0.262 pm for the seventh-degree response fits. The response
chart uses eight training rings and eight displaced holdout rings at 16
azimuths. The A5 qualification uses a separate 24-azimuth reference, a
ninth-degree fit, full and quarter launch radii, and a 0.025 mm integration
step. Removing only the fifth-degree sixfold component and refitting all
remaining columns increased the full-pupil pre-adjustment holdout RMS from
0.0036 to 19.16 pm.

The executable study and individual positive/negative observations are
retained with the local results. The coefficient projection and its analytical
regression are maintained in the repository.
[Response experiment](../../outputs/probe-channel-research-20261101/model.py)
· [Coefficient projection](../../scripts/probe_channel_response.py)
· [Analytical regression](../../tests/test_probe_channel_response.py)

## Physical drive audit

The subsequent [stage 2 inventory](probe_stage12_20261101/README.md#step-2-drive-inventory) distinguishes these mathematical perturbation directions from verified winding or supply freedoms. Use it before registering a direction as a physical channel.

### Current fixed-axis A5 rule

- Use HP1/HP2 scalar strength with both installation angles frozen, together
  with TL22/TL21/TL12 and ADL for matching. Use the installed Mini Condenser
  scalar and shared Objective scalar for physical focus placement. The latter
  scales both objective lobes; do not label it an independent upper-lens drive.
- These eight installed scalar excitations have eight distinguishable sampled
  field responses. After exact linear protection of C1, A2, C3 and alpha95,
  the measured local A5 response has two resolved components. Adding Mini
  improves their scaled singular-value ratio from about 474 to 18.4. This
  depends on the documented diagnostic steps and does not establish OEM
  winding calibration or a large-range independent A5 knob.
- Re-measure the native finite-source crossover after every candidate.
  Preserving pencil C1 does not preserve the RMS waist after other
  aberrations change. Explicit shared-Objective compensation is permissible;
  image-plane refocus is not a measurement of that compensation.
- The executed coupled candidate reduces A5 by only about 0.30% and outer
  sixth-harmonic current by about 3.54%, while protecting A2/C3 within 5 pm
  at 30 mrad and retaining the original current. D4/C5 changes are about
  28.7/24.3 pm, and the inner threefold tail and D95/RMS focus separation
  increase. It is a review candidate, not an accepted independent-A5
  correction. The original accepted A5 input remains authoritative.
- Keep all source IDs/weights, angle, transmission, radial-shell harmonics,
  inner threefold current and per-order coefficients in the acceptance record.
  A six-lobed current image alone cannot identify A5 or exclude D4/C5 coupling.

[Drive audit](probe_a5_physical_20261101/driver-audit.json) |
[Measured response](probe_a5_physical_20261101/protected-response.json) |
[Outcome](probe_a5_physical_20261101/outcome.json) |
[Every executed image](probe_a5_physical_20261101/IMAGE_HISTORY.md)

### Authorized joint C3/C5/A5 search

The subsequent [joint probe search](probe_joint_20261101/README.md) permits
controlled aberration coupling and evaluates the actual current distribution.
Its objective is increased fixed-axis core current together with reduced D95,
RMS and shell-resolved threefold/sixfold tails. The preceding 5 pm all-other-term
budget qualified an isolated A5 knob; it is not a permanent constraint on the
new joint optimization. All coefficients remain measured and reported.

Use the same eight registered scalar drives and keep source, apertures, field
profiles, angles of the multipoles, C2/C3 condenser lenses and auxiliary drives
fixed. C3 as an aberration coefficient is distinct from the C3 condenser lens.
After every candidate, find the real Mini/Objective crossover, execute the
required shared-Objective placement and then compare native specimen images.
Refresh the response near a new operating region; the initial linear response
does not predict finite coupled moves accurately.

An improvement in sixfold current does not by itself mean the A5 coefficient
decreased. Similarly, the central 50 pm disk current is an integrated current,
not the maximum image-pixel value. Retain both geometric coefficients and
full-cohort current measurements, including the far-angle source members
outside the reference pencil's approximately 30 mrad fitting domain.

[Joint measured outcome](probe_joint_20261101/outcome.json) |
[All joint-search images](probe_joint_20261101/IMAGE_HISTORY.md)
