# Corrector service controls and aberration fitting

For detached error injection, fixed-reference aberration comparisons and
electrical correction headroom, see [Assembly error studies](assembly_error_studies.md).
The **Assembly error study** tab reuses these intrinsic/learning routes and
does not fit or apply a new learning matrix.

## Probe service inputs

Open **Hardware Tuning → Corrector → Probe service channels (C1–A3)**.
The service input is a relative change in percentage points of the model's
output maxima. The table shows each output's maximum, current percentage and
separate intrinsic or learning role. Applying again adds another increment.
The complete combined move is validated before any live write.

The operator clarified on 2026-11-04 that `will change` denotes intrinsic
control and that `affect`, `changes slightly` and all incidental small
changes denote corrector-learning compensation. Accordingly, the default
learning matrix is zero and its execution switch is off.

| Tool | Intrinsic output group | Learning output group |
| --- | --- | --- |
| `AT_A2` upper | HP1/HP2 differential strength at fixed mount angles | DPH1x/y, DPH2x/y, QPH1x |
| `AT_A2` lower | TL21/TL22 differential excitation | DP22x/y |
| `AT_A2fine` upper / lower | HPolx / HPoly | QPoly / QPolx |
| `AT_A1fine` | QPol | Small DP11x/y changes |
| `AT_A3C` | QPC | Small DP21/DP22 changes |
| `AT_S3` | QPol | Small dipole and beam-shift/tilt changes |
| `AT_C3` | Model common HP1/HP2 excitation | Local dipole and quadrupole trims |
| `AT_B2` and `AT_B2fine` | Same DP12 x/y drive | No learning outputs identified in the notes |

Common and differential tools start with equal normalized weights of +1 or
−1. These are explicit simulator basis choices. Unmeasured weights in the
focus relay and some lower inputs remain zero until supplied. The C3
assignment retains the `AT_C2` versus `AT_C3` source-label discrepancy.
The QPol-based A1 and S3 tools share a primary field. Their selective
aberration response requires calibration. Natural coupling during propagation
is retained even with learning off.

For output `i`, the physical increment is its installed maximum multiplied
by `(intrinsic_gain + enabled_learning_gain) * delta_percent / 100`.
The learning gains are separately editable user calibration data. No
screenshot current or observed current ratio seeds this matrix.

For direct absolute settings, open **Probe outputs · maximum and percent**.
Select a physical output, read its model maximum and set its percentage.
Multipole and deflector percentages are signed. Lens excitation is positive.
The normal/skew vector retains the existing peak-magnitude limit, and the
physical coil limits apply to summed scan and DP12 drives.

Scan calibration, held-record restoration and worker/cache result installation
check that combined drive before accepting it. An over-limit result is rejected
and the previous calibrated drives and records are retained. Alignment is not
clipped or rescaled to make a scan fit. A complete calibration checks its final
coil ratios and scan amplitude together; a standalone pivot adjustment checks
the resulting drive against the retained amplitude.

Profiles and working points preserve the physical outputs and the new
`probe_service_percent` settings. Historical `probe_service_gains` count
maps remain inert archival metadata. Loading old data preserves its actual
field values and starts the percentage interface with learning off.
Programmatic service calls require the explicit `delta_percent=` keyword,
which prevents an old positional digit count from becoming a large percentage.

## Installed frames, optical transfer and reported aberrations

The component placement dialog and 3D dimension editor provide distinct
azimuth and tilt controls for main-column components.

| Control | Unit | Action |
| --- | --- | --- |
| Assembly azimuth about column Z | degrees | Rotates owned CAD, windings, mechanical bores and field bases together |
| Mechanical azimuth error (field only) | degrees | Adds an in-plane field rotation representing mechanical alignment error |
| Optical tilt X / Y | mrad | Tilts the optical axis about the transverse diameter axes |

Zero assembly azimuth aligns the local reference with column +X. Positive
azimuth turns +X toward +Y about downstream +Z. Shared DP channels edit their
physical winding host. DP12 follows the AC host. Explicitly owned mechanical
children inherit their parent's assembly azimuth. Independently mounted
components use the column reference even when they share a packing parent.
Apertures expose assembly azimuth and the existing effective plane tilts.

Let `a` be assembly azimuth and `e` the field azimuth error. The material
rotation is `Rz(a)`. The optical rotation is
`Rz(a) @ Rz(legacy_z) @ Ry(tilt_y) @ Rx(tilt_x) @ Rz(e)` for one placement
owner. Assembly azimuth rotates the complete installed field about column Z;
changing `e` preserves the tilted field-axis direction and position. Field
errors belong to the field's own placement owner. A parent source's
error leaves independently driven child sources at their own settings.
Legacy optical Z values retain their original Euler order and appear in the editor when
nonzero. New azimuth parameters default to zero in older inputs. Existing
hexapole `orientation_rad` remains the internal field-basis phase and is
applied once inside the hexapole law. The new assembly azimuth rotates that
entire field along with its declared material geometry.

The simulator uses a fixed right-handed laboratory frame with +Z downstream.
An element's field basis is attached to its local transverse plane. The
existing placement transform carries that field into the laboratory frame.
The main hexapole `orientation_rad` and the auxiliary winding bases stay
fixed during service operations. An optical-frame roll from the placement
model is applied once. The finite-column propagator then includes the
signed magnetic-lens rotation.

For a local harmonic of multiplicity `m`, active rotation through `theta`
transforms its complex strength by `exp(i*m*theta)`. A passive rotation
of the coordinate axes through `beta` transforms the same coefficient by
`exp(-i*m*beta)`. Thus two independent dipole bases are separated by 90°,
quadrupole bases by 45°, and hexapole bases by 30°. These are electrical
field-basis angles. A zero field has no uniquely defined polar angle.

The optical comparison of hexapole directions uses their conjugate aperture
planes. Published CESCOR optics relates the hexapole midplanes to the
objective's coma-free aperture plane through transfer optics. The complex
probe-aberration coordinate is an angular pupil coordinate at the specimen.
Camera or scanned-image orientation adds the corresponding calibrated image
transform. [Müller et al., 2006, Fig. 2 and Eqs. 5–7](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf)

The relevant relative hexapole phase includes both installed orientation and
optical rotation through TL21/TL22. Their strength mismatch and relative
orientation mismatch generate different components of threefold astigmatism.
The two A2 service inputs therefore act on different hardware groups.
The exact angular zero linking this DCORPRIME's raw outputs to its camera
requires instrument calibration data. [Müller et al., 2008, section 5.3](https://doi.org/10.1016/j.phpro.2008.07.093)

The ordinary angular-aberration convention uses multiplicities 2 for A1,
1 for B2, 3 for A2, 2 for S3 and 4 for A3. QPC remains a physical quadrupole
even when a service tool uses its coupled response to adjust A3.
The service-input labels, electrical quadratures and fitted aberration
components have separate meanings.

## Shared deflection hardware

The user confirmed on 2026-11-07 that, for this instrument, the AC Scan Coil
used without a probe corrector is the same physical hardware called DP12 in
the CEOS probe-corrector system. `ac_deflector` represents that single physical
host; `probe_dp12_scan_deflector` is its CEOS control/reference alias. The
alias adds no winding, material body or independent magnetic field. Scan and
alignment inputs must share the host's geometry, pose and physical drive limits.

DP12's additional software reference is virtual; the AC scan hardware it
controls is physical. The current model distributes a static angular increment
between the two finite AC coils. For upper and lower coil
centres `zu`, `zl` and virtual reference `zr`, the lower fraction is
`(zr - zu) / (zl - zu)` and the upper fraction is its complement.
This preserves angle and first axial moment in a field-free drift. The
propagator uses the actual coil positions, lengths and other column fields.
Raster and wobble commands superpose on the same two physical fields.
The upstream BSh/BTlt outputs use difference and sum modes on the existing
beam deflector. These numerical distribution rules remain engineering choices;
the user's confirmation establishes the DP12/AC hardware identity, not those
coefficients. OEM winding geometry, gains and specimen-plane shift or tilt
purity remain unmeasured.

### Historical count-interface verification

The following record predates the operator's intrinsic/learning
classification. It retains the formerly combined move for traceability.
The current default A2 lower tool drives the two transfer lenses only.

At the [archived 300 kV input design](development/probe_stage12_20261101/inputs/baseline.json),
the unified particle solver propagated the same 2,000 original positive-current
gun-exit records from 450 mm to the specimen entrance at 1599.1999975 mm,
with a maximum integration step of 0.05 mm. Applying +9 counts on the lower
A2 input changed TL21 by +0.09 and TL22 by −0.08 percentage points, and
DP22x/y by −0.0005/−0.0002 mrad. The source IDs, weights, initial energies
and times were checked against the archived array hashes. Both runs delivered
10.00 µA with 2,000 surviving and 2,000 effective particles. At the same
plane the current-weighted RMS displacement between matching particles was
20.2055 nm. This measures the combined response of these model drives.
A separate aberration measurement is needed to isolate the A2 coefficient.

## Model A1/A2 coefficient fit

The second tab measures the current simulator's response and adjusts its
electrical drive groups. It accepts requested relative changes in the two
Cartesian components of an aberration coefficient. The service tab above
accepts increments of an individual instrument input.

| Channel | Probe corrector | Image corrector, including B-COR |
| --- | --- | --- |
| A1 coarse | DP21/DP22 X/Y deflection | DPH1/DPH2 X/Y deflection |
| A1 fine | QPol normal/skew | Objective stigmator normal/skew |
| A2 coarse | Main HP1/HP2 scalar strength and TL21/TL22 excitation | Main HP1/HP2 and installed HP3 normal/skew |
| A2 fine | HPol normal/skew electrical drives | Auxiliary HPol normal/skew |

Coarse/fine therefore selects different physical hardware. It is not a change
of a GUI step multiplier, and the labels do not assert a manufacturer-specific
sensitivity ratio. A channel requires its hardware to be installed and enabled.
The probe fitter preserves each installation angle. The separate image
corrector fitter retains its historical polar-strength/orientation model.

## Use

1. Select the installed probe or image corrector and one of the four channels.
2. Set the diagnostic pupil (initially 0.5 mrad) and maximum axial integration
   step (initially 0.05 mm), then select
   **Calculate response**. Calibration is an explicit background calculation.
3. Enter signed normal/skew **coefficient increments in nm**. The panel shows
   the physical drive group and individual-axis local limits. Simultaneous
   normal/skew requests may have a tighter combined limit.
4. Select **Calculate and apply change**. An independent propagated candidate
   must achieve the requested increment before any live hardware is changed.
   The panel reports the before/after A1, A2, C3 and C1 values. Recalibrate after
   each successful change or after modifying any physical input.
5. Recalculate the coherent beam, select the desired observation Z and read
   its phase plate through the existing section viewer. A stale wave cannot
   represent newly applied hardware settings.

Each calibration captures a new nominal first-order reference. Its before/after
pair shares that reference; absolute fitted coefficients from separate
calibrations are not directly comparable. In particular, a new reference can
absorb an already-applied stigmator's linear A1 action. The independent-pupil
fit RMS shown in the panel helps assess the polynomial diagnostic. The actual
wave retains the entire stigmator action. For later constraints on equal phase
plates, compare the propagated complex waves at matching physical coordinates
and source conditions, rather than comparing separately re-referenced fits.

The actual propagated wave retains tilt, defocus, higher-order coupling,
apertures and specimen effects supported by the existing propagation path.
No fitted low-order phase is subtracted from that wave. A mixed-state beam
still has per-mode phases; an incoherent sum does not acquire a single phase.

## Finite thickness and validity

All current hexapoles retain their distributed Gaussian axial field:
`effective_length_mm` is its FWHM, and `strength_m3` is the peak coefficient.
A shorter physical element remains finite. Editing field length at fixed peak
changes integrated excitation as well as the nonlinear interaction during
propagation. Editing length while preserving integrated excitation can still
change the third-order response. A zero-strength element has no field action
regardless of its length. Mechanical envelope length and effective field
length remain separate parameters; neither is a measured 3D magnetic map.

Hexapole and quadrupole inputs are checked at runtime edits, snapshot restore
and propagation-plan construction. Non-finite values, invalid effective
lengths and strengths beyond the declared limit are rejected. Signed physical
strengths remain valid within their bounds. Failed edits do not partially move
the associated hardware.

Calibration holds the source pupil, nominal transfer and coefficient
normalization fixed across every perturbation. Recalculating that reference
for each stigmator perturbation would absorb the first-order A1 response.
Signed full/half-step physical perturbations check response rank, conditioning
and local linearity. A normalized minimum-norm solve uses each drive's measured
step, bounds the requested adjustment, then propagates it again. Only the
measured physical settings are committed atomically; cancellation or changed
input identity prevents publication. Changes in field width, positions,
strengths and the rest of the captured physical state invalidate calibration.

The reported aberrations are finite-pupil ray diagnostics of this working
point. C1, C3, displacement and direction are coupled outputs, not constraints
silently held fixed. This diagnostic does not certify finite-source behavior,
interception, scattering, arbitrary pupil sizes, integration-step convergence
or an experimental microscope. Refine the integration step and pupil as
appropriate before relying on a numerical response for an optimization study.

## Regression coverage

`test_corrector_parameter_validation.py` covers rejected edits, restore and
propagation entry. `test_corrector_channels.py` covers calibration, bounded
candidate checks and state-safe application. `test_corrector_channel_gui.py`
covers explicit execution, stale inputs, cancellation and publication.
`test_hexapole_thickness_response.py` checks analytical finite-thickness
responses, and `test_corrector_channel_wave_transport.py` checks actual complex
wave and full-phase changes, signed normal/skew symmetry and axial refinement.
These enter the `field-ui` and `coherent-development` acceptance inventories;
passing bounded cases is not a full microscope qualification.
