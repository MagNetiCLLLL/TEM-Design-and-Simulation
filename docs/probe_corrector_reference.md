# Probe-corrector reference and validation

The user-supplied photographs of an older **Probe Corrector DCOR (Service)**
interface are references for separating hardware channels, calibrated controls,
and correction diagnostics. They are not a magnetic-field calibration or a
production drawing of the simulated corrector. Those older screenshots show
**STEM @ 200 kV**. Additional photographs identified by the user as the
**Iliad Ultra CEOS probe corrector** show **DCORPRIME SERVICE UI**, operating
at **STEM @ 80 kV**. The current default simulation uses **300 kV**.

## Reading the DCOR screenshots

- **Internal Single Channels / Total Output** shows absolute channel outputs,
  with units including A, mA and microampere. The two main HP channels are near
  105 mA; transfer-lens channels are near 0.75 and 1.03 A. These values alone
  do not specify magnetic field, integrated multipole strength, effective
  length, polarity, or excitation dependence.
- **Calibrated Alignment Tools** shows adjustments associated with aberrations
  and alignment effects, such as StigA1, StigA2, B2, C3, BeamShift and BeamTilt.
  Zero displayed adjustments can coexist with nonzero absolute channel outputs.
  A calibrated control can require coordinated changes to several channels.
- The displayed **Phase Plate** is a phase diagnostic in angular space. It is
  not an image of the real-space probe. Its outer angular extent does not by
  itself specify the working convergence angle or probe diameter.

The general optical roles relevant to the simulator are primary hexapoles for
the nonlinear correction, quadrupole trims for nonsymmetric adjustment,
dipole steering for alignment, and round transfer lenses for the relay between
correction elements. Exact assignments of ambiguous labels such as DPH, HP and
auxiliary channels require additional manufacturer documentation. Neither
element order nor polarity should be inferred from a screenshot label alone.

In a two-stage hexapole corrector, the combined hexapole action can supply
negative spherical aberration while compensating the unwanted threefold
distortion. Cancellation depends on the complete relay and operating condition;
the presence of two enabled hexapoles does not establish correction. See
[JEOL's Cs corrector explanation](https://www.jeol.com/words/emterms/20121023.021059.php)
and [aberration terminology](https://www.jeol.com/words/emterms/20231204.php).

The simulator therefore retains field strengths and excitation settings in its
own units. No A-to-field conversion, OEM accuracy, or manufacturer's calibrated
aberration-control response is inferred from the photographed current values.

## Devices and control-channel markers

The probe layout separates electrical controls from material parts. Fresh
**C3 + Probe Corrector** and **C3 + Probe Corrector + Image Corrector**
assemblies link DPH1/DPH2 and DP11/DP21/DP22 to five explicit X/Y winding
inserts. The channel remains selectable and has no second material body.
Its position, effective field thickness and rigid pose come from its insert.
Older saved manifests without these links retain their original model.

DPH1 and DPH2 now act at the optical reference of HP1 and HP2 respectively.
This implements the published optical arrangement, using a coincident midplane
as an explicit approximation; it does not establish an exact OEM coil centre.
The two DPH inserts occupy space inside the corresponding HP envelopes.
DP21/DP22 flank the transfer doublet and DP11 precedes TL12 at their existing
model positions. The main HP and round-lens positions and strengths are not
changed by this reconstruction.

QPH1/QPH2, HPC/QPC and HPol/QPol remain channel-only records with unverified
physical hosts. DP12 remains a virtual alignment reference. The classification
does not remove their optical fields or imply the absence of real hardware.
Existing unlinked channel envelopes remain compatibility metadata rather than
independent material bodies, vacuum walls or FEM iron.

### DPH structure evidence and modelling boundary

- [Müller et al., CEOS, 2006, pp. 443–444 / Figs. 2–3](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf)
  identifies eight X/Y alignment deflectors, including DPH1/DPH2, and places
  them at the corresponding hexapole stages in the optical schematic. Figure 3
  shows a hexapole's pole pins, coils and yoke, but does not identify the DPH
  winding arrangement. The published HP dimensions belong to that design;
  they are not transplanted into the present model.
- [CEOS-hosted 2025 workshop, Lu, p. 2](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/MC2025_CEOS_PL_20250831.pdf#page=2)
  shows the CESCOR DPH/HP optical relationship and the additional alignment
  elements. This is an optical diagram, not a scale construction drawing.
- [CEOS patent EP1012866B1, description of local alignment fields](https://patents.google.com/patent/EP1012866B1/en)
  discusses superposing dipole and quadrupole fields on hexapole fields to
  compensate local misalignment. It supports overlapping optical fields, not
  a particular DPH housing or electrical connection.
- [CEOS patent US20090079531A1, Figs. 5–6 and 10–11](https://patents.google.com/patent/US20090079531A1/en)
  describes concentric, separately driven dipole windings and installation in
  a lens bore or recess. This is the basis for a feasible generic winding
  insert representation, not proof that these named CESCOR channels use that
  exact patented construction.
- [FEI patent EP2325863A2, paragraphs 0070–0071](https://patents.google.com/patent/EP2325863A2/en)
  discusses the CEOS Rose corrector used on Titan 80–300 and multipoles that
  simultaneously generate hexapole and alignment dipole/quadrupole fields.
  It independently supports superposed field operation, without supplying
  DPH dimensions or a channel-to-winding map.

Each insert is an **engineering reconstruction**, not an OEM CAD part. Its
14.5/14.6 mm mechanical length and 4 mm effective magnetic length retain the
previous model allocations. The 22.2 mm inner and 23.8 mm outer diameters fit
outside the existing 20 mm vacuum passage and its liner. The HP envelope's
material opening is 24 mm to leave room for the DPH insert; its optical clear
bore and vacuum passage remain 20 mm. The rendered orthogonal winding pairs
show their two-axis role; turn count, resistance, inductance, current-to-field
calibration and a magnetostatic solution of these windings are not claimed.

The existing finite-field solver remains authoritative: an X/Y kick command
specifies integrated transverse field at the reference beam momentum, spread
across the configured effective thickness. Particle and coherent propagation
read that same field. There is no additional thin kick or field inferred from
the display mesh. Disabling a channel removes its field while leaving its
installed hardware in place. Zero deflection is also not hardware removal.

### Axial order, radial nesting and overall length

Positive Z is downstream. The physical order is BSh/BTlt → ADL →
**HP2 with DPH2 inside** → TL22 → DP22 → central relay region → DP21 →
TL21 → **HP1 with DPH1 inside** → DP11 → TL1 (stored key `probe_tl12_lens`)
→ AC scan pair → mini condenser / objective. The QPH/HPol/QPol and other
field-only markers do not add separate material stages to this sequence.

For the default C3 + Probe Corrector module (module-local Z; add 450 mm
for the default gun's source-absolute Z):

| Stage | Start–end Z (mm) | Field centre Z (mm) | Relation |
|---|---:|---:|---|
| HP2 | 685.4–703.6 | 694.5 | DPH2 occupies the same optical centre |
| DPH2 winding insert | 687.25–701.75 | 694.5 | Inside HP2, not upstream of it |
| TL22 | 703.6–729.1 | 716.35 | Before DP22 |
| DP22 winding insert | 729.1–743.6 | 736.35 | Between TL22 and the relay middle |
| DP21 winding insert | 772.6–787.2 | 779.9 | Between the relay middle and TL21 |
| TL21 | 787.2–812.7 | 799.95 | Before HP1 |
| HP1 | 841.8–859.9 | 850.85 | DPH1 occupies the same optical centre |
| DPH1 winding insert | 843.6–858.1 | 850.85 | Inside HP1, not upstream of it |
| DP11 winding insert | 889.0–903.6 | 896.3 | After HP1, before TL1 |
| TL1 | 903.6–929.0 | 916.3 | Before the separate AC scan pair |

At either DPH/HP centre, inside-to-outside diameters are: vacuum 20 mm,
liner outer diameter 20.5 mm, DPH winding inner/outer diameter 22.2/23.8 mm,
HP material opening 24 mm. These give radial gaps of 0.85 mm before the
winding and 0.10 mm outside it. The two winding layers are orthogonal X/Y
steering pairs, not two additional serial lenses. The TOMLs and 3D part notes
record these axial and radial relationships separately.

[CEOS's German CESCOR specifications](https://www.ceos-gmbh.de/de/produkte/stemkorrektoren/cescor)
explicitly give **H × W × D = 255 × 337 × 255 mm**. In particular, 337 mm
is the width, not the axial height. The published specifications do not
identify flange locations or which adapter/transfer optics are included.
[DCOR/ASCOR specifications](https://www.ceos-gmbh.de/de/produkte/stemkorrektoren/dcor-ascor)
give different device dimensions; they must not substitute for CESCOR.

The current model's HP2-to-HP1 core envelope is **174.5 mm**; ADL-to-TL1
coupled optics span **309 mm**; including BSh/BTlt spans **359 mm**. These
are three distinct scopes, not three measurements of the manufacturer's
housing. The optical core is compatible in scale with the 255 mm device
height, but a flange-to-flange mechanical match remains unverified.
[FEI's coupling-optics patent, Fig. 1 description](https://patents.google.com/patent/US8841630B2/en)
also distinguishes core-corrector and coupling optics, without establishing
the enclosure boundaries of this particular model.

Both assembly TOMLs store the published H/W/D reference and the keys defining
each model span. The new coils fit inside the existing span; no new serial
length is added, and no source or lens setting is changed to force an external
dimension match. A fictitious OEM housing is not added. The HP material
envelopes (18.1/18.2 mm length, 75 mm OD, 20 mm clear passage) remain provisional;
they are not the 2006 paper's specific 30 mm pole length, 152 mm yoke OD and
8 mm bore. Thus this update establishes topology and active deflector fields,
not exact reproduction of the manufacturer's mechanics or Cs calibration.

### Executed DPH/DP validation (10 October 2026)

`scripts/validate_corrector_deflectors.py` exercised the default C3 + Probe
Corrector with 3,000 identical Tip rays, the existing source and optical
settings, Cs and hexapoles enabled. The gun prefix used its existing CPU
solver; the post-gun column and finite-coil comparisons used actual CUDA.
There were 2,401 surviving rays. Each of DPH2, DP22, DP21, DPH1 and DP11 was
checked with a baseline, positive/negative X and Y at 0.001 and 0.0005 mrad,
and nonzero commands with the channel disabled (55 finite-coil cases total).
An independent DOP853 reference included the scalar electric field and the
resulting energy/momentum variation. The largest relative response discrepancy
was **5.29e-10**; disabling each channel reproduced the baseline coordinates
and alive masks exactly. This verifies the implemented integrated-field
response, not the winding's current-to-field calibration.

The unchanged baseline gave HP2-to-HP1 `||B||₂ = 5.16e-14 m/rad`, unit
first-order magnification, sample RMS radius **0.125874 nm**, and sample D95
**0.495304 nm**. Source waists remained above ADL and between TL1 and the mini
condenser; the latter was at **Z = 1400.000702 mm**. The TL22–TL21 region had a
radial minimum at **Z = 1208.148342 mm**, but its two principal transverse
waists were separated by approximately **6.23 µm**. The width difference was
only about 0.013 nm on a roughly 1 µm principal width; this is an approximate
common waist. The strict simultaneous two-axis waist check remains failed.
Neither the optical settings nor its tolerance were changed to hide that result.

The bounded run took **406 s**. Its original `report.json` retains a final
`FAILED` status because editing UI/model support files during execution changed
the whole-source implementation fingerprint. Separate post-run comparison
confirmed exact equality of the captured and current physical inputs, resolved
assembly and numerically reconstructed field plans. It does not prove bytewise
identity of every executing file or convert this run into a current-code
end-to-end pass. The five numerical deflector checks and this final identity
audit are reported separately. No full coherent propagation was repeated.

Local evidence is under `outputs/corrector_dph_20261010/particle_run2`:
`inputs.json`, `protocol.json`, `coordinates.npz`, the original `report.json`,
`completed_numerical_summary.json`, `postrun_equivalence.json`, and
`corrector_particle_validation.png`. The axial/radial geometry figure and its
provenance are in `outputs/corrector_geometry_relation_20261010`.

Fresh assemblies load the new TOML bindings. Restart and assemble the selected
probe-corrector column to adopt them; an older saved instrument or result is not
silently rewritten. Linked field placement is edited through the physical coil
insert (and the HP reference for DPH), while its X/Y drive remains on the same
existing channel. Saving, reloading, undo/redo and independent mechanical copies
are covered by focused regressions.

## Image-corrector channels and scan hardware

The same distinction applies to the image corrector. CEOS's
[2025 workshop, page 26](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/01_How-do-aberration-correctors-work-pdf.pdf#page=26)
separates the principal OL/TL/HP/ADL optics from additional alignment channels:
HPol, QPol, DP11/DP12/DPH1/DP21/DP22/DPH2, ISh, DSh and DSt. Both image
manifests mark these auxiliary records as control channels with an unverified
physical host. Older records are classified by their canonical keys as well.
Their optical fields and coordinates remain available, but legacy envelopes
do not become separate material bodies, vacuum walls or beam stops. This does
not imply that the channels lack physical coils or multipoles.

OL post, HP1/HP2, transfer lenses and the adapter retain their existing optical
and provisional mechanical models. The schematic does not supply OEM internal
dimensions or prove a physical host for an auxiliary channel. In particular,
stored DPH parent links and the DP12/TL12 overlap describe the existing model,
not verified manufacturer packaging. TL12 remains its existing effective-lens
representation; no new housing or host is inferred from the channel diagram.
Corrector ISh/DSh names alone do not establish that they share the microscope's
main image/diffraction-deflector pair.

The pre-corrector beam deflector and the downstream AC scan pair remain
separate model hardware. CEOS's
[CESCOR description](https://www.ceos-gmbh.de/de/produkte/residualsCEXCOR)
places the scan between the corrector and objective, and distinguishes BTlt
above the corrector from DP12 below it for coma alignment. Generic microscope
beam/scan reuse therefore does not establish a common physical pair across
the corrector. This evidence supports the ordering, not the simulator's
millimetre coordinates or a particular manufacturer's full wiring map.

## Additional DCORPRIME reference

The two additional user photographs are preserved without alteration:

- [Service alignment tools, 80 kV](references/ceos_dcorprime/service_alignment_tools_80kv.jpg)
- [Internal single channels, 80 kV](references/ceos_dcorprime/internal_single_channels_80kv.jpg)

These extend the interface reference with three distinct kinds of information:

1. **Internal single channels:** named hardware outputs and their units.
   The photographed page includes HP1/HP2, QPH1/QPH2, DPH1/DPH2, transfer
   lenses, steering channels, and channels labelled QPC and HPC. Record these
   as UI labels; their field geometry, longitudinal positions and calibration
   matrices are not specified by the photographs.
2. **Service alignment tools:** named coordinated adjustments, including
   AT_A1, AT_A1fine, AT_A2, AT_A2fine, AT_B2, AT_B2fine, AT_A3C, AT_C3,
   AT_S3, AT_A4, AT_B4, AT_B4pure, AT_D4, AT_A5C and AT_ApDist.
   Other visible controls include HP1_P/HP2_P, QPH1_P/QPH2_P,
   TL2asym/TL2sym/TL2SymFoc, DP2asym/DP2sym, BeamShift, TableauTilt,
   BeamTilt, TiltOL, ShiftOL, UserBeamShift and RestoreShift.
   These controls belong to a different layer from absolute hardware outputs;
   their displayed zero offsets do not establish zero element excitation.
3. **Optical state:** a phase-plate view, an estimated probe-shape view,
   microscope properties and several separate probe-size measures. A future
   simulator interface should preserve these distinctions and identify the
   basis of each displayed quantity.

Both photographs display the following working-state values (transcribed
from the UI, not independently measured or adopted as simulator defaults):

| Display label | Displayed value |
|---|---:|
| STEM voltage | 80 kV |
| Probe semi aperture | 20.2 mrad |
| Probe current | 131 pA |
| Chromatic aberration | 1.35 mm |
| Energy spread FWHM | 0.4 eV |
| Diffraction limit | 126 pm |
| + Source size | 123 pm |
| + Cc × dE | 0.54 eV mm |
| Optimum D50 | 180 pm |
| Total D50 | 180 pm |
| D FWHM | 157 pm |
| I FWHM | 41.76% |

The screenshot's probe metrics are labelled D50 and FWHM. Preserve those
definitions separately from this simulator's geometric RMS radius and D95;
the displayed numbers are not interchangeable. The photographs alone do not
specify the vendor's full probe-size calculation or how its individual
contributions are combined.

The optical-state panel also shows **Exclude 1st & 2nd order**,
**Correction: None** and **No geometric aberrations**. These are the displayed
diagnostic selections/status, not evidence that the physical instrument has
zero residual aberrations or that its corrector elements are switched off.
The pictured probe and phase plate should therefore be retained together with
those settings when used as visual references.

This reference addition does not change the current simulator geometry,
300 kV calibration, field settings or correction algorithm. In
particular, QPC/HPC labels alone do not justify adding field sources or
assigning their physical parameters.

## Inspecting a calculated probe

Run a calculation, then open **Scanning Image → Probe Aberrations**. The sample
probe summary uses all finite surviving incident rays and their current weights
at the physical sample plane. It reports geometric RMS radius, 95%-current
diameter D95, twofold and threefold shape moments, and a local linear estimate
of the waist offset. Positive waist offset means downstream of the sample.
These values need not equal statistics computed from the decimated points
displayed in the transverse ray plot.

For centered positions `z = (x - <x>) + i(y - <y>)`, the shape moments are

`M_n = |<z^n>| / <|z|^n>`, for `n = 2, 3`, with current-weighted averages.

Both range from zero to one and are invariant to translation, rotation and
uniform scaling. An equilateral three-lobed pattern can have `M_2 = 0` and
`M_3 = 1`: a circular covariance ellipse does not guarantee a round spot.
Finite-source sampling also contributes to these moments. Low values alone
do not exclude higher-order structure, broad radial tails, or residual
aberrations, and the moments are not A1 or A2 coefficients.

The coefficient table is a separate, compact reference-ring diagnostic. It
estimates a C3 residual ratio by comparing transverse ray errors with nonlinear
fields off/on. The errors are **positions**, reported in nm, not angles in rad.
It does not fit the full aberration set to the actual illumination bundle.

An em dash **—** means that a residual coefficient has **not been measured**.
It does not mean a measured zero. An explicitly supplied coefficient is marked
as configured, and the existing wave model retains its configured/default
coefficients independently of the geometric shape measurements. A circular
wave image produced with default-zero nonsymmetric coefficients does not prove
that the traced corrector has eliminated those aberrations.

## Acceptance of a corrected operating point

Evaluate these properties together, on the same physical sample plane:

1. **Numerical convergence:** hold optical settings and deterministic source
   rays fixed while refining the integration step. Compare RMS radius, D95,
   waist offset and both shape moments. A setting fitted at one step must remain
   acceptable at finer steps; numerical drift must be small relative to the
   claimed improvement and probe dimensions.
2. **Focus and illumination:** verify the waist relative to the sample, the
   requested current-weighted convergence angle, and surviving current. Enlarging
   a defocused spot, shrinking the pupil, or rejecting most rays can conceal
   asymmetric errors without delivering the requested probe.
3. **Shape and size:** compare the same-scale probe distribution, twofold and
   threefold moments, radial containment and tails. Repeat with adequate source
   sampling; do not select an operating point solely by its covariance or one
   normalized moment.
4. **Aberration scope:** distinguish the reference-ring C3 estimate from a fit
   over the actual pupil. Assess a centered-source bundle as well as the finite
   source, so source broadening cannot hide deterministic optical errors.
5. **Wave interpretation:** geometric ray containment is not a diffraction-limited
   FWHM. A quantitative intensity-probe claim additionally requires converged
   wave sampling and the relevant pupil, residual phase, source coherence and
   energy distribution.

Acceptance tolerances and the validated voltage, aperture, assembly and lens
settings should accompany each calibration. A good default working point does
not certify arbitrary later channel edits or another instrument configuration.

## Recalibrated default and reproduction

The 10 October 2026 **Nanoprobe** preset is a 30 mrad working point for the
current FEG, C3 + Probe Corrector column, 300 kV and 100 µm C2 aperture at
absolute Z 765 mm. The sample centre is Z 1599.2 mm; the incident probe is
measured at the physical upper surface, Z 1599.1999975 mm for the default
5 nm thickness. The Tip definition, round-lens positions and Objective
excitation of 68.9801% are unchanged by this recalibration. The subsequent AC
scan-pair relocation and its separate scan validation are documented below.

The preset restores these round-lens settings and both principal hexapoles:

| Lens | Excitation (%) | Field direction |
|---|---:|---|
| C1 | 51.666666666666664 | +Z |
| C2 | 25.25772717253661 | +Z |
| C3 | 46.059815218359184 | +Z |
| ADL | 79.80694335935594 | +Z |
| TL22 | 59.93152503582202 | −Z |
| TL21 | 59.95527906841963 | +Z |
| TL1 (stored key `probe_tl12_lens`) | 49.977014835272826 | +Z |
| Mini Condenser | 43.88646617222637 | −Z |
| Objective | 68.9801 | +Z |

| Hexapole | Strength (m⁻³) | Azimuth (rad) |
|---|---:|---:|
| HP2 | 694323.0354215894 | 0 |
| HP1 | 416774.6329835517 | −0.041975232520065166 |

Three distinct checks establish the relay and focus; a crossing in one plotted
projection is insufficient:

- **HP2 → HP1 conjugacy:** the first-order position map
  `r_HP1 = A r_HP2 + B theta_HP2` has `||B||₂ ≈ 5.17 × 10⁻¹⁴ m/rad`, with both
  singular values of `A` approximately one. This is unit-magnification imaging
  of the HP planes. It does not assert that the full four-dimensional transfer
  is `−I` or afocal; the remaining blocks and magnetic rotation also matter.
- **An executed crossover between TL1 and CM:** the current-weighted physical
  ray bundle converges and then diverges near **Z 1400.00070 mm**, with RMS
  radius **2.15835 nm**. Both principal transverse widths have a local minimum.
  The waist lies between TL1's hardware exit at 1379 mm and CM's entrance at
  1512.5 mm. Its finite radius reflects the finite source and aberrations; it
  is not a requirement that every ray intersect the axis at one point.
- **Sample focus:** the same emitted population reaches a focused probe at the
  sample entrance after the combined CM/Objective field.

The physical particle validation uses 3,000 emitted Tip samples, including their
individual energies and weights, in **Custom** mode with spherical aberration
and hexapoles active. The physical source-to-gun-exit implementation runs on
CPU; the post-gun column executes on **CUDA GPU**, with recorded backend
receipts. On the detached validation state only, AC raster, Descan raster and
AC wobble are disabled, while the physical coils and their static biases remain
unchanged. Normal GUI scan defaults are not disabled.

| Quantity | Result |
|---|---:|
| Surviving incident rays / emitted rays | 2,401 / 3,000 |
| Surviving current fraction | 80.0333% |
| Sample geometric RMS radius, 0.025 mm step | 0.1258745 nm |
| Sample 95%-current diameter D95, 0.025 mm step | 0.4953038 nm |
| Sample 95%-current convergence semi-angle | approximately 30.0067 mrad |
| Sample covariance waist offset, 0.025 mm step | approximately −0.02884 nm |
| RMS-radius change, 0.05 → 0.025 mm step | approximately 0.00000118 nm |
| Covariance waist-offset change, 0.05 → 0.025 mm step | approximately 0.02884 nm |

At the 0.05 mm fit step, the same-ray nonlinear displacement relative to the
linear column was about **0.6890 nm** with spherical aberration and no
hexapoles, and **0.0004426 nm** with the fitted hexapoles. This comparison holds
the round-lens settings and physical source population fixed. The earlier
**Ideal** mode suppressed these terms. Since the 10 October 2026 definition
change, Ideal retains Cs and HP using the same parameterised field laws; a
linear comparison must explicitly disable those operators in its detached
diagnostic calculation.

These results establish a bounded particle working point and two-step
transport consistency. They do not establish source-sampling convergence,
wave-probe resolution, an OEM current calibration or experimental microscope
performance. The older 25 mrad / 1,000-ray and 15,000-ray figures described
different optics and are superseded as default-preset claims.

### Applying and validating the defaults

Restart and create a fresh default assembly to obtain the catalog CM magnitude.
The 43.88646617222637% installation default applies specifically to **C3 +
Probe Corrector**. C2, C3, C3 + Image Corrector and the double-corrector column
retain their historical 10% startup magnitude; their optics were not retuned by
this calibration. This selection initializes the single runtime CM control,
not a second source of operating values.
An existing instrument intentionally retains its live absolute CM excitation
when changing Microprobe/Nanoprobe modes: applying a lens preset alone does not
reset it. To adopt this working point in an existing assembly, set the Mini
Condenser magnitude to **43.88646617222637%**, select **Nanoprobe**, and apply
the lens preset. Nanoprobe uses the negative field direction; Microprobe uses
the positive direction at the same magnitude. Use **Custom** optics and
3,000 rays with a 0.05 mm or finer step to compare the nonlinear probe metrics.
For the static figures above, disable the raster and wobble drives using their
existing controls. Inspect Z near 1400.00070 mm in **Ray Diagram**, then the
sample entrance and **Scanning Image → Probe Aberrations**.

For a reproducible fresh-catalog check without modifying instrument settings:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_probe_relay.py --rays 3000 --figure
```

This requires an available CUDA GPU and records the actual column backend;
it does not silently accept CPU fallback. It saves `report.json`, `inputs.json`
and the optional `validation.png` under `outputs/probe_relay_validation`.
The report includes the input identity, source-bundle identity, HP transfer,
executed crossover checkpoints, sample metrics and explicit pass/fail checks
at 0.05 and 0.025 mm. It neither refits nor saves new catalog parameters. Add
`--compare-aberrations` only when the separate identical-source linear / Cs-only
/ Cs-plus-hexapole comparison is required; that adds three column traces.

Microprobe was rechecked at the same positive CM magnitude with C2
**25.32063230456724%** and C3 **33.43002947440911%**, preserving its other lens
settings. Its 3,000-ray Custom/CUDA calculation at the finer step gives
**D95 = 2.000144 µm**, **alpha95 = 0.265107 mrad**, and all 3,000 rays surviving.
This checks compatibility of the 2 µm default, not every requested Microprobe
diameter. Projection-mode lens settings remain unchanged.

### AC placement and finite scan validation

The **C3 + Probe Corrector** installation now places the AC pair around the
executed TL1-to-CM source waist. Only this assembly's AC position changed:

| Component or observation | Absolute Z (mm) |
|---|---:|
| TL1 hardware exit | 1379 |
| AC upper coil body / centre | 1380–1395 / 1387.5 |
| Source waist, scan disabled | approximately 1400.0007 |
| AC lower coil body / centre | 1405–1420 / 1412.5 |
| Next condenser stigmator entrance | 1436.5 |
| Mini Condenser body / centre | 1512.5–1532.5 / 1522.5 |

The 40 mm pair moved upstream by 91.5 mm. It has 1 mm axial clearance after
TL1 and 16.5 mm before the next stigmator; the two 15 mm coils retain their
10 mm gap. Actual assembly meshes retain both complete coil bodies. This is
an envelope/layout check, not measured OEM coil geometry or winding clearance.
With these adjacent envelopes and the same pair dimensions, simple non-overlap
permits a pair centre between 1399 and 1416.5 mm (endpoints touch). Placing the
waist inside the inter-coil gap further restricts that interval. This screening
does not qualify every position; the selected centre **1400 mm** was executed
and validated. No round-lens, HP, Objective or Tip parameter was changed by
this scan relocation.

A source waist and a scan pivot are different constraints. The waist near
1400 mm is imaged onto the specimen; tilting about that same conjugate source
plane alone would not provide the requested specimen translation. The two
finite coils instead act together to produce a common **effective scan pivot**
in the downstream installed field near **1572.664866 mm**. A second verified
root occurs near **1593.206504 mm**. At both roots the canonical transfer to the
sample satisfies `D_can ≈ 0` while `B` is nonzero. The independently checked
`||D_can||₂` values are approximately 1.5e-8 and 5.0e-9. These are internal
effective pivots, not a claim about the unique field-free cardinal front focal
plane of an isolated CM/Objective assembly. The source image near the scan pair
and the distinct objective aperture/field-ray plane are consistent with the
topology discussed by [Müller et al., CEOS, 2006](https://www.ceos-gmbh.de/de/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf).

Automatic calibration now holds **canonical** sample direction fixed, matching
the diffraction-plane convention. The actual mechanical direction remains
reported and can change inside the Objective magnetic field. Old held records
retain their stored commands and do not silently acquire the new constraint.
Select Automatic recalibration or Calibrate and hold to adopt it.

The new optical validation used the same 3,000 physical Tip rays, Custom
fields including Cs and HP, one executed incident prefix, and five frozen scan
states: the centre and four corners of a **1 µm × 1 µm** FOV. Descan was off.
GPU receipts confirm CUDA for the post-gun paths; the existing gun path is CPU.

| Executed quantity | Result |
|---|---:|
| Surviving rays, every scan state | 2401 / 3000 |
| Sample geometric RMS radius across scan states | 0.125873–0.126449 nm |
| Largest sample-position error | 0.009699 nm |
| Largest centroid motion at the first effective pivot | 0.033203 nm |
| Objective diffraction observation Z | 1605.193496 mm |
| Largest centroid motion there, Descan off | 0.033743 nm |
| HP2 → HP1 `||B||₂` | 5.17e-14 m/rad |
| Largest physical coil-axis drive at this FOV | 0.211643 mrad |

Calibration uses specimen centre 1599.2 mm; the physical probe is measured at
the entrance 1599.1999975 mm. Refining only the post-AC trace from 0.05 to
0.025 mm, with the identical already-executed 0.05 mm upstream prefix, changed
the compared centroids by at most 0.0000102 nm and sample RMS by 0.00000257 nm.
The separate relay validation above supplies the upstream refinement check.
These are five frozen particle states in vacuum, not a complete raster,
specimen diffraction-intensity or coherent-image validation. Screen stability
at another installed receiver still requires its own conjugacy check.

Reproduce the bounded check with an available CUDA GPU:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_ac_scan_pivot.py --fov-nm 1000
```

The script saves exact inputs, backend receipts, pass/fail results and a plot
under `outputs/ac_scan_pivot_validation`; it does not apply settings to a live
GUI. The completed 10 October run is in `outputs/ac_scan_pivot_20261010`.
Its original `report.json` is retained; `report.reviewed.json` adds stricter
scan-radius/diffraction/step gates and independent pivot transfer checks using
the same saved measurements, with both implementation identities recorded.
All eleven reviewed checks passed. Adopt the geometry by restarting and
assembling **C3 + Probe Corrector**; merely reapplying a lens preset does not
rebuild installed hardware positions.

### Further adjustments

The old Direct Alignment interpolation table and approximate
0.25 mrad/µm aperture scaling belonged to the previous relay. They have been
removed. Direct Alignment retains its general solver, using the current optics
and deterministic candidate search followed by physical validation. The 30 mrad
default does not demonstrate that every aperture/convergence combination in the
editable range is reachable or corrected.

The separate `scripts/calibrate_nanoprobe_corrector.py` tool fits a new
hexapole/focus candidate; its report is not a replacement for the relay checks
above. It does not overwrite instrument configurations, and `--skip-focus`
produces diagnostic candidates only. Changes to source, voltage, aperture,
convergence or relay lenses require renewed focus, crossover, conjugacy and
aberration checks before new defaults are accepted.

## Integration consistency

Post-gun propagation advances canonical transverse momenta with RK4 and
evaluates the continuous magnetic and multipole fields at the actual interval
nodes and midpoints. This removes numerical differentiation of Bz from the ray
equations. Thin lens and aberration actions occur at their physical planes.
NumPy, Numba and CUDA use the same step formula; public results retain physical
ray slopes. A finer step now verifies the fitted optics instead of moving the
focus by a large discretisation error.

## DPH response with the revised Ideal Optics definition

On 10 October 2026, Ideal Optics was changed to retain configured HP, Cs,
chromatic and actual-energy effects. Its distinction is the use of parameterised
fields rather than a material/pole-piece field solution; see
[Simulation model levels](SIMULATION_MODES.md). No source or optical strength
was retuned for the following check.

`scripts/validate_dph_aberrations.py` executed 26 downstream CUDA comparisons
in 51.1 seconds. The completed record is
`outputs/dph_aberrations_20261010/gpu_run2/report.json` (PASS). It uses an
explicit historical entrance population, with 2,401 common surviving rays
from the original 3,000 source rays. Every comparison uses the same ray IDs
and current weights; no new clipping occurred. This is a conditional downstream
check, not a newly executed gun or complete coherent-wave calculation.

Each DPH was changed by +/-0.05 mrad independently along X and Y. Checkpoints
cover the full HP field, its body entrance/centre/exit, and the specimen. Each
distribution is centred separately before comparing its shape. Temporarily
setting only the target HP strength to zero provides a counterfactual control;
its hardware, the other HP and all configured Cs terms remain active.

| Observation | DPH2 / HP2 | DPH1 / HP1 |
| --- | ---: | ---: |
| Beam-centre displacement at HP centre | approximately +/-25 nm | approximately +/-25 nm |
| Target-HP-dependent centred specimen response, X/Y | 0.184981-0.184989 nm RMS | 0.271473-0.271475 nm RMS |
| Specimen radial RMS, baseline | 0.125873 nm | 0.125873 nm |
| Specimen radial RMS with the DPH perturbation | 0.23673-0.23883 nm | 0.29821-0.30058 nm |

The target-HP-dependent response is the difference of the centred odd DPH
responses with that HP on/off. It includes subsequent propagation through the
remaining optics; it is not an independently measured local aberration
coefficient. A displaced hexapole produces constant and quadrupole terms in
addition to its original nonlinear field. Therefore a changed centred shape,
including affine astigmatic distortion, is relevant; a purely non-affine
residual is not required to establish this effect.

For DPH2 +X, refining the axial step from 0.05 to 0.025 mm changed the
target-HP-dependent centred response by 6.17e-18 m (relative 3.33e-8).
This refinement qualifies that specimen response only, not every reported
axis or local slope. Ideal and Custom, using the same analytic inputs, produced
exactly equal saved coordinates, angles, energies and flight times for the
baseline and +X cases. Their numerical propagation-plan arrays also agree,
with nonzero HP and Cs operators.

All 26 execution receipts report CUDA GPU. Before/after source inventories and
historical input hashes are unchanged. Complete input graphs differ only in
the actual-backend readout, which changes from CPU to CUDA GPU; the requested
backend and all physical controls are identical. The earlier `gpu_run1`
failure is retained: its final audit incorrectly treated that execution
readout as a physical input change. The corrected audit saves both raw graphs
and excludes only this named readout, without weakening physical-input checks.

The saved `dph_inside_hp.png` shows the internal passage displacement;
`dph_sample_shape.png` shows centred specimen distributions. Local regressions
also compare finite dipole/HP propagation with independent DOP853 integration
and check the complete translated cubic wave phase, including its constant,
tilt and curvature terms. The compact-ring C3 diagnostic now uses the same
active DPH events in its three comparison traces; it remains an approximate
diagnostic rather than a complete fitted aberration tensor.
