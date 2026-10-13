# B-COR additional image-corrector assembly

The selectable B-COR configuration is an engineering reconstruction of the
three-hexapole optical topology. It is separate from the existing two-hexapole
CETCOR configuration. It does not establish that the model reproduces a
commercial B-COR's C3, B3, C5 or field-of-view performance.

## Sources and what they establish

- [CEOS B-COR poster](https://www.ceos-gmbh.de/user/pages/02.products/Poster-internet.pdf):
  the ordering of the objective, TL11/TL12, HP1, TL21/TL22, HP2,
  TL31/TL32, HP3 and adapter lens. The poster distinguishes axial and
  off-axis correction. Its ray diagram is not a dimensioned manufacturing drawing.
- [CEOS patent EP1941531A2](https://patents.google.com/patent/EP1941531A2/en),
  especially Figs. 1–2 and the associated description: three principal
  hexapole fields, paired imaging between their reference planes, and relative
  field orientation specified in the rotating Larmor frame. It permits sixfold
  multipoles and compound round-lens relays. It does not provide the strengths
  or mechanical dimensions needed to reproduce the commercial instrument.
- The existing CETCOR/CESCOR family references in `cad-image-corrector-channels.md`
  support the generic pole-pin, bulk-winding and return-yoke construction.
  Reusing that construction for B-COR is a declared family inference; it is not
  a verified B-COR cross-section.

The original poster and patent-page snapshot are stored under
`cad-library/B-COR/`; `source.json` records URLs, byte counts and SHA-256 values.

## Mechanical allocation

Two new modules preserve the previous source, condenser, probe corrector and
objective parameters: `C3_BCOR.toml` and `C3_ProbeCorrector_BCOR.toml`.
The new TL31 and TL32 use the same 180 mm outside diameter, 36 mm axial envelope,
pole, yoke, housing and bulk-winding detail as TL21 and TL22. HP3 uses the same
180 mm outside diameter, 44 mm envelope and 30 mm field length as the retained
HP1/HP2 reconstruction. The three HP centres are spaced by 144 mm. These values
are simulator allocations, not measured CEOS dimensions.

Adding the second relay moves the adapter lens, exit controls, SAD plane and
column exit downstream by 144 mm. Module assembly then translates the complete
projector/recording module from the new exit. No upstream optical coordinate or
strength is changed merely by selecting B-COR.

No DPH3, DP31/DP32 or extra support has been invented. Existing DPH1/DPH2 and
other two-hexapole-model steering inserts remain available in their allocated
locations; their presence and exact placement in commercial B-COR are not
independently established by the poster. Their reconstruction provenance is
retained. They should not be interpreted as a manufacturer parts list.

The new round-lens material components and HP3 use the same analytic solid
definitions for STEP export and native 3D rendering. Material placement follows
the existing mechanical rules; optical tilt does not rotate the material shell.

## Optical qualification boundary

TOML stores each principal hexapole's strength and laboratory-frame orientation.
Laboratory angles must be interpreted together with the integrated Larmor
rotation between planes; copying a diagram's drawn rotation is insufficient.
The extra hardware is functional, but a first-order relay check alone does not
qualify nonlinear spherical-aberration or off-axis-coma cancellation. No
manufacturer correction performance is claimed.

## Selecting the assembly

Open **Configure instrument**, leave **C3 lens** installed, enable
**Image corrector**, choose **B-COR · 3 hexapoles**, then click **Assemble**.
The probe corrector is independent: keeping it installed selects
`C3 + Probe Corrector + B-COR`; otherwise the selection is `C3 + B-COR`.
The existing CETCOR option and the startup assembly remain unchanged.

Particle and coherent propagation consume the same installed TL31, TL32 and
HP3 components. There is no separate B-COR wave-only strength or GPU selector.
Changing an active HP3 strength changes the executed complex wave, not only a
drawn component. The original source parameters and upstream probe optics are
retained by the configuration dialog.

## Default first-order working point

The two new B-COR manifests contain the following **300 kV simulator working
point**. These are not CEOS coil-current measurements. The existing CETCOR
manifests are unchanged.

| Lens | Default excitation (%) |
| --- | ---: |
| TL11 | 47.22262895320412 |
| TL12 | 72.24615725655114 |
| TL21 | 60.00013491700115 |
| TL22 | 60.00017147578498 |
| TL31 | 60.0001347376614 |
| TL32 | 60.00017190693384 |
| ADL | 56.28443499997931 |

`scripts/solve_bcor_relay.py` generated candidates with a first-order Larmor
model and checked a halved integration step. A separate actual-CUDA
laboratory-frame finite-difference propagation in `scripts/validate_bcor.py`
then verified the saved defaults of both assemblies. HP1→HP2 and HP2→HP3
have position-transfer `||B||₂ < 2.34e-11 m/rad` and unit-position-magnification
errors below `2.8e-10`. The sample-to-SAD position-transfer residual is below
`2.9e-12 m/rad`. The checks use a 0.025 mm integration step and a
`1e-7 m/rad` conjugacy limit; local receipts are in
`outputs/bcor-20261013/final-gpu-relays/`. The validator explicitly disables
the analytic axisymmetric shortcut for this independent GPU measurement.

These are **position conjugacies**, not full negative-identity maps or proof
of nonlinear cancellation. The laboratory rotation is retained. In particular,
the mechanical midpoint between TL31 and TL32 is not a sample-conjugate image
plane at this working point (`||B||₂ ≈ 4.29e-3 m/rad`); it is recorded as a
diagnostic, not marked as a conjugate plane. The model therefore does not
claim to reproduce every fundamental-ray relation in the CEOS poster.
The three finite hexapole strengths/orientations remain explicitly provisional.

Newly installed hardware takes the selected manifest defaults. Previously
installed and edited component drives are retained when switching assemblies;
in that case the above default-working-point validation does not apply to
the edited working point. Saving/restoring an operating profile retains its
chosen corrector and settings.

The system-residual observer now samples the installed image corrector's SAD
output rather than stopping at the upstream objective image. Its coefficients
remain sample-referred. This axial-pupil observer does not measure off-axis
coma, and enabling a corrector is not labelled as verified correction.

## Geometric checks, 2026-10-13

- The focused hardware regressions cover both manifests, unchanged
  upstream parameters, ordered optical references, complete added lens material,
  HP3 native mesh generation and common native/STEP solid definitions.
- Actual CadQuery 2.6.1 export and STEP re-import passed for TL31, TL32 and HP3
  and their aggregate: four STEP files, 23 analytic solids. The aggregate file
  contains the three additions only, not a complete microscope assembly.
- OpenCascade checked the complete image-corrector material section in each
  of the two new variants: 106 solids per variant, 476 evaluated pairs total,
  zero material-volume collisions, 24 contacting pairs per variant, and
  0.16 mm minimum positive clearance. Unchanged upstream parts and physical
  vacuum liners were outside this bounded exact-solid audit. This is geometric
  validation of the declared reconstruction, not manufacturer qualification.
- Local receipts are in `outputs/bcor-hardware-20261013/step/step-manifest.json`
  and `outputs/bcor-hardware-20261013/collision/report.json`.

## Executed software and device checks

The final focused run passed 253 cases with zero failures/errors/skips; the
receipt is `outputs/bcor-20261013/final-tests.xml`. This covers installation,
switching, dormant-state/profile preservation, malformed manifest rejection,
shared field action, CAD construction, residual-plane selection and CI scope
registration. An additional final run passed 32 existing profile-contract
cases. GUI selection checks used Qt's offscreen platform.

On an actual NVIDIA GeForce RTX 5090, CPU versus CUDA propagation from before
HP1 through the B-COR SAD output differed by at most `5.58e-19 m` in position
and `1.39e-17 rad` in angle for the five diagnostic rays. A coherent HP3-local
continuation differed by `6.82e-16` in relative complex-field norm. These are
bounded backend-parity checks, not a full source-to-detector coherent run or
a demonstration of corrected imaging over the field of view.
