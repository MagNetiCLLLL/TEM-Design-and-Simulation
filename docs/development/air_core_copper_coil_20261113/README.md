# Air-core copper-coil round-lens model: sources and reference inputs

Date: 2026-11-13. This document defines the geometry/current inputs of an independent air-core round-lens design model. It does not identify an OEM winding, qualify an installed operating preset, or replace the existing Gaussian calibration. No new finite-source or coherent calculation was performed to prepare this metadata.

## Physical scope

An explicitly declared circular excitation winding is represented by a continuous, uniform azimuthal current density over its rectangular radial/axial cross-section. The surrounding magnetic permeability is that of vacuum. Pole pieces, yokes, saturation, hysteresis, copper resistivity, eddy currents and thermal response do not alter this field model. Passive material and vacuum passage still define whatever mechanical interception the selected particle executor supports.

Coil inner and outer diameters locate the current source; the vacuum-tube inner diameter limits the accessible beam region. They are separate inputs. At fixed outer diameter, changing the winding inner diameter changes both the occupied current region and its radial thickness. No Gaussian width is inferred from that change.

The new selection applies to round lenses only. Corrector hexapoles, quadrupoles, stigmators, scan/alignment deflectors, filter prisms and electrostatic devices retain their declared native field models. A circular solenoid would have the wrong symmetry for these elements. Their existing winding details are not evidence for a substitute axisymmetric field.

The existing project pose convention is retained: owner optical X/Y tilt perturbs the magnetic-source axis, while the material CAD does not acquire that physical tilt. This is an optical-axis misalignment model, not a mechanically tilted copper-winding reconstruction. Independent coil CAD offsets, rotations or noncircular deformations that cannot be expressed by the declared round source are rejected. The controlled study below uses centred, untilted windings.

## Selecting the model

1. In **Simulation > Finite-Field Optics**, select **Shared potentials and geometry**.
2. In **Simulation > Column round-lens field**, select **Copper coils in vacuum (ignore pole pieces)**. The same choice is available in the lens-model inspector. This source currently supports classical particles; coherent entry points reject it explicitly before propagation.
3. For the separate Gaussian option, select the Gaussian source and use **Simulation > Gaussian geometry relation > Link Gaussian widths to coil length**. The historical default is **Fixed Gaussian calibration**.

The Gaussian rule is `a_actual = a_saved * L_current / L_reference`, using the sum of authoritative current-bearing winding intervals. It keeps optical profile centres, Gaussian term amplitudes, excitation and polarity fixed. Objective upper/lower lobe overlap can therefore change the combined peak. It does not conserve the field-squared integral or refocus the beam. Coil radial dimensions do not determine Gaussian width under this deliberately axial-only rule; the copper-coil source resolves their field effect directly. Explicit FEM/imported-field owners remain independent of this Gaussian rule.

Both choices persist through saved states, operating profiles, immutable instrument snapshots and worker input serialization. Numerical accuracy controls do not switch field laws. The initial reference ampere-turns are editable assembly inputs; existing excitation controls scale them. An old Gaussian operating point is not automatically retuned on selection.

The completed [single-coil intrinsic-aberration study](../copper_coil_aberration_20261113/README.md) covers seven declared geometry/current cases and one finite-source half-step repeat, each with 1024 original positive-current particles followed by a transported 1024-ID append.

## Public evidence and its limits

| Primary source | What it supports | What it does not supply |
| --- | --- | --- |
| [MIT, *Electromagnetic Fields and Energy*, section 8.2, example 8.2.1](https://web.mit.edu/6.013_book/www/chapter8/8.2.html) | Biot-Savart superposition and the finite circular-solenoid axial field; finite-coil fringe fields are part of the solution. | An electron-microscope winding specification or an instrument current calibration. |
| [FEI patent US20150021476A1, embodiments 1-2 and figures 1-3](https://patents.google.com/patent/US20150021476A1/en) | An actual charged-particle lens distinguishes the main electric coil, magnetic yoke/insert and beam bore; it also describes an auxiliary adjustment coil. Its focusing mechanism includes magnetic saturation. | The present simulator's coil dimensions, NI values or a claim that omitting the magnetic body reproduces that patented lens. |
| [JEOL patent DE3139905A1, figures 2 and 4](https://patents.google.com/patent/DE3139905A1/en) | Explicit NI is used as a lens excitation quantity. The documented 100 kV three-pole example discusses values near 2700-4000 ampere-turns and unequal excitations in two gaps. | A universal TEM current, a 300 kV air-core calibration, or the turns/current of the simulator's coils. |
| [CEOS, *Basics*](https://www.ceos-gmbh.de/en/basics) and [STEM aberration-correctors](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren) | Spherical-aberration correctors combine non-round multipoles with additional round lenses; the two categories cannot be interchanged. | Public winding counts, conductor currents or dimensional drawings for the simulated corrector's individual coils. |

The project uses **3000 ampere-turns at 100% excitation as an explicit engineering initialization for every round owner**. This is a chosen common design input, not a value extracted from the FEI patent or a calibration transferred from the JEOL example. It was not inferred by matching an old Gaussian peak. Public material found in this review does not establish the actual N, conductor I, packing factor, resistance, cooling or safe duty cycle of the modelled instrument. Those quantities are not fabricated.

## Input contract and normalization

The existing part geometry remains authoritative:

| Location | Field | Meaning |
| --- | --- | --- |
| Round-lens owner | `air_core_reference_ampere_turns` | Positive total NI of this owner at 100% model excitation. Bundled initial value: 3000.0 A turns. |
| Round-lens owner | `air_core_reference_status` | Bundled value: `engineering_reference_not_oem_not_particle_qualified`. |
| Round-lens owner | `air_core_reference_source`, `air_core_reference_source_urls` | Provenance and explicit distinction between the physical law and the chosen numeric initialization. |
| Round-lens owner | `gaussian_reference_coil_length_mm` | Frozen original total current-bearing axial length used only by the separately selected Gaussian axial-length coupling. It is not a magnetic-field measurement. |
| Excitation-coil child | `field_source_key` | The independently driven round-lens owner. Nested Mini Condenser and Objective remain separate owners. |
| Excitation-coil child | `coil_ampere_turn_fraction` | Positive fraction of its owner's NI. All fractions of one owner sum to one. Existing bundled owners each have one coil record and fraction 1.0. |
| Excitation-coil child | `mechanical_inner_diameter_mm`, `mechanical_outer_diameter_mm` | Winding-region boundaries. Radial thickness is half their difference. |
| Excitation-coil child | Existing physical axial intervals | Current-bearing axial support. Objective split layers take precedence over the historical child display envelope. |

The current convention is

`NI_owner = enabled * polarity * (percent / 100) * air_core_reference_ampere_turns`.

For coil share `f`, inner radius `Ri`, outer radius `Ro` and total occupied axial length `L`,

`J_phi = f * NI_owner / [(Ro - Ri) * L]`.

The cross-section here is radial thickness times occupied axial length, not the annular area pi*(Ro^2-Ri^2). NI already accounts for complete turns around the axis. Multiple separated axial intervals divide that same coil's total NI through the uniform cross-section measure; they do not each receive a fresh copy of the owner's entire NI.

With coil centre at zero, a uniformly filled annular winding has axial field

`B0(z) = mu0 * NI / [2 L (Ro-Ri)] * integral_Ri^Ro [(z+L/2)/sqrt(R^2+(z+L/2)^2) - (z-L/2)/sqrt(R^2+(z-L/2)^2)] dR`.

Separate intervals/coil records superpose with their declared shares. The on-axis formula is also available as an independent validation expression in `reference_benchmarks.finite_solenoid_axis`; its fixed quadrature order is not by itself a convergence certificate for every geometry. Off-axis fields must follow the same physical current source, rather than an unrelated Gaussian fitted after a diameter edit. Finite field tails must not be silently clipped at a display support face.

At fixed Ro, L and NI, increasing Ri removes the more central turns from the source distribution and raises J_phi in the remaining thinner cross-section. The central field decreases. Holding J_phi fixed instead reduces total NI as the cross-section shrinks. Holding central B fixed requires a separately recorded change of NI. FWHM, fringe extent and aberrations are outputs to calculate, not universal diameter-scaling rules.

## Round-lens coverage

All twenty distinct installed round-owner keys already had explicit circular excitation-coil geometry. No duplicate coil was added. The seven column variants contain 70 stored owner/coil records; the standalone recording template and shared projector stack contribute four each. The shared stack supplies recording assemblies that reference it. These are storage counts, not 78 simultaneous lenses in one microscope.

| Stored TOML | Round owners | Existing coil records | Missing geometry/reference after update |
| --- | ---: | ---: | ---: |
| `configs/instruments/column/C2.toml` | 4 | 4 | 0 |
| `configs/instruments/column/C3.toml` | 5 | 5 | 0 |
| `configs/instruments/column/C3_ProbeCorrector.toml` | 9 | 9 | 0 |
| `configs/instruments/column/C3_ImageCorrector.toml` | 10 | 10 | 0 |
| `configs/instruments/column/C3_ProbeCorrector_ImageCorrector.toml` | 14 | 14 | 0 |
| `configs/instruments/column/C3_BCOR.toml` | 12 | 12 | 0 |
| `configs/instruments/column/C3_ProbeCorrector_BCOR.toml` | 16 | 16 | 0 |
| `configs/instruments/project_and_recording_system/NoEnergyFilter.toml` | 4 | 4 | 0 |
| `configs/subassemblies/projector_stack.toml` | 4 | 4 | 0 |

Covered keys are C1 lens, C2 lens, C3 lens; probe ADL/`adapter_lens`, TL12, TL21, TL22; Mini Condenser; Objective; image TL11, TL12, TL21, TL22 and adapter; B-COR TL31/TL32; Diffraction, Intermediate, Projector 1 and Projector 2. The ADL alias is one existing driven round lens, not a reason to fabricate a second source.

The Objective provides an important geometry distinction: in C2, its legacy child display envelope is 162 mm, but the authoritative upper/lower current-bearing layers have total length 290 mm. `magnetic_geometry.objective_layer_intervals_mm` is shared with the existing CAD/geometry-field binding and supplies those intervals. The frozen Gaussian reference length uses the same total; it is not taken from the display envelope.

The machine-readable [coil-coverage.json](coil-coverage.json) records all 78 owners, the existing coil IDs, dimensions, physical intervals and input hashes. Regenerate it without solving a field or transporting particles:

```text
python scripts/audit_round_coil_metadata.py --output docs/development/air_core_copper_coil_20261113/coil-coverage.json
```

## Compatibility and qualification

The metadata update preserves every pre-existing parsed TOML value. Gaussian default selection, field shapes, excitations, geometry and old result records are not retuned by adding a reference current. Old documents without these optional fields remain readable; selecting the new model requires explicit current/geometry and must not invent missing inputs.

The new model changes the field distribution and generally changes focus, rotations, crossover locations, transmission and aberrations. An operating preset tuned with the old Gaussian is not thereby qualified under the air-core field. All new finite-source physical tests still require at least 1000 original positive-current particles, preserved clipping/current ledgers and appropriate sampling/step checks. New coherent calculations retain the repository's usable-CUDA requirement.

Static metadata tests cover complete bundled references, valid physical intervals, split Objective layers, rejected invalid NI, incomplete metadata, missing coils, incorrect shares and attempted multipole reinterpretation. These tests establish input consistency only; they are not finite-source, measured-field, OEM or thermal qualification.

The retained [software-validation summary](validation/summary.json) and JUnit reports cover the new field, geometry/current ownership, serialization, mapped-field independence, model controls, profiles, parameter explanations and editor behavior. The legacy product-quadrature profile fixtures now select their intended source schema explicitly. Navigation tests use synthetic zero-current display lines and forbid physical propagation; they no longer launch a nine-particle preview. This bounded validation is separate from the numerical sensitivities of the finite-source research and does not claim a complete repository-suite run.
