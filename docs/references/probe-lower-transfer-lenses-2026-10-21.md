# Internal structure review of TL21 and downstream TL12

> Follow-up update: this document retains the inspection baseline and validation records from before the three copper-end contact fixes. TL22–DP22, DP21–TL21, and DP11–TL12 were subsequently addressed with explicit copper winding lengths, insulation at both ends, and positive assembly gaps; see the [end-insulation and clearance fix record](probe-contact-isolation-2026-10-21.md) for the current structure and evidence from that work. References below to unresolved contacts and retained copper describe the earlier stage only.

The scope is TL21 and TL12 below TL22 (saved key `probe_tl12_lens`, abbreviated TL1 in earlier material). All lengths are in module-local mm, with +Z downstream. Add 450 mm to convert the default assembly to global coordinates. This work only adds passive mechanical structure, without changing lens positions, pole gaps, excitation directions, strengths, or propagation formulas.

## Sources and structural limits

[Müller et al. 2006, Fig.2, pp.443–444](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf) distinguishes the TL21/TL22 transfer doublet between the hexapoles from the downstream TL11/TL12 transfer system, but does not provide material manufacturing dimensions for this project's lenses. [Hitachi JP2967966B2, conventional Fig.8, [0005]](https://patents.google.com/patent/JP2967966B2/en) describes metal cooling plates at both winding ends and electrical insulation between the winding and plates. This work uses that generic round-lens structural family; **it is not certification of CEOS OEM internal construction**.

All five existing material parts are retained: upper/lower pole pieces, the bulk excitation winding pack, the return yoke, and the housing. Each housing gains an insulation layer and a passive copper plate at each end, totaling four independent regions. They share the `SolidSpec`, native 3D, and analytic STEP paths; plate copper is not part of the excitation winding and contributes no additional superposed field. Mechanical fields are saved as `mechanical_transfer_end_*`; each lens stores its own winding interfaces and must not reuse TL22's winding OD or absolute Z.

## Saved dimensions and end space

| Item | TL21 | TL12 |
| --- | ---: | ---: |
| Housing axial range | 787.2–812.7 | 903.6–929.0 |
| Original winding axial range | 788.475–811.425 | 904.87–927.73 |
| Original winding ID / OD | 78.33 / 93.3164759 | 78.33 / 93.63 |
| Original space at each end | 1.275 | 1.27 |
| New insulation thickness | 0.1 | 0.1 |
| New copper-plate thickness | 0.5 | 0.5 |
| Remaining space beyond the new layers at each end | 0.675 | 0.67 |

The 0.1/0.5 mm values are explicitly saved engineering allocations, not measurements from papers or photographs. Insulation and copper plates have the same radial ranges as their respective original windings; TL21's OD93.3164759 and TL12's OD93.63 both differ from TL22's OD93.5109425.

| Region | TL21 local Z | TL12 local Z |
| --- | --- | --- |
| Upper copper plate | 787.875–788.375 | 904.27–904.77 |
| Upper insulation | 788.375–788.475 | 904.77–904.87 |
| Lower insulation | 811.425–811.525 | 927.73–927.83 |
| Lower copper plate | 811.525–812.025 | 927.83–928.33 |

Both pole-piece bores remain 20, the actual continuous vacuum-tube OD is 19.2, and nominal radial clearance is **0.4**. The actual tube ID is 5.76; the declared clear beam bore of 20 must not be mistaken for the actual tube ID. New plate layers lie outside radius 39.165 and do not intrude into the tube or original pole pieces.

## Retained existing contacts and structural gaps

- The downstream copper end face of DP21 contacts the upper TL21 pole piece at Z787.2.
- The downstream copper end face of DP11 contacts the upper TL12 pole piece at Z903.6.

These are **zero-gap material-face contacts**, not positive mechanical clearances or established electrical isolation. They were not concealed by moving optical parts, shortening copper winding packs, enlarging pole bores, or adding overlapping insulation. The new insulation at the outer winding ends cannot resolve contacts between deflector winding packs and pole pieces near the beam axis.

The original TL21 yoke ID101.3164759 and TL12 yoke ID101.63 still lack a confirmed radial magnetic-circuit connection to the pole shoulders of OD66.33. The new copper plates are not ferromagnetic return-path connections. Water channels, screws, individual turns, leads, host mounting, and material thermal/electrical/magnetic parameters remain unreconstructed; no improvement in cooling, eddy-current response, or spherical-aberration performance is claimed.

## Dimension coupling and validation scope

Winding ID/OD edits update the corresponding end-plate ID/OD; pole-gap edits preserve the optical center and pole-piece rear faces while updating the gap-facing end faces. Copying the parent component remaps coil references, and moving it moves all saved interfaces together; copying a housing alone preserves self-contained structure, with references serving only as provenance. Parameters use the same dimension, provenance, and impact descriptions as other components. Enabling an explicit Box/STEP replacement disables the native recipe; restoring it triggers validation, and failures do not corrupt the draft or history.

The dedicated regression file is `tests/test_cad_probe_transfer_lens_hardware.py`, covering configuration scope, preservation of the five original material regions, all 36 internal material pairs, closed meshes, end-face positions, actual tube-wall clearance, existing copper contacts, dimension coupling, copy/save/undo, replacement precedence, and invalid parameters. Tube-wall checks with 32/64/128 segments are compared with the analytic 0.4 mm cylindrical clearance; faceted-surface results must not be treated as OEM assembly tolerances.

The review also fixed a floating-point boundary issue in the generic end-layer recipe: a very small positive layer thickness can round to zero at the current nonzero Z coordinate, whereas checking only total stack thickness previously accepted a degenerate individual layer. Each boundary of the four layers is now checked for strict ordering using the actual geometry-construction expressions. Configuration validation and edit submission reject unrepresentable thicknesses without artificially increasing the physical minimum thickness or changing normal configurations.

The **47 dedicated regressions passed**, and **130 tests passed** when combined with the existing C3, Adapter, and TL22 suites. The latest receipt is `outputs/probe-lower-details-20261021/lens-layer-boundary-regression.xml` (the earlier 37-test record is retained as history). The same directory's `step/step-manifest.json` records successful readback of 9 analytic STEP files for downstream components and the vacuum tube; the existing export validator checks solid counts, volumes, bounds, names, materials, and colors. In the isolated CAD environment, `test_cad_step.py`, `test_cad_kernel.py`, and `test_cad_assembly_validation.py` had **26 passes**, with per-case XML records saved in `exact-cad-regression.xml` and `exact-cad-assembly-regression.xml`.

These checks establish consistency of saved geometry and software paths, not validation of physical instrument dimensions, thermal behavior, electrical insulation, or complete magnetic-field FEM.
