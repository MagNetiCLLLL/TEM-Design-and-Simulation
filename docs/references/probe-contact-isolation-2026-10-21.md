# Insulation and Clearance Revision for Three Probe Corrector Copper End Contacts

This round addresses end contacts at TL22–DP22, DP21–TL21 and DP11–TL12 (stored key `probe_tl12_lens`). Following the [previous internal-structure review](probe-lower-detail-review-2026-10-21.md), it uses native material models, existing CAD / STEP formats and background validation, without desktop interaction. Coordinates are column-module local Z in mm, positive downstream along the electron beam; add 450 mm to the tables for default global positions.

## Evidence and design choice

[CEOS US20090079531A1](https://patents.google.com/patent/US20090079531A1/en) Fig.4 and its description establish conductor-cover insulation and full-surface bonding for flexible windings. Fig.10 and 11 show multipole-coil structural families mounted in lens bores or recesses and note that larger recesses may change lens fields. They do not establish factory end seals, dimensions or assembly tolerances for this project's DP22/DP21/DP11.

[Hitachi JP2967966B2](https://patents.google.com/patent/JP2967966B2/en), conventional structure Fig.8 and [0005], describes a thin insulating layer between a lens excitation coil and a copper cooling plate. This only supplements evidence for insulation separation, not a specific DP end-seal shape or material grade.

This round symmetrically shortens mechanical copper windings about fixed installation centres and completes existing insulation at both ends. It reuses `winding_axial_length_mm`, without duplicate cross-component dimensions, pole-shoe machining or lens magnetic-gap changes. Copper retracts 0.14 per end: 0.04 for insulating edge seals and 0.10 for nominal mechanical clearance. These are **explicit engineering reconstruction dimensions, not OEM measurements or voltage-withstand design results**.

The alternative of recessing pole shoes was not adopted: although supported as a structural family, it changes magnetic material, whereas the present issue can be resolved using the existing independent mechanical copper-length parameter. No unverified fasteners, connectors, wiring or mounts are added.

## Material boundaries before and after modification

| Component | Original copper interval | Revised copper interval | Revised full insulation interval | Actual copper length |
| --- | --- | --- | --- | ---: |
| DP22 | 729.10–743.60 | 729.24–743.46 | 729.20–743.50 | 14.22 |
| DP21 | 772.60–787.20 | 772.74–787.06 | 772.70–787.10 | 14.32 |
| DP11 | 889.00–903.60 | 889.14–903.46 | 889.10–903.50 | 14.32 |

| Interface | Original pole-shoe contact plane | Original copper-to-pole gap | Revised copper-to-pole gap | Revised insulation-to-pole gap |
| --- | ---: | ---: | ---: | ---: |
| TL22 lower pole shoe—DP22 upper end | 729.10 | 0 | 0.14 | 0.10 |
| DP21 lower end—TL21 upper pole shoe | 787.20 | 0 | 0.14 | 0.10 |
| DP11 lower end—TL12 upper pole shoe | 903.60 | 0 | 0.14 | 0.10 |

Values come from saved configuration and the common material-boundary function. Full insulation intervals include copper, radial carriers, covers, bonding regions, conductor-gap insulation and both edge seals; copper envelopes alone cannot establish absence of intersections.

All three copper layers retain radii 11.10–11.90, and complete radial stacks remain 11.06–11.94. Actual continuous vacuum-tube inner/outer diameters remain 5.76 / 19.20, retaining 1.46 nominal radial clearance from tube exterior to stack interior. The configured 20 beam-clearance declaration is not the actual tube inner diameter. Main pole-shoe bores and minimum radial clearances to the tube remain 20 and 0.40 respectively. Windings outside these three contacts, including DPH1 and DPH2, are not shortened.

## Dimension editing and material linkage

`C3_ProbeCorrector`, `C3_ProbeCorrector_ImageCorrector` and `C3_ProbeCorrector_BCOR` save the same recipe. The three mechanical installation lengths remain 14.5, 14.6 and 14.6; optical centres, effective thicknesses, field calibration and control channels are unchanged.

`winding_axial_length_mm` controls actual copper boundaries about a fixed centre. Carriers, covers, bonding and conductor-gap insulation share those boundaries, with edge-seal thickness extending them at both ends. Existing actual-length/window constraints, layer-thickness representability and neighbour-overlap validation remain active. Clearances after dimension changes must be recomputed from complete materials; 0.10 is not automatically maintained through all later edits.

This round also fixes two editing-description issues associated with explicit copper length: selected copper end faces now point to actual copper length rather than read-only installation length; the actual-length effect description explicitly concerns material geometry and explains independence from effective-field thickness. Old DPH2 without a separate copper length retains the `length_mm` path. With a replacement base model selected, retained native copper length is labelled inactive rather than claiming to modify the replacement.

## Validation record

Raw receipts, before/after parameter comparisons and geometry outputs are in `outputs/probe-contact-isolation-20261021/`, excluded from source publication.

- **478 unique related regression tests passed**, with no final failures, errors or skips. Coverage includes dimension linkage across three configurations, copper end-face parameter paths, native/CAD meshes, parameter descriptions, save/reload/undo/restore, mechanical-overlap lists and physical-model overlays in magnetic-field views. `final-validation.json` replaces historical records with latest complete-test-file reruns, deduplicating each case without counting old failures as passes.
- Independent CAD environment `test_cad_assembly_validation.py`: **8 passed**, none skipped; see `exact-cad-assembly-regression.xml`.
- The three configurations build 339, 345 and 368 material regions. For each, the three modified components are checked against column neighbours and the actual vacuum tube using 498 nearby material pairs and 537 exact distances. No volume intersection or subthreshold overlap; the three copper–pole distances are 0.139999999999873, insulation–pole distances 0.099999999999909, and tube radial clearance 1.46. See `geometry-verification-summary.json` and three `after-exact-contact-*.json` files. This is not recertification of every pair of unchanged components.
- **5 STEP files passed rereading**, covering three interface assemblies, the actual continuous vacuum tube and the combined assembly. The latter contains 67 named material regions and 85 connected solids; names, materials/colours, volumes and bounds are checked by the existing exporter. See `step/step-manifest.json`.
- All six deliberately invalid drafts restoring original copper lengths or increasing edge seals to 0.15 were reported by exact CAD as intersections between the corresponding end caps and pole shoes. Draft editing remains possible, but configuration validation, in-place save and Save As reject them without rewriting originals. See `invalid-draft-exact-collisions.json`. Coarse checks may first identify the parent lens; exact lists identify specific pole shoes and material regions.
- Comparison of 35 configurations found only the agreed 27 field changes (three configurations × three windings × copper length/edge-seal side/source note); all other values and component ordering remain unchanged. Nontarget materials are preserved; the three groups' existing 36 regions shorten with actual copper length, and each configuration gains three end caps. See `material-preservation.json`.
- In the bounded CPU comparison at 300 kV, FEG, C3 + Probe Corrector, STEM Nanoprobe Normal, without energy filter, all 18 field/trajectory arrays in each ideal/analytical mode were bitwise unchanged. The comparison uses 9 deterministic trajectories, Z450–1810, maximum step 0.5, 39 field sources and 13 dipoles. A C3 excitation +1% positive control produced approximately 3.20894e-4 T field change and 4.34007e-7 m trajectory change, demonstrating sensitivity to real input changes. See `transport-preservation.json`.

Initial sandbox temporary-file/STEP permission issues were rerun with normal permissions offscreen. Tests were also corrected for old assumptions about actual copper intervals, original stored versus parsed Descan metadata, and editable conflicting drafts. Geometry tolerances, atomic saving and propagation logic were not relaxed or changed. Early receipts remain traceable; final results supersede them through complete-module reruns. No desktop interaction, full coherent calculation, full-source sampling, material FEM/BEM or real-instrument validation was performed.

## Physical-model boundaries

This changes the mechanical axial length of three copper winding groups, not merely colour or display scale. Current propagation uses independently saved effective-field thicknesses and calibration, rather than solving fields from these copper CAD loops. It can therefore verify preserved optical inputs, fields and trajectories in the current model, but cannot establish that shortening a real current path leaves its field exactly unchanged. Actual geometric modification may require recalibration or material-resolved FEM / BEM.

Existing sources do not establish insulation grade, dielectric strength, creepage paths, thermal expansion, vacuum compatibility, manufacturing tolerances or external attachment. The 0.04 insulation and 0.10 clearance remove direct copper–iron end-face contact in this reconstruction; they do not complete electrical, thermal or manufacturing certification for real hardware.
