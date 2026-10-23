# Internal-structure inspection of the Probe Corrector downstream of TL22

> Subsequent update: this document preserves the inspection baseline and validation records before repair of three copper-end contacts. TL22–DP22, DP21–TL21, and DP11–TL12 were subsequently addressed using explicit copper winding lengths, insulating seals at both ends, and positive assembly clearances; see the [end-insulation and clearance repair record](probe-contact-isolation-2026-10-21.md) for the current structure and evidence from that round. “Unresolved contacts” and “copper retained” below describe only the earlier stage.

The baseline for this round is `512d8c0`. Scope extends from the lower end of TL22 to TL12 and the DP12 reference plane, with inspection and validation through backend code only. Coordinates are module-local millimeters, with +Z downstream; default installed global Z is local Z plus 450 mm. Existing optical parameters, component positions, effective field lengths, and propagation equations were not adjusted.

## Component changes and sources

| Downstream order | Treatment in this round | Original materials retained |
| --- | --- | --- |
| DP22 | Added 0.04 mm insulating edge seal at lower end | Four copper winding packs and original radial laminate retained |
| HPC, QPC | Checked channel identities; still no separate material solids | No physical coils or poles inferred |
| DP21 | Added 0.04 mm insulating edge seal at upper end | Four copper winding packs retained |
| TL21 | Added 0.1 mm insulation and a 0.5 mm passive copper plate at each end | Upper/lower pole pieces, winding pack, yoke, and housing retained |
| QPH1 | Retained control-channel identity | No unknown host added |
| HP1 / DPH1 | HP1 gains six hollow formers and six equivalent core-bonding regions; DPH1 gains 0.04 mm seals at both ends | Six pole pins, six excitation winding packs, yoke, and four deflection copper winding packs retained |
| HPol, QPol | Retained control-channel identities | No unknown hosts added |
| DP11 | Added 0.04 mm insulating edge seal at upper end | Four copper winding packs retained |
| TL12 | Added end insulation/copper plates using its own winding interfaces | Original five material parts retained |
| DP12 | Retained virtual alignment reference | No duplicate downstream AC scan coils generated |

Hexapole topology and control-channel relationships follow [Müller et al. 2006, pp.443–444, Fig.2/3](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf). Winding-former/core-bonding principles follow [JEOL JP2004241190A, [0005], Fig.7(c,d)](https://patents.google.com/patent/JP2004241190A/en); the source supports generic multipole construction, not the actual CEOS HP1 former shape. DPH/DP laminates follow [CEOS US20090079531A1, Fig.1/4/6](https://patents.google.com/patent/US20090079531A1/en); end seals are an explicit engineering inference within that family. Lens end plates follow [Hitachi JP2967966B2, conventional Fig.8, [0005]](https://patents.google.com/patent/JP2967966B2/en).

All added millimeter values are saved engineering-reconstruction allocations, not manufacturer dimensions. Individual wires, screws, water circuits, fittings, or host mounting were not inferred. Passive copper plates add no magnetic sources, and bonding/insulation colors do not calibrate material performance.

All three Probe configurations save these structures. The main Probe configuration gains 25 material regions; Probe + Image and Probe + B-COR each gain 53, including 28 radial support/insulation regions previously missing from four deflector groups. Comparison with the isolated baseline's analytical material catalogs preserves the shapes and materials of all 310, 288, and 311 original regions, respectively.

HP1 former walls and end rims are both 0.25 mm thick; supports derive from its own window and core dimensions. Net axial bonding thickness is 0.474 mm, not HP2's 0.478 mm. TL21 and TL12 use their own winding ODs of 93.3164759 / 93.63 mm and retain 0.675 / 0.67 mm axial space beyond the added end laminates, respectively.

## Vacuum tube and neighboring mechanical clearances

| Interface | Current nominal result / mm | Explanation |
| --- | ---: | --- |
| Actual vacuum tube ID / OD | 5.76 / 19.2 | A component's 20 mm clear-beam declaration is not the tube ID |
| Tube exterior—DP22/DP21/DPH1/DP11 support | Radial 1.46 | Support inner radius 11.06, tube outer radius 9.6 |
| Tube exterior—HP1 pole tips | Radial 2.4 | HP1 material bore diameter 24 |
| DPH1 outer layer—HP1 pole tips | Radial 0.06 | Exact minimum solid distance approximately 0.0599999999999983 |
| DPH1 edge seal—HP1 installation-envelope end | 1.76 at each end | Axial host accommodation, not an OEM mounting method |
| Tube exterior—TL21/TL12 pole pieces | Radial 0.4 | Pole-piece bore diameter 20 |
| TL21—HP1 | Axial 29.1 | Original part positions retained |
| HP1—new DP11 upper edge seal | Axial 29.06 | Original copper-start spacing 29.1; new seal extends 0.04 upstream |

The following three interfaces remain zero-clearance end-face contacts between original copper winding packs and pole pieces, explicitly listed in tests and records:

| Component pair | Contact local Z / mm | Current limitation |
| --- | ---: | --- |
| TL22 lower pole piece—DP22 upper end | 729.1 | Copper end not isolated |
| DP21 lower end—TL21 upper pole piece | 787.2 | Copper end not isolated |
| DP11 lower end—TL12 upper pole piece | 903.6 | Copper end not isolated |

These contacts have no positive clearance; zero overlap volume does not establish electrical isolation. This round seals only the free sides, and validation rejects an additional seal on the contacting sides above. No optical component was moved, original copper shortened, or unsupported pole-piece groove cut to eliminate contacts. Radial connections from lens pole shoulders to the original return yokes remain incompletely reconstructed; new copper plates cannot complete a ferromagnetic return path.

## Dimension coupling and fixes in this round

- All new dimensions enter existing TOML, parameter-provenance, undo/redo, copy, and save formats; native 3D and analytical STEP share the same `SolidSpec`.
- Winding ID/OD changes update matching end plates; HP1 window/former dimensions update corresponding support regions; DPH/DP seal thickness updates only the selected side. All 32 actual dimension edits across four deflector groups changed their corresponding material shapes while preserving optical fields.
- When changing the length of an independently copied lens, ManifestEditor previously failed to scale end-plate contact coordinates, unlike the 3D editor. Both entry points now share axial scaling rules; conflicting explicit bulk coordinates and scaling are rejected before writing. Scaling applies only to independent mechanical copies, not the native optical assembly.
- An extremely small positive end-seal/lens-layer thickness could round to zero at its absolute Z; old validation accepted it and generated a degenerate mesh. Representable positive length is now checked for every actual material-layer boundary, without imposing an arbitrary physical minimum thickness.
- Explicit Box/STEP replacements retain precedence; the saved native recipe is validated again on restoration, and failures do not contaminate drafts, history, or source files. New lens-end geometry enters the existing field-map identity checks; old imported maps require review, and no new field is solved automatically.

## Backend validation evidence

Reproducible artifacts are under `outputs/probe-lower-details-20261021/`; they are not OEM CAD or real-instrument acceptance evidence:

- `material-preservation.json`: all original analytical shapes/materials preserved in three configurations; `transport-preservation.json` additionally confirms that 35 configurations gain only 192 structural fields, with all existing field values and component orders retained.
- `after/{probe,probe-image,probe-bcor}/audit.json`: 130/182/195 installed entries and 456/468/493 meshes, respectively; zero mesh issues, CAD construction errors, or material omissions.
- `exact-clearance.json`: 336 in-column material regions; 764 candidate pairs checked between this round's components and full-column neighbors/actual continuous tubes; 593 exact intersection operations, zero material interpenetrations, and zero below-threshold overlaps. Pairs with bounding-box distance over 2 mm were excluded; this scope is not full-instrument manufacturing-tolerance or dynamic-assembly validation. Contact faces are retained separately. The original `assembly-clearance.json` records 16 interface flags from 48-segment triangle meshes; exact OCC checks confirmed no volumetric interpenetration, without relaxing tolerances.
- `step/step-manifest.json`: all 9 STEP files passed round-trip checks, covering downstream components and actual tubes; the combined assembly's 93 named material regions correspond to 117 connected solids, with counts, volumes, bounds, names, materials, and colors checked.
- `transport-preservation.json`: at 300 kV, STEM Nanoprobe Normal, all 18 arrays in each production ideal/analytical route are bitwise identical; 39 field sources and 13 dipoles, with no duplicate fields added; 9 particles, 730 saved planes, Z450–1810 mm, maximum step 0.5 mm. A +1% C3-strength positive control changes fields/trajectories, confirming a responsive comparison. This check excludes full-source sampling, aperture/specimen cutoff, coherent propagation, and material FEM field solutions.
- After replacing earlier results with final whole-file reruns, **all 1,234 distinct pytest cases passed, with zero failures, errors, or skips**; original XML and deduplication details are in `final-validation.json` in this directory. A further **26 actual CAD STEP/kernel/assembly tests all passed**. Some initial save tests encountered sandbox atomic-replacement restrictions and were rerun with normal backend permissions; real atomic writes were not replaced. Two obsolete coil-ID rejection assertions and a vacuum-segmentation impact assertion in the baseline were updated after reproduction against isolated HEAD; propagation behavior was not changed to satisfy old tests. New tests in this round and 13 test files omitted in the previous round were added to their acceptance scopes; the acceptance runner's independent-subprocess test was rechecked in a temporary directory outside the repository to avoid inheriting outer Git/pytest roots.

See the [deflector record](probe-lower-deflector-details-2026-10-21.md), [transfer-lens record](probe-lower-transfer-lenses-2026-10-21.md), `tests/test_probe_hp1_support.py`, and `tests/test_copied_axial_dimension_sync.py` for individual details and dedicated tests.
