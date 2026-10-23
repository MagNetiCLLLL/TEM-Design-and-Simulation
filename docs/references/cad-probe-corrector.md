# Probe Corrector Internal 3D Structure

Review date: 2026-10-18. The scope is the main hexapoles HP1/HP2 in the current C3 + Probe Corrector, together with their transfer lenses and deflection windings; this round does not rebuild the image corrector. Following the main hexapole improvements, this round adds carriers, radial insulation coverage and interlayer bonding to DPH1/DPH2 and DP11/DP21/DP22. Improvement here means mechanical representation, parameter independence and geometric validation, not optical operating-point optimization or field recalibration.

## Structural evidence and accuracy

[Müller et al., 2006, Advancing the Hexapole Cs-Corrector for the Scanning Transmission Electron Microscope](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=3), p.443 and Figure 3 on p.444, support six radial ferromagnetic pole pins, pole tips narrowing toward the beam axis, windings around radial cores and a common return yoke. Figure 3 is an interior photograph; its caption explicitly says that the vacuum liner is not shown.

The photograph cannot establish this project's exact pole-face curvature, turn count, manufacturing dimensions or material curves. The hardware outer diameter, bore and pole length in the paper were not transferred to this project. Current values are all labelled `engineering_reconstruction_not_oem`; flat tapered pole noses and rectangular bulk windings are engineering approximations at the same level, not manufacturer drawings. Detail follows the [common CAD policy](CAD_DETAIL_POLICY.md): retain material volumes without adding unsupported cooling, fasteners, individual turns or insulating supports.

## Actual improvements to the main hexapoles in the previous stage

The main hexapole comparison baseline is the existing 13-solid HP model before modification, not the earlier version showing only an annular envelope. It already had six pole pins, six copper windings and one yoke; these existing solids are not described as added hardware. This round of auxiliary-component changes retains that main hexapole geometry.

| Part | Before modification | Improvement in this round |
| --- | --- | --- |
| Pole nose and pole pin | 39° annular-sector pole shoe joined to a narrow rectangular stem | Flat tapered pole nose facing the axis, joined to a radial core; each pole pin remains one connected solid |
| Winding | Rectangular solid minus a rectangular window; dimensions from in-function ratios | Retains the bulk rectangular winding, explicitly defining its radial axis, window and core clearance; radial position and dimensions are saved independently |
| Mechanical pole length | Implicitly 80% of installation-envelope length | Explicitly saved as `mechanical_hexapole_pole_length_mm`; old files can still obtain defaults from the original envelope ratio |
| Internal dimensions | Pole width, winding dimensions, yoke thickness, etc. hardcoded in the generator | Saved as millimetre `mechanical_hexapole_*` parameters using the regular dimension editor and save format |
| Parameter constraints | Mainly envelope, material-bore and solid-construction checks | Loading and edit submission check finite numbers, positive dimensions, pole/window/winding containment, circumferential neighbour clearances and winding-corner/yoke clearances |

This improves internal-structure readability, adjustability and geometric consistency; it does not demonstrate lower spherical aberration or better correction performance.

## Current configured dimensions

The table below is read from `configs/instruments/column/C3_ProbeCorrector.toml`; all units are mm. Radial length follows the pole pin, width follows its local tangent, and height/pole length follows beam-axis Z. `coil_radial_inset_mm` is measured outward from the material-bore radius.

| Item / field suffix | HP1 | HP2 |
| --- | ---: | ---: |
| Installation-envelope length `length_mm` | 18.1 | 18.2 |
| Mechanical pole length `pole_length_mm` | 14.48 | 14.56 |
| Pole-tip width `tip_width_mm` | 2.448 | 2.448 |
| Pole-root width `root_width_mm` | 4.896 | 4.896 |
| Radial taper length `taper_length_mm` | 8.925 | 8.925 |
| Core height `stem_height_mm` | 7.24 | 7.28 |
| Radial yoke thickness `yoke_thickness_mm` | 5.61 | 5.61 |
| Winding radial start offset `coil_radial_inset_mm` | 10.71 | 10.71 |
| Winding radial length `coil_radial_length_mm` | 6.885 | 6.885 |
| Winding width `coil_width_mm` | 12.24 | 12.24 |
| Winding height `coil_height_mm` | 12.67 | 12.74 |
| Window width `coil_window_width_mm` | 6.885 | 6.885 |
| Window height `coil_window_height_mm` | 8.688 | 8.736 |
| Optical effective-field width `effective_length_mm`, Gaussian FWHM | 10.0 | 6.0 |

Except for the explicitly listed `length_mm` and `effective_length_mm`, these fields have the `mechanical_hexapole_` prefix. Both main hexapoles retain a 75 mm outer diameter, 24 mm material bore and configured 20 mm beam-clearance bore; the latter cannot be treated as a measured inner diameter of the separate vacuum-tube material model. The one-sided tangential winding/core clearance is 0.9945 mm; one-sided axial clearances are 0.724 mm for HP1 and 0.728 mm for HP2. These are clearances in the reconstructed geometry, not assembly tolerances or measurements.

Mechanical pole length, copper-winding height, installation-envelope length and effective-field width are independent. Changing explicit mechanical dimensions does not rewrite `effective_length_mm`; shrinking the installation envelope until saved internal dimensions no longer fit is rejected, without automatically compressing the interior.

## Format shared with other components

TOML supplies dimensions; `probe_hexapole_geometry.py` parses and validates them, and `cad_multipoles.py` generates `SolidSpec`. Each main HP still outputs `pole_pin_1`–`pole_pin_6`, `coil_1`–`coil_6` and `return_yoke`, retaining the `soft_magnetic` and `copper_winding` categories and module-local millimetre coordinates.

The same solid recipe supplies native 3D meshes and analytic STEP, retaining the `temsim-analytic-material-cad-v1` catalog format and `temsim-analytic-step-v1` export record. STEP is built from analytic solids, not display triangles. Global installation transforms, component identities and material colours follow the existing workflow. Mechanical material categories do not imply calibrated permeability or B–H curves.

ADL, TL21, TL22 and TL12 retain the existing round-lens material subparts. DPH1/DPH2 and DP retain their copper windings, hosts and positions; the five added passive material regions also use the same `SolidSpec`, native 3D and STEP routes. Carriers and covers use `insulator`; bonding layers record a separate material role without uncalibrated dielectric, magnetic or thermal parameters. HPC, QPC, HPol, QPol, QPH and other control channels with unconfirmed physical hosts receive neither conjectured independent material solids nor extra optical excitation. The existing vacuum tube remains its own structure and is not duplicated inside HP.

## Internal components other than hexapoles

The table describes current material-model coverage; “modelled” does not mean complete verification of the manufacturer's hardware.

| Component | Current material structure and installation position | Treatment and limitations in this round |
| --- | --- | --- |
| DPH2 / DPH1 | 4 closed copper windings per group; orthogonal X/Y saddle pairs occupy two layers, installed inside HP2 / HP1 material bores respectively | Adds 2 carriers, 2 radial covers and 1 interlayer bonding region per group; the hexapole's own 6 excitation windings are distinct from these deflection windings |
| DP22 / DP21 | 4 windings of the same type each; currently downstream of TL22 and upstream of TL21 respectively | Adds the same five passive regions with shared parameters, editing and STEP construction; saved positions are not OEM installation dimensions |
| DP11 | 4 windings, currently between HP1 and TL12 | Adds the same five passive regions; retains separate material components and a separate STEP |
| ADL / TL22 / TL21 / TL12 | Each unit contains upper/lower pole shoes, an annular excitation winding, return yoke and housing | Already follows other round lenses' material-subpart format; not redesigned this round |
| QPH1/2, HPC/QPC, HPol/QPol | Optical channels exist; actual coil hosts remain unconfirmed | No independent material solids; names cannot establish pole counts or justify independent iron cores |
| DP12 | Virtual alignment reference | No extra deflector material solid; the downstream AC scan coil is another modelled component |

The 20 original copper windings in the five DPH/DP groups remain unchanged. The main hexapole stage's 66-solid assembly and checks are historical baselines. Adding 5 passive regions per group increases the recipe from 66 to 91 solids; old checks cannot validate the additions.

Four windings represent two pairs for two-axis deflection, not a quadrupole lens. Copper-winding inner/outer diameters remain 22.2/23.8 mm, each layer is 0.32 mm thick, and angular span is 120°. These dimensions describe bulk conductor occupancy, not wire diameter, turn count or individual printed traces.

## Evidence and differences for this round of deflector supports

The [CEOS multipole-coils patent](https://patents.google.com/patent/US20090079531A1/en), Fig. 1, supports a flexible carrier rolled into a cylinder; Fig. 4 describes conductor-cover insulation and interlayer bonding; Fig. 6 supports concentric windings in two directions; Figs. 10–11 describe bore or recess mounting. This evidence supports the `bonded_flexible_pcb_equivalent` structural family, not proof that the named channels use the same factory construction. The patent does not establish this model's layer thicknesses, polymer or adhesive type.

| Item | Previous model | Model in this round |
| --- | --- | --- |
| Winding carrier | Only 4 copper windings, no material carrier | One cylindrical insulating substrate per layer, supporting its copper windings |
| Radial insulation | 0.16 mm gap between conductor layers | Allocated to 0.04 mm X cover, 0.08 mm bonding layer and 0.04 mm Y carrier, plus an X carrier and Y cover |
| Interlayer attachment | No visible material structure | Adds a bonding region representing bonded relative positioning; adhesive type, thermal conductivity and joint strength are unspecified |
| Host mounting | Only HP nesting or position between lenses recorded | Retains the relationships; no unsupported rigid formers, screws, clamps, shoulders or connectors |

The following radial material ranges are engineering allocations in mm, relative to the local beam axis. Carriers and covers are each 0.04 mm thick. Bonding thickness is the retained copper-layer spacing minus adjacent insulating layers, currently 0.08 mm.

| Material region | Inner radius | Outer radius | Addition status |
| --- | ---: | ---: | --- |
| X carrier | 11.06 | 11.10 | Added |
| X copper winding | 11.10 | 11.42 | Two existing windings |
| X radial cover | 11.42 | 11.46 | Added |
| Interlayer bonding region | 11.46 | 11.54 | Added |
| Y carrier | 11.54 | 11.58 | Added |
| Y copper winding | 11.58 | 11.90 | Two existing windings |
| Y radial cover | 11.90 | 11.94 | Added |

The support stack's total inner/outer diameters are 22.12/23.88 mm, extending 0.04 mm radially on each side of the original copper windings and retaining 0.06 mm outer radial clearance in the 24 mm HP material bore. Axial positions and original copper geometry are unchanged.

These cylindrical substrates, covers and bonding regions are bulk equivalent stack volumes representing radial coverage only. They do not complete conformal conductor-sidewall insulation, inter-turn gaps, vias, winding seams or the manufacturing stack. Interlayer bonding does not reconstruct the insert's actual attachment to its host. The improvement is the readability of carriers, insulation and interlayer connections, without claims of purer fields, better cooling or improved spherical-aberration performance.

The separate vacuum tube currently has actual CAD material inner/outer diameters of 5.76/19.2 mm, distinct from the configured 20 mm beam-clearance bore; the previous 20/20.5 mm description does not apply to the current material recipe. This round retains that tube, with 1.46 mm radial clearance to the deflector carrier. The discrepancy between the beam-clearance declaration and actual tube bore still needs separate review and must not be obscured by this support modelling.

All added materials are passive mechanical structures. They add no optical channels and do not directly derive deflection fields from CAD. Dimensions remain labelled `engineering_reconstruction_not_oem`; material colours do not imply validated properties.

## Background export and checks

Generate the catalog and previews without operating the desktop:

```powershell
.venv/Scripts/python.exe scripts/export_probe_corrector_model.py --output outputs/probe_corrector_auxiliary_20261018 --preview
```

The output directory stores `catalog.json`, `comparison.json` and native 3D background previews. The deflector-support comparison is `deflector-support-comparison.png`; assembly and enlarged deflector views retain `probe-corrector-assembly.png` and `probe-deflectors.png`. With a saved pre-change catalog, add `--baseline-catalog <before-catalog.json>` for a same-scale comparison. Previous-stage `outputs/probe_corrector_structure_20261018` remains historical; its 66-solid checks do not replace this round's validation. STEP still uses the existing exporter:

```powershell
environment/cad/Scripts/python.exe scripts/export_cad_models.py outputs/probe_corrector_auxiliary_20261018/catalog.json --output outputs/probe_corrector_auxiliary_20261018/step
```

The STEP exporter rereads files to check solid counts, volumes, bounds, names, colours and material categories. Geometry checks must cover unchanged original copper, added-material counts and radial ordering in five groups, positive thicknesses, contact without volumetric intersections, HP/actual-vacuum-tube clearances, and preserved axial centres and optical inputs. Main hexapoles retain checks for sixfold 60° symmetry, connected pole pins, yoke connections and winding/core clearances. This is the validation scope; results follow. These checks do not replace 3D field calculations or hardware validation.

Background results: all 323 related regressions passed; 10 STEP files passed rereading. Per-solid catalog comparison found 25 added passive solids, with the original 66 solids' shapes and materials unchanged. Including the actual vacuum-tube segment over the same axial range, 92 analytic solids had no material-volume overlaps beyond tolerance, with 90 material contacts recorded. Minimum DPH/HP outer radial clearance was 0.06 mm. Original TOML optical values, positions and effective-field lengths were unchanged. Results are in `regression-final.xml`, `comparison.json`, `configuration-comparison.json`, `step/step-manifest.json` and `collision/report.json`.

New dimensions use the same edit, undo and save mechanisms as other components. Carrier/cover values cannot be nonpositive, Boolean or nonfinite; a positive-thickness bonding region and beam-clearance constraints are required. Parent–child overlap permission cannot allow the DPH support to enter the HP bore wall. This checks nominal coaxial installation envelopes; arbitrary movement/tilt still requires exact geometry checks. References and existing ordinary TEM deflector hardware are listed in the [ordinary deflector study](cad-multipoles.md#ordinary-tem-deflectors-follow-on-review-2026-10-18).

## 2026-10-20 component review: conductor-gap insulation

The 91 solids and 323 tests above are historical results from the previous radial-support model. Component review found five existing radial annular layers, but empty spaces between copper windings and inside winding windows, without conductor-side insulation representing a filled flexible stack.

This round enables `winding_support_fill_conductor_gaps = true` only for DPH2, DP22, DP21, DPH1 and DP11 in current `C3_ProbeCorrector.toml`. Evidence remains the cover insulation in [CEOS US20090079531A1](https://patents.google.com/patent/US20090079531A1/en) Fig. 4 and the full text's full-surface bonding to avoid interlayer air gaps. Selecting a filled stack is an **engineering inference about the structural family**, not verification of these DPH/DP factory interiors. Resin, dielectric constant and thermal properties are unspecified.

Each group gains X and Y `conductor_gap_insulation` material regions. Each is exactly the original copper layer's complete annulus minus its two original copper windings, filling windows and angular inter-winding gaps. No new dimensions are introduced; regions remain within original diameters and axial extents. Original copper, five radial supports, HP clearances, optical strengths, effective lengths, physical hosts and shared channels remain unchanged. A region may contain disconnected volumes touching adjacent carriers/covers; it is not a separately removable part.

Old files without the switch or with explicit `false` retain the original five-layer structure; only Boolean values are accepted. Native display subtracts existing copper meshes to avoid intersections from different circumferential sampling; STEP subtracts the same windings' analytic surfaces. Both preserve original copper. This still omits turn-by-turn insulation, vias, terminals, seams and host attachment, and makes no field or spherical-aberration improvement claim.

Validation: 260 related background/offscreen regressions and 15 independent CAD regressions passed; 275 analytic material-volume pairs across five inserts passed overlap checks (threshold `1e-9 mm³`), with 90 contacts. The 55 material regions were exported as 5 individual STEP files and 1 collection STEP; solid counts, volumes, bounds, names, materials and RGBA colours passed rereading. Comparison confirmed 10 additions, with original 45-region shapes/materials and all optical inputs unchanged. Results are under `outputs/corrector_detail_20261020/`: `regression.xml`, `analytic-contacts.json`, `configuration-comparison.json` and `step/step-manifest.json`. STEP checks also fixed a shared issue: colours must be saved per solid when a region contains disconnected solids. Only solids without explicit colours inherit defaults; existing solid, surface or curve colours are not overwritten. Rereading restores a region colour only when every solid has the same saved colour, without hiding missing or mixed colours.

## 2026-10-20 HP2 winding-former and core-bonding adaptation

HP2's original 13 material regions—six inward-tapering pole pins, six copper windings and a common yoke—remain unchanged. HP2 in all three Probe Corrector configurations gains six hollow winding formers and six equivalent core-bonding regions, totalling 25 regions. HP1 and old files without the new profile retain their previous model.

The primary source [JEOL multipole manufacturing patent JP2004241190A](https://patents.google.com/patent/JP2004241190A/en), [0005] and Fig.7(c,d), describes windings on hollow winding members bonded to pole cores; [0055] also permits bobbins. This supports only the **structural principle of connecting winding members to cores**. It is not a CEOS HP2 production drawing or proof of this instrument's assembly. Hexapole topology remains based on [Müller et al. 2006, p.443–444, Fig.3](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf). No removable cores, screws, locating pins or extra yokes from another patent were mixed in.

New profile: `hexapole_support_profile = "bonded_radial_coil_formers"`. Two dimensions are explicit engineering assumptions: 0.25 mm former wall thickness and 0.25 mm flange thickness at each radial end. Rectangular cross-sections/flanges and unspecified-grade insulating material are also adaptation assumptions. Original copper windows of 6.885 × 8.736 mm and core stems of 4.896 × 7.28 mm remain unchanged; former inner windows are therefore 6.385 × 8.236 mm. Remaining one-sided tangential/axial spaces of 0.7445/0.478 mm become **volume-equivalent bonding fill**, not a claim of such thick adhesive films in real hardware. Dielectric, thermal, mechanical and vacuum properties remain unspecified; individual turn enamel is unresolved.

Winding axes are radial. Former walls touch original copper-window walls; flanges touch radial copper end faces; equivalent bonding joins former interiors to original core stems. Materials contact without intersection, with six copies at 60° intervals. Added materials start at radius 22.46 mm, well outside HP2's original 12 mm material-bore radius, without axial end caps or DPH2-channel restriction. HP2's original 6 mm effective-field length, 18.2 mm physical envelope, 24/75 mm diameters and all optical positions remain unchanged.

DPH2's full material outer radius of 11.94 mm retains 0.06 mm minimum radial clearance to the 12 mm HP2 pole-tip bore; the actual vacuum tube's 9.6 mm outer radius is also unaffected. HP2's common-yoke lower end remains at local Z=703.6 mm, in planar contact with TL22's original upper end, not a positive axial gap. New formers/fill extend only to local Z=700.87 mm, away from this contact. Arbitrarily moved, tilted or scaled neighbours still require actual assembly-geometry checks; these coaxial clearances do not guarantee all arrangements.

Native mesh and STEP share the solid description. Wall/flange edits change passive materials only. Replacement CAD retains but disables this recipe; restoring native geometry revalidates it, and invalid restoration cannot partially change a draft. Passive material categories are not new FEM materials or field solutions and do not alter current analytic hexapoles, spherical aberration or electron propagation; CAD-based field recomputation needs separate material and boundary definitions.


## 2026-10-20 DPH2 axial edge insulation

DPH2 adds two 0.04 mm passive annular edge seals to its original 11 regions, retaining copper, 4 mm effective-field length and optical channels. Full material extents, HP2/vacuum-tube/TL22 clearances, inference limits and actual before/after field/particle validation are in the [DPH2 axial edge-insulation review](dph2-edge-insulation.md). This is an engineering adaptation of a covered-stack family, not a claimed verified factory DPH2 interior shape.

## Winding length versus installation allocation (2026-10-21)

Image-corrector DPH/DP inserts with an explicit `winding_axial_length_mm` use that field for the copper length about their saved centre. For example, BCOR/CETCOR DP11 allocates 6 mm axially but saves 4 mm of copper. The existing `winding_extent_note` explicitly states that the remaining allocation is not material or a measured fringe-field profile. Therefore `length_mm` is now labelled as a read-only installation/field allocation in the dimension editor, while `winding_axial_length_mm` is the editable physical size. Editing it updates native and CAD copper without moving the centre or changing the effective field thickness, channel calibration or operating controls. It must still fit the allocation and leave the configured saddle-loop window open.

The distinction is conditional: existing probe inserts without an explicit copper length, including DPH2, still use their saved `length_mm` as the physical winding length. Their original dimension remains editable. No automatic copper scaling from an optical allocation is introduced. These are the same engineering winding reconstructions already documented above, not additional OEM dimensions.

## 2026-10-21: review downstream of TL22

After the historical baselines above, this round reviews DP22, DP21, TL21, DPH1, HP1, DP11 and TL12, adding explicit winding formers/bonding to HP1 and separately dimensioned end stacks to TL21/TL12, and selecting DPH/DP edge-seal sides by available space. All three Probe configurations adopt these passive structures. Original copper, ferromagnetic materials and optical values are retained.

Individual changes, tube clearances, three unresolved copper–pole-shoe zero-clearance contacts, dimension-editing consistency fixes and background evidence are in the [overall review](probe-lower-detail-review-2026-10-21.md), [deflector record](probe-lower-deflector-details-2026-10-21.md) and [transfer-lens record](probe-lower-transfer-lenses-2026-10-21.md). Earlier material-region counts apply only to their respective historical stages.


## 2026-10-21: follow-up on three DP copper end contacts

DP22, DP21 and DP11 use explicit `winding_axial_length_mm`: fixed-centre copper retracts 0.14 mm per end, with 0.04 mm insulation at both ends, leaving 0.10 mm nominal assembly clearance to adjacent pole shoes. Optical allocations/effective thicknesses, all pole shoes, magnetic gaps and vacuum tubes are retained. Dimensions are engineering reconstructions; public sources support the insulation family but not this instrument's manufacturing dimensions or voltage ratings. Copper shapes are intentionally shortened, so the previous “all copper shapes unchanged” conclusion no longer applies. See the [end-insulation and clearance repair record](probe-contact-isolation-2026-10-21.md).
